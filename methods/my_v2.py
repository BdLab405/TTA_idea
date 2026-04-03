import logging

import os
import tqdm
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from methods.base import TTAMethod
from models.model import split_up_model
from augmentations.transforms_cotta import get_tta_transforms
from datasets.data_loading import get_source_loader
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import SymmetricCrossEntropy
from utils.misc import ema_update_model
from collections import defaultdict

logger = logging.getLogger(__name__)


@ADAPTATION_REGISTRY.register()
class MY2(TTAMethod):
    """RMT (MY) method with integrated LAW-style per-parameter adaptive learning rates.

    本类在不改变原始运行逻辑的前提下做了重构：
    - 将原来冗长的 __init__ 拆分为若干私有 helper（便于阅读与维护）
    - 保留所有原始计算流程：prototype 提取、warmup、contrastive loss、LAW 风格 lr 调整

    依赖（在外部环境中必须存在）：
      - get_source_loader, get_tta_transforms, split_up_model
      - SymmetricCrossEntropy, ema_update_model
      - logger, ADAPTATION_REGISTRY, TTAMethod
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        # ------------------------- 基本配置读取 -------------------------
        self.cfg = cfg
        self.num_classes = num_classes
        self.device = getattr(self, "device", "cuda" if torch.cuda.is_available() else "cpu")

        # 数据/批大小
        batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH

        # 源域 loader
        _, self.src_loader = get_source_loader(
            dataset_name=cfg.CORRUPTION.DATASET,
            adaptation=cfg.MODEL.ADAPTATION,
            preprocess=model.model_preprocess,
            data_root_dir=cfg.DATA_DIR,
            batch_size=batch_size_src,
            ckpt_path=cfg.MODEL.CKPT_PATH,
            num_samples=cfg.SOURCE.NUM_SAMPLES,
            percentage=cfg.SOURCE.PERCENTAGE,
            workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()),
            use_clip=cfg.MODEL.USE_CLIP,
        )
        self.src_loader_iter = iter(self.src_loader)

        # 对比学习与损失超参
        self.contrast_mode = cfg.CONTRAST.MODE
        self.temperature = cfg.CONTRAST.TEMPERATURE
        self.base_temperature = self.temperature
        self.projection_dim = cfg.CONTRAST.PROJECTION_DIM

        self.lambda_ce_src = cfg.MY.LAMBDA_CE_SRC
        self.lambda_ce_trg = cfg.MY.LAMBDA_CE_TRG
        self.lambda_cont = cfg.MY.LAMBDA_CONT

        self.m_teacher_momentum = cfg.M_TEACHER.MOMENTUM

        # warmup 与 lr
        self.warmup_steps = cfg.MY.NUM_SAMPLES_WARM_UP // batch_size_src
        self.final_lr = cfg.OPTIM.LR
        arch_name = cfg.MODEL.ARCH
        ckpt_path = cfg.MODEL.CKPT_PATH

        # transforms / loss
        self.tta_transform = get_tta_transforms(self.img_size)
        self.symmetric_cross_entropy = SymmetricCrossEntropy()

        # EMA 模型复制
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        # feature extractor / classifier 切分
        self.feature_extractor, self.classifier = split_up_model(self.model, arch_name, self.dataset_name)

        # prototypes
        proto_dir_path = os.path.join(cfg.CKPT_DIR, "prototypes")
        self.prototypes_src = self._load_or_extract_prototypes(proto_dir_path, arch_name, ckpt_path)
        self.prototypes_src = self.prototypes_src.to(self.device).unsqueeze(1)
        self.prototype_labels_src = torch.arange(start=0, end=self.num_classes, step=1).to(self.device).long()

        # projector
        if self.dataset_name == "domainnet126":
            self.projector = nn.Identity()
        else:
            num_channels = self.prototypes_src.shape[-1]
            self.projector = nn.Sequential(
                nn.Linear(num_channels, self.projection_dim),
                nn.ReLU(),
                nn.Linear(self.projection_dim, self.projection_dim),
            ).to(self.device)
            # 将 projector 的参数加入外部 optimizer（原逻辑）
            self.optimizer.add_param_group({
                'params': self.projector.parameters(),
                'lr': self.optimizer.param_groups[0]["lr"],
            })

        # ------------------------- LAW 相关初始化 -------------------------
        self.tau = cfg.MY.LAW_TAU
        self.eps = 1e-8
        self.grad_weight = defaultdict(lambda: 0.0)
        self.trainable_dict = {k: v for k, v in self.model.named_parameters() if v.requires_grad}

        base_opt = self.optimizer.param_groups[0]
        self.base_lr = base_opt['lr']
        self.betas = base_opt.get('betas', (0.9, 0.999))
        self.weight_decay = base_opt.get('weight_decay', 0.0)

        # ==== CHANGE: 构建一次性的 per-parameter param_groups（只做一次，避免每步重建 optimizer）
        import torch.optim as optim

        # 将 trainable_dict 的顺序固定下来
        self.trainable_names = list(self.trainable_dict.keys())  # name list, order matters
        self.trainable_params = [self.trainable_dict[n] for n in self.trainable_names]

        # 创建 param_groups：每个参数一个 param_group（也可以按层合并；这里按参数最灵活）
        param_groups = []
        for p in self.trainable_params:
            param_groups.append({
                'params': [p],
                'lr': self.base_lr,
                'betas': self.betas,
                'weight_decay': self.weight_decay,
            })

        # 用这些 param_groups 替换/重建 optimizer（只执行一次）
        self.optimizer = optim.Adam(param_groups)

        # 保存 param_group 对应的参数名列表，用于后续按索引更新 lr
        self.param_group_names = self.trainable_names.copy()
        # 也将 grad_weight 的键确保存在并为 tensor
        for name in self.param_group_names:
            if isinstance(self.grad_weight[name], float):
                self.grad_weight[name] = torch.zeros_like(self.trainable_dict[name].data)

        # warmup checkpoint handling
        if self.warmup_steps > 0:
            self._handle_warmup(arch_name, ckpt_path)

        # 保存当前模型/优化器状态快照
        self.models = [self.model, self.model_ema, self.projector]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    # ------------------------------------------------------------------
    # 私有工具方法（保持原逻辑）
    # ------------------------------------------------------------------
    def _load_or_extract_prototypes(self, proto_dir_path: str, arch_name: str, ckpt_path: str) -> torch.Tensor:
        """Load prototypes from disk if存在，否则从源数据中提取并保存。

        返回：prototypes tensor, shape [num_classes, feat_dim]
        """
        if self.dataset_name == "domainnet126":
            fname = f"protos_{self.dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth"
        else:
            fname = f"protos_{self.dataset_name}_{arch_name}.pth"
        fname = os.path.join(proto_dir_path, fname)

        if os.path.exists(fname):
            logger.info("Loading class-wise source prototypes...")
            prototypes = torch.load(fname)
            return prototypes

        # 否则提取
        os.makedirs(proto_dir_path, exist_ok=True)
        features_src = torch.tensor([])
        labels_src = torch.tensor([])
        logger.info("Extracting source prototypes...")
        with torch.no_grad():
            for data in tqdm.tqdm(self.src_loader):
                x, y = data[0], data[1]
                tmp_features = self.feature_extractor(x.to(self.device))
                features_src = torch.cat([features_src, tmp_features.view(tmp_features.shape[:2]).cpu()], dim=0)
                labels_src = torch.cat([labels_src, y], dim=0)
                if len(features_src) > 100000:
                    break

        prototypes = torch.tensor([])
        for i in range(self.num_classes):
            mask = labels_src == i
            prototypes = torch.cat([prototypes, features_src[mask].mean(dim=0, keepdim=True)], dim=0)

        torch.save(prototypes, fname)
        return prototypes

    def _init_projector(self):
        """如果需要，按原逻辑初始化 projector 并把参数加入 optimizer。
        (此函数在当前实现中为示例，实际初始化已在 __init__ 处理)
        """
        num_channels = self.prototypes_src.shape[-1]
        self.projector = nn.Sequential(
            nn.Linear(num_channels, self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim),
        ).to(self.device)
        self.optimizer.add_param_group({
            'params': self.projector.parameters(),
            'lr': self.optimizer.param_groups[0]["lr"],
        })

    def _handle_warmup(self, arch_name: str, ckpt_path: str):
        """处理 warmup 检查点的加载或训练并保存（保留原逻辑）。"""
        warmup_ckpt_path = os.path.join(self.cfg.CKPT_DIR, "warmup")
        if self.dataset_name == "domainnet126":
            source_domain = ckpt_path.split(os.sep)[-1].split('_')[1]
            ckpt_fname = f"ckpt_warmup_{self.dataset_name}_{source_domain}_{arch_name}_bs{self.src_loader.batch_size}.pth"
        else:
            ckpt_fname = f"ckpt_warmup_{self.dataset_name}_{arch_name}_bs{self.src_loader.batch_size}.pth"
        ckpt_path_full = os.path.join(warmup_ckpt_path, ckpt_fname)

        if os.path.exists(ckpt_path_full):
            logger.info("Loading warmup checkpoint...")
            checkpoint = torch.load(ckpt_path_full)
            self.model.load_state_dict(checkpoint["model"])
            self.model_ema.load_state_dict(checkpoint["model_ema"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"Loaded from {ckpt_path_full}")
        else:
            os.makedirs(warmup_ckpt_path, exist_ok=True)
            self.warmup()
            torch.save({
                "model": self.model.state_dict(),
                "model_ema": self.model_ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }, ckpt_path_full)

    # ------------------------------------------------------------------
    # warmup / contrastive / loss_calculation（保持原逻辑，仅做格式重排）
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def warmup(self):
        """Warm-up on source data (与原实现一致)。"""
        logger.info("Starting warm up...")

        for i in range(self.warmup_steps):
            for par in self.optimizer.param_groups:
                par["lr"] = self.final_lr * (i + 1) / self.warmup_steps

            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            imgs_src = batch[0].to(self.device)

            outputs = self.model(imgs_src)
            outputs_ema = self.model_ema(imgs_src)
            loss = self.symmetric_cross_entropy(outputs, outputs_ema).mean(0)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.model_ema = ema_update_model(
                model_to_update=self.model_ema,
                model_to_merge=self.model,
                momentum=self.m_teacher_momentum,
                device=self.device,
                update_all=True,
            )

        logger.info("Finished warm up...")

        for par in self.optimizer.param_groups:
            par["lr"] = self.final_lr

    def contrastive_loss(self, features, labels=None, mask=None):
        """Contrastive loss（保留原有实现）。"""
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def loss_calculation(self, x):
        """计算 RMT 主损失：self-training + contrastive（保留原逻辑）。"""
        imgs_test = x[0]
        features_test = self.feature_extractor(imgs_test)
        outputs_test = self.classifier(features_test)

        features_aug_test = self.feature_extractor(self.tta_transform(imgs_test))
        outputs_aug_test = self.classifier(features_aug_test)

        outputs_ema = self.model_ema(imgs_test)

        with torch.no_grad():
            dist = F.cosine_similarity(
                x1=self.prototypes_src.repeat(1, features_test.shape[0], 1),
                x2=features_test.view(1, features_test.shape[0], features_test.shape[1]).repeat(
                    self.prototypes_src.shape[0], 1, 1),
                dim=-1,
            )
            _, indices = dist.topk(1, largest=True, dim=0)
            indices = indices.squeeze(0)

        features = torch.cat([
            self.prototypes_src[indices],
            features_test.view(features_test.shape[0], 1, -1),
            features_aug_test.view(features_test.shape[0], 1, -1),
        ], dim=1)

        loss_contrastive = self.contrastive_loss(features=features, labels=None)
        loss_self_training = (
            0.5 * self.symmetric_cross_entropy(outputs_test, outputs_ema) +
            0.5 * self.symmetric_cross_entropy(outputs_aug_test, outputs_ema)
        ).mean(0)

        loss = self.lambda_ce_trg * loss_self_training + self.lambda_cont * loss_contrastive
        outputs = outputs_test + outputs_ema
        return outputs, loss

    # ------------------------------------------------------------------
    # forward_and_adapt: LAW 风格 per-parameter lr 逻辑（保持原实现）
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """主适配函数：

        1) 先用模型 logits 的 argmax 作为伪标签做 NLL backward，累积 grad^2 到 self.grad_weight（LAW 部分）
        2) 基于 grad_weight 做 min-max 标准化并计算 lr 权重（lr = base_lr * (w**tau)），重建 optimizer
        3) 用新 optimizer 对 RMT 主损失做 backward/step（兼容 mixed precision）
        """
        imgs_test = x[0]

        # 输出与主损失（RMT 逻辑）
        outputs, main_loss = self.loss_calculation(x)

        # 重新计算 logits（未与 ema 相加）用于伪标签（与原实现一致）
        features_test = self.feature_extractor(imgs_test)
        logits_est = self.classifier(features_test)

        pseudo_label = logits_est.max(1)[1].view(-1)
        nll_loss = F.nll_loss(F.log_softmax(logits_est, dim=1), pseudo_label)

        # 反向传播以获得 param.grad
        nll_loss.backward()

        # 累积 grad^2
        min_weight, max_weight = 1e8, -1e8
        for name, param in self.trainable_dict.items():
            if param.grad is None:
                continue
            if isinstance(self.grad_weight[name], float):
                self.grad_weight[name] = torch.zeros_like(param.grad.data)
            self.grad_weight[name] += (param.grad.data ** 2)
            w = (self.grad_weight[name].mean().item()) ** 0.5
            min_weight = min(min_weight, w)
            max_weight = max(max_weight, w)

        # ===== CHANGE: 使用累计的 self.grad_weight（FIM 近似），计算 per-parameter lr 权重并直接更新已有 optimizer.param_groups
        # 可选：对 grad_weight 做 EMA 平滑，避免过度抖动
        ema_m = getattr(self, "law_grad_ema_momentum", 0.9)  # 可在 cfg 中设置 self.law_grad_ema_momentum
        for name, param in self.trainable_dict.items():
            if param.grad is None:
                continue
            # 确保 grad_weight[name] 是 tensor
            if isinstance(self.grad_weight[name], float):
                self.grad_weight[name] = torch.zeros_like(param.grad.data)
            # EMA 平滑（可选）
            self.grad_weight[name] = ema_m * self.grad_weight[name] + (1.0 - ema_m) * (param.grad.data ** 2)

        # 计算每个参数的“标量统计值”（mean -> sqrt）
        grad_stats = []
        for name in self.param_group_names:
            gw = self.grad_weight[name]
            stat = gw.mean().item() ** 0.5
            grad_stats.append(stat)
        grad_stats = torch.tensor(grad_stats, dtype=torch.float32)

        # min-max 归一化（带 eps 避免除0）
        min_val = grad_stats.min().item()
        max_val = grad_stats.max().item()
        denom = (max_val - min_val + self.eps)
        if denom < 1e-12:
            lr_weights = torch.ones_like(grad_stats)
        else:
            lr_weights = (grad_stats - min_val) / denom

        # tau 指数缩放
        lr_weights = lr_weights.clamp(min=1e-12) ** (self.tau)

        # 将 lr_weights 应用于已有 optimizer.param_groups（按索引对应 param_group_names）
        for idx, group in enumerate(self.optimizer.param_groups):
            # 对于 projector 或额外 param_group（如果有额外组），做兜底处理
            if idx < len(lr_weights):
                group_lr = float(self.base_lr * lr_weights[idx])
            else:
                # 如果 param_groups 数量不匹配（例如额外的 projector group），使用 base_lr
                group_lr = float(self.base_lr)
            # 你可能希望设置一个最小/最大 lr 限制，避免过小或过大
            min_lr = getattr(self.cfg.MY, "MIN_LAW_LR", 1e-6)
            max_lr = getattr(self.cfg.MY, "MAX_LAW_LR", 1.0)
            group['lr'] = max(min_lr, min(group_lr, max_lr))

        # 不重建 optimizer，不清空状态
        # 后续直接 self.optimizer.step() 即可

        # 用新 optimizer 更新主损失
        self.optimizer.zero_grad()
        if getattr(self, 'mixed_precision', False) and self.device == "cuda":
            self.scaler.scale(main_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            main_loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

        # 更新 EMA
        self.model_ema = ema_update_model(
            model_to_update=self.model_ema,
            model_to_merge=self.model,
            momentum=self.m_teacher_momentum,
            device=self.device,
            update_all=True,
        )

        return outputs

    @torch.no_grad()
    def forward_sliding_window(self, x):
        imgs_test = x[0]
        outputs_test = self.model(imgs_test)
        outputs_ema = self.model_ema(imgs_test)
        return outputs_test + outputs_ema

    def configure_model(self):
        """配置模型：保持 eval 模式但允许 BN/Norm/Conv 的参数求导（与原逻辑一致）。"""
        self.model.eval()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)



