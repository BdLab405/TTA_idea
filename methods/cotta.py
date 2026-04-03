"""
Builds upon: https://github.com/qinenergy/cotta
Corresponding paper: https://arxiv.org/abs/2203.13591
+ Memory bank & centroid loss from: "Online Adaptive Fault Diagnosis With Test-Time Domain Adaptation"
"""

import torch
import torch.nn as nn
import torch.jit

from methods.base import TTAMethod
from models.model import split_up_model
from augmentations.transforms_cotta import get_tta_transforms
from utils.registry import ADAPTATION_REGISTRY
from utils.misc import ema_update_model
from collections import deque
import torch.nn.functional as F

# ---------------- 新增 MemoryBank 实现 ----------------
class MemoryBank:
    def __init__(self, feature_dim, num_classes, max_timepoints=5, device='cpu'):
        """
        Stores last max_timepoints batches of (features, pseudo_labels) for centroid computation.
        Based on Online Adaptive Fault Diagnosis: last 5 timepoints, centroid loss starts at 6th batch.
        """
        self.max_timepoints = max_timepoints
        self.device = device
        self.queues = deque(maxlen=max_timepoints)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.total_batches = 0

    def enqueue(self, feats, labels):
        # feats: [B, D] cpu tensor, labels: [B]
        self.queues.append((feats.detach().cpu(), labels.detach().cpu()))
        self.total_batches += 1

    def compute_centroids(self):
        D = self.feature_dim
        sums = torch.zeros(self.num_classes, D)
        counts = torch.zeros(self.num_classes)
        for feats, labels in self.queues:
            for cls in torch.unique(labels):
                mask = (labels == cls)
                if mask.sum() == 0:
                    continue
                cls_idx = int(cls.item())
                sums[cls_idx] += feats[mask].sum(dim=0)
                counts[cls_idx] += mask.sum().item()
        centroids = torch.zeros(self.num_classes, D)
        for k in range(self.num_classes):
            if counts[k] > 0:
                centroids[k] = sums[k] / counts[k]
        return centroids  # cpu tensor

    def num_batches(self):
        return len(self.queues)

def centroid_loss_current_batch(feats, pseudo_labels, memory_bank):
    if memory_bank.num_batches() == 0:
        return torch.tensor(0., device=feats.device)
    centroids = memory_bank.compute_centroids().to(feats.device)
    feats_n = F.normalize(feats, dim=1)
    centroids_n = F.normalize(centroids, dim=1)
    cent_for_samples = centroids_n[pseudo_labels.long()]
    cos = (feats_n * cent_for_samples).sum(dim=1)
    return (1.0 - cos).mean()


@ADAPTATION_REGISTRY.register()
class CoTTA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.mt = cfg.M_TEACHER.MOMENTUM
        self.rst = cfg.COTTA.RST
        self.ap = cfg.COTTA.AP
        self.n_augmentations = cfg.TEST.N_AUGMENTATIONS
        self.arch_name = cfg.MODEL.ARCH

        # Setup EMA and anchor/source model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        self.model_anchor = self.copy_model(self.model)
        for param in self.model_anchor.parameters():
            param.detach_()

        self.models = [self.model, self.model_ema, self.model_anchor]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        self.softmax_entropy = softmax_entropy_cifar if "cifar" in self.dataset_name else softmax_entropy_imagenet
        self.transform = get_tta_transforms(self.img_size,
                                            gaussian_std=cfg.COTTA.GAUSSIAN_STD,
                                            sp_prob=cfg.COTTA.GAUSSIAN_STD)

        # ---------------- 新增 memory bank ----------------
        self.feature_dim = cfg.MODEL.FEATURE_DIM if hasattr(cfg.MODEL, "FEATURE_DIM") else 256
        self.num_classes = num_classes
        self.memory_bank = MemoryBank(feature_dim=self.feature_dim,
                                      num_classes=num_classes,
                                      max_timepoints=cfg.COTTA.MEMORY_TP,
                                      device=self.device)
        self.lambda_centroid = cfg.COTTA.LAMBDA_CENTROID
        self.warmup_batches = cfg.COTTA.WARMUP_BATCHES
        self.batch_idx = 0

    def extract_features(self, x):
        """
        通用特征提取函数：自动兼容 ResNet、ResNet1D、timm 等模型
        """
        model = self.model
        # 1. 使用 split_up_model 切分
        encoder, classifier = split_up_model(model, self.arch_name, self.dataset_name)
        # 2. 用 encoder 提取特征
        with torch.no_grad():
            feats = encoder(x)

        return feats

    def loss_calculation(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)

        # Create prediction of anchor (source) model
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(imgs_test), dim=1).max(1)[0]

        # Augmentation-averaged Prediction
        ema_outputs = []
        if anchor_prob.mean(0) < self.ap:
            for _ in range(self.n_augmentations):
                outputs_ = self.model_ema(self.transform(imgs_test)).detach()
                ema_outputs.append(outputs_)
            outputs_ema = torch.stack(ema_outputs).mean(0)
        else:
            outputs_ema = self.model_ema(imgs_test)

        # ----- CoTTA original loss (soft CE / KL divergence) -----
        loss = self.softmax_entropy(outputs, outputs_ema).mean(0)

        # ----- 新增 centroid loss -----
        with torch.no_grad():
            pseudo_labels = outputs_ema.softmax(dim=1).argmax(dim=1)
        # 2) student 特征（**不要 detach**，这个 feats 需要参与反向传播）
        feats = self.extract_features(imgs_test)  # requires_grad=True (默认)
        # 3) 计算 centroid loss（只用 memory bank 中的历史数据 —— 不包含当前 batch）
        loss_centroid = torch.tensor(0.0, device=loss.device)
        # 要求：达到 warmup 且 memory bank 已收集到足够的 timepoints（通常为 memory_bank.max_timepoints）
        if (self.batch_idx >= self.warmup_batches) and (
                self.memory_bank.num_batches() >= self.memory_bank.max_timepoints):
            loss_centroid = centroid_loss_current_batch(feats, pseudo_labels, self.memory_bank)
        # 4) 把当前 batch 的特征（detach 后）与伪标签加入 memory bank（用于后续计算历史质心）
        self.memory_bank.enqueue(feats.detach().cpu(), pseudo_labels.cpu())
        total_loss = loss + self.lambda_centroid * loss_centroid
        return outputs_ema, total_loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs_ema, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs_ema, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Teacher update (EMA)
        self.model_ema = ema_update_model(
            model_to_update=self.model_ema,
            model_to_merge=self.model,
            momentum=self.mt,
            device=self.device,
            update_all=True
        )

        # Stochastic restore (CoTTA 原始实现)
        # 安全策略：仅在 batch_idx > warmup_batches 后启用 stochastic restore
        # if self.rst > 0. and self.batch_idx >= self.warmup_batches:
        #     with torch.no_grad():
        #         for nm, m in self.model.named_modules():
        #             for npp, p in m.named_parameters():
        #                 if npp in ['weight', 'bias'] and p.requires_grad:
        #                     mask = (torch.rand(p.shape) < self.rst).float().to(self.device)
        #                     p.data = self.model_states[0][f"{nm}.{npp}"] * mask + p * (1. - mask)
        self.batch_idx += 1
        return outputs_ema

    @torch.no_grad()
    def forward_sliding_window(self, x):
        imgs_test = x[0]
        return self.model_ema(imgs_test)

    def configure_model(self):
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


@torch.jit.script
def softmax_entropy_cifar(x, x_ema) -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_imagenet(x, x_ema) -> torch.Tensor:
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)
