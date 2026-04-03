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

logger = logging.getLogger(__name__)


@ADAPTATION_REGISTRY.register()  # 把 RMT 方法注册到一个“适应方法的注册表”中，方便框架统一调用
class RMT(TTAMethod):  # RMT 类，继承自 TTAMethod（表示这是一个测试时自适应的方法）
    def __init__(self, cfg, model, num_classes):
        """
        RMT 的初始化函数，主要做三件事：
        1. 从配置文件 cfg 中读取各种参数；
        2. 设置源域数据加载器、对比学习模块、EMA 模型、投影头等组件；
        3. 如果有预训练好的“原型（prototype）”或 warmup checkpoint，就加载进来。
        """
        super().__init__(cfg, model, num_classes)  # 调用父类初始化（会保存 cfg / model / 类别数 等）

        # -----------------------
        # 1. 数据加载器（source data loader）
        # -----------------------
        # 源域 batch_size 的选择逻辑：
        # 如果 TEST.BATCH_SIZE > 1，就直接用；否则用 WINDOW_LENGTH 代替
        batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH

        # 获取源域数据加载器（src_loader）
        # - dataset_name：源域数据集名字（如 CIFAR-10 / ImageNet）
        # - adaptation：适配方法
        # - preprocess：模型的预处理步骤
        # - data_root_dir：数据根目录
        # - batch_size：上面计算好的 batch 大小
        # - ckpt_path：模型权重路径
        # - num_samples / percentage：控制是否减少源域样本（加快训练或做实验）
        # - workers：最大数据加载进程数
        # - use_clip：是否用 CLIP 特殊预处理
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
            use_clip=cfg.MODEL.USE_CLIP
        )
        self.src_loader_iter = iter(self.src_loader)  # 把源域数据加载器转成迭代器，方便后续按 batch 取数据

        # -----------------------
        # 2. 对比学习相关参数
        # -----------------------
        self.contrast_mode = cfg.CONTRAST.MODE             # 对比学习的模式（one / all）
        self.temperature = cfg.CONTRAST.TEMPERATURE        # 温度系数（softmax 缩放，调节难易度）
        self.base_temperature = self.temperature           # 基础温度（通常等于上面那个）
        self.projection_dim = cfg.CONTRAST.PROJECTION_DIM  # 投影维度，用于降维和计算对比损失

        # -----------------------
        # 3. 各类损失函数的权重
        # -----------------------
        self.lambda_ce_src = cfg.RMT.LAMBDA_CE_SRC  # 源域交叉熵损失的权重
        self.lambda_ce_trg = cfg.RMT.LAMBDA_CE_TRG  # 目标域交叉熵损失的权重
        self.lambda_cont = cfg.RMT.LAMBDA_CONT     # 对比损失的权重

        # -----------------------
        # 4. EMA（指数滑动平均）教师模型相关
        # -----------------------
        self.m_teacher_momentum = cfg.M_TEACHER.MOMENTUM  # EMA 更新的动量参数（通常 0.99 左右）

        # -----------------------
        # 5. Warmup 参数（预热）
        # -----------------------
        self.warmup_steps = cfg.RMT.NUM_SAMPLES_WARM_UP // batch_size_src  # warmup 要走多少个 step
        self.final_lr = cfg.OPTIM.LR  # 最终学习率
        arch_name = cfg.MODEL.ARCH    # 模型架构名称（如 resnet50）
        ckpt_path = cfg.MODEL.CKPT_PATH  # 模型权重路径

        # -----------------------
        # 6. TTA 数据增强（测试时数据增强）
        # -----------------------
        self.tta_transform = get_tta_transforms(self.img_size)  # 获取 TTA 的数据增强变换

        # -----------------------
        # 7. 损失函数
        # -----------------------
        self.symmetric_cross_entropy = SymmetricCrossEntropy()  # 对称交叉熵（比普通 CE 更鲁棒）

        # -----------------------
        # 8. EMA 模型（teacher 模型）
        # -----------------------
        self.model_ema = self.copy_model(self.model)  # 复制一份模型作为 EMA 模型
        for param in self.model_ema.parameters():
            param.detach_()  # EMA 模型的参数不参与梯度更新（只靠 momentum 更新）

        # -----------------------
        # 9. 拆分模型（特征提取器 & 分类器）
        # -----------------------
        self.feature_extractor, self.classifier = split_up_model(self.model, arch_name, self.dataset_name)

        # -----------------------
        # 10. 源域原型（prototypes）
        # -----------------------
        proto_dir_path = os.path.join(cfg.CKPT_DIR, "prototypes")  # 原型存放目录
        if self.dataset_name == "domainnet126":
            # 如果是 domainnet126 数据集，文件名里需要带上源域信息
            fname = f"protos_{self.dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth"
        else:
            fname = f"protos_{self.dataset_name}_{arch_name}.pth"
        fname = os.path.join(proto_dir_path, fname)

        # 如果原型文件存在，就直接加载
        if os.path.exists(fname):
            logger.info("Loading class-wise source prototypes...")
            self.prototypes_src = torch.load(fname)
        else:
            # 否则，先提取特征，再计算各类的 prototype（类中心向量）
            os.makedirs(proto_dir_path, exist_ok=True)
            features_src = torch.tensor([])
            labels_src = torch.tensor([])
            logger.info("Extracting source prototypes...")
            with torch.no_grad():
                for data in tqdm.tqdm(self.src_loader):  # 遍历源域数据
                    x, y = data[0], data[1]
                    tmp_features = self.feature_extractor(x.to(self.device))  # 提取特征
                    features_src = torch.cat([features_src, tmp_features.view(tmp_features.shape[:2]).cpu()], dim=0)
                    labels_src = torch.cat([labels_src, y], dim=0)
                    if len(features_src) > 100000:  # 最多抽取 100k 个特征（防止太大）
                        break

            # 按类别计算 prototype（每类特征的平均值）
            self.prototypes_src = torch.tensor([])
            for i in range(self.num_classes):
                mask = labels_src == i
                self.prototypes_src = torch.cat(
                    [self.prototypes_src, features_src[mask].mean(dim=0, keepdim=True)], dim=0
                )

            torch.save(self.prototypes_src, fname)  # 存起来，下次直接加载

        # 把原型送到 GPU，并加一维（方便计算相似度时 repeat）
        self.prototypes_src = self.prototypes_src.to(self.device).unsqueeze(1)
        # 每个 prototype 对应的类别标签（0 ~ num_classes-1）
        self.prototype_labels_src = torch.arange(start=0, end=self.num_classes, step=1).to(self.device).long()

        # -----------------------
        # 11. 投影层（projector）
        # -----------------------
        if self.dataset_name == "domainnet126":
            # domainnet126 自带良好的特征分布，不需要 projector
            self.projector = nn.Identity()
        else:
            # 否则，加一个 2 层 MLP projector（特征降维+非线性变换）
            num_channels = self.prototypes_src.shape[-1]
            self.projector = nn.Sequential(
                nn.Linear(num_channels, self.projection_dim),
                nn.ReLU(),
                nn.Linear(self.projection_dim, self.projection_dim)
            ).to(self.device)
            # 把 projector 参数加入 optimizer
            self.optimizer.add_param_group({
                'params': self.projector.parameters(),
                'lr': self.optimizer.param_groups[0]["lr"]
            })

        # -----------------------
        # 12. Warmup 阶段
        # -----------------------
        if self.warmup_steps > 0:
            warmup_ckpt_path = os.path.join(cfg.CKPT_DIR, "warmup")
            if self.dataset_name == "domainnet126":
                source_domain = ckpt_path.split(os.sep)[-1].split('_')[1]
                ckpt_path = f"ckpt_warmup_{self.dataset_name}_{source_domain}_{arch_name}_bs{self.src_loader.batch_size}.pth"
            else:
                ckpt_path = f"ckpt_warmup_{self.dataset_name}_{arch_name}_bs{self.src_loader.batch_size}.pth"
            ckpt_path = os.path.join(warmup_ckpt_path, ckpt_path)

            if os.path.exists(ckpt_path):
                # 如果 warmup checkpoint 存在，就加载
                logger.info("Loading warmup checkpoint...")
                checkpoint = torch.load(ckpt_path)
                self.model.load_state_dict(checkpoint["model"])
                self.model_ema.load_state_dict(checkpoint["model_ema"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info(f"Loaded from {ckpt_path}")
            else:
                # 否则，执行 warmup 并保存结果
                os.makedirs(warmup_ckpt_path, exist_ok=True)
                self.warmup()
                torch.save({
                    "model": self.model.state_dict(),
                    "model_ema": self.model_ema.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }, ckpt_path)

        # -----------------------
        # 13. 保存模型和优化器的状态
        # -----------------------
        self.models = [self.model, self.model_ema, self.projector]  # 要一起管理的模型
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    @torch.enable_grad()  # 确保在可能的 no_grad 测试上下文中仍然启用梯度计算
    def warmup(self):
        # 打印日志：开始 warm up 阶段
        logger.info(f"Starting warm up...")

        # 在 warmup 阶段，循环执行 self.warmup_steps 次
        for i in range(self.warmup_steps):

            # ---------- (1) 动态调整学习率 ----------
            # 在 warmup 中，学习率会从接近 0 线性增加到最终目标学习率 self.final_lr
            # i 从 0 开始，所以 (i+1)/self.warmup_steps 会在 (0,1] 区间变化
            for par in self.optimizer.param_groups:
                par["lr"] = self.final_lr * (i + 1) / self.warmup_steps

            # ---------- (2) 从源数据加载一个 batch ----------
            try:
                # 从源数据迭代器中取下一个 batch
                batch = next(self.src_loader_iter)
            except StopIteration:
                # 如果迭代器到头了，就重新创建一个新的迭代器
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            # 取出源域图像，并放到 GPU / 指定设备上
            imgs_src = batch[0].to(self.device)

            # ---------- (3) 模型前向计算 ----------
            # 通过当前模型得到输出
            outputs = self.model(imgs_src)

            # 通过 teacher 模型（EMA 更新的平均模型）得到输出
            outputs_ema = self.model_ema(imgs_src)

            # ---------- (4) 损失计算 ----------
            # 计算对称交叉熵（symmetric cross entropy），用来保持学生模型和教师模型的一致性
            # .mean(0) 是对 batch 取平均
            loss = self.symmetric_cross_entropy(outputs, outputs_ema).mean(0)

            # ---------- (5) 反向传播与优化 ----------
            # 计算梯度
            loss.backward()
            # 更新参数
            self.optimizer.step()
            # 将梯度清零，避免梯度累积
            self.optimizer.zero_grad()

            # ---------- (6) EMA 更新 ----------
            # 使用指数滑动平均（EMA, Exponential Moving Average）更新 teacher 模型
            # 保持 teacher 模型为 student 模型的稳定版本
            self.model_ema = ema_update_model(
                model_to_update=self.model_ema,  # teacher 模型
                model_to_merge=self.model,  # student 模型
                momentum=self.m_teacher_momentum,  # EMA 动量参数，决定更新速度
                device=self.device,
                update_all=True  # 更新所有参数
            )

        # 打印日志：warmup 完成
        logger.info(f"Finished warm up...")

        # ---------- (7) 重设学习率 ----------
        # 最后确保所有参数组的学习率恢复为最终学习率（final_lr）
        for par in self.optimizer.param_groups:
            par["lr"] = self.final_lr

    # Integrated from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    # 从 SupContrast (Supervised Contrastive Learning) 代码中整合而来
    def contrastive_loss(self, features, labels=None, mask=None):
        # features: [batch_size, n_views, feature_dim]
        #   - 一个 batch 内的样本，每个样本可能有多个视角/增强版本 (n_views)
        # labels: [batch_size]，样本的类别标签（可选）
        # mask: [batch_size, batch_size]，表示正样本对的掩码矩阵（可选）

        batch_size = features.shape[0]

        # ---------- (1) 检查输入条件 ----------
        if labels is not None and mask is not None:
            # labels 和 mask 只能二选一，不能同时给
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # 如果都没给，就默认只有自己和自己是正样本
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            # 如果提供了标签，用标签生成 mask
            labels = labels.contiguous().view(-1, 1)  # [batch_size, 1]
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # mask[i][j] = 1 表示样本 i 和样本 j 属于同一类
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            # 如果提供了 mask，就直接用
            mask = mask.float().to(self.device)

        # ---------- (2) 特征展开 ----------
        contrast_count = features.shape[1]
        # 每个样本有多少个视角（augmentations）

        # 把第二维 (n_views) 展开，变成 [batch_size * n_views, feature_dim]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # 过 projector 投影到对比学习空间（比如 MLP 映射）
        contrast_feature = self.projector(contrast_feature)

        # L2 归一化，使特征在球面上（常见于对比学习）
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

        # ---------- (3) 选择 anchor ----------
        if self.contrast_mode == 'one':
            # 每个样本只用第一个视角作为 anchor
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # 所有视角都作为 anchor
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # ---------- (4) 计算相似度 logits ----------
        # anchor_feature * contrast_feature^T / temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # 数值稳定性处理：减去最大值（避免 exp 溢出）
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # ---------- (5) 构造 mask ----------
        # 把 mask 扩展到 [batch_size*anchor_count, batch_size*contrast_count]
        mask = mask.repeat(anchor_count, contrast_count)

        # 构造 logits_mask，用来去掉样本和自己的对比（self-contrast）
        logits_mask = torch.scatter(
            torch.ones_like(mask),  # 初始化为全 1
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0  # 对角线设为 0
        )

        # 最终的有效 mask
        mask = mask * logits_mask

        # ---------- (6) 计算 log_prob ----------
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # ---------- (7) 计算正样本的 log-likelihood 平均值 ----------
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # ---------- (8) 计算最终损失 ----------
        # 温度缩放：放大梯度或调节软化程度
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        # [anchor_count, batch_size] → 取平均
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def loss_calculation(self, x):
        imgs_test = x[0]
        # x 代表测试 batch
        # x[0] 是测试图片 (imgs_test)，x[1] 可能是标签（这里不用）

        # ---------- (1) 正常前向传播 ----------
        # 原始测试图片走一次模型的特征提取器 + 分类器
        features_test = self.feature_extractor(imgs_test)  # 提取特征
        outputs_test = self.classifier(features_test)  # 分类预测结果

        # ---------- (2) 增强后的测试图片 ----------
        # 对测试图片做数据增强 (tta_transform)，再走一遍模型
        features_aug_test = self.feature_extractor(self.tta_transform(imgs_test))
        outputs_aug_test = self.classifier(features_aug_test)

        # ---------- (3) EMA 模型的预测 ----------
        # 用 teacher model（指数滑动平均的副本）对原始测试图片预测
        outputs_ema = self.model_ema(imgs_test)

        # ---------- (4) 原型对齐 ----------
        with torch.no_grad():
            # self.prototypes_src: 预先计算的 source 原型向量 (每类一个)
            # dist[i, j] 表示 第 i 个原型 与 第 j 个测试样本的相似度（余弦相似度）
            dist = F.cosine_similarity(
                x1=self.prototypes_src.repeat(1, features_test.shape[0], 1),
                # 把 test 特征扩展，形状对齐 [n_prototypes, batch, feature_dim]
                x2=features_test.view(1, features_test.shape[0], features_test.shape[1]).repeat(
                    self.prototypes_src.shape[0], 1, 1),
                dim=-1
            )

            # 对每个测试样本，找到最近的 source 原型（最相似的原型）
            _, indices = dist.topk(1, largest=True, dim=0)  # [1, batch]
            indices = indices.squeeze(0)  # [batch]

        # ---------- (5) 组装对比学习的输入 ----------
        # 每个测试样本的 3 个视角:
        #   (a) 最近的源域原型
        #   (b) 测试图片的原始特征
        #   (c) 测试图片增强后的特征
        features = torch.cat([
            self.prototypes_src[indices],  # [batch, 1, feature_dim]
            features_test.view(features_test.shape[0], 1, -1),  # [batch, 1, feature_dim]
            features_aug_test.view(features_test.shape[0], 1, -1)  # [batch, 1, feature_dim]
        ], dim=1)  # 拼接成 [batch, 3, feature_dim]

        # 对比学习损失：鼓励 (原型-测试特征-增强特征) 之间的一致性
        loss_contrastive = self.contrastive_loss(features=features, labels=None)

        # ---------- (6) 自训练损失 ----------
        # outputs_test / outputs_aug_test 要和 teacher model (EMA) 一致
        loss_self_training = (
                0.5 * self.symmetric_cross_entropy(outputs_test, outputs_ema) +
                0.5 * self.symmetric_cross_entropy(outputs_aug_test, outputs_ema)
        ).mean(0)

        # 总损失初始值：目标域自训练损失 + 对比损失
        loss = self.lambda_ce_trg * loss_self_training + self.lambda_cont * loss_contrastive

        # ---------- (7) 如果需要，还加上源域监督训练 ----------
        if self.lambda_ce_src > 0:
            # 从 source loader 取一个 batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            imgs_src, labels_src = batch[0], batch[1]
            # 正常有监督训练（源域的标签是真实标签）
            features_src = self.feature_extractor(imgs_src.to(self.device))
            outputs_src = self.classifier(features_src)
            loss_ce_src = F.cross_entropy(outputs_src, labels_src.to(self.device).long())

            # 把源域损失加到总损失中
            loss += self.lambda_ce_src * loss_ce_src

        # ---------- (8) 融合预测 ----------
        # 最终预测结果 = 测试图片预测 + teacher 预测
        outputs = outputs_test + outputs_ema
        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        # 前向推理 + 模型在测试时的自适应更新
        # 即：一边预测，一边用损失函数更新参数

        if self.mixed_precision and self.device == "cuda":
            # 如果启用了混合精度，并且当前设备是 GPU
            with torch.cuda.amp.autocast():
                # 在 autocast 环境中，前向计算和 loss 计算将自动混合精度
                outputs, loss = self.loss_calculation(x)

            # 使用梯度缩放器（GradScaler）来避免混合精度下的梯度下溢
            self.scaler.scale(loss).backward()  # 反向传播（缩放后的梯度）
            self.scaler.step(self.optimizer)  # 更新参数
            self.scaler.update()  # 更新缩放因子
            self.optimizer.zero_grad()  # 清空梯度

        else:
            # 如果不是混合精度模式，就按普通 FP32 流程来
            outputs, loss = self.loss_calculation(x)
            loss.backward()  # 反向传播
            self.optimizer.step()  # 参数更新
            self.optimizer.zero_grad()  # 清空梯度

        # 每次参数更新后，也要更新教师模型 (EMA model)
        self.model_ema = ema_update_model(
            model_to_update=self.model_ema,  # 被更新的教师模型
            model_to_merge=self.model,  # 学生模型（最新更新）
            momentum=self.m_teacher_momentum,  # EMA 衰减系数
            device=self.device,
            update_all=True  # 是否更新所有参数
        )
        return outputs  # 返回预测结果

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        对单样本进行滑动窗口推理
        x: 使用滑动窗口缓冲生成的输入数据
        return: 模型预测结果
        """
        imgs_test = x[0]  # 从输入取出图像
        outputs_test = self.model(imgs_test)  # 学生模型前向
        outputs_ema = self.model_ema(imgs_test)  # 教师模型前向
        return outputs_test + outputs_ema  # 融合预测（集成）

    def configure_model(self):
        """配置模型，控制哪些参数可训练，哪些保持冻结"""

        # self.model.train()
        self.model.eval()
        # 设成 eval 模式，避免随机深度 (stochastic depth) 等训练时的随机性
        # 但测试时归一化 (test-time normalization) 仍然有效

        self.model.requires_grad_(False)
        # 先禁用整个模型的梯度（冻结）

        # 遍历模型的所有子模块，选择性打开需要更新的部分
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # 对于 2D BatchNorm：打开梯度
                m.requires_grad_(True)
                # 不要用累计统计量（running_mean/var），只用 batch 内的统计
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

            elif isinstance(m, nn.BatchNorm1d):
                # 对于 1D BatchNorm：必须保持 train 模式
                # 因为单样本 TTA 下，BN1d 如果是 eval 模式会直接用无意义的 running stats
                m.train()
                m.requires_grad_(True)

            else:
                # 其他层（比如全连接、卷积等），也要开启梯度
                m.requires_grad_(True)


