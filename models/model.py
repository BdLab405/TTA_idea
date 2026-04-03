import json
import logging
import os

import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from open_clip import create_model_and_transforms, get_tokenizer
from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from typing import Union
from copy import deepcopy
from models import resnet26
from models.custom_clip import ClipTestTimePromptTuning
from packaging import version
from datasets.cls_names import get_class_names
from datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_V2_MASK, IMAGENET_D109_MASK
from datasets.prompts import *


logger = logging.getLogger(__name__)


def get_torchvision_model(model_name: str, weight_version: str = "IMAGENET1K_V1"):
    """
    Restore a pre-trained model from torchvision
    Further details can be found here: https://pytorch.org/vision/0.14/models.html
    Input:
        model_name: Name of the model to create and initialize with pre-trained weights
        weight_version: Name of the pre-trained weights to restore
    Returns:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    assert version.parse(torchvision.__version__) >= version.parse("0.13"), "Torchvision version has to be >= 0.13"

    # check if the specified model name is available in torchvision
    available_models = torchvision.models.list_models(module=torchvision.models)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in torchvision. Choose from: {available_models}")

    # get the weight object of the specified model and the available weight initialization names
    model_weights = torchvision.models.get_model_weights(model_name)
    available_weights = [init_name for init_name in dir(model_weights) if "IMAGENET1K" in init_name]

    # check if the specified type of weights is available
    if weight_version not in available_weights:
        raise ValueError(f"Weight type '{weight_version}' is not supported for torchvision model '{model_name}'."
                         f" Choose from: {available_weights}")

    # restore the specified weights
    model_weights = getattr(model_weights, weight_version)

    # setup the specified model and initialize it with the specified pre-trained weights
    model = torchvision.models.get_model(model_name, weights=model_weights)

    # get the transformation and add the input normalization to the model
    transform = model_weights.transforms()
    model = normalize_model(model, transform.mean, transform.std)
    logger.info(f"Successfully restored '{weight_version}' pre-trained weights"
                f" for model '{model_name}' from torchvision!")

    # create the corresponding input transformation
    preprocess = transforms.Compose([transforms.Resize(transform.resize_size, interpolation=transform.interpolation),
                                     transforms.CenterCrop(transform.crop_size),
                                     transforms.ToTensor()])
    return model, preprocess


def get_timm_model(model_name: str):
    """
    Restore a pre-trained model from timm: https://github.com/huggingface/pytorch-image-models/tree/main/timm
    Quickstart: https://huggingface.co/docs/timm/quickstart
    Input:
        model_name: Name of the model to create and initialize with pre-trained weights
    Returns:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    # check if the defined model name is supported as pre-trained model
    available_models = timm.list_models(pretrained=True)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in timm. Choose from: {available_models}")

    # setup pre-trained model
    model = timm.create_model(model_name, pretrained=True)
    logger.info(f"Successfully restored the weights of '{model_name}' from timm.")

    # restore the input pre-processing
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config)

    # if there is an input normalization, add it to the model and remove it from the input pre-processing
    for transf in preprocess.transforms[::-1]:
        if isinstance(transf, transforms.Normalize):
            # add input normalization to the model
            model = normalize_model(model, mean=transf.mean, std=transf.std)
            preprocess.transforms.remove(transf)
            break

    return model, preprocess


class ResNetDomainNet126(torch.nn.Module):
    """
    Architecture used for DomainNet-126
    """
    def __init__(self, arch: str = "resnet50", checkpoint_path: str = None, num_classes: int = 126, bottleneck_dim: int = 256):
        super().__init__()

        self.arch = arch
        self.bottleneck_dim = bottleneck_dim
        self.weight_norm_dim = 0

        # 1) ResNet backbone (up to penultimate layer)
        if not self.use_bottleneck:
            model = torchvision.models.get_model(self.arch, weights="IMAGENET1K_V1")
            modules = list(model.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules)
            self._output_dim = model.fc.in_features
        # 2) ResNet backbone + bottlenck (last fc as bottleneck)
        else:
            model = torchvision.models.get_model(self.arch, weights="IMAGENET1K_V1")
            model.fc = torch.nn.Linear(model.fc.in_features, self.bottleneck_dim)
            bn = torch.nn.BatchNorm1d(self.bottleneck_dim)
            self.encoder = torch.nn.Sequential(model, bn)
            self._output_dim = self.bottleneck_dim

        self.fc = torch.nn.Linear(self.output_dim, num_classes)

        if self.use_weight_norm:
            self.fc = torch.nn.utils.weight_norm(self.fc, dim=self.weight_norm_dim)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)
        else:
            logger.warning(f"No checkpoint path was specified. Continue with ImageNet pre-trained weights!")

        # add input normalization to the model
        self.encoder = nn.Sequential(ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), self.encoder)

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, checkpoint_path):
        # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # state_dict = dict()
        # model_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint.keys() else checkpoint["model"]

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 初始化空 state_dict
        state_dict = dict()

        # 针对两种格式做兼容判断：
        # - 若 checkpoint 是一个包含 'state_dict' 的字典，则提取出来
        # - 若 checkpoint 本身就是 state_dict（你当前情况），则直接用
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model_state_dict = checkpoint["state_dict"]
        else:
            model_state_dict = checkpoint

        for name, param in model_state_dict.items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        if not self.use_bottleneck:
            backbone_params.extend(self.encoder.parameters())
        # case 2)
        else:
            resnet = self.encoder[1][0]
            for module in list(resnet.children())[:-1]:
                backbone_params.extend(module.parameters())
            # bottleneck fc + (bn) + classifier fc
            extra_params.extend(resnet.fc.parameters())
            extra_params.extend(self.encoder[1][1].parameters())
            extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.weight_norm_dim >= 0


class BaseModel(torch.nn.Module):
    """
    Change the model structure to perform the adaptation "AdaContrast" for other datasets
    """
    def __init__(self, model, arch_name: str, dataset_name: str):
        super().__init__()

        self.encoder, self.fc = split_up_model(model, arch_name=arch_name, dataset_name=dataset_name)
        if isinstance(self.fc, nn.Sequential):
            for module in self.fc.modules():
                if isinstance(module, nn.Linear):
                    self._num_classes = module.out_features
                    self._output_dim = module.in_features
        elif isinstance(self.fc, nn.Linear):
            self._num_classes = self.fc.out_features
            self._output_dim = self.fc.in_features
        else:
            raise ValueError("Unable to detect output dimensions")

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_dim(self):
        return self._output_dim


class ImageNetXMaskingLayer(torch.nn.Module):
    """ Following: https://github.com/hendrycks/imagenet-r/blob/master/eval.py
    """
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x[:, self.mask]


class ImageNetXWrapper(torch.nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

        self.masking_layer = ImageNetXMaskingLayer(mask)

    def forward(self, x):
        logits = self.model(self.normalize(x))
        return self.masking_layer(logits)


class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.normalize(x)
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x


class ZeroShotCLIP(nn.Module):
    def __init__(self, cfg, model, device, normalize):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.device = device
        self.normalize = normalize
        self.prompt_mode = cfg.CLIP.PROMPT_MODE
        self.freeze_text_encoder = cfg.CLIP.FREEZE_TEXT_ENCODER
        self.class_names = get_class_names(cfg.CORRUPTION.DATASET)
        self.tokenize = get_tokenizer(cfg.MODEL.ARCH)
        self.logit_scale = self.model.logit_scale.data

        assert self.prompt_mode in ["custom", "ensemble", "cupl", "all_prompts"]

        # get the prompt templates
        prompt_templates = cfg.CLIP.PROMPT_TEMPLATE
        if self.prompt_mode in ["ensemble", "all_prompts"]:
            try:
                prompt_templates = eval(f"{cfg.CORRUPTION.DATASET.split('_')[0]}_templates")
            except NameError:
                logger.warning(f"Could not find dataset specific prompt templates! Using ImageNet prompt templates!")
                prompt_templates = eval("imagenet_templates")
            logger.info(f"Using the following prompt templates: {prompt_templates}")

        if self.prompt_mode not in ["custom", "ensemble"]:
            # load CuPL prompts
            with open(cfg.CLIP.PROMPT_PATH) as f:
                gpt3_prompts = json.load(f)
            logger.info(f"Successfully restored CuPL prompts from '{cfg.CLIP.PROMPT_PATH}'")

        # extract the text features for faster inference
        with torch.no_grad():
            all_texts = []
            self.text_features = []
            for c_name in self.class_names:
                texts = [template.format(c_name) for template in prompt_templates] if self.prompt_mode != "cupl" else []
                if self.prompt_mode in ["cupl", "all_prompts"]:
                    texts += [t for t in gpt3_prompts[c_name]]

                all_texts += texts
                texts = self.tokenize(texts).to(self.device)
                class_embeddings = model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                self.text_features.append(class_embedding)

            self.text_features = torch.stack(self.text_features, dim=0).to(self.device)
            self.tokenized_texts_all = self.tokenize(all_texts).to(self.device)

        # prevents test-time adaptation methods from unfreezing parameters in the text encoder
        if self.freeze_text_encoder:
            self.model.transformer = None

    @property
    def dtype(self):
        return next(self.model.visual.parameters()).dtype

    def forward(self, imgs_test, return_features=False):
        # normalize the input images
        imgs_test = self.normalize(imgs_test.type(self.dtype))

        if self.freeze_text_encoder or self.cfg.MODEL.ADAPTATION == "source" or "norm" in self.cfg.MODEL.ADAPTATION:
            # get and normalize the image features
            img_features = self.model.encode_image(imgs_test)
            img_features = img_features / img_features.norm(dim=1, keepdim=True)

            # use pre-extracted text features since no text encoder updates are performed
            text_features = self.text_features
        else:
            img_features, text_features, _ = self.model(imgs_test, self.tokenized_texts_all)

        # cosine similarity as logits
        logits_per_image = self.logit_scale.exp() * img_features @ text_features.T

        if return_features:
            return logits_per_image, img_features, text_features
        else:
            return logits_per_image


def get_model(cfg, num_classes: int, device: Union[str, torch.device]):
    """
    根据配置加载并初始化模型，并准备相应的输入预处理方法。
    参数：
        cfg: 配置文件对象，包含模型架构、权重路径等参数
        num_classes: 类别数，用于构建输出层
        device: 运行模型的设备（'cuda' 或 'cpu'）
    返回：
        base_model: 构建好的模型（可能附带了归一化层等）
        preprocess: 对应的输入预处理方法（图像增强等）
    """
    preprocess = None  # 初始化预处理方法为 None

    # 如果启用了使用 CLIP 模型（多模态模型），使用 open_clip 的接口加载
    if cfg.MODEL.USE_CLIP:
        # 加载 CLIP 模型和其对应的预处理 transform
        base_model, _, preprocess = create_model_and_transforms(
            cfg.MODEL.ARCH,
            pretrained=cfg.MODEL.WEIGHTS,  # 是否加载预训练权重
            device=device,
            precision=cfg.CLIP.PRECISION  # 精度（如 fp16）
        )

        # 提取图像归一化操作（通常是最后一步 transform）
        normalization = preprocess.transforms[-1]
        # 去掉预处理中的归一化步骤，因为我们会将其集成到模型里头
        preprocess.transforms = preprocess.transforms[:-1]

        # 如果配置使用 TPT（Test-time Prompt Tuning）方法
        if cfg.MODEL.ADAPTATION == "tpt":
            # 使用 TPT 封装 CLIP 模型，支持 prompt-tuning 的自适应
            base_model = ClipTestTimePromptTuning(
                base_model,
                normalization,
                cfg.MODEL.ARCH,
                cfg.CORRUPTION.DATASET,
                n_ctx=cfg.TPT.N_CTX,                  # prompt token 个数
                ctx_init=cfg.TPT.CTX_INIT,            # 初始 prompt 内容
                class_token_pos=cfg.TPT.CLASS_TOKEN_POS  # class token 位置
            )

            # 如果提供了 prompt 初始化的权重文件（如 CoOp 模型的软 prompt）
            if cfg.MODEL.CKPT_PATH:
                # 加载保存的 soft prompt 参数
                pretrained_ctx = torch.load(cfg.MODEL.CKPT_PATH)['state_dict']['ctx']
                assert pretrained_ctx.shape[0] == cfg.TPT.N_CTX  # 确保 shape 一致
                with torch.no_grad():
                    # 拷贝 soft prompt 到模型中
                    base_model.prompt_learner.ctx.copy_(pretrained_ctx)
                    base_model.prompt_learner.ctx_init_state = pretrained_ctx
                logger.info("Successfully restored pre-trained soft prompt (CoOp)")
        else:
            # 否则走 Zero-Shot 的 CLIP 流程
            base_model = ZeroShotCLIP(cfg, base_model, device, normalize=normalization)

    # 如果是 domainnet126 数据集，使用定制模型结构
    elif cfg.CORRUPTION.DATASET == "domainnet126":
        base_model = ResNetDomainNet126(
            arch=cfg.MODEL.ARCH,
            checkpoint_path=cfg.MODEL.CKPT_PATH,
            num_classes=num_classes
        )

    # ============================================================
    # ★ 新增分支: 自定义 1D ResNet 模型
    # 假设你在配置文件中设置 cfg.MODEL.ARCH = "resnet1d_8"
    # ============================================================
    elif cfg.MODEL.ARCH == "resnet1d_8":
        print(f"Loading custom 1D Model: {cfg.MODEL.ARCH}")

        # 实例化模型
        base_model = ResNet1D_8(num_classes=num_classes)

        # 加载权重
        if cfg.MODEL.CKPT_PATH and os.path.exists(cfg.MODEL.CKPT_PATH):
            # 注意：map_location 确保权重加载到 CPU 防止显存冲突，后面统一 .to(device)
            checkpoint = torch.load(cfg.MODEL.CKPT_PATH, map_location="cpu")

            # 兼容处理：有些保存代码会多一层 'state_dict' 或 'model' 键
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # 加载参数
            try:
                base_model.load_state_dict(state_dict)
                logger.info(f"Successfully restored model from: {cfg.MODEL.CKPT_PATH}")
            except RuntimeError as e:
                print(f"❌ 权重加载失败，请检查 Key 是否匹配: {e}")
        else:
            logger.info("⚠️ Warning: No checkpoint path provided for ResNet1D, using random init.")

        # 定义预处理
        # 1D 数据通常在 Dataset __getitem__ 里做完了 Z-score
        # 这里的 preprocess 通常用于 TTA 框架在测试时对 raw input 做处理
        # 如果你的 TestDataloader 输出已经是 Tensor，这里可以是 Identity
        preprocess = None


    # 否则尝试以下几种方式加载模型
    else:
        try:
            # Step 1: 加载模型和预处理器
            base_model, preprocess = get_torchvision_model(cfg.MODEL.ARCH, weight_version=cfg.MODEL.WEIGHTS)

            # Step 2: 替换 FC 层
            model_core = base_model[1] if isinstance(base_model, nn.Sequential) else base_model
            if hasattr(model_core, "fc"):
                model_core.fc = nn.Linear(512, num_classes)

            # Step 3: 加载你训练的权重
            if cfg.MODEL.CKPT_PATH and os.path.exists(cfg.MODEL.CKPT_PATH):
                checkpoint = torch.load(cfg.MODEL.CKPT_PATH, map_location="cpu")
                state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
                base_model.load_state_dict(state_dict)
                logger.info(f"Successfully restored model from: {cfg.MODEL.CKPT_PATH}")

        except ValueError:
            try:
                # 如果 torchvision 不支持，尝试 timm 库
                base_model, preprocess = get_timm_model(cfg.MODEL.ARCH)
            except ValueError:
                try:
                    # 如果还是失败，看看是否支持自定义模型
                    if cfg.MODEL.ARCH == "resnet26_gn":
                        base_model = resnet26.build_resnet26()
                        checkpoint = torch.load(cfg.MODEL.CKPT_PATH, map_location="cpu")
                        base_model.load_state_dict(checkpoint['net'])  # 加载权重
                        base_model = normalize_model(base_model, resnet26.MEAN, resnet26.STD)
                    else:
                        raise ValueError(f"Model {cfg.MODEL.ARCH} is not supported!")

                    logger.info(f"Successfully restored model '{cfg.MODEL.ARCH}' from: {cfg.MODEL.CKPT_PATH}")
                except ValueError:
                    # 最后再尝试使用 robustbench 提供的鲁棒性模型接口
                    dataset_name = cfg.CORRUPTION.DATASET.split("_")[0]  # 只取主干名
                    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, dataset_name, ThreatModel.corruptions)

        # 对于某些 ImageNet 的变种数据集，需要加掩码限制输出类别
        if cfg.CORRUPTION.DATASET in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
            mask = eval(f"{cfg.CORRUPTION.DATASET.upper()}_MASK")  # 获得 mask 向量
            base_model = ImageNetXWrapper(base_model, mask=mask)  # 包装模型

    return base_model.to(device), preprocess  # 返回模型和预处理 transform


def split_up_model(model, arch_name: str, dataset_name: str):
    """
    Split up the model into an encoder and a classifier.
    This is required for methods like RMT and AdaContrast
    Input:
        model: Model to be split up
        arch_name: Name of the network
        dataset_name: Name of the dataset
    Returns:
        encoder: The encoder of the model
        classifier The classifier of the model
    """
    if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model, model.model.pretrained_cfg["classifier"]):
        # split up models loaded from timm
        classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
        encoder = model
        encoder.model.reset_classifier(0)
        if isinstance(model, ImageNetXWrapper):
            encoder = nn.Sequential(encoder.normalize, encoder.model)

    elif arch_name == "Standard" and dataset_name in {"cifar10", "cifar10_c"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_WRN":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_ResNeXt":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:2], nn.ReLU(), *list(model.children())[2:-1], nn.Flatten())
        classifier = model.classifier
    elif dataset_name == "domainnet126":
        encoder = model.encoder
        classifier = model.fc
        # 2. 处理你的 1D 模型 (插入在这里)
    elif arch_name == "resnet1d_8":
        encoder = nn.Sequential(
            model.initial_conv,
            model.layer1,
            model.layer2,
            model.layer3,
            model.avg_pool,
            nn.Flatten()
        )
        classifier = model.fc
    elif "resnet" in arch_name or "resnext" in arch_name or "wide_resnet" in arch_name or arch_name in {"Standard_R50", "Hendrycks2020AugMix", "Hendrycks2020Many", "Geirhos2018_SIN"}:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "densenet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "efficientnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool, nn.Flatten())
        classifier = model.model.classifier
    elif "mnasnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.layers, nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "shufflenet" in arch_name:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.fc
    elif "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    elif "swin_" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute, model.model.avgpool, model.model.flatten)
        classifier = model.model.head
    elif "convnext" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool)
        classifier = model.model.classifier
    elif arch_name == "mobilenet_v2":
        encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

    # add a masking layer to the classifier
    if dataset_name in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
        mask = eval(f"{dataset_name.upper()}_MASK")
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(mask))

    return encoder, classifier


# === 必须包含之前的 1D 模型定义 ===
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D_8(nn.Module):
    def __init__(self, num_classes=13):
        super(ResNet1D_8, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = BasicBlock1D(64, 64, stride=1)
        self.layer2 = BasicBlock1D(64, 128, stride=2)
        self.layer3 = BasicBlock1D(128, 256, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.initial_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
