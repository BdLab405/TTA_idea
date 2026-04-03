import os
import logging
import random
import numpy as np
import time
import webdataset as wds

import torch
import torchvision
import torchvision.transforms as transforms

from typing import Union
from conf import complete_data_dir_path, generalization_dataset_names, ds_name2pytorch_ds_name
from datasets.imagelist_dataset import ImageList, FGVCAircraft
from datasets.imagenet_subsets import create_imagenet_subset
from datasets.corruptions_datasets import create_cifarc_dataset, create_imagenetc_dataset, create_roll_dataset, create_roll_dataset_1d
from datasets.imagenet_d_utils import create_symlinks_and_get_imagenet_visda_mapping
from datasets.imagenet_dict import map_dict
from augmentations.transforms_adacontrast import get_augmentation_versions, get_augmentation
from augmentations.transforms_augmix import AugMixAugmenter
from robustbench.data import load_roll_source, load_roll_source_1d

logger = logging.getLogger(__name__)


def identity(x):
    return x


def get_transform(dataset_name: str, adaptation: str, preprocess: Union[transforms.Compose, None], use_clip: bool,
                  n_views: int = 64):
    """
    Get the transformation pipeline
    Note that the data normalization is done within the model
    Input:
        dataset_name: Name of the dataset
        adaptation: Name of the adaptation method
        preprocess: Input pre-processing from restored model (if available)
        use_clip: If the underlying model is based on CLIP
        n_views Number of views for test-time augmentation
    Returns:
        transforms: The data pre-processing (and augmentation)
    """
    if use_clip:
        if adaptation in ["tpt", "vte"]:
            base_transform = transforms.Compose([preprocess.transforms[0], preprocess.transforms[1]])
            preproc = transforms.Compose([transforms.ToTensor()])  # the input normalization is done within the model
            use_augmix = True if dataset_name in generalization_dataset_names else False
            transform = AugMixAugmenter(base_transform, preproc, dataset_name=dataset_name,
                                        n_views=n_views - 1, use_augmix=use_augmix)
        else:
            transform = preprocess

    elif adaptation in ["memo", "ttaug"]:
        base_transform = transforms.Compose(
            [preprocess.transforms[0], preprocess.transforms[1]]) if preprocess else None
        preproc = transforms.Compose([transforms.ToTensor()])
        transform = AugMixAugmenter(base_transform, preproc, dataset_name=dataset_name, n_views=n_views,
                                    use_augmix=True)

    elif adaptation == "adacontrast":
        # adacontrast requires specific transformations
        if dataset_name in ["cifar10", "cifar100", "cifar10_c", "cifar100_c"]:
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=(32, 32),
                                                  crop_size=32)
        elif dataset_name in ["imagenet_c", "ccc"]:
            # note that ImageNet-C and CCC are already resized and centre cropped (to size 224)
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=(224, 224),
                                                  crop_size=224)
        elif dataset_name == "domainnet126":
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=(256, 256),
                                                  crop_size=224)
        else:
            resize_size = 256
            crop_size = 224
            # try to get the correct resize & crop size from the pre-process
            if isinstance(preprocess, transforms.Compose):
                for transf in preprocess.transforms:
                    if isinstance(transf, transforms.Resize):
                        resize_size = transf.size
                    elif isinstance(transf,
                                    (transforms.CenterCrop, transforms.RandomCrop, transforms.RandomResizedCrop)):
                        crop_size = transf.size

            # use classical ImageNet transformation procedure
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=resize_size,
                                                  crop_size=crop_size)
    else:
        # create non-method specific transformation
        if dataset_name in ["cifar10", "cifar100"]:
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name in ["cifar10_c", "cifar100_c"]:
            transform = None
        elif dataset_name in ["imagenet_c", "ccc"]:
            # note that ImageNet-C and CCC are already resized and centre cropped (to size 224)
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name == "domainnet126":
            transform = get_augmentation(aug_type="test", res_size=(256, 256), crop_size=224)
        else:
            if preprocess:
                # set transform to the corresponding input transformation of the restored model
                transform = preprocess
            else:
                # use classical ImageNet transformation procedure
                transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()])

    return transform


def get_test_loader(setting: str, adaptation: str, dataset_name: str, preprocess: Union[transforms.Compose, None],
                    data_root_dir: str, domain_name: str, domain_names_all: list, severity: int, num_examples: int,
                    rng_seed: int, use_clip: bool, n_views: int = 64, delta_dirichlet: float = 0.,
                    batch_size: int = 128, shuffle: bool = False, workers: int = 4):
    """
    构造测试集加载器（Test DataLoader）
    输入参数：
        setting: 当前使用的实验设定（如是否重置、是否混合域等）
        adaptation: 自适应方法名称（如 rmt、tent 等）
        dataset_name: 数据集名称（如 cifar10_c）
        preprocess: 模型自带的图像预处理方法
        data_root_dir: 数据根目录
        domain_name: 当前要测试的域（或 corruption 类型）
        domain_names_all: 所有域的名称，用于“混合域”等设定
        severity: 图像扰动强度（1~5）
        num_examples: 本次测试使用的图像样本数量
        rng_seed: 随机种子
        use_clip: 是否使用 CLIP 模型
        n_views: 测试时每张图像的增强次数
        delta_dirichlet: 类间分布扰动（用于相关性设定）
        batch_size: 每次加载的样本数量
        shuffle: 是否打乱顺序（一般为 False）
        workers: 使用的线程数
    返回：
        test_loader: pytorch 的测试数据加载器
    """

    # 设定随机种子，确保所有方法对测试集顺序一致（保证可复现）
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    # 根据数据集名称补全实际数据路径（如把“imagenet_c”映射到“ImageNet-C”）
    data_dir = complete_data_dir_path(data_root_dir, dataset_name)

    # 根据数据集、自适应方法等获取图像预处理 pipeline（如 Resize、Normalize 等）
    transform = get_transform(dataset_name, adaptation, preprocess, use_clip, n_views)

    # 创建测试集
    if domain_name == "none":
        # 如果是源域，则直接加载源数据
        test_dataset, _ = get_source_loader(dataset_name, adaptation, preprocess,
                                            data_root_dir, batch_size, use_clip, n_views,
                                            train_split=False, workers=workers)
    else:
        # 针对 CIFAR-C 数据集（如 cifar10_c）
        if dataset_name in ["cifar10_c", "cifar100_c"]:
            test_dataset = create_cifarc_dataset(dataset_name=dataset_name,
                                                 severity=severity,
                                                 data_dir=data_dir,
                                                 corruption=domain_name,
                                                 corruptions_seq=domain_names_all,
                                                 transform=transform,
                                                 setting=setting)
        # 针对 轴承 数据集（如 PU_1d_8c_2048）
        elif dataset_name in ["PU_1d_8c_2048", "PU_13c", "PHM", "JNU_4c"]:
            test_dataset = create_roll_dataset(data_dir=data_dir,
                                               domain_name=domain_name,
                                               domain_names=domain_names_all,
                                               transform=None,
                                               setting=setting)

        # 针对 轴承 数据集（如 PU_1d_8c_2048）
        elif dataset_name in ["PU_13c_1d"]:
            # ★ 路由到新的 1D 数据集创建函数
            test_dataset = create_roll_dataset_1d(
                data_dir=data_dir,
                domain_name=domain_name,
                domain_names=domain_names_all,
                transform=preprocess,  # 对于 TTA，这里可能包含额外变换，但基础 Z-score 在 Dataset 内部
                setting=setting
            )

        # 针对 ImageNet-C 数据集
        elif dataset_name == "imagenet_c":
            test_dataset = create_imagenetc_dataset(n_examples=num_examples,
                                                    severity=severity,
                                                    data_dir=data_dir,
                                                    corruption=domain_name,
                                                    corruptions_seq=domain_names_all,
                                                    transform=transform,
                                                    setting=setting)
        # 针对 Imagenet-K/R/A/V2，这些是标准的 image folder 格式
        elif dataset_name in ["imagenet_k", "imagenet_r", "imagenet_a", "imagenet_v2"]:
            test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

        # CCC 是 webdataset 格式（按序列编号 tar 文件组织）
        elif dataset_name == "ccc":
            url = f'https://mlcloud.uni-tuebingen.de:7443/datasets/CCC/{domain_name}/serial_{{00000..99999}}.tar'
            test_dataset = (wds.WebDataset(url)
                            .decode("pil")
                            .to_tuple("input.jpg", "output.cls")
                            .map_tuple(transform, identity))

        # 对于 ImageNet-D 和 DomainNet-126，需要使用 list.txt 加 label 映射
        elif dataset_name in ["imagenet_d", "imagenet_d109", "domainnet126"]:
            # ImageNet-D 可能需要先建软链接（如果不存在）
            if dataset_name in ["imagenet_d", "imagenet_d109"]:
                for dom_name in domain_names_all:
                    if not os.path.exists(os.path.join(data_dir, dom_name)):
                        logger.info(f"Creating symbolical links for ImageNet-D {dom_name}...")
                        domainnet_dir = os.path.join(complete_data_dir_path(data_root_dir, "domainnet126"), dom_name)
                        create_symlinks_and_get_imagenet_visda_mapping(domainnet_dir, map_dict)

            # 创建标签文件列表（多个 domain 合并或单一 domain）
            if "mixed_domains" in setting:
                data_files = [os.path.join("datasets", f"{dataset_name}_lists", dom_name + "_list.txt") for dom_name in
                              domain_names_all]
            else:
                data_files = [os.path.join("datasets", f"{dataset_name}_lists", domain_name + "_list.txt")]

            test_dataset = ImageList(image_root=data_dir,
                                     label_files=data_files,
                                     transform=transform)

        # 如果是泛化数据集（如 flowers102、oxford_pets 等）
        elif dataset_name in generalization_dataset_names:
            if not os.path.exists(data_dir):
                # 自动下载数据
                ds_name = ds_name2pytorch_ds_name(dataset_name)
                eval(f"torchvision.datasets.{ds_name}")(root=data_root_dir, download=True)

            # 特例：fgvc_aircraft 需要特殊处理
            if dataset_name == "fgvc_aircraft":
                test_dataset = FGVCAircraft(image_root=data_dir, transform=transform, split="test")
            else:
                # 其他数据集通过 json list 加载
                data_list_paths = [os.path.join("datasets", f"other_lists", f"split_zhou_{dataset_name}.json")]
                test_dataset = ImageList(image_root=data_dir, label_files=data_list_paths, transform=transform,
                                         split="test")

        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported!")

    try:
        # 尝试对样本打乱（确保测试顺序一致）
        random.shuffle(test_dataset.samples)

        # 如果指定了样本数量限制，则随机抽样
        if num_examples != -1:
            num_samples_orig = len(test_dataset)
            test_dataset.samples = random.sample(test_dataset.samples, k=min(num_examples, num_samples_orig))

        # 混合域设定下，输出提示
        if "mixed_domains" in setting:
            logger.info(f"Successfully mixed the file paths of the following domains: {domain_names_all}")

        # 排序样本（用于类相关测试）
        if "correlated" in setting:
            if delta_dirichlet > 0.:
                logger.info(
                    f"Using Dirichlet distribution with delta={delta_dirichlet} to temporally correlated samples by class labels...")
                test_dataset.samples = sort_by_dirichlet(delta_dirichlet, samples=test_dataset.samples)
            else:
                logger.info(f"Sorting the file paths by class labels...")
                test_dataset.samples.sort(key=lambda x: x[1])
    except AttributeError:
        logger.warning(
            "Attribute 'samples' is missing. Continuing without shuffling, sorting or subsampling the files...")

    # 返回 PyTorch DataLoader 对象
    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                       drop_last=False)


def get_source_loader(dataset_name: str, adaptation: str, preprocess: Union[transforms.Compose, None],
                      data_root_dir: str, batch_size: int, use_clip: bool = False, n_views: int = 64,
                      train_split: bool = True, ckpt_path: str = None, num_samples: int = -1,
                      percentage: float = 1.0, workers: int = 4):
    """
    Create the source data loader
    Input:
        dataset_name: Name of the dataset
        adaptation: Name of the adaptation method
        preprocess: Input pre-processing from restored model (if available)
        data_root_dir: Path of the data root directory
        batch_size: The number of samples to process in each iteration
        use_clip: If the underlying model is based on CLIP
        n_views: Number of views for test-time augmentation
        train_split: Whether to use the training or validation split
        ckpt_path: Path to a checkpoint which determines the source domain for DomainNet-126
        num_samples: Number of source samples used during training
        percentage: (0, 1] Percentage of source samples used during training
        workers: Number of workers used for data loading
    Returns:
        source_dataset: The source dataset
        source_loader: The source data loader
    """

    # create the correct source dataset name
    src_dataset_name = dataset_name.split("_")[0] if dataset_name != "ccc" else "imagenet"

    # complete the data root path to the full dataset path
    data_dir = complete_data_dir_path(data_root_dir, dataset_name=src_dataset_name)

    # get the data transformation
    transform = get_transform(src_dataset_name, adaptation, preprocess, use_clip, n_views)

    # create the source dataset
    if dataset_name in ["cifar10", "cifar10_c"]:
        source_dataset = torchvision.datasets.CIFAR10(root=data_root_dir,
                                                      train=train_split,
                                                      download=True,
                                                      transform=transform)
    elif dataset_name in ["PU_1d_8c_2048", "PU_13c"]:
        data_dir = complete_data_dir_path(data_root_dir, dataset_name)
        # 加载某个领域作为源域数据
        domain_names = ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10']
        domain_str = os.path.splitext(os.path.basename(ckpt_path))[0].split('_', 2)[-1]
        index = domain_names.index(domain_str)
        source_dataset = load_roll_source(domain_index=index, domain_names=domain_names, data_root_dir=data_dir, transform=transform)

    elif dataset_name in ["PU_13c_1d"]:
        data_dir = complete_data_dir_path(data_root_dir, dataset_name)
        # 加载某个领域作为源域数据
        domain_names = ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10']
        domain_str = os.path.splitext(os.path.basename(ckpt_path))[0].split('_', 2)[-1]
        index = domain_names.index(domain_str)
        source_dataset = load_roll_source_1d(domain_index=index, domain_names=domain_names, data_root_dir=data_dir)

    elif dataset_name in ["PHM"]:
        data_dir = complete_data_dir_path(data_root_dir, dataset_name)
        # 加载某个领域作为源域数据
        domain_names = ['30hz', '35hz', '40hz', '45hz']
        domain_str = os.path.splitext(os.path.basename(ckpt_path))[0].split('_', 2)[-1]
        index = domain_names.index(domain_str)
        source_dataset = load_roll_source(domain_index=index, domain_names=domain_names, data_root_dir=data_dir, transform=transform)

    elif dataset_name in ["JNU_4c"]:
        data_dir = complete_data_dir_path(data_root_dir, dataset_name)
        # 加载某个领域作为源域数据
        domain_names = ['600', '800', '1000']
        domain_str = os.path.splitext(os.path.basename(ckpt_path))[0].split('_', 2)[-1]
        index = domain_names.index(domain_str)
        source_dataset = load_roll_source(domain_index=index, domain_names=domain_names, data_root_dir=data_dir, transform=transform)

    elif dataset_name in ["cifar100", "cifar100_c"]:
        source_dataset = torchvision.datasets.CIFAR100(root=data_root_dir,
                                                       train=train_split,
                                                       download=True,
                                                       transform=transform)
    elif dataset_name in ["imagenet", "imagenet_c", "imagenet_k", "ccc"]:
        split = "train" if train_split else "val"
        source_dataset = torchvision.datasets.ImageNet(root=data_dir,
                                                       split=split,
                                                       transform=transform)
    elif dataset_name in ["domainnet126"]:
        src_domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
        source_data_list = [os.path.join("datasets", f"{dataset_name}_lists", f"{src_domain}_list.txt")]
        source_dataset = ImageList(image_root=data_dir,
                                   label_files=source_data_list,
                                   transform=transform)
        logger.info(f"Loading source data from list: {source_data_list[0]}")
    elif dataset_name in ["imagenet_r", "imagenet_a", "imagenet_v2", "imagenet_d", "imagenet_d109"]:
        split = "train" if train_split else "val"
        source_dataset = create_imagenet_subset(data_dir=data_dir,
                                                test_dataset_name=dataset_name,
                                                split=split,
                                                transform=transform)
    else:
        raise ValueError("Dataset not supported.")

    if percentage < 1.0 or num_samples >= 0:  # reduce the number of source samples
        assert percentage > 0.0, "The percentage of source samples has to be in range 0.0 < percentage <= 1.0"
        assert num_samples > 0, "The number of source samples has to be at least 1"
        if src_dataset_name in ["cifar10", "cifar100"]:
            nr_src_samples = source_dataset.data.shape[0]
            nr_reduced = min(num_samples, nr_src_samples) if num_samples > 0 else int(
                np.ceil(nr_src_samples * percentage))
            inds = random.sample(range(0, nr_src_samples), nr_reduced)
            source_dataset.data = source_dataset.data[inds]
            source_dataset.targets = [source_dataset.targets[k] for k in inds]
        else:
            nr_src_samples = len(source_dataset.samples)
            nr_reduced = min(num_samples, nr_src_samples) if num_samples > 0 else int(
                np.ceil(nr_src_samples * percentage))
            source_dataset.samples = random.sample(source_dataset.samples, nr_reduced)

        logger.info(
            f"Number of images in source loader: {nr_reduced}/{nr_src_samples} \t Reduction factor = {nr_reduced / nr_src_samples:.4f}")

    # create the source data loader
    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers,
                                                drop_last=False)
    logger.info(
        f"Number of images and batches in source loader: #img = {len(source_dataset)} #batches = {len(source_loader)}")
    return source_dataset, source_loader


def sort_by_dirichlet(delta_dirichlet: float, samples: list):
    """
    Adapted from: https://github.com/TaesikGong/NOTE/blob/main/learner/dnn.py
    Sort classes according to a dirichlet distribution
    Input:
        delta_dirichlet: Parameter of the distribution
        samples: List containing all data sample pairs (file_path, class_label)
    Returns:
        samples_sorted: List containing the temporally correlated samples
    """

    N = len(samples)
    samples_sorted = []
    class_labels = np.array([val[1] for val in samples])
    num_classes = int(np.max(class_labels) + 1)
    dirichlet_numchunks = num_classes

    time_start = time.time()
    time_duration = 120  # seconds until program terminates if no solution was found

    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
    min_size = -1
    min_size_thresh = 10
    while min_size < min_size_thresh:  # prevent any chunk having too less data
        idx_batch = [[] for _ in range(dirichlet_numchunks)]
        idx_batch_cls = [[] for _ in range(dirichlet_numchunks)]  # contains data per each class
        for k in range(num_classes):
            idx_k = np.where(class_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(delta_dirichlet, dirichlet_numchunks))

            # balance
            proportions = np.array(
                [p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

            # store class-wise data
            for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                idx_j.append(idx)

        # exit loop if no solution was found after a certain while
        if time.time() > time_start + time_duration:
            raise ValueError(
                f"Could not correlated sequence using dirichlet value '{delta_dirichlet}'. Try other value!")

    sequence_stats = []

    # create temporally correlated sequence
    for chunk in idx_batch_cls:
        cls_seq = list(range(num_classes))
        np.random.shuffle(cls_seq)
        for cls in cls_seq:
            idx = chunk[cls]
            samples_sorted.extend([samples[i] for i in idx])
            sequence_stats.extend(list(np.repeat(cls, len(idx))))

    return samples_sorted
