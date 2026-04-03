import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from robustbench.model_zoo.enums import BenchmarkDataset
from robustbench.zenodo_download import DownloadError, zenodo_download
from robustbench.loaders import CustomImageFolder, CustomCifarDataset, Custom1DDataset


PREPROCESSINGS = {
    'Res256Crop224': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()]),
    'Crop288': transforms.Compose([transforms.CenterCrop(288),
                                   transforms.ToTensor()]),
    'none': transforms.Compose([transforms.ToTensor()]),
}


def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor


def load_cifar10(
        n_examples: Optional[int] = None,
        data_dir: str = './data',
        prepr: Optional[str] = 'none') -> Tuple[torch.Tensor, torch.Tensor]:
    transforms_test = PREPROCESSINGS[prepr]
    dataset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               transform=transforms_test,
                               download=True)
    return _load_dataset(dataset, n_examples)


def load_cifar100(
        n_examples: Optional[int] = None,
        data_dir: str = './data',
        prepr: Optional[str] = 'none') -> Tuple[torch.Tensor, torch.Tensor]:
    transforms_test = PREPROCESSINGS[prepr]
    dataset = datasets.CIFAR100(root=data_dir,
                                train=False,
                                transform=transforms_test,
                                download=True)
    return _load_dataset(dataset, n_examples)


def load_imagenet(
        n_examples: Optional[int] = 5000,
        data_dir: str = './data',
        prepr: str = 'Res256Crop224') -> Tuple[torch.Tensor, torch.Tensor]:
    transforms_test = PREPROCESSINGS[prepr]
    imagenet = CustomImageFolder(data_dir + '/val', transforms_test)
    
    test_loader = data.DataLoader(imagenet, batch_size=n_examples,
                                  shuffle=False, num_workers=4)

    x_test, y_test, paths = next(iter(test_loader))
    
    return x_test, y_test


CleanDatasetLoader = Callable[[Optional[int], str], Tuple[torch.Tensor,
                                                          torch.Tensor]]
_clean_dataset_loaders: Dict[BenchmarkDataset, CleanDatasetLoader] = {
    BenchmarkDataset.cifar_10: load_cifar10,
    BenchmarkDataset.cifar_100: load_cifar100,
    BenchmarkDataset.imagenet: load_imagenet,
}


def load_clean_dataset(dataset: BenchmarkDataset, n_examples: Optional[int],
                       data_dir: str, prepr: Optional[str] = 'none') -> Tuple[torch.Tensor, torch.Tensor]:
    return _clean_dataset_loaders[dataset](n_examples, data_dir, prepr)


CORRUPTIONS = ("shot_noise", "motion_blur", "snow", "pixelate",
               "gaussian_noise", "defocus_blur", "brightness", "fog",
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
               "jpeg_compression", "elastic_transform")

ZENODO_CORRUPTIONS_LINKS: Dict[BenchmarkDataset, Tuple[str, Set[str]]] = {
    BenchmarkDataset.cifar_10: ("2535967", {"CIFAR-10-C.tar"}),
    BenchmarkDataset.cifar_100: ("3555552", {"CIFAR-100-C.tar"})
}

CORRUPTIONS_DIR_NAMES: Dict[BenchmarkDataset, str] = {
    BenchmarkDataset.cifar_10: "CIFAR-10-C",
    BenchmarkDataset.cifar_100: "CIFAR-100-C",
    BenchmarkDataset.imagenet: "ImageNet-C"
}


def load_roll(
    n_examples: int = -1,                     # 默认加载全部样本
    data_dir: str = './data/PU_1d_8c_2048',   # 数据所在目录
    domains: Sequence[str] = ('N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10'),
    shuffle: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 加载标签（共7200）
    labels_path = os.path.join(data_dir, 'labels.npy')
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"标签文件不存在: {labels_path}")
    labels = np.load(labels_path).astype(np.uint8)

    x_list = []
    y_list = []

    # 加载每个 domain 的图像数据
    for domain in domains:
        domain_path = os.path.join(data_dir, f'{domain}.npy')
        if not os.path.isfile(domain_path):
            raise FileNotFoundError(f"{domain}.npy 不存在！")

        data = np.load(domain_path).astype(np.uint8)  # (样本数, 32, 32, 3)
        x_list.append(data)
        y_list.append(labels)

    # 拼接所有 domain 图像（共 N 个）
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list)

    if shuffle:
        indices = np.random.permutation(len(x))
        x, y = x[indices], y[indices]

    # 格式转换：HWC -> CHW，归一化
    x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32) / 255.0  # -> (N, 3, 32, 32)
    x = torch.tensor(x)
    y = torch.tensor(y)

    # 根据 n_examples 决定是否截断
    if n_examples > 0:
        x = x[:n_examples]
        y = y[:n_examples]

    return x, y


def load_roll_1d(data_dir: str, domains: list, full_domain: list):
    """
    辅助函数：加载指定域的数据和标签
    """
    x_list = []
    y_list = []

    # 加载标签全集
    labels_path = os.path.join(data_dir, 'labels.npy')
    all_labels = np.load(labels_path)

    # 获取域名列表以确定索引
    # 注意：这里需要硬编码或者从外部传入完整的 domain_list 顺序，才能正确切分 label

    for dom in domains:
        path = os.path.join(data_dir, f"{dom}.npy")
        if not os.path.exists(path):
            continue

        data = np.load(path).astype(np.float32)  # [N, 2048]

        # 确定标签
        if dom in full_domain:
            idx = full_domain.index(dom)
            n_samples = len(data)
            # 同样处理标签切片逻辑
            if len(all_labels) == n_samples:  # 单份标签模式
                labels = all_labels
            else:  # 拼接标签模式
                labels = all_labels[idx * n_samples: (idx + 1) * n_samples]

            # ★ 修复点 1：确保 NumPy 标签是整数
            labels = labels.astype(np.int64)
        else:
            # 如果域名不在标准列表里，无法确定标签，抛错或由外部处理
            raise ValueError(f"Unknown domain {dom} for label slicing")

        x_list.append(torch.tensor(data))
        # ★ 修复点 2：将标签张量指定为 long (整数) 类型
        y_list.append(torch.tensor(labels, dtype=torch.long))

    if not x_list:
        return torch.tensor([]), torch.tensor([])

    return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)


def load_roll_source(domain_index: int = 0, domain_names = None, data_root_dir: str = './data', transform=None):
    """
    加载 PU 数据集中某个领域的样本，作为源域数据。

    参数：
        domain_index: 选取哪个领域（0~3）
        data_root_dir: 数据根目录，应包含 PU_1d_8c_2048
        transform: 图像预处理方法
    返回：
        CustomCifarDataset 对象
    """
    if domain_names is None:
        domain_names = ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10']
    domain_name = domain_names[domain_index]

    domain_path = os.path.join(data_root_dir, f'{domain_name}.npy')
    labels_path = os.path.join(data_root_dir, 'labels.npy')

    x_data = np.load(domain_path)  # (3600, 32, 32, 3)
    y_data = np.load(labels_path)  # (7200, )，你只取前 3600 就行

    # 假设每个 domain 都是按顺序存放的
    samples = [[x_data[i], y_data[i], domain_name] for i in range(len(x_data))]
    return CustomCifarDataset(samples=samples, transform=transform)


def load_roll_source_1d(domain_index: int = 0, domain_names=None, data_root_dir: str = './dataset/JNU_1d_2048_2000/JNU_4c',
                     transform=None):
    if domain_names is None:
        # 修改为你的 JNU 数据集域名
        domain_names = ['600', '800', '1000']

    domain_name = domain_names[domain_index]

    # 加载数据 [N, 2048]
    domain_path = os.path.join(data_root_dir, f'{domain_name}.npy')
    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"Source domain file not found: {domain_path}")

    x_data = np.load(domain_path).astype(np.float32)

    # 加载标签
    # 你的 labels.npy 包含了所有域的标签拼接
    # 假设每个域样本数相同 (2000)，我们需要切片获取当前域的标签
    labels_path = os.path.join(data_root_dir, 'labels.npy')
    all_labels = np.load(labels_path)

    samples_per_domain = len(x_data)
    start_idx = domain_index * samples_per_domain
    end_idx = start_idx + samples_per_domain

    # 校验边界，防止越界
    if end_idx > len(all_labels):
        # 如果 labels.npy 只是单份拷贝（即所有域共享同一套标签模板），直接取全部
        if len(all_labels) == samples_per_domain:
            y_data = all_labels
        else:
            raise ValueError(f"Label file length {len(all_labels)} mismatch with requested range {start_idx}-{end_idx}")
    else:
        y_data = all_labels[start_idx:end_idx]

    # ★ 修复点：确保标签是整数类型
    y_data = y_data.astype(np.int64)
    # 构建 samples 列表
    samples = [
        [x_data[i], int(y_data[i]), domain_name]  # <-- 关键修改：使用 int() 确保原生 Python 整数
        # 或者 [x_data[i], y_data[i].item(), domain_name]
        for i in range(len(x_data))
    ]
    # 使用新的 Dataset 类
    return Custom1DDataset(samples=samples, transform=transform)



# load_cifar10c 是 load_corruptions_cifar 的简单封装，指定数据集为 CIFAR-10。
def load_cifar10c(
        n_examples: int = 10000,            # 要加载的测试样本数
        severity: int = 5,                  # 扰动的严重程度（1~5）
        data_dir: str = './data',           # 数据所在目录
        shuffle: bool = False,              # 是否打乱样本顺序
        corruptions: Sequence[str] = CORRUPTIONS,  # 要加载的扰动类型
        prepr: Optional[str] = 'none'       # 预处理方式（未使用）
) -> Tuple[torch.Tensor, torch.Tensor]:     # 返回图像与标签张量
    return load_corruptions_cifar(BenchmarkDataset.cifar_10, n_examples,
                                  severity, data_dir, corruptions, shuffle)


def load_cifar100c(
        n_examples: int = 10000,
        severity: int = 5,
        data_dir: str = './data',
        shuffle: bool = False,
        corruptions: Sequence[str] = CORRUPTIONS,
        prepr: Optional[str] = 'none'
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_cifar(BenchmarkDataset.cifar_100, n_examples,
                                  severity, data_dir, corruptions, shuffle)


def load_imagenetc(
        n_examples: Optional[int] = 5000,
        severity: int = 5,
        data_dir: str = './data',
        shuffle: bool = False,
        corruptions: Sequence[str] = CORRUPTIONS,
        prepr: str = 'Res256Crop224'
) -> Tuple[torch.Tensor, torch.Tensor]:
    transforms_test = PREPROCESSINGS[prepr]

    assert len(corruptions) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"
    # TODO: generalize this (although this would probably require writing a function similar to `load_corruptions_cifar`
    #  or alternatively creating yet another CustomImageFolder class that fetches images from multiple corruption types
    #  at once -- perhaps this is a cleaner solution)

    data_folder_path = Path(data_dir) / CORRUPTIONS_DIR_NAMES[BenchmarkDataset.imagenet] / corruptions[0] / str(severity)
    imagenet = CustomImageFolder(data_folder_path, transforms_test)

    test_loader = data.DataLoader(imagenet, batch_size=n_examples,
                                  shuffle=shuffle, num_workers=2)

    x_test, y_test, paths = next(iter(test_loader))

    return x_test, y_test


CorruptDatasetLoader = Callable[[int, int, str, bool, Sequence[str]],
                                Tuple[torch.Tensor, torch.Tensor]]
CORRUPTION_DATASET_LOADERS: Dict[BenchmarkDataset, CorruptDatasetLoader] = {
    BenchmarkDataset.cifar_10: load_cifar10c,
    BenchmarkDataset.cifar_100: load_cifar100c,
    BenchmarkDataset.imagenet: load_imagenetc,
}


def load_corruptions_cifar(
        dataset: BenchmarkDataset,                  # 枚举类型，代表要加载的CIFAR-10/100
        n_examples: int = 10000,                    # 要加载的图像数量
        severity: int = 5,                          # 扰动等级（1~5）
        data_dir: str = './data',                   # 数据所在的根目录
        corruptions: Sequence[str] = CORRUPTIONS,   # 扰动类型列表
        shuffle: bool = False                       # 是否打乱顺序
) -> Tuple[torch.Tensor, torch.Tensor]:            # 返回两个Tensor（图像、标签）

    assert 1 <= severity <= 5                      # 校验扰动等级必须在 1~5 之间
    n_total_cifar = 10000                          # 每个 corruption + severity 的样本数固定是 10000 张

    if not os.path.exists(data_dir):               # 如果 data 文件夹不存在，就创建
        os.makedirs(data_dir)

    data_dir = Path(data_dir)                      # 转换为 Path 对象便于路径拼接
    data_root_dir = data_dir / CORRUPTIONS_DIR_NAMES[dataset]
    # 根据数据集名获得其子目录，比如CIFAR-10-C就叫'cifar10-c'

    if not data_root_dir.exists():
        # 如果数据还没下载，就从 Zenodo 链接自动下载
        zenodo_download(*ZENODO_CORRUPTIONS_LINKS[dataset], save_dir=data_dir)

    # Download labels
    labels_path = data_root_dir / 'labels.npy'     # 标签文件路径
    if not os.path.isfile(labels_path):
        raise DownloadError("Labels are missing, try to re-download them.")
    labels = np.load(labels_path)                  # 读入所有标签（50000个）


    x_test_list, y_test_list = [], []              # 用于存储所有图像和标签
    n_pert = len(corruptions)                      # 总共选了几种扰动

    for corruption in corruptions:                 # 遍历每种扰动
        corruption_file_path = data_root_dir / (corruption + '.npy')
        if not corruption_file_path.is_file():
            raise DownloadError(f"{corruption} file is missing")

        images_all = np.load(corruption_file_path) # 读取.npy文件 shape=(50000, 32, 32, 3)

        # 取出对应severity的1万张图（每个severity一万张，按顺序存放）
        images = images_all[(severity - 1) * n_total_cifar : severity * n_total_cifar]

        # 计算要从每种扰动中抽取多少张图
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])         # 加入图像列表
        y_test_list.append(labels[:n_img])         # 加入标签列表（统一的）


    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)

    if shuffle:                                     # 如果要求打乱顺序
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]


    x_test = np.transpose(x_test, (0, 3, 1, 2))     # 从[H, W, C]转换为[C, H, W]
    x_test = x_test.astype(np.float32) / 255        # 归一化到0~1之间
    x_test = torch.tensor(x_test)[:n_examples]      # 转换为Tensor，并只取前n个样本
    y_test = torch.tensor(y_test)[:n_examples]

    return x_test, y_test

