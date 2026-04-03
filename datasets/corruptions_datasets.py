
import os
import json
import torch
import logging
from typing import Optional, Sequence

from robustbench.data import CORRUPTIONS, PREPROCESSINGS, load_cifar10c, load_cifar100c, load_roll, load_roll_1d
from robustbench.loaders import CustomImageFolder, CustomCifarDataset, Custom1DDataset

logger = logging.getLogger(__name__)


def create_roll_dataset(
        data_dir: str = './data/PU_1d_8c_2048',
        domain_name: str = "N09_M07_F10",
        domain_names: Sequence[str] = ('N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10'),
        transform=None,
        setting: str = 'continual'):
    """
    创建 PU 数据集对象（仿照 CIFAR-C 方式）

    参数：
    - data_dir: 存放 .npy 文件的目录
    - domains: 要使用的所有领域名（应与 .npy 文件对应）
    - transform: 可选的 transform（用于测试阶段）
    - setting: continual 或 mixed_domains 等模式

    返回：
    - CustomCifarDataset 对象
    """

    domain_names = domain_names if "mixed_domains" in setting else [domain_name]

    x_test = torch.tensor([])
    y_test = torch.tensor([])
    domain = []

    for dom in domain_names:
        x_tmp, y_tmp = load_roll(data_dir=data_dir, domains=[dom])
        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        domain += [dom] * x_tmp.shape[0]

    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()

    samples = [[x_test[i], y_test[i], domain[i]] for i in range(x_test.shape[0])]
    return CustomCifarDataset(samples=samples, transform=transform)


def create_roll_dataset_1d(
        data_dir: str = './dataset/JNU_1d_2048_2000/JNU_4c',
        domain_name: str = "600",
        domain_names: list = ['600', '800', '1000'],
        transform=None,
        setting: str = 'continual'):
    # 确定要加载哪些域
    if "mixed_domains" in setting:
        target_domains = domain_names
    else:
        target_domains = [domain_name]

    x_test = torch.tensor([])
    y_test = torch.tensor([])
    domain_labels = []

    for dom in target_domains:
        # 调用上面的辅助函数
        x_tmp, y_tmp = load_roll_1d(data_dir=data_dir, domains=[dom], full_domain=domain_names)

        if len(x_tmp) > 0:
            x_test = torch.cat([x_test, x_tmp], dim=0)
            y_test = torch.cat([y_test, y_tmp], dim=0)
            domain_labels += [dom] * x_tmp.shape[0]

    # ★ 移除原本的 transpose 操作，1D 数据不需要 (H, W, C) -> (C, H, W)
    # x_test shape is already [Total_N, 2048]

    # 转回 numpy 以便存入 list
    x_test_np = x_test.numpy()
    y_test_np = y_test.numpy()

    # ★ 修复点：确保标签在存入 samples 列表时是原生 Python 整数 (int)
    samples = [
        [x_test_np[i], int(y_test_np[i]), domain_labels[i]]
        for i in range(len(x_test_np))
    ]

    return Custom1DDataset(samples=samples, transform=transform)


def create_cifarc_dataset(
        dataset_name: str = 'cifar10_c',
        severity: int = 5,
        data_dir: str = './data',
        corruption: str = "gaussian_noise",
        corruptions_seq: Sequence[str] = CORRUPTIONS,
        transform=None,
        setting: str = 'continual'):
    """
    构造 CIFAR-10-C 或 CIFAR-100-C 数据集对象（用于测试）

    参数说明：
    - dataset_name: 数据集名称，支持 'cifar10_c' 和 'cifar100_c'
    - severity: 扰动强度（1~5）
    - data_dir: 数据存储的根目录
    - corruption: 当前扰动类型（如 gaussian_noise）
    - corruptions_seq: 所有扰动序列（通常用于 mixed_domains）
    - transform: 图像预处理方法
    - setting: 当前实验设定（如 continual、mixed_domains 等）

    返回：
    - CustomCifarDataset 对象，用于 PyTorch DataLoader 加载
    """

    domain = []  # 用于记录每张图片的扰动类型（domain）
    x_test = torch.tensor([])  # 保存所有测试图像数据
    y_test = torch.tensor([])  # 保存所有图像对应的标签

    # 如果是 mixed_domains 模式，则使用多个 corruption；否则只使用当前指定的 corruption
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]

    # 遍历每一个扰动类型
    for cor in corruptions_seq:
        if dataset_name == 'cifar10_c':
            # 加载特定 corruption 的 CIFAR-10-C 测试集图像和标签
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
        elif dataset_name == 'cifar100_c':
            # 加载 CIFAR-100-C
            x_tmp, y_tmp = load_cifar100c(severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported!")

        # 将当前 corruption 的样本添加到总样本中
        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        # 记录每张图像对应的 corruption 类型（即 domain 名称）
        domain += [cor] * x_tmp.shape[0]

    # 转换为 NumPy 格式，并调整通道维度为 HWC（高度、宽度、通道）
    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()

    # 将每个样本打包为一个三元组：[图像，标签，domain]
    samples = [[x_test[i], y_test[i], domain[i]] for i in range(x_test.shape[0])]

    # 返回自定义的 CIFAR 数据集对象，支持 transform
    return CustomCifarDataset(samples=samples, transform=transform)


def create_imagenetc_dataset(
    n_examples: Optional[int] = -1,
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    # create the dataset which loads the default test list from robust bench containing 5000 test samples
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]
    corruption_dir_path = os.path.join(data_dir, corruptions_seq[0], str(severity))
    dataset_test = CustomImageFolder(corruption_dir_path, transform)

    if "mixed_domains" in setting or "correlated" in setting or n_examples != -1:
        # load imagenet class to id mapping from robustbench
        with open(os.path.join("robustbench", "data", "imagenet_class_to_id_map.json"), 'r') as f:
            class_to_idx = json.load(f)

        if n_examples != -1 or "correlated" in setting:
            # create file path of file containing all 50k image ids
            file_path = os.path.join("datasets", "imagenet_list", "imagenet_val_ids_50k.txt")
        else:
            # create file path of default test list from robustbench
            file_path = os.path.join("robustbench", "data", "imagenet_test_image_ids.txt")

        # load file containing file ids
        with open(file_path, 'r') as f:
            fnames = f.readlines()

        item_list = []
        for cor in corruptions_seq:
            corruption_dir_path = os.path.join(data_dir, cor, str(severity))
            item_list += [(os.path.join(corruption_dir_path, fn.split('\n')[0]), class_to_idx[fn.split(os.sep)[0]]) for fn in fnames]
        dataset_test.samples = item_list

    return dataset_test
