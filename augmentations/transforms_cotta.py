# KATANA: Simple Post-Training Robustness Using Test Time Augmentations
# https://arxiv.org/pdf/2109.08191v1.pdf
import PIL
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, Compose, Lambda
from numpy import random

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn_like(img) * self.std + self.mean
        return torch.clamp(img + noise, 0.0, 1.0)


class Clip(torch.nn.Module):
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clamp(img, self.min_val, self.max_val)


class SaltAndPepperNoise(torch.nn.Module):
    """Salt-and-pepper noise: 随机将像素置 0 或 1"""
    def __init__(self, prob=0.01):
        super().__init__()
        self.prob = prob

    def forward(self, img):
        mask = torch.rand_like(img)
        img = torch.where(mask < self.prob / 2, torch.zeros_like(img),
                          torch.where(mask > 1 - self.prob / 2, torch.ones_like(img), img))
        return img


class PoissonNoise(torch.nn.Module):
    """Poisson noise: 泊松噪声"""
    def forward(self, img):
        vals = 2 ** torch.ceil(torch.log2(torch.tensor(len(torch.unique(img))) + 1))
        noisy = torch.poisson(img * vals) / vals
        return torch.clamp(noisy, 0.0, 1.0)


def get_tta_transforms(img_size, gaussian_std=0.01, sp_prob=0.01,
                       soft=False, padding_mode='edge', cotta_augs=True):
    """
    改写版数据增强，符合 Online Adaptive Fault Diagnosis 论文：
    - 仅使用三种噪声增强（Gaussian, Salt-and-Pepper, Poisson）
    - 不再进行颜色扰动或仿射变换
    兼容旧参数 soft, padding_mode, cotta_augs（保留接口但不使用）
    """
    return Compose([
        Clip(0.0, 1.0),
        GaussianNoise(0, gaussian_std),
        SaltAndPepperNoise(prob=sp_prob),
        PoissonNoise(),
        Clip(0.0, 1.0)
    ])

