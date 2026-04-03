import os
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from robustbench.model_zoo.architectures.utils_architectures import normalize_model

# === 配置参数 ===
DATA_DIR = './data/PU_13c'  # PHM PU_13c
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_CLASSES = 13
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels_file = 'labels.npy'

# === 你要训练的所有域 ===
domain_files_list = [
    'N09_M07_F10.npy',
    'N15_M01_F10.npy',
    'N15_M07_F04.npy',
    'N15_M07_F10.npy'
]

# domain_files_list = [
#     '30hz.npy',
#     '35hz.npy',
#     '40hz.npy',
#     '45hz.npy'
# ]

# domain_files_list = [
#     '600.npy',
#     '800.npy',
#     '1000.npy'
# ]

# === 数据集定义 ===
class PUDataset(Dataset):
    def __init__(self, data_paths, labels_path, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        for path in data_paths:
            x = np.load(os.path.join(DATA_DIR, path))  # shape: [N, 32, 32, 3]
            self.data.append(x)

        self.data = np.concatenate(self.data, axis=0)  # shape: [Total_N, 32, 32, 3]
        self.labels = np.load(os.path.join(DATA_DIR, labels_path))
        assert len(self.data) == len(self.labels), "❌ 数据与标签长度不一致"

        print(f"✅ 加载数据 shape: {self.data.shape}, 标签 shape: {self.labels.shape}")
        print("标签分布:", dict(Counter(self.labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        # img = np.uint8(img * 255)
        img = transforms.ToPILImage()(img)

        if self.transform:
            img = self.transform(img)

        return img, label

# === 公共 transform ===
transform = transforms.Compose([
    transforms.ToTensor()
])

# === 遍历每个域，单独训练并保存模型 ===
mean = [0, 0, 0]
std = [1, 1, 1]

for domain_file in domain_files_list:
    print(f"\n🚀 正在训练域：{domain_file}")

    save_path = f'./source_resnet18_{domain_file.replace(".npy", "")}.pth'
    domain_files = [domain_file]

    # 构建数据集和加载器
    full_dataset = PUDataset(domain_files, labels_file, transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 构建模型
    model = torchvision.models.resnet18()
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = normalize_model(model, mean, std).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # === 训练 ===
    for epoch in tqdm(range(NUM_EPOCHS), desc=f"[{domain_file}] Epochs"):
        model.train()
        total_loss = 0.0
        total_correct = 0

        # 去掉内部 tqdm
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        avg_loss = total_loss / len(train_dataset)
        acc = total_correct / len(train_dataset)
        # print(f"✅ Train Loss: {avg_loss:.4f}, Accuracy: {acc*100:.2f}%")

    # === 测试 ===
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    acc = correct / len(test_dataset)
    print(f"🎯 Test Accuracy on {domain_file}: {acc*100:.2f}%")

    # === 保存模型 ===
    torch.save(model.state_dict(), save_path)
    print(f"💾 模型已保存到 {save_path}")
