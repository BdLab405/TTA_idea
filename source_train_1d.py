import os
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# === 配置参数 ===
DATA_DIR = './data/PU_13c_1d'  # PHM PU_13c
BATCH_SIZE = 64
NUM_EPOCHS = 10
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


# ============================================================================
# ★ 1. 模型：ResNet-8 (1D)
# ============================================================================
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


# ============================================================================
# ★ 2. Dataset
# ============================================================================
class PUDataset(Dataset):
    def __init__(self, data_paths, labels_path):
        self.data_list = []
        self.label_list = []

        base_labels = np.load(os.path.join(DATA_DIR, labels_path))

        for path in data_paths:
            file_path = os.path.join(DATA_DIR, path)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"数据文件未找到: {file_path}")

            d = np.load(file_path)
            self.data_list.append(d)

            if len(d) != len(base_labels):
                raise ValueError(f"文件 {path} 的样本数 ({len(d)}) 与标签文件 ({len(base_labels)}) 不一致！")

            self.label_list.append(base_labels)

        self.data = np.concatenate(self.data_list, axis=0)
        self.labels = np.concatenate(self.label_list, axis=0)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # print(f"✅ 数据加载完成: Data Shape {self.data.shape} (N, Length)")
        # print("标签分布:", dict(Counter(self.labels.numpy())))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]

        signal = signal.unsqueeze(0)
        mean = signal.mean()
        std = signal.std() + 1e-8
        signal = (signal - mean) / std

        return signal, label


# ============================================================================
# ★ 3. 新增：跨域测试函数
# ============================================================================
def evaluate_on_other_domains(model, current_domain, all_domains):
    """
    ★ NEW：训练完一个域后，用最佳模型测试其它域的完整数据集
    """
    model.eval()
    print("\n📌 开始跨域测试（其他域的测试集准确率）：")

    for domain in all_domains:
        if domain == current_domain:
            continue

        test_dataset = PUDataset([domain], labels_file)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                correct += (outputs.argmax(dim=1) == targets).sum().item()

        acc = correct / len(test_dataset)
        print(f"  🔹 目标域 {domain} | 测试准确率: {acc*100:.2f}%")


# ============================================================================
# ★ 4. 主训练流程（加入跨域评估）
# ============================================================================
if __name__ == '__main__':
    save_dir = './re_checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    for domain_file in domain_files_list:
        print(f"\n🚀 正在训练域：{domain_file}")

        model_name = f'source_resnet1d_{domain_file.replace(".npy", "")}.pth'
        save_path = os.path.join(save_dir, model_name)

        current_domains = [domain_file]
        full_dataset = PUDataset(current_domains, labels_file)

        train_size = int(0.6 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        generator = torch.Generator().manual_seed(42)
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator=generator
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = ResNet1D_8(num_classes=NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        best_acc = 0.0

        # -------------------------------
        # 训练
        # -------------------------------
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0.0
            total_correct = 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)

            for inputs, targets in loop:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                total_correct += (outputs.argmax(dim=1) == targets).sum().item()

                loop.set_postfix(loss=loss.item())

            # 测试
            model.eval()
            test_correct = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    test_correct += (outputs.argmax(dim=1) == targets).sum().item()

            test_acc = test_correct / len(test_dataset)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), save_path)

        print(f"✅ 域 {domain_file} 训练结束 | 最佳测试精度: {best_acc*100:.2f}%")
        print(f"💾 模型已保存到: {save_path}")

        # ★ NEW：加载最佳模型，做跨域测试
        best_model = ResNet1D_8(num_classes=NUM_CLASSES).to(DEVICE)
        best_model.load_state_dict(torch.load(save_path, map_location=DEVICE))

        evaluate_on_other_domains(best_model, domain_file, domain_files_list)
