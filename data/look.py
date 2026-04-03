import numpy as np
import matplotlib.pyplot as plt
import os

# 设置路径（根据你自己的数据路径修改）
data_dir = "./CIFAR-10-C"  # 注意区分大小写
corruption_name = "gaussian_noise"
severity = 5  # 1~5级
n_total = 10000  # 每种 severity 下的数据量固定为 10000

# 1. 加载图像数据（取出对应 severity 的部分）
corruption_path = os.path.join(data_dir, corruption_name + ".npy")
images_all = np.load(corruption_path)  # shape: (50000, 32, 32, 3)

# 取出对应 severity 的数据
start_idx = (severity - 1) * n_total
end_idx = severity * n_total
images = images_all[start_idx:end_idx]  # shape: (10000, 32, 32, 3)

print(f"✅ 图像 shape: {images.shape}, dtype: {images.dtype}")

# 2. 加载 labels
labels_path = os.path.join(data_dir, "labels.npy")
labels = np.load(labels_path)  # shape: (10000,)
labels = labels[:n_total]  # 按理说与 severity 无关，是重复的

print(f"✅ 标签 shape: {labels.shape}, dtype: {labels.dtype}")
print(f"前10个标签: {labels[:10]}")

# 3. 显示前几张图像
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(images[i].astype(np.uint8))  # 注意：原始就是uint8格式的图像（0~255）
    plt.title(str(labels[i]))
    plt.axis("off")

plt.tight_layout()
plt.show()
