# TTA_idea

## Introduction

This project focuses on **bearing fault diagnosis under source-free test-time adaptation (TTA) settings**.

To move beyond conventional offline methods, this work considers a more challenging scenario:
**online adaptation with streaming data**, where only **single-sample input** is available and **no source data can be accessed**.

To address these challenges, we propose:

> **PA-PLR (Prototype Alignment with Parameter-wise Learning Rate)**

The method is designed to:

* Enable **online and continuous model adaptation**
* Handle **single-sample updates**
* Mitigate **catastrophic forgetting**

Key ideas include:

* **Prototype alignment** for stable feature adaptation
* **EMA teacher model** for reliable pseudo supervision
* **Parameter-wise adaptive learning rate** to improve optimization robustness

---

## Datasets

Please manually download the following datasets:

* PU dataset

Organize them under `./data` as follows:

---

## Requirements

Recommended environment:

* python == 3.11
* pytorch >= 2.3
* numpy
* scipy
* sklearn
* tqdm

---

## How to Run

### 1. Source Model Training

```bash id="r8n3fj"
python source_train.py
```

⚠️ Please modify the configuration (e.g., dataset path, hyperparameters) inside the script before running.

---

### 2. Target Domain Adaptation

```bash id="p3xk2q"
bash run/pu_my.sh
```

⚠️ You can adjust parameters (e.g., learning rate, epochs) in the `.sh` file as needed.

---

## Notes

* This project focuses on **test-time adaptation without source data**
* The method supports **online inference and updating**
* Make sure dataset paths are correctly configured before running

---

## Contact

If you have any questions, feel free to open an issue.

- [bd_lab@163.com](mailto:bd_lab@163.com)
