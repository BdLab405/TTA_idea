import os
import re
import sys
import pandas as pd

# ------------------------------------------------------
# 🔹 从日志中提取指定参数（支持命名空间）
# ------------------------------------------------------
def extract_params_from_log(log_path, target_keys):
    """
    从日志文件中提取目标参数（如 MY.LAMBDA_CE_TRG）。
    target_keys 是完整带命名空间的键名列表。
    """
    params_found = {}
    current_namespace = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 命名空间块，如 "MY:" 或 "RMT:"
            ns_match = re.match(r"^([A-Z_]+):$", line)
            if ns_match:
                current_namespace = ns_match.group(1)
                continue

            # 参数行，如 "LAMBDA_CE_TRG: 1.0"
            kv_match = re.match(r"^([A-Z0-9_]+):\s*([^\s#]+)", line)
            if kv_match and current_namespace:
                key_full = f"{current_namespace}.{kv_match.group(1)}"
                val = kv_match.group(2)

                if key_full in target_keys:
                    # 尝试自动类型转换
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                    params_found[key_full] = val

    return params_found

# ------------------------------------------------------
# 🔹 日志解析主逻辑
# ------------------------------------------------------
def parse_logs(timestamp, log_dir, target_keys, method=None):
    """
    从日志目录中解析所有与 timestamp 匹配的日志文件，
    提取精度、参数信息等。
    """
    # 支持递归查找（防止不同方法分文件夹）
    log_files = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if timestamp in f and f.endswith(".txt"):
                log_files.append(os.path.join(root, f))

    log_files = sorted(log_files)
    if not log_files:
        raise FileNotFoundError(f"No log files found for timestamp {timestamp} in {log_dir}")

    dataset, results = None, {}

    # 遍历日志文件
    for log_path in log_files:
        src_domain = None

        # 提取参数信息（命名空间支持）
        params_found = extract_params_from_log(log_path, target_keys)

        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                # 数据集
                if "DATASET:" in line and dataset is None:
                    match = re.search(r"DATASET:\s*(\S+)", line)
                    if match:
                        dataset = match.group(1)

                # 源域
                if "Successfully restored model from:" in line:
                    match = re.search(r"source_[^_]+_(\S+)\.pth", line)
                    if match:
                        src_domain = match.group(1)

                # 精度信息
                if "accuracy %" in line:
                    match = re.search(r"\[(\w+)\].*:\s*([\d.]+)%", line)
                    if match and src_domain:
                        tgt_domain = match.group(1)[:-1]  # 去掉末尾数字
                        acc = float(match.group(2))
                        key = f"{src_domain}→{tgt_domain}"
                        results[key] = acc

        # 每个日志一个记录（包含参数）
        data = {
            "timestamp": timestamp,
            "method": method if method else "all",
            "dataset": dataset,
        }
        data.update(params_found)
        data.update(results)

    return pd.DataFrame([data])


# ------------------------------------------------------
# 🔹 主入口
# ------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python parse_logs.py <timestamp> <log_dir> <output_file> <method> [param_keys ...]")
        sys.exit(1)

    timestamp = sys.argv[1]
    log_dir = sys.argv[2]
    method = sys.argv[3]
    out_file = sys.argv[4]
    target_keys = sys.argv[5:]  # 可变参数列表（如 ["MY.LAMBDA_CE_TRG", "MY.LAMBDA_CONT", ...]）

    # 拼接路径：log_dir/method/out_file
    save_dir = os.path.join(log_dir)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, out_file)

    # 解析日志
    df = parse_logs(timestamp, log_dir, target_keys, method)

    # 写入/追加
    if os.path.exists(out_path):
        old_df = pd.read_excel(out_path)
        new_df = pd.concat([old_df, df], ignore_index=True)
        new_df.to_excel(out_path, index=False)
        print(f"✅ Appended results for {timestamp} into {out_path}")
    else:
        df.to_excel(out_path, index=False)
        print(f"✅ Created {out_path} with results for {timestamp}")
