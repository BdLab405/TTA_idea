#!/bin/bash

# 全局变量：domains
METHOD="cotta"  # ←←← 只需要改这里！改成别的名字即可（如 "law"、"rmt"）
CFG_FILE="cfgs/pu/${METHOD}.yaml"
OUT_DIR="./output/PU_13c_1d/"
OUT_FILE="PU_${METHOD}_results.xlsx"
domains=("N09_M07_F10" "N15_M01_F10" "N15_M07_F04" "N15_M07_F10")

# 参数定义（字典方式）
declare -A params
#params[COTTA.AP]="0.8 0.83 0.86 0.89 0.92 0.95 0.98"
#params[COTTA.LAMBDA_CENTROID]="0.05 0.1 0.2 0.5 0.75 1.0"
#params[COTTA.WARMUP_BATCHES]="1 2 5 10 20"
#params[COTTA.SP_PROB]="0.001 0.005 0.01 0.02 0.05"
params[COTTA.AP]="0.92"
params[COTTA.LAMBDA_CENTROID]="0.1"
params[COTTA.WARMUP_BATCHES]="5"
params[COTTA.SP_PROB]="0.01"


# 获取参数key列表（自动）
param_keys=("${!params[@]}")
echo "${param_keys[@]}"

# 遍历参数
for param_name in "${!params[@]}"; do
    values="${params[$param_name]}"

    # 遍历参数取值
    for val in $values; do
        echo "🔹 Running with $param_name=$val"

        # 每个参数循环一个统一的时间戳
        timestamp=$(date +"%y%m%d_%H%M%S")
        echo "⚙️ Running param $param_name over value: $val (timestamp=$timestamp)"

        # -------------------------一轮实验------------------------
        for i in "${!domains[@]}"; do
            source="${domains[$i]}"

            # 目标域 = 除了当前源域的其他域
            targets_str="["
            for j in "${!domains[@]}"; do
                if [ "$i" -ne "$j" ]; then
                    if [ "$targets_str" = "[" ]; then
                        targets_str="'${domains[$j]}'"
                    else
                        targets_str="$targets_str,'${domains[$j]}'"
                    fi
                fi
            done
            targets_str="[$targets_str]"

            echo "🧹 Cleaning warmup and prototype checkpoints..."
#            rm -f ckpt/warmup/ckpt_warmup_PU_13c_resnet18_bs64.pth
#            rm -f ckpt/prototypes/protos_PU_13c_resnet18.pth
            rm -f ckpt/warmup/ckpt_warmup_PU_13c_1d_resnet1d_8_bs128.pth
            rm -f ckpt/prototypes/protos_PU_13c_1d_resnet1d_8.pth

            echo "🚀 Running test_time.py with $param_name=$val, source=$source, targets=$targets_str"
            python test_time.py \
                --cfg "$CFG_FILE" \
                LOG_TIME $timestamp \
                LOG_SUFFIX $((i+1)) \
                MODEL.CKPT_PATH ./checkpoints/source_resnet1d_${source}.pth \
                CORRUPTION.TYPE "$targets_str" \
                $param_name $val
        done
        # -------------------------一轮实验结束------------------------

        # 🧩 自动解析日志，传入所有参数key
        python parse_logs.py "$timestamp" "$OUT_DIR" "$METHOD" "$OUT_FILE" "${param_keys[@]}"

    done
done
