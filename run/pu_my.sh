#!/bin/bash

# 全局变量：domains
METHOD="my"  # ←←← 只需要改这里
CFG_FILE="cfgs/pu/${METHOD}.yaml"
OUT_DIR="./output/PU_13c_1d/"
OUT_FILE="PU_${METHOD}_results.xlsx"
domains=("N09_M07_F10" "N15_M01_F10" "N15_M07_F04" "N15_M07_F10")

# 新增变量：MODEL CKPT 前缀路径（避免硬编码）
CKPT_PREFIX="./checkpoints/source_resnet1d"

# 参数定义
declare -A params
params[OPTIM.LR]="5e-4"
params[TEST.BATCH_SIZE]="64"
params[MY.LAMBDA_CE_TRG]="1.0"
params[MY.LAMBDA_CONT]="1.0"
params[MY.NUM_SAMPLES_WARM_UP]="5000"
params[MY.LAW_LR_BASE]="0.0005"
params[MY.LAW_TAU]="1.0"

# 参数 key
param_keys=("${!params[@]}")
echo "${param_keys[@]}"

# 遍历参数
for param_name in "${!params[@]}"; do
    values="${params[$param_name]}"

    for val in $values; do
        echo "🔹 Running with $param_name=$val"

        timestamp=$(date +"%y%m%d_%H%M%S")
        echo "⚙️ Running param $param_name over value: $val (timestamp=$timestamp)"

        # -------------------------一轮实验------------------------
        for i in "${!domains[@]}"; do
            source="${domains[$i]}"

            # 生成目标域
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

            # ------------------ 清空 warmup 和 prototypes ------------------
            echo "🧹 Cleaning warmup and prototype checkpoints..."
            rm -f ckpt/warmup/*
            rm -f ckpt/prototypes/*
            # ---------------------------------------------------------------

            echo "🚀 Running test_time.py with $param_name=$val, source=$source, targets=$targets_str"

            python test_time.py \
                --cfg "$CFG_FILE" \
                LOG_TIME $timestamp \
                LOG_SUFFIX $((i+1)) \
                MODEL.CKPT_PATH "${CKPT_PREFIX}_${source}.pth" \
                CORRUPTION.TYPE "$targets_str" \
                $param_name $val
        done
        # -------------------------一轮实验结束----------------------------

        python parse_logs.py "$timestamp" "$OUT_DIR" "$METHOD" "$OUT_FILE" "${param_keys[@]}"

    done
done
