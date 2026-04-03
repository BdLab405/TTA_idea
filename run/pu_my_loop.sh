#!/bin/bash

# 全局变量：domains
METHOD="my"
CFG_FILE="cfgs/pu/${METHOD}.yaml"
OUT_DIR="./output/PU_13c_1d/"
OUT_FILE="PU_${METHOD}_results.xlsx"
domains=("N09_M07_F10" "N15_M01_F10" "N15_M07_F04" "N15_M07_F10")

# 新增变量：MODEL CKPT 前缀路径
CKPT_PREFIX="./checkpoints/source_resnet1d"

# ==================== 1. 参数定义 ====================
declare -A params
# 在这里定义你要进行全排列组合的参数
params[OPTIM.LR]="1e-3 2e-3 5e-3 7e-3 1e-2"
params[TEST.BATCH_SIZE]="48 64 96"
#params[MY.LAMBDA_CE_TRG]="0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0"
#params[MY.LAMBDA_CONT]="0.1 0.25 0.5 0.75 1.0 2.0 3.0"
params[MY.NUM_SAMPLES_WARM_UP]="30000 50000 70000 100000"
params[MY.LAW_LR_BASE]="0.0001 0.001 0.005"

#params[MY.LAW_TAU]="0.1 0.2 0.5 1.0 1.5 2.0 3.0"

# 获取所有参数的键名 (例如: OPTIM.LR TEST.BATCH_SIZE ...)
param_keys=("${!params[@]}")

echo "📋 Detected parameters for Grid Search: ${param_keys[@]}"
echo "------------------------------------------------------"

# ==================== 2. 递归函数定义 ====================
# 函数：run_grid_search
# 参数1: 递归深度 (当前处理第几个参数)
# 参数2: 累积的命令行参数字符串
function run_grid_search() {
    local depth=$1
    local current_args=$2

    # --- 基准情况：所有参数都已遍历完，执行实验 ---
    if [ "$depth" -eq "${#param_keys[@]}" ]; then

        timestamp=$(date +"%y%m%d_%H%M%S")
        echo "🔥 Running Combination: $current_args (Timestamp: $timestamp)"

        # ------------------------- 域循环 (Domains Loop) ------------------------
        for i in "${!domains[@]}"; do
            source="${domains[$i]}"

            # 生成目标域字符串 (Source/Target Logic)
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
            # 注意：在全排列组合中，每次组合可能都需要清理，视你的逻辑而定
            # echo "🧹 Cleaning warmup and prototype checkpoints..."
            rm -f ckpt/warmup/*
            rm -f ckpt/prototypes/*
            # ---------------------------------------------------------------

            echo "🚀 [Source: $source] Executing test_time.py..."

            # 执行 Python 脚本，传入累积的参数 ($current_args)
            python test_time.py \
                --cfg "$CFG_FILE" \
                LOG_TIME $timestamp \
                LOG_SUFFIX $((i+1)) \
                MODEL.CKPT_PATH "${CKPT_PREFIX}_${source}.pth" \
                CORRUPTION.TYPE "$targets_str" \
                $current_args
        done
        # ------------------------- 实验结束 ----------------------------

        # 解析日志
        # 注意：这里传给 parse_logs 的 param_keys 是所有的 key，
        # 具体的 value 应该包含在 $timestamp 对应的 log 文件里，
        # 或者你需要修改 parse_logs.py 来适应这种情况。
        python parse_logs.py "$timestamp" "$OUT_DIR" "$METHOD" "$OUT_FILE" "${param_keys[@]}"

        return
    fi

    # --- 递归步骤：遍历当前层级参数的所有值 ---
    local key="${param_keys[$depth]}"
    local values="${params[$key]}"

    for val in $values; do
        # 递归调用：深度+1，将当前 "key value" 追加到参数字符串中
        run_grid_search $((depth + 1)) "$current_args $key $val"
    done
}

# ==================== 3. 开始执行 ====================
# 从第0个参数开始，初始参数字符串为空
run_grid_search 0 ""