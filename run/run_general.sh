#!/bin/bash
# 通用脚本：执行单个 METHOD 的完整实验流程

METHOD="$1"                # 方法名，如 my, rmt, law
CFG_FILE="$2"              # cfg 路径
# declare -A params=()     <-- 🔴 必须注释掉或删除这一行！否则会清空父脚本传来的参数！
# params 变量现在直接从父脚本继承

# 处理 Domains (间接引用)
# 因为使用了 source，DOMAINS 变量在当前作用域可见，所以 ${!3} 这种间接引用现在可以正常工作了
domains_ref="$3"
domains=("${!domains_ref}")
CKPT_PREFIX="$4"           # checkpoint 前缀
OUT_DIR="$5"
OUT_FILE="$6"

# ------- 输出参数检查 -------
echo "🔧 METHOD = $METHOD"
echo "🔧 CFG_FILE = $CFG_FILE"
echo "🔧 CKPT_PREFIX = $CKPT_PREFIX"
echo "🔧 OUT_DIR = $OUT_DIR"
echo "🔧 OUT_FILE = $OUT_FILE"
echo "🔧 Domains: ${domains[*]}"

# ------- 获取参数 key -------
param_keys=("${!params[@]}")
echo "📝 Parameters found: ${param_keys[*]}"

# ------- 遍历参数组合 -------
for param_name in "${!params[@]}"; do
    values="${params[$param_name]}"

    for val in $values; do
        echo "🔹 Running with $param_name=$val"

        timestamp=$(date +"%y%m%d_%H%M%S")

        # ------- 遍历 domains -------
        for i in "${!domains[@]}"; do
            source="${domains[$i]}"

            # 生成 targets 列表（排除当前 source）
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

            echo "🧹 Cleaning checkpoints..."
            rm -f ckpt/warmup/*
            rm -f ckpt/prototypes/*

            echo "🚀 Running test_time.py..."
            python test_time.py \
                --cfg "$CFG_FILE" \
                LOG_TIME $timestamp \
                LOG_SUFFIX $((i+1)) \
                MODEL.CKPT_PATH "${CKPT_PREFIX}_${source}.pth" \
                CORRUPTION.TYPE "$targets_str" \
                $param_name $val
        done

        # ------- 解析日志 -------
        python parse_logs.py \
            "$timestamp" "$OUT_DIR" "$METHOD" "$OUT_FILE" "${param_keys[@]}"

    done
done
