#!/bin/bash
# 调度脚本：依次运行多个方法（MY / RMT / LAW / ROID / SANTA / COTTA）
# 所有参数均保留原注释，方便你随时切换

# -----------------------
# 共享设置
# -----------------------
DOMAINS=("N09_M07_F10" "N15_M01_F10" "N15_M07_F04" "N15_M07_F10")
CKPT_PREFIX="./checkpoints/source_resnet1d"
OUT_DIR="./output/PU_13c_1d/"

#############################################
################    方法：MY    ##############
#############################################
#METHOD="my"
#CFG_FILE="cfgs/pu/${METHOD}.yaml"
#OUT_FILE="PU_${METHOD}_results.xlsx"
#
#declare -A params
#params[OPTIM.LR]="1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2"
#params[TEST.BATCH_SIZE]="32 64 128 256"
#params[MY.LAMBDA_CE_TRG]="0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0"
#params[MY.LAMBDA_CONT]="0.1 0.25 0.5 0.75 1.0 2.0 3.0"
#params[MY.NUM_SAMPLES_WARM_UP]="5000 10000 20000 50000 100000"
#params[MY.LAW_LR_BASE]="0.0001 0.0002 0.0005 0.001 0.002 0.005"
#params[MY.LAW_TAU]="0.1 0.2 0.5 1.0 1.5 2.0 3.0"
#
#echo "▶ Running METHOD = $METHOD"
#source ./run/run_general.sh "$METHOD" "$CFG_FILE" "DOMAINS[@]" "$CKPT_PREFIX" "$OUT_DIR" "$OUT_FILE"


############################################
###############    方法：RMT   ##############
############################################
METHOD="rmt"
CFG_FILE="cfgs/pu/${METHOD}.yaml"
OUT_FILE="PU_${METHOD}_results.xlsx"

declare -A params
params[OPTIM.LR]="5e-3 2e-3"
#params[TEST.BATCH_SIZE]="48"
#params[RMT.LAMBDA_CE_TRG]="0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0"
#params[RMT.LAMBDA_CONT]="0.1 0.25 0.5 0.75 1.0 2.0 3.0"
#params[RMT.NUM_SAMPLES_WARM_UP]="5000 10000 20000 50000 100000"

#params[RMT.LAMBDA_CE_TRG]="1.0"
#params[RMT.LAMBDA_CONT]="1.0"
#params[RMT.NUM_SAMPLES_WARM_UP]="50000"

echo "▶ Running METHOD = $METHOD"
source ./run/run_general.sh "$METHOD" "$CFG_FILE" "DOMAINS[@]" "$CKPT_PREFIX" "$OUT_DIR" "$OUT_FILE"


############################################
###############    方法：LAW   ##############
############################################
METHOD="law"
CFG_FILE="cfgs/pu/${METHOD}.yaml"
OUT_FILE="PU_${METHOD}_results.xlsx"

declare -A params
params[OPTIM.LR]="5e-3 2e-3"
#params[TEST.BATCH_SIZE]="48"
#params[LAW.TAU]="0.1 0.2 0.5 1.0 1.5 2.0 3.0"
#params[LAW.TAU]="1.0"

echo "▶ Running METHOD = $METHOD"
source ./run/run_general.sh "$METHOD" "$CFG_FILE" "DOMAINS[@]" "$CKPT_PREFIX" "$OUT_DIR" "$OUT_FILE"


############################################
###############    方法：ROID   #############
############################################
METHOD="roid"
CFG_FILE="cfgs/pu/${METHOD}.yaml"
OUT_FILE="PU_${METHOD}_results.xlsx"

declare -A params
params[OPTIM.LR]="5e-3 2e-3"
#params[TEST.BATCH_SIZE]="48"
#params[ROID.MOMENTUM_SRC]="0.9 0.91 0.92 0.95 0.99 0.999"
#params[ROID.MOMENTUM_PROBS]="0.8 0.85 0.9 0.95 0.99"
#params[ROID.TEMPERATURE]="0.1 0.2 0.4 0.5 0.7 0.9"

#params[ROID.MOMENTUM_SRC]="0.99"
#params[ROID.MOMENTUM_PROBS]="0.9"
#params[ROID.TEMPERATURE]="0.4"

echo "▶ Running METHOD = $METHOD"
source ./run/run_general.sh "$METHOD" "$CFG_FILE" "DOMAINS[@]" "$CKPT_PREFIX" "$OUT_DIR" "$OUT_FILE"


############################################
###############   方法：SANTA   #############
############################################
METHOD="santa"
CFG_FILE="cfgs/pu/${METHOD}.yaml"
OUT_FILE="PU_${METHOD}_results.xlsx"

declare -A params
params[OPTIM.LR]="5e-3 2e-3"
#params[TEST.BATCH_SIZE]="48"
#params[SANTA.LAMBDA_CE_TRG]="0.2 0.5 0.75 1.0 2.0 3.0"
#params[SANTA.LAMBDA_CONT]="0.2 0.5 0.75 1.0 2.0 3.0"

#params[SANTA.LAMBDA_CE_TRG]="1.0"
#params[SANTA.LAMBDA_CONT]="1.0"

echo "▶ Running METHOD = $METHOD"
source ./run/run_general.sh "$METHOD" "$CFG_FILE" "DOMAINS[@]" "$CKPT_PREFIX" "$OUT_DIR" "$OUT_FILE"


############################################
###############   方法：COTTA   #############
############################################
METHOD="cotta"
CFG_FILE="cfgs/pu/${METHOD}.yaml"
OUT_FILE="PU_${METHOD}_results.xlsx"

declare -A params
params[OPTIM.LR]="5e-3 2e-3"
#params[TEST.BATCH_SIZE]="48"
#params[COTTA.AP]="0.8 0.83 0.86 0.89 0.92 0.95 0.98"
#params[COTTA.LAMBDA_CENTROID]="0.05 0.1 0.2 0.5 0.75 1.0"
#params[COTTA.WARMUP_BATCHES]="1 2 5 10 20"
#params[COTTA.SP_PROB]="0.001 0.005 0.01 0.02 0.05"

#params[COTTA.AP]="0.92"
#params[COTTA.LAMBDA_CENTROID]="0.1"
#params[COTTA.WARMUP_BATCHES]="5"
#params[COTTA.SP_PROB]="0.01"

echo "▶ Running METHOD = $METHOD"
source ./run/run_general.sh "$METHOD" "$CFG_FILE" "DOMAINS[@]" "$CKPT_PREFIX" "$OUT_DIR" "$OUT_FILE"

