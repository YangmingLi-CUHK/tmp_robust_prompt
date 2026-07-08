#!/bin/bash
# =============================================================================
# Meeting 14 实验矩阵
# GPPT Baseline + GCL Linear Probe (attacked) + RobustPrompt-T 单 Filter 调参
# =============================================================================
# 用法:
#   bash run_all_experiments.sh              # 直接运行（前台）
#   nohup bash run_all_experiments.sh &      # 后台运行，日志在 nohup.out
#
# 实验量:
#   GPPT baseline:                 2 BB × 6 ptb × 5 seeds =   60
#   GCL Linear Probe (attacked):  2 BB × 5 ptb             =   10
#   RobustPrompt-T sim_pt:        2 BB × 5 val × 6 ptb × 5 =  300
#   RobustPrompt-T degree_pt:     2 BB × 4 val × 6 ptb × 5 =  240
#   RobustPrompt-T out_detect_pt: 2 BB × 5 val × 6 ptb × 5 =  300
#   ─────────────────────────────────────────────────────────────
#   Total:                                                    910 runs
#   预计耗时: ~30-60s/run × 910 ≈ 8-15 小时 (RTX 5090)
# =============================================================================

set -euo pipefail

# ============================== Config ==============================
STABLE_BB="./pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_permE.aug2_dropN.lr_0.001.ratio_0.2.seed_1.pth"
PEAK_BB="./pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_permE.aug2_maskN.lr_0.001.ratio_0.3.seed_1.pth"

PTB_RATES=("0.0" "0.05" "0.1" "0.15" "0.2" "0.25")
SEEDS="1 2 3 4 5"
DEVICE=0
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Base params (Meeting 13 optimal: lr=0.01, pt=0.25, kl=0.001, no MSE)
BASE_PARAMS="--task NodeTask --dataset_name Cora --preprocess_method none \
  --gnn_type GCN --hid_dim 256 --num_layer 2 --epochs 200 \
  --shot_num 5 --run_split 1 --filter_mode original \
  --prompt_lr 0.01 --pt_threshold 0.25 \
  --weight_mse 0.0 --weight_kl 0.001 --weight_constraint 0.2"

mkdir -p logs/baselines
mkdir -p logs/single_filter_sim
mkdir -p logs/single_filter_degree
mkdir -p logs/single_filter_ood

run_cmd() {
    local label="$1"
    local cmd="$2"
    echo "[$(date +%H:%M:%S)] $label"
    echo "  $cmd"
    eval "$cmd"
    echo ""
}

# ========================== Section 1: GPPT Baseline ==========================
echo "=========================================================="
echo " SECTION 1: GPPT Baseline (60 runs)"
echo "=========================================================="

for BB_LABEL in "stable" "peak"; do
    if [ "$BB_LABEL" = "stable" ]; then
        BB_PATH="$STABLE_BB"
    else
        BB_PATH="$PEAK_BB"
    fi

    for PTB in "${PTB_RATES[@]}"; do
        LOG="logs/baselines/gppt_${BB_LABEL}_${PTB}_${TIMESTAMP}.log"
        CMD="CUDA_VISIBLE_DEVICES=$DEVICE python MyTask.py \
          --pre_train_model_path '$BB_PATH' \
          --prompt_type GPPT \
          --attack_downstream --specified --attack_method Meta_Self-${PTB} \
          --seed $SEEDS \
          $BASE_PARAMS \
          > $LOG 2>&1"
        run_cmd "GPPT $BB_LABEL ptb=$PTB" "$CMD"
    done
done

# ===================== Section 2: GCL Linear Probe (Attacked) =====================
echo "=========================================================="
echo " SECTION 2: GCL Linear Probe on Attacked Graphs (10 runs)"
echo "=========================================================="

for BB_LABEL in "stable" "peak"; do
    if [ "$BB_LABEL" = "stable" ]; then
        BB_PATH="$STABLE_BB"
    else
        BB_PATH="$PEAK_BB"
    fi

    # ptb=0.0 is same as clean — already evaluated
    for PTB in "0.05" "0.1" "0.15" "0.2" "0.25"; do
        LOG="logs/baselines/gcl_lp_${BB_LABEL}_${PTB}_${TIMESTAMP}.log"
        CMD="CUDA_VISIBLE_DEVICES=$DEVICE python eval_pretrain.py \
          --checkpoint '$BB_PATH' \
          --attack_method Meta_Self-${PTB} \
          --shot 5 --split 1 \
          --device $DEVICE \
          > $LOG 2>&1"
        run_cmd "GCL-LP $BB_LABEL ptb=$PTB" "$CMD"
    done
done

# ================ Section 3: RobustPrompt-T Single-Filter Sweep ================
echo "=========================================================="
echo " SECTION 3: RobustPrompt-T Single-Filter Sweep (840 runs)"
echo "=========================================================="

# --- 3a: sim_pt only (degree=-1, ood=-1.0) ---
echo "--- 3a: sim_pt sweep ---"
SIM_VALS=("0.2" "0.3" "0.4" "0.5" "0.6")

for BB_LABEL in "stable" "peak"; do
    if [ "$BB_LABEL" = "stable" ]; then
        BB_PATH="$STABLE_BB"
    else
        BB_PATH="$PEAK_BB"
    fi
    for SIM in "${SIM_VALS[@]}"; do
        for PTB in "${PTB_RATES[@]}"; do
            LOG="logs/single_filter_sim/sim${SIM}_${BB_LABEL}_${PTB}_${TIMESTAMP}.log"
            CMD="CUDA_VISIBLE_DEVICES=$DEVICE python MyTask.py \
              --pre_train_model_path '$BB_PATH' \
              --prompt_type RobustPrompt-T \
              --pt_sim_threshold $SIM \
              --pt_degree_threshold -1 \
              --pt_out_detect_threshold -1.0 \
              --attack_downstream --specified --attack_method Meta_Self-${PTB} \
              --seed $SEEDS \
              $BASE_PARAMS \
              > $LOG 2>&1"
            run_cmd "sim=$SIM $BB_LABEL ptb=$PTB" "$CMD"
        done
    done
done

# --- 3b: degree_pt only (sim=-1.0, ood=-1.0) ---
echo "--- 3b: degree_pt sweep ---"
DEG_VALS=("1" "2" "3" "5")

for BB_LABEL in "stable" "peak"; do
    if [ "$BB_LABEL" = "stable" ]; then
        BB_PATH="$STABLE_BB"
    else
        BB_PATH="$PEAK_BB"
    fi
    for DEG in "${DEG_VALS[@]}"; do
        for PTB in "${PTB_RATES[@]}"; do
            LOG="logs/single_filter_degree/deg${DEG}_${BB_LABEL}_${PTB}_${TIMESTAMP}.log"
            CMD="CUDA_VISIBLE_DEVICES=$DEVICE python MyTask.py \
              --pre_train_model_path '$BB_PATH' \
              --prompt_type RobustPrompt-T \
              --pt_sim_threshold -1.0 \
              --pt_degree_threshold $DEG \
              --pt_out_detect_threshold -1.0 \
              --attack_downstream --specified --attack_method Meta_Self-${PTB} \
              --seed $SEEDS \
              $BASE_PARAMS \
              > $LOG 2>&1"
            run_cmd "deg=$DEG $BB_LABEL ptb=$PTB" "$CMD"
        done
    done
done

# --- 3c: out_detect_pt only (sim=-1.0, degree=-1) ---
echo "--- 3c: out_detect_pt sweep ---"
OOD_VALS=("0.3" "0.4" "0.5" "0.6" "0.7")

for BB_LABEL in "stable" "peak"; do
    if [ "$BB_LABEL" = "stable" ]; then
        BB_PATH="$STABLE_BB"
    else
        BB_PATH="$PEAK_BB"
    fi
    for OOD in "${OOD_VALS[@]}"; do
        for PTB in "${PTB_RATES[@]}"; do
            LOG="logs/single_filter_ood/ood${OOD}_${BB_LABEL}_${PTB}_${TIMESTAMP}.log"
            CMD="CUDA_VISIBLE_DEVICES=$DEVICE python MyTask.py \
              --pre_train_model_path '$BB_PATH' \
              --prompt_type RobustPrompt-T \
              --pt_sim_threshold -1.0 \
              --pt_degree_threshold -1 \
              --pt_out_detect_threshold $OOD \
              --attack_downstream --specified --attack_method Meta_Self-${PTB} \
              --seed $SEEDS \
              $BASE_PARAMS \
              > $LOG 2>&1"
            run_cmd "ood=$OOD $BB_LABEL ptb=$PTB" "$CMD"
        done
    done
done

echo "=========================================================="
echo " ALL DONE — $(date)"
echo " Logs: logs/baselines/ logs/single_filter_sim/ logs/single_filter_degree/ logs/single_filter_ood/"
echo "=========================================================="
