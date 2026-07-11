#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate LnL

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="./pre_trained_model_raw/Cora.GraphCL.GCN.64_hidden_dim.aug1_dropN.aug2_maskN.lr_0.001.ratio_0.2.seed_4.pth"
LOG_DIR="logs/focusedcleaner_lp_single_filter_64_dropN_maskN_seed4"
mkdir -p "$LOG_DIR"

PTBS="0.0 0.05 0.1 0.15 0.2 0.25"

# Disable RobustPrompt-T prompt-tip selectors so the reported filter_module
# metrics isolate the focusedcleaner_lp filter as much as this pipeline allows.
SIM_OFF="${SIM_OFF:--2.0}"
DEG_OFF="${DEG_OFF:--1}"
OOD_OFF="${OOD_OFF:--2.0}"

FILTER_LP_EPOCHS="${FILTER_LP_EPOCHS:-50}"
FILTER_LP_LR="${FILTER_LP_LR:-0.1}"
FILTER_LP_NEG_RATIO="${FILTER_LP_NEG_RATIO:-1.0}"
FILTER_LP_THRESHOLD_MODE="${FILTER_LP_THRESHOLD_MODE:-gmean}"
FILTER_LP_MAX_TRAIN_PAIRS="${FILTER_LP_MAX_TRAIN_PAIRS:-200000}"
FILTER_LP_PCA_DIM="${FILTER_LP_PCA_DIM:--1}"
FILTER_LP_HIDDEN_DIM="${FILTER_LP_HIDDEN_DIM:-0}"

run_one() {
  local ptb="$1"
  local tag="focusedcleaner_lp"

  local log="${LOG_DIR}/peak_${tag}_sim${SIM_OFF}_deg${DEG_OFF}_ood${OOD_OFF}_ptb${ptb}.log"
  echo "[$(date '+%F %T')] START peak ${tag} ptb=${ptb}"

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" MyTask.py \
    --pre_train_model_path "$MODEL" \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 64 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode focusedcleaner_lp --no_attention \
    --filter_lp_hidden_dim "$FILTER_LP_HIDDEN_DIM" \
    --filter_lp_epochs "$FILTER_LP_EPOCHS" \
    --filter_lp_lr "$FILTER_LP_LR" \
    --filter_lp_neg_ratio "$FILTER_LP_NEG_RATIO" \
    --filter_lp_threshold_mode "$FILTER_LP_THRESHOLD_MODE" \
    --filter_lp_max_train_pairs "$FILTER_LP_MAX_TRAIN_PAIRS" \
    --filter_lp_pca_dim "$FILTER_LP_PCA_DIM" \
    --pt_sim_threshold "$SIM_OFF" \
    --pt_degree_threshold "$DEG_OFF" \
    --pt_out_detect_threshold "$OOD_OFF" \
    --pt_threshold 0.25 --weight_mse 0.0 --weight_kl 0.001 --prompt_lr 0.001 \
    --attack_downstream --specified --attack_method Meta_Self-${ptb} \
    > "$log" 2>&1

  echo "[$(date '+%F %T')] DONE peak ${tag} ptb=${ptb}"
}

for ptb in $PTBS; do
  run_one "$ptb"
done

"$PYTHON_BIN" analyze_combo_filters.py \
  --log-dir "$LOG_DIR" \
  --out-prefix "${LOG_DIR}/focusedcleaner_lp_single_filter_summary"

echo "FocusedCleaner-LP single-filter experiments done."
