#!/usr/bin/env bash
# ============================================================================
# Inductive RobustPrompt-I Combo Sweep — OUR MODIFIED VERSION.
# Uses RobustPrompt-I with 3 defense tips (sim / degree / out_detect).
# Key features vs original:
#   - out_detect_pt fully implemented (original: pass)
#   - Real attention (original: F.normalize overwrite = fake attention)
#   - tau-tune two-pass edge pruning in Tune() (original: single-pass pruning in forward)
#   - Gradient clipping (original: none)
#   - filter_module support (original: none)
#   - Safe degree normalization (+1e-12 epsilon, original: no epsilon)
# Each tip is activated via its threshold; set to -1 (or -1.0) to disable.
# ============================================================================
set -uo pipefail

# ---- Config ----
PEAK_BB="./pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_permE.aug2_maskN.lr_0.001.ratio_0.3.seed_1.pth"
DEVICE=0
SEEDS="1 2 3 4 5"
PTB_RATES="0.0 0.05 0.1 0.15 0.2 0.25"
LOG_DIR="logs/inductive_robustprompt"
mkdir -p "$LOG_DIR"

# Active thresholds (when ON):
SIM_THR=0.2
DEG_THR=1
OOD_THR=0.4
# Disabled values:
SIM_OFF=-1.0
DEG_OFF=-1
OOD_OFF=-1.0

# Shared base params (aligned with run_5filter_combos.sh)
BASE_PARAMS="--task NodeTask --dataset_name Cora --preprocess_method none \
  --gnn_type GCN --prompt_type RobustPrompt-I --shot_num 5 --run_split 1 \
  --hid_dim 256 --num_layer 2 --epochs 200 \
  --pt_threshold 0.25 --weight_mse 0.1 --weight_kl 0.1 --weight_constraint 0.2 \
  --filter_mode original"

run_one() {
  local label="$1" sim="$2" deg="$3" ood="$4"
  for ptb in $PTB_RATES; do
    local log="$LOG_DIR/${label}_ptb${ptb}.log"
    local extra=""
    if [ "$ptb" != "0.0" ]; then
      extra="--attack_downstream --specified --attack_method Meta_Self-${ptb}"
    fi
    echo "[$(date +%H:%M:%S)] $label ptb=$ptb"
    CUDA_VISIBLE_DEVICES=$DEVICE python MyTask.py \
      --pre_train_model_path "$PEAK_BB" \
      --pt_sim_threshold $sim --pt_degree_threshold $deg \
      --pt_out_detect_threshold $ood \
      --seed $SEEDS \
      $BASE_PARAMS $extra \
      > "$log" 2>&1 \
      || echo "[$(date +%H:%M:%S)] $label ptb=$ptb FAILED (exit=$?)" | tee -a "$LOG_DIR/_errors.log"
  done
}

# ---- Single filters (3) ----
echo "=== Single Filter ==="
run_one "sim"         $SIM_THR $DEG_OFF $OOD_OFF
run_one "degree"      $SIM_OFF $DEG_THR $OOD_OFF
run_one "ood"         $SIM_OFF $DEG_OFF $OOD_THR

# ---- Two-filter combos (3) ----
echo "=== Two-Filter Combos ==="
run_one "sim+degree"  $SIM_THR $DEG_THR $OOD_OFF
run_one "sim+ood"     $SIM_THR $DEG_OFF $OOD_THR
run_one "degree+ood"  $SIM_OFF $DEG_THR $OOD_THR

# ---- Three-filter combo (1) ----
echo "=== Three-Filter Combo ==="
run_one "sim+degree+ood" $SIM_THR $DEG_THR $OOD_THR

echo "=== ALL DONE ==="
