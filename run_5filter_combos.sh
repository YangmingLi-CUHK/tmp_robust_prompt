#!/usr/bin/env bash
# ============================================================================
# 5-Filter Combo Sweep: sim / degree / out_detect / nsp / focusedcleaner
# Uses RobustPrompt-T-NSP which supports all 5 tips natively.
# Each tip is activated via its threshold; set to -1 (or -1.0) to disable.
# ============================================================================
set -uo pipefail

# ---- Config ----
PEAK_BB="./pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_permE.aug2_maskN.lr_0.001.ratio_0.3.seed_1.pth"
DEVICE=0
SEEDS="1 2 3 4 5"
PTB_RATES="0.0 0.05 0.1 0.15 0.2 0.25"
LOG_DIR="logs/5filter_combos"
mkdir -p "$LOG_DIR"

# Active thresholds (when ON):
SIM_THR=0.2
DEG_THR=1
OOD_THR=0.4
NSP_THR=0.3
FC_THR=0.5
# Disabled values:
SIM_OFF=-1.0
DEG_OFF=-1
OOD_OFF=-1.0
NSP_OFF=-1.0
FC_OFF=-1.0

# Shared base params
BASE_PARAMS="--task NodeTask --dataset_name Cora --preprocess_method none \
  --gnn_type GCN --prompt_type RobustPrompt-T-NSP --shot_num 5 --run_split 1 \
  --hid_dim 256 --num_layer 2 --epochs 200 \
  --pt_threshold 0.25 --weight_mse 0.1 --weight_kl 0.1 --weight_constraint 0.2 \
  --filter_mode original"

declare -A TIP_LABELS=(
  ["sim"]="s"
  ["degree"]="d"
  ["ood"]="o"
  ["nsp"]="n"
  ["focusedcleaner"]="f"
)

run_one() {
  local label="$1" sim="$2" deg="$3" ood="$4" nsp="$5" fc="$6"
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
      --pt_out_detect_threshold $ood --pt_nsp_threshold $nsp \
      --pt_focusedcleaner_threshold $fc \
      --seed $SEEDS \
      $BASE_PARAMS $extra \
      > "$log" 2>&1 \
      || echo "[$(date +%H:%M:%S)] $label ptb=$ptb FAILED (exit=$?)" | tee -a "$LOG_DIR/_errors.log"
  done
}

# ---- Single filters (5) ----
echo "=== Single Filter ==="
run_one "sim"         $SIM_THR $DEG_OFF $OOD_OFF $NSP_OFF $FC_OFF
run_one "degree"      $SIM_OFF $DEG_THR $OOD_OFF $NSP_OFF $FC_OFF
run_one "ood"         $SIM_OFF $DEG_OFF $OOD_THR $NSP_OFF $FC_OFF
run_one "nsp"         $SIM_OFF $DEG_OFF $OOD_OFF $NSP_THR $FC_OFF
run_one "fc"          $SIM_OFF $DEG_OFF $OOD_OFF $NSP_OFF $FC_THR

# ---- Two-filter combos (10) ----
echo "=== Two-Filter Combos ==="
run_one "sim+deg"     $SIM_THR $DEG_THR $OOD_OFF $NSP_OFF $FC_OFF
run_one "sim+ood"     $SIM_THR $DEG_OFF $OOD_THR $NSP_OFF $FC_OFF
run_one "sim+nsp"     $SIM_THR $DEG_OFF $OOD_OFF $NSP_THR $FC_OFF
run_one "sim+fc"      $SIM_THR $DEG_OFF $OOD_OFF $NSP_OFF $FC_THR
run_one "deg+ood"     $SIM_OFF $DEG_THR $OOD_THR $NSP_OFF $FC_OFF
run_one "deg+nsp"     $SIM_OFF $DEG_THR $OOD_OFF $NSP_THR $FC_OFF
run_one "deg+fc"      $SIM_OFF $DEG_THR $OOD_OFF $NSP_OFF $FC_THR
run_one "ood+nsp"     $SIM_OFF $DEG_OFF $OOD_THR $NSP_THR $FC_OFF
run_one "ood+fc"      $SIM_OFF $DEG_OFF $OOD_THR $NSP_OFF $FC_THR
run_one "nsp+fc"      $SIM_OFF $DEG_OFF $OOD_OFF $NSP_THR $FC_THR

# ---- Three-filter combos (10) ----
echo "=== Three-Filter Combos ==="
run_one "sim+deg+ood"   $SIM_THR $DEG_THR $OOD_THR $NSP_OFF $FC_OFF
run_one "sim+deg+nsp"   $SIM_THR $DEG_THR $OOD_OFF $NSP_THR $FC_OFF
run_one "sim+deg+fc"    $SIM_THR $DEG_THR $OOD_OFF $NSP_OFF $FC_THR
run_one "sim+ood+nsp"   $SIM_THR $DEG_OFF $OOD_THR $NSP_THR $FC_OFF
run_one "sim+ood+fc"    $SIM_THR $DEG_OFF $OOD_THR $NSP_OFF $FC_THR
run_one "sim+nsp+fc"    $SIM_THR $DEG_OFF $OOD_OFF $NSP_THR $FC_THR
run_one "deg+ood+nsp"   $SIM_OFF $DEG_THR $OOD_THR $NSP_THR $FC_OFF
run_one "deg+ood+fc"    $SIM_OFF $DEG_THR $OOD_THR $NSP_OFF $FC_THR
run_one "deg+nsp+fc"    $SIM_OFF $DEG_THR $OOD_OFF $NSP_THR $FC_THR
run_one "ood+nsp+fc"    $SIM_OFF $DEG_OFF $OOD_THR $NSP_THR $FC_THR

# ---- Four-filter combos (5) ----
echo "=== Four-Filter Combos ==="
run_one "sim+deg+ood+nsp"    $SIM_THR $DEG_THR $OOD_THR $NSP_THR $FC_OFF
run_one "sim+deg+ood+fc"     $SIM_THR $DEG_THR $OOD_THR $NSP_OFF $FC_THR
run_one "sim+deg+nsp+fc"     $SIM_THR $DEG_THR $OOD_OFF $NSP_THR $FC_THR
run_one "sim+ood+nsp+fc"     $SIM_THR $DEG_OFF $OOD_THR $NSP_THR $FC_THR
run_one "deg+ood+nsp+fc"     $SIM_OFF $DEG_THR $OOD_THR $NSP_THR $FC_THR

# ---- Five-filter combo (1) ----
echo "=== Five-Filter Combo ==="
run_one "all5" $SIM_THR $DEG_THR $OOD_THR $NSP_THR $FC_THR

echo "=== ALL DONE ==="
