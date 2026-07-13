#!/usr/bin/env bash
# Edge-anomaly evaluation sweep for all 1/2/3-filter methods.
# The original run_5filter_combos.sh is intentionally left unchanged.
set -euo pipefail

PEAK_BB="${PEAK_BB:-./pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_permE.aug2_maskN.lr_0.001.ratio_0.3.seed_1.pth}"
DEVICE="${DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-1 2 3 4 5}"
# ptb=0 has no positive attack edges, so edge AUC/F1 are undefined and omitted by default.
PTB_RATES="${PTB_RATES:-0.05 0.1 0.15 0.2 0.25}"
LOG_DIR="${LOG_DIR:-logs/5filter_edge_detection}"
OUT_DIR="${OUT_DIR:-results/5filter_edge_detection}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

SIM_THR="${SIM_THR:-0.2}"
DEG_THR="${DEG_THR:-1}"
OOD_THR="${OOD_THR:-0.4}"
NSP_THR="${NSP_THR:-0.3}"
FC_THR="${FC_THR:-0.5}"
SIM_OFF=-1.0
DEG_OFF=-1
OOD_OFF=-1.0
NSP_OFF=-1.0
FC_OFF=-1.0

read -r -a SEED_ARRAY <<< "$SEEDS"

run_one() {
  local label="$1" sim="$2" degree="$3" ood="$4" nsp="$5" fc="$6"
  local ptb log
  for ptb in $PTB_RATES; do
    log="$LOG_DIR/${label}_ptb${ptb}.log"
    echo "[$(date '+%F %T')] START method=$label ptb=$ptb"
    CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" MyTask.py \
      --pre_train_model_path "$PEAK_BB" \
      --task NodeTask --dataset_name Cora --preprocess_method none \
      --gnn_type GCN --prompt_type RobustPrompt-T-NSP --shot_num 5 --run_split 1 \
      --hid_dim 256 --num_layer 2 --epochs 200 \
      --pt_threshold 0.25 --weight_mse 0.1 --weight_kl 0.1 --weight_constraint 0.2 \
      --filter_mode original \
      --pt_sim_threshold "$sim" --pt_degree_threshold "$degree" \
      --pt_out_detect_threshold "$ood" --pt_nsp_threshold "$nsp" \
      --pt_focusedcleaner_threshold "$fc" \
      --seed "${SEED_ARRAY[@]}" \
      --attack_downstream --specified --attack_method "Meta_Self-${ptb}" \
      > "$log" 2>&1
    echo "[$(date '+%F %T')] DONE method=$label ptb=$ptb"
  done
}

echo "=== Single filters ==="
run_one "sim"        "$SIM_THR" "$DEG_OFF" "$OOD_OFF" "$NSP_OFF" "$FC_OFF"
run_one "degree"     "$SIM_OFF" "$DEG_THR" "$OOD_OFF" "$NSP_OFF" "$FC_OFF"
run_one "ood"        "$SIM_OFF" "$DEG_OFF" "$OOD_THR" "$NSP_OFF" "$FC_OFF"
run_one "nsp"        "$SIM_OFF" "$DEG_OFF" "$OOD_OFF" "$NSP_THR" "$FC_OFF"
run_one "fc"         "$SIM_OFF" "$DEG_OFF" "$OOD_OFF" "$NSP_OFF" "$FC_THR"

echo "=== Two-filter combinations ==="
run_one "sim+degree"    "$SIM_THR" "$DEG_THR" "$OOD_OFF" "$NSP_OFF" "$FC_OFF"
run_one "sim+ood"       "$SIM_THR" "$DEG_OFF" "$OOD_THR" "$NSP_OFF" "$FC_OFF"
run_one "sim+nsp"       "$SIM_THR" "$DEG_OFF" "$OOD_OFF" "$NSP_THR" "$FC_OFF"
run_one "sim+fc"        "$SIM_THR" "$DEG_OFF" "$OOD_OFF" "$NSP_OFF" "$FC_THR"
run_one "degree+ood"    "$SIM_OFF" "$DEG_THR" "$OOD_THR" "$NSP_OFF" "$FC_OFF"
run_one "degree+nsp"    "$SIM_OFF" "$DEG_THR" "$OOD_OFF" "$NSP_THR" "$FC_OFF"
run_one "degree+fc"     "$SIM_OFF" "$DEG_THR" "$OOD_OFF" "$NSP_OFF" "$FC_THR"
run_one "ood+nsp"       "$SIM_OFF" "$DEG_OFF" "$OOD_THR" "$NSP_THR" "$FC_OFF"
run_one "ood+fc"        "$SIM_OFF" "$DEG_OFF" "$OOD_THR" "$NSP_OFF" "$FC_THR"
run_one "nsp+fc"        "$SIM_OFF" "$DEG_OFF" "$OOD_OFF" "$NSP_THR" "$FC_THR"

echo "=== Three-filter combinations ==="
run_one "sim+degree+ood" "$SIM_THR" "$DEG_THR" "$OOD_THR" "$NSP_OFF" "$FC_OFF"
run_one "sim+degree+nsp" "$SIM_THR" "$DEG_THR" "$OOD_OFF" "$NSP_THR" "$FC_OFF"
run_one "sim+degree+fc"  "$SIM_THR" "$DEG_THR" "$OOD_OFF" "$NSP_OFF" "$FC_THR"
run_one "sim+ood+nsp"    "$SIM_THR" "$DEG_OFF" "$OOD_THR" "$NSP_THR" "$FC_OFF"
run_one "sim+ood+fc"     "$SIM_THR" "$DEG_OFF" "$OOD_THR" "$NSP_OFF" "$FC_THR"
run_one "sim+nsp+fc"     "$SIM_THR" "$DEG_OFF" "$OOD_OFF" "$NSP_THR" "$FC_THR"
run_one "degree+ood+nsp" "$SIM_OFF" "$DEG_THR" "$OOD_THR" "$NSP_THR" "$FC_OFF"
run_one "degree+ood+fc"  "$SIM_OFF" "$DEG_THR" "$OOD_THR" "$NSP_OFF" "$FC_THR"
run_one "degree+nsp+fc"  "$SIM_OFF" "$DEG_THR" "$OOD_OFF" "$NSP_THR" "$FC_THR"
run_one "ood+nsp+fc"     "$SIM_OFF" "$DEG_OFF" "$OOD_THR" "$NSP_THR" "$FC_THR"

"$PYTHON_BIN" analyze_5filter_edge_detection.py --log-dir "$LOG_DIR" --out-dir "$OUT_DIR"
"$PYTHON_BIN" plot_5filter_edge_detection.py \
  --summary "$OUT_DIR/summary.csv" \
  --out-dir "$OUT_DIR/figures"

echo "=== ALL DONE: $OUT_DIR ==="
