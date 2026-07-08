#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="logs/combo_filters_midmetrics_deg_sweep"
mkdir -p "$LOG_DIR"

PTBS="0.05 0.1 0.15 0.2 0.25"
DEGS="1 2 3 5"

MODEL_STABLE="./pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_permE.aug2_dropN.lr_0.001.ratio_0.2.seed_1.pth"
MODEL_PEAK="./pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_permE.aug2_maskN.lr_0.001.ratio_0.3.seed_1.pth"

run_one() {
  local bb="$1"
  local model="$2"
  local tag="$3"
  local sim="$4"
  local deg="$5"
  local ood="$6"
  local ptb="$7"

  local log="${LOG_DIR}/${bb}_${tag}_sim${sim}_deg${deg}_ood${ood}_ptb${ptb}.log"
  echo "[$(date '+%F %T')] START ${bb} ${tag} sim=${sim} deg=${deg} ood=${ood} ptb=${ptb}"

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" MyTask.py \
    --pre_train_model_path "$model" \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --attack_downstream --specified --attack_method Meta_Self-${ptb} \
    --pt_sim_threshold "$sim" \
    --pt_degree_threshold "$deg" \
    --pt_out_detect_threshold "$ood" \
    > "$log" 2>&1

  echo "[$(date '+%F %T')] DONE ${bb} ${tag} deg=${deg} ptb=${ptb}"
}

for ptb in $PTBS; do
  for deg in $DEGS; do
    for bb in stable peak; do
      if [ "$bb" = "stable" ]; then
        model="$MODEL_STABLE"
      else
        model="$MODEL_PEAK"
      fi

      run_one "$bb" "$model" deg_ood_degscan -1.0 "$deg" 0.5 "$ptb"
      run_one "$bb" "$model" sim_deg_degscan 0.3 "$deg" -1.0 "$ptb"
      run_one "$bb" "$model" all3_degscan 0.3 "$deg" 0.5 "$ptb"
    done
  done
done

python analyze_combo_filters.py \
  --log-dir "$LOG_DIR" \
  --out-prefix "${LOG_DIR}/combo_filters_midmetrics_deg_sweep_summary"
