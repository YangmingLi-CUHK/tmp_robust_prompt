#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <Citeseer-SVD1433-GraphCL-checkpoint.pth>" >&2
    exit 2
fi

CHECKPOINT="$1"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

CHECKPOINT_NAME="$(basename "$CHECKPOINT")"
case "$CHECKPOINT_NAME" in
    Citeseer.GraphCL.GCN.256_hidden_dim.preprocess_svd_1433.aug1_*.aug2_*.lr_0.001.ratio_*.seed_*.pth)
        ;;
    *)
        echo "Unexpected checkpoint configuration: $CHECKPOINT_NAME" >&2
        echo "Expected Citeseer GraphCL/GCN/256/SVD-1433/lr-0.001 checkpoint." >&2
        exit 2
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
PTB_RATES=("0.00" "0.05" "0.10" "0.15" "0.20" "0.25")
SEEDS=(1 2 3 4 5)
LOG_DIR="logs/citeseer_svd1433_to_cora_rprompt_original"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
    --pre_train_model_path "$CHECKPOINT"
    --pretrain_dataset_name Citeseer
    --task NodeTask
    --dataset_name Cora
    --preprocess_method none
    --gnn_type GCN
    --prompt_type RobustPrompt-T
    --prompt_variant original
    --shot_num 5
    --run_split 1
    --hid_dim 256
    --num_layer 2
    --epochs 200
    --seed "${SEEDS[@]}"
    --filter_mode original
    --prompt_lr 0.01
    --pt_threshold 0.5
    --weight_mse 0.1
    --weight_kl 0.3
    --weight_constraint 0.2
    --temperature 1.0
    --pt_sim_threshold 0.2
    --pt_degree_threshold 1
    --pt_out_detect_threshold 0.4
    --p_plus
    --no_attention
    --cosine_constraint
)

echo "Checkpoint: $CHECKPOINT"
echo "Transfer: Citeseer SVD-1433 -> Cora raw 1433-D BoW (shape-only compatibility)"

for ptb in "${PTB_RATES[@]}"; do
    log_path="$LOG_DIR/$(basename "$CHECKPOINT" .pth)_Meta_Self_${ptb}.log"
    echo "[$(date '+%F %T')] Meta_Self-$ptb"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" MyTask.py \
        "${COMMON_ARGS[@]}" \
        --attack_downstream \
        --specified \
        --attack_method "Meta_Self-$ptb" \
        2>&1 | tee "$log_path"
done

echo "Completed Cora RobustPrompt-T-original at 6 pollution levels."
