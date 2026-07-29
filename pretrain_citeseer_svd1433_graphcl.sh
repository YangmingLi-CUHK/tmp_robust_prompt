#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
AUG_METHODS=("dropN" "permE" "maskN")
AUG_RATIOS=(0.1 0.2 0.3)
SEEDS=(1 2 3 4 5)
LEARNING_RATE=0.001
SVD_OUT_DIM=1433

TOTAL=$((${#AUG_METHODS[@]} * ${#AUG_METHODS[@]} * ${#AUG_RATIOS[@]} * ${#SEEDS[@]}))
COUNT=0

echo "Citeseer SVD-${SVD_OUT_DIM} GraphCL grid: $TOTAL runs"
echo "Data: DeepRobust/Nettack Citeseer LCC (2110 nodes, 3668 paper edges)"
echo "Setting: METIS parts=200, GCN, hidden=256, layers=2, epochs=200, lr=$LEARNING_RATE"
echo "Boundary: SVD makes the input shape compatible with Cora; it does not align feature semantics."

for seed in "${SEEDS[@]}"; do
    for aug_ratio in "${AUG_RATIOS[@]}"; do
        for aug1 in "${AUG_METHODS[@]}"; do
            for aug2 in "${AUG_METHODS[@]}"; do
                COUNT=$((COUNT + 1))
                echo "[$COUNT/$TOTAL] aug1=$aug1 aug2=$aug2 ratio=$aug_ratio seed=$seed"

                CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" MyPretrain.py \
                    --task GraphCL \
                    --dataset_name Citeseer \
                    --preprocess_method svd \
                    --svd_out_dim "$SVD_OUT_DIM" \
                    --gnn_type GCN \
                    --hid_dim 256 \
                    --num_layer 2 \
                    --epochs 200 \
                    --seed "$seed" \
                    --device 0 \
                    --aug1 "$aug1" \
                    --aug2 "$aug2" \
                    --lr "$LEARNING_RATE" \
                    --aug_ratio "$aug_ratio"
            done
        done
    done
done

echo "Completed $TOTAL Citeseer SVD-${SVD_OUT_DIM} GraphCL runs."
