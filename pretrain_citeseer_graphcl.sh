#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-0}"
AUG_METHODS=("dropN" "permE" "maskN")
AUG_RATIOS=(0.1 0.2 0.3)
SEEDS=(1 2 3 4 5)
LEARNING_RATE=0.001

TOTAL=$((${#AUG_METHODS[@]} * ${#AUG_METHODS[@]} * ${#AUG_RATIOS[@]} * ${#SEEDS[@]}))
COUNT=0

echo "Citeseer GraphCL grid: $TOTAL runs"
echo "Data: DeepRobust/Nettack Citeseer LCC (2110 nodes, 3668 paper edges, 9446 runtime edges)"
echo "Setting: METIS parts=200, GCN, hidden=256, layers=2, epochs=200, lr=$LEARNING_RATE"

for seed in "${SEEDS[@]}"; do
    for aug_ratio in "${AUG_RATIOS[@]}"; do
        for aug1 in "${AUG_METHODS[@]}"; do
            for aug2 in "${AUG_METHODS[@]}"; do
                COUNT=$((COUNT + 1))
                echo "[$COUNT/$TOTAL] aug1=$aug1 aug2=$aug2 ratio=$aug_ratio seed=$seed"

                "$PYTHON_BIN" MyPretrain.py \
                    --task GraphCL \
                    --dataset_name Citeseer \
                    --preprocess_method none \
                    --gnn_type GCN \
                    --hid_dim 256 \
                    --num_layer 2 \
                    --epochs 200 \
                    --seed "$seed" \
                    --device "$DEVICE" \
                    --aug1 "$aug1" \
                    --aug2 "$aug2" \
                    --lr "$LEARNING_RATE" \
                    --aug_ratio "$aug_ratio"
            done
        done
    done
done

echo "Completed $TOTAL Citeseer GraphCL runs."
