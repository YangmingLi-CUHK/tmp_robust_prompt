#!/bin/bash
# =============================================================================
# GraphCL 预训练 5-seed 网格搜索
# 超参空间: aug1×aug2×lr×aug_ratio×seed = 3×3×3×3×5 = 405 组实验
# 权重保存到 pre_trained_model_raw/
# 命名: {Cora.GraphCL.GCN.256_hidden_dim.aug1_X.aug2_Y.lr_Z.ratio_R.seed_N.pth}
# =============================================================================

AUG_METHODS=("dropN" "permE" "maskN")
LEARNING_RATES=(0.01 0.005 0.001)
AUG_RATIOS=(0.1 0.2 0.3)
SEEDS=(1 2 3 4 5)

TOTAL=$(( ${#AUG_METHODS[@]} * ${#AUG_METHODS[@]} * ${#LEARNING_RATES[@]} * ${#AUG_RATIOS[@]} * ${#SEEDS[@]} ))
echo "Total experiments: $TOTAL"

COUNT=0
for seed in "${SEEDS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for aug_ratio in "${AUG_RATIOS[@]}"; do
            for aug1 in "${AUG_METHODS[@]}"; do
                for aug2 in "${AUG_METHODS[@]}"; do
                    COUNT=$((COUNT + 1))
                    echo "=========================================================="
                    echo "[$COUNT/$TOTAL] GraphCL: aug1=$aug1, aug2=$aug2, lr=$lr, aug_ratio=$aug_ratio, seed=$seed"
                    echo "=========================================================="

                    python MyPretrain.py \
                        --task GraphCL \
                        --dataset_name Cora \
                        --preprocess_method none \
                        --gnn_type GCN \
                        --hid_dim 256 \
                        --num_layer 2 \
                        --epochs 200 \
                        --seed "$seed" \
                        --device 0 \
                        --aug1 "$aug1" \
                        --aug2 "$aug2" \
                        --lr "$lr" \
                        --aug_ratio "$aug_ratio"
                done
            done
        done
    done
done

echo "All $TOTAL pretraining tasks finished!"
