#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/logs/citeseer_svd100_to_cora_svd100_graphcl_135}"
CORA_SVD_CACHE="${CORA_SVD_CACHE:-$PROJECT_ROOT/data/preprocessed/cora_clean_full_l1_svd_100.pt}"
CITESEER_SVD_CACHE="$PROJECT_ROOT/data/deeprobust/citeseer_nettack_lcc_l1_svd_100.pt"

AUG_METHODS=("dropN" "permE" "maskN")
AUG_RATIOS=("0.1" "0.2" "0.3")
SEEDS=(1 2 3 4 5)
LEARNING_RATE="0.001"
SVD_OUT_DIM=100
HIDDEN_DIM=256
NUM_LAYERS=2
EPOCHS=200
SHOT=5
SPLIT=1

TOTAL=$((${#AUG_METHODS[@]} * ${#AUG_METHODS[@]} * ${#AUG_RATIOS[@]} * ${#SEEDS[@]}))
RESULTS_CSV="$OUTPUT_DIR/per_seed_results_incremental.csv"
SUMMARY_CSV="$OUTPUT_DIR/group_summary_incremental.csv"
MANIFEST="$OUTPUT_DIR/manifest.tsv"
RUN_LOG_DIR="$OUTPUT_DIR/run_logs"
SOURCE_CACHE_RECEIPT="$OUTPUT_DIR/citeseer_svd100_cache_receipt.json"

mkdir -p "$RUN_LOG_DIR"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
EXPECTED_MANIFEST="$OUTPUT_DIR/.manifest.expected.$$"
cleanup_manifest_temp() {
    rm -f "$EXPECTED_MANIFEST"
}
trap cleanup_manifest_temp EXIT

{
    printf 'field\tvalue\n'
    printf 'experiment\tciteseer_svd100_to_cora_svd100_graphcl_linear_probe\n'
    printf 'git_commit\t%s\n' "$GIT_COMMIT"
    printf 'source_scope\tDeepRobust/Nettack Citeseer LCC: 2110 nodes, 3668 undirected edges, 3703 BoW features\n'
    printf 'source_pipeline\traw_BoW->L1_row_normalization->independent_SVD100->METIS_200->GraphCL\n'
    printf 'source_cache\traw-feature SHA256 + SVD-feature SHA256 + Torch/PyG versions recorded before training\n'
    printf 'target_scope\tfull clean Cora: 2708 nodes, 1433 BoW features, 5-shot/split-1\n'
    printf 'target_pipeline\traw_BoW->L1_row_normalization->independent_SVD100->frozen_GCN->all-node_zscore->multinomial_logistic\n'
    printf 'target_cache\traw-feature SHA256 + SVD-feature SHA256 + Torch/PyG versions validated before reuse\n'
    printf 'feature_alignment\tindependent_svd_shape_only; no shared SVD basis or semantic alignment\n'
    printf 'grid\taug1(3 ordered choices)*aug2(3 ordered choices)*ratio(3)*seed(5)=135 runs; 27 groups*5 seeds\n'
    printf 'augmentations\tdropN,permE,maskN; same-view pairs allowed\n'
    printf 'augmentation_ratios\t0.1,0.2,0.3; shared by both GraphCL views\n'
    printf 'pretrain_seeds\t1,2,3,4,5\n'
    printf 'encoder\tGCN 100->256->256; 2 layers; mean graph pooling\n'
    printf 'graphcl\tepochs=200; batch_size=10; temperature=0.1; projection=256->256->256; final-epoch GNN saved\n'
    printf 'graphcl_views\tone augmented-view pair is generated before training and reused for all 200 epochs\n'
    printf 'optimizer\tAdam; lr=0.001; weight_decay=0.0001; early stopping disabled\n'
    printf 'probe\tfrozen encoder; full-graph embeddings; all-node z-score; sklearn multinomial LogisticRegression(lbfgs,max_iter=10000)\n'
    printf 'selection\trank only complete 5-seed groups by validation mean; test is not used for selection\n'
    printf 'accuracy_summary\tall 5 seeds; simple mean and population std; no seed removed\n'
} > "$EXPECTED_MANIFEST"

if [[ -f "$MANIFEST" ]]; then
    if ! cmp -s "$EXPECTED_MANIFEST" "$MANIFEST"; then
        echo "ERROR: existing manifest does not match this script: $MANIFEST" >&2
        echo "Use a new OUTPUT_DIR for a different experiment or code revision." >&2
        exit 2
    fi
else
    mv "$EXPECTED_MANIFEST" "$MANIFEST"
fi

echo "Citeseer-SVD100 GraphCL -> Cora-SVD100 frozen linear probe"
echo "Grid: $TOTAL runs = 27 hyperparameter groups x 5 pretraining seeds"
echo "Results: $RESULTS_CSV"
echo "Group summary: $SUMMARY_CSV"
echo "Boundary: Citeseer and Cora fit SVD100 independently; this aligns shape, not feature semantics."

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
    eval_citeseer_svd100_to_cora_svd100.py \
    --prepare_source_cache_only \
    --source_cache_path "$CITESEER_SVD_CACHE" \
    --source_cache_receipt "$SOURCE_CACHE_RECEIPT" \
    --results_csv "$RESULTS_CSV"

COUNT=0
for seed in "${SEEDS[@]}"; do
    for aug_ratio in "${AUG_RATIOS[@]}"; do
        for aug1 in "${AUG_METHODS[@]}"; do
            for aug2 in "${AUG_METHODS[@]}"; do
                COUNT=$((COUNT + 1))
                RUN_KEY="a1=${aug1}|a2=${aug2}|r=${aug_ratio}|s=${seed}"
                CHECKPOINT_NAME="Citeseer.GraphCL.GCN.${HIDDEN_DIM}_hidden_dim.preprocess_svd_${SVD_OUT_DIM}.aug1_${aug1}.aug2_${aug2}.lr_${LEARNING_RATE}.ratio_${aug_ratio}.seed_${seed}.pth"
                CHECKPOINT_PATH="$PROJECT_ROOT/pre_trained_model_raw/$CHECKPOINT_NAME"

                if [[ -f "$RESULTS_CSV" ]] && awk -F',' -v key="$RUN_KEY" \
                    'NR > 1 && $1 == key && $2 == "ok" { found=1 } END { exit !found }' \
                    "$RESULTS_CSV"; then
                    echo "[$COUNT/$TOTAL] verifying recorded run: $RUN_KEY"
                    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
                        eval_citeseer_svd100_to_cora_svd100.py \
                        --checkpoint "$CHECKPOINT_PATH" \
                        --results_csv "$RESULTS_CSV" \
                        --summary_csv "$SUMMARY_CSV" \
                        --source_cache_receipt "$SOURCE_CACHE_RECEIPT" \
                        --skip_recorded
                    continue
                fi

                SAFE_KEY="a1_${aug1}__a2_${aug2}__r_${aug_ratio}__s_${seed}"
                ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
                LOG_PATH="$RUN_LOG_DIR/${SAFE_KEY}.attempt_${ATTEMPT_ID}.log"

                echo "[$COUNT/$TOTAL] $RUN_KEY"
                {
                    echo "run_key=$RUN_KEY"
                    echo "checkpoint=$CHECKPOINT_PATH"
                    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

                    if [[ -e "$CHECKPOINT_PATH" ]]; then
                        QUARANTINE_PATH="${CHECKPOINT_PATH}.unrecorded.${ATTEMPT_ID}"
                        echo "Unrecorded checkpoint provenance is unknown; preserving it at $QUARANTINE_PATH"
                        mv -- "$CHECKPOINT_PATH" "$QUARANTINE_PATH"
                    fi

                    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" MyPretrain.py \
                        --task GraphCL \
                        --dataset_name Citeseer \
                        --preprocess_method svd \
                        --svd_out_dim "$SVD_OUT_DIM" \
                        --gnn_type GCN \
                        --hid_dim "$HIDDEN_DIM" \
                        --num_layer "$NUM_LAYERS" \
                        --epochs "$EPOCHS" \
                        --seed "$seed" \
                        --device 0 \
                        --aug1 "$aug1" \
                        --aug2 "$aug2" \
                        --lr "$LEARNING_RATE" \
                        --aug_ratio "$aug_ratio"

                    if [[ ! -s "$CHECKPOINT_PATH" ]]; then
                        echo "ERROR: expected checkpoint was not created: $CHECKPOINT_PATH" >&2
                        exit 1
                    fi

                    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
                        eval_citeseer_svd100_to_cora_svd100.py \
                        --checkpoint "$CHECKPOINT_PATH" \
                        --validate_checkpoint_only

                    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
                        eval_citeseer_svd100_to_cora_svd100.py \
                        --checkpoint "$CHECKPOINT_PATH" \
                        --device 0 \
                        --shot "$SHOT" \
                        --split "$SPLIT" \
                        --cache_path "$CORA_SVD_CACHE" \
                        --results_csv "$RESULTS_CSV" \
                        --summary_csv "$SUMMARY_CSV" \
                        --source_cache_receipt "$SOURCE_CACHE_RECEIPT" \
                        --log_path "$LOG_PATH" \
                        --skip_recorded

                    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                } 2>&1 | tee -a "$LOG_PATH"
            done
        done
    done
done

if [[ ! -f "$RESULTS_CSV" || ! -f "$SUMMARY_CSV" ]]; then
    echo "ERROR: final result tables are missing." >&2
    exit 1
fi
SUCCESSFUL_RUNS="$(awk -F',' 'NR > 1 && $2 == "ok" { count++ } END { print count + 0 }' "$RESULTS_CSV")"
COMPLETE_GROUPS="$(awk -F',' 'NR > 1 && $2 == "complete_5_of_5" { count++ } END { print count + 0 }' "$SUMMARY_CSV")"
if [[ "$SUCCESSFUL_RUNS" -ne 135 || "$COMPLETE_GROUPS" -ne 27 ]]; then
    echo "ERROR: final completeness check failed: runs=$SUCCESSFUL_RUNS/135, groups=$COMPLETE_GROUPS/27" >&2
    exit 1
fi

echo "Completed all $TOTAL runs."
echo "Per-seed results: $RESULTS_CSV"
echo "Five-seed group summary (ranked by validation mean): $SUMMARY_CSV"
