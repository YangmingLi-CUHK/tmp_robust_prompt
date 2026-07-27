#!/usr/bin/env bash
# End-to-end MetaAttack poisoning experiment.
#
# For every rate r:
#   Phase 1: train a fresh supervised GCN teacher on A_M-r.
#   Phase 2: freeze that poisoned teacher and train AttrPrompt on A_M-r.
#   Test:    evaluate both teacher and prompt on A_M-r.
#
# Checkpoints are isolated by rate. Phase 2 verifies the teacher's adjacency
# fingerprint before loading it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_attack_fewshot/Cora/shot_5/1/Meta_Self/raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/save_cora/poisoned_teacher_attrprompt}"
PTB_RATES="${PTB_RATES:-0.0 0.05 0.1 0.15 0.2 0.25}"
PROMPT_TYPE="${PROMPT_TYPE:-dynamic}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-400}"
PROMPT_EPOCHS="${PROMPT_EPOCHS:-200}"
SEEDS="${SEEDS:-10}"
CUDA_ID="${CUDA_ID:-0}"
USE_IB="${USE_IB:-1}"
RETRAIN_TEACHER="${RETRAIN_TEACHER:-1}"

cd "$SCRIPT_DIR"
mkdir -p "$OUTPUT_ROOT"

ib_args=()
if [[ "$USE_IB" == "1" ]]; then
    ib_args+=(--IB)
fi

echo "End-to-end poisoned-teacher AttrPrompt experiment"
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Rates: $PTB_RATES"
echo "Prompt: $PROMPT_TYPE"

for ptb in $PTB_RATES; do
    tag="${ptb/./p}"
    rate_root="${OUTPUT_ROOT}/M${tag}"
    teacher_root="${rate_root}/GCN"
    prompt_root="${rate_root}/AttrPrompt_${PROMPT_TYPE}"
    mkdir -p "$teacher_root" "$prompt_root"

    strict_args=()
    if [[ "$ptb" != "0" && "$ptb" != "0.0" ]]; then
        strict_args+=(--strict_no_clean_forward)
    fi

    echo
    echo "================================================================"
    echo "M-${ptb}: teacher train/val/test graph = Meta_Self_Cora_${ptb}.pt"
    echo "M-${ptb}: prompt reference/student/test base = same graph"
    echo "================================================================"

    if [[ "$RETRAIN_TEACHER" == "1" ]]; then
        python train_cora_gcn.py \
            --data_root "$DATA_ROOT" \
            --save_root "$teacher_root" \
            --train_ptb "$ptb" \
            --epochs "$TEACHER_EPOCHS" \
            --hidden 16 \
            --lr 0.01 \
            --dropout 0.5 \
            --weight_decay 5e-4 \
            --patience 100 \
            --seeds "$SEEDS" \
            --cuda_id "$CUDA_ID" \
            "${strict_args[@]}" \
            2>&1 | tee "${teacher_root}/run.log"
    fi

    for ((seed=0; seed<SEEDS; seed++)); do
        checkpoint="${teacher_root}/model_${seed}.pth"
        metadata="${teacher_root}/model_${seed}.meta.json"
        if [[ ! -f "$checkpoint" || ! -f "$metadata" ]]; then
            echo "Missing poisoned teacher or metadata for M-${ptb}, seed ${seed}" >&2
            exit 1
        fi
    done

    python train_cora_attrprompt.py \
        --data_root "$DATA_ROOT" \
        --pretrain_root "$teacher_root" \
        --save_root "$prompt_root" \
        --prompt_type "$PROMPT_TYPE" \
        --epochs "$PROMPT_EPOCHS" \
        --hidden 16 \
        --lr 0.01 \
        --attack_iters 1 \
        --step_size 0.02 \
        --seeds "$SEEDS" \
        --cuda_id "$CUDA_ID" \
        --train_ptb "$ptb" \
        --ptb_rates "$ptb" \
        --require_teacher_metadata \
        "${strict_args[@]}" \
        "${ib_args[@]}" \
        2>&1 | tee "${prompt_root}/run.log"
done

python summarize_poisoned_pipeline.py \
    --output_root "$OUTPUT_ROOT" \
    --ptb_rates $PTB_RATES \
    --prompt_type "$PROMPT_TYPE"

echo
echo "Completed: ${OUTPUT_ROOT}"
echo "Combined table: ${OUTPUT_ROOT}/poisoned_pipeline_summary.csv"
