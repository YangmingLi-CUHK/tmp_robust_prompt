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
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
RESUME="${RESUME:-0}"
PROMPT_TYPE="${PROMPT_TYPE:-dynamic}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-400}"
PROMPT_EPOCHS="${PROMPT_EPOCHS:-200}"
SEEDS="${SEEDS:-10}"
CUDA_ID="${CUDA_ID:-0}"
USE_IB="${USE_IB:-1}"
RETRAIN_TEACHER="${RETRAIN_TEACHER:-1}"

cd "$SCRIPT_DIR"
mkdir -p "$OUTPUT_ROOT"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_ID may contain only letters, numbers, dot, underscore, and dash." >&2
    exit 1
fi

canonical_rate_string="$(python rate_utils.py $PTB_RATES)"
read -r -a canonical_rates <<< "$canonical_rate_string"

RUN_ROOT="${OUTPUT_ROOT}/runs/${RUN_ID}"
RATE_MANIFEST="${RUN_ROOT}/rates.txt"
if [[ -e "$RUN_ROOT" ]]; then
    if [[ "$RESUME" != "1" ]]; then
        echo "Run directory already exists; refusing to overwrite: $RUN_ROOT" >&2
        echo "Choose another RUN_ID, or set RESUME=1 deliberately." >&2
        exit 1
    fi
    if [[ ! -f "$RATE_MANIFEST" ]]; then
        echo "Cannot resume: missing rate manifest $RATE_MANIFEST" >&2
        exit 1
    fi
    saved_rate_string="$(<"$RATE_MANIFEST")"
    if [[ "$saved_rate_string" != "$canonical_rate_string" ]]; then
        echo "Cannot resume with a different rate set." >&2
        echo "Saved:     $saved_rate_string" >&2
        echo "Requested: $canonical_rate_string" >&2
        exit 1
    fi
    if [[ -f "${RUN_ROOT}/poisoned_pipeline_summary.csv" ]]; then
        echo "Run is already complete; refusing to overwrite its CSV files." >&2
        exit 1
    fi
else
    mkdir -p "$RUN_ROOT"
    printf '%s\n' "$canonical_rate_string" > "$RATE_MANIFEST"
fi
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"

ib_args=()
if [[ "$USE_IB" == "1" ]]; then
    ib_args+=(--IB)
fi

echo "End-to-end poisoned-teacher AttrPrompt experiment"
echo "Data root: $DATA_ROOT"
echo "Run ID: $RUN_ID"
echo "Run root: $RUN_ROOT"
echo "Canonical rates: ${canonical_rates[*]}"
echo "Prompt: $PROMPT_TYPE"

for ptb in "${canonical_rates[@]}"; do
    tag="M${ptb/./p}"
    rate_root="${RUN_ROOT}/${tag}"
    teacher_root="${rate_root}/GCN"
    prompt_root="${rate_root}/AttrPrompt_${PROMPT_TYPE}"
    rate_csv="${rate_root}/result_${tag}.csv"

    if [[ "$RESUME" == "1" && -f "$rate_csv" ]]; then
        echo "M-${ptb} already complete; keeping $rate_csv"
        continue
    fi

    mkdir -p "$teacher_root" "$prompt_root"

    strict_args=()
    if [[ "$ptb" != "0.00" ]]; then
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
            2>&1 | tee "${teacher_root}/run_${ATTEMPT_ID}.log"
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
        2>&1 | tee "${prompt_root}/run_${ATTEMPT_ID}.log"

    python summarize_poisoned_pipeline.py \
        --output_root "$RUN_ROOT" \
        --ptb_rates "$ptb" \
        --prompt_type "$PROMPT_TYPE" \
        --mode per-rate
done

python summarize_poisoned_pipeline.py \
    --output_root "$RUN_ROOT" \
    --ptb_rates "${canonical_rates[@]}" \
    --prompt_type "$PROMPT_TYPE" \
    --mode combined

echo
echo "Completed: ${RUN_ROOT}"
echo "Combined table: ${RUN_ROOT}/poisoned_pipeline_summary.csv"
echo "Per-rate tables: ${RUN_ROOT}/M*/result_M*.csv"
