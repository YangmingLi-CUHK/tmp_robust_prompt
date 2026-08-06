#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/citeseer_svd100_to_cora_svd100_transductive_2bb_3methods_corrected_budget_180}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

ARGS=(
    --python-bin "$PYTHON_BIN"
    --device 0
    --output-dir "$OUTPUT_DIR"
)
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    ARGS+=(--preflight-only)
fi

"$PYTHON_BIN" run_citeseer_svd100_to_cora_svd100_transductive_2bb_3methods_corrected_budget_180.py \
    "${ARGS[@]}" \
    "$@"
