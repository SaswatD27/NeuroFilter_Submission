#!/usr/bin/env bash

set -euo pipefail

ROOT=/path/to/code
RUN_ROOT="$ROOT/logs/cmpl_alltrain80_delta_vs_displacement_sharedtest_v1"
RUNNER="$ROOT/notebooks/cmpl_alltrain40_delta_vs_displacement_sharedtest.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-qwen_2_5_32b}"

mkdir -p "$RUN_ROOT"
export MPLCONFIGDIR="$RUN_ROOT/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"
exec > >(tee -a "$RUN_ROOT/00_driver.out") 2>&1

echo "Python: $(command -v "$PYTHON_BIN")"
echo "Artifacts: $RUN_ROOT"
echo "Models: $MODELS"
echo "Training subjects: 0-19 and 40-59"
echo "Test subjects: 20-39 and 60-79"

EXTRA_ARGS=()
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--validate-only)
fi
if [[ "${CRESCENDO_ONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--crescendo-only)
fi

for MODEL_KEY in $MODELS; do
    MODEL_DIR="$RUN_ROOT/$MODEL_KEY"
    mkdir -p "$MODEL_DIR"
    echo "Starting $MODEL_KEY"
    "$PYTHON_BIN" -u "$RUNNER" --model "$MODEL_KEY" --trajectory-count 80 --artifact-dir "$MODEL_DIR" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$MODEL_DIR/run.out"
done

echo "All requested models completed."
