#!/usr/bin/env bash

set -euo pipefail

ROOT=/path/to/code
RUN_ROOT="$ROOT/logs/cmpl80_offline_repeated_train_test_50_50_v1"
RUNNER="$ROOT/notebooks/cmpl80_offline_repeated_train_test_runner.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-qwen_2_5_32b llama_3_3_70b gpt_oss_20b}"
REPEATS="${REPEATS:-10}"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/00_driver.out") 2>&1

echo "Python: $(command -v "$PYTHON_BIN")"
echo "Artifacts: $RUN_ROOT"
echo "Models: $MODELS"
echo "Repeats: $REPEATS"
echo "This run is offline and makes no API calls."

EXTRA_ARGS=()
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--validate-only)
fi

for MODEL_KEY in $MODELS; do
    MODEL_DIR="$RUN_ROOT/$MODEL_KEY"
    mkdir -p "$MODEL_DIR"
    echo "Starting $MODEL_KEY"
    "$PYTHON_BIN" -u "$RUNNER" --model "$MODEL_KEY" --output-dir "$MODEL_DIR" --repeats "$REPEATS" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$MODEL_DIR/run.out"
done

echo "All requested offline repeated evaluations completed."
