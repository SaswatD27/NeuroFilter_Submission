#!/usr/bin/env bash

set -euo pipefail

ROOT=/path/to/code
MODEL_KEY=llama_3_3_70b
RUNNER="$ROOT/notebooks/run_cmpl80_threshold0_pyrit_tap20_skeletonkey20.py"
OUTPUT_DIR="$ROOT/logs/cmpl80_threshold0_tap_goat_goblin_v3/$MODEL_KEY"
PYTHON_COMMAND="${PYTHON_BIN:-python}"

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/run.out") 2>&1

echo "Python: $(command -v "$PYTHON_COMMAND")"
echo "Model: $MODEL_KEY"
echo "Output: $OUTPUT_DIR"
echo "Attacks: GOAT and Goblin"
echo "Promptfoo maximum turns: 20"

EXTRA_ARGS=()
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--validate-only)
elif [[ -z "${PROMPTFOO_API_KEY:-}" ]]; then
    echo "PROMPTFOO_API_KEY must be set."
    exit 1
fi

"$PYTHON_COMMAND" -u "$RUNNER" --model "$MODEL_KEY" --attacks goat goblin --promptfoo-max-turns 20 "${EXTRA_ARGS[@]}"
