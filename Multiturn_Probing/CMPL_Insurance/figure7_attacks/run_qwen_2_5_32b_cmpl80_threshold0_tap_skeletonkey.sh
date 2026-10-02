#!/usr/bin/env bash

set -euo pipefail

RUN_ROOT=/path/to/code
MODEL_KEY=qwen_2_5_32b
OUTPUT_DIR="$RUN_ROOT/logs/cmpl80_threshold0_tap_skeletonkey_seeded_v1/$MODEL_KEY"
RUNNER="$RUN_ROOT/notebooks/run_cmpl80_threshold0_pyrit_tap_skeletonkey.py"
PYTHON_COMMAND="${PYTHON_BIN:-python}"
ATTACK_NAMES="${ATTACKS:-tap skeleton_key}"

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/run.out") 2>&1

echo "Python: $(command -v "$PYTHON_COMMAND")"
echo "Model: $MODEL_KEY"
echo "Output: $OUTPUT_DIR"

EXTRA_ARGS=()
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--validate-only)
elif [[ -z "${ATTACKER_OPENAI_ENDPOINT:-}" || -z "${ATTACKER_OPENAI_API_KEY:-}" ]]; then
    echo "ATTACKER_OPENAI_ENDPOINT and ATTACKER_OPENAI_API_KEY must be set."
    exit 1
fi

"$PYTHON_COMMAND" -u "$RUNNER" --model "$MODEL_KEY" --attacks $ATTACK_NAMES "${EXTRA_ARGS[@]}"

