#!/usr/bin/env bash

set -euo pipefail

CMPL_ROOT=/path/to/code
CMPL_RUN_DIR="$CMPL_ROOT/logs/cmpl_alltrain80_delta_vs_displacement_live_cmpl_v1/qwen_2_5_32b"
CMPL_RUNNER="$CMPL_ROOT/notebooks/cmpl80_delta_vs_displacement_live_cmpl.py"
CMPL_PLOTTER="$CMPL_ROOT/notebooks/plot_cmpl80_delta_vs_displacement_live_cmpl.py"
CMPL_PYTHON="${PYTHON_BIN:-python}"

mkdir -p "$CMPL_RUN_DIR"
exec > >(tee -a "$CMPL_RUN_DIR/00_driver.out") 2>&1

echo "Python: $(command -v "$CMPL_PYTHON")"
echo "Artifacts: $CMPL_RUN_DIR"
echo "Training subjects: 0-19 and 40-59"
echo "Live test subjects: 20-39 and 60-79"
echo "Live conversations: 40 malicious + 40 benign for each probe"
echo "Threshold: 0 for both probes"
echo "Delta conversations: reuse existing complete CMPL threshold-0 test"
echo "New API generation: displacement conversations only"

CMPL_EXTRA_ARGS=()
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
    CMPL_EXTRA_ARGS+=(--validate-only)
else
    CMPL_PRIMARY_ENDPOINT="${CMPL_OPENAI_ENDPOINT:-${ATTACKER_OPENAI_ENDPOINT:-}}"
    CMPL_PRIMARY_API_KEY="${CMPL_OPENAI_API_KEY:-${ATTACKER_OPENAI_API_KEY:-}}"
    if [[ -z "$CMPL_PRIMARY_ENDPOINT" || -z "$CMPL_PRIMARY_API_KEY" ]]; then
        echo "Set CMPL_OPENAI_ENDPOINT and CMPL_OPENAI_API_KEY, or ATTACKER_OPENAI_ENDPOINT and ATTACKER_OPENAI_API_KEY."
        exit 1
    fi
fi

"$CMPL_PYTHON" -u "$CMPL_RUNNER" --output-dir "$CMPL_RUN_DIR" --methods displacement "${CMPL_EXTRA_ARGS[@]}" 2>&1 | tee -a "$CMPL_RUN_DIR/run.out"

if [[ "${VALIDATE_ONLY:-0}" != "1" ]]; then
    "$CMPL_PYTHON" -u "$CMPL_PLOTTER" --run-dir "$CMPL_RUN_DIR" 2>&1 | tee -a "$CMPL_RUN_DIR/plot.out"
fi

echo "Qwen live CMPL delta-versus-displacement evaluation completed."
