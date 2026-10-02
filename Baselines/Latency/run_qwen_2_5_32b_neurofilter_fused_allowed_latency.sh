#!/usr/bin/env bash

set -euo pipefail

LATENCY_ROOT=/path/to/code
LATENCY_SCRIPT="$LATENCY_ROOT/notebooks/benchmark_qwen_2_5_32b_neurofilter_fused_allowed_latency.py"
LATENCY_OUTPUT="$LATENCY_ROOT/logs/qwen_2_5_32b_neurofilter_fused_allowed_latency_v1"
LATENCY_PYTHON="${PYTHON_BIN:-python}"

mkdir -p "$LATENCY_OUTPUT" "$LATENCY_OUTPUT/.matplotlib" "$LATENCY_OUTPUT/.cache"
exec > >(tee -a "$LATENCY_OUTPUT/00_driver.out") 2>&1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="$LATENCY_OUTPUT/.matplotlib"
export XDG_CACHE_HOME="$LATENCY_OUTPUT/.cache"

echo "Python: $(command -v "$LATENCY_PYTHON")"
echo "Output: $LATENCY_OUTPUT"
echo "Model: Qwen 2.5 32B Instruct"
echo "Data: 20 saved benign CMPL Insurance test conversations (400 turns)"
echo "Measurement: normal generation versus the same generation with NeuroFilter"
echo "Extra NeuroFilter model forwards: 0"
echo "API calls: disabled"

LATENCY_EXTRA_ARGS=()
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
    LATENCY_EXTRA_ARGS+=(--validate-only)
fi

"$LATENCY_PYTHON" -u "$LATENCY_SCRIPT" --output-dir "$LATENCY_OUTPUT" "${LATENCY_EXTRA_ARGS[@]}" 2>&1 | tee -a "$LATENCY_OUTPUT/run.out"

echo "Corrected NeuroFilter latency benchmark finished."
