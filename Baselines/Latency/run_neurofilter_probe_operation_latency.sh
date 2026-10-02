#!/usr/bin/env bash

set -euo pipefail

BENCHMARK_ROOT=/path/to/code
BENCHMARK_SCRIPT="$BENCHMARK_ROOT/notebooks/benchmark_neurofilter_probe_operation_latency.py"
BENCHMARK_OUTPUT="$BENCHMARK_ROOT/logs/neurofilter_probe_operation_latency_v1"
BENCHMARK_PYTHON="${PYTHON_BIN:-python}"

mkdir -p "$BENCHMARK_OUTPUT"
exec > >(tee -a "$BENCHMARK_OUTPUT/00_driver.out") 2>&1

export PYTHONNOUSERSITE=1

echo "Python: $(command -v "$BENCHMARK_PYTHON")"
echo "Output: $BENCHMARK_OUTPUT"
echo "Measurements: single-turn 7B and multi-turn 32B probe operations"
echo "Models loaded: none"
echo "API calls: none"

BENCHMARK_EXTRA_ARGS=()
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
    BENCHMARK_EXTRA_ARGS+=(--validate-only)
fi

"$BENCHMARK_PYTHON" -u "$BENCHMARK_SCRIPT" --output-dir "$BENCHMARK_OUTPUT" "${BENCHMARK_EXTRA_ARGS[@]}" 2>&1 | tee -a "$BENCHMARK_OUTPUT/run.out"

echo "Probe operation latency benchmark finished."
