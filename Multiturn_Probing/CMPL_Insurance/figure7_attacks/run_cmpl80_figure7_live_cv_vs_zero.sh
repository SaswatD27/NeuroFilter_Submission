#!/usr/bin/env bash

set -euo pipefail

FIG7_ROOT=/path/to/code
FIG7_RUN_ROOT="$FIG7_ROOT/logs/cmpl80_figure7_live_cv_vs_zero_v1"
FIG7_RUNNER="$FIG7_ROOT/notebooks/cmpl80_figure7_live_eval.py"
FIG7_PLOTTER="$FIG7_ROOT/notebooks/plot_cmpl80_figure7_live_cv_vs_zero.py"
FIG7_PYTHON="${PYTHON_BIN:-python}"
FIG7_MODELS="${MODELS:-qwen_2_5_32b llama_3_3_70b gpt_oss_20b}"

mkdir -p "$FIG7_RUN_ROOT"
exec > >(tee -a "$FIG7_RUN_ROOT/00_driver.out") 2>&1

echo "Python: $(command -v "$FIG7_PYTHON")"
echo "Artifacts: $FIG7_RUN_ROOT"
echo "Models: $FIG7_MODELS"
echo "Training subjects: 0-19 and 40-59"
echo "Live test subjects: 20-39 and 60-79"

FIG7_EXTRA_ARGS=()
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
    FIG7_EXTRA_ARGS+=(--validate-only)
else
    if [[ -z "${ATTACKER_OPENAI_ENDPOINT:-}" || -z "${ATTACKER_OPENAI_API_KEY:-}" ]]; then
        echo "ATTACKER_OPENAI_ENDPOINT and ATTACKER_OPENAI_API_KEY must be set."
        exit 1
    fi
fi

for FIG7_MODEL in $FIG7_MODELS; do
    FIG7_MODEL_DIR="$FIG7_RUN_ROOT/$FIG7_MODEL"
    mkdir -p "$FIG7_MODEL_DIR"
    echo "Starting $FIG7_MODEL"
    "$FIG7_PYTHON" -u "$FIG7_RUNNER" --model "$FIG7_MODEL" --output-dir "$FIG7_MODEL_DIR" "${FIG7_EXTRA_ARGS[@]}" 2>&1 | tee -a "$FIG7_MODEL_DIR/run.out"
done

if [[ "${VALIDATE_ONLY:-0}" != "1" ]]; then
    "$FIG7_PYTHON" -u "$FIG7_PLOTTER" --run-root "$FIG7_RUN_ROOT" 2>&1 | tee -a "$FIG7_RUN_ROOT/plot.out"
fi

echo "All requested Figure 7 evaluations completed."
