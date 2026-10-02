#!/usr/bin/env bash

set -euo pipefail

RUN_ROOT=/path/to/code
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_DIR="$RUN_ROOT/logs/cmpl80_new_models_fair_threshold0_v1/qwen_3_8_27b"
RUNNER="$RUN_ROOT/notebooks/trajectoryprobe_cmpl_insurance_multiturn_w_acc_qwen_3.8_27B_repeatedsplits_cmplcount_v1.py"
BASELINE_CACHE="$RUN_ROOT/logs/trajectoryprobe_live_cmpl_training_cache_Qwen3_8_27B_FP8_insurance.jsonl"
TRAINING_DONE="$RUN_DIR/.training_complete"
LIVE_DONE="$RUN_DIR/.live_complete"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/00_driver.out") 2>&1

if [[ -z "${CMPL_OPENAI_ENDPOINT:-${ATTACKER_OPENAI_ENDPOINT:-}}" ]]; then
    echo "Missing CMPL_OPENAI_ENDPOINT or ATTACKER_OPENAI_ENDPOINT."
    exit 1
fi
if [[ -z "${CMPL_OPENAI_API_KEY:-${ATTACKER_OPENAI_API_KEY:-}}" ]]; then
    echo "Missing CMPL_OPENAI_API_KEY or ATTACKER_OPENAI_API_KEY."
    exit 1
fi
if [[ ! -f "$BASELINE_CACHE" ]]; then
    echo "Missing reusable subjects 0-19 training cache: $BASELINE_CACHE"
    exit 1
fi

echo "Python: $PYTHON_BIN"
echo "Output: $RUN_DIR"
echo "Training subjects: 0-19 and 40-59 (40 attack + 40 benign)"
echo "Live test subjects: 20-39 and 60-79 (40 attack + 40 benign)"
echo "Live threshold: 0"

if [[ ! -f "$TRAINING_DONE" ]]; then
    "$PYTHON_BIN" -u "$RUNNER" --trajectory-count 80 --stage training --repeated-splits 10 --validation-fraction 0.30 --artifact-log-dir "$RUN_DIR" --baseline-training-cache "$BASELINE_CACHE" --live-threshold zero --force-retrain-probe 2>&1 | tee -a "$RUN_DIR/01_training.out"
    touch "$TRAINING_DONE"
else
    echo "Training is already complete; reusing the CMPL-80 probe."
fi

if [[ ! -f "$LIVE_DONE" ]]; then
    "$PYTHON_BIN" -u "$RUNNER" --trajectory-count 80 --stage live --repeated-splits 10 --validation-fraction 0.30 --artifact-log-dir "$RUN_DIR" --baseline-training-cache "$BASELINE_CACHE" --live-threshold zero 2>&1 | tee -a "$RUN_DIR/02_live_threshold0.out"
    touch "$LIVE_DONE"
else
    echo "Threshold-0 live testing is already complete."
fi

echo "Qwen 3.8 CMPL-80 threshold-0 evaluation completed."
