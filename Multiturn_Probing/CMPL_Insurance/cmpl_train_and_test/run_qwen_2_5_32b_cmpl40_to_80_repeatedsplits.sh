#!/usr/bin/env bash

set -euo pipefail

QWEN_ROOT=/path/to/code
QWEN_NOTEBOOK_DIR="$QWEN_ROOT/notebooks"
QWEN_RUN_LOG_DIR="$QWEN_ROOT/logs/qwen_2_5_32b_40_to_80_v1"

QWEN_BASELINE_SCRIPT="$QWEN_NOTEBOOK_DIR/trajectoryprobe_cmpl_insurance_multiturn_w_acc_Qwen_2_5_32B_kfoldcrossval_cmpl_train_cmpltest.py"
QWEN_REPEATED_SCRIPT="$QWEN_NOTEBOOK_DIR/trajectoryprobe_cmpl_insurance_multiturn_w_acc_Qwen_2_5_32B_repeatedsplits_cmplcount_v1.py"

QWEN_BASELINE_CACHE="$QWEN_RUN_LOG_DIR/baseline40_training_cache.jsonl"
QWEN_BASELINE_TRACE="$QWEN_RUN_LOG_DIR/baseline40_training_trace.jsonl"
QWEN_BASELINE_ONLINE="$QWEN_RUN_LOG_DIR/baseline40_live_test.jsonl"

QWEN_BASELINE_DONE="$QWEN_RUN_LOG_DIR/.baseline40_complete"
QWEN_REPEATED_TRAINING_DONE="$QWEN_RUN_LOG_DIR/.repeated80_training_complete"
QWEN_REPEATED_LIVE_DONE="$QWEN_RUN_LOG_DIR/.repeated80_live_complete"

mkdir -p "$QWEN_RUN_LOG_DIR"
exec > >(tee -a "$QWEN_RUN_LOG_DIR/00_driver.out") 2>&1

if [[ -z "${CMPL_OPENAI_ENDPOINT:-${ATTACKER_OPENAI_ENDPOINT:-}}" ]]; then
    echo "Missing CMPL_OPENAI_ENDPOINT or ATTACKER_OPENAI_ENDPOINT."
    exit 1
fi

if [[ -z "${CMPL_OPENAI_API_KEY:-${ATTACKER_OPENAI_API_KEY:-}}" ]]; then
    echo "Missing CMPL_OPENAI_API_KEY or ATTACKER_OPENAI_API_KEY."
    exit 1
fi

echo "Python: $(command -v python)"
echo "Run logs: $QWEN_RUN_LOG_DIR"

if [[ ! -f "$QWEN_BASELINE_DONE" ]]; then
    echo "Starting the 40-trajectory baseline run."
    python -u "$QWEN_BASELINE_SCRIPT" --train-subject-start 0 --train-subject-end 19 --subject-start 20 --subject-end 39 --cmpl-training-cache "$QWEN_BASELINE_CACHE" --cmpl-training-trace-log "$QWEN_BASELINE_TRACE" --online-output "$QWEN_BASELINE_ONLINE" --force-retrain-probe --training-only 2>&1 | tee -a "$QWEN_RUN_LOG_DIR/01_baseline40.out"
    touch "$QWEN_BASELINE_DONE"
else
    echo "The 40-trajectory baseline run is already complete; skipping it."
fi

if [[ ! -f "$QWEN_REPEATED_TRAINING_DONE" ]]; then
    echo "Starting repeated-split training with 80 trajectories."
    python -u "$QWEN_REPEATED_SCRIPT" --trajectory-count 80 --stage training --repeated-splits 10 --validation-fraction 0.30 --artifact-log-dir "$QWEN_RUN_LOG_DIR" --baseline-training-cache "$QWEN_BASELINE_CACHE" --force-retrain-probe 2>&1 | tee -a "$QWEN_RUN_LOG_DIR/02_repeatedsplits80_training.out"
    touch "$QWEN_REPEATED_TRAINING_DONE"
else
    echo "The 80-trajectory repeated-split training stage is already complete; skipping it."
fi

if [[ ! -f "$QWEN_REPEATED_LIVE_DONE" ]]; then
    echo "Starting the 80-trajectory live test."
    python -u "$QWEN_REPEATED_SCRIPT" --trajectory-count 80 --stage live --repeated-splits 10 --validation-fraction 0.30 --artifact-log-dir "$QWEN_RUN_LOG_DIR" --baseline-training-cache "$QWEN_BASELINE_CACHE" 2>&1 | tee -a "$QWEN_RUN_LOG_DIR/03_repeatedsplits80_live.out"
    touch "$QWEN_REPEATED_LIVE_DONE"
else
    echo "The 80-trajectory live test is already complete; skipping it."
fi

echo "All Qwen CMPL stages are complete."
