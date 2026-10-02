#!/usr/bin/env bash

set -euo pipefail

LLAMA_ROOT=/path/to/code
LLAMA_NOTEBOOK_DIR="$LLAMA_ROOT/notebooks"
LLAMA_RUN_LOG_DIR="$LLAMA_ROOT/logs/llama_3_3_70b_40_to_80_v1"

LLAMA_BASELINE_SCRIPT="$LLAMA_NOTEBOOK_DIR/trajectoryprobe_cmpl_insurance_multiturn_w_acc_llama_3_3_70B_kfoldcrossval_cmpl_train_cmpltest.py"
LLAMA_REPEATED_SCRIPT="$LLAMA_NOTEBOOK_DIR/trajectoryprobe_cmpl_insurance_multiturn_w_acc_llama_3_3_70B_repeatedsplits_cmplcount_v1.py"

LLAMA_BASELINE_CACHE="$LLAMA_RUN_LOG_DIR/baseline40_training_cache.jsonl"
LLAMA_BASELINE_TRACE="$LLAMA_RUN_LOG_DIR/baseline40_training_trace.jsonl"
LLAMA_BASELINE_ONLINE="$LLAMA_RUN_LOG_DIR/baseline40_live_test.jsonl"

LLAMA_BASELINE_DONE="$LLAMA_RUN_LOG_DIR/.baseline40_complete"
LLAMA_REPEATED_TRAINING_DONE="$LLAMA_RUN_LOG_DIR/.repeated80_training_complete"
LLAMA_REPEATED_LIVE_DONE="$LLAMA_RUN_LOG_DIR/.repeated80_live_complete"

mkdir -p "$LLAMA_RUN_LOG_DIR"
exec > >(tee -a "$LLAMA_RUN_LOG_DIR/00_driver.out") 2>&1

if [[ -z "${CMPL_OPENAI_ENDPOINT:-${ATTACKER_OPENAI_ENDPOINT:-}}" ]]; then
    echo "Missing CMPL_OPENAI_ENDPOINT or ATTACKER_OPENAI_ENDPOINT."
    exit 1
fi

if [[ -z "${CMPL_OPENAI_API_KEY:-${ATTACKER_OPENAI_API_KEY:-}}" ]]; then
    echo "Missing CMPL_OPENAI_API_KEY or ATTACKER_OPENAI_API_KEY."
    exit 1
fi

echo "Python: $(command -v python)"
echo "Run logs: $LLAMA_RUN_LOG_DIR"

if [[ ! -f "$LLAMA_BASELINE_DONE" ]]; then
    echo "Starting the 40-trajectory baseline run."
    python -u "$LLAMA_BASELINE_SCRIPT" --train-subject-start 0 --train-subject-end 19 --subject-start 20 --subject-end 39 --cmpl-training-cache "$LLAMA_BASELINE_CACHE" --cmpl-training-trace-log "$LLAMA_BASELINE_TRACE" --online-output "$LLAMA_BASELINE_ONLINE" --force-retrain-probe --training-only 2>&1 | tee -a "$LLAMA_RUN_LOG_DIR/01_baseline40.out"
    touch "$LLAMA_BASELINE_DONE"
else
    echo "The 40-trajectory baseline run is already complete; skipping it."
fi

if [[ ! -f "$LLAMA_REPEATED_TRAINING_DONE" ]]; then
    echo "Starting repeated-split training with 80 trajectories."
    python -u "$LLAMA_REPEATED_SCRIPT" --trajectory-count 80 --stage training --repeated-splits 10 --validation-fraction 0.30 --artifact-log-dir "$LLAMA_RUN_LOG_DIR" --baseline-training-cache "$LLAMA_BASELINE_CACHE" --force-retrain-probe 2>&1 | tee -a "$LLAMA_RUN_LOG_DIR/02_repeatedsplits80_training.out"
    touch "$LLAMA_REPEATED_TRAINING_DONE"
else
    echo "The 80-trajectory repeated-split training stage is already complete; skipping it."
fi

if [[ ! -f "$LLAMA_REPEATED_LIVE_DONE" ]]; then
    echo "Starting the 80-trajectory live test."
    python -u "$LLAMA_REPEATED_SCRIPT" --trajectory-count 80 --stage live --repeated-splits 10 --validation-fraction 0.30 --artifact-log-dir "$LLAMA_RUN_LOG_DIR" --baseline-training-cache "$LLAMA_BASELINE_CACHE" 2>&1 | tee -a "$LLAMA_RUN_LOG_DIR/03_repeatedsplits80_live.out"
    touch "$LLAMA_REPEATED_LIVE_DONE"
else
    echo "The 80-trajectory live test is already complete; skipping it."
fi

echo "All Llama CMPL stages are complete."
