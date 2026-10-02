#!/usr/bin/env bash

set -euo pipefail

GPT_OSS_ROOT=/path/to/code
GPT_OSS_NOTEBOOK_DIR="$GPT_OSS_ROOT/notebooks"
GPT_OSS_RUN_LOG_DIR="$GPT_OSS_ROOT/logs/gpt_oss_20b_40_to_80_v1"

GPT_OSS_BASELINE_SCRIPT="$GPT_OSS_NOTEBOOK_DIR/trajectoryprobe_cmpl_insurance_multiturn_w_acc_gpt_oss_20B_kfoldcrossval_layerthreshold_cmpl_train_cmpltest.py"
GPT_OSS_REPEATED_SCRIPT="$GPT_OSS_NOTEBOOK_DIR/trajectoryprobe_cmpl_insurance_multiturn_w_acc_gpt_oss_20B_repeatedsplits_cmplcount_v1.py"

GPT_OSS_BASELINE_CACHE="$GPT_OSS_RUN_LOG_DIR/baseline40_training_cache.jsonl"
GPT_OSS_BASELINE_TRACE="$GPT_OSS_RUN_LOG_DIR/baseline40_training_trace.jsonl"
GPT_OSS_BASELINE_ONLINE="$GPT_OSS_RUN_LOG_DIR/baseline40_live_test.jsonl"

GPT_OSS_BASELINE_DONE="$GPT_OSS_RUN_LOG_DIR/.baseline40_complete"
GPT_OSS_REPEATED_TRAINING_DONE="$GPT_OSS_RUN_LOG_DIR/.repeated80_training_complete"
GPT_OSS_REPEATED_LIVE_DONE="$GPT_OSS_RUN_LOG_DIR/.repeated80_live_complete"

mkdir -p "$GPT_OSS_RUN_LOG_DIR"
exec > >(tee -a "$GPT_OSS_RUN_LOG_DIR/00_driver.out") 2>&1

if [[ -z "${CMPL_OPENAI_ENDPOINT:-${ATTACKER_OPENAI_ENDPOINT:-}}" ]]; then
    echo "Missing CMPL_OPENAI_ENDPOINT or ATTACKER_OPENAI_ENDPOINT."
    exit 1
fi

if [[ -z "${CMPL_OPENAI_API_KEY:-${ATTACKER_OPENAI_API_KEY:-}}" ]]; then
    echo "Missing CMPL_OPENAI_API_KEY or ATTACKER_OPENAI_API_KEY."
    exit 1
fi

echo "Python: $(command -v python)"
echo "Run logs: $GPT_OSS_RUN_LOG_DIR"

if [[ ! -f "$GPT_OSS_BASELINE_DONE" ]]; then
    echo "Starting the 40-trajectory baseline run."
    python -u "$GPT_OSS_BASELINE_SCRIPT" --train-subject-start 0 --train-subject-end 19 --subject-start 20 --subject-end 39 --cmpl-training-cache "$GPT_OSS_BASELINE_CACHE" --cmpl-training-trace-log "$GPT_OSS_BASELINE_TRACE" --online-output "$GPT_OSS_BASELINE_ONLINE" --force-retrain-probe --training-only 2>&1 | tee -a "$GPT_OSS_RUN_LOG_DIR/01_baseline40.out"
    touch "$GPT_OSS_BASELINE_DONE"
else
    echo "The 40-trajectory baseline run is already complete; skipping it."
fi

if [[ ! -f "$GPT_OSS_REPEATED_TRAINING_DONE" ]]; then
    echo "Starting repeated-split training with 80 trajectories."
    python -u "$GPT_OSS_REPEATED_SCRIPT" --trajectory-count 80 --stage training --repeated-splits 10 --validation-fraction 0.30 --artifact-log-dir "$GPT_OSS_RUN_LOG_DIR" --baseline-training-cache "$GPT_OSS_BASELINE_CACHE" --force-retrain-probe 2>&1 | tee -a "$GPT_OSS_RUN_LOG_DIR/02_repeatedsplits80_training.out"
    touch "$GPT_OSS_REPEATED_TRAINING_DONE"
else
    echo "The 80-trajectory repeated-split training stage is already complete; skipping it."
fi

if [[ ! -f "$GPT_OSS_REPEATED_LIVE_DONE" ]]; then
    echo "Starting the 80-trajectory live test."
    python -u "$GPT_OSS_REPEATED_SCRIPT" --trajectory-count 80 --stage live --repeated-splits 10 --validation-fraction 0.30 --artifact-log-dir "$GPT_OSS_RUN_LOG_DIR" --baseline-training-cache "$GPT_OSS_BASELINE_CACHE" 2>&1 | tee -a "$GPT_OSS_RUN_LOG_DIR/03_repeatedsplits80_live.out"
    touch "$GPT_OSS_REPEATED_LIVE_DONE"
else
    echo "The 80-trajectory live test is already complete; skipping it."
fi

echo "All GPT-OSS CMPL stages are complete."
