#!/usr/bin/env bash

set -euo pipefail

ROOT=/path/to/code
TRAIN_SCRIPT="$ROOT/notebooks/cmpl_alltrain40_delta_vs_displacement_sharedtest.py"
LIVE_SCRIPT="$ROOT/notebooks/cmpl_alltrain40_delta_vs_displacement_live_crescendo_v2.py"
TRAIN_DIR="$ROOT/logs/cmpl_alltrain80_delta_vs_displacement_sharedtest_v1/qwen_2_5_32b"
LIVE_DIR="$ROOT/logs/cmpl_alltrain80_delta_vs_displacement_live_crescendo_v2/qwen_2_5_32b"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$TRAIN_DIR" "$LIVE_DIR"
exec > >(tee -a "$LIVE_DIR/00_driver.out") 2>&1

echo "Python: $(command -v "$PYTHON_BIN")"
echo "Training artifacts: $TRAIN_DIR"
echo "Live artifacts: $LIVE_DIR"
echo "Training subjects: 0-19 and 40-59"
echo "Live Crescendo test subjects: 20-39 and 60-79"

"$PYTHON_BIN" -u "$TRAIN_SCRIPT" --model qwen_2_5_32b --trajectory-count 80 --artifact-dir "$TRAIN_DIR" --training-only 2>&1 | tee -a "$TRAIN_DIR/retrain_displacement_50_50.out"

"$PYTHON_BIN" -u "$LIVE_SCRIPT" --trajectory-count 80 --artifact-dir "$LIVE_DIR" 2>&1 | tee -a "$LIVE_DIR/run.out"

echo "Qwen alltrain80 delta-versus-displacement live Crescendo comparison completed."
