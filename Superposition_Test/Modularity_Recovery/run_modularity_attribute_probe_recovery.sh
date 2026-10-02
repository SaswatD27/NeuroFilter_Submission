#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {qwen_2_5_32b|gpt_oss_20b|llama_3_3_70b}"
    exit 2
fi

MODEL_KEY=$1
case "$MODEL_KEY" in
    qwen_2_5_32b|gpt_oss_20b|llama_3_3_70b) ;;
    *)
        echo "Unknown model: $MODEL_KEY"
        exit 2
        ;;
esac

ROOT=/path/to/code
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_DIR="$ROOT/logs/modularity_attribute_probe_recovery_v1/$MODEL_KEY"
SCRIPT="$ROOT/notebooks/recover_modularity_attribute_probes.py"
export MPLCONFIGDIR="$ROOT/temp_modularity_attribute_probe_recovery_v1/_matplotlib"
export XDG_CACHE_HOME="$ROOT/temp_modularity_attribute_probe_recovery_v1/_xdg_cache"

mkdir -p "$RUN_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
echo "Python: $PYTHON_BIN"
echo "Model: $MODEL_KEY"
echo "Output: $RUN_DIR"
echo "Activation cache: $ROOT/temp_modularity_attribute_probe_recovery_v1/$MODEL_KEY"
echo "API calls: none"
echo "This is the full attribute-superposition/AutoDAN cosine experiment and is resumable."

"$PYTHON_BIN" -u "$SCRIPT" --model "$MODEL_KEY" 2>&1 | tee -a "$RUN_DIR/run.out"
