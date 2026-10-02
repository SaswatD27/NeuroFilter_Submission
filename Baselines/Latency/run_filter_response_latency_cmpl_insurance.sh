#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/path/to/code/notebooks/benchmark_filter_response_latency.py
OUTPUT=/path/to/code/logs/filter_response_latency_cmpl_insurance_v1

# Prevent incompatible packages in ~/.local from overriding this conda env.
export PYTHONNOUSERSITE=1

mkdir -p "$OUTPUT"

echo "Python: $PYTHON"
echo "Output: $OUTPUT"
echo "API calls: disabled"
echo "Target response generation inside timing: disabled"
echo "Methods: single-turn NeuroFilter and Llama Guard 4; multi-turn NeuroFilter, Llama Guard 4, and Agentic Firewall"

run_method() {
    local setting=$1
    local method=$2
    local log="$OUTPUT/${setting}_${method}.out"
    echo "Starting $setting $method; log: $log"
    "$PYTHON" -u "$SCRIPT" --setting "$setting" --filter "$method" --output-dir "$OUTPUT" >> "$log" 2>&1
}

if [[ $# -eq 0 ]]; then
    METHODS=(single_neurofilter single_llama_guard_4 multi_neurofilter multi_llama_guard_4 multi_agentic_firewall)
else
    METHODS=("$@")
fi

for method in "${METHODS[@]}"; do
    case "$method" in
        single_neurofilter) run_method single_turn neurofilter ;;
        single_llama_guard_4) run_method single_turn llama_guard_4 ;;
        multi_neurofilter) run_method multi_turn neurofilter ;;
        multi_llama_guard_4) run_method multi_turn llama_guard_4 ;;
        multi_agentic_firewall) run_method multi_turn agentic_firewall ;;
        *)
            echo "Unknown method: $method" >&2
            echo "Allowed: single_neurofilter single_llama_guard_4 multi_neurofilter multi_llama_guard_4 multi_agentic_firewall" >&2
            exit 2
            ;;
    esac
done

"$PYTHON" -u "$SCRIPT" --summarize-only --output-dir "$OUTPUT" >> "$OUTPUT/summary.out" 2>&1
echo "Completed requested methods. Summary: $OUTPUT/latency_summary.md"
