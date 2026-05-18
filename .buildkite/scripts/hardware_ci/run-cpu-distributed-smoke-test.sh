#!/bin/bash
set -euox pipefail
export VLLM_CPU_CI_ENV=0
export VLLM_CPU_KVCACHE_SPACE=1 # avoid OOM

MODE=${1:-all}

run_scenario() {
    local label="$1" result_file="$2"
    shift 2
    echo "--- $label"
    vllm serve meta-llama/Llama-3.2-3B-Instruct "$@" --max-model-len=4096 &
    local server_pid=$!
    timeout 600 bash -c "until curl localhost:8000/v1/models > /dev/null 2>&1; do sleep 1; done" || exit 1
    vllm bench serve \
        --backend vllm \
        --dataset-name random \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --num-prompts 20 \
        --result-dir ./test_results \
        --result-filename "$result_file" \
        --save-result \
        --endpoint /v1/completions
    kill -s SIGTERM "$server_pid"; wait "$server_pid" || true
    if [ "$(jq '.failed' "./test_results/$result_file")" -ne 0 ]; then
        echo "Some requests were failed in $label!"
        exit 1
    fi
}

case "$MODE" in
    tp_pp) run_scenario "PP+TP" tp_pp.json -tp=2 -pp=2 ;;
    dp_tp) run_scenario "DP+TP" dp_tp.json -tp=2 -dp=2 ;;
    all)
        run_scenario "PP+TP" tp_pp.json -tp=2 -pp=2
        run_scenario "DP+TP" dp_tp.json -tp=2 -dp=2
        ;;
    *) echo "ERROR: unknown mode '$MODE' (expected: tp_pp | dp_tp | all)" >&2; exit 1 ;;
esac
