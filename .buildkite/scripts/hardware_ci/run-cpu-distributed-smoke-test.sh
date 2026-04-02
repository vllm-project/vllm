#!/bin/bash
set -euox pipefail
export VLLM_CPU_CI_ENV=0
export VLLM_CPU_KVCACHE_SPACE=1 # avoid OOM

echo "--- PP+TP"
NUMA_COUNT=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)
echo "System Check: Found $NUMA_COUNT NUMA node(s)."

if [ "$NUMA_COUNT" -lt 4 ]; then
    echo "Warning: NUMA count ($NUMA_COUNT) is less than 4. Skipping vLLM server startup."
else
    vllm serve meta-llama/Llama-3.2-3B-Instruct -tp=2 -pp=2 --max-model-len=4096 &
    server_pid=$!
    timeout 600 bash -c "until curl localhost:8000/v1/models > /dev/null 2>&1; do sleep 1; done" || exit 1
    vllm bench serve \
        --backend vllm \
        --dataset-name random \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --num-prompts 20 \
        --result-dir ./test_results \
        --result-filename tp_pp.json \
        --save-result \
        --endpoint /v1/completions
    kill -s SIGTERM $server_pid; wait $server_pid || true
    failed_req=$(jq '.failed' ./test_results/tp_pp.json)
    if [ "$failed_req" -ne 0 ]; then
      echo "Some requests were failed!"
      exit 1
    fi
fi
echo "--- DP+TP"
NUMA_COUNT=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)
echo "System Check: Found $NUMA_COUNT NUMA node(s)."

if [ "$NUMA_COUNT" -lt 4 ]; then
    echo "Warning: NUMA count ($NUMA_COUNT) is less than 4. Skipping vLLM server startup."
else
    vllm serve meta-llama/Llama-3.2-3B-Instruct -tp=2 -dp=2 --max-model-len=4096 &
    server_pid=$!
    timeout 600 bash -c "until curl localhost:8000/v1/models > /dev/null 2>&1; do sleep 1; done" || exit 1
    vllm bench serve \
        --backend vllm \
        --dataset-name random \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --num-prompts 20 \
        --result-dir ./test_results \
        --result-filename dp_pp.json \
        --save-result \
        --endpoint /v1/completions
    kill -s SIGTERM $server_pid; wait $server_pid || true
    failed_req=$(jq '.failed' ./test_results/dp_pp.json)
    if [ "$failed_req" -ne 0 ]; then
      echo "Some requests were failed!"
      exit 1
    fi
fi
