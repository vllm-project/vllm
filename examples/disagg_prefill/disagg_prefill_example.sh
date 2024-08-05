#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export VLLM_PORT=12345

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# prefilling instance
VLLM_RPC_PORT=5570 VLLM_DISAGG_PREFILL_ROLE=prefill CUDA_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8100 \
    -tp 1 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.8 \
    --max-model-len 10000 &

# decoding instance
VLLM_RPC_PORT=5580 VLLM_DISAGG_PREFILL_ROLE=decode CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8200 \
    -tp 1 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.8 \
    --max-model-len 10000 &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# sending an example request
# in disaggregated prefilling, there are two steps of sending a request:
#   1. send the request to prefill instance, with max_tokens set to 1
#   2. send the request again to decode instance, no modification

# send to prefill instance, let it only do prefill by setting max_token=1
curl -m 5 http://localhost:8100/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
"prompt": "'$i' San Francisco is a",
"max_tokens": 1,
"temperature": 0
}'

# send to decode instance
curl -m 5 http://localhost:8100/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
"prompt": "'$i' San Francisco is a",
"max_tokens": 50,
"temperature": 0
}'


# clean up
ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9