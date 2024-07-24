#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_PORT=12345
export NCCL_BUFFSIZE=2147483648

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# prefilling instance
VLLM_DISAGG_PREFILL_ROLE=prefill CUDA_VISIBLE_DEVICES=0,1,2,3 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model neuralmagic/Meta-Llama-3-70B-Instruct-FP8 \
    --port 8100 \
    -tp 4 \
    --enable-prefix-caching &

# decoding instance
VLLM_DISAGG_PREFILL_ROLE=decode CUDA_VISIBLE_DEVICES=4,5,6,7 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model neuralmagic/Meta-Llama-3-70B-Instruct-FP8 \
    --port 8200 \
    -tp 4 \
    --enable-prefix-caching &


wait_for_server 8100
wait_for_server 8200

# sending an example request
# in disaggregated prefilling, there are two steps of sending a request:
#   1. send the request to prefill instance, with max_tokens set to 1
#   2. send the request again to decode instance, no modification

# send to prefill instance
curl http://localhost:8100/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
"prompt": "San Francisco is a",
"max_tokens": 1,
"temperature": 0
}' &

# send to decode instance
curl http://localhost:8200/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
"prompt": "San Francisco is a",
"max_tokens": 5,
"temperature": 0
}'

