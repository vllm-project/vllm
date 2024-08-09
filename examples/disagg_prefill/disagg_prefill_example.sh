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
    --max-model-len 10000 \
    --gpu-memory-utilization 0.8 &

# decoding instance
VLLM_RPC_PORT=5580 VLLM_DISAGG_PREFILL_ROLE=decode CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8200 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.8 &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# launch a proxy server that opens the service at port 8000
python3 ../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py &
sleep 1

# serve an example request
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}'

# clean up
ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9