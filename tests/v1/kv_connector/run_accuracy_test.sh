#!/bin/bash

set -xe

# Model to run.
MODEL_NAME=Qwen/Qwen3-0.6B

# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

# Waits for vLLM to start.
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# Prefill instance.
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5559 vllm serve $MODEL_NAME \
    --port 8100 \
    --enforce-eager \
    --disable-log-requests \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

# Decode instance.
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5558 vllm serve $MODEL_NAME \
    --port 8200 \
    --enforce-eager \
    --disable-log-requests \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# Proxy server.
python ${GIT_ROOT}/tests/v1/kv_connector/toy_proxy_server.py --port 8192 &

# Run lm eval.
python -m pytest -s -x ${GIT_ROOT}/tests/v1/kv_connector/test_accuracy.py
