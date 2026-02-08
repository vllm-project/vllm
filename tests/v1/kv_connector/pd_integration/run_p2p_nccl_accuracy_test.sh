#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# P2pNccl Connector accuracy test script
# Uses disagg_prefill_proxy_server.py which encodes KV addresses in request IDs

set -xe

# Required for NCCL on some systems (consistent with other distributed tests)
export NCCL_CUMEM_HOST_ENABLE=0

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-0.6B"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.3}
GIT_ROOT=$(git rev-parse --show-toplevel)

# Get the host IP that vLLM will use for KV transfer
# This must match what P2pNcclEngine binds to
KV_HOST=$(python3 -c "from vllm.utils.network_utils import get_ip; print(get_ip())")

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr) 2>/dev/null' SIGINT SIGTERM EXIT

wait_for_server() {
  local port=$1
  timeout 600 bash -c "until curl -s localhost:${port}/v1/completions > /dev/null; do sleep 1; done"
}

# Cleanup any existing instances
pkill -f "vllm serve" || true
sleep 2

# Prefill instance (kv_producer) - port 8100, kv_port 14579
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_NAME \
  --port 8100 \
  --max-model-len 4096 \
  --enforce-eager \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_port":14579}' &

# Decode instance (kv_consumer) - port 8200, kv_port 14580 (different!)
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME \
  --port 8200 \
  --max-model-len 4096 \
  --enforce-eager \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_port":14580}' &

echo "Waiting for prefill instance on port 8100 to start..."
wait_for_server 8100
echo "Waiting for decode instance on port 8200 to start..."
wait_for_server 8200

# Start proxy server with P2pNccl request ID encoding
# This proxy encodes prefill and decode KV addresses in the request ID
# Note: --kv-host must match the IP that vLLM's P2pNcclEngine binds to
python3 ${GIT_ROOT}/benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py \
  --port 8192 \
  --prefill-url http://localhost:8100 \
  --decode-url http://localhost:8200 \
  --kv-host ${KV_HOST} \
  --prefill-kv-port 14579 \
  --decode-kv-port 14580 &

sleep 5

# Run accuracy test
TEST_MODEL=$MODEL_NAME python3 -m pytest -s -x ${GIT_ROOT}/tests/v1/kv_connector/pd_integration/test_accuracy.py

pkill -f "vllm serve" || true
echo "P2pNccl accuracy test completed!"
