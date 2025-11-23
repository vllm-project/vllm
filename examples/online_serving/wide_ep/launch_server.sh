#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Launch script for the DeepSeek-V3 Wide-EP benchmarking recipe.
# Only COORDINATOR_IP and START_RANK come from the caller; everything else
# matches the published benchmark so results stay reproducible.
#
# Required environment variables:
#   COORDINATOR_IP   - IP address of the coordinator node
#   START_RANK       - Starting rank for this node (0, 8, 16, 24)
#
set -euo pipefail

COORDINATOR_IP=${COORDINATOR_IP:-}
START_RANK=${START_RANK:-}

COORDINATOR_PORT=13345
DATA_PARALLEL_SIZE=32
DATA_PARALLEL_SIZE_LOCAL=8
API_SERVER_COUNT=8
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=512
MODEL="deepseek-ai/DeepSeek-V3-0324"

if [[ -z "$COORDINATOR_IP" || -z "$START_RANK" ]]; then
    echo "Error: set COORDINATOR_IP and START_RANK before running this benchmark script."
    echo "Example: COORDINATOR_IP=10.0.0.1 START_RANK=8 ./launch_server.sh"
    exit 1
fi

# Set environment variables
export VLLM_USE_DEEP_GEMM=1
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_MOE_DP_CHUNK_SIZE=512
export VLLM_SKIP_P2P_CHECK=1
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
export NVIDIA_GDRCOPY=enabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
export NVSHMEM_QP_DEPTH=1512
export GLOO_SOCKET_IFNAME=eth0

# Launch vLLM server
uv run vllm serve "$MODEL" \
  --data-parallel-hybrid-lb \
  --api-server-count "$API_SERVER_COUNT" \
  --data-parallel-address "$COORDINATOR_IP" \
  --data-parallel-rpc-port "$COORDINATOR_PORT" \
  --tensor-parallel-size 1 \
  --data-parallel-size "$DATA_PARALLEL_SIZE" \
  --data-parallel-size-local "$DATA_PARALLEL_SIZE_LOCAL" \
  --data-parallel-start-rank "$START_RANK" \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-dbo \
  --dbo-decode-token-threshold 32 \
  --async-scheduling \
  --enable-eplb \
  --eplb-config '{"window_size":"1000","step_interval":"3000","num_redundant_experts":"32","log_balancedness":"False"}' \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --compilation_config '{"pass_config":{"enable_fusion":true,"enable_attn_fusion":true,"enable_noop":true},"custom_ops":["+rms_norm"],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both"}'

