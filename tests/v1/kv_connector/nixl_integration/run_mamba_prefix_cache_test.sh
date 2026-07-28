#!/bin/bash
set -xe

# E2E test: Mamba hybrid prefix cache hits in PD disaggregation.
# Spins up a 1P1D setup with a Mamba hybrid model and verifies
# repeated prompts yield non-zero D-side prefix cache hits.

PREFILL_GPU_ID=${PREFILL_GPU_ID:-0}
DECODE_GPU_ID=${DECODE_GPU_ID:-1}
MODEL=${MODEL:-"ibm-granite/granite-4.0-h-tiny"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
VLLM_SERVE_EXTRA_ARGS=${VLLM_SERVE_EXTRA_ARGS:-}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-FLASHINFER}

echo "Running Mamba prefix cache test (GPUs: P=$PREFILL_GPU_ID, D=$DECODE_GPU_ID, model=$MODEL, backend=$ATTENTION_BACKEND)"

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

# Resolve repository root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)}"

trap 'kill $(jobs -pr) 2>/dev/null' SIGINT SIGTERM EXIT

wait_for_server() {
  local port=$1
  timeout 600 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm serve" || true
  sleep 2
}

cleanup_instances

EXTRA_ARGS=()
if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
  IFS=',' read -r -a EXTRA_ARGS <<< "$VLLM_SERVE_EXTRA_ARGS"
fi
if [[ -n "$ATTENTION_BACKEND" ]]; then
  EXTRA_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi

# Start prefill instance
PREFILL_PORT=8001
CUDA_VISIBLE_DEVICES=$PREFILL_GPU_ID \
VLLM_SSM_CONV_STATE_LAYOUT=DS \
VLLM_KV_CACHE_LAYOUT=HND \
VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
vllm serve $MODEL \
  --port $PREFILL_PORT \
  --enforce-eager \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-model-len 16384 \
  --block-size 128 \
  --trust-remote-code \
  --enable-prefix-caching \
  --mamba-cache-mode all \
  --kv-transfer-config "$KV_CONFIG" \
  "${EXTRA_ARGS[@]}" &

# Start decode instance
DECODE_PORT=8002
CUDA_VISIBLE_DEVICES=$DECODE_GPU_ID \
VLLM_SSM_CONV_STATE_LAYOUT=DS \
VLLM_KV_CACHE_LAYOUT=HND \
VLLM_NIXL_SIDE_CHANNEL_PORT=6000 \
vllm serve $MODEL \
  --port $DECODE_PORT \
  --enforce-eager \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-model-len 16384 \
  --block-size 128 \
  --trust-remote-code \
  --enable-prefix-caching \
  --mamba-cache-mode all \
  --kv-transfer-config "$KV_CONFIG" \
  "${EXTRA_ARGS[@]}" &

echo "Waiting for prefill instance on port $PREFILL_PORT..."
wait_for_server "$PREFILL_PORT"
echo "Waiting for decode instance on port $DECODE_PORT..."
wait_for_server "$DECODE_PORT"

# Start proxy
PROXY_PORT=8192
python3 "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
  --port $PROXY_PORT \
  --prefiller-ports $PREFILL_PORT \
  --decoder-ports $DECODE_PORT &

sleep 5

echo "Running Mamba prefix cache test..."
PREFILL_PORT=$PREFILL_PORT \
DECODE_PORT=$DECODE_PORT \
PROXY_PORT=$PROXY_PORT \
python3 -m pytest -s -v \
  "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_mamba_prefix_cache.py"

echo "Mamba prefix cache test passed!"

cleanup_instances
