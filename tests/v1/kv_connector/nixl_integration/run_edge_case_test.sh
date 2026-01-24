#!/bin/bash
set -xe

# Parse command line arguments
KV_BUFFER_DEVICE="cuda"  # Default to cuda
PREFILL_GPU_ID=4         # Default GPU IDs
DECODE_GPU_ID=5
while [[ $# -gt 0 ]]; do
  case $1 in
    --kv_buffer_device)
      KV_BUFFER_DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--kv_buffer_device <cuda|cpu>]"
      exit 1
      ;;
  esac
done

echo "Running edge case tests with kv_buffer_device=$KV_BUFFER_DEVICE (GPUs: $PREFILL_GPU_ID, $DECODE_GPU_ID)"

# Build the kv-transfer-config once
if [[ "$KV_BUFFER_DEVICE" == "cuda" ]]; then
  KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
else
  KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"$KV_BUFFER_DEVICE\"}"
fi

# Models to run
MODELS=(
    "Qwen/Qwen3-0.6B"
)

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

# Function to clean up previous instances
cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm serve" || true
  sleep 2
}

# Handle to get model-specific arguments for deepseek
get_model_args() {
  local model_name=$1
  local extra_args=""

  if [[ "$model_name" == "deepseek-ai/deepseek-vl2-tiny" ]]; then
    extra_args="--hf_overrides '{\"architectures\": [\"DeepseekVLV2ForCausalLM\"]}' --trust-remote-code"
  fi

  echo "$extra_args"
}


# Function to run tests for a specific model
run_tests_for_model() {
  local model_name=$1
  echo "================================"
  echo "Testing model: $model_name"
  echo "================================"

  # Get model-specific arguments
  local model_args=$(get_model_args "$model_name")

  # Start prefill instance
  PREFILL_PORT=8001

  BASE_CMD="CUDA_VISIBLE_DEVICES=$PREFILL_GPU_ID VLLM_NIXL_SIDE_CHANNEL_PORT=5559 vllm serve $model_name \
  --port $PREFILL_PORT \
  --enforce-eager \
  --gpu-memory-utilization 0.2 \
  --kv-transfer-config '$KV_CONFIG'"

  if [ -n "$model_args" ]; then
  FULL_CMD="$BASE_CMD $model_args"
  else
  FULL_CMD="$BASE_CMD"
  fi

  eval "$FULL_CMD &"

  # Start decode instance
  DECODE_PORT=8002

  # Build the command with or without model-specific args
  BASE_CMD="CUDA_VISIBLE_DEVICES=$DECODE_GPU_ID VLLM_NIXL_SIDE_CHANNEL_PORT=6000 vllm serve $model_name \
  --port $DECODE_PORT \
  --enforce-eager \
  --gpu-memory-utilization 0.2 \
  --kv-transfer-config '$KV_CONFIG'"

  if [ -n "$model_args" ]; then
  FULL_CMD="$BASE_CMD $model_args"
  else
  FULL_CMD="$BASE_CMD"
  fi

  eval "$FULL_CMD &"

  # Wait for all instances to start
  echo "Waiting for prefill instance on port $PORT to start..."
  wait_for_server $PREFILL_PORT
  echo "Waiting for decode instance on port $PORT to start..."
  wait_for_server $DECODE_PORT

  # Build the command for the proxy server with all the hosts and ports
  PROXY_PORT=8192
  PROXY_CMD="python ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port $PROXY_PORT"
  PROXY_CMD+=" --prefiller-ports ${PREFILL_PORT}"
  PROXY_CMD+=" --decoder-ports ${DECODE_PORT}"
  # Start the proxy server
  echo "Starting proxy server with command: $PROXY_CMD"
  $PROXY_CMD &

  # Wait for the proxy to start
  sleep 5

  # Run lm eval for this model
  echo "Running tests for $model_name"
  PREFILL_PORT=$PREFILL_PORT DECODE_PORT=$DECODE_PORT PROXY_PORT=$PROXY_PORT python -m pytest -s -v ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_edge_cases.py

  # Clean up before running next model
  cleanup_instances
  sleep 3
}

# Run tests for each model
for model in "${MODELS[@]}"; do
  run_tests_for_model "$model"
done

echo "All tests completed!"
