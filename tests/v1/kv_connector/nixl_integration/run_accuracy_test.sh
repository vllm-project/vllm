#!/bin/bash
set -xe

# Parse command line arguments
KV_BUFFER_DEVICE="cuda"  # Default to cuda
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

echo "Running accuracy tests with kv_buffer_device=$KV_BUFFER_DEVICE"

DECODER_KV_LAYOUT=${DECODER_KV_LAYOUT:-"HND"} # Default to HND, optional NHD
if [[ "$DECODER_KV_LAYOUT" == "NHD" ]]; then
  KV_CONFIG_HETERO_LAYOUT=',"enable_permute_local_kv":"True"'
else
  KV_CONFIG_HETERO_LAYOUT=''
fi

# Build the kv-transfer-config once
if [[ "$KV_BUFFER_DEVICE" == "cuda" ]]; then
  KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"'${KV_CONFIG_HETERO_LAYOUT}'}'
else
  KV_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"$KV_BUFFER_DEVICE\""${KV_CONFIG_HETERO_LAYOUT}"}"
fi

# Models to run
MODEL_NAMES=${MODEL_NAMES:-}
if [[ -n "$MODEL_NAMES" ]]; then
  MODELS=("$MODEL_NAMES")
else
  MODELS=(
      "Qwen/Qwen3-0.6B"
  )
fi

# Number of prefill and decode instances to create
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1} # Default to 1
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}   # Default to 1
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.2}
PREFILL_BLOCK_SIZE=${PREFILL_BLOCK_SIZE:-128}
DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE:-128}

# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

SMI_BIN=$(which nvidia-smi || which rocm-smi)

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

get_num_gpus() {
  if [[ "$SMI_BIN" == *"nvidia"* ]]; then
    echo "$($SMI_BIN --query-gpu=name --format=csv,noheader | wc -l)"
  else
    echo "$($SMI_BIN -l | grep GPU | wc -l)"
  fi
}

# Function to run tests for a specific model
run_tests_for_model() {
  local model_name=$1
  echo "================================"
  echo "Testing model: $model_name"
  echo "================================"

  # Get model-specific arguments
  local model_args=$(get_model_args "$model_name")

  # Arrays to store all hosts and ports
  PREFILL_HOSTS=()
  PREFILL_PORTS=()
  DECODE_HOSTS=()
  DECODE_PORTS=()

  # Start prefill instances
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    # Calculate GPU ID - we'll distribute across available GPUs
    GPU_ID=$((i % $(get_num_gpus)))
    NEXT_GPU=${GPU_ID}
    # If PREFILLER_TP_SIZE is more than 1
    for (( j=1; j < PREFILLER_TP_SIZE; j++ )); do
      NEXT_GPU=$(((GPU_ID + j) % $(get_num_gpus)))
      GPU_ID="${GPU_ID},${NEXT_GPU}"
    done

    # Calculate port number (base port + instance number)
    PORT=$((8100 + i))
    # Calculate side channel port. Avoid clash with with TP workers.
    SIDE_CHANNEL_PORT=$((5559 + i))

    echo "Starting prefill instance $i on GPU $GPU_ID, port $PORT"

    # Build the command with or without model-specific args
    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID \
    VLLM_KV_CACHE_LAYOUT='HND' \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
    vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --block-size ${PREFILL_BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --kv-transfer-config '$KV_CONFIG'"

    if [ -n "$model_args" ]; then
    FULL_CMD="$BASE_CMD $model_args"
    else
    FULL_CMD="$BASE_CMD"
    fi

    eval "$FULL_CMD &"

    # Store host and port for proxy configuration
    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=($PORT)
  done

  # Start decode instances
  for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    # Calculate GPU ID - we'll distribute across available GPUs, starting from after prefill GPUs
    GPU_ID=$(((i + NEXT_GPU + 1) % $(get_num_gpus)))
    # If DECODER_TP_SIZE is more than 1
    for (( j=1; j < DECODER_TP_SIZE; j++ )); do
      NEXT_GPU=$(((GPU_ID + j) % $(get_num_gpus)))
      GPU_ID="${GPU_ID},${NEXT_GPU}"
    done
    # Calculate port number (base port + instance number)
    PORT=$((8200 + i))
    # Calculate side channel port
    SIDE_CHANNEL_PORT=$((5659 + i * $DECODER_TP_SIZE))

    echo "Starting decode instance $i on GPU $GPU_ID, port $PORT"

    # Build the command with or without model-specific args
    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID \
    VLLM_KV_CACHE_LAYOUT=$DECODER_KV_LAYOUT \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
    vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --block-size ${DECODE_BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --kv-transfer-config '$KV_CONFIG'"
  
  # DP-EP attention mode
  if [[ -z "$DP_EP" ]]; then
    BASE_CMD="${BASE_CMD} --tensor-parallel-size $DECODER_TP_SIZE"
  else
    echo "DP-EP Attention enabled, deploying with dp=DECODER_TP_SIZE and tp=1"
    BASE_CMD="${BASE_CMD} --data-parallel-size $DECODER_TP_SIZE \
    --tensor-parallel-size 1 --enable-expert-parallel"
  fi

    if [ -n "$model_args" ]; then
    FULL_CMD="$BASE_CMD $model_args"
    else
    FULL_CMD="$BASE_CMD"
    fi

    eval "$FULL_CMD &"

    # Store host and port for proxy configuration
    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=($PORT)
  done

  # Wait for all instances to start
  for PORT in "${PREFILL_PORTS[@]}"; do
    echo "Waiting for prefill instance on port $PORT to start..."
    wait_for_server $PORT
  done

  for PORT in "${DECODE_PORTS[@]}"; do
    echo "Waiting for decode instance on port $PORT to start..."
    wait_for_server $PORT
  done

  # Build the command for the proxy server with all the hosts and ports
  PROXY_CMD="python3 ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192"

  # Add all prefill hosts and ports
  PROXY_CMD+=" --prefiller-hosts ${PREFILL_HOSTS[@]}"
  PROXY_CMD+=" --prefiller-ports ${PREFILL_PORTS[@]}"

  # Add all decode hosts and ports
  PROXY_CMD+=" --decoder-hosts ${DECODE_HOSTS[@]}"
  PROXY_CMD+=" --decoder-ports ${DECODE_PORTS[@]}"

  # Start the proxy server
  echo "Starting proxy server with command: $PROXY_CMD"
  $PROXY_CMD &

  # Wait for the proxy to start
  sleep 5

  # Run lm eval for this model
  echo "Running tests for $model_name"
  TEST_MODEL=$model_name python3 -m pytest -s -x ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_accuracy.py

  # Clean up before running next model
  cleanup_instances
  sleep 3
}

# Run tests for each model
for model in "${MODELS[@]}"; do
  run_tests_for_model "$model"
done

echo "All tests completed!"
