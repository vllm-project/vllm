#!/bin/bash

set -xe

# Model to run.
MODEL_NAME=Qwen/Qwen3-0.6B

# Number of prefill and decode instances to create
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1} # Default to 1
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-2}   # Default to 2

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

# Arrays to store all hosts and ports
PREFILL_HOSTS=()
PREFILL_PORTS=()
DECODE_HOSTS=()
DECODE_PORTS=()

# Start prefill instances
for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
  # Calculate GPU ID - we'll distribute across available GPUs
  GPU_ID=$((i % $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)))
  # Calculate port number (base port + instance number)
  PORT=$((8100 + i))
  # Calculate side channel port
  SIDE_CHANNEL_PORT=$((5559 + i))

  echo "Starting prefill instance $i on GPU $GPU_ID, port $PORT"

  CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $MODEL_NAME \
    --port $PORT \
    --enforce-eager \
    --disable-log-requests \
    --gpu-memory-utilization 0.2 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

  # Store host and port for proxy configuration
  PREFILL_HOSTS+=("localhost")
  PREFILL_PORTS+=($PORT)
done

# Start decode instances
for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
  # Calculate GPU ID - we'll distribute across available GPUs, starting from after prefill GPUs
  GPU_ID=$(((i + NUM_PREFILL_INSTANCES) % $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)))
  # Calculate port number (base port + instance number)
  PORT=$((8200 + i))
  # Calculate side channel port
  SIDE_CHANNEL_PORT=$((5659 + i))

  echo "Starting decode instance $i on GPU $GPU_ID, port $PORT"

  CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $MODEL_NAME \
    --port $PORT \
    --enforce-eager \
    --disable-log-requests \
    --gpu-memory-utilization 0.2 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

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
PROXY_CMD="python ${GIT_ROOT}/tests/v1/kv_connector/toy_proxy_server.py --port 8192"

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

# Run lm eval.
python -m pytest -s -x ${GIT_ROOT}/tests/v1/kv_connector/test_accuracy.py
