#!/bin/bash
set -xe

# Parse command line arguments
ATTENTION_BACKEND=""  # Default to empty (use vllm default)

while [[ $# -gt 0 ]]; do
  case $1 in
    --attention-backend)
      ATTENTION_BACKEND="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--attention-backend <backend>]"
      exit 1
      ;;
  esac
done

if [[ -n "$ATTENTION_BACKEND" ]]; then
  echo "Using attention backend: $ATTENTION_BACKEND"
fi
if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
  echo "vLLM serve extra args: $VLLM_SERVE_EXTRA_ARGS"
fi

# Mooncake discovers peers via its bootstrap server, so no side-channel
# or layout env vars are needed.
KV_CONFIG_P='{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
KV_CONFIG_D='{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'

# Models to run
MODEL_NAME=${MODEL_NAME:-}
if [[ -n "$MODEL_NAME" ]]; then
  MODEL="$MODEL_NAME"
else
  MODEL="Qwen/Qwen3-0.6B"
fi

# Number of prefill and decode instances to create
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1} # Default to 1
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}   # Default to 1
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.2}
BLOCK_SIZE=${BLOCK_SIZE:-128}
# Comma-separated extra args for vllm serve (e.g. --max-model-len,2048)
VLLM_SERVE_EXTRA_ARGS=${VLLM_SERVE_EXTRA_ARGS:-}

# Resolve the repository root from the script location instead of `.git`.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)}"

SMI_BIN=$(which nvidia-smi || which rocm-smi || echo "")

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

get_num_gpus() {
  if [[ "$SMI_BIN" == *"nvidia"* ]]; then
    $SMI_BIN --query-gpu=name --format=csv,noheader | wc -l
  elif [[ "$SMI_BIN" == *"rocm"* ]]; then
    $SMI_BIN -l | grep -c GPU
  else
    # works for non-cuda platforms,
    # assuming at least 1 device and
    # let system to decide which card to use
    echo "1"
  fi
}

# Function to run tests for a specific model
run_tests_for_model() {
  local model_name=$1
  echo "================================"
  echo "Testing model: $model_name"
  echo "================================"

  # Proxy args: repeated --prefill URL BOOTSTRAP_PORT and --decode URL.
  PROXY_ARGS=()

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
    # Bootstrap server port for this prefiller. Avoid clash across instances.
    BOOTSTRAP_PORT=$((8998 + i))

    echo "Starting prefill instance $i on GPU $GPU_ID, port $PORT, bootstrap $BOOTSTRAP_PORT"

    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID \
    VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP_PORT \
    vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --kv-transfer-config '$KV_CONFIG_P'"
    if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
      IFS=',' read -r -a extra_args <<< "$VLLM_SERVE_EXTRA_ARGS"
      for arg in "${extra_args[@]}"; do
        BASE_CMD="${BASE_CMD} $arg"
      done
    fi

    # Add attention backend config if specified
    if [[ -n "$ATTENTION_BACKEND" ]]; then
      BASE_CMD="${BASE_CMD} --attention-backend=$ATTENTION_BACKEND"
    fi

    eval "$BASE_CMD &"

    PROXY_ARGS+=(--prefill "http://localhost:${PORT}" "$BOOTSTRAP_PORT")
    PREFILL_PORTS+=("$PORT")
  done

  # Start decode instances
  for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    # Calculate GPU ID - distribute starting from after prefill GPUs
    GPU_ID=$(((i + NEXT_GPU + 1) % $(get_num_gpus)))
    # If DECODER_TP_SIZE is more than 1
    for (( j=1; j < DECODER_TP_SIZE; j++ )); do
      NEXT_GPU=$(((GPU_ID + j) % $(get_num_gpus)))
      GPU_ID="${GPU_ID},${NEXT_GPU}"
    done
    # Calculate port number (base port + instance number)
    PORT=$((8200 + i))

    echo "Starting decode instance $i on GPU $GPU_ID, port $PORT"

    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID \
    vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --block-size ${BLOCK_SIZE} \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --kv-transfer-config '$KV_CONFIG_D'"
    if [[ -n "$VLLM_SERVE_EXTRA_ARGS" ]]; then
      IFS=',' read -r -a extra_args <<< "$VLLM_SERVE_EXTRA_ARGS"
      for arg in "${extra_args[@]}"; do
        BASE_CMD="${BASE_CMD} $arg"
      done
    fi

    # Add attention backend config if specified
    if [[ -n "$ATTENTION_BACKEND" ]]; then
      BASE_CMD="${BASE_CMD} --attention-backend=$ATTENTION_BACKEND"
    fi

    eval "$BASE_CMD &"

    PROXY_ARGS+=(--decode "http://localhost:${PORT}")
    DECODE_PORTS+=("$PORT")
  done

  # Wait for all instances to start
  for PORT in "${PREFILL_PORTS[@]}"; do
    echo "Waiting for prefill instance on port $PORT to start..."
    wait_for_server "$PORT"
  done

  for PORT in "${DECODE_PORTS[@]}"; do
    echo "Waiting for decode instance on port $PORT to start..."
    wait_for_server "$PORT"
  done

  # Start the proxy server
  echo "Starting proxy server on port 8192"
  python3 "${GIT_ROOT}/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py" \
    --port 8192 "${PROXY_ARGS[@]}" &

  # Wait for the proxy to start
  sleep 5

  # Run lm eval for this model
  echo "Running tests for $model_name"
  TEST_MODEL=$model_name python3 -m pytest -s -x \
    "${GIT_ROOT}/tests/v1/kv_connector/mooncake_integration/test_accuracy.py"

  # Clean up before running next model
  cleanup_instances
  sleep 3
}

run_tests_for_model "$MODEL"

echo "All tests completed!"
