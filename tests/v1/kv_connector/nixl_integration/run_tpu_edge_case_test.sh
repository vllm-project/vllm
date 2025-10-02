#!/bin/bash
set -xe

# Hosts / ports
PREFILL_HOST=${PREFILL_HOST:-"localhost"}
PREFILL_PORT=${PREFILL_PORT:-8100}
PREFILL_NIXL_SIDE_PORT=${PREFILL_NIXL_SIDE_PORT:-5577}
DECODE_HOST=${DECODE_HOST:-"localhost"}
DECODE_PORT=${DECODE_PORT:-8200}
PROXY_HOST=${PROXY_HOST:-"localhost"}
PROXY_PORT=${PROXY_PORT:-8192}
BASELINE_HOST=${BASELINE_HOST:-"localhost"}
BASELINE_PORT=${BASELINE_PORT:-9290}


# Model to run.
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-3B-Instruct"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
BLOCK_SIZE=${BLOCK_SIZE:-32}


# execution env
GIT_ROOT=$(git rev-parse --show-toplevel)
EXP_ROOT="${GIT_ROOT}/tests/v1/kv_connector/nixl_integration"
CONDA_PATH=${CONDA_PATH:-"/home/${USER}/anaconda3"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"nixl"}

OUTPUT_FILE=${OUTPUT_FILE:-"${EXP_ROOT}/.tpu_accuracy_test_outputs.txt"}

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

# Waits for vLLM server to start.
wait_for_server() {
  local host=$1
  local port=$2
  timeout 1200 bash -c "
    until curl -s ${host}:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9 || true
    # pkill -f python || true
    echo "Cleanup complete. Exiting."
}


launch_pd() {
  PREFILL_BASE_CMD="source ${CONDA_PATH}/bin/activate ${CONDA_ENV_NAME};
  UCX_TLS=tcp \
  VLLM_MULTIPROC_EXECUTE_MODEL_TIMEOUT_S=200 \
  VLLM_LOGGING_LEVEL=DEBUG \
  VLLM_USE_V1=1 \
  VLLM_NIXL_SIDE_CHANNEL_HOST=${PREFILL_HOST} \
  VLLM_NIXL_SIDE_CHANNEL_PORT=${PREFILL_NIXL_SIDE_PORT} \
  PJRT_DEVICE=TPU \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve $MODEL_NAME \
      --host ${PREFILL_HOST} \
      --port ${PREFILL_PORT} \
      --max-model-len ${MAX_MODEL_LEN}\
      --seed 42 \
      --block-size ${BLOCK_SIZE} \
      --enforce-eager \
      --gpu-memory-utilization 0.5 \
      --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"cpu\"}'"


  DECODE_BASE_CMD="source ${CONDA_PATH}/bin/activate ${CONDA_ENV_NAME};
  UCX_TLS=tcp \
  VLLM_MULTIPROC_EXECUTE_MODEL_TIMEOUT_S=200 \
  VLLM_LOGGING_LEVEL=DEBUG \
  VLLM_USE_V1=1 \
  PJRT_DEVICE=TPU \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve $MODEL_NAME \
      --host ${DECODE_HOST} \
      --port ${DECODE_PORT} \
      --max-model-len ${MAX_MODEL_LEN}\
      --seed 42 \
      --block-size ${BLOCK_SIZE} \
      --enforce-eager \
      --gpu-memory-utilization 0.5 \
      --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"cpu\"}'"

  echo ${PREFILL_BASE_CMD}
  echo ${DECODE_BASE_CMD}
  sleep 2

  # execute on hosts
  ssh -tt ${PREFILL_HOST} "${PREFILL_BASE_CMD}" &
  ssh -tt ${DECODE_HOST} "${DECODE_BASE_CMD}" &
  sleep 1
  wait_for_server ${PREFILL_HOST} ${PREFILL_PORT}
  sleep 1
  wait_for_server ${DECODE_HOST} ${DECODE_PORT}
  sleep 1
}

launch_pd_proxy(){
  PROXY_BASE_CMD="source ${CONDA_PATH}/bin/activate ${CONDA_ENV_NAME};
  python3 ${EXP_ROOT}/toy_proxy_server.py \
  --prefiller-host ${PREFILL_HOST} --prefiller-port ${PREFILL_PORT} \
  --decoder-host ${DECODE_HOST} --decoder-port ${DECODE_PORT} \
  --host=${PROXY_HOST} --port ${PROXY_PORT}"
  echo ${PROXY_BASE_CMD}
  ssh -tt ${PROXY_HOST} "${PROXY_BASE_CMD}" &
}


# run disagg. & do exact-match with the outputs from baseline
launch_pd
launch_pd_proxy
sleep 10

PREFILL_HOST=${PREFILL_HOST} \
PREFILL_PORT=${PREFILL_PORT} \
DECODE_HOST=${DECODE_HOST} \
DECODE_PORT=${DECODE_PORT} \
PROXY_HOST=${PROXY_HOST} \
PROXY_PORT=${PROXY_PORT} python -m pytest -s -v ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_edge_cases.py