#!/bin/bash
set -e

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
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-0.6B"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
BLOCK_SIZE=${BLOCK_SIZE:-64}
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
KV_BUFFER_DEVICE=${KV_BUFFER_DEVICE:-"xpu"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

# Build mask-local device indices. If ZE_AFFINITY_MASK=6,7, then
# --device-ids 0,1 selects physical devices 6 and 7.
if [[ -n "${ZE_AFFINITY_MASK:-}" ]]; then
  IFS=',' read -r -a SHARED_DEVICES <<< "${ZE_AFFINITY_MASK}"
  NUM_AVAILABLE_DEVICES=${#SHARED_DEVICES[@]}
else
  NUM_AVAILABLE_DEVICES=$((PREFILLER_TP_SIZE + DECODER_TP_SIZE))
fi
AVAILABLE_DEVICES=()
for ((i=0; i<NUM_AVAILABLE_DEVICES; i++)); do
  AVAILABLE_DEVICES+=("$i")
done

compute_gpu_id() {
  local start=$1
  local tp_size=$2
  local gpu_id="${AVAILABLE_DEVICES[$start]}"
  for (( j=1; j<tp_size; j++ )); do
    gpu_id="${gpu_id},${AVAILABLE_DEVICES[$((start + j))]}"
  done
  echo "${gpu_id}"
}


# execution env
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)}"
EXP_ROOT="${GIT_ROOT}/tests/v1/kv_connector/nixl_integration"

OUTPUT_FILE=${OUTPUT_FILE:-"${EXP_ROOT}/.xpu_accuracy_test_outputs.txt"}

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

cleanup() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm serve" || true
  sleep 2
}

wait_for_server() {
  local host=$1
  local port=$2
  timeout 1200 bash -c "
    until curl -s ${host}:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

launch_baseline() {
  local BASELINE_GPU_ID
  BASELINE_GPU_ID=$(compute_gpu_id 0 1)

  BASELINE_BASE_CMD="
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ENABLE_V1_MULTIPROCESSING=1 vllm serve $MODEL_NAME \
      --host ${BASELINE_HOST} \
      --port ${BASELINE_PORT} \
      --max-model-len ${MAX_MODEL_LEN}\
      --seed 42 \
      -tp 1 \
      --device-ids ${BASELINE_GPU_ID} \
      --block-size ${BLOCK_SIZE} \
      --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
      --dtype float16 \
      --enforce-eager"
  echo "${BASELINE_BASE_CMD}"
  bash -c "${BASELINE_BASE_CMD}" &
  sleep 10
  wait_for_server "${BASELINE_HOST}" "${BASELINE_PORT}"
}

launch_pd() {
  local PREFILL_GPU_ID
  PREFILL_GPU_ID=$(compute_gpu_id 0 "${PREFILLER_TP_SIZE}")
  local DECODE_GPU_ID
  DECODE_GPU_ID=$(compute_gpu_id "${PREFILLER_TP_SIZE}" "${DECODER_TP_SIZE}")
  local kv_config="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"${KV_BUFFER_DEVICE}\"}"

  sleep 2

  echo "Starting prefill on GPU ${PREFILL_GPU_ID}, port ${PREFILL_PORT}"
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=200 \
  VLLM_NIXL_SIDE_CHANNEL_HOST="${PREFILL_HOST}" \
  VLLM_NIXL_SIDE_CHANNEL_PORT="${PREFILL_NIXL_SIDE_PORT}" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ENABLE_V1_MULTIPROCESSING=1 \
  vllm serve "$MODEL_NAME" \
    --host "${PREFILL_HOST}" \
    --port "${PREFILL_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --seed 42 \
    --block-size "${BLOCK_SIZE}" \
    --enforce-eager \
    --dtype float16 \
    -tp "${PREFILLER_TP_SIZE}" \
    --device-ids "${PREFILL_GPU_ID}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --kv-transfer-config "${kv_config}" &

  echo "Starting decode on GPU ${DECODE_GPU_ID}, port ${DECODE_PORT}"
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=200 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ENABLE_V1_MULTIPROCESSING=1 \
  vllm serve "$MODEL_NAME" \
    --host "${DECODE_HOST}" \
    --port "${DECODE_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --seed 42 \
    --block-size "${BLOCK_SIZE}" \
    --enforce-eager \
    -tp "${DECODER_TP_SIZE}" \
    --device-ids "${DECODE_GPU_ID}" \
    --dtype float16 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --kv-transfer-config "${kv_config}" &

  sleep 1
  wait_for_server "${PREFILL_HOST}" "${PREFILL_PORT}"
  sleep 1
  wait_for_server "${DECODE_HOST}" "${DECODE_PORT}"
  sleep 1
}

launch_pd_proxy(){
  PROXY_BASE_CMD="
  python3 ${EXP_ROOT}/toy_proxy_server.py \
  --prefiller-host ${PREFILL_HOST} --prefiller-port ${PREFILL_PORT} \
  --decoder-host ${DECODE_HOST} --decoder-port ${DECODE_PORT} \
  --host=${PROXY_HOST} --port ${PROXY_PORT}"
  echo "${PROXY_BASE_CMD}"
  bash -c "${PROXY_BASE_CMD}" &
  sleep 2
}

run_tests(){
  local service_url=$1
  local mode=$2
  python3 "${EXP_ROOT}/test_disagg_accuracy.py" --service_url="${service_url}" --model_name="${MODEL_NAME}" --mode="${mode}" --file_name="${OUTPUT_FILE}"
}


# run non-disagg. baseline & save outputs
launch_baseline
run_tests "http://${BASELINE_HOST}:${BASELINE_PORT}" "baseline"
cleanup
sleep 10


# run disagg. & do exact-match with the outputs from baseline
launch_pd
launch_pd_proxy
run_tests "http://${PROXY_HOST}:${PROXY_PORT}" "disagg"
echo "-----P/D success----"

rm "${OUTPUT_FILE}"
cleanup

exit 0
