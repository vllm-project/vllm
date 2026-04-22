#!/bin/bash
set -euo pipefail

declare -a PIDS=()
declare -a PID_NAMES=()
declare -a PID_LOGS=()
SCRIPT_NAME=$(basename "$0")

###############################################################################
# Configuration -- override via env before running
###############################################################################
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
# MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./logs}"
mkdir -p "$LOG_PATH"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_DECODE_PORT="${PREFILL_DECODE_PORT:-19535}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-0}"
GPU_PD="${GPU_PD:-0}"

# Device platform and affinity env name.
# DEVICE_PLATFORM supports: cuda, xpu
DEVICE_PLATFORM="${DEVICE_PLATFORM:-cuda}"
if [[ -z "${DEVICE_AFFINITY_ENV:-}" ]]; then
    if [[ "${DEVICE_PLATFORM,,}" == "xpu" ]]; then
        DEVICE_AFFINITY_ENV="ZE_AFFINITY_MASK"
    else
        DEVICE_AFFINITY_ENV="CUDA_VISIBLE_DEVICES"
    fi
fi
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-12000}"   # wait_for_server timeout
NUM_PROMPTS="${NUM_PROMPTS:-100}"             # number of prompts to send in benchmark

# Serve args
GPU_MEMORY_UTILIZATION_E="${GPU_MEMORY_UTILIZATION_E:-0.01}"
GPU_MEMORY_UTILIZATION_PD="${GPU_MEMORY_UTILIZATION_PD:-0.7}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

###############################################################################
# Helpers
###############################################################################
# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

START_TIME=$(date +"%Y%m%d_%H%M%S")
ENC_LOG=$LOG_PATH/encoder_${START_TIME}.log
PD_LOG=$LOG_PATH/pd_${START_TIME}.log
PROXY_LOG=$LOG_PATH/proxy_${START_TIME}.log

log() {
    printf '[%s] [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$SCRIPT_NAME" "$*"
}

track_pid() {
    local name=$1
    local pid=$2
    local log_file=$3
    PIDS+=("$pid")
    PID_NAMES+=("$name")
    PID_LOGS+=("$log_file")
    log "Started ${name}: pid=${pid}, log=${log_file}"
}

wait_for_server() {
    local port=$1
    local name=${2:-service}
    log "Waiting for ${name} on port ${port} (timeout=${TIMEOUT_SECONDS}s)"
    if timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:$port/v1/chat/completions > /dev/null; do
            sleep 1
        done"; then
        log "${name} is ready on port ${port}"
        return 0
    fi
    log "Timed out waiting for ${name} on port ${port}"
    return 1
}

# Cleanup function
cleanup() {
    log "Stopping everything..."
    trap - INT TERM USR1   # prevent re-entrancy

    # Kill all tracked PIDs
    for idx in "${!PIDS[@]}"; do
        pid=${PIDS[$idx]}
        name=${PID_NAMES[$idx]}
        log_file=${PID_LOGS[$idx]}
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping ${name}: pid=${pid}, log=${log_file}"
            kill "$pid" 2>/dev/null
        fi
    done

    # Wait a moment for graceful shutdown
    wait "${PIDS[@]}" 2>/dev/null || true

    # Force kill any remaining processes
    for idx in "${!PIDS[@]}"; do
        pid=${PIDS[$idx]}
        name=${PID_NAMES[$idx]}
        log_file=${PID_LOGS[$idx]}
        if kill -0 "$pid" 2>/dev/null; then
            log "Force killing ${name}: pid=${pid}, log=${log_file}"
            kill -9 "$pid" 2>/dev/null
        fi
    done

    # Kill the entire process group as backup
    kill -- -$$ 2>/dev/null

    log "All processes stopped."
    exit 0
}

trap cleanup INT
trap cleanup USR1
trap cleanup TERM

log "Configuration:"
log "  MODEL=${MODEL}"
log "  LOG_PATH=${LOG_PATH}"
log "  GIT_ROOT=${GIT_ROOT}"
log "  Ports: encoder=${ENCODE_PORT}, prefill_decode=${PREFILL_DECODE_PORT}, proxy=${PROXY_PORT}"
log "  Devices: ${DEVICE_AFFINITY_ENV} encoder=${GPU_E}, prefill_decode=${GPU_PD}"
log "  Serve args: max_num_seqs=${MAX_NUM_SEQS}, max_model_len=${MAX_MODEL_LEN}"
log "  Log files: encoder=${ENC_LOG}, prefill_decode=${PD_LOG}, proxy=${PROXY_LOG}"

###############################################################################
# Encoder worker
###############################################################################
log "Launching encoder worker on port ${ENCODE_PORT}"
env "$DEVICE_AFFINITY_ENV=$GPU_E" \
VLLM_EC_MOONCAKE_BOOTSTRAP_PORT=9198 \
vllm serve "$MODEL" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION_E" \
    --port "$ENCODE_PORT" \
    --mm-encoder-only \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --allowed-local-media-path "${GIT_ROOT}"/tests/v1/ec_connector/integration \
    --ec-transfer-config "{
        \"ec_connector\": \"MooncakeECConnector\",
        \"ec_role\": \"ec_producer\",
        \"ec_connector_extra_config\": {
            \"protocol\": \"rdma\",
            \"device_name\": \"mlx5_2,mlx5_4\",
            \"transfer_buffer_size\": \"1073741824\"
        }
    }" \
    >"${ENC_LOG}" 2>&1 &

track_pid "encoder" "$!" "$ENC_LOG"

###############################################################################
# Prefill+Decode worker
###############################################################################
log "Launching prefill+decode worker on port ${PREFILL_DECODE_PORT}"
env "$DEVICE_AFFINITY_ENV=$GPU_PD" vllm serve "$MODEL" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION_PD" \
    --port "$PREFILL_DECODE_PORT" \
    --enable-request-id-headers \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --allowed-local-media-path "${GIT_ROOT}"/tests/v1/ec_connector/integration \
    --ec-transfer-config "{
        \"ec_connector\": \"MooncakeECConnector\",
        \"ec_role\": \"ec_consumer\",
        \"ec_connector_extra_config\": {
            \"protocol\": \"rdma\",
            \"device_name\": \"mlx5_2,mlx5_4\",
            \"transfer_buffer_size\": \"1073741824\"
        }
    }" \
    >"${PD_LOG}" 2>&1 &

track_pid "prefill_decode" "$!" "$PD_LOG"

# Wait for workers
wait_for_server "$ENCODE_PORT" "encoder"
wait_for_server "$PREFILL_DECODE_PORT" "prefill_decode"

###############################################################################
# Proxy
###############################################################################
log "Launching EPD proxy on port ${PROXY_PORT}"
python ../disagg_epd_proxy.py \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
    >"${PROXY_LOG}" 2>&1 &

track_pid "proxy" "$!" "$PROXY_LOG"

wait_for_server "$PROXY_PORT" "proxy"
log "All services are up."

###############################################################################
# Benchmark
###############################################################################
# log "Running benchmark (stream): num_prompts=${NUM_PROMPTS}, proxy_port=${PROXY_PORT}"

# vllm bench serve \
#     --model "$MODEL" \
#     --dataset-name random-mm \
#     --num-prompts "$NUM_PROMPTS" \
#     --random-input-len 400 \
#     --random-output-len 100 \
#     --random-range-ratio 0.0 \
#     --random-mm-base-items-per-request 1 \
#     --random-mm-num-mm-items-range-ratio 0 \
#     --random-mm-limit-mm-per-prompt '{"image":3,"video":0}' \
#     --random-mm-bucket-config '{(560, 560, 1): 1.0}' \
#     --ignore-eos \
#     --backend openai-chat \
#     --endpoint /v1/chat/completions \
#     --temperature 0 \
#     --port "$PROXY_PORT"

# log "Benchmark finished."

###############################################################################
# Single request with local image
###############################################################################
log "Running single request with local image (non-stream)"
curl http://127.0.0.1:"${PROXY_PORT}"/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "'"${MODEL}"'",
    "temperature": 0,
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file://'"${GIT_ROOT}"'/tests/v1/ec_connector/integration/hato.jpg"}},
        {"type": "text", "text": "What is in this image?"}
    ]}
    ]
    }'

# cleanup
log "Single request finished; cleanup starting"
cleanup
