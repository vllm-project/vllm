#!/bin/bash
set -euo pipefail

declare -a PIDS=()
declare -a PID_NAMES=()
declare -a PID_LOGS=()
SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

###############################################################################
# Configuration -- override via env before running
###############################################################################
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
# MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-${SCRIPT_DIR}/logs}"
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
NUM_WARMUPS="${NUM_WARMUPS:-0}"               # warmup prompts excluded from metrics

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
PYTHON="${PYTHON:-${GIT_ROOT}/.venv/bin/python}"

START_TIME=$(date +"%Y%m%d_%H%M%S")
ENC_LOG=$LOG_PATH/encoder_${START_TIME}.log
PD_LOG=$LOG_PATH/pd_${START_TIME}.log
PROXY_LOG=$LOG_PATH/proxy_${START_TIME}.log
BENCH_LOG=$LOG_PATH/bench_${START_TIME}.log
RESULT_JSON=$LOG_PATH/single_request_${START_TIME}.json

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

warn_if_same_device() {
    local left_name=$1
    local left_device=$2
    local right_name=$3
    local right_device=$4

    if [[ "$left_device" == "$right_device" ]]; then
        log "WARNING: ${left_name} and ${right_name} both use ${DEVICE_AFFINITY_ENV}=${left_device}."
        log "         This is useful for smoke tests, but the benchmark includes serial encoder and decode work on one device."
    fi
}

terminate_pid_tree() {
    local pid=$1
    local child

    while read -r child; do
        if [[ -n "$child" ]]; then
            terminate_pid_tree "$child"
        fi
    done < <(pgrep -P "$pid" 2>/dev/null || true)

    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
    fi
}

force_terminate_pid_tree() {
    local pid=$1
    local child

    while read -r child; do
        if [[ -n "$child" ]]; then
            force_terminate_pid_tree "$child"
        fi
    done < <(pgrep -P "$pid" 2>/dev/null || true)

    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
    fi
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
    local exit_code=$?
    log "Stopping everything..."
    trap - INT TERM USR1   # prevent re-entrancy

    # Kill all tracked PIDs
    for idx in "${!PIDS[@]}"; do
        pid=${PIDS[$idx]}
        name=${PID_NAMES[$idx]}
        log_file=${PID_LOGS[$idx]}
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping ${name}: pid=${pid}, log=${log_file}"
            terminate_pid_tree "$pid"
        fi
    done

    # Wait a moment for graceful shutdown
    sleep 2

    # Force kill any remaining processes
    for idx in "${!PIDS[@]}"; do
        pid=${PIDS[$idx]}
        name=${PID_NAMES[$idx]}
        log_file=${PID_LOGS[$idx]}
        if kill -0 "$pid" 2>/dev/null; then
            log "Force killing ${name}: pid=${pid}, log=${log_file}"
            force_terminate_pid_tree "$pid"
        fi
    done

    log "All processes stopped."
    exit "$exit_code"
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
warn_if_same_device "encoder" "$GPU_E" "prefill_decode" "$GPU_PD"

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
"${PYTHON}" "${EXAMPLE_ROOT}/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
    --log-request-timing \
    >"${PROXY_LOG}" 2>&1 &

track_pid "proxy" "$!" "$PROXY_LOG"

wait_for_server "$PROXY_PORT" "proxy"
log "All services are up."

###############################################################################
# Benchmark
###############################################################################
log "Running benchmark (stream): num_prompts=${NUM_PROMPTS}, num_warmups=${NUM_WARMUPS}, proxy_port=${PROXY_PORT}"

vllm bench serve \
    --model "$MODEL" \
    --dataset-name random-mm \
    --num-prompts "$NUM_PROMPTS" \
    --num-warmups "$NUM_WARMUPS" \
    --random-input-len 400 \
    --random-output-len 100 \
    --random-range-ratio 0.0 \
    --random-mm-base-items-per-request 1 \
    --random-mm-num-mm-items-range-ratio 0 \
    --random-mm-limit-mm-per-prompt '{"image":3,"video":0}' \
    --random-mm-bucket-config '{(560, 560, 1): 1.0}' \
    --ignore-eos \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --temperature 0 \
    --port "$PROXY_PORT" \
    2>&1 | tee "$BENCH_LOG"

log "Benchmark finished; result saved to ${BENCH_LOG}"

###############################################################################
# Single request with local image
###############################################################################
log "Running single request with local image (non-stream)"
curl -fsS http://127.0.0.1:"${PROXY_PORT}"/v1/chat/completions \
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
    }' | tee "$RESULT_JSON"

echo
log "Single request JSON saved to ${RESULT_JSON}"

# cleanup
log "Single request finished; cleanup starting"
cleanup
