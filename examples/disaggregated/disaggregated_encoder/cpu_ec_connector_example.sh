#!/bin/bash
# Local two-process demo for ECCPUConnector.
#
# Starts a producer (encoder-only) on $GPU_E and a consumer on $GPU_PD on the
# same host, then issues a single end-to-end request:
#   1. POST image to producer with max_tokens=1; harvest ec_transfer_params
#      from the response body.
#   2. POST the same prompt to the consumer with ec_transfer_params injected
#      into sampling_params.extra_args. The consumer pulls the encoding from
#      the producer over NIXL (intra-node shm by default) and decodes.
#
# Override any value by setting the env var before invoking, e.g.
#   GPU_E=2 GPU_PD=3 NUM_EC_BLOCKS=512 bash cpu_ec_connector_example.sh
set -euo pipefail

###############################################################################
# Configuration
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./logs}"
mkdir -p "$LOG_PATH"

PRODUCER_PORT="${PRODUCER_PORT:-19534}"
CONSUMER_PORT="${CONSUMER_PORT:-19535}"
PRODUCER_SIDE_CHANNEL_PORT="${PRODUCER_SIDE_CHANNEL_PORT:-5601}"
CONSUMER_SIDE_CHANNEL_PORT="${CONSUMER_SIDE_CHANNEL_PORT:-5602}"

GPU_E="${GPU_E:-0}"
GPU_PD="${GPU_PD:-1}"

DEVICE_PLATFORM="${DEVICE_PLATFORM:-cuda}"
if [[ -z "${DEVICE_AFFINITY_ENV:-}" ]]; then
    if [[ "${DEVICE_PLATFORM,,}" == "xpu" ]]; then
        DEVICE_AFFINITY_ENV="ZE_AFFINITY_MASK"
    else
        DEVICE_AFFINITY_ENV="CUDA_VISIBLE_DEVICES"
    fi
fi

NUM_EC_BLOCKS="${NUM_EC_BLOCKS:-80000}"

GPU_MEMORY_UTILIZATION_E="${GPU_MEMORY_UTILIZATION_E:-0.01}"
GPU_MEMORY_UTILIZATION_PD="${GPU_MEMORY_UTILIZATION_PD:-0.7}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"

###############################################################################
# Helpers
###############################################################################
GIT_ROOT=$(git rev-parse --show-toplevel)
IMAGE_PATH="${IMAGE_PATH:-${GIT_ROOT}/tests/v1/ec_connector/integration/hato.jpg}"

START_TIME=$(date +"%Y%m%d_%H%M%S")
PRODUCER_LOG="${LOG_PATH}/producer_${START_TIME}.log"
CONSUMER_LOG="${LOG_PATH}/consumer_${START_TIME}.log"

declare -a PIDS=()
declare -a TMPFILES=()

wait_for_server() {
    local port=$1
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:$port/v1/chat/completions > /dev/null; do
            sleep 1
        done"
}

cleanup() {
    trap - INT TERM
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    for f in "${TMPFILES[@]}"; do
        rm -f "$f"
    done
}
trap cleanup INT TERM EXIT

###############################################################################
# Producer (encoder-only)
###############################################################################
echo "[setup] starting producer on GPU $GPU_E (port $PRODUCER_PORT)..."
env "$DEVICE_AFFINITY_ENV=$GPU_E" \
    VLLM_EC_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_EC_SIDE_CHANNEL_PORT="$PRODUCER_SIDE_CHANNEL_PORT" \
    vllm serve "$MODEL" \
        --port "$PRODUCER_PORT" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION_E" \
        --enforce-eager \
        --no-enable-prefix-caching \
        --mm-encoder-only \
        --max-num-batched-tokens 114688 \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --allowed-local-media-path "$(dirname "$IMAGE_PATH")" \
        --ec-transfer-config '{
            "ec_connector": "ECCPUConnector",
            "ec_role": "ec_producer",
            "engine_id": "ec-producer-0",
            "ec_connector_extra_config": {
                "num_ec_blocks": '"$NUM_EC_BLOCKS"'
            }
        }' \
        > "$PRODUCER_LOG" 2>&1 &
PIDS+=($!)

###############################################################################
# Consumer
###############################################################################
echo "[setup] starting consumer on GPU $GPU_PD (port $CONSUMER_PORT)..."
env "$DEVICE_AFFINITY_ENV=$GPU_PD" \
    VLLM_EC_SIDE_CHANNEL_HOST=127.0.0.1 \
    VLLM_EC_SIDE_CHANNEL_PORT="$CONSUMER_SIDE_CHANNEL_PORT" \
    vllm serve "$MODEL" \
        --port "$CONSUMER_PORT" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION_PD" \
        --enforce-eager \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --allowed-local-media-path "$(dirname "$IMAGE_PATH")" \
        --ec-transfer-config '{
            "ec_connector": "ECCPUConnector",
            "ec_role": "ec_consumer",
            "ec_connector_extra_config": {
                "num_ec_blocks": '"$NUM_EC_BLOCKS"'
            }
        }' \
        > "$CONSUMER_LOG" 2>&1 &
PIDS+=($!)

echo "[setup] waiting for both servers..."
wait_for_server "$PRODUCER_PORT"
wait_for_server "$CONSUMER_PORT"
echo "[setup] producer + consumer up. logs: $PRODUCER_LOG  $CONSUMER_LOG"

###############################################################################
# Step 1: encode on producer (max_tokens=1)
#
# Image is inlined as base64 — producer never fetches HTTP URLs in EC
# disaggregation. The response body carries `ec_transfer_params`.
###############################################################################
# jq --arg is passed via argv, which overflows for large images. Stage the
# base64 payload in a temp file and read it with --rawfile instead.
B64_FILE=$(mktemp); TMPFILES+=("$B64_FILE")
base64 -w0 "$IMAGE_PATH" > "$B64_FILE"

PRODUCER_REQ_FILE=$(mktemp); TMPFILES+=("$PRODUCER_REQ_FILE")
jq -n \
    --arg model "$MODEL" \
    --rawfile b64 "$B64_FILE" \
    '{
        model: $model,
        max_tokens: 1,
        messages: [{
            role: "user",
            content: [
                {type: "image_url", image_url: {url: ("data:image/jpeg;base64," + $b64)}},
                {type: "text", text: "Describe this image in one sentence."}
            ]
        }]
    }' > "$PRODUCER_REQ_FILE"

echo "[step 1] POST encode to producer..."
PRODUCER_RES=$(curl -sS "http://127.0.0.1:${PRODUCER_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary "@$PRODUCER_REQ_FILE")

EC_TRANSFER_PARAMS=$(echo "$PRODUCER_RES" | jq -c '.ec_transfer_params')
if [[ "$EC_TRANSFER_PARAMS" == "null" || -z "$EC_TRANSFER_PARAMS" ]]; then
    echo "ERROR: producer response did not include ec_transfer_params"
    echo "$PRODUCER_RES" | jq .
    exit 1
fi
echo "[step 1] got ec_transfer_params: $EC_TRANSFER_PARAMS"

###############################################################################
# Step 2: decode on consumer with ec_transfer_params injected
#
# Consumer pulls bytes from producer over NIXL using the params, skipping
# its own vision encoder.
###############################################################################
CONSUMER_REQ_FILE=$(mktemp); TMPFILES+=("$CONSUMER_REQ_FILE")
jq -n \
    --arg model "$MODEL" \
    --rawfile b64 "$B64_FILE" \
    --argjson params "$EC_TRANSFER_PARAMS" \
    '{
        model: $model,
        max_tokens: 64,
        messages: [{
            role: "user",
            content: [
                {type: "image_url", image_url: {url: ("data:image/jpeg;base64," + $b64)}},
                {type: "text", text: "Describe this image in one sentence."}
            ]
        }],
        extra_body: {ec_transfer_params: $params}
    }' > "$CONSUMER_REQ_FILE"

echo "[step 2] POST decode to consumer..."
curl -sS "http://127.0.0.1:${CONSUMER_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary "@$CONSUMER_REQ_FILE" | jq .

echo "[done]"
