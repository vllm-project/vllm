#!/bin/bash
set -euo pipefail

# =========================
# Configuration Parameters
# =========================

PREFILL_PORT=8100
DECODE_PORT=8200

MODEL="facebook/opt-125m"

PREFILL_LOG="/tmp/prefill.log"
DECODE_LOG="/tmp/decode.log"

START_TIMEOUT=120
WAIT_INTERVAL=1

PROMPT="San Francisco is a"

PORTS=($PREFILL_PORT $DECODE_PORT)
LOGS=($PREFILL_LOG $DECODE_LOG)
STAGES=("prefill" "decode")
GPUS=(0 1)

# =========================
# Function Definitions
# =========================

# Function to check if a command exists
command_exists() {
    command -v "$1" &>/dev/null
}

# Function to log messages with timestamps
log() {
    local message="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $message"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -i :"$port" -t &>/dev/null; then
        log "Error: Port $port is in use."
        exit 1
    fi
}

# Function to start a vllm server
start_vllm_server() {
    local gpu_id=$1
    local stage=$2
    local port=$3
    local log_file=$4
    CUDA_VISIBLE_DEVICES="$gpu_id" PD_SEPARATE_STAGE="$stage" \
        vllm serve "$MODEL" --enforce-eager --port "$port" --dtype=float16 > "$log_file" 2>&1 &
}

# Function to wait for a vllm endpoint to become ready
wait_for_endpoint() {
    local port=$1
    local elapsed=0
    while true; do
        if curl --output /dev/null --silent --fail "http://localhost:$port/v1/models"; then
            log "vllm on port $port is ready!"
            break
        fi
        if [ $elapsed -ge $START_TIMEOUT ]; then
            log "Error: vllm on port $port is not ready after $START_TIMEOUT seconds."
            log "Check log file for more details."
            exit 1
        fi
        sleep $WAIT_INTERVAL
        elapsed=$((elapsed + WAIT_INTERVAL))
    done
}

# Function to clean up background processes on exit
cleanup() {
    log "Cleaning up background processes..."
    pkill -f "vllm serve" || true
}

trap cleanup EXIT

# =========================
# Main Script Execution
# =========================

# Check for required commands
for cmd in vllm curl lsof nvidia-smi; do
    if ! command_exists "$cmd"; then
        log "Error: Required command '$cmd' is not installed."
        exit 1
    fi
done

# Check if INFINITY is supported
OUTPUT=$(python3 -c "from infinity import check_infinity_supported; \
result = check_infinity_supported(); \
print(result)" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Infinity is not supported: $OUTPUT"
    exit $EXIT_CODE
fi

# Check if there are at least 2 GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    log "Error: Less than 2 GPUs detected."
    exit 1
fi

# Check if the ports are not in use
for port in "${PORTS[@]}"; do
    check_port "$port"
done

# Start vllm servers
for i in "${!PORTS[@]}"; do
    log "Starting vllm server (${STAGES[$i]}) on port ${PORTS[$i]}..."
    start_vllm_server "${GPUS[$i]}" "${STAGES[$i]}" "${PORTS[$i]}" "${LOGS[$i]}"
done

# Wait for vllm endpoints to become ready
for port in "${PORTS[@]}"; do
    wait_for_endpoint "$port"
done

log "All vllm endpoints are ready!"

# Prepare JSON data
DATA=$(jq -n \
    --arg model "$MODEL" \
    --arg prompt "$PROMPT" \
    '{model: $model, prompt: $prompt}')

log "Sending request to prefill and decode..."

# Send requests
prefill_output=$(curl -s "http://localhost:${PREFILL_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$DATA")

decode_output=$(curl -s "http://localhost:${DECODE_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$DATA")

# Display outputs
printf "Prefill output:\n%s\n\nDecode output:\n%s\n" "$prefill_output" "$decode_output"
