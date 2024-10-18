#!/bin/bash
set -euo pipefail

# =========================
# Configuration Parameters
# =========================

# Replace these with the actual IP addresses of your hosts
PREFILL_HOST="10.192.18.145"
DECODE_HOST="10.192.24.218"

INFINITY_HOST=10.192.18.145

PORT=8000

MODEL="facebook/opt-125m"

PREFILL_LOG="/tmp/prefill.log"
DECODE_LOG="/tmp/decode.log"

START_TIMEOUT=120
WAIT_INTERVAL=1

PROMPT="San Francisco is a"

STAGES=("prefill" "decode")
HOSTS=("$PREFILL_HOST" "$DECODE_HOST")
GPUS=(0 0)
LOGS=("$PREFILL_LOG" "$DECODE_LOG")

# Conda environments for each host
PREFILL_CONDA_ENV="qian2"
DECODE_CONDA_ENV="qian"
CONDA_ENVS=("$PREFILL_CONDA_ENV" "$DECODE_CONDA_ENV")


# =========================
# Function Definitions
# =========================

# Function to check if a host is the local machine
is_local_host() {
    local host_ip="$1"
    local local_ips
    local_ips=$(hostname -I)
    if [[ "$host_ip" == "127.0.0.1" || "$host_ip" == "localhost" ]]; then
        return 0
    fi
    for ip in $local_ips; do
        if [[ "$host_ip" == "$ip" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to check if a command exists on a host
command_exists_on_host() {
    local host="$1"
    local conda_env="$2"
    local cmd="$3"
    if is_local_host "$host"; then
        source ~/.bashrc
        conda activate "$conda_env"
        command -v "$cmd" &>/dev/null
    else
        ssh "$host" "bash -c 'source ~/.bashrc; conda activate $conda_env; command -v $cmd &>/dev/null'"
    fi
}

# Function to log messages with timestamps
log() {
    local message="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $message"
}

# Function to start a vllm server on a host
start_vllm_server_on_host() {
    local host="$1"
    local conda_env="$2"
    local gpu_id="$3"
    local stage="$4"
    local port="$5"
    local log_file="$6"
    if is_local_host "$host"; then
        source ~/.bashrc
        conda activate "$conda_env"
        CUDA_VISIBLE_DEVICES="$gpu_id" PD_SEPARATE_STAGE="$stage" INFINITE_STORE_SERVER=\"$INFINITY_HOST\" \
            vllm serve "$MODEL" --enforce-eager --port "$port" --dtype=float16 > "$log_file" 2>&1 &
    else
        ssh "$host" "bash -c 'source ~/.bashrc; conda activate $conda_env; \
         CUDA_VISIBLE_DEVICES=\"$gpu_id\" PD_SEPARATE_STAGE=\"$stage\" INFINITE_STORE_SERVER=\"$INFINITY_HOST\" \
            vllm serve \"$MODEL\" --enforce-eager --port \"$port\" --dtype=float16 > \"$log_file\" 2>&1 &'"
    fi
}

# Function to wait for a vllm endpoint to become ready on a host
wait_for_endpoint() {
    local host="$1"
    local port="$2"
    local elapsed=0
    while true; do
        if curl --output /dev/null --silent --fail "http://$host:$port/v1/models"; then
            log "vllm on $host:$port is ready!"
            break
        fi
        if [ $elapsed -ge $START_TIMEOUT ]; then
            log "Error: vllm on $host:$port is not ready after $START_TIMEOUT seconds."
            log "Check log file on the host for more details."
            exit 1
        fi
        sleep $WAIT_INTERVAL
        elapsed=$((elapsed + WAIT_INTERVAL))
    done
}

# Function to clean up background processes on hosts
cleanup() {
    log "Cleaning up background processes..."
    for i in "${!HOSTS[@]}"; do
        host="${HOSTS[$i]}"
        conda_env="${CONDA_ENVS[$i]}"
        if is_local_host "$host"; then
            pkill -f 'vllm serve' || true
        else
            ssh "$host" "pkill -f 'vllm serve' || true"
        fi
    done
}

trap cleanup EXIT

# =========================
# Main Script Execution
# =========================

echo aaaaa
# Check for required commands on hosts
for i in "${!HOSTS[@]}"; do
    host="${HOSTS[$i]}"
    conda_env="${CONDA_ENVS[$i]}"
    for cmd in vllm curl nvidia-smi; do
        if ! command_exists_on_host "$host" "$conda_env" "$cmd"; then
            log "Error: Required command '$cmd' is not installed on host $host in conda environment '$conda_env'."
            exit 1
        fi
    done
done
echo aaaaa1
# Check if Infinity is supported on hosts
for i in "${!HOSTS[@]}"; do
    host="${HOSTS[$i]}"
    conda_env="${CONDA_ENVS[$i]}"
    if is_local_host "$host"; then
        source ~/.bashrc
        conda activate "$conda_env"
        OUTPUT=$(python3 -c 'from infinistore import check_supported; result = check_supported(); print(result)' 2>&1)
        EXIT_CODE=$?
    else
        OUTPUT=$(ssh "$host" "bash -c 'source ~/.bashrc; conda activate $conda_env; python3 -c \"from infinistore import check_supported; result = check_supported(); print(result)\"' 2>&1")

        
        EXIT_CODE=$?
    fi

    echo $host: $OUTPUT $EXIT_CODE
    if [ $EXIT_CODE -ne 0 ]; then
        log "Error: Infinity is not supported on host $host: $OUTPUT"
        exit $EXIT_CODE
    fi
done

echo aaaaa2
# Check if there is at least 1 GPU on each host
for i in "${!HOSTS[@]}"; do
    host="${HOSTS[$i]}"
    conda_env="${CONDA_ENVS[$i]}"
    if is_local_host "$host"; then
        source ~/.bashrc
        conda activate "$conda_env"
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        GPU_COUNT=$(ssh "$host" "bash -c 'source ~/.bashrc; conda activate $conda_env; nvidia-smi --query-gpu=name --format=csv,noheader | wc -l'")
    fi
    if [ "$GPU_COUNT" -lt 1 ]; then
        log "Error: No GPUs detected on host $host."
        exit 1
    fi
done

# Start vllm servers on hosts
for i in "${!HOSTS[@]}"; do
    host="${HOSTS[$i]}"
    conda_env="${CONDA_ENVS[$i]}"
    log "Starting vllm server (${STAGES[$i]}) on ${HOSTS[$i]}:${PORT}..."
    start_vllm_server_on_host "$host" "$conda_env" "${GPUS[$i]}" "${STAGES[$i]}" "$PORT" "${LOGS[$i]}"
done

# Wait for vllm endpoints to become ready on hosts
for i in "${!HOSTS[@]}"; do
    wait_for_endpoint "${HOSTS[$i]}" "$PORT"
done

log "All vllm endpoints are ready!"

# Prepare JSON data
DATA=$(jq -n \
    --arg model "$MODEL" \
    --arg prompt "$PROMPT" \
    '{model: $model, prompt: $prompt}')

log "Sending request to prefill and decode..."

# Send requests to hosts
prefill_output=$(curl -s "http://${PREFILL_HOST}:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$DATA")

decode_output=$(curl -s "http://${DECODE_HOST}:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$DATA")

# Display outputs
printf "Prefill output:\n%s\n\nDecode output:\n%s\n" "$prefill_output" "$decode_output"