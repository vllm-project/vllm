#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache}"
TIMEOUT="${TIMEOUT:-600}"

SERVER_PID=""

cleanup() {
    echo "Stopping server..."
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "Done."
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local deadline=$((SECONDS + TIMEOUT))
    echo "Waiting for server on port $PORT..."
    while (( SECONDS < deadline )); do
        if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
            echo "Server ready."
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Server did not start within ${TIMEOUT}s"
    return 1
}

rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

###############################################################################
# Start server with ec_both
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU" \
vllm serve "$MODEL" \
    --port "$PORT" \
    --enforce-eager \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_both",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    "$@" &

SERVER_PID=$!
wait_for_server

###############################################################################
# Benchmark -- dataset contains duplicate images, exercises cache hits
###############################################################################
echo "Running benchmark ($NUM_PROMPTS prompts)..."
vllm bench serve \
    --model "$MODEL" \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat \
    --seed 0 \
    --num-prompts "$NUM_PROMPTS" \
    --port "$PORT"

echo "Benchmark complete."
