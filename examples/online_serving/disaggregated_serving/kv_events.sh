#!/bin/bash
# This file demonstrates the KV cache event publishing
# We will launch a vllm instances configured to publish KV cache
# events and launch a simple subscriber to log those events.

set -xe

echo "🚧🚧 Warning: The usage of KV cache events is experimental and subject to change 🚧🚧"
sleep 1

MODEL_NAME=${HF_MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

vllm serve $MODEL_NAME \
    --port 8100 \
    --max-model-len 100 \
    --enforce-eager \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-events-config \
    '{"enable_kv_cache_events": true, "publisher": "zmq", "topic": "kv-events"}' &

wait_for_server 8100

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python3 "$SCRIPT_DIR/kv_events_subscriber.py" &
sleep 1

# serve two example requests
output1=$(curl -X POST -s http://localhost:8100/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "Explain quantum computing in simple terms a 5-year-old could understand.",
"max_tokens": 80,
"temperature": 0
}')

output2=$(curl -X POST -s http://localhost:8100/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "Explain quantum computing in simple terms a 50-year-old could understand.",
"max_tokens": 80,
"temperature": 0
}')

# Cleanup commands
pkill -9 -u "$USER" -f python
pkill -9 -u "$USER" -f vllm

sleep 1

echo "Cleaned up"

# Print the outputs of the curl requests
echo ""
echo "Output of first request: $output1"
echo "Output of second request: $output2"

echo "🎉🎉 Successfully finished 2 test requests! 🎉🎉"
echo ""
