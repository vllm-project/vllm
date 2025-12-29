#!/bin/bash
# Smoke test for vLLM server
#
# This script performs a basic smoke test of the vLLM server by:
# 1. Starting a vLLM server with a small model in dummy mode
# 2. Waiting for the server to become ready (health check)
# 3. Testing the /health endpoint
# 4. Testing the /v1/completions endpoint with a simple prompt
#
# Usage:
#   smoke-test-server.sh [MODEL] [PORT]
#
# Arguments:
#   MODEL    - Model name to use (default: Qwen/Qwen3-0.6B)
#   PORT     - Port to run the server on (default: 8000)
#   TIMEOUT  - Server startup timeout in seconds (default: 30)
#
# Example:
#   smoke-test-server.sh
#   smoke-test-server.sh meta-llama/Llama-2-7b-hf 8001 60
#
# Exit codes:
#   0 - Success
#   1 - Server failed to start or endpoints failed
#
set -e

MODEL="${1:-Qwen/Qwen3-0.6B}"
PORT="${2:-8000}"
TIMEOUT="${3:-30}"

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ]; then
    echo "Cleaning up server process (PID: $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
# trap for cleanup on exit, interrupt, or termination
trap cleanup EXIT INT TERM

# Start server
vllm serve "$MODEL" \
  --max-model-len 2048 \
  --load-format dummy \
  --hf-overrides '{"num_hidden_layers": 2}' \
  --enforce-eager \
  --port "$PORT" &
SERVER_PID=$!

# Wait for server to start
WAIT_INTERVAL=2
MAX_ATTEMPTS=$((TIMEOUT/WAIT_INTERVAL))
for i in $(seq 1 $MAX_ATTEMPTS); do
  if curl -s "http://localhost:$PORT/health" > /dev/null; then
    echo "Server started successfully"
    break
  fi
  if [ "$i" -eq $MAX_ATTEMPTS ]; then
    echo "Server failed to start"
    exit 1
  fi
  sleep $WAIT_INTERVAL
done

# Test health endpoint
curl -f "http://localhost:$PORT/health"

# Test completion
curl -f "http://localhost:$PORT/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"Hello\", \"max_tokens\": 5}"

echo "Smoke test passed"
