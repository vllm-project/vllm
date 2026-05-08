#!/usr/bin/env bash
set -euo pipefail

# Run accuracy test against a regular (no KV connector) vLLM serve instance.
# Mirrors the "regular" task from llm-scripts/llmd/Justfile.

# Config from Justfile
# MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8}"
MODEL="${MODEL:-openai/gpt-oss-20b}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-1}"
MEMORY_UTIL="${MEMORY_UTIL:-0.85}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
EAGER="${EAGER:-}"
BACKEND="${BACKEND:-AUTO}"
PORT="${PORT:-8192}"

GIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cleanup() {
  echo "Stopping vLLM server..."
  pkill -f "vllm serve" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_server() {
  echo "Waiting for vLLM server on port $PORT..."
  timeout 600 bash -c "
    until curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
      sleep 2
    done
  " || { echo "Server failed to start"; exit 1; }
  echo "Server ready."
}

# VLLM_WORKER_MULTIPROC_METHOD=spawn \
# chg run -g"$PREFILL_TP_SIZE" -- vllm serve "$MODEL" \
echo "Starting vLLM serve (regular, no KV connector)..."
  VLLM_LOGGING_LEVEL="INFO" \
  chg run -g"$PREFILL_TP_SIZE" -- vllm serve "$MODEL" \
    --port "$PORT" \
    $EAGER \
    --tensor-parallel-size "$PREFILL_TP_SIZE" \
    --enforce-eager \
    --gpu-memory-utilization "$MEMORY_UTIL" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --attention-backend "$BACKEND" \
    --block-size "$BLOCK_SIZE" \
  &

wait_for_server

echo "Running accuracy test..."
TEST_MODEL="$MODEL" python3 -m pytest -s -x \
  "$GIT_ROOT/tests/v1/kv_connector/nixl_integration/test_accuracy.py"
