#!/usr/bin/env bash
set -euxo pipefail

# Nightly e2e test for prefetch offloading with a MoE model.
# Runs DeepSeek-V2-Lite with prefetch offloading of MoE expert weights
# and validates GSM8K accuracy matches baseline (no offloading).
#
# args: [THRESHOLD] [NUM_QUESTIONS] [START_PORT]
THRESHOLD=${1:-0.25}
NUM_Q=${2:-1319}
PORT=${3:-8030}
OUT_DIR=${OUT_DIR:-/tmp/vllm-scheduled}
mkdir -p "${OUT_DIR}"

wait_for_server() {
  local port=$1
  timeout 600 bash -c '
    until curl -sf "http://127.0.0.1:'"$port"'/health" > /dev/null; do
      sleep 1
    done'
}

MODEL="deepseek-ai/DeepSeek-V2-Lite"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

vllm serve "$MODEL" \
  --max-model-len 2048 \
  --offload-group-size 8 \
  --offload-num-in-group 2 \
  --offload-prefetch-step 1 \
  --offload-params w13_weight w2_weight \
  --port "$PORT" &
SERVER_PID=$!
wait_for_server "$PORT"

TAG=$(echo "$MODEL" | tr '/: \\n' '_____')
OUT="${OUT_DIR}/${TAG}_prefetch_offload.json"
python3 tests/evals/gsm8k/gsm8k_eval.py --host http://127.0.0.1 --port "$PORT" --num-questions "${NUM_Q}" --save-results "${OUT}"
python3 - <<PY
import json; acc=json.load(open('${OUT}'))['accuracy']
print(f"${MODEL} prefetch_offload: accuracy {acc:.3f}")
assert acc >= ${THRESHOLD}, f"${MODEL} prefetch_offload accuracy {acc}"
PY

cleanup
SERVER_PID=
