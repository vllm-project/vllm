#!/usr/bin/env bash
set -euxo pipefail

# args: [THRESHOLD] [NUM_QUESTIONS] [START_PORT]
THRESHOLD=${1:-0.8}
NUM_Q=${2:-1319}
PORT=${3:-8050}
OUT_DIR=${OUT_DIR:-/tmp/vllm-scheduled}
mkdir -p "${OUT_DIR}"

wait_for_server() {
  local port=$1
  timeout 600 bash -c '
    until curl -sf "http://127.0.0.1:'"$port"'/health" > /dev/null; do
      sleep 1
    done'
}

MODEL="Qwen/Qwen3-30B-A3B-FP8"
BACK="allgather_reducescatter"

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

VLLM_DEEP_GEMM_WARMUP=skip \
vllm serve "$MODEL" \
--enforce-eager \
--data-parallel-size 4 \
--enable-expert-parallel \
--enable-eplb \
--all2all-backend "$BACK" \
--eplb-config '{"window_size":20, "step_interval":100, "use_async":true}' \
--trust-remote-code \
--max-model-len 2048 \
--port "$PORT" &
SERVER_PID=$!
wait_for_server "$PORT"

TAG=$(echo "$MODEL" | tr '/: \\n' '_____')
OUT="${OUT_DIR}/${TAG}_${BACK}.json"
python3 tests/evals/gsm8k/gsm8k_eval.py --host http://127.0.0.1 --port "$PORT" --num-questions "${NUM_Q}" --save-results "${OUT}"
python3 - <<PY
import json; acc=json.load(open('${OUT}'))['accuracy']
print(f"${MODEL} ${BACK}: accuracy {acc:.3f}")
assert acc >= ${THRESHOLD}, f"${MODEL} ${BACK} accuracy {acc}"
PY
