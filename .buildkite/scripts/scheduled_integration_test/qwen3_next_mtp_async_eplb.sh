#!/usr/bin/env bash
set -euxo pipefail

# args: [THRESHOLD] [NUM_QUESTIONS] [START_PORT]
THRESHOLD=${1:-0.25}
NUM_Q=${2:-1319}
PORT=${3:-8040}
OUT_DIR=${OUT_DIR:-/tmp/vllm-scheduled}
mkdir -p "${OUT_DIR}"

wait_for_server() {
  local port=$1
  timeout 600 bash -c '
    until curl -sf "http://127.0.0.1:'"$port"'/health" > /dev/null; do
      sleep 1
    done'
}

MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"

# Set BACKENDS based on platform
if command -v rocm-smi &> /dev/null || [[ -d /opt/rocm ]] || [[ -n "${ROCM_PATH:-}" ]]; then
  # ROCm platform
  BACKENDS=("allgather_reducescatter")
  # Disable MOE padding for ROCm since it is causing eplb to fail
  export VLLM_ROCM_MOE_PADDING=0
else
  # Non-ROCm platform (CUDA/other)
  BACKENDS=("deepep_high_throughput" "deepep_low_latency")
fi

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

for BACK in "${BACKENDS[@]}"; do
  VLLM_DEEP_GEMM_WARMUP=skip \
  VLLM_ALL2ALL_BACKEND=$BACK \
  vllm serve "$MODEL" \
    --enforce-eager \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-config '{"window_size":200,"step_interval":600,"use_async":true}' \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}' \
    --trust-remote-code \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --port $PORT &
  SERVER_PID=$!
  wait_for_server $PORT

  TAG=$(echo "$MODEL" | tr '/: \\n' '_____')
  OUT="${OUT_DIR}/${TAG}_${BACK}.json"
  python3 tests/evals/gsm8k/gsm8k_eval.py --host http://127.0.0.1 --port $PORT --num-questions ${NUM_Q} --save-results ${OUT}
  python3 - <<PY
import json; acc=json.load(open('${OUT}'))['accuracy']
print(f"${MODEL} ${BACK}: accuracy {acc:.3f}")
assert acc >= ${THRESHOLD}, f"${MODEL} ${BACK} accuracy {acc}"
PY

  cleanup
  SERVER_PID=
  sleep 1
  PORT=$((PORT+1))
done
