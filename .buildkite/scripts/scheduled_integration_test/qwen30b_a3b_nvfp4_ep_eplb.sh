#!/usr/bin/env bash
set -euxo pipefail

# args: [THRESHOLD] [NUM_QUESTIONS] [START_PORT] [DATA_PARALLEL_SIZE] [TENSOR_PARALLEL_SIZE]
THRESHOLD=${1:-0.8}
NUM_Q=${2:-1319}
PORT=${3:-8060}
DATA_PARALLEL_SIZE=${4:-2}
TENSOR_PARALLEL_SIZE=${5:-2}
OUT_DIR=${OUT_DIR:-/tmp/vllm-scheduled}
mkdir -p "${OUT_DIR}"

# NVFP4 first-start runs the FlashInfer FP4 autotuner + cutedsl warmup, which can
# exceed the default engine-ready timeout on a cold cache. Give it room.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}

wait_for_server() {
  local port=$1
  timeout 600 bash -c '
    until curl -sf "http://127.0.0.1:'"$port"'/health" > /dev/null; do
      sleep 1
    done'
}

# ModelOpt NVFP4 checkpoint (exercises EPLB + NVFP4 MoE on Blackwell).
MODEL="nvidia/Qwen3-30B-A3B-FP4"
BACKENDS=("deepep_high_throughput" "deepep_low_latency")

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
  vllm serve "$MODEL" \
    --enforce-eager \
    --enable-eplb \
    --all2all-backend "$BACK" \
    --eplb-config '{"window_size":10, "step_interval":100, "num_redundant_experts":0, "log_balancedness":true, "use_async":false}' \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --data-parallel-size "${DATA_PARALLEL_SIZE}" \
    --enable-expert-parallel \
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

  cleanup
  SERVER_PID=
  sleep 1
  PORT=$((PORT+1))
done
