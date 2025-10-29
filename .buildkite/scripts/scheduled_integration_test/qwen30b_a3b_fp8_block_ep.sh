#!/usr/bin/env bash
set -euxo pipefail

# Args: [THRESH] [NUM_QUESTIONS] [START_PORT]
THRESH=${1:-0.8}
NUM_Q=${2:-1319}
PORT=${3:-8020}

wait_for_server() {
  local port=$1
  timeout 600 bash -c '
    until curl -sf "http://127.0.0.1:'"$port"'/health" > /dev/null; do
      sleep 1
    done'
}

MODEL="QWen/Qwen3-30B-A3B-FP8"
BACKENDS=("deepep_high_throughput" "deepep_low_latency")

for BACK in "${BACKENDS[@]}"; do
  VLLM_ALL2ALL_BACKEND=$BACK \
  vllm serve "$MODEL" \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --trust-remote-code \
    --max-model-len 2048 \
    --port $PORT &
  wait_for_server $PORT

  OUT=/tmp/qwen30b_a3b_fp8_block_${BACK}.json
  python3 tests/evals/gsm8k/gsm8k_eval.py --host http://127.0.0.1 --port $PORT --num-questions ${NUM_Q} --save-results ${OUT}
  python3 - <<PY
import json; acc=json.load(open('${OUT}'))['accuracy']
print(f"${MODEL} ${BACK}: accuracy {acc:.3f}")
assert acc >= ${THRESH}, f"${MODEL} ${BACK} accuracy {acc}"
PY

  pkill -f "vllm serve" || true
  sleep 2
  PORT=$((PORT+1))
done
