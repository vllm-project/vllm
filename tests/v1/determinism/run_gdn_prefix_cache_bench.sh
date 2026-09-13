#!/usr/bin/env bash
# TTFT smoke: shared 2000-token prefix, BI=1, prefix cache on vs off.
#
# Start two servers yourself (or one at a time), then point BASE_URL here.
#
#   export VLLM_BATCH_INVARIANT=1
#   export CUDA_VISIBLE_DEVICES=0,1
#   vllm serve "$VLLM_GDN_PC_TEST_MODEL" \
#     --tensor-parallel-size 2 \
#     --max-model-len 3264 \
#     --max-num-seqs 64 \
#     --gpu-memory-utilization 0.95 \
#     --trust-remote-code \
#     --enable-prefix-caching \
#     --port 8000
#
#   BASE_URL=http://127.0.0.1:8000 TAG=on \
#     bash tests/v1/determinism/run_gdn_prefix_cache_bench.sh
#
# Repeat with --no-enable-prefix-caching and TAG=off.
set -euo pipefail

MODEL="${VLLM_GDN_PC_TEST_MODEL:?set VLLM_GDN_PC_TEST_MODEL to a GDN checkpoint}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
TAG="${TAG:-on}"
RESULT_DIR="${RESULT_DIR:-./gdn_pc_bench_results}"
mkdir -p "${RESULT_DIR}"

bench_one() {
  local conc="$1"
  local nprompt="$2"
  local rep="$3"
  vllm bench serve \
    --backend openai \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --endpoint /v1/completions \
    --dataset-name prefix_repetition \
    --prefix-repetition-prefix-len 2000 \
    --prefix-repetition-suffix-len 0 \
    --prefix-repetition-num-prefixes 1 \
    --prefix-repetition-output-len 200 \
    --num-prompts "${nprompt}" \
    --max-concurrency "${conc}" \
    --request-rate inf \
    --percentile-metrics ttft,tpot,itl \
    --save-result \
    --result-dir "${RESULT_DIR}" \
    --result-filename "bench_${TAG}_c${conc}_r${rep}.json"
}

echo "warmup TAG=${TAG} BASE_URL=${BASE_URL}"
bench_one 1 4 warmup || true
for REP in 1 2; do
  bench_one 1 8 "${REP}"
  bench_one 16 32 "${REP}"
done
