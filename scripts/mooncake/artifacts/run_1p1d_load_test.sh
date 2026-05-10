#!/usr/bin/env bash
set -euo pipefail

# Load test extracted from the post_serve section of:
# pd_tp_dep_mooncake_offload_nixl_400G_producer_random_python_bench.yaml

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-nvidia/Kimi-K2.5-NVFP4}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
URL="${URL:-${ROUTER_URL:-http://127.0.0.1:${ROUTER_PORT}}}"

VLLM_BENCH_REPO="${VLLM_BENCH_REPO:-/home/${USER}/vllm}"
BENCH_PYTHON="${BENCH_PYTHON:-${VLLM_BENCH_REPO}/.venv/bin/python}"

NUM_PROMPTS="${NUM_PROMPTS:-75}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-75}"
MULTI_TURN_NUM_TURNS="${MULTI_TURN_NUM_TURNS:-30}"
RANDOM_PREFIX_LEN="${RANDOM_PREFIX_LEN:-20000}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-10000}"
PER_TURN_INPUT_LEN="${PER_TURN_INPUT_LEN:-2048}"
LIMIT_MIN_TOKENS="${LIMIT_MIN_TOKENS:-900}"
LIMIT_MAX_TOKENS="${LIMIT_MAX_TOKENS:-900}"

if [[ -z "${LOG_DIR:-}" ]]; then
  latest_log_dir="$(find "${SCRIPT_DIR}/logs" -maxdepth 1 -type d -name 'kimi-1p1d_*' 2>/dev/null | sort | tail -n 1 || true)"
  LOG_DIR="${latest_log_dir:-${SCRIPT_DIR}/logs/load-test_$(date +%Y%m%d_%H%M%S)}"
fi

PERF_DIR="${PERF_DIR:-${LOG_DIR}/perf}"
DATASET_PATH="${DATASET_PATH:-${PERF_DIR}/multi_turn_random.json}"
STATS_PATH="${STATS_PATH:-${PERF_DIR}/multi_turn_stats.json}"
RUN_LOG="${RUN_LOG:-${PERF_DIR}/load_test.log}"

mkdir -p "${PERF_DIR}"

echo "=== 1P/1D load test ===" | tee "${RUN_LOG}"
echo "URL:             ${URL}" | tee -a "${RUN_LOG}"
echo "Model:           ${MODEL}" | tee -a "${RUN_LOG}"
echo "Log dir:         ${LOG_DIR}" | tee -a "${RUN_LOG}"
echo "Perf dir:        ${PERF_DIR}" | tee -a "${RUN_LOG}"
echo "Prompts:         ${NUM_PROMPTS}" | tee -a "${RUN_LOG}"
echo "Concurrency:     ${MAX_CONCURRENCY}" | tee -a "${RUN_LOG}"
echo "Dataset:         ${DATASET_PATH}" | tee -a "${RUN_LOG}"
echo "Stats:           ${STATS_PATH}" | tee -a "${RUN_LOG}"
echo "Started:         $(date -Iseconds)" | tee -a "${RUN_LOG}"
echo "======================" | tee -a "${RUN_LOG}"

"${BENCH_PYTHON}" \
  "${VLLM_BENCH_REPO}/benchmarks/multi_turn/gen_random_multi_turn.py" \
  --tokenizer "${MODEL}" \
  --num-prompts "${NUM_PROMPTS}" \
  --multi-turn-num-turns "${MULTI_TURN_NUM_TURNS}" \
  --random-prefix-len "${RANDOM_PREFIX_LEN}" \
  --random-input-len "${RANDOM_INPUT_LEN}" \
  --per-turn-input-len "${PER_TURN_INPUT_LEN}" \
  --output "${DATASET_PATH}" \
  2>&1 | tee -a "${RUN_LOG}"

"${BENCH_PYTHON}" \
  "${VLLM_BENCH_REPO}/benchmarks/multi_turn/benchmark_serving_multi_turn.py" \
  --model "${MODEL}" \
  --trust-remote-code \
  --url "${URL}" \
  --input-file "${DATASET_PATH}" \
  --num-clients "${NUM_PROMPTS}" \
  --max-active-conversations "${MAX_CONCURRENCY}" \
  --limit-min-tokens "${LIMIT_MIN_TOKENS}" \
  --limit-max-tokens "${LIMIT_MAX_TOKENS}" \
  --stats-json-output "${STATS_PATH}" \
  2>&1 | tee -a "${RUN_LOG}"

echo "Finished:        $(date -Iseconds)" | tee -a "${RUN_LOG}"
