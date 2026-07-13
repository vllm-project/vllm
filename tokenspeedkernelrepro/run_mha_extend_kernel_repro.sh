#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/app/tokspd}
BENCH_DIR=${BENCH_DIR:-${SCRIPT_DIR}}
VLLM_ROOT=${VLLM_ROOT:-${ROOT}/tokspd-int}
OUT_DIR=${OUT_DIR:-/tmp/rocm_tokenspeed_mha_extend_kernel_repro}
GPU=${GPU:-0}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-50}
DTYPE=${DTYPE:-bf16}
NO_SINKS=${NO_SINKS:-0}

export PYTHONPATH="${VLLM_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --dtype "${DTYPE}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
)
if [[ "${NO_SINKS}" == "1" ]]; then
  COMMON_ARGS+=(--no-sinks)
fi

run_logged() {
  local name=$1
  shift
  local log_path="${OUT_DIR}/${name}.log"
  echo "== ${name} =="
  echo "$*" | tee "${log_path}"
  HIP_VISIBLE_DEVICES="${GPU}" "$@" 2>&1 | tee -a "${log_path}"
  echo
}

# Native extend corner case: a one-token extend should behave like decode, but
# this exposes whether the extend kernel has a high fixed per-request cost.
run_logged extend_full_q1 \
  python "${BENCH_DIR}/bench_extend_decomposition_perf.py" \
    "${COMMON_ARGS[@]}" \
    --requests 128 \
    --query-len 1 \
    --seq-len 1024

# Multi-token full-attention extend. Keep this in the repro because native
# extend is not universally slower; this row prevents over-attributing the gap.
run_logged extend_full_q8 \
  python "${BENCH_DIR}/bench_extend_decomposition_perf.py" \
    "${COMMON_ARGS[@]}" \
    --requests 128 \
    --query-len 8 \
    --seq-len 1024

# Sliding-window extend is common for the model path we were evaluating.
run_logged extend_sliding128_q8 \
  python "${BENCH_DIR}/bench_extend_decomposition_perf.py" \
    "${COMMON_ARGS[@]}" \
    --requests 128 \
    --query-len 8 \
    --seq-len 1024 \
    --sliding-window 128

# Request-level mixed batch comparison against ROCM_AITER_UNIFIED_ATTN.
# The tokenspeed_native_extend row routes extend requests through the native MHA
# extend wrapper while keeping the rest of the batch shape identical.
run_logged mixed_balanced_native_extend \
  python "${BENCH_DIR}/bench_mixed_vs_aiter_unified.py" \
    "${COMMON_ARGS[@]}" \
    --num-decodes 128 \
    --decode-seq-len 1024 \
    --num-extends 16 \
    --extend-query-len 16 \
    --extend-seq-len 1024 \
    --num-prefills 8 \
    --prefill-query-len 64

# Larger near-chunk-limit mixed batch. This is closer to the serving path where
# max-num-batched-tokens constrains how much prefill/extend work enters a step.
run_logged mixed_prefill_heavy_native_extend \
  python "${BENCH_DIR}/bench_mixed_vs_aiter_unified.py" \
    "${COMMON_ARGS[@]}" \
    --num-decodes 512 \
    --decode-seq-len 1024 \
    --num-extends 16 \
    --extend-query-len 16 \
    --extend-seq-len 1024 \
    --num-prefills 28 \
    --prefill-query-len 256

python "${BENCH_DIR}/summarize_mha_extend_kernel_repro.py" \
  --out "${OUT_DIR}/summary.md" \
  "${OUT_DIR}"/*.log

echo "Summary written to ${OUT_DIR}/summary.md"
cat "${OUT_DIR}/summary.md"
