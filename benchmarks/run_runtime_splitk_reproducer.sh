#!/usr/bin/env bash
# Runner for the staged BlockScale SplitK zero-init runtime reproducer.
#
# Pins the same AITER / tuned-CSV environment the real Qwen3-Next server and
# the Stage A mini-graph use, so the only variable under test is the graph
# shape. All extra args are forwarded to the Python harness, e.g.:
#
#   bash benchmarks/run_runtime_splitk_reproducer.sh \
#     --scenario seq parallel --num-blocks 8 --m 4 --k 2048
#
set -euo pipefail

cd "$(dirname "$0")/.."
VLLM_ROOT="$(pwd)"

AITER_DIR="${AITER_DIR:-/home/AMD/samremes/dev/aiter}"
TUNED_CSV="${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE:-${AITER_DIR}/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_next_80b_a3b_filtered_built.csv}"

# vLLM is not pip-installed with metadata in every env; make the local
# checkout importable. aiter is never pip-installed, so it always needs PYTHONPATH.
exec env \
  PYTHONPATH="${VLLM_ROOT}:${AITER_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  HF_HUB_OFFLINE=1 \
  VLLM_TARGET_DEVICE=rocm \
  VLLM_ROCM_USE_AITER=1 \
  VLLM_ROCM_USE_AITER_TRITON_GEMM=1 \
  AMDGCN_USE_BUFFER_OPS=0 \
  VLLM_DISABLE_COMPILE_CACHE=1 \
  TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
  AITER_CONFIG_GEMM_A8W8_BLOCKSCALE="${TUNED_CSV}" \
  python benchmarks/debug_vllm_runtime_splitk_reproducer.py "$@"
