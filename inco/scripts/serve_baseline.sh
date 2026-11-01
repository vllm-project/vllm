#!/usr/bin/env bash
# Launch the vLLM server for the baseline sweep.
#
# Launched ONCE at maximum batch size; only client concurrency is swept, so a
# single engine configuration (one set of CUDA graphs, one KV cache) backs the
# whole Pareto curve.
#
# Standard perf features left at their defaults ON (asserted by the client's
# /server_info audit before any measurement is taken):
#   * CUDA graphs        - piecewise + full-decode via torch.compile
#   * async scheduling   - CPU/GPU overlap ("overlap scheduler")
#   * chunked prefill    - via --max-num-batched-tokens
#   * prefix caching     - flushed between points so it cannot flatter results
#   * KV cache size      - pinned, so the curve does not silently change when
#                          the provider hands out a different H100 variant
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/workload.env"

# Note the flag is --kv-cache-memory-bytes; vLLM's own startup hint says
# --kv-cache-memory, which this tree does not accept.
kv_args=()
if [[ "${INCO_KV_CACHE_GIB:-0}" -gt 0 ]]; then
  kv_args=(--kv-cache-memory-bytes "$((INCO_KV_CACHE_GIB * 1024 * 1024 * 1024))")
fi

exec vllm serve "${INCO_MODEL}" \
  --host "${INCO_HOST}" \
  --port "${INCO_PORT}" \
  --served-model-name "${INCO_MODEL}" \
  --tensor-parallel-size "${INCO_NUM_GPUS}" \
  --dtype "${INCO_DTYPE}" \
  --max-model-len "${INCO_MAX_MODEL_LEN}" \
  --max-num-seqs "${INCO_MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${INCO_MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${INCO_GPU_MEMORY_UTILIZATION}" \
  --async-scheduling \
  --enable-prefix-caching \
  "${kv_args[@]}" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  "$@"
