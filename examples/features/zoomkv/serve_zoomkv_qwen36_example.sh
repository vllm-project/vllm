#!/usr/bin/env bash
# Experimental ZoomKV K+V CPU-offload performance-test template.
# Host-specific GPU/model defaults remain overridable through the environment.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
export PYTHONPATH=${PYTHONPATH:-.}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-fork}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}

CPU_BYTES_PER_RANK=${ZOOMKV_CPU_BYTES_PER_RANK:-25769803776}
if [[ ! "$CPU_BYTES_PER_RANK" =~ ^[1-9][0-9]*$ ]]; then
  echo "ZOOMKV_CPU_BYTES_PER_RANK must be a positive integer byte count" >&2
  exit 2
fi

# Build JSON with a JSON encoder so environment overrides cannot produce
# malformed --attention-config input.
ATTENTION_CONFIG=$(
  ZOOMKV_CPU_BYTES_PER_RANK="$CPU_BYTES_PER_RANK" python - <<'PY'
import json
import os

print(json.dumps({
    "backend": "ZOOMKV",
    "zoomkv_sink_size": 64,
    "zoomkv_local_size": 256,
    "zoomkv_chunk_size": 16,
    "zoomkv_chunk_candidates": 200,
    "zoomkv_dense_chunks": 60,
    "zoomkv_dense_topk": 8,
    "zoomkv_sparse_topk": 4,
    "zoomkv_final_topk": 100,
    "zoomkv_full_attention_threshold": 3072,
    "zoomkv_enable_offload": True,
    "zoomkv_cpu_bytes_per_rank": int(os.environ["ZOOMKV_CPU_BYTES_PER_RANK"]),
    "zoomkv_offload_unit_tokens": 64,
    "zoomkv_strict_kernels": True,
    "zoomkv_dense_fallback": False,
}, separators=(",", ":")))
PY
)

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH:-/data/qyl/models/Qwen3.6-27B}" \
  --served-model-name "${SERVED_MODEL_NAME:-Qwen3.6-27B}" \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  --block-size 16 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}" \
  --no-enable-prefix-caching \
  --no-enable-log-requests \
  --disable-log-stats \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --attention-config "$ATTENTION_CONFIG"
