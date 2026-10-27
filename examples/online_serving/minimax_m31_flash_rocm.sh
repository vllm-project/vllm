#!/usr/bin/env bash
# MiniMax-M3.1-Flash TP4 serving recipe for AMD MI350X/MI355X.
#
# Precision actually served:
#   official NVFP4 checkpoint, routed experts requantized NVFP4→MXFP4 at load,
#   FP8 main KV + FP8 indexer. This is not native NVFP4 serving.
#
# Do not enable monokernel (--compilation-config.monokernel) or dual-batch
# overlap for this model until they are proven on the all-sparse M3.1 layout.
set -euo pipefail

MODEL="${MODEL:-MiniMaxAI/MiniMax-M3.1-Flash}"
TP="${TP:-4}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS="${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-1}"

exec vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --block-size 128 \
  --kv-cache-dtype fp8 \
  --attention-config '{"indexer_kv_dtype": "fp8"}' \
  --quantization-config.moe mxfp4 \
  --speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": ${NUM_SPECULATIVE_TOKENS}, \"kv_cache_dtype\": \"fp8\", \"draft_sample_method\": \"probabilistic\"}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --reasoning-parser minimax_m3 \
  --tool-call-parser minimax_m3 \
  --enable-auto-tool-choice \
  "$@"
