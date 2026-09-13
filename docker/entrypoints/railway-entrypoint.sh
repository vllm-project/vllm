#!/bin/sh
# Entrypoint for docker/Dockerfile.railway.
#
# Railway injects PORT and expects the service to listen on it. The model and
# any extra `vllm serve` flags come from environment variables so the image
# never needs rebuilding to change them:
#   VLLM_MODEL             HF repo or local path (default: Qwen/Qwen3-0.6B)
#   VLLM_ARGS              extra `vllm serve` flags, whitespace separated
#   VLLM_CPU_KVCACHE_SPACE KV cache size in GiB (default: 2)
#   OMP_NUM_THREADS        OpenMP threads (default: nproc)
set -eu

MODEL="${VLLM_MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"

export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"

# shellcheck disable=SC2086
set -- ${VLLM_ARGS:-}

exec vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" "$@"
