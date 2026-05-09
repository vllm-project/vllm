#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
    exec vllm serve "$@"
fi

exec vllm serve "${DAMORK_MODEL}" \
    --host "${DAMORK_HOST}" \
    --port "${DAMORK_PORT}" \
    --served-model-name "${DAMORK_SERVED_MODEL_NAME}" \
    --max-model-len "${DAMORK_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${DAMORK_GPU_MEMORY_UTILIZATION}" \
    --enforce-eager
