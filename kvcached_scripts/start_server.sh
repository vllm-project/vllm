#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

DEFAULT_MODEL=meta-llama/Llama-3.1-8B
DEFAULT_VLLM_PORT=12346

MODEL=${1:-$DEFAULT_MODEL}
VLLM_PORT=${2:-$DEFAULT_VLLM_PORT}

source "$ENGINE_DIR/.venv/bin/activate"
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ENABLE_KVCACHED=false
export KVCACHED_IPC_NAME=VLLM

vllm serve "$MODEL" \
--disable-log-requests \
--no-enable-prefix-caching \
--port="$VLLM_PORT"
