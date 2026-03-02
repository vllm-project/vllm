#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
PORT="${PORT:-8000}"
GPU="${GPU:-1}"
EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache}"

rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

CUDA_VISIBLE_DEVICES="$GPU" \
vllm serve "$MODEL" \
    --port "$PORT" \
    --enable-log-requests \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_both",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    "$@"
