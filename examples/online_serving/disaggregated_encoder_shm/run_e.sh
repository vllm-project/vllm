#!/bin/bash

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/user/ec_cache}"
rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
    --gpu-memory-utilization 0.01 \
    --port "23001" \
    --enforce-eager \
    --conver "mm_encoder_only" \
    --enable-request-id-headers \
    --served-model-name model_name \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs 128 \
    --ec-transfer-config '{
        "ec_connector": "SHMConnector",
        "ec_role": "ec_producer",
        "ec_ip": "127.0.0.1",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'",
            "listen_ports": [30161],
            "engine_id": 0,
            "producer_instances": 1,
            "consumer_instances": 1,
            "producer": {
                "dp_size": 1,
                "tp_size": 1
            },
            "consumer": {
                "dp_size": 1,
                "tp_size": 1
            }
        }
    }'
