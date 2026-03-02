#!/bin/bash

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
VLLM_NIXL_SIDE_CHANNEL_PORT=5601

CUDA_VISIBLE_DEVICES=2 vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "43001" \
    --enforce-eager \
    --enable-request-id-headers \
    --served-model-name model_name \
    --max-model-len 32768  \
    --max-num-seqs 128 \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer"
    }'
