#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# NOTE: For correct KV cache transfer, ensure all processes use the same PYTHONHASHSEED to keep the hash of the KV cache consistent across processes.
export PYTHONHASHSEED=0

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <prefiller | decoder> [model]"
    exit 1
fi

if [[ $# -eq 1 ]]; then
    echo "Using default model: meta-llama/Llama-3.1-8B-Instruct"
    MODEL="meta-llama/Llama-3.1-8B-Instruct"
else
    echo "Using model: $2"
    MODEL=$2
fi


if [[ $1 == "prefiller" ]]; then
    # Prefiller listens on port 7100
    prefill_config_file=$SCRIPT_DIR/configs/lmcache-prefiller-config.yaml

    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE=$prefill_config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=${PREFILLER_DEVICE_ID:-0} \
        vllm serve $MODEL \
        --port 7100 \
        --enforce-eager \
        --no-enable-prefix-caching \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'




elif [[ $1 == "decoder" ]]; then
    # Decoder listens on port 7200
    decode_config_file=$SCRIPT_DIR/configs/lmcache-decoder-config.yaml

    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE=$decode_config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=${DECODER_DEVICE_ID:-1} \
        vllm serve $MODEL \
        --port 7200 \
        --enforce-eager \
        --no-enable-prefix-caching \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}'


else
    echo "Invalid role: $1"
    echo "Should be either prefill, decode"
    exit 1
fi
