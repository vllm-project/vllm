#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <prefiller | decoder> [model]"
    exit 1
fi

if [[ $# -eq 1 ]]; then
    echo "Using default model: meta-llama/Llama-3.1-8B-Instruct"
    MODEL="/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/"
else
    echo "Using model: $2"
    MODEL=$2
fi


if [[ $1 == "prefiller" ]]; then
    # Prefiller listens on port 8100
    prefill_config_file=$SCRIPT_DIR/configs/lmcache-config-lm.yaml

    #UCX_TLS=tcp \
    LMCACHE_CONFIG_FILE=$prefill_config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        LMCACHE_REMOTE_SERDE=naive \
        LMCACHE_CHUNK_SIZE=256 \
        vllm serve $MODEL \
        --port 1100 \
        --gpu_memory_utilization 0.5 \
        --disable-log-requests \
        --tensor_parallel_size 2 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'


elif [[ $1 == "decoder" ]]; then
    # Decoder listens on port 8200
    decode_config_file=$SCRIPT_DIR/configs/lmcache-config-lm.yaml

    #UCX_TLS=tcp \
    LMCACHE_CONFIG_FILE=$decode_config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        LMCACHE_REMOTE_SERDE=naive \
        LMCACHE_CHUNK_SIZE=256 \
        vllm serve $MODEL \
        --port 1200 \
        --gpu_memory_utilization 0.5 \
        --disable-log-requests \
        --tensor_parallel_size 2 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1"}}'


else
    echo "Invalid role: $1"
    echo "Should be either prefill, decode"
    exit 1
fi
