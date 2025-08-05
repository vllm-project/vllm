#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <prefiller | decoder> [server] [tp] [model]"
    exit 1
fi

SERVER="lm"
TP_SIZE=1
MODEL="llama3.1/Meta-Llama-3.1-8B-Instruct"

if [[ $# -eq 1 ]]; then
    echo "Using default server: $SERVER"
    echo "Using default tp: $TP_SIZE"
    echo "Using default model: $MODEL"
else
    SERVER=$2
    TP_SIZE=$3
    MODEL=$4
    echo "Using server: $SERVER"
    echo "Using tp: $TP_SIZE"
    echo "Using model: $MODEL"
fi


if [[ $1 == "prefiller" ]]; then
    if [[ $SERVER == "lm" ]]; then
        # Prefiller listens on port 8100
        prefill_config_file=$SCRIPT_DIR/configs/lmcache-config-lm.yaml
    elif [[ $SERVER == "redis" ]]; then
        # Prefiller listens on port 6379
        prefill_config_file=$SCRIPT_DIR/configs/lmcache-config-redis.yaml
    else
        echo "Invalid server: $2"
        exit 1
    fi

    #UCX_TLS=tcp \
    LMCACHE_CONFIG_FILE=$prefill_config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        RANK=0 \
        vllm serve $MODEL \
        --port 1100 \
        --disable-log-requests \
        --enforce-eager \
        --tensor_parallel_size $TP_SIZE \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'


elif [[ $1 == "decoder" ]]; then
    if [[ $SERVER == "lm" ]]; then
        # Decoder listens on port 8100
        decode_config_file=$SCRIPT_DIR/configs/lmcache-config-lm.yaml
    elif [[ $SERVER == "redis" ]]; then
        # Decoder listens on port 6379
        decode_config_file=$SCRIPT_DIR/configs/lmcache-config-redis.yaml
    else
        echo "Invalid server: $2"
        exit 1
    fi

    #UCX_TLS=tcp \
    LMCACHE_CONFIG_FILE=$decode_config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        RANK=1 \
        vllm serve $MODEL \
        --port 1200 \
        --disable-log-requests \
        --enforce-eager \
        --tensor_parallel_size $TP_SIZE \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1"}}'


else
    echo "Invalid role: $1"
    echo "Should be either prefill, decode"
    exit 1
fi
