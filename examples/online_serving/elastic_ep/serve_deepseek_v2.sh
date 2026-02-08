#!/bin/bash

HOST="0.0.0.0"
PORT=8006
DATA_PARALLEL_SIZE=4
REDUNDANT_EXPERTS=0
LOCAL_MODEL_PATH="/models/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0"
MODEL_NAME="deepseek-ai/DeepSeek-V2-Lite"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dp)
            DATA_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --re)
            REDUNDANT_EXPERTS="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --local-model)
            MODEL_NAME=$LOCAL_MODEL_PATH
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dp SIZE                    Set data parallel size (default: 4)"
            echo "  --re SIZE                    Set redundant experts (default: 0)"
            echo "  --host HOST                  Set host address (default: 0.0.0.0)"
            echo "  --port PORT                  Set port number (default: 8006)"
            echo "  --model MODEL_NAME           Set model name or path"
            echo "  -h, --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting vLLM server for $MODEL_NAME with data parallel size: $DATA_PARALLEL_SIZE and redundant experts: $REDUNDANT_EXPERTS"

export RAY_DEDUP_LOGS=0
export VLLM_USE_DEEP_GEMM=1

vllm serve $MODEL_NAME \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --data-parallel-size-local $DATA_PARALLEL_SIZE \
    --data-parallel-backend ray \
    --enforce-eager \
    --enable-expert-parallel \
    --enable-eplb \
    --all2all-backend pplx \
    --eplb-config.num_redundant_experts $REDUNDANT_EXPERTS \
    --trust-remote-code \
    --host $HOST \
    --port $PORT
