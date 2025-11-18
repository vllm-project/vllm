#!/bin/bash

MODEL_NAME="deepseek-ai/DeepSeek-V2-Lite"
LOCAL_MODEL_PATH="/models/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0"
HOST="localhost"
PORT=8006
NUM_PROMPTS=20
REQUEST_RATE=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --local-model)
            MODEL_NAME=$LOCAL_MODEL_PATH
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --request-rate)
            REQUEST_RATE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL_NAME           Set model name or path (default: deepseek-ai/DeepSeek-V2-Lite)"
            echo "  --local-model                Use local model path (convenience option)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

vllm bench serve \
    --model $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --num-prompts $NUM_PROMPTS \
    --request-rate $REQUEST_RATE
