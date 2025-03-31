#!/bin/bash
set -xe

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

python3 disagg_prefill_proxy_xpyd.py &

MODEL_NAME=${HF_MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}

## 2P2D, TP=1
## prefilling instance, which is the KV producer
#CUDA_VISIBLE_DEVICES=4 vllm serve $MODEL_NAME \
#    --host 0.0.0.0 \
#    --port 20001 \
#    --served-model-name base_model \
#    --max-model-len 8192 \
#    --gpu-memory-utilization 0.8 \
#    --kv-transfer-config \
#    '{"proxy_ip":"0.0.0.0","proxy_port":"30001","kv_connector":"P2pConnector","kv_role":"kv_producer","http_port":"20001","kv_port":"21001"}' &
#
## prefilling instance, which is the KV producer
#CUDA_VISIBLE_DEVICES=5 vllm serve $MODEL_NAME \
#    --host 0.0.0.0 \
#    --port 20002 \
#    --served-model-name base_model \
#    --max-model-len 8192 \
#    --gpu-memory-utilization 0.8 \
#    --kv-transfer-config \
#    '{"proxy_ip":"0.0.0.0","proxy_port":"30001","kv_connector":"P2pConnector","kv_role":"kv_producer","http_port":"20002","kv_port":"22001"}' &
#
## decoding instance, which is the KV consumer
#CUDA_VISIBLE_DEVICES=6 vllm serve $MODEL_NAME \
#    --host 0.0.0.0 \
#    --port 20003 \
#    --served-model-name base_model \
#    --max-model-len 8192 \
#    --gpu-memory-utilization 0.8 \
#    --kv-transfer-config \
#    '{"proxy_ip":"0.0.0.0","proxy_port":"30001","kv_connector":"P2pConnector","kv_role":"kv_consumer","http_port":"20003","kv_port":"23001"}' &
#
## decoding instance, which is the KV consumer
#CUDA_VISIBLE_DEVICES=7 vllm serve $MODEL_NAME \
#    --host 0.0.0.0 \
#    --port 20004 \
#    --served-model-name base_model \
#    --max-model-len 8192 \
#    --gpu-memory-utilization 0.8 \
#    --kv-transfer-config \
#    '{"proxy_ip":"0.0.0.0","proxy_port":"30001","kv_connector":"P2pConnector","kv_role":"kv_consumer","http_port":"20004","kv_port":"24001"}' &


# 2P2D, TP=2
# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=0,1 vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port 20001 \
    --tensor-parallel-size 2 \
    --served-model-name base_model \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"proxy_ip":"0.0.0.0","proxy_port":"30001","kv_connector":"P2pConnector","kv_role":"kv_producer","http_port":"20001","kv_port":"21001"}' &

# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=2,3 vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port 20002 \
    --tensor-parallel-size 2 \
    --served-model-name base_model \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"proxy_ip":"0.0.0.0","proxy_port":"30001","kv_connector":"P2pConnector","kv_role":"kv_producer","http_port":"20002","kv_port":"22001"}' &

# decoding instance, which is the KV consumer
CUDA_VISIBLE_DEVICES=4,5 vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port 20003 \
    --tensor-parallel-size 2 \
    --served-model-name base_model \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"proxy_ip":"0.0.0.0","proxy_port":"30001","kv_connector":"P2pConnector","kv_role":"kv_consumer","http_port":"20003","kv_port":"23001"}' &

# decoding instance, which is the KV consumer
CUDA_VISIBLE_DEVICES=6,7 vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port 20004 \
    --tensor-parallel-size 2 \
    --served-model-name base_model \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"proxy_ip":"0.0.0.0","proxy_port":"30001","kv_connector":"P2pConnector","kv_role":"kv_consumer","http_port":"20004","kv_port":"24001"}' &