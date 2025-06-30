#!/bin/bash

echo "Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change."

MODEL=meta-llama/Llama-3.1-8B-Instruct

PIDS=()

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Please set it to your Hugging Face token."
        exit 1
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN is not a valid Hugging Face token. Please set it to your Hugging Face token."
        exit 1
    fi
    echo "HF_TOKEN is set and valid."
}

check_num_gpus() {
    # can you check if the number of GPUs are >=2 via nvidia-smi?
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    python3 -c "import $1" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        if [ "$1" == "nixl" ]; then
            echo "$1 is not installed. Please refer to https://github.com/ai-dynamo/nixl for installation."
        else
            echo "$1 is not installed. Please install it via pip install $1."
        fi
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    kill -- -$$            # negative PID  ==  “this whole process-group”
    wait                   # reap children so we don't leave zombies
    exit 0
}

wait_for_server() {
  local port=$1
  local timeout_seconds=1200
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server"
      return 1
    fi

    sleep 1
  done
}


main() {
    check_hf_token
    check_num_gpus
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching prefiller, decoder and proxy..."
    echo "Please check prefiller.log, decoder.log and proxy.log for logs."

    # launch proxy
    python3 disagg_proxy_p2p_nccl_xpyd.py &
    PIDS+=($!)

    # 1P
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 20005 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --disable-log-request \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > prefill1.log 2>&1 &
    PIDS+=($!)

    # 3D
    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 20009 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --disable-log-request \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > decode1.log 2>&1 &
    PIDS+=($!)


    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 20003 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --disable-log-request \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20003","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > decode2.log 2>&1 &
    PIDS+=($!)

    VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 20008 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --disable-log-request \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20008","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > decode3.log 2>&1 &
    PIDS+=($!)


    wait_for_server 20003
    wait_for_server 20005
    wait_for_server 20008
    wait_for_server 20009

    echo "All servers are up. Starting benchmark..."

    # begin benchmark
    cd ../../../benchmarks/
    python3 benchmark_serving.py --port 10001 --seed $(date +%s) \
        --model $MODEL \
        --dataset-name random --random-input-len 7500 --random-output-len 200 \
        --num-prompts 200 --burstiness 100 --request-rate 2 | tee benchmark.log

    echo "Benchmarking done. Cleaning up..."

    cleanup

}

main