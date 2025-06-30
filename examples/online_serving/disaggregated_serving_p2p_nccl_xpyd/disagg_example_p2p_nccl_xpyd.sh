#!/bin/bash

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30001}

# In this example we run 1P3D. You can modify the following variables for 
# XpYd support.
# Currently we only support same TP and PP between prefill and decode instances.
PREFILL_PORT=${PREFILL_PORT:-20005}
DECODE_PORTS=${DECODE_PORTS:-20009,20003,20008}
PREFILL_GPU=${PREFILL_GPU:-0}
DECODE_GPUS=${DECODE_GPUS:-1,2,3}

echo "Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change."
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Prefill GPU: $PREFILL_GPU, Port: $PREFILL_PORT"
echo "  Decode GPUs: $DECODE_GPUS, Ports: $DECODE_PORTS"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

PIDS=()

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    local files=("disagg_proxy_p2p_nccl_xpyd.py")
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Required file $file not found in $(pwd)"
            exit 1
        fi
    done
}

check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Please set it to your Hugging Face token."
        echo "Example: export HF_TOKEN=your_token_here"
        exit 1
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN is not a valid Hugging Face token. Please set it to your Hugging Face token."
        exit 1
    fi
    echo "HF_TOKEN is set and valid."
}

check_num_gpus() {
    # Check if the number of GPUs are >=2 via nvidia-smi
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
    if ! python3 -c "import $1" > /dev/null 2>&1; then
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
    kill -- -$$            # negative PID  ==  "this whole process-group"
    wait                   # reap children so we don't leave zombies
    exit 0
}

wait_for_server() {
  local port=$1
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      echo "Server on port $port is ready."
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server on port $port"
      return 1
    fi

    sleep 1
  done
}

main() {
    check_required_files
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
    echo "Please check prefill1.log, decode1.log, decode2.log, decode3.log and proxy.log for logs."

    # launch proxy
    python3 disagg_proxy_p2p_nccl_xpyd.py &
    PIDS+=($!)

    # Parse GPU and port arrays
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    # 1P (1 Producer) - Prefill server
    echo "Starting prefill server on GPU $PREFILL_GPU, port $PREFILL_PORT..."
    CUDA_VISIBLE_DEVICES=$PREFILL_GPU VLLM_USE_V1=1 vllm serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $PREFILL_PORT \
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
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"21001\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$PREFILL_PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > prefill1.log 2>&1 &
    PIDS+=($!)

    # 3D (3 Decoders) - Decode servers
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local kv_port=$((22001 + i))
        
        echo "Starting decode server $((i+1)) on GPU $gpu_id, port $port..."
        VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=$gpu_id vllm serve $MODEL \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $port \
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
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > decode$((i+1)).log 2>&1 &
        PIDS+=($!)
    done

    # Wait for all servers to start with error handling
    echo "Waiting for all servers to start..."
    for port in $PREFILL_PORT "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start server on port $port"
            cleanup
            exit 1
        fi
    done

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