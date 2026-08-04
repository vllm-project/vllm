#!/bin/bash

echo "Warning: LMCache disaggregated prefill support for vLLM v1 is experimental and subject to change."


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
    python -c "import $1" > /dev/null 2>&1
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
    trap - INT TERM USR1   # prevent re-entrancy
    
    # Kill all tracked PIDs
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill "$pid" 2>/dev/null
        fi
    done
    
    # Wait a moment for graceful shutdown
    sleep 2
    
    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done
    
    # Kill the entire process group as backup
    kill -- -$$ 2>/dev/null
    
    echo "All processes stopped."
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
    ensure_python_library_installed lmcache
    ensure_python_library_installed nixl
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching prefiller, decoder and proxy..."
    echo "Please check prefiller.log, decoder.log and proxy.log for logs."

    # Launch the proxy first
    python3 ../disagg_proxy_server.py \
        --host localhost \
        --port 9100 \
        --prefiller-host localhost \
        --prefiller-port 7100 \
        --num-prefillers 2 \
        --decoder-host localhost \
        --decoder-port 7200  \
        --decoder-init-port 7300 \
        --decoder-alloc-port 7400 \
        --proxy-host localhost \
        --proxy-port 7500 \
        --num-decoders 2 \
        > >(tee proxy.log)    2>&1 &
    proxy_pid=$!
    PIDS+=($proxy_pid)

    # Launch the decoder
    bash disagg_vllm_launcher.sh decoder1  \
        > >(tee decoder1.log)  2>&1 &
    decoder_pid=$!
    PIDS+=($decoder_pid)

    sleep 5
    # Launch the second decoder 
    bash disagg_vllm_launcher.sh decoder2  \
        > >(tee decoder2.log)  2>&1 &
    decoder_pid=$!
    PIDS+=($decoder_pid)
    wait_for_server 7200
    wait_for_server 7201


    # Launch the prefillers next
    bash disagg_vllm_launcher.sh prefiller1 \
        > >(tee prefiller1.log) 2>&1 &
    prefiller_pid=$!
    PIDS+=($prefiller_pid)

    sleep 5  # Don't launch the second prefiller too quickly
    bash disagg_vllm_launcher.sh prefiller2 \
        > >(tee prefiller2.log) 2>&1 &
    prefiller2_pid=$!
    PIDS+=($prefiller2_pid)

    wait_for_server 7100
    wait_for_server 7101
    wait_for_server 9100

    echo "==================================================="
    echo "All servers are up. You can send request now..."
    echo "Press Ctrl-C to terminate all instances."

    # Keep the script running until interrupted
    echo "Script is running. Waiting for termination signal..."
    echo "==================================================="

    while true; do
        sleep 1
    done
}

main
