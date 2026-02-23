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
    # can you check if the number of GPUs are >=2 via nvidia-smi/rocm-smi?
    which rocm-smi > /dev/null 2>&1
    if [ $? -ne 0 ]; then
	num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
	num_gpus=$(rocm-smi --showid | grep Instinct | wc -l)
    fi

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

    bash disagg_vllm_launcher.sh prefiller \
        > >(tee prefiller.log) 2>&1 &
    prefiller_pid=$!
    PIDS+=($prefiller_pid)

    bash disagg_vllm_launcher.sh decoder  \
        > >(tee decoder.log)  2>&1 &
    decoder_pid=$!
    PIDS+=($decoder_pid)

    python3 disagg_proxy_server.py \
        --host localhost \
        --port 9000 \
        --prefiller-host localhost \
        --prefiller-port 8100 \
        --decoder-host localhost \
        --decoder-port 8200  \
        > >(tee proxy.log)    2>&1 &
    proxy_pid=$!
    PIDS+=($proxy_pid)

    wait_for_server 8100
    wait_for_server 8200
    wait_for_server 9000

    echo "All servers are up. Starting benchmark..."

    # begin benchmark
    cd ../../../../benchmarks/
    vllm bench serve --port 9000 --seed $(date +%s) \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset-name random --random-input-len 7500 --random-output-len 200 \
        --num-prompts 200 --burstiness 100 --request-rate 3.6 | tee benchmark.log

    echo "Benchmarking done. Cleaning up..."

    cleanup

}

main
