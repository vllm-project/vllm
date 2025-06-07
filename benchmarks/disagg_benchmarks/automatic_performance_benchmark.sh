#!/bin/bash

# License: This script is provided for managing vLLM servers. It automates the process of starting servers
# with specific configurations and running benchmarks with loop. 
# It supports starting multi chunked/prefill/decode backend on different cards.

# Functionality Description:
# This script is designed to manage the operation of servers using vLLM, with support for different configurations
# such as "chunked" and "disaggregated". It sets up specific hardware configurations for different tasks
# (e.g., Prefill, Decode) and manages their power limits. Additionally, it checks for open ports, waits for server
# readiness, and starts proxy servers. Benchmarks are then executed based on various input parameters.
# The script is used to launch servers, execute tests, and record the results.

# Set script to exit automatically if an error occurs
set -e

RUN_LOG_DIR="./log"
mkdir -p "$RUN_LOG_DIR"

c_cards=(0 1) # Specify the cards used for non-P/D separation
p_cards=(0)   # Specify the cards used for Prefill
d_cards=(1 2) # Specify the cards used for Decode

methods=("chunked" "disagg")
inputs=(128 256 512 1024 2048)
outputs=(128 256 512 1024 2048)
prompts=(10 40 70 100 128 256 512)
request_rates=("inf")

# Check if a port is open on localhost
check_port() {
    local port=$1
    if nc -z localhost $port; then
        echo "Port $port is already in use. Exiting."
        exit 1
    fi
}

# Wait for a server to be ready on a specified port
wait_for_server() {
    local port=$1
    local timeout=300
    local start_time=$(date +%s)
    while ! nc -z localhost $port; do
        sleep 1
        local current_time=$(date +%s)
        if ((current_time - start_time > timeout)); then
            echo "Timeout waiting for server on port $port. Exiting."
            exit 1
        fi
    done
}

# Check if a port is open with a timeout
check_port_open() {
    local port=$1
    if nc -z -w 1 localhost $port; then
        return 0  # Port is open
    else
        return 1  # Port is not open
    fi
}

# Set the NVIDIA GPU power limit
set_nvidia_power_limit() {
    local card=$1
    local limit=$2
    if [ "$limit" -eq 1 ]; then
        sudo nvidia-smi -i "$card" -lgc 525,525
    else
        sudo nvidia-smi -i "$card" -rgc
    fi
}

# Start chunked server setup
start_chunked() {
    pkill -9 python3 || echo "No python3 processes to kill."
    sleep 1

    local ports=()

    for ((i = 0; i < ${#c_cards[@]}; i++)); do
        local card=${c_cards[i]}

        start_port=18000
        port=$((start_port + i))
        ports+=("$port")
        
        CUDA_VISIBLE_DEVICES="$card" python3 -m vllm.entrypoints.openai.api_server \
            --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
            --port "$port" \
            --max-model-len 4096 \
            --trust-remote-code \
            > "$RUN_LOG_DIR/vllm_server_chunked_${card}.log" 2>&1 &

        echo "Started chunked server on card $card with port $port"
    done

    echo -n "Waiting for vLLM servers..."
    for port in ${ports[@]}; do 
        wait_for_server "$port"
    done
    echo " All vLLM servers ready."

    # Start proxy server
    echo "Starting proxy server..."
    python3 ./auto_round_robin_proxy.py --ports "${ports[@]}" --proxy_port 8001 & 
    proxy_pid=$!

    # Wait for proxy server to be ready
    echo -n "Waiting for proxy server..."
    wait_for_server 8001
    echo " Proxy ready (PID: $proxy_pid)"

    # Check if proxy server started successfully
    if ! nc -z localhost 8001; then
        echo "Proxy server failed to start on port 8001"
        return 1  # Return failure
    fi

    return 0  # Return success
}

# Start disaggregated server setup
start_disagg() {
    pkill -9 python3 || echo "No python3 processes to kill."
    sleep 1

    p_num=${#p_cards[@]}
    d_num=${#d_cards[@]}
    kv_parallel_size=$((p_num + d_num))

    local start_port_p=18000  # Starting port for p_cards
    local start_port_d=19000  # Starting port for d_cards

    for ((i = 0; i < ${#p_cards[@]}; i++)); do
        local card=${p_cards[i]}
        p_port=$((start_port_p + i))
        p_ports+=("$p_port")

        kv_transfer_config=$(jq -n \
        --arg kv_connector "TelecclConnector" \
        --arg kv_role "kv_producer" \
        --arg kv_rank "$i" \
        --arg kv_parallel_size "$kv_parallel_size" \
        --arg kv_buffer_size "5e9" \
        '{kv_connector: $kv_connector, kv_role: $kv_role, kv_rank: ($kv_rank | tonumber), kv_parallel_size: ($kv_parallel_size | tonumber), kv_buffer_size: ($kv_buffer_size | tonumber)}')
            
        CUDA_VISIBLE_DEVICES="$card" python3 -m vllm.entrypoints.openai.api_server \
            --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
            --port "$p_port" \
            --max-model-len 4096 \
            --trust-remote-code \
            --kv-transfer-config "$kv_transfer_config" \
            > "$RUN_LOG_DIR/vllm_server_prefill_${card}.log" 2>&1 &

        echo "Started server on card $card with port $p_port"
    done

    for ((i = 0; i < ${#d_cards[@]}; i++)); do
        local card=${d_cards[i]}
        d_port=$((start_port_d + i))
        d_ports+=("$d_port")

        kv_rank=$(( ${#p_cards[@]} + i ))
        kv_transfer_config=$(jq -n \
        --arg kv_connector "TelecclConnector" \
        --arg kv_role "kv_consumer" \
        --arg kv_rank "$kv_rank" \
        --arg kv_parallel_size "$kv_parallel_size" \
        --arg kv_buffer_size "5e9" \
        '{kv_connector: $kv_connector, kv_role: $kv_role, kv_rank: ($kv_rank | tonumber), kv_parallel_size: ($kv_parallel_size | tonumber), kv_buffer_size: ($kv_buffer_size | tonumber)}')
        
        CUDA_VISIBLE_DEVICES="$card" python3 -m vllm.entrypoints.openai.api_server \
            --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
            --port "$d_port" \
            --max-model-len 4096 \
            --trust-remote-code \
            --kv-transfer-config "$kv_transfer_config" \
            > "$RUN_LOG_DIR/vllm_server_decode_${card}.log" 2>&1 &

        echo "Started server on card $card with port $d_port"
    done

    echo -n "Waiting for vLLM servers..."
    for port in "${p_ports[@]}" "${d_ports[@]}"; do 
        wait_for_server "$port"
    done
    echo " All vLLM servers ready."

    # Start proxy server
    echo "Starting proxy server..."
    python3 ./auto_disagg_prefill_proxy_server.py --p_ports "${p_ports[@]}" --d_ports "${d_ports[@]}" --proxy_port 8000 & 
    proxy_pid=$!

    # Wait for proxy server to be ready
    echo -n "Waiting for proxy server..."
    wait_for_server 8000
    echo " Proxy ready (PID: $proxy_pid)"    
    
    # Check if proxy server started successfully
    if ! nc -z localhost 8000; then
        echo "Proxy server failed to start on port 8000"
        return 1  # Return failure
    fi

    return 0  # Return success
}

# Run the Python benchmarking script
run_python_script() {
    local method=$1
    local input=$2
    local output=$3
    local prompts=$4
    local request_rate=$5
    local run_log_dir=$6
    local limit=$7

    local -n c_cards_ref=$8
    local -n p_cards_ref=$9
    local -n d_cards_ref=$10

    if [[ "$method" == "chunked" ]]; then
        local port=8001
        local c_num=${#c_cards_ref[@]}
        result_filename=${method}_${c_num}c_limit${limit}_in${input}_out${output}_${prompts}prompts_${request_rate}rate.json
    elif [[ "$method" == "disagg" ]]; then
        local port=8000
        local p_num=${#p_cards_ref[@]}
        local d_num=${#d_cards_ref[@]}
        result_filename=${method}_${p_num}p${d_num}d_limit${limit}_in${input}_out${output}_${prompts}prompts_${request_rate}rate.json
    else
        echo "Unknown Method: $method"
    fi

    python3 /workspace/deepseek/zer/vllm/benchmarks/benchmark_serving.py \
        --backend vllm \
        --model /workspace/deepseek/zer/DeepSeek-V2-Lite/ \
        --dataset-name "sonnet" \
        --dataset-path "/workspace/deepseek/zer/vllm/benchmarks/sonnet_4x.txt" \
        --sonnet-input-len "$input" \
        --sonnet-output-len "$output" \
        --sonnet-prefix-len 22 \
        --num-prompts "$prompts" \
        --port "$port" \
        --request-rate "$request_rate" \
        --save-result \
        --result-dir "$run_log_dir" \
        --result-filename "$result_filename"
}

# Terminate all python3 processes
pkill -9 python3 || echo "No python3 processes to kill."
sleep 1

for method in "${methods[@]}"; do
    for input in "${inputs[@]}"; do
        for output in "${outputs[@]}"; do
            for prompts in "${prompts[@]}"; do
                for request_rate in "${request_rates[@]}"; do
                    if [ "$method" == "chunked" ]; then
                        if ! check_port_open 8001; then
                            echo "Port 8001 is not open. Starting."
                            if ! start_chunked; then
                                echo "Failed to start chunked servers. Exiting."
                                exit 1
                            fi
                        fi
                        for limit in 0 1; do
                            for card in "${c_cards[@]}"; do
                                set_nvidia_power_limit "$card" "$limit"
                            done
                            run_python_script "$method" "$input" "$output" "$prompts" "$request_rate" "$RUN_LOG_DIR" "$limit" c_cards p_cards d_cards
                        done
                    elif [ "$method" == "disagg" ]; then
                        if ! check_port_open "8000"; then
                            echo "Port 8000 is not open. Starting."
                            if ! start_disagg; then
                                echo "Failed to start disagg servers. Exiting."
                                exit 1
                            fi
                        fi
                        for limit in 1; do
                            for card in "${p_cards[@]}"; do
                                set_nvidia_power_limit "$card" 0
                            done
                            for card in "${d_cards[@]}"; do
                                set_nvidia_power_limit "$card" "$limit"
                            done
                            run_python_script "$method" "$input" "$output" "$prompts" "$request_rate" "$RUN_LOG_DIR" "$limit" c_cards p_cards d_cards
                        done
                    else
                        echo "unknown method"
                    fi
                done
            done
        done
    done
done
