#!/bin/bash

# This script aims to tune the best server parameter combinations to maximize throughput for given requirement. 
# The current server parameter combination is  max_num_seqs and max_num_batched_tokens
# It also supports additional requirement: e2e latency and prefix cache. 

# Pre-requisite:
# 1. Checkout to your branch, install/ update the correct running env. For TPU, activate conda env and install the corresponding torch, xla version. 
# 2. If the model is customized, replace the MODEL's config with the customized config.
# 3. Set variables (ALL REQUIRED)
#   BASE: your directory for vllm repo
#   MODEL: the model served by vllm
#   DOWNLOAD_DIR: directory to download and load model weights.
#   INPUT_LEN: request input len
#   OUTPUT_LEN: request output len
#   MIN_CACHE_HIT_PCT: prefix cache rate
#   MAX_LATENCY_ALLOWED_MS: (e2e) latency requirement. If there's no latency requirement, set it to a large number like 1000000000
# 4. Run the script, it might take a long time, you can use tmux to avoid the script stop if disconnection happens.
# 5. The final result will be saved in RESULT file. 


# Example use cases 
# 1. Given input_len=1800, output_len=20, what's the best max_num_seqs and max_num_batched_tokens to get highest throughput?
# Use INPUT_LEN=1800,  OUTPUT_LEN=20, MIN_CACHE_HIT_PCT=0, MAX_LATENCY_ALLOWED_MS=100000000000
# 2. If we have latency requirement to be lower than 500ms, what's the best server parameter?
# Use INPUT_LEN=1800,  OUTPUT_LEN=20, MIN_CACHE_HIT_PCT=0, MAX_LATENCY_ALLOWED_MS=500
# 3. If we want to reach 60% prefix cache, what's the best server parameter? 
# Use INPUT_LEN=1800,  OUTPUT_LEN=20, MIN_CACHE_HIT_PCT=60, MAX_LATENCY_ALLOWED_MS=500

# --- Configuration ---
# Script tag for organization
readonly TAG=$(date +"%Y_%m_%d_%H_%M")
# Base directory for logs and code (using $HOME is generally safer than ~ in scripts)
readonly BASE_DIR="$HOME"
# VLLM code directory
readonly VLLM_DIR="$BASE_DIR/vllm"
# Log folder for this run
readonly LOG_FOLDER="$BASE_DIR/auto_tune/$TAG"
# Main result file path
readonly RESULT_FILE="$LOG_FOLDER/result.txt"
# Model identifier
readonly MODEL="Qwen/Qwen2.5-32B"
# Tensor Parallelism size
readonly TP=4
# Input and Output lengths for benchmarking
readonly INPUT_LEN=1800
readonly OUTPUT_LEN=128
# Minimum required prefix cache hit percentage (used to calculate prefix length)
readonly MIN_CACHE_HIT_PERCENT=0 # Set between 0 and 100
# Maximum allowed E2E latency in milliseconds (set high if no strict requirement)
readonly MAX_LATENCY_ALLOWED_MS=1000000000 # Example: 1 second = 1000 ms
# Optional: Specify download directory if using a mounted disk for models
readonly DOWNLOAD_DIR="" # Example: "/mnt/models" - leave empty if not needed
# Port for the vLLM server
readonly VLLM_PORT=8004
# Conda environment name
readonly CONDA_ENV_NAME="qwen" # Changed from "vllm" in original comment, adjust if needed
# Path to the benchmark dataset file
readonly DATASET_PATH="$VLLM_DIR/benchmarks/sonnet_4x.txt" # Assumes sonnet_4x.txt exists here
# Number of prompts to use in benchmark_serving.py
readonly NUM_PROMPTS=1000
# Time (in seconds) to wait for the server to start (only used in background mode)
readonly SERVER_WAIT_TIMEOUT=600 # 10 minutes
# --- End Configuration ---

# --- Global Variables ---
# For tracking best result (only used in background mode)
best_throughput=0
best_goodput=0
best_max_num_seqs=0
best_num_batched_tokens=0
# For managing server PID (only used in background mode)
server_pid=""
# --- End Global Variables ---

# --- Logging Functions ---
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# --- Helper Functions ---

setup_environment() {
    log_info "Setting up environment..."
    export VLLM_USE_V1=1
    # if ! command -v conda &> /dev/null; then
        # if [[ -f "$HOME/miniconda3/bin/activate" ]]; then
             # . "$HOME/miniconda3/bin/activate"
        # else
            # log_error "Conda command not found. Please ensure Conda is installed and initialized."
            # exit 1
        # fi
    # fi
    # conda init
    # conda activate "$CONDA_ENV_NAME"
    # if [[ $? -ne 0 ]]; then
        # log_error "Failed to activate conda environment '$CONDA_ENV_NAME'." 
        # exit 1
    # else
        # log_info "Activated conda environment: $CONDA_ENV_NAME"
    # fi

    log_info "Changing directory to $VLLM_DIR"
    cd "$VLLM_DIR"
    if [[ $? -ne 0 ]]; then
        log_error "Failed to change directory to $VLLM_DIR. Does it exist? Attempting to continue..."
        exit 1
    fi
}

# --- Server Management Functions ---

# ** Starts server in FOREGROUND and waits **
start_vllm_server_foreground() {
    local max_num_seqs=$1
    local max_num_batched_tokens=$2
    local vllm_log_file=$3

    log_info "Starting vLLM server in FOREGROUND (Max Seqs: $max_num_seqs, Max Batched Tokens: $max_num_batched_tokens)... Log: $vllm_log_file"
    log_info "Script will WAIT here until server exits. Press Ctrl+C to terminate manually."
    rm -f "$vllm_log_file" # Clean previous log

    # Round max_model_len up to nearest 64
    local rounded_max_model_len=$(( ((INPUT_LEN + OUTPUT_LEN + 63) / 64) * 64 ))

    local server_cmd=(
        vllm serve "$MODEL"
        --disable-log-requests
        --port "$VLLM_PORT"
        --gpu-memory-utilization 0.98
        --max-num-seqs "$max_num_seqs"
        --max-num-batched-tokens "$max_num_batched_tokens"
        --tensor-parallel-size "$TP"
        --no-enable-prefix-caching
        --load-format dummy
        --max-model-len "$rounded_max_model_len"
    )
    if [[ -n "$DOWNLOAD_DIR" ]]; then
        server_cmd+=(--download-dir "$DOWNLOAD_DIR")
    fi

    # Execute in foreground, redirect output
    "${server_cmd[@]}" > "$vllm_log_file" 2>&1
    local server_exit_status=$?

    if [[ $server_exit_status -ne 0 ]]; then
        log_error "vLLM server command finished with error status: $server_exit_status. Check log: $vllm_log_file"
        return 1
    else
        log_info "vLLM server command finished successfully."
        return 0
    fi
}

# ** Starts server in BACKGROUND and waits for it to be ready **
start_vllm_server_background() {
    local max_num_seqs=$1
    local max_num_batched_tokens=$2
    local vllm_log_file=$3

    log_info "Starting vLLM server in BACKGROUND (Max Seqs: $max_num_seqs, Max Batched Tokens: $max_num_batched_tokens)... Log: $vllm_log_file"
    rm -f "$vllm_log_file" # Clean previous log

    # Round max_model_len up to nearest 64
    local rounded_max_model_len=$(( ((INPUT_LEN + OUTPUT_LEN + 63) / 64) * 64 ))

    local server_cmd=(
        vllm serve "$MODEL"
        --disable-log-requests
        --port "$VLLM_PORT"
        --gpu-memory-utilization 0.98
        --max-num-seqs "$max_num_seqs"
        --max-num-batched-tokens "$max_num_batched_tokens"
        --tensor-parallel-size "$TP"
        --no-enable-prefix-caching
        --load-format dummy
        --max-model-len "$rounded_max_model_len"
    )
    if [[ -n "$DOWNLOAD_DIR" ]]; then
        server_cmd+=(--download-dir "$DOWNLOAD_DIR")
    fi

    # Execute in background, redirect output
    "${server_cmd[@]}" > "$vllm_log_file" 2>&1 &
    # Check immediate launch status
    if [[ $? -ne 0 ]]; then
        log_error "Failed to launch vLLM server command itself."
        server_pid="" # Ensure pid is clear
        return 1 # Indicate failure
    fi
    # Store PID of background process
    server_pid=$!
    log_info "Server background process potentially started with PID: $server_pid"

    log_info "Waiting up to $SERVER_WAIT_TIMEOUT seconds for server startup confirmation..."
    local wait_time=0
    while [[ $wait_time -lt $SERVER_WAIT_TIMEOUT ]]; do
        # Check if the process is still running
        if ! ps -p "$server_pid" > /dev/null; then
             log_error "Server background process $server_pid disappeared unexpectedly. Check log: $vllm_log_file"
             server_pid="" # Clear PID as it's gone
             return 1 # Failure
        fi
        # Check log file for success message
        if grep -Fq "Application startup complete" "$vllm_log_file"; then
            log_info "Server startup complete (PID: $server_pid)."
            return 0 # Success
        fi
        sleep 10
        wait_time=$((wait_time + 10))
    done

    log_error "Server failed to confirm startup within $SERVER_WAIT_TIMEOUT seconds (PID: $server_pid). Check log: $vllm_log_file"
    # Try to kill the potentially hanging process
    stop_vllm_server # Call stop function to attempt cleanup
    return 1 # Failure
}

# ** Stops the server (primarily used in background mode) **
stop_vllm_server() {
    # Check if server_pid is set and corresponds to a running process
    if [[ -n "$server_pid" ]] && ps -p "$server_pid" > /dev/null; then
        log_info "Stopping server process with PID: $server_pid"
        kill "$server_pid" # Try graceful shutdown
        sleep 5
        if ps -p "$server_pid" > /dev/null; then
           log_info "Server PID $server_pid still alive, sending SIGKILL..."
           kill -9 "$server_pid"
           sleep 2
        fi
    else
        # Only log if we actually expected a PID
        if [[ -n "$server_pid" ]]; then
             log_info "Server PID $server_pid not found or already stopped."
        fi
        # Optional: Add pkill fallback if needed, but less precise
        # log_info "Attempting pkill as fallback..."
        # pkill -f "vllm serve.*--port $VLLM_PORT"
    fi
    server_pid="" # Reset PID after attempting to stop
    log_info "Server stop attempt finished."
}


# --- Benchmark and Parsing Functions (only used in background mode) ---

run_single_benchmark() {
    local request_rate=$1
    local bm_log_file=$2
    local max_num_batched_tokens=$3
    local max_num_seqs=$4

    log_info "Running benchmark (Req Rate: $request_rate," \
             " Max Seqs: $max_num_seqs," \
             " Max Batched Tokens: $max_num_batched_tokens)... Log: $bm_log_file"
    python benchmarks/benchmark_serving.py \
        --backend vllm \
        --model "$MODEL" \
        --dataset-name sonnet \
        --dataset-path "$DATASET_PATH" \
        --sonnet-input-len "$INPUT_LEN" \
        --sonnet-output-len "$OUTPUT_LEN" \
        --sonnet-prefix-len 0 \
        --ignore-eos \
        --disable-tqdm \
        --request-rate "$request_rate" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --goodput "e2el:${MAX_LATENCY_ALLOWED_MS}" \
        --num-prompts "$NUM_PROMPTS" \
        --port "$VLLM_PORT" > "$bm_log_file"
    local bench_status=$?
    if [[ $bench_status -ne 0 ]]; then
        log_error "Benchmark script failed (Exit Status: $bench_status) for Req Rate: $request_rate. Check log: $bm_log_file"
        return 1
    fi
    log_info "Benchmark finished successfully for Req Rate: $request_rate."
    return 0
}

parse_metric() {
    local log_file=$1
    local pattern=$2
    local value
    if [[ ! -r "$log_file" ]]; then log_error "Cannot read log: $log_file"; echo "0"; return; fi
    if grep -oP 'foobar' <<< '' &> /dev/null; then
       value=$(grep -oP "${pattern}\s*:\s*\K[0-9.]+" "$log_file" | head -n 1)
    else
       if [[ "$pattern" == "Request throughput" ]]; then value=$(grep "Request throughput (req/s):" "$log_file" | sed 's/[^0-9.]//g' | head -n 1);
       elif [[ "$pattern" == "Mean E2EL" ]]; then value=$(grep "Mean E2EL (ms):" "$log_file" | awk '{print $NF}' | head -n 1);
       elif [[ "$pattern" == "Request goodput" ]]; then value=$(grep "Request goodput (req/s):" "$log_file" | sed 's/[^0-9.]//g' | head -n 1);
       elif [[ "$pattern" == "Output token throughput" ]]; then value=$(grep "Output token throughput (tok/s):" "$log_file" | sed 's/[^0-9.]//g' | head -n 1);
       elif [[ "$pattern" == "Total Token throughput" ]]; then value=$(grep "Total Token throughput (tok/s):" "$log_file" | sed 's/[^0-9.]//g' | head -n 1);
       else value=""; fi
    fi
    echo "${value:-0}"
}

# --- Benchmark Trial Function (Only for Background Mode) ---

run_background_benchmark_trial() {
    local max_num_batched_tokens=$1
    local max_num_seqs=$2
    local vllm_log_file="$LOG_FOLDER/vllm_log_${max_num_batched_tokens}_${max_num_seqs}.txt"
    local trial_throughput=0
    local trial_goodput=0
    local trial_output_token_throughput=0
    local trial_total_token_throughput=0
    local trial_e2el=-1
    local trial_request_rate="N/A"
    local meet_latency_requirement=0

    log_info "===== Starting Background Trial: Max Seqs=$max_num_seqs, Max Batched Tokens=$max_num_batched_tokens ====="

    # Ensure any previous server instance is stopped (relies on global server_pid)
    stop_vllm_server

    # Start server in BACKGROUND
    if ! start_vllm_server_background "$max_num_seqs" "$max_num_batched_tokens" "$vllm_log_file"; then
        log_error "Failed to start server for this trial. Skipping."
        printf "TRIAL FAILED (Server Start): max_num_seqs: %s, max_num_batched_tokens: %s\n" \
            "$max_num_seqs" "$max_num_batched_tokens" >> "$RESULT_FILE"
        return 1 # Indicate trial failure
    fi

    # --- Benchmarking Logic (only runs if server started ok) ---
    local bm_log_inf="$LOG_FOLDER/bm_log_${max_num_batched_tokens}_${max_num_seqs}_requestrate_inf.txt"
    local benchmark_succeeded=0 # Flag
    if run_single_benchmark "inf" "$bm_log_inf" "$max_num_batched_tokens" "$max_num_seqs"; then
        benchmark_succeeded=1
        trial_throughput=$(parse_metric "$bm_log_inf" "Request throughput")
        trial_e2el=$(parse_metric "$bm_log_inf" "Mean E2EL")
        trial_goodput=$(parse_metric "$bm_log_inf" "Request goodput")
        trial_output_token_throughput=$(parse_metric "$bm_log_inf" "Output token throughput")
        trial_total_token_throughput=$(parse_metric "$bm_log_inf" "Total Token throughput")
        trial_request_rate="inf"
        log_info "Inf Rate Results -" \
                 " Request throughput (req/s): ${trial_throughput}," \
                 " Request goodput (req/s): ${trial_goodput}" \
                 " Output token throughput (tok/s): ${trial_output_token_throughput}," \
                 " Total Token throughput (tok/s): ${trial_total_token_throughput}" \
                 " Mean E2EL (ms): ${trial_e2el},"

        if (( $(echo "$trial_e2el > 0 && $trial_e2el <= $MAX_LATENCY_ALLOWED_MS" | bc -l) )); then
            meet_latency_requirement=1
            log_info "Latency requirement met with request rate 'inf'."
        else
            log_info "Latency requirement NOT met with 'inf' rate. Searching..."
            local request_rate_search=$((${trial_throughput%.*} + 1))
            request_rate_search=$((request_rate_search > 0 ? request_rate_search : 1))
            while ((request_rate_search > 0)); do
                local bm_log_timed="$LOG_FOLDER/bm_log_${max_num_seqs}_${max_num_batched_tokens}_requestrate_${request_rate_search}.txt"
                if ! run_single_benchmark "$request_rate_search" "$bm_log_timed" "$max_num_batched_tokens" "$max_num_seqs"; then
                    log_error "Benchmark failed for rate $request_rate_search. Stopping search."
                    meet_latency_requirement=0; break
                fi
                local current_e2el=$(parse_metric "$bm_log_timed" "Mean E2EL")
                log_info "Tested Rate: $request_rate_search -> E2EL: ${current_e2el}ms"
                if (( $(echo "$current_e2el > 0 && $current_e2el <= $MAX_LATENCY_ALLOWED_MS" | bc -l) )); then
                    log_info "Latency requirement met at rate $request_rate_search."
                    meet_latency_requirement=1
                    trial_throughput=$(parse_metric "$bm_log_timed" "Request throughput")
                    trial_goodput=$(parse_metric "$bm_log_timed" "Request goodput")
                    trial_output_token_throughput=$(parse_metric "$bm_log_timed" "Output token throughput")
                    trial_total_token_throughput=$(parse_metric "$bm_log_timed" "Total Token throughput")
                    trial_e2el=$current_e2el
                    trial_request_rate=$request_rate_search
                    break
                fi
                request_rate_search=$((request_rate_search - 1))
            done
            if (( ! meet_latency_requirement )); then log_info "Could not find rate meeting latency req."; fi
        fi
    else
         log_error "Initial benchmark run (inf rate) failed. Skipping detailed results for this trial."
         trial_result_line="TRIAL FAILED (Inf Benchmark): max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens"
    fi # End benchmark run check

    # --- Log results and update best (only if initial benchmark ran) ---
    if (( benchmark_succeeded )); then
        local trial_result_line="max_num_seqs: $max_num_seqs,"
        trial_result_line+=" max_num_batched_tokens: $max_num_batched_tokens,"
        trial_result_line+=" request_rate: $trial_request_rate,"
        trial_result_line+=" e2el: $trial_e2el,"
        trial_result_line+=" throughput: $trial_throughput,"
        trial_result_line+=" goodput: $trial_goodput"
        trial_result_line+=" output_token_throughput: $trial_output_token_throughput,"
        trial_result_line+=" total_token_throughput: $trial_total_token_throughput"
        if (( meet_latency_requirement )) && (( $(echo "$trial_throughput > $best_throughput" | bc -l) )); then
             log_info ">>> New best throughput found! ($trial_throughput > $best_throughput) <<<"
             best_throughput=$trial_throughput; best_goodput=$trial_goodput
             best_max_num_seqs=$max_num_seqs; best_num_batched_tokens=$max_num_batched_tokens
             trial_result_line+=" (NEW BEST)"
        elif (( ! meet_latency_requirement )); then
             trial_result_line+=" (LATENCY REQ. NOT MET)"
        fi
        log_info "Trial Result: $trial_result_line"
        printf "%s\n" "$trial_result_line" >> "$RESULT_FILE"
    else
        # Log the failure line if benchmark didn't even run
        printf "%s\n" "$trial_result_line" >> "$RESULT_FILE"
    fi

    # Stop the server (important!)
    stop_vllm_server

    log_info "===== Finished Background Trial: Max Seqs=$max_num_seqs, Max Batched Tokens=$max_num_batched_tokens ====="
    printf '=%.0s' $(seq 1 80); echo
    return 0
}

# --- Main Execution ---
main() {
    log_info "Starting vLLM Script"
    log_info "Tag: $TAG"
    log_info "Model: $MODEL"
    log_info "Result File: $RESULT_FILE"

    mkdir -p "$LOG_FOLDER"
    if [[ $? -ne 0 ]]; then log_error "Failed to create log folder: $LOG_FOLDER. Exiting."; exit 1; fi
    log_info "Log folder ready: $LOG_FOLDER"

    setup_environment

    # Define parameter ranges (used for background mode, defaults for foreground)
    local num_seqs_list="128 256"
    # For Qwen-32B, it should be >= 2048
    local num_batched_tokens_list="512 1024 2048 4096"
    # Use the first value as default for foreground mode if needed
    local default_num_seqs=$(echo $num_seqs_list | awk '{print $1}')
    local default_num_batched_tokens=$(echo $num_batched_tokens_list | awk '{print $1}')

    # --- Record git hash and headers (only needed for background/benchmark mode) ---
    local current_hash=$(git rev-parse HEAD 2>/dev/null || echo "N/A")
    log_info "Current vLLM git hash: $current_hash"
    printf "Run Tag: %s\n" "$TAG" > "$RESULT_FILE"
    printf "Model: %s\n" "$MODEL" >> "$RESULT_FILE"
    printf "TP: %s, InputLen: %s, OutputLen: %s\n" "$TP" "$INPUT_LEN" "$OUTPUT_LEN" >> "$RESULT_FILE"
    printf "Max Latency Allowed (ms): %s\n" "$MAX_LATENCY_ALLOWED_MS" >> "$RESULT_FILE"
    printf "Timestamp: %s\n" "$(date)" >> "$RESULT_FILE"
    printf "vLLM Git Hash: %s\n" "$current_hash" >> "$RESULT_FILE"
    echo "----------------------------------------" >> "$RESULT_FILE"
    # --- End Header ---

    for num_seqs in $num_seqs_list; do
        for num_batched_tokens in $num_batched_tokens_list; do
            run_background_benchmark_trial "$num_batched_tokens" "$num_seqs" 
        done
    done

    log_info "All benchmark permutations finished."
    log_info "----------------------------------------"
    log_info "Best Configuration Found:"
    log_info "  Max Num Seqs: $best_max_num_seqs"
    log_info "  Max Num Batched Tokens: $best_num_batched_tokens"
    log_info "  Best Throughput (req/s): $best_throughput"
    log_info "  Goodput at Best Throughput (req/s): $best_goodput"
    log_info "----------------------------------------"

    echo "----------------------------------------" >> "$RESULT_FILE"
    echo "Best Overall Result:" >> "$RESULT_FILE"
    printf "best_max_num_seqs: %s\n" "$best_max_num_seqs" >> "$RESULT_FILE"
    printf "best_num_batched_tokens: %s\n" "$best_num_batched_tokens" >> "$RESULT_FILE"
    printf "best_throughput: %s\n" "$best_throughput" >> "$RESULT_FILE"
    printf "best_goodput: %s\n" "$best_goodput" >> "$RESULT_FILE"
    log_info "Script finished (background mode)."
    exit 0
}

# Execute the main function
main