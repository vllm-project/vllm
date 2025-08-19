#!/bin/bash

# This script aims to tune the best server parameter combinations to maximize throughput for given requirement.
# See details in README (benchmarks/auto_tune/README.md).

TAG=$(date +"%Y_%m_%d_%H_%M")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE="$SCRIPT_DIR/../../.."
MODEL="meta-llama/Llama-3.1-8B-Instruct"
SYSTEM="TPU"
TP=1
DOWNLOAD_DIR=""
INPUT_LEN=4000
OUTPUT_LEN=16
MAX_MODEL_LEN=4096
MIN_CACHE_HIT_PCT=0
MAX_LATENCY_ALLOWED_MS=100000000000
NUM_SEQS_LIST="128 256"
NUM_BATCHED_TOKENS_LIST="512 1024 2048 4096"

LOG_FOLDER="$BASE/auto-benchmark/$TAG"
RESULT="$LOG_FOLDER/result.txt"
PROFILE_PATH="$LOG_FOLDER/profile"

echo "result file: $RESULT"
echo "model: $MODEL"

rm -rf $LOG_FOLDER
rm -rf $PROFILE_PATH
mkdir -p $LOG_FOLDER
mkdir -p $PROFILE_PATH

cd "$BASE/vllm"

pip install -q datasets

current_hash=$(git rev-parse HEAD)
echo "hash:$current_hash" >> "$RESULT"
echo "current_hash: $current_hash"

TOTAL_LEN=$((INPUT_LEN + OUTPUT_LEN))
RED='\033[0;31m'
if (( TOTAL_LEN > MAX_MODEL_LEN )); then
    echo -e "${RED}FAILED: INPUT_LEN($INPUT_LEN) + OUTPUT_LEN($OUTPUT_LEN) = $TOTAL_LEN, which is > MAX_MODEL_LEN = $MAX_MODEL_LEN.\033[0m" >&2
    exit 1
fi

best_throughput=0
best_max_num_seqs=0
best_num_batched_tokens=0
best_goodput=0
best_request_rate=0

start_server() {
    local gpu_memory_utilization=$1
    local max_num_seqs=$2
    local max_num_batched_tokens=$3
    local vllm_log=$4
    local profile_dir=$5

    pkill -if vllm

    # Define the common arguments as a bash array.
    # Each argument and its value are separate elements.
    local common_args_array=(
        "$MODEL"
        "--disable-log-requests"
        "--port" "8004"
        "--gpu-memory-utilization" "$gpu_memory_utilization"
        "--max-num-seqs" "$max_num_seqs"
        "--max-num-batched-tokens" "$max_num_batched_tokens"
        "--tensor-parallel-size" "$TP"
        "--enable-prefix-caching"
        "--load-format" "dummy"
        "--download-dir" "$DOWNLOAD_DIR"
        "--max-model-len" "$MAX_MODEL_LEN"
    )

    # Use the array expansion "${common_args_array[@]}"
    # This correctly passes each element as a separate argument.
    if [[ -n "$profile_dir" ]]; then
        # Start server with profiling enabled
        VLLM_USE_V1=1 VLLM_SERVER_DEV_MODE=1 VLLM_TORCH_PROFILER_DIR=$profile_dir \
            vllm serve "${common_args_array[@]}" > "$vllm_log" 2>&1 &
    else
        # Start server without profiling
        VLLM_USE_V1=1 VLLM_SERVER_DEV_MODE=1 \
            vllm serve "${common_args_array[@]}" > "$vllm_log" 2>&1 &
    fi

    # wait for 10 minutes...
    server_started=0
    for i in {1..60}; do
        RESPONSE=$(curl -s -X GET "http://0.0.0.0:8004/health" -w "%{http_code}" -o /dev/stdout)
        STATUS_CODE=$(echo "$RESPONSE" | tail -n 1)
        if [[ "$STATUS_CODE" -eq 200 ]]; then
            server_started=1
            break
        else
            sleep 10
        fi
    done

    if (( ! server_started )); then
        echo "server did not start within 10 minutes. Please check server log at $vllm_log".
        return 1
    else
        return 0
    fi
}

run_benchmark() {
    local max_num_seqs=$1
    local max_num_batched_tokens=$2
    local gpu_memory_utilization=$3
    echo "max_num_seq: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens"
    local vllm_log="$LOG_FOLDER/vllm_log_${max_num_seqs}_${max_num_batched_tokens}.txt"
    echo "vllm_log: $vllm_log"
    echo
    rm -f $vllm_log
    pkill -if vllm

    echo "starting server..."
    # Call start_server without a profile_dir to avoid profiling overhead
    start_server $gpu_memory_utilization $max_num_seqs $max_num_batched_tokens $vllm_log ""
    result=$?
    if [[ "$result" -eq 1 ]]; then
        echo "server failed to start. gpu_memory_utilization:$gpu_memory_utilization, max_num_seqs:$max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens"
    else
        echo "server started."
    fi
    echo

    echo "run benchmark test..."
    meet_latency_requirement=0
    # get a basic qps by using request-rate inf
    bm_log="$LOG_FOLDER/bm_log_${max_num_seqs}_${max_num_batched_tokens}_requestrate_inf.txt"
    prefix_len=$(( INPUT_LEN * MIN_CACHE_HIT_PCT / 100 ))
    adjusted_input_len=$(( INPUT_LEN - prefix_len ))
    # --profile flag is removed from this call
    vllm bench serve \
        --backend vllm \
        --model $MODEL  \
        --dataset-name random \
        --random-input-len $adjusted_input_len \
        --random-output-len $OUTPUT_LEN \
        --ignore-eos \
        --disable-tqdm \
        --request-rate inf \
        --percentile-metrics ttft,tpot,itl,e2el \
        --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
        --num-prompts 1000 \
        --random-prefix-len $prefix_len \
        --port 8004 &> "$bm_log"
    throughput=$(grep "Request throughput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
    e2el=$(grep "P99 E2EL (ms):" "$bm_log" | awk '{print $NF}')
    goodput=$(grep "Request goodput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')

    if (( $(echo "$e2el <= $MAX_LATENCY_ALLOWED_MS" | bc -l) )); then
        meet_latency_requirement=1
        request_rate=inf
    fi

    if (( ! meet_latency_requirement )); then
    # start from request-rate as int(throughput) + 1
        request_rate=$((${throughput%.*} + 1))
        while ((request_rate > 0)); do
            # clear prefix cache
            curl -X POST http://0.0.0.0:8004/reset_prefix_cache
            sleep 5
            bm_log="$LOG_FOLDER/bm_log_${max_num_seqs}_${max_num_batched_tokens}_requestrate_${request_rate}.txt"
            vllm bench serve \
                --backend vllm \
                --model $MODEL  \
                --dataset-name random \
                --random-input-len $adjusted_input_len \
                --random-output-len $OUTPUT_LEN \
                --ignore-eos \
                --disable-tqdm \
                --request-rate $request_rate \
                --percentile-metrics ttft,tpot,itl,e2el \
                --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
                --num-prompts 100 \
                --random-prefix-len $prefix_len \
                --port 8004 &> "$bm_log"
            throughput=$(grep "Request throughput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
            e2el=$(grep "P99 E2EL (ms):" "$bm_log" | awk '{print $NF}')
            goodput=$(grep "Request goodput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
            if (( $(echo "$e2el <= $MAX_LATENCY_ALLOWED_MS" | bc -l) )); then
                meet_latency_requirement=1
                break
            fi
            request_rate=$((request_rate-1))
        done
    fi
    # write the results and update the best result.
    if ((meet_latency_requirement)); then
        echo "max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens, request_rate: $request_rate, e2el: $e2el, throughput: $throughput, goodput: $goodput"
        echo "max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens, request_rate: $request_rate, e2el: $e2el, throughput: $throughput, goodput: $goodput" >> "$RESULT"
        if (( $(echo "$throughput > $best_throughput" | bc -l) )); then
            best_throughput=$throughput
            best_max_num_seqs=$max_num_seqs
            best_num_batched_tokens=$max_num_batched_tokens
            best_goodput=$goodput
            best_request_rate=$request_rate
        fi
    else
        echo "max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens does not meet latency requirement ${MAX_LATENCY_ALLOWED_MS}"
        echo "max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens does not meet latency requirement ${MAX_LATENCY_ALLOWED_MS}" >> "$RESULT"
    fi

    echo "best_max_num_seqs: $best_max_num_seqs, best_num_batched_tokens: $best_num_batched_tokens, best_throughput: $best_throughput"

    pkill -if vllm
    sleep 10
    printf '=%.0s' $(seq 1 20)
    return 0
}

read -r -a num_seqs_list <<< "$NUM_SEQS_LIST"
read -r -a num_batched_tokens_list <<< "$NUM_BATCHED_TOKENS_LIST"

# first find out the max gpu-memory-utilization without HBM OOM.
gpu_memory_utilization=0.98
find_gpu_memory_utilization=0
while (( $(echo "$gpu_memory_utilization >= 0.9" | bc -l) )); do
    # Pass empty string for profile_dir argument
    start_server $gpu_memory_utilization "${num_seqs_list[-1]}" "${num_batched_tokens_list[-1]}" "$LOG_FOLDER/vllm_log_gpu_memory_utilization_$gpu_memory_utilization.log" ""
    result=$?
    if [[ "$result" -eq 0 ]]; then
        find_gpu_memory_utilization=1
        break
    else
        gpu_memory_utilization=$(echo "$gpu_memory_utilization - 0.01" | bc)
    fi
done

if [[ "$find_gpu_memory_utilization" -eq 1 ]]; then
    echo "Using gpu_memory_utilization=$gpu_memory_utilization to serve model."
else
    echo "Cannot find a proper gpu_memory_utilization over 0.9 to serve the model, please check logs in $LOG_FOLDER."
    exit 1
fi

for num_seqs in "${num_seqs_list[@]}"; do
    for num_batched_tokens in "${num_batched_tokens_list[@]}"; do
        run_benchmark $num_seqs $num_batched_tokens $gpu_memory_utilization
    done
done
echo "finish permutations"

# =================================================================================
# FINAL PROFILING RUN FOR THE BEST CONFIGURATION
# =================================================================================
if (( $(echo "$best_throughput > 0" | bc -l) )); then
    echo
    echo "Benchmark tuning finished. Now running profiling on the best configuration found..."
    echo "Best config: max_num_seqs: $best_max_num_seqs, max_num_batched_tokens: $best_num_batched_tokens, throughput: $best_throughput"
    echo

    vllm_log="$LOG_FOLDER/vllm_log_BEST_PROFILE.txt"
    bm_log="$LOG_FOLDER/bm_log_BEST_PROFILE.txt"

    # Start server with the best params and profiling ENABLED
    echo "Starting server for profiling..."
    start_server $gpu_memory_utilization $best_max_num_seqs $best_num_batched_tokens "$vllm_log" "$PROFILE_PATH"

    # Run benchmark with the best params and the --profile flag
    echo "Running benchmark with profiling..."
    prefix_len=$(( INPUT_LEN * MIN_CACHE_HIT_PCT / 100 ))
    adjusted_input_len=$(( INPUT_LEN - prefix_len ))
    vllm bench serve \
        --backend vllm \
        --model $MODEL \
        --dataset-name random \
        --random-input-len $adjusted_input_len \
        --random-output-len $OUTPUT_LEN \
        --ignore-eos \
        --disable-tqdm \
        --request-rate $best_request_rate \
        --percentile-metrics ttft,tpot,itl,e2el \
        --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
        --num-prompts 100 \
        --random-prefix-len $prefix_len \
        --port 8004 \
        --profile &> "$bm_log"
else
    echo "No configuration met the latency requirements. Skipping final profiling run."
fi
pkill -if vllm
echo "best_max_num_seqs: $best_max_num_seqs, best_num_batched_tokens: $best_num_batched_tokens, best_throughput: $best_throughput, profile saved in: $PROFILE_PATH"
echo "best_max_num_seqs: $best_max_num_seqs, best_num_batched_tokens: $best_num_batched_tokens, best_throughput: $best_throughput, profile saved in: $PROFILE_PATH" >> "$RESULT"
