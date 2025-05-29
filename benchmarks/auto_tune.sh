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

TAG=$(date +"%Y_%m_%d_%H_%M")
BASE=""
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DOWNLOAD_DIR=""
INPUT_LEN=4000
OUTPUT_LEN=16
MIN_CACHE_HIT_PCT_PCT=0
MAX_LATENCY_ALLOWED_MS=100000000000

LOG_FOLDER="$BASE/auto-benchmark/$TAG"
RESULT="$LOG_FOLDER/result.txt"

echo "result file$ $RESULT"
echo "model: $MODEL"
echo

rm -rf $LOG_FOLDER
mkdir -p $LOG_FOLDER

cd "$BASE/vllm"
# create sonnet-4x.txt so that we can sample 2048 tokens for input
echo "" > benchmarks/sonnet_4x.txt
for _ in {1..4}
do
cat benchmarks/sonnet.txt >> benchmarks/sonnet_4x.txt
done

pip install datasets

current_hash=$(git rev-parse HEAD)
echo "hash:$current_hash" >> "$RESULT"
echo "current_hash: $current_hash"

best_throughput=0
best_max_num_seqs=0
best_num_batched_tokens=0
best_goodput=0
run_benchmark() {
    local max_num_seqs=$1
    local max_num_batched_tokens=$2
    echo "max_num_seq: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens"
    local vllm_log="$LOG_FOLDER/vllm_log_${max_num_seqs}_${max_num_batched_tokens}.txt"
    echo "vllm_log: $vllm_log"
    echo
    rm -f $vllm_log

    # start the server
    VLLM_USE_V1=1 VLLM_SERVER_DEV_MODE=1 vllm serve $MODEL \
        --disable-log-requests \
        --port 8004 \
        --gpu-memory-utilization 0.98 \
        --max-num-seqs $max_num_seqs \
        --max-num-batched-tokens $max_num_batched_tokens \
        --tensor-parallel-size 1 \
        --enable-prefix-caching \
        --load-format dummy \
        --download-dir $DOWNLOAD_DIR \
        --max-model-len $(( INPUT_LEN+OUTPUT_LEN )) > "$vllm_log" 2>&1 &
    echo "wait for 10 minutes.."
    echo
    # wait for 10 minutes...
    server_started=0
    for i in {1..60}; do        
        if grep -Fq "Application startup complete" "$vllm_log"; then
            echo "Application started"
            server_started=1
            break
        else
            # echo "wait for 10 seconds..."
            sleep 10
        fi
    done
 
    if (( ! server_started )); then
        echo "server did not start within 10 minutes, terminate the benchmarking. Please check server log at $vllm_log"
        echo "pkill -f vllm"
        echo
        pkill vllm
        sleep 10
        return 1
    fi
    
    echo "run benchmark test..."
    echo
    meet_latency_requirement=0
    # get a basic qps by using request-rate inf
    bm_log="$LOG_FOLDER/bm_log_${max_num_seqs}_${max_num_batched_tokens}_requestrate_inf.txt"
    prefix_len=$(( INPUT_LEN * MIN_CACHE_HIT_PCT / 100 ))
    python benchmarks/benchmark_serving.py \
        --backend vllm \
        --model $MODEL  \
        --dataset-name sonnet \
        --dataset-path benchmarks/sonnet_4x.txt \
        --sonnet-input-len $INPUT_LEN \
        --sonnet-output-len $OUTPUT_LEN \
        --ignore-eos \
        --disable-tqdm \
        --request-rate inf \
        --percentile-metrics ttft,tpot,itl,e2el \
        --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
        --num-prompts 100 \
        --sonnet-prefix-len $prefix_len \
        --port 8004 > "$bm_log"
    through_put=$(grep "Request throughput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
    e2el=$(grep "P99 E2EL (ms):" "$bm_log" | awk '{print $NF}')
    goodput=$(grep "Request goodput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')

    if (( $(echo "$e2el <= $MAX_LATENCY_ALLOWED_MS" | bc -l) )); then
        meet_latency_requirement=1
    fi

    if (( ! meet_latency_requirement )); then
    # start from request-rate as int(through_put) + 1
        request_rate=$((${through_put%.*} + 1))
        while ((request_rate > 0)); do
            # clear prefix cache
            curl -X POST http://0.0.0.0:8004/reset_prefix_cache
            sleep 5
            bm_log="$LOG_FOLDER/bm_log_${max_num_seqs}_${max_num_batched_tokens}_requestrate_${request_rate}.txt"
            python benchmarks/benchmark_serving.py \
                --backend vllm \
                --model $MODEL  \
                --dataset-name sonnet \
                --dataset-path benchmarks/sonnet_4x.txt \
                --sonnet-input-len $INPUT_LEN \
                --sonnet-output-len $OUTPUT_LEN \
                --ignore_eos \
                --disable-tqdm \
                --request-rate $request_rate \
                --percentile-metrics ttft,tpot,itl,e2el \
                --goodput e2el:$MAX_LATENCY_ALLOWED_MS \
                --num-prompts 100 \
                --sonnet-prefix-len $prefix_len \
                --port 8004 > "$bm_log"
            through_put=$(grep "Request throughput (req/s):" "$bm_log" | sed 's/[^0-9.]//g')
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
        echo "max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens, request_rate: $request_rate, e2el: $e2el, through put: $through_put, goodput: $goodput"
        echo "max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens, request_rate: $request_rate, e2el: $e2el, through put: $through_put, goodput: $goodput" >> "$RESULT"
        if (( $(echo "$through_put > $best_throughput" | bc -l) )); then
            best_throughput=$through_put
            best_max_num_seqs=$max_num_seqs
            best_num_batched_tokens=$max_num_batched_tokens
            best_goodput=$goodput
        fi
    else
        echo "max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens does not meet latency requirement ${MAX_LATENCY_ALLOWED_MS}"
        echo "max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens does not meet latency requirement ${MAX_LATENCY_ALLOWED_MS}" >> "$RESULT"
    fi

    echo "best_max_num_seqs: $best_max_num_seqs, best_num_batched_tokens: $best_num_batched_tokens, best_throughput: $best_throughput"

    echo "pkill -f vllm"
    echo
    pkill vllm
    sleep 10
    rm -f $vllm_log
    printf '=%.0s' $(seq 1 20)
    return 0
}


num_seqs_list="128 256"
num_batched_tokens_list="512 1024 2048 4096"
for num_seqs in $num_seqs_list; do
    for num_batched_tokens in $num_batched_tokens_list; do
        run_benchmark $num_seqs $num_batched_tokens
        exit 0
    done
done
echo "finish permutations"
echo "best_max_num_seqs: $best_max_num_seqs, best_num_batched_tokens: $best_num_batched_tokens, best_throughput: $best_throughput"
echo "best_max_num_seqs: $best_max_num_seqs, best_num_batched_tokens: $best_num_batched_tokens, best_throughput: $best_throughput" >> "$RESULT"

