#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# EPD (Encoder-Prefill-Decode) Correctness Test
# 
# This script tests that EPD disaggregation produces the same outputs as baseline.
# It runs:
# 1. Baseline: Single vLLM instance
# 2. EPD: 1E + 1PD setup
# 3. Baseline for (E + P + D): 1P + 1D vLLM instances disagg
# 4. EPD: 1E + 1P + 1D setup

# For GPU usage

# set -xe

# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

# Model to test
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

# Set 1 to use multimodal prompts; else to use text-only
USE_MM_PROMPTS="${USE_MM_PROMPTS:-1}"
MM_FLAG=""
if [ $USE_MM_PROMPTS = "1" ]; then
    MM_FLAG="--use_mm_prompts"
fi

# GPU configuration
GPU_E="${GPU_E:-0}"
GPU_P="${GPU_P:-1}"
GPU_D="${GPU_D:-2}"
GPU_SINGLE="${GPU_SINGLE:-$GPU_P}"
GPU_PD="${GPU_PD:-$GPU_P}"

# Port
ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
PREFILL_DECODE_PORT="${PREFILL_DECODE_PORT:-19537}"
ENDPOINT_PORT="${ENDPOINT_PORT:-10001}"

# Storage path for encoder cache
EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache_test}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

# Output file for baseline comparison and logs
LOG_PATH="${LOG_PATH:-/tmp}"
BASELINE_FILE="${BASELINE_FILE:-/tmp/vllm_baseline.txt}"
BASELINE_PD_FILE="${BASELINE_PD_FILE:-/tmp/vllm_epd_baseline.txt}"

mkdir -p $LOG_PATH

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

# Wait for server to be ready
wait_for_server() {
    local port=$1
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:${port}/v1/chat/completions > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

# Cleanup function
cleanup_instances() {
    echo "Cleaning up any running vLLM instances..."
    pkill -f "vllm serve" || true
    pkill -f "disagg_epd_proxy.py" || true
    sleep 2
}

# Function to run baseline (single instance)
run_baseline() {
    echo "================================"
    echo "Running BASELINE (single instance)"
    echo "================================"
    
    cleanup_instances
    rm -rf "$EC_SHARED_STORAGE_PATH"
    
    local PORT=$ENDPOINT_PORT
    
    # Start baseline instance
    echo "Starting baseline instance on GPU $GPU_SINGLE, port $PORT"
    CUDA_VISIBLE_DEVICES="$GPU_SINGLE" vllm serve "$MODEL" \
        --port $PORT \
        --enforce-eager \
        --gpu-memory-utilization 0.7 \
        --max-num-seqs 128 \
        --allowed-local-media-path ${GIT_ROOT}/tests/v1/ec_connector/integration \
        > $LOG_PATH/baseline.log 2>&1 &
    
    local BASELINE_PID=$!
    
    # Wait for baseline to start
    echo "Waiting for baseline instance to start..."
    wait_for_server $PORT

    curl http://127.0.0.1:$PORT/v1/models
    echo ""
    
    # Run test in baseline mode
    echo "Running baseline..."

    python "${GIT_ROOT}/tests/v1/ec_connector/integration/test_epd_correctness.py" \
        --service_url "http://localhost:$PORT" \
        --model_name "$MODEL" \
        --mode baseline \
        --baseline_file "$BASELINE_FILE" \
        $MM_FLAG
    
    # Cleanup baseline
    echo "Stopping baseline instance..."
    kill $BASELINE_PID 2>/dev/null || true
    sleep 2
    cleanup_instances
}

# Function to run EPD with 1E + 1PD
run_epd_1e_1pd() {
    echo "================================"
    echo "Running EPD (1E + 1PD)"
    echo "================================"
    
    cleanup_instances
    rm -rf "$EC_SHARED_STORAGE_PATH"
    mkdir -p "$EC_SHARED_STORAGE_PATH"
    
    local ENCODE_PORT=$ENCODE_PORT
    local PREFILL_DECODE_PORT=$PREFILL_DECODE_PORT
    local PROXY_PORT=$ENDPOINT_PORT
    
    declare -a PIDS=()
    
    # Start encoder instance
    echo "Starting encoder instance on GPU $GPU_E, port $ENCODE_PORT"
    CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
        --port $ENCODE_PORT \
        --enforce-eager \
        --gpu-memory-utilization 0.01 \
        --enable-request-id-headers \
        --no-enable-prefix-caching \
        --max-num-batched-tokens 114688 \
        --max-num-seqs 128 \
        --allowed-local-media-path ${GIT_ROOT}/tests/v1/ec_connector/integration \
        --ec-transfer-config '{
            "ec_connector": "ECSharedStorageConnector",
            "ec_role": "ec_producer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
            }
        }' \
        > $LOG_PATH/1e1pd_encoder.log 2>&1 &
    PIDS+=($!)
    
    # Start prefill+decode instance
    echo "Starting PD instance on GPU $GPU_PD, port $PREFILL_DECODE_PORT"
    CUDA_VISIBLE_DEVICES="$GPU_PD" vllm serve "$MODEL" \
        --port $PREFILL_DECODE_PORT \
        --enforce-eager \
        --gpu-memory-utilization 0.7 \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --allowed-local-media-path ${GIT_ROOT}/tests/v1/ec_connector/integration \
        --ec-transfer-config '{
            "ec_connector": "ECSharedStorageConnector",
            "ec_role": "ec_consumer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
            }
        }' \
        > $LOG_PATH/1e1pd_pd.log 2>&1 &
    PIDS+=($!)
    
    # Wait for instances to start
    echo "Waiting for encoder instance..."
    wait_for_server $ENCODE_PORT
    echo "Waiting for PD instance..."
    wait_for_server $PREFILL_DECODE_PORT

    # Start proxy
    echo "Starting EPD proxy on port $PROXY_PORT"
    python "${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
        --host "0.0.0.0" \
        --port $PROXY_PORT \
        --encode-servers-urls "http://localhost:$ENCODE_PORT" \
        --prefill-servers-urls "disable" \
        --decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
        > $LOG_PATH/1e1pd_proxy.log 2>&1 &
    PIDS+=($!)
    
    # Wait for proxy
    echo "Waiting for proxy..."
    wait_for_server $PROXY_PORT

    curl http://127.0.0.1:$PROXY_PORT/v1/models
    curl http://127.0.0.1:$PROXY_PORT/health
    echo ""

    echo "All EPD (1E+1PD) services are up!"
    
    # Run test in disagg mode
    echo "Running EPD (1E+1PD) correctness test..."
    
    python "${GIT_ROOT}/tests/v1/ec_connector/integration/test_epd_correctness.py" \
        --service_url "http://localhost:$PROXY_PORT" \
        --model_name "$MODEL" \
        --mode disagg \
        --baseline_file "$BASELINE_FILE" \
        $MM_FLAG
    
    # Cleanup
    echo "✓✓ 1E+1PD Correctness Test finished"
    echo "Stopping EPD (1E+1PD) instances..."
    for pid in "${PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    sleep 2
    cleanup_instances
}

# Function to run baseline for 1E + 1P + 1D (PD disagg)
run_baseline_1p_1d() {
    echo "================================"
    echo "Running PD BASELINE (1P + 1D)"
    echo "================================"
    
    cleanup_instances
    rm -rf "$EC_SHARED_STORAGE_PATH"
    mkdir -p "$EC_SHARED_STORAGE_PATH"
    
    local PREFILL_PORT=$PREFILL_PORT
    local DECODE_PORT=$DECODE_PORT
    local PROXY_PORT=$ENDPOINT_PORT
    
    declare -a PIDS=()
    
    # Start prefill instance
    echo "Starting prefill instance on GPU $GPU_P, port $PREFILL_PORT"
    CUDA_VISIBLE_DEVICES="$GPU_P" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
    vllm serve "$MODEL" \
        --port $PREFILL_PORT \
        --enforce-eager \
        --gpu-memory-utilization 0.7 \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --allowed-local-media-path ${GIT_ROOT}/tests/v1/ec_connector/integration \
        --kv-transfer-config '{
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer"
        }' \
        > $LOG_PATH/1p1d_prefill.log 2>&1 &
    PIDS+=($!)
    
    # Start decode instance
    echo "Starting decode instance on GPU $GPU_D, port $DECODE_PORT"
    CUDA_VISIBLE_DEVICES="$GPU_D" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=6000 \
    vllm serve "$MODEL" \
        --port $DECODE_PORT \
        --enforce-eager \
        --gpu-memory-utilization 0.7 \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --allowed-local-media-path ${GIT_ROOT}/tests/v1/ec_connector/integration \
        --kv-transfer-config '{
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer"
        }' \
        > $LOG_PATH/1p1d_decode.log 2>&1 &
    PIDS+=($!)
    
    # Wait for instances to start
    echo "Waiting for prefill instance..."
    wait_for_server $PREFILL_PORT
    echo "Waiting for decode instance..."
    wait_for_server $DECODE_PORT
    
    # Start proxy
    echo "Starting EPD proxy on port $PROXY_PORT"
    python "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
        --host "0.0.0.0" \
        --port $PROXY_PORT \
        --prefiller-ports $PREFILL_PORT \
        --decoder-ports $DECODE_PORT \
        > $LOG_PATH/1p1d_proxy.log 2>&1 &
    PIDS+=($!)
    
    # Wait for proxy
    echo "Waiting for proxy..."
    wait_for_server $PROXY_PORT

    curl http://127.0.0.1:$PROXY_PORT/healthcheck
    echo ""

    echo "All PD (1P+1D) services are up!"
    
    # Run test in baseline mode
    echo "Running PD disagg baseline..."
    
    python "${GIT_ROOT}/tests/v1/ec_connector/integration/test_epd_correctness.py" \
        --service_url "http://localhost:$PROXY_PORT" \
        --model_name "$MODEL" \
        --mode baseline_pd \
        --baseline_file "$BASELINE_PD_FILE" \
        $MM_FLAG
    
    # Cleanup
    echo "Stopping PD (1P+1D) instances..."
    for pid in "${PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    sleep 2
    cleanup_instances
}

# Function to run EPD with 1E + 1P + 1D
run_epd_1e_1p_1d() {
    echo "================================"
    echo "Running EPD (1E + 1P + 1D)"
    echo "================================"
    
    cleanup_instances
    rm -rf "$EC_SHARED_STORAGE_PATH"
    mkdir -p "$EC_SHARED_STORAGE_PATH"
    
    local ENCODE_PORT=$ENCODE_PORT
    local PREFILL_PORT=$PREFILL_PORT
    local DECODE_PORT=$DECODE_PORT
    local PROXY_PORT=$ENDPOINT_PORT
    
    declare -a PIDS=()
    
    # Start encoder instance
    echo "Starting encoder instance on GPU $GPU_E, port $ENCODE_PORT"
    CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
        --port $ENCODE_PORT \
        --enforce-eager \
        --gpu-memory-utilization 0.01 \
        --enable-request-id-headers \
        --no-enable-prefix-caching \
        --max-num-batched-tokens 114688 \
        --max-num-seqs 128 \
        --allowed-local-media-path ${GIT_ROOT}/tests/v1/ec_connector/integration \
        --ec-transfer-config '{
            "ec_connector": "ECSharedStorageConnector",
            "ec_role": "ec_producer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
            }
        }' \
        > $LOG_PATH/1e1p1d_encoder.log 2>&1 &
    PIDS+=($!)
    
    # Start prefill instance
    echo "Starting prefill instance on GPU $GPU_P, port $PREFILL_PORT"
    CUDA_VISIBLE_DEVICES="$GPU_P" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
    vllm serve "$MODEL" \
        --port $PREFILL_PORT \
        --enforce-eager \
        --gpu-memory-utilization 0.7 \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --allowed-local-media-path ${GIT_ROOT}/tests/v1/ec_connector/integration \
        --ec-transfer-config '{
            "ec_connector": "ECSharedStorageConnector",
            "ec_role": "ec_consumer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
            }
        }' \
        --kv-transfer-config '{
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer"
        }' \
        > $LOG_PATH/1e1p1d_prefill.log 2>&1 &
    PIDS+=($!)
    
    # Start decode instance
    echo "Starting decode instance on GPU $GPU_D, port $DECODE_PORT"
    CUDA_VISIBLE_DEVICES="$GPU_D" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=6000 \
    vllm serve "$MODEL" \
        --port $DECODE_PORT \
        --enforce-eager \
        --gpu-memory-utilization 0.7 \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --allowed-local-media-path ${GIT_ROOT}/tests/v1/ec_connector/integration \
        --kv-transfer-config '{
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer"
        }' \
        > $LOG_PATH/1e1p1d_decode.log 2>&1 &
    PIDS+=($!)
    
    # Wait for instances to start
    echo "Waiting for encoder instance..."
    wait_for_server $ENCODE_PORT
    echo "Waiting for prefill instance..."
    wait_for_server $PREFILL_PORT
    echo "Waiting for decode instance..."
    wait_for_server $DECODE_PORT
    
    # Start proxy
    echo "Starting EPD proxy on port $PROXY_PORT"
    python "${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
        --host "0.0.0.0" \
        --port $PROXY_PORT \
        --encode-servers-urls "http://localhost:$ENCODE_PORT" \
        --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
        --decode-servers-urls "http://localhost:$DECODE_PORT" \
        > $LOG_PATH/1e1p1d_proxy.log 2>&1 &
    PIDS+=($!)
    
    # Wait for proxy
    echo "Waiting for proxy..."
    wait_for_server $PROXY_PORT

    curl http://127.0.0.1:$PROXY_PORT/v1/models
    curl http://127.0.0.1:$PROXY_PORT/health
    echo ""

    echo "All EPD (1E+1P+1D) services are up!"
    
    # Run test in disagg mode
    echo "Running EPD (1E+1P+1D) correctness test..."
    
    python "${GIT_ROOT}/tests/v1/ec_connector/integration/test_epd_correctness.py" \
        --service_url "http://localhost:$PROXY_PORT" \
        --model_name "$MODEL" \
        --mode disagg \
        --baseline_file "$BASELINE_PD_FILE" \
        $MM_FLAG
    
    # Cleanup
    echo "✓✓ 1E+1P+1D Correctness Test finished"
    echo "Stopping EPD (1E+1P+1D) instances..."
    for pid in "${PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    sleep 2
    cleanup_instances
}

# Main execution
echo "================================"
echo "EPD Correctness Test Suite"
echo "Model: $MODEL"
echo "================================"

# Step 1: Run baseline
run_baseline

# Step 2: Test 1E + 1PD
run_epd_1e_1pd

# Step 3: Test baseline 1P + 1D
run_baseline_1p_1d

# Step 4: Test 1E + 1P + 1D
run_epd_1e_1p_1d

# Cleanup output file
rm -f "$BASELINE_FILE"
rm -f "$BASELINE_PD_FILE"

echo "================================"
echo "✓✓ All EPD correctness tests finished!"
echo "================================"
