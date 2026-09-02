#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Scale-out EC connector E2E test
#
# Tests the disaggregated multimodal flow over the scale-out endpoints,
# following the bash-script style of tests/v1/kv_connector/nixl_integration:
#
#   1. Baseline: a single vLLM instance serving /v1/chat/completions
#   2. Scale-out:
#        render   (GPU-less, `vllm launch render`)
#        encode   (EC producer, encode-only, --tokens-only)
#        prefill  (EC consumer, --tokens-only)
#
# The Python client renders each multimodal request once, sends the full
# kwargs_data to the encode instance, then sends metadata-only features
# (mm_metadata, no kwargs_data) plus ec_transfer_params to the prefill
# instance, and derenders the output tokens. Outputs are compared against
# the baseline for exact equality.
#
# Usage:
#   bash tests/entrypoints/scale_out/ec_integration/run_scale_out_ec_e2e_test.sh
#
# Requires 2 GPUs by default (encode on GPU 0, prefill on GPU 1); the
# baseline reuses GPU 0 after the encode instance is stopped.

# set -xe

# Resolve the repository root from the script location instead of `.git`.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)}"

# Model and engine configuration
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"

# GPU configuration. encode + prefill run concurrently on separate GPUs;
# the baseline reuses GPU_E after the scale-out instances are stopped.
GPU_E="${GPU_E:-0}"
GPU_PD="${GPU_PD:-1}"
GPU_SINGLE="${GPU_SINGLE:-$GPU_E}"

# Ports
RENDER_PORT="${RENDER_PORT:-19600}"
ENCODE_PORT="${ENCODE_PORT:-19601}"
PREFILL_PORT="${PREFILL_PORT:-19602}"
BASELINE_PORT="${BASELINE_PORT:-19603}"

# Storage path for the ECExampleConnector encoder cache
EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_scale_out_test}"

# Python interpreter (override to .venv/bin/python for local runs)
PYTHON="${PYTHON:-python3}"

# Output files and logs
LOG_PATH="${LOG_PATH:-/tmp}"
BASELINE_FILE="${BASELINE_FILE:-/tmp/vllm_scale_out_ec_baseline.txt}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

mkdir -p "$LOG_PATH"

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr) 2>/dev/null || true' SIGINT SIGTERM EXIT

wait_for_server() {
    local port=$1
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:${port}/health > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

cleanup_instances() {
    echo "Cleaning up any running vLLM instances..."
    pkill -TERM -f "vllm serve" || true
    pkill -TERM -f "vllm launch render" || true
    sleep 3
    pkill -9 -f "vllm serve" || true
    pkill -9 -f "vllm launch render" || true
    sleep 2
}

# Step 1: baseline single instance
run_baseline() {
    echo "================================"
    echo "Running BASELINE (single instance)"
    echo "================================"

    cleanup_instances
    rm -rf "$EC_SHARED_STORAGE_PATH"

    echo "Starting baseline instance on GPU $GPU_SINGLE, port $BASELINE_PORT"
    env CUDA_VISIBLE_DEVICES="$GPU_SINGLE" vllm serve "$MODEL" \
        --port "$BASELINE_PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs "$MAX_NUM_SEQS" \
        > "$LOG_PATH"/scale_out_ec_baseline.log 2>&1 &
    local BASELINE_PID=$!

    echo "Waiting for baseline instance to start..."
    wait_for_server "$BASELINE_PORT"

    "$PYTHON" "${SCRIPT_DIR}/test_scale_out_ec_e2e.py" \
        --mode baseline \
        --service_url "http://localhost:$BASELINE_PORT" \
        --model_name "$MODEL" \
        --baseline_file "$BASELINE_FILE"

    echo "Stopping baseline instance..."
    kill "$BASELINE_PID" 2>/dev/null || true
    sleep 2
    cleanup_instances
}

# Step 2: render + encode (EC producer) + prefill (EC consumer)
run_scale_out_ec() {
    echo "================================"
    echo "Running SCALE-OUT EC (render + encode + prefill)"
    echo "================================"

    cleanup_instances
    rm -rf "$EC_SHARED_STORAGE_PATH"
    mkdir -p "$EC_SHARED_STORAGE_PATH"

    declare -a PIDS=()

    # GPU-less render/derender server
    echo "Starting render server on port $RENDER_PORT"
    vllm launch render "$MODEL" \
        --port "$RENDER_PORT" \
        > "$LOG_PATH"/scale_out_ec_render.log 2>&1 &
    PIDS+=($!)

    # Encode-only EC producer instance. It runs the vision encoder and
    # publishes embeddings through the ECExampleConnector shared storage.
    echo "Starting encode instance on GPU $GPU_E, port $ENCODE_PORT"
    env CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
        --tokens-only \
        --port "$ENCODE_PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager \
        --gpu-memory-utilization 0.01 \
        --no-enable-prefix-caching \
        --max-num-batched-tokens 114688 \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --ec-transfer-config '{
            "ec_connector": "ECExampleConnector",
            "ec_role": "ec_producer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
            }
        }' \
        > "$LOG_PATH"/scale_out_ec_encode.log 2>&1 &
    PIDS+=($!)

    # Prefill/decode EC consumer instance. It receives metadata-only
    # features plus ec_transfer_params and loads the embeddings that the
    # encode instance published.
    echo "Starting prefill instance on GPU $GPU_PD, port $PREFILL_PORT"
    env CUDA_VISIBLE_DEVICES="$GPU_PD" vllm serve "$MODEL" \
        --tokens-only \
        --port "$PREFILL_PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager \
        --enable-mm-embeds \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --ec-transfer-config '{
            "ec_connector": "ECExampleConnector",
            "ec_role": "ec_consumer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
            }
        }' \
        > "$LOG_PATH"/scale_out_ec_prefill.log 2>&1 &
    PIDS+=($!)

    echo "Waiting for render instance..."
    wait_for_server "$RENDER_PORT"
    echo "Waiting for encode instance..."
    wait_for_server "$ENCODE_PORT"
    echo "Waiting for prefill instance..."
    wait_for_server "$PREFILL_PORT"

    echo "All scale-out EC services are up!"

    "$PYTHON" "${SCRIPT_DIR}/test_scale_out_ec_e2e.py" \
        --mode disagg \
        --render_url "http://localhost:$RENDER_PORT" \
        --encode_url "http://localhost:$ENCODE_PORT" \
        --prefill_url "http://localhost:$PREFILL_PORT" \
        --model_name "$MODEL" \
        --baseline_file "$BASELINE_FILE"

    echo "Stopping scale-out EC instances..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    cleanup_instances
}

# Main execution
echo "================================"
echo "Scale-out EC E2E Test Suite"
echo "Model: $MODEL"
echo "================================"

run_baseline
run_scale_out_ec

rm -f "$BASELINE_FILE"

echo "================================"
echo "All scale-out EC e2e tests finished!"
echo "================================"
