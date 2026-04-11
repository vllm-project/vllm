#!/usr/bin/env bash
# num_stages sweep for TQ Triton decode kernel _tq_decode_stage1
# Tests num_stages=1,2,3 for k8v4 (GPU2) and turboquant_4bit_nc (GPU3)
# Usage: bash tools/num_stages_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KERNEL_FILE="$SCRIPT_DIR/vllm/v1/attention/ops/triton_turboquant_decode.py"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
MODEL="Qwen/Qwen3-4B"
RESULTS_FILE="$SCRIPT_DIR/tools/num_stages_results.txt"

echo "=== TQ Triton num_stages sweep ===" | tee "$RESULTS_FILE"
echo "Date: $(date)" | tee -a "$RESULTS_FILE"
echo "Kernel: $KERNEL_FILE" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

patch_num_stages() {
    local ns=$1
    # Replace num_stages=N in the _tq_decode_stage1 launch (line ~548)
    sed -i "s/^\(        num_stages=\)[0-9]\+,$/\1${ns},/" "$KERNEL_FILE"
    echo "  patched num_stages=$ns in $KERNEL_FILE"
}

run_bench() {
    local gpu=$1
    local port=$2
    local preset=$3
    local ns=$4

    echo ""
    echo "--- preset=$preset  num_stages=$ns  GPU=$gpu ---" | tee -a "$RESULTS_FILE"

    # Start server
    echo "  Starting server on GPU $gpu port $port..."
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --kv-cache-dtype "$preset" \
        --port "$port" \
        --max-model-len 32768 \
        --disable-log-requests \
        > /tmp/vllm_gpu${gpu}_ns${ns}.log 2>&1 &
    SERVER_PID=$!

    # Wait for server to be ready
    echo "  Waiting for server (pid=$SERVER_PID)..."
    local max_wait=120
    local elapsed=0
    while ! curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 3
        elapsed=$((elapsed + 3))
        if [ $elapsed -ge $max_wait ]; then
            echo "  ERROR: server did not start after ${max_wait}s" | tee -a "$RESULTS_FILE"
            kill $SERVER_PID 2>/dev/null || true
            return 1
        fi
    done
    echo "  Server ready after ${elapsed}s"

    # Run benchmark
    echo "  Running bench..."
    BENCH_OUT=$(
        $PYTHON -m sglang.bench_serving \
            --backend vllm \
            --port "$port" \
            --model "$MODEL" \
            --dataset-name random \
            --random-input-len 64 \
            --random-output-len 1024 \
            --num-prompts 200 \
            --request-rate inf 2>&1
    )

    echo "$BENCH_OUT" >> "$RESULTS_FILE"

    # Extract key metrics
    THROUGHPUT=$(echo "$BENCH_OUT" | grep -oP 'Output token throughput.*?:\s*\K[\d.]+' | head -1 || echo "N/A")
    MEDIAN_TTFT=$(echo "$BENCH_OUT" | grep -oP 'Median TTFT.*?:\s*\K[\d.]+' | head -1 || echo "N/A")
    echo "  output_tok/s=$THROUGHPUT  median_ttft_ms=$MEDIAN_TTFT" | tee -a "$RESULTS_FILE"

    # Kill server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 2
}

# ===== Sweep k8v4 on GPU 2 =====
PRESET="turboquant_k8v4"
GPU=2
PORT=8502
echo "### PRESET: $PRESET  GPU: $GPU ###" | tee -a "$RESULTS_FILE"

for NS in 1 2 3; do
    patch_num_stages $NS
    run_bench $GPU $PORT "$PRESET" $NS
done

# Restore to 2 (default)
patch_num_stages 2

# ===== Sweep turboquant_4bit_nc on GPU 3 =====
PRESET="turboquant_4bit_nc"
GPU=3
PORT=8503
echo "" | tee -a "$RESULTS_FILE"
echo "### PRESET: $PRESET  GPU: $GPU ###" | tee -a "$RESULTS_FILE"

for NS in 1 2 3; do
    patch_num_stages $NS
    run_bench $GPU $PORT "$PRESET" $NS
done

# Restore to 2
patch_num_stages 2

echo "" | tee -a "$RESULTS_FILE"
echo "=== Sweep complete. Results in $RESULTS_FILE ===" | tee -a "$RESULTS_FILE"
