#!/usr/bin/env bash
# Run a single benchmark: start server, bench, kill server, print results
# Usage: bash tools/run_single_bench.sh <GPU> <PORT> <PRESET> <NS_LABEL>
# NS_LABEL is just for logging (the kernel file must already be patched)
set -euo pipefail

GPU=$1
PORT=$2
PRESET=$3
NS_LABEL=$4
PYTHON=/home/vibhav.agarwal/vllm-tq/.venv/bin/python
MODEL=Qwen/Qwen3-4B
LOG_DIR=/home/vibhav.agarwal/vllm-tq/tools/sweep_logs

echo ">>> START  preset=$PRESET num_stages=$NS_LABEL gpu=$GPU port=$PORT"

# Start server
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --kv-cache-dtype "$PRESET" \
    --port "$PORT" \
    --max-model-len 32768 \
    --disable-log-stats \
    > "$LOG_DIR/gpu${GPU}_${PRESET}_ns${NS_LABEL}.log" 2>&1 &
SERVER_PID=$!

# Wait for server ready (max 150s)
for i in $(seq 1 50); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "    server ready at t=${i}*3s"
        break
    fi
    sleep 3
    if [ $i -eq 50 ]; then
        echo "ERROR: server did not start"
        kill -9 $SERVER_PID 2>/dev/null || true
        exit 1
    fi
done

# Run benchmark
BENCH_LOG="$LOG_DIR/bench_${PRESET}_ns${NS_LABEL}.log"
$PYTHON -m sglang.bench_serving \
    --backend vllm \
    --port "$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 64 \
    --random-output-len 1024 \
    --num-prompts 200 \
    --request-rate inf \
    > "$BENCH_LOG" 2>&1

# Extract metrics
OUT_TPS=$(grep -oP 'Output token throughput.*?:\s*\K[\d.]+' "$BENCH_LOG" | head -1 || echo "N/A")
MEDIAN_ITL=$(grep -oP 'Median ITL.*?:\s*\K[\d.]+' "$BENCH_LOG" | head -1 || echo "N/A")
MEDIAN_TPOT=$(grep -oP 'Median TPOT.*?:\s*\K[\d.]+' "$BENCH_LOG" | head -1 || echo "N/A")
echo "    RESULT: output_tok/s=$OUT_TPS  median_itl_ms=$MEDIAN_ITL  median_tpot_ms=$MEDIAN_TPOT"

# Kill server and release GPU memory
kill -9 $SERVER_PID 2>/dev/null || true
# Also kill EngineCore child processes
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 5
