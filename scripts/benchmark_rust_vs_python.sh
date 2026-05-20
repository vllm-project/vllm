#!/bin/bash
set -e

# Configuration
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GRPC_PORT="${GRPC_PORT:-50051}"
HTTP_PORT="${HTTP_PORT:-8000}"
MAX_TIME="${MAX_TIME:-3}"
MAX_REQUESTS="${MAX_REQUESTS:-500}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-./experiments}"

echo "============================================"
echo "vLLM Rust vs Python API Server Benchmark"
echo "============================================"
echo "Model: $MODEL"
echo "gRPC Port: $GRPC_PORT"
echo "HTTP Port: $HTTP_PORT"
echo ""

# Ensure experiment directory exists
mkdir -p "$EXPERIMENT_DIR"

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up..."
    kill $GRPC_PID 2>/dev/null || true
    kill $API_PID 2>/dev/null || true
}
trap cleanup EXIT

# Start Python gRPC backend
echo "[1/4] Starting Python gRPC server..."
python -m vllm.entrypoints.grpc_server \
    --model "$MODEL" \
    --port "$GRPC_PORT" &
GRPC_PID=$!

echo "Waiting for model to load (this may take a while)..."
sleep 60

# Verify gRPC server is healthy
echo "Checking gRPC server health..."
if ! grpcurl -plaintext "localhost:$GRPC_PORT" vllm.grpc.engine.VllmEngine/HealthCheck; then
    echo "ERROR: gRPC server not responding"
    exit 1
fi
echo "gRPC server is healthy"

# Benchmark 1: Python API Server
echo ""
echo "[2/4] Benchmarking Python API Server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$HTTP_PORT" &
API_PID=$!
sleep 10

genai-bench benchmark \
    --api-backend vllm \
    --api-base "http://localhost:$HTTP_PORT" \
    --task text-to-text \
    --model-tokenizer "$MODEL" \
    --api-model-name "$MODEL" \
    --traffic-scenario "D(100,100)" \
    --traffic-scenario "D(2000,200)" \
    --num-concurrency 1 \
    --num-concurrency 4 \
    --num-concurrency 16 \
    --num-concurrency 64 \
    --max-time-per-run "$MAX_TIME" \
    --max-requests-per-run "$MAX_REQUESTS" \
    --experiment-base-dir "$EXPERIMENT_DIR" \
    --experiment-folder-name "python-baseline"

kill $API_PID
wait $API_PID 2>/dev/null || true
sleep 2

# Benchmark 2: Rust API Server
echo ""
echo "[3/4] Benchmarking Rust API Server..."
./rust/target/release/vllm-api-server \
    --grpc-addr "localhost:$GRPC_PORT" \
    --port "$HTTP_PORT" \
    --model "$MODEL" &
API_PID=$!
sleep 5

genai-bench benchmark \
    --api-backend vllm \
    --api-base "http://localhost:$HTTP_PORT" \
    --task text-to-text \
    --model-tokenizer "$MODEL" \
    --api-model-name "$MODEL" \
    --traffic-scenario "D(100,100)" \
    --traffic-scenario "D(2000,200)" \
    --num-concurrency 1 \
    --num-concurrency 4 \
    --num-concurrency 16 \
    --num-concurrency 64 \
    --max-time-per-run "$MAX_TIME" \
    --max-requests-per-run "$MAX_REQUESTS" \
    --experiment-base-dir "$EXPERIMENT_DIR" \
    --experiment-folder-name "rust-prototype"

kill $API_PID
wait $API_PID 2>/dev/null || true

# Generate comparison report
echo ""
echo "[4/4] Generating comparison report..."
genai-bench excel \
    --experiment-folder "$EXPERIMENT_DIR" \
    --excel-name rust-vs-python-comparison \
    --metric-percentile p90 p99 mean

echo ""
echo "============================================"
echo "Benchmark complete!"
echo "Results saved to: $EXPERIMENT_DIR"
echo "============================================"
