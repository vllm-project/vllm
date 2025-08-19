#!/bin/bash

export TORCH_COMPILE_CACHE_DIR="/lustre/fsw/portfolios/hw/users/sshrestha/.cache/torch_compile_cache"
export VLLM_CACHE_ROOT="/lustre/fsw/portfolios/hw/users/sshrestha/.cache/vllm"
export XDG_CACHE_HOME="/lustre/fsw/portfolios/hw/users/sshrestha/.cache"


# Simple script to start vLLM server and run benchmarks

# Configuration
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct"
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=8
SERVER_PORT=8000
BENCHMARK_WAIT_TIME=30  # seconds to wait for server to start

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting vLLM server and benchmark script${NC}"
echo "Model: $MODEL"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Server Port: $SERVER_PORT"
echo

# Start vLLM server in background
echo -e "${YELLOW}Starting vLLM server...${NC}"
# vllm serve $MODEL \
#   --trust-remote-code \
#   --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
#   --port $SERVER_PORT &

VLLM_ALL2ALL_BACKEND=deepep_low_latency VLLM_USE_DEEP_GEMM=1 vllm serve $MODEL --trust-remote-code --tensor-parallel-size=$TENSOR_PARALLEL_SIZE --data-parallel-size=$DATA_PARALLEL_SIZE --enable-expert-parallel --port $SERVER_PORT --max-model-len 16384 --gpu_memory_utilization=0.9 &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Function to cleanup server on script exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "Stopping vLLM server (PID: $SERVER_PID)"
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null
    fi
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Wait for server to start
echo -e "${YELLOW}Waiting ${BENCHMARK_WAIT_TIME} seconds for server to start...${NC}"
sleep $BENCHMARK_WAIT_TIME

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}ERROR: vLLM server failed to start or crashed${NC}"
    exit 1
fi

# Test if server is responding
echo -e "${YELLOW}Testing server connectivity...${NC}"
if curl -s http://localhost:$SERVER_PORT/health > /dev/null; then
    echo -e "${GREEN}Server is responding${NC}"
else
    echo -e "${YELLOW}Server health check failed, but continuing with benchmark...${NC}"
fi

echo

# Run benchmark
echo -e "${YELLOW}Starting benchmark...${NC}"
vllm bench serve \
  --model $MODEL \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 10000 \
  --num-prompts 16 \
  --ignore-eos \
  --base-url http://localhost:$SERVER_PORT

echo -e "${GREEN}Benchmark completed${NC}"

# Server will be automatically stopped by the cleanup function
