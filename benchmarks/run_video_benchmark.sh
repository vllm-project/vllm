#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Video Benchmark Runner for Hybrid Attention VL Models
#
# This script runs comparative benchmarks for video inference with:
# 1. Standard attention (baseline)
# 2. Hybrid SSM + sliding window attention
#
# The benchmark uses the baby_reading video asset from HuggingFace.
#
# Usage:
#   ./benchmarks/run_video_benchmark.sh [MODEL_PATH] [OUTPUT_DIR]
#
# Examples:
#   ./benchmarks/run_video_benchmark.sh Qwen/Qwen2.5-VL-3B-Instruct ./video_results
#   ./benchmarks/run_video_benchmark.sh Qwen/Qwen2.5-VL-7B-Instruct ./video_results
#
# Environment variables:
#   NUM_FRAMES     - Number of video frames to sample (default: 16)
#   NUM_ITERS      - Number of benchmark iterations (default: 5)
#   NUM_WARMUP     - Number of warmup iterations (default: 2)
#   MAX_TOKENS     - Maximum tokens to generate (default: 128)
#   GPU_MEM_UTIL   - GPU memory utilization (default: 0.9)
#   TENSOR_PARALLEL - Tensor parallel size (default: 1)

set -e

# Set multiprocessing method to avoid CUDA initialization conflicts
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Default configuration
MODEL_PATH="${1:-Qwen/Qwen2.5-VL-3B-Instruct}"
OUTPUT_DIR="${2:-./video_benchmark_results}"
NUM_FRAMES="${NUM_FRAMES:-16}"
NUM_ITERS="${NUM_ITERS:-5}"
NUM_WARMUP="${NUM_WARMUP:-2}"
MAX_TOKENS="${MAX_TOKENS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
QUESTION="${QUESTION:-Describe what is happening in this video in detail.}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Video Benchmark Suite for Hybrid Attention VL Models"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Number of frames: $NUM_FRAMES"
echo "Benchmark iterations: $NUM_ITERS"
echo "Warmup iterations: $NUM_WARMUP"
echo "Max tokens: $MAX_TOKENS"
echo "GPU memory utilization: $GPU_MEM_UTIL"
echo "Tensor parallel size: $TENSOR_PARALLEL"
echo "Max model length: $MAX_MODEL_LEN"
echo "Question: $QUESTION"
echo "============================================================"

# Check if opencv is installed
echo ""
echo "Checking dependencies..."
python3 -c "import cv2" 2>/dev/null || {
    echo "WARNING: opencv-python not installed. Installing..."
    pip install opencv-python
}

echo ""
echo "============================================================"
echo "Running STANDARD attention benchmark"
echo "============================================================"

standard_output="$OUTPUT_DIR/standard_results.json"

python benchmarks/video_benchmark.py \
    --model "$MODEL_PATH" \
    --num-frames "$NUM_FRAMES" \
    --num-iterations "$NUM_ITERS" \
    --num-warmup "$NUM_WARMUP" \
    --max-tokens "$MAX_TOKENS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --question "$QUESTION" \
    --output-file "$standard_output"

echo "Standard attention results saved to: $standard_output"

echo ""
echo "============================================================"
echo "Running HYBRID attention benchmark"
echo "============================================================"

hybrid_output="$OUTPUT_DIR/hybrid_results.json"

python benchmarks/video_benchmark.py \
    --model "$MODEL_PATH" \
    --num-frames "$NUM_FRAMES" \
    --num-iterations "$NUM_ITERS" \
    --num-warmup "$NUM_WARMUP" \
    --max-tokens "$MAX_TOKENS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --question "$QUESTION" \
    --use-hybrid-attention \
    --output-file "$hybrid_output"

echo "Hybrid attention results saved to: $hybrid_output"

echo ""
echo "============================================================"
echo "Benchmark Complete!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  No JSON files found"
echo ""

# Print comparison summary
if [ -f "$standard_output" ] && [ -f "$hybrid_output" ]; then
    echo "============================================================"
    echo "Comparison Summary"
    echo "============================================================"
    
    python3 << EOF
import json
import os

output_dir = "$OUTPUT_DIR"

# Load results
with open(f"{output_dir}/standard_results.json") as f:
    standard_data = json.load(f)

with open(f"{output_dir}/hybrid_results.json") as f:
    hybrid_data = json.load(f)

# Extract metrics
std_results = standard_data.get("results", [{}])[0]
hyb_results = hybrid_data.get("results", [{}])[0]

print(f"{'Metric':<30} {'Standard':<15} {'Hybrid':<15} {'Diff':<10}")
print("-" * 70)

# Check for errors
if std_results.get("error"):
    print(f"Standard benchmark failed: {std_results['error']}")
elif hyb_results.get("error"):
    print(f"Hybrid benchmark failed: {hyb_results['error']}")
else:
    # Latency comparison
    std_lat = std_results.get("avg_latency_seconds", 0) * 1000
    hyb_lat = hyb_results.get("avg_latency_seconds", 0) * 1000
    lat_diff = ((hyb_lat - std_lat) / std_lat * 100) if std_lat > 0 else 0
    print(f"{'Avg Latency (ms)':<30} {std_lat:>12.1f}   {hyb_lat:>12.1f}   {lat_diff:>+7.1f}%")
    
    std_p50 = std_results.get("p50_latency_seconds", 0) * 1000
    hyb_p50 = hyb_results.get("p50_latency_seconds", 0) * 1000
    p50_diff = ((hyb_p50 - std_p50) / std_p50 * 100) if std_p50 > 0 else 0
    print(f"{'P50 Latency (ms)':<30} {std_p50:>12.1f}   {hyb_p50:>12.1f}   {p50_diff:>+7.1f}%")
    
    std_p99 = std_results.get("p99_latency_seconds", 0) * 1000
    hyb_p99 = hyb_results.get("p99_latency_seconds", 0) * 1000
    p99_diff = ((hyb_p99 - std_p99) / std_p99 * 100) if std_p99 > 0 else 0
    print(f"{'P99 Latency (ms)':<30} {std_p99:>12.1f}   {hyb_p99:>12.1f}   {p99_diff:>+7.1f}%")
    
    # Throughput comparison
    std_tput = std_results.get("throughput_tokens_per_second", 0)
    hyb_tput = hyb_results.get("throughput_tokens_per_second", 0)
    tput_diff = ((hyb_tput - std_tput) / std_tput * 100) if std_tput > 0 else 0
    print(f"{'Throughput (tok/s)':<30} {std_tput:>12.1f}   {hyb_tput:>12.1f}   {tput_diff:>+7.1f}%")
    
    std_gen = std_results.get("generation_tokens_per_second", 0)
    hyb_gen = hyb_results.get("generation_tokens_per_second", 0)
    gen_diff = ((hyb_gen - std_gen) / std_gen * 100) if std_gen > 0 else 0
    print(f"{'Generation (tok/s)':<30} {std_gen:>12.1f}   {hyb_gen:>12.1f}   {gen_diff:>+7.1f}%")
    
    # Token counts
    print("")
    print(f"{'Total Input Tokens':<30} {std_results.get('total_input_tokens', 0):>12}   {hyb_results.get('total_input_tokens', 0):>12}")
    print(f"{'Total Output Tokens':<30} {std_results.get('total_output_tokens', 0):>12}   {hyb_results.get('total_output_tokens', 0):>12}")
    
    # Memory info
    std_mem = std_results.get("memory", {}).get("post_benchmark", {})
    hyb_mem = hyb_results.get("memory", {}).get("post_benchmark", {})
    if std_mem.get("available") and hyb_mem.get("available"):
        print("")
        std_used = std_mem.get("used_memory_gib", 0)
        hyb_used = hyb_mem.get("used_memory_gib", 0)
        mem_diff = ((hyb_used - std_used) / std_used * 100) if std_used > 0 else 0
        print(f"{'GPU Memory Used (GiB)':<30} {std_used:>12.2f}   {hyb_used:>12.2f}   {mem_diff:>+7.1f}%")

print("")
EOF
fi

echo "============================================================"
echo "Quick Run Commands"
echo "============================================================"
echo ""
echo "# Run comparison mode (both standard and hybrid):"
echo "python benchmarks/video_benchmark.py --compare-modes --output-file results.json"
echo ""
echo "# Run with different frame counts:"
echo "NUM_FRAMES=8 ./benchmarks/run_video_benchmark.sh"
echo "NUM_FRAMES=32 ./benchmarks/run_video_benchmark.sh"
echo ""
echo "# Run with different model:"
echo "./benchmarks/run_video_benchmark.sh Qwen/Qwen2.5-VL-7B-Instruct ./results_7b"
echo ""
echo "============================================================"
echo "Troubleshooting"
echo "============================================================"
echo ""
echo "If you see 'CUDA device busy' errors, try:"
echo "  1. Check for other GPU processes: nvidia-smi"
echo "  2. Kill any stale processes: pkill -f vllm"
echo "  3. Set specific GPU: CUDA_VISIBLE_DEVICES=0 ./benchmarks/run_video_benchmark.sh"
echo ""

