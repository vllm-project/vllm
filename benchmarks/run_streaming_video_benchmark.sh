#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Streaming Video Benchmark Runner for SSM + Sliding Window Attention
#
# This script runs comparative benchmarks for streaming video inference:
# 1. Standard attention (baseline) - full KV cache, O(n) memory
# 2. Hybrid SSM + sliding window - fixed SSM state, O(1) memory scaling
#
# The benchmark demonstrates memory efficiency gains for long videos and
# concurrent query throughput with shared SSM state.
#
# Usage:
#   ./benchmarks/run_streaming_video_benchmark.sh [MODEL_PATH] [OUTPUT_DIR]
#
# Examples:
#   # Default: 3B model, all scenarios
#   ./benchmarks/run_streaming_video_benchmark.sh
#
#   # Specific model
#   ./benchmarks/run_streaming_video_benchmark.sh Qwen/Qwen2.5-VL-7B-Instruct
#
#   # Custom output directory
#   ./benchmarks/run_streaming_video_benchmark.sh Qwen/Qwen2.5-VL-3B-Instruct ./my_results
#
#   # Run specific scenario
#   SCENARIO=long-video ./benchmarks/run_streaming_video_benchmark.sh
#
# Environment variables:
#   SCENARIO       - Specific scenario to run (single-query, multi-query,
#                    continuous-query, long-video, memory-scaling)
#   NUM_FRAMES     - Override number of frames (default: scenario-specific)
#   CONCURRENT_Q   - Number of concurrent queries (default: 3)
#   MAX_TOKENS     - Maximum tokens to generate (default: 128)
#   GPU_MEM_UTIL   - GPU memory utilization (default: 0.9)
#   TENSOR_PARALLEL - Tensor parallel size (default: 1)
#   RUN_COMPARISON - Set to "true" to run standard vs hybrid comparison
#   SKIP_STANDARD  - Set to "true" to skip standard attention benchmark
#   SKIP_HYBRID    - Set to "true" to skip hybrid attention benchmark

set -e

# Set multiprocessing method to avoid CUDA initialization conflicts
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Default configuration
MODEL_PATH="${1:-Qwen/Qwen2.5-VL-3B-Instruct}"
OUTPUT_DIR="${2:-./streaming_benchmark_results}"
SCENARIO="${SCENARIO:-}"
NUM_FRAMES="${NUM_FRAMES:-}"
CONCURRENT_Q="${CONCURRENT_Q:-3}"
MAX_TOKENS="${MAX_TOKENS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
NUM_WARMUP="${NUM_WARMUP:-1}"
RUN_COMPARISON="${RUN_COMPARISON:-true}"
SKIP_STANDARD="${SKIP_STANDARD:-false}"
SKIP_HYBRID="${SKIP_HYBRID:-false}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Streaming Video Benchmark Suite"
echo "SSM + Sliding Window vs Standard Attention"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Scenario: ${SCENARIO:-all}"
echo "Concurrent queries: $CONCURRENT_Q"
echo "Max tokens: $MAX_TOKENS"
echo "GPU memory utilization: $GPU_MEM_UTIL"
echo "Tensor parallel size: $TENSOR_PARALLEL"
echo "Max model length: $MAX_MODEL_LEN"
echo "Run comparison: $RUN_COMPARISON"
echo "============================================================"

# Check dependencies
echo ""
echo "Checking dependencies..."
python3 -c "import cv2" 2>/dev/null || {
    echo "WARNING: opencv-python not installed. Installing..."
    pip install opencv-python
}

# Function to run a scenario
run_scenario() {
    local scenario_name=$1
    local num_frames=$2
    local output_suffix=$3

    echo ""
    echo "============================================================"
    echo "Running scenario: $scenario_name"
    echo "Frames: $num_frames"
    echo "============================================================"

    local common_args="--model $MODEL_PATH \
        --scenario $scenario_name \
        --max-tokens $MAX_TOKENS \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --tensor-parallel-size $TENSOR_PARALLEL \
        --max-model-len $MAX_MODEL_LEN \
        --num-warmup $NUM_WARMUP"

    if [ -n "$num_frames" ]; then
        common_args="$common_args --num-frames $num_frames"
    fi

    if [ "$RUN_COMPARISON" = "true" ]; then
        # Run comparison mode
        echo "Running standard vs hybrid comparison..."
        python benchmarks/streaming_video_benchmark.py \
            $common_args \
            --compare-modes \
            --output-file "$OUTPUT_DIR/${scenario_name}_comparison${output_suffix}.json"
    else
        # Run individual benchmarks
        if [ "$SKIP_STANDARD" != "true" ]; then
            echo "Running STANDARD attention..."
            python benchmarks/streaming_video_benchmark.py \
                $common_args \
                --output-file "$OUTPUT_DIR/${scenario_name}_standard${output_suffix}.json"
        fi

        if [ "$SKIP_HYBRID" != "true" ]; then
            echo "Running HYBRID attention..."
            python benchmarks/streaming_video_benchmark.py \
                $common_args \
                --use-hybrid-attention \
                --output-file "$OUTPUT_DIR/${scenario_name}_hybrid${output_suffix}.json"
        fi
    fi
}

# Run benchmarks
if [ -n "$SCENARIO" ]; then
    # Run specific scenario
    run_scenario "$SCENARIO" "$NUM_FRAMES" ""
else
    # Run all scenarios for comprehensive comparison
    echo ""
    echo "Running all benchmark scenarios..."

    # 1. Memory scaling test - most important for demonstrating O(1) vs O(n)
    echo ""
    echo "=== Phase 1: Memory Scaling Test ==="
    run_scenario "memory-scaling" "${NUM_FRAMES:-64}" ""

    # 2. Multi-query test - demonstrates SSM state sharing
    echo ""
    echo "=== Phase 2: Concurrent Query Test ==="
    run_scenario "multi-query" "${NUM_FRAMES:-32}" ""

    # 3. Long video test - stress test for memory efficiency
    echo ""
    echo "=== Phase 3: Long Video Stress Test ==="
    run_scenario "long-video" "${NUM_FRAMES:-128}" ""

    # 4. Continuous query test - real-time Q&A scenario
    echo ""
    echo "=== Phase 4: Continuous Query Test ==="
    run_scenario "continuous-query" "${NUM_FRAMES:-48}" ""
fi

echo ""
echo "============================================================"
echo "Generating Visualizations"
echo "============================================================"

# Generate visualization plots
python benchmarks/visualize_video_benchmark.py \
    --results-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/plots" \
    --format png \
    --streaming-mode 2>/dev/null || {
    echo "Note: Visualization may need streaming mode support. Skipping..."
}

echo ""
echo "============================================================"
echo "Benchmark Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  No JSON files found"
echo ""
if [ -d "$OUTPUT_DIR/plots" ]; then
    ls -la "$OUTPUT_DIR/plots"/*.png 2>/dev/null || echo "  No plot files found"
fi

# Print summary if comparison was run
if [ "$RUN_COMPARISON" = "true" ] && [ -z "$SCENARIO" ]; then
    echo ""
    echo "============================================================"
    echo "Key Findings Summary"
    echo "============================================================"

    python3 << 'EOF'
import json
import os
import sys

output_dir = os.environ.get("OUTPUT_DIR", "./streaming_benchmark_results")

def load_comparison(scenario):
    path = f"{output_dir}/{scenario}_comparison.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def print_scenario_summary(name, data):
    if not data or len(data.get("results", [])) < 2:
        return
    
    results = data["results"]
    standard = next((r for r in results if r["config"] == "standard_attention"), None)
    hybrid = next((r for r in results if r["config"] == "hybrid_attention"), None)
    
    if not standard or not hybrid:
        return
    
    print(f"\n{name.upper()}:")
    print("-" * 40)
    
    # Memory comparison
    std_mem = standard.get("peak_memory_gib", 0)
    hyb_mem = hybrid.get("peak_memory_gib", 0)
    if std_mem > 0 and hyb_mem > 0:
        savings = (std_mem - hyb_mem) / std_mem * 100
        print(f"  Memory: {std_mem:.2f} -> {hyb_mem:.2f} GiB ({savings:+.1f}%)")
    
    # Memory growth
    std_growth = standard.get("memory_growth_rate_gib_per_frame", 0) * 1000
    hyb_growth = hybrid.get("memory_growth_rate_gib_per_frame", 0) * 1000
    if std_growth > 0:
        print(f"  Growth: {std_growth:.3f} -> {hyb_growth:.3f} MiB/frame")
    
    # Latency
    std_lat = standard.get("avg_query_latency_seconds", 0) * 1000
    hyb_lat = hybrid.get("avg_query_latency_seconds", 0) * 1000
    if std_lat > 0 and hyb_lat > 0:
        delta = (hyb_lat - std_lat) / std_lat * 100
        print(f"  Latency: {std_lat:.1f} -> {hyb_lat:.1f} ms ({delta:+.1f}%)")

# Load and summarize each scenario
scenarios = ["memory-scaling", "multi-query", "long-video", "continuous-query"]
for scenario in scenarios:
    data = load_comparison(scenario)
    if data:
        print_scenario_summary(scenario, data)

print()
EOF
fi

echo ""
echo "============================================================"
echo "Quick Run Commands"
echo "============================================================"
echo ""
echo "# Run specific scenario:"
echo "SCENARIO=memory-scaling ./benchmarks/run_streaming_video_benchmark.sh"
echo ""
echo "# Run with more frames (long video test):"
echo "NUM_FRAMES=256 SCENARIO=long-video ./benchmarks/run_streaming_video_benchmark.sh"
echo ""
echo "# Run hybrid-only benchmark:"
echo "RUN_COMPARISON=false SKIP_STANDARD=true ./benchmarks/run_streaming_video_benchmark.sh"
echo ""
echo "# Run with 7B model:"
echo "./benchmarks/run_streaming_video_benchmark.sh Qwen/Qwen2.5-VL-7B-Instruct"
echo ""
echo "============================================================"
echo "Understanding the Results"
echo "============================================================"
echo ""
echo "Key metrics to look for:"
echo ""
echo "1. MEMORY GROWTH RATE (MiB/frame):"
echo "   - Standard attention: Should grow linearly (~0.5-2 MiB/frame)"
echo "   - Hybrid SSM: Should be near-zero (O(1) memory)"
echo ""
echo "2. PEAK MEMORY:"
echo "   - Hybrid should show significant reduction (30-70%)"
echo ""
echo "3. CONCURRENT QUERY THROUGHPUT:"
echo "   - Hybrid enables higher QPS with shared SSM state"
echo ""
echo "4. MEMORY SCALING PLOT:"
echo "   - Standard: Linear growth curve"
echo "   - Hybrid: Flat/constant memory line"
echo ""

