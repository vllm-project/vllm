#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Hybrid Architecture Evaluation Suite
#
# This script runs the comprehensive evaluation procedure defined in the research plan.
# It executes three key experiments to validate the "Zero-Init" Hybrid Architecture:
# 1. Memory Scaling (O(1) proof)
# 2. Throughput Analysis
# 3. Long Video Stability
#
# Usage:
#   ./benchmarks/run_evaluation.sh [MODEL_PATH]
#

set -e

MODEL_PATH="${1:-Qwen/Qwen2.5-VL-3B-Instruct}"
OUTPUT_DIR="./evaluation_results"
mkdir -p "$OUTPUT_DIR"

# Enable the benchmark mode for Zero-Init inference
export VLLM_HYBRID_SSM_MODE="benchmark"
# Fix import path for benchmarks module
export PYTHONPATH=$PYTHONPATH:.

echo "================================================================="
echo "Starting Hybrid Architecture Evaluation"
echo "Model: $MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Mode: Zero-Init Benchmark (SSM execution enabled, output zeroed)"
echo "================================================================="

# Check requirements
if ! python3 -c "import cv2" &> /dev/null; then
    echo "Error: opencv-python is required. Please install it."
    exit 1
fi

echo ""
echo "-----------------------------------------------------------------"
echo "Experiment 1: Memory Scaling Verification"
echo "Comparison: Standard Attention vs. Hybrid (SSM+SW)"
echo "Hypothesis: Hybrid memory growth should be near-zero per frame."
echo "-----------------------------------------------------------------"

python -m benchmarks.streaming_video_benchmark \
    --model "$MODEL_PATH" \
    --scenario memory-scaling \
    --num-frames 64 \
    --compare-modes \
    --output-file "$OUTPUT_DIR/exp1_memory_scaling.json" \
    --gpu-memory-utilization 0.9

echo ""
echo "-----------------------------------------------------------------"
echo "Experiment 2: Throughput Analysis (Concurrent Queries)"
echo "Scenario: 5 concurrent queries on a 32-frame video"
echo "Hypothesis: Shared SSM state enables high throughput."
echo "-----------------------------------------------------------------"

python -m benchmarks.streaming_video_benchmark \
    --model "$MODEL_PATH" \
    --scenario multi-query \
    --use-hybrid-attention \
    --num-frames 32 \
    --output-file "$OUTPUT_DIR/exp2_throughput_hybrid.json"

# Run baseline for comparison
python -m benchmarks.streaming_video_benchmark \
    --model "$MODEL_PATH" \
    --scenario multi-query \
    --num-frames 32 \
    --output-file "$OUTPUT_DIR/exp2_throughput_standard.json"


echo ""
echo "-----------------------------------------------------------------"
echo "Experiment 3: Long Video Stability Stress Test"
echo "Scenario: 128 frames (simulated long context)"
echo "Hypothesis: Successful completion without OOM."
echo "-----------------------------------------------------------------"

python -m benchmarks.streaming_video_benchmark \
    --model "$MODEL_PATH" \
    --scenario long-video \
    --use-hybrid-attention \
    --num-frames 128 \
    --output-file "$OUTPUT_DIR/exp3_stability.json"

echo ""
echo "================================================================="
echo "Evaluation Complete!"
echo " Results saved to $OUTPUT_DIR"
echo "================================================================="
echo "Summary of Findings:"
echo "1. Memory Scaling: Check exp1_memory_scaling.json for 'memory_savings_percent'"
echo "2. Throughput: Compare exp2_throughput_hybrid.json vs standard"
echo "3. Stability: Check exp3_stability.json for successful completion"
echo "================================================================="
