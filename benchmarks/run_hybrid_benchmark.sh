#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Hybrid Attention Benchmark Runner
#
# This script runs comparative benchmarks for three attention configurations:
# 1. Full attention (baseline)
# 2. Sliding window only
# 3. Hybrid SSM + sliding window
#
# Usage:
#   ./benchmarks/run_hybrid_benchmark.sh [MODEL_PATH] [OUTPUT_DIR]
#
# Examples:
#   ./benchmarks/run_hybrid_benchmark.sh meta-llama/Llama-3.2-1B ./results
#   ./benchmarks/run_hybrid_benchmark.sh meta-llama/Llama-3.2-3B ./results
#   ./benchmarks/run_hybrid_benchmark.sh mistralai/Mistral-7B-v0.1 ./results

set -e

# Default configuration
MODEL_PATH="${1:-meta-llama/Llama-3.2-1B}"
OUTPUT_DIR="${2:-./hybrid_benchmark_results}"
INPUT_LENGTHS="${INPUT_LENGTHS:-512,1024,2048,4096}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_WARMUP="${NUM_WARMUP:-3}"
NUM_ITERS="${NUM_ITERS:-5}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
DTYPE="${DTYPE:-auto}"
SEED="${SEED:-42}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Hybrid Attention Benchmark Suite"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Input lengths: $INPUT_LENGTHS"
echo "Prompts per length: $NUM_PROMPTS"
echo "Output length: $OUTPUT_LEN"
echo "Warmup iterations: $NUM_WARMUP"
echo "Benchmark iterations: $NUM_ITERS"
echo "GPU memory utilization: $GPU_MEM_UTIL"
echo "Data type: $DTYPE"
echo "Random seed: $SEED"
echo "============================================================"

# Array of configurations to benchmark
CONFIGS=("full" "sliding" "hybrid")

# Run benchmarks for each configuration
for config in "${CONFIGS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running benchmark: $config"
    echo "============================================================"
    
    output_file="$OUTPUT_DIR/${config}_results.json"
    
    # Skip if results already exist (useful for resuming)
    if [ -f "$output_file" ] && [ "${SKIP_EXISTING:-false}" = "true" ]; then
        echo "Results already exist, skipping: $output_file"
        continue
    fi
    
    python benchmarks/benchmark_hybrid_attention.py \
        --model "$MODEL_PATH" \
        --config "$config" \
        --input-lengths "$INPUT_LENGTHS" \
        --num-prompts "$NUM_PROMPTS" \
        --output-len "$OUTPUT_LEN" \
        --num-warmup-iters "$NUM_WARMUP" \
        --num-benchmark-iters "$NUM_ITERS" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --dtype "$DTYPE" \
        --seed "$SEED" \
        --trust-remote-code \
        --output-json "$output_file"
    
    echo "Results saved to: $output_file"
done

echo ""
echo "============================================================"
echo "Generating visualizations"
echo "============================================================"

# Generate visualization plots
python benchmarks/visualize_hybrid_benchmark.py \
    --results-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/plots" \
    --format png

echo ""
echo "============================================================"
echo "Benchmark Complete!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "Plots saved to: $OUTPUT_DIR/plots"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  No JSON files found"
echo ""
ls -la "$OUTPUT_DIR/plots"/*.png 2>/dev/null || echo "  No plot files found"
echo ""

# Print summary if all results exist
if [ -f "$OUTPUT_DIR/full_results.json" ] && \
   [ -f "$OUTPUT_DIR/sliding_results.json" ] && \
   [ -f "$OUTPUT_DIR/hybrid_results.json" ]; then
    echo "============================================================"
    echo "Quick Summary"
    echo "============================================================"
    
    # Extract key metrics using Python
    python3 << 'EOF'
import json
import os

output_dir = os.environ.get('OUTPUT_DIR', './hybrid_benchmark_results')
configs = ['full', 'sliding', 'hybrid']
labels = {
    'full': 'Full Attention',
    'sliding': 'Sliding Window', 
    'hybrid': 'Hybrid (SSM+SW)'
}

print(f"{'Configuration':<20} {'Input Len':<12} {'Throughput':<15} {'Avg Latency':<15}")
print("-" * 62)

for config in configs:
    filepath = f"{output_dir}/{config}_results.json"
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = json.load(f)
        
        for length, result in sorted(data.get('by_input_length', {}).items(), key=lambda x: int(x[0])):
            if 'error' not in result:
                throughput = result.get('throughput', {}).get('tokens_per_second', 0)
                latency = result.get('latency', {}).get('avg_seconds', 0) * 1000
                print(f"{labels[config]:<20} {length:<12} {throughput:>10.1f} tok/s {latency:>10.1f} ms")
EOF
fi

echo ""
echo "For detailed analysis, see: $OUTPUT_DIR/plots/summary.md"

