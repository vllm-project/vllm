#!/bin/bash
# Example script for running the prefill/decode benchmark

echo "Running vLLM Prefill/Decode Benchmark Examples"
echo "=============================================="

# Basic benchmark with small configuration
echo "1. Basic benchmark (batch=1, input=512, output=128):"
torchrun --nproc-per-node=1 HTTP/benchmark_prefill_decode.py \
    --tensor-parallel-size 1 \
    --batch-size 1 \
    --input-length 512 \
    --output-length 128 \
    --num-iterations 3 \
    --warmup-iterations 1

echo -e "\n\n2. Larger batch benchmark (batch=4, input=1024, output=256):"
torchrun --nproc-per-node=1 HTTP/benchmark_prefill_decode.py \
    --tensor-parallel-size 1 \
    --batch-size 4 \
    --input-length 1024 \
    --output-length 256 \
    --num-iterations 5 \
    --warmup-iterations 2 \
    --output-file prefill_decode_batch4.csv

echo -e "\n\n3. With token parallelism (if supported):"
torchrun --nproc-per-node=1 HTTP/benchmark_prefill_decode.py \
    --tensor-parallel-size 1 \
    --enable-token-parallel \
    --token-parallel-size 1 \
    --batch-size 2 \
    --input-length 512 \
    --output-length 128 \
    --num-iterations 3 \
    --output-file prefill_decode_token_parallel.csv

echo -e "\n\nBenchmark completed! Check the CSV files for detailed results."
