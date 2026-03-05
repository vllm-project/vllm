# Hybrid SSM + Sliding-Window Attention Benchmarking Suite

This benchmarking suite compares three attention configurations to evaluate the memory and performance trade-offs of the hybrid SSM + sliding-window attention approach.

## Configurations

| Configuration | Description | Use Case |
|---------------|-------------|----------|
| `full` | Full attention with complete KV cache | Baseline - maximum quality |
| `sliding` | Sliding window attention without SSM | Memory-efficient, limited context |
| `hybrid` | Sliding window + SSM history branch | Memory-efficient with extended context |

## Quick Start

### 1. Run All Benchmarks

```bash
# Make the script executable
chmod +x benchmarks/run_hybrid_benchmark.sh

# Run with default settings (Llama-3.2-1B)
./benchmarks/run_hybrid_benchmark.sh

# Run with a specific model
./benchmarks/run_hybrid_benchmark.sh meta-llama/Llama-3.2-3B ./my_results

# Run with custom settings
INPUT_LENGTHS="256,512,1024" NUM_PROMPTS=20 ./benchmarks/run_hybrid_benchmark.sh
```

### 2. Run Individual Configurations

```bash
# Full attention baseline
python benchmarks/benchmark_hybrid_attention.py \
    --model meta-llama/Llama-3.2-1B \
    --config full \
    --input-lengths 512,1024,2048,4096 \
    --num-prompts 50 \
    --output-json results/full_results.json

# Sliding window only
python benchmarks/benchmark_hybrid_attention.py \
    --model meta-llama/Llama-3.2-1B \
    --config sliding \
    --sliding-window-override 2048 \
    --input-lengths 512,1024,2048,4096 \
    --num-prompts 50 \
    --output-json results/sliding_results.json

# Hybrid SSM + sliding window
python benchmarks/benchmark_hybrid_attention.py \
    --model meta-llama/Llama-3.2-1B \
    --config hybrid \
    --sliding-window-override 2048 \
    --input-lengths 512,1024,2048,4096 \
    --num-prompts 50 \
    --output-json results/hybrid_results.json
```

### 3. Generate Visualizations

```bash
python benchmarks/visualize_hybrid_benchmark.py \
    --results-dir ./results \
    --output-dir ./results/plots \
    --format png
```

## Benchmark Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Required | HuggingFace model path or name |
| `--config` | Required | Configuration: `full`, `sliding`, or `hybrid` |
| `--input-lengths` | `512,1024,2048,4096` | Comma-separated input lengths |
| `--num-prompts` | `100` | Prompts per input length |
| `--output-len` | `128` | Output tokens per request |
| `--num-warmup-iters` | `3` | Warmup iterations |
| `--num-benchmark-iters` | `10` | Benchmark iterations |
| `--gpu-memory-utilization` | `0.9` | GPU memory utilization |
| `--sliding-window-override` | `4096` | Override sliding window size |
| `--dtype` | `auto` | Model dtype |
| `--seed` | `42` | Random seed |

## Output Files

The benchmark generates:

```
results/
├── full_results.json       # Full attention results
├── sliding_results.json    # Sliding window results
├── hybrid_results.json     # Hybrid attention results
└── plots/
    ├── memory_comparison.png
    ├── throughput_comparison.png
    ├── latency_percentiles.png
    ├── latency_vs_input_length.png
    ├── memory_efficiency.png
    └── summary.md
```

## Metrics Collected

### Memory Metrics
- **KV Cache Memory**: Memory allocated for KV cache (GiB)
- **Model Memory**: Memory used by model weights (GiB)
- **Peak GPU Memory**: Maximum GPU memory usage (GiB)

### Throughput Metrics
- **Tokens/second**: Total tokens processed per second
- **Requests/second**: Requests completed per second

### Latency Metrics
- **Average latency**: Mean request latency (ms)
- **P50/P90/P99 latency**: Percentile latencies (ms)

## Example Results

After running the benchmarks, you'll see output like:

```
Configuration          Input Len   Throughput      Avg Latency
--------------------------------------------------------------
Full Attention         512         5000.0 tok/s    150.0 ms
Full Attention         1024        4200.0 tok/s    180.0 ms
Sliding Window         512         5500.0 tok/s    140.0 ms
Sliding Window         1024        4800.0 tok/s    160.0 ms
Hybrid (SSM+SW)        512         5200.0 tok/s    145.0 ms
Hybrid (SSM+SW)        1024        4500.0 tok/s    170.0 ms
```

## Recommended Models for Testing

| GPU Memory | Recommended Model |
|------------|-------------------|
| 24GB (RTX 3090/4090) | `meta-llama/Llama-3.2-1B` |
| 40GB (A100-40GB) | `meta-llama/Llama-3.2-3B` |
| 80GB (A100-80GB/H100) | `meta-llama/Llama-3.1-8B` or `mistralai/Mistral-7B-v0.1` |

## Test Data Generation

Generate custom test datasets:

```bash
# Random prompts
python benchmarks/hybrid_benchmark_data.py \
    --tokenizer meta-llama/Llama-3.2-1B \
    --input-lengths 512,1024,2048,4096 \
    --num-prompts 100 \
    --pattern random \
    --output-file test_prompts.json

# Shared prefix prompts (for prefix caching tests)
python benchmarks/hybrid_benchmark_data.py \
    --tokenizer meta-llama/Llama-3.2-1B \
    --input-lengths 512,1024,2048,4096 \
    --num-prompts 100 \
    --pattern shared_prefix \
    --output-file shared_prefix_prompts.json
```

## Interpreting Results

### Memory Efficiency
The hybrid approach should show:
- **Lower KV cache memory** compared to full attention
- **Similar or slightly higher model memory** (due to SSM parameters)
- **Better throughput per GiB** of memory

### Throughput
- Hybrid should be **comparable to sliding window** on short sequences
- Hybrid should **outperform full attention** on long sequences (memory-bound)

### Latency
- Hybrid adds **minimal overhead** for SSM computation
- P99 latency should remain close to sliding window baseline

## Troubleshooting

### Out of Memory
```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.8

# Use smaller batch size (fewer prompts)
--num-prompts 20

# Use shorter sequences
--input-lengths 256,512,1024
```

### Model Loading Issues
```bash
# Trust remote code for custom models
--trust-remote-code

# Specify max model length
--max-model-len 4096
```

## Files in This Suite

| File | Description |
|------|-------------|
| `benchmark_hybrid_attention.py` | Main benchmark script |
| `visualize_hybrid_benchmark.py` | Visualization generator |
| `hybrid_benchmark_data.py` | Test data generator |
| `run_hybrid_benchmark.sh` | Convenience runner script |
| `README_hybrid_benchmark.md` | This documentation |

