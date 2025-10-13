# vLLM Attention Benchmarking Suite

Fast, flexible benchmarking for vLLM attention and MLA backends with an extended batch specification grammar.

## Quick Start

```bash
cd benchmarks/attention_benchmarks

# Test the parser
python test_batch_spec.py
# ✓ All tests pass

# Run one of the 4 research studies
python benchmark.py --config configs/cutlass_numsplits.yaml
python benchmark.py --config configs/hopper_head_count.yaml
python benchmark.py --config configs/flashinfer_vs_cutlass.yaml
python benchmark.py --config configs/reorder_threshold.yaml

# Or run custom benchmarks
python benchmark.py \
    --backends flash flashinfer \
    --batch-specs "q2k" "8q1kv1k" "2q2k_32q1kv1k" \
    --output-csv results.csv
```

## Simplified Batch Specification Grammar

Express workloads concisely using query length and KV cache size:

```python
"q2k"              # 2048-token prefill (q_len=2048, kv_len=2048)
"q1kv1k"           # Decode: 1 token with 1K KV cache
"8q1kv1k"          # 8 decode requests
"q4kv1k"           # 4-token extend (e.g., spec decode)
"2q2k_32q1kv1k"    # Mixed: 2 prefills + 32 decodes
"16q4kv1k"         # 16 spec decode (4 tokens each)
```

### Grammar Rule

```
Format: (<count>?) q<q_len>(k?) (kv<kv_len>(k?))?

- count:  Number of identical requests (optional, default=1)
- q_len:  Query length (number of new tokens)
- kv_len: Total KV cache length (optional, defaults to q_len for prefill)
- 'k':    Multiplies value by 1024

Mixed batches: Use _ to combine (e.g., "2q2k_32q1kv1k")
```

**Note**: Decode, prefill, and spec decode are just different query lengths - no special syntax needed!

## Research Studies

The suite includes 4 pre-configured studies to answer key MLA optimization questions. Each study is a single YAML file you can run directly:

### Study 1: CUTLASS MLA num-splits Optimization

**Question:** Should we revert the CUTLASS MLA num-splits heuristic (PRs #24966, #25509)?

```bash
python benchmark.py --config configs/cutlass_numsplits.yaml
```

Tests CUTLASS MLA with different `num_kv_splits` values (1, 2, 4, 8, 16, 32) across various batch sizes and compares against auto-selection.

### Study 2: FlashAttn MLA vs FlashMLA on Hopper

**Question:** Does head count matter for FlashAttn MLA vs FlashMLA on Hopper GPUs?

```bash
# Test with default head count (128)
python benchmark.py --config configs/hopper_head_count.yaml

# Test with different head counts
for heads in 16 32 64 128 256; do
    python benchmark.py --config configs/hopper_head_count.yaml \
        --num-q-heads $heads \
        --output-csv hopper_heads_${heads}.csv
done
```

Compares FlashAttn MLA and FlashMLA performance with varying attention head counts.

### Study 3: FlashInfer-MLA vs Optimized CUTLASS

**Question:** Is FlashInfer-MLA better than CUTLASS MLA after num-splits optimization?

```bash
python benchmark.py --config configs/flashinfer_vs_cutlass.yaml
```

Compares FlashInfer-MLA against CUTLASS MLA with optimized `num_kv_splits` values.

### Study 4: Reorder Batch Threshold Optimization (Decode vs Prefill Crossover)

**Question:** At what query length does the prefill pipeline become faster than the decode pipeline?

**Methodology:** Reproduces the original `benchmark_mla_threshold.py` study using the new interface:
- For each query length (1-2048), test BOTH decode and prefill pipelines
- Find the crossover point where prefill becomes faster
- Analyze how this varies across batch sizes (1-256)

```bash
python benchmark.py --config configs/reorder_threshold.yaml
```

Tests query lengths from 1-2048 (fine-grained steps at low values, coarser at high values) across 9 batch sizes. For each query length, compares:
- **Decode pipeline**: `threshold >= query_length`
- **Prefill pipeline**: `threshold < query_length`

Outputs the optimal threshold (last query length where decode is faster) for each batch size.

---

## Universal Benchmark

The `benchmark.py` script handles **all** backends - both standard attention and MLA.

### Standard Attention (Flash/Triton/FlashInfer)

```bash
python benchmark.py \
    --backends flash triton flashinfer \
    --batch-specs "q2k" "8q1kv1k" "2q2k_32q1kv1k" \
    --num-layers 10 \
    --repeats 5 \
    --output-csv results.csv
```

### MLA Backends

```bash
# Compare all MLA backends
python benchmark.py \
    --backends cutlass_mla flashinfer_mla flash_attn_mla flashmla \
    --batch-specs "64q1kv1k" "64q1kv4k" \
    --output-csv mla_results.csv
```

### Parameter Sweeps

#### CUTLASS MLA num-splits Optimization

```bash
python benchmark.py \
    --backend cutlass_mla \
    --batch-specs "64q1kv1k" "64q1kv4k" "64q1kv16k" \
    --num-splits 1 2 4 8 16 \
    --compare-auto \
    --output-json optimal_splits.json
```

**Answers:** What is the optimal `num_kv_splits` for CUTLASS MLA?

#### Reorder Batch Threshold Optimization

```bash
python benchmark.py \
    --backend flashmla \
    --batch-specs "q4kv1k" "q8kv2k" \
    --thresholds 1 4 16 64 256 512 \
    --output-csv threshold_sweep.csv
```

**Answers:** What's the optimal `reorder_batch_threshold` for speculative decoding?

### All Command-Line Options

```
--backends BACKEND [BACKEND ...]    # flash, triton, flashinfer, cutlass_mla,
                                    # flashinfer_mla, flash_attn_mla, flashmla
--backend BACKEND                   # Single backend (alternative to --backends)
--batch-specs SPEC [SPEC ...]       # Batch specifications (default: ["q2k", "8q1kv1k"])

# Model configuration
--num-layers N                      # Number of layers (default: 10)
--head-dim N                        # Head dimension (default: 128)
--num-q-heads N                     # Query heads (default: 32)
--num-kv-heads N                    # KV heads (default: 8)
--block-size N                      # Block size (default: 16)

# Benchmark settings
--device DEVICE                     # Device (default: cuda:0)
--repeats N                         # Repetitions (default: 1)
--warmup-iters N                    # Warmup iterations (default: 3)
--profile-memory                    # Profile memory usage

# MLA-specific parameter sweeps
--num-splits N [N ...]              # CUTLASS MLA: Test multiple num_kv_splits
--thresholds N [N ...]              # FlashMLA/FlashAttn MLA: Test multiple thresholds
--compare-auto                      # CUTLASS MLA: Also test auto num_kv_splits

# Output
--output-csv FILE                   # Save to CSV
--output-json FILE                  # Save to JSON
```

## Hardware Requirements

| Backend | Hardware |
|---------|----------|
| Flash/Triton/FlashInfer | Any CUDA GPU |
| CUTLASS MLA | Blackwell (SM100+) |
| FlashAttn MLA | Hopper (SM90+) |
| FlashMLA | Hopper (SM90+) |
| FlashInfer-MLA | Any CUDA GPU |

## Using MLA Runner Directly

All MLA backends are available in `mla_runner.py`:

```python
from mla_runner import (
    run_cutlass_mla_benchmark,
    run_flashinfer_mla_benchmark,
    run_flashattn_mla_benchmark,
    run_flashmla_benchmark,
)
from common import BenchmarkConfig

config = BenchmarkConfig(
    backend="cutlass_mla",
    batch_spec="64q1kv4k",
    num_layers=10,
    head_dim=576,
    num_q_heads=128,
    num_kv_heads=1,
    block_size=128,
    device="cuda:0",
    repeats=5,
    warmup_iters=3,
)

# CUTLASS MLA with specific num_kv_splits
result = run_cutlass_mla_benchmark(config, num_kv_splits=4)
print(f"Time: {result['mean']:.6f}s, Throughput: {result['throughput']:.1f} tok/s")

# FlashInfer-MLA
result = run_flashinfer_mla_benchmark(config)

# FlashAttn MLA (Hopper SM90+)
result = run_flashattn_mla_benchmark(config, reorder_batch_threshold=64)

# FlashMLA (Hopper SM90+)
result = run_flashmla_benchmark(config, reorder_batch_threshold=64)
```

## Python API

```python
from batch_spec import parse_batch_spec, format_batch_spec, get_batch_stats
from common import BenchmarkConfig, BenchmarkResult, ResultsFormatter

# Parse batch specs
requests = parse_batch_spec("2q2k_q4kv1k_32s1k")
print(format_batch_spec(requests))
# "2 prefill (2x2k), 1 specdecode (1xq4s1k), 32 decode (32x1k)"

# Get batch statistics
stats = get_batch_stats(requests)
print(f"Total tokens: {stats['total_tokens']}")
print(f"Num decode: {stats['num_decode']}, Num prefill: {stats['num_prefill']}")

# Format results
formatter = ResultsFormatter()
formatter.save_csv(results, "output.csv")
formatter.save_json(results, "output.json")
```

## File Structure

```
attention_benchmarks/
├── README.md                      # This file
│
├── batch_spec.py                  # Grammar parser (tested)
├── common.py                      # Infrastructure
├── runner.py                      # Standard attention helpers
├── mla_runner.py                  # MLA helpers (ALL 4 backends)
├── test_batch_spec.py             # Tests (all passing)
│
├── benchmark.py                   # Universal benchmark script
│
└── configs/                       # Pre-configured studies
    ├── cutlass_numsplits.yaml         # CUTLASS num-splits optimization
    ├── hopper_head_count.yaml         # FlashAttn vs FlashMLA head count
    ├── flashinfer_vs_cutlass.yaml     # FlashInfer vs optimized CUTLASS
    └── reorder_threshold.yaml         # Reorder threshold optimization
```

## Tips

**1. Warmup matters** - Use `--warmup-iters 10` for stable results

**2. Multiple repeats** - Use `--repeats 20` for low variance

**3. Save results** - Always use `--output-csv` or `--output-json`

**4. Test incrementally** - Start with `--num-layers 1 --repeats 1`

**5. Extended grammar** - Leverage spec decode, chunked prefill patterns

**6. Parameter sweeps** - Use `--num-splits` or `--thresholds` to find optimal values

## Troubleshooting

**Import errors?**
```bash
source /path/to/vllm/.venv/bin/activate
```

**Backend not supported?**
- Check hardware requirements above
- Some backends need Hopper/Blackwell

**OOM?**
- Reduce batch size: `"32s1k"` → `"16s1k"`
- Reduce sequence length: `"64q1kv16k"` → `"64q1kv4k"`

## What's Included

✅ Extended batch spec grammar with tests (all passing!)
✅ Universal benchmark script for all backends
✅ Standard attention support (Flash/Triton/FlashInfer)
✅ MLA runner with ALL 4 backends
✅ Parameter sweep modes (num-splits, thresholds)
✅ Rich console output + CSV/JSON export
✅ Pre-built configuration files (optional)

**~5,000 lines of code, fully simplified, ready to benchmark!** 🚀
