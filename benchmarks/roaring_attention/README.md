# RoaringBitmap Sparse Attention Benchmarks

## Overview

This directory contains benchmarks comparing different sparse attention mask implementations for vLLM.

## Key Findings

### ✅ When RoaringBitmaps Win

RoaringBitmaps are effective for:
- **Structured sparsity patterns** (sliding windows, block-local patterns)
- **Long contexts** (>50k tokens) where O(n²) memory becomes prohibitive
- **Frequent set operations** (finding attention overlaps between sequences)
- **Distributed systems** (efficient serialization)

### ❌ When RoaringBitmaps Don't Help

Avoid RoaringBitmaps for:
- **Dense attention** (<80% sparsity)
- **Truly random sparse patterns** (no compression benefit)
- **Small contexts** (<10k tokens, overhead not worth it)
- **Simple allocate/free patterns** (see failed block pool attempt)

## Benchmark Results

### Configuration
- 1,000 blocks (16k tokens)
- Sliding window: 128 blocks
- Global attention: 16 blocks  
- Random sparse: ~1% additional connections
- **Actual density: 15.4%** (84.6% sparse)

### Memory Usage

| Method | Memory | Relative to Dense |
|--------|--------|-------------------|
| Dense Tensor | 976 KB | 1.00x |
| RoaringBitmap | 674 KB | 0.69x |
| Python COO | 3.78 MB | 3.98x |

**Note**: Python COO performs poorly due to set/dict overhead. A C++ implementation would be more competitive.

### Operation Performance

| Operation | Dense | Roaring | Speedup |
|-----------|-------|---------|---------|
| Query (100x) | 3.1ms | 0.4ms | 7.5x |
| Intersection (100x) | 1.7ms | 0.3ms | 5.4x |

### Scaling Projections

For 1M tokens (62,500 blocks):
- **Dense**: 3.64 GB (O(n²) growth)
- **RoaringBitmap**: 66-200 MB (depends on pattern)
- **Savings**: 94-98% for structured patterns

## Running the Benchmarks

```bash
# Original benchmark (has measurement issues)
python sparse_attention_benchmark.py --num-blocks 1000

# Corrected benchmark with accurate measurements
python sparse_attention_benchmark_fixed.py --num-blocks 1000

# Test different sparsity patterns
python sparse_attention_benchmark_fixed.py --num-blocks 1000 --window-size 256 --sparsity 0.95
```

## Implementation Recommendations

1. **Start with opt-in flag**: `--use-roaring-attention-masks`
2. **Profile with real models**: Results vary by attention pattern
3. **Consider hybrid approach**: Use RoaringBitmaps only for layers with structured sparsity
4. **Monitor memory usage**: Ensure actual savings before committing

## Files

- `sparse_attention_benchmark.py`: Original benchmark (flawed measurements)
- `sparse_attention_benchmark_fixed.py`: Corrected benchmark with accurate memory tracking
- `test_roaring_attention.py`: Unit tests for RoaringAttentionMask class

## Dependencies

```bash
pip install pyroaring torch numpy
```

## Limitations

1. **Pattern-dependent benefits**: Highly dependent on attention structure
2. **Python overhead**: C++ implementation would show better results
3. **GPU transfer cost**: Not measured, needs real integration testing
4. **Dynamic patterns**: Fixed patterns show best compression

## Future Work

- Benchmark with actual model attention patterns (Mistral, Llama)
- Test GPU memory transfer overhead
- Implement C++ version for production
- Profile with dynamic/learned sparsity patterns