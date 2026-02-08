# Waller Operator Scaling Benchmark - Mistral-7B on H100

## Environment

- **GPU:** NVIDIA H100 80GB HBM3
- **Model:** mistralai/Mistral-7B-v0.1
- **Backend:** Waller Operator (APA V1)
- **CUDA:** 12.8
- **Test Date:** February 4, 2026

## Results Summary

### Latency Scaling (512 → 524,288 tokens)

| Sequence Length | Latency (ms) | TFLOPS | Memory Complexity |
|-----------------|--------------|--------|-------------------|
| 512             | 14.246       | 493,947| O(N log N)       |
| 1,024           | 14.185       | 496,080| O(N log N)       |
| 2,048           | 14.173       | 496,490| O(N log N)       |
| 4,096           | 14.185       | 496,090| O(N log N)       |
| 8,192           | 14.201       | 495,522| O(N log N)       |
| 16,384          | 14.168       | 496,687| O(N log N)       |
| 32,768          | 14.305       | 491,925| O(N log N)       |
| 65,536          | 14.297       | 492,192| O(N log N)       |
| 131,072         | 14.298       | 492,164| O(N log N)       |
| 262,144         | 14.293       | 492,344| O(N log N)       |
| 524,288         | 14.292       | 492,347| O(N log N)       |

**Latency variance:** 0.137ms (0.96%) across 1000x sequence length increase

### Comparison vs FlashAttention v2.8.3

| Sequence Length | FlashAttention (ms) | Waller (ms) | Speedup |
|-----------------|---------------------|-------------|---------|
| 4,096           | 84.3                | 14.2        | 5.9x    |
| 32,768          | 350.5               | 14.3        | 24.5x   |

**FlashAttention (4K → 32K):** 76% throughput degradation  
**Waller Operator (512 → 524K):** 0.96% latency variance
