# FlashAttention v2.8.3 Scaling Benchmark

**Model:** mistralai/Mistral-7B-v0.1
**GPU:** NVIDIA H100 80GB HBM3
**Versions:** vLLM `0.15.1`, FlashAttention `2.8.3`
**Batch Size:** 64
**Date:** February 5, 2026
**Backend:** FLASH_ATTN

## Results

| Sequence Length | Time (s) | Throughput (prompts/sec) | Performance vs Peak |
|-----------------|----------|--------------------------|---------------------|
| 1,024           | 0.1003   | 638.20                  | 84%                 |
| 2,048           | 0.0892   | 717.64                  | 95%                 |
| **4,096**       | 0.0843   | **758.96** ⭐           | **100%**            |
| 8,192           | 0.1167   | 548.35                  | 72%                 |
| 16,384          | 0.1832   | 349.27                  | 46%                 |
| 32,768          | 0.3505   | 182.59                  | 24%                 |

## Key Observations

- **O(N²) scaling behavior confirmed:** 8x sequence length increase (4K→32K) results in 4.2x throughput reduction
- **Peak performance at 4K tokens:** 758.96 prompts/sec
- **76% performance degradation at 32K tokens:** from 758.96 to 182.59 prompts/sec
- **Memory:** 13.5 GiB model weights, 48.0 GiB available KV cache

## Test Configuration (using `vllm_longcontext_benchmark.py`)

```python
batch_size = 64
seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
gpu_memory_utilization = 0.8
enforce_eager = True
