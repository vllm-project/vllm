# Video Benchmark Summary

**Model:** Qwen/Qwen2.5-VL-3B-Instruct  
**Frames:** 48  
**Input Tokens:** 36035

## Configuration Comparison

| Configuration | Avg Latency (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Throughput (tok/s) | Gen Speed (tok/s) |
|--------------|------------------|----------|----------|----------|-------------------|-------------------|
| Hybrid Attention | 1821.22 | 1818.65 | 1826.09 | 1829.03 | 4027.5 | 70.3 |
| Standard Attention | 1828.40 | 1826.88 | 1833.70 | 1835.80 | 4011.7 | 70.0 |

## Performance Delta (Hybrid vs Standard)

| Metric | Hybrid | Standard | Δ (%) | Better |
|--------|--------|----------|-------|--------|
| Avg Latency (ms) | 1821.22 | 1828.40 | -0.39% | Hybrid ✓ |
| Throughput (tok/s) | 4027.52 | 4011.71 | +0.39% | Hybrid ✓ |