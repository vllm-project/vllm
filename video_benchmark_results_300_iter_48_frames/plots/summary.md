# Video Benchmark Summary

**Model:** Qwen/Qwen2.5-VL-3B-Instruct  
**Frames:** 48  
**Input Tokens:** 2162100

## Configuration Comparison

| Configuration | Avg Latency (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Throughput (tok/s) | Gen Speed (tok/s) |
|--------------|------------------|----------|----------|----------|-------------------|-------------------|
| Hybrid SSM + SW | 1869.40 | 1864.40 | 1882.76 | 1917.41 | 3923.7 | 68.5 |
| Standard Full Attention | 1861.98 | 1861.01 | 1865.91 | 1874.79 | 3939.3 | 68.7 |

## Performance Delta (Hybrid vs Standard)

| Metric | Hybrid | Standard | Δ (%) | Better |
|--------|--------|----------|-------|--------|
| Avg Latency (ms) | 1869.40 | 1861.98 | +0.40% | Standard |
| Throughput (tok/s) | 3923.72 | 3939.35 | -0.40% | Standard |