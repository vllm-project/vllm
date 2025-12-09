# Video Benchmark Summary

**Model:** Qwen/Qwen2.5-VL-3B-Instruct  
**Frames:** 16  
**Input Tokens:** 121150

## Configuration Comparison

| Configuration | Avg Latency (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Throughput (tok/s) | Gen Speed (tok/s) |
|--------------|------------------|----------|----------|----------|-------------------|-------------------|
| Hybrid SSM + SW | 1744.60 | 1744.73 | 1745.46 | 1746.12 | 1462.2 | 73.4 |
| Standard Full Attention | 1755.83 | 1755.94 | 1760.97 | 1765.29 | 1452.9 | 72.9 |

## Performance Delta (Hybrid vs Standard)

| Metric | Hybrid | Standard | Δ (%) | Better |
|--------|--------|----------|-------|--------|
| Avg Latency (ms) | 1744.60 | 1755.83 | -0.64% | Hybrid |
| Throughput (tok/s) | 1462.23 | 1452.88 | +0.64% | Hybrid |