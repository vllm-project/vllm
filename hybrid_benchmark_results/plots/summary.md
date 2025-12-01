# Hybrid Attention Benchmark Summary

## Configuration Comparison

| Configuration | Input Length | Throughput (tok/s) | Avg Latency (ms) | P99 Latency (ms) | KV Cache (GiB) |
|--------------|--------------|-------------------|------------------|------------------|----------------|
| Full Attention | 512 | 22659.8 | 1412.2 | 1728.7 | None |
| Full Attention | 1024 | 33606.4 | 1714.0 | 2415.3 | None |
| Full Attention | 2048 | 46759.5 | 2326.8 | 3787.1 | None |
| Full Attention | 4096 | 58144.1 | 3632.4 | 6876.9 | None |
| Sliding Window | 512 | 15190.4 | 2106.6 | 2417.9 | None |
| Sliding Window | 1024 | 20773.4 | 2772.8 | 3437.9 | None |
| Sliding Window | 2048 | 26154.9 | 4159.8 | 5554.0 | None |
| Sliding Window | 4096 | 25562.1 | 8262.2 | 11134.5 | None |
| Hybrid (SSM + SW) | 512 | 15140.4 | 2113.5 | 2428.2 | None |
| Hybrid (SSM + SW) | 1024 | 20769.9 | 2773.2 | 3446.7 | None |
| Hybrid (SSM + SW) | 2048 | 26158.5 | 4159.3 | 5560.5 | None |
| Hybrid (SSM + SW) | 4096 | 25546.6 | 8267.2 | 11150.1 | None |