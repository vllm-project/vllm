# Hybrid Attention Benchmark Summary

## Configuration Comparison

| Configuration | Input Length | Throughput (tok/s) | Avg Latency (ms) | P99 Latency (ms) | KV Cache (GiB) |
|--------------|--------------|-------------------|------------------|------------------|----------------|
| Full Attention | 512 | 23080.8 | 1386.4 | 1706.8 | None |
| Full Attention | 1024 | 33903.0 | 1699.0 | 2402.8 | None |
| Full Attention | 2048 | 46923.9 | 2318.6 | 3805.8 | None |
| Full Attention | 4096 | 58209.3 | 3628.3 | 6921.1 | None |
| Sliding Window | 512 | 15305.8 | 2090.7 | 2408.5 | None |
| Sliding Window | 1024 | 20724.0 | 2779.4 | 3442.1 | None |
| Sliding Window | 2048 | 26234.1 | 4147.3 | 5551.3 | None |
| Sliding Window | 4096 | 25501.3 | 8281.9 | 11165.0 | None |
| Hybrid (SSM + SW) | 512 | 15302.1 | 2091.2 | 2411.2 | None |
| Hybrid (SSM + SW) | 1024 | 20854.8 | 2762.0 | 3440.9 | None |
| Hybrid (SSM + SW) | 2048 | 26196.9 | 4153.2 | 5569.1 | None |
| Hybrid (SSM + SW) | 4096 | 25499.4 | 8282.6 | 11172.3 | None |