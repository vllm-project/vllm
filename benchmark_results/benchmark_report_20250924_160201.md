# MLFQ vs Default Scheduler Benchmark Report
============================================================

## Summary

| Workload | Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Success Rate |
|----------|----------|-----------|-------------------|-----------------|-----------------|--------------|
| short_requests_100 | 100 | MLFQ | 835.22 | 0.231 | 0.306 | 100.00% |
| short_requests_100 | 100 | Default | 842.26 | 0.229 | 0.299 | 100.00% |
| mixed_length_100 | 100 | MLFQ | 1322.66 | 0.288 | 0.329 | 100.00% |
| mixed_length_100 | 100 | Default | 1313.38 | 0.289 | 0.356 | 100.00% |
| long_requests_100 | 100 | MLFQ | 868.52 | 0.558 | 0.622 | 100.00% |
| long_requests_100 | 100 | Default | 874.55 | 0.554 | 0.598 | 100.00% |
| burst_load_100 | 100 | MLFQ | 1167.20 | 0.513 | 0.565 | 100.00% |
| burst_load_100 | 100 | Default | 1165.44 | 0.514 | 0.566 | 100.00% |

## Detailed Analysis

### Short Requests

| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |
|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|
| 100 | MLFQ | 835.22 | 0.231 | 0.306 | -0.8% | -0.8% |
| 100 | Default | 842.26 | 0.229 | 0.299 | - | - |

### Mixed Length

| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |
|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|
| 100 | MLFQ | 1322.66 | 0.288 | 0.329 | +0.7% | +0.4% |
| 100 | Default | 1313.38 | 0.289 | 0.356 | - | - |

### Long Requests

| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |
|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|
| 100 | MLFQ | 868.52 | 0.558 | 0.622 | -0.7% | -0.7% |
| 100 | Default | 874.55 | 0.554 | 0.598 | - | - |

### Burst Load

| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |
|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|
| 100 | MLFQ | 1167.20 | 0.513 | 0.565 | +0.2% | +0.1% |
| 100 | Default | 1165.44 | 0.514 | 0.566 | - | - |

## Overall Conclusion

**Overall Performance:**
- Average Throughput Improvement: -0.0%
- Average Latency Improvement: -0.2%

❌ MLFQ scheduler shows decreased throughput performance.
❌ MLFQ scheduler shows increased latency.