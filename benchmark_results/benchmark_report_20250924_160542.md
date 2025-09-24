# MLFQ vs Default Scheduler Benchmark Report
============================================================

## Summary

| Workload | Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Success Rate |
|----------|----------|-----------|-------------------|-----------------|-----------------|--------------|
| short_requests_5 | 5 | MLFQ | 439.75 | 0.202 | 0.202 | 100.00% |
| short_requests_5 | 5 | Default | 488.47 | 0.182 | 0.182 | 100.00% |
| mixed_length_5 | 5 | MLFQ | 765.72 | 0.326 | 0.326 | 100.00% |
| mixed_length_5 | 5 | Default | 765.16 | 0.327 | 0.327 | 100.00% |
| long_requests_5 | 5 | MLFQ | 823.07 | 0.607 | 0.607 | 100.00% |
| long_requests_5 | 5 | Default | 807.20 | 0.619 | 0.619 | 100.00% |
| burst_load_5 | 5 | MLFQ | 681.76 | 0.220 | 0.220 | 100.00% |
| burst_load_5 | 5 | Default | 626.96 | 0.239 | 0.239 | 100.00% |

## Detailed Analysis

### Short Requests

| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |
|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|
| 5 | MLFQ | 439.75 | 0.202 | 0.202 | -10.0% | -11.1% |
| 5 | Default | 488.47 | 0.182 | 0.182 | - | - |

### Mixed Length

| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |
|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|
| 5 | MLFQ | 765.72 | 0.326 | 0.326 | +0.1% | +0.1% |
| 5 | Default | 765.16 | 0.327 | 0.327 | - | - |

### Long Requests

| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |
|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|
| 5 | MLFQ | 823.07 | 0.607 | 0.607 | +2.0% | +1.9% |
| 5 | Default | 807.20 | 0.619 | 0.619 | - | - |

### Burst Load

| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |
|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|
| 5 | MLFQ | 681.76 | 0.220 | 0.220 | +8.7% | +8.0% |
| 5 | Default | 626.96 | 0.239 | 0.239 | - | - |

## Overall Conclusion

**Overall Performance:**
- Average Throughput Improvement: +0.8%
- Average Latency Improvement: +0.8%

✅ MLFQ scheduler shows improved throughput performance.
✅ MLFQ scheduler shows improved latency performance.