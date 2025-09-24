# MLFQ vs Default Scheduler Benchmark Report
============================================================

## Summary

| Workload | Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Success Rate |
|----------|----------|-----------|-------------------|-----------------|-----------------|--------------|
| short_requests_2 | 2 | MLFQ | 227.06 | 0.176 | 0.176 | 100.00% |
| short_requests_2 | 2 | Default | 245.02 | 0.163 | 0.163 | 100.00% |
| mixed_length_2 | 2 | MLFQ | 306.56 | 0.326 | 0.326 | 100.00% |
| mixed_length_2 | 2 | Default | 302.97 | 0.330 | 0.330 | 100.00% |
| long_requests_2 | 2 | MLFQ | 336.30 | 0.595 | 0.595 | 100.00% |
| long_requests_2 | 2 | Default | 317.73 | 0.629 | 0.629 | 100.00% |
| burst_load_2 | 2 | MLFQ | 273.82 | 0.219 | 0.219 | 100.00% |
| burst_load_2 | 2 | Default | 247.99 | 0.242 | 0.242 | 100.00% |

## Detailed Analysis

### Short Requests

| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |
|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|
| 2 | MLFQ | 227.06 | 0.176 | 0.176 | -7.3% | -7.9% |
| 2 | Default | 245.02 | 0.163 | 0.163 | - | - |

### Mixed Length

| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |
|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|
| 2 | MLFQ | 306.56 | 0.326 | 0.326 | +1.2% | +1.2% |
| 2 | Default | 302.97 | 0.330 | 0.330 | - | - |

### Long Requests

| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |
|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|
| 2 | MLFQ | 336.30 | 0.595 | 0.595 | +5.8% | +5.5% |
| 2 | Default | 317.73 | 0.629 | 0.629 | - | - |

### Burst Load

| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |
|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|
| 2 | MLFQ | 273.82 | 0.219 | 0.219 | +10.4% | +9.4% |
| 2 | Default | 247.99 | 0.242 | 0.242 | - | - |

## Overall Conclusion

**Overall Performance:**
- Average Throughput Improvement: +2.7%
- Average Latency Improvement: +3.6%

✅ MLFQ scheduler shows improved throughput performance.
✅ MLFQ scheduler shows improved latency performance.