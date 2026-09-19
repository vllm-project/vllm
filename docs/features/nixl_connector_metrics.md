# NIXL Connector Metrics Aggregation

This document describes how NIXL connector metrics are collected, aggregated, and reported in vLLM.

## Overview

The NIXL connector collects detailed telemetry for each KV cache transfer between prefill (P) and decode (D) instances. These metrics are exposed via:

1. **Periodic CLI logging** - Human-readable summary printed periodically
2. **Prometheus metrics** - Histograms and counters for monitoring dashboards

## Data Flow

```
NIXL Telemetry (per-transfer)
        │
        ▼
NixlKVConnectorStats.record_transfer()
        │
        ▼
Stats Aggregation (reduce())
        │
        ├──► CLI Logging (human-readable)
        │
        └──► Prometheus Metrics (histograms/counters)
```

## Core Classes

### `NixlKVConnectorStats`

Located in `vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py`.

**Raw data collected per transfer:**
- `transfer_duration` - End-to-end transfer time (µs from NIXL `xferDuration`)
- `post_duration` - Time to post transfer to RDMA backend (µs from NIXL `postDuration`)
- `bytes_transferred` - Total bytes moved
- `num_descriptors` - Number of NIXL memory descriptors
- `num_failed_transfers` - Failed transfer count
- `num_failed_notifications` - Failed notification count
- `num_kv_expired_reqs` - Requests with expired KV blocks on P instance

### Aggregation Pipeline

#### 1. Per-Engine Collection

Each vLLM engine (worker) maintains its own `NixlKVConnectorStats` instance. Raw telemetry is recorded via:

```python
# Called from worker on each completed transfer
stats.record_transfer(nixl_telemetry)

# Called on failures
stats.record_failed_transfer()
stats.record_failed_notification()
stats.record_kv_expired_req()
```

#### 2. Periodic Reduction (`reduce()`)

The `reduce()` method computes compact representative statistics from raw data:

```python
def reduce(self) -> dict[str, int | float]:
    # Returns CLI-ready dict with:
    # - Num successful transfers
    # - Avg/P90 xfer time (ms)
    # - Avg/P90 post time (ms)
    # - Avg MB per transfer
    # - Throughput (MB/s)
    # - Avg number of descriptors
```

**Key aggregation logic:**
- Only successful transfers are reported in CLI logging
- Failed transfers are tracked separately via Prometheus counters
- Time units converted: µs → ms (×1e3) for human readability
- Bytes converted: bytes → MB (÷ 2²⁰) for readability
- Percentiles computed using NumPy (P90)

#### 3. Cross-Engine Aggregation

The `aggregate()` method combines stats from multiple engines:

```python
def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
    for k, v in other.data.items():
        self.data[k].extend(other.data[k])
    return self
```

This allows combining stats across:
- Multiple vLLM workers (data parallelism)
- Multiple TP ranks
- Multiple PP stages

#### 4. Prometheus Metrics Export

The `NixlPromMetrics.observe()` method pushes aggregated data to Prometheus:

```python
def observe(self, transfer_stats_data: dict, engine_idx: int = 0):
    # Histograms: xfer_time, post_time, bytes_transferred, num_descriptors
    # Counters: failed_transfers, failed_notifications, kv_expired_reqs
```

**Prometheus Metrics Exported:**
| Metric | Type | Description |
|--------|------|-------------|
| `vllm:nixl_xfer_time_seconds` | Histogram | Per-transfer RDMA copy duration |
| `vllm:nixl_post_time_seconds` | Histogram | Time to submit to RDMA backend |
| `vllm:nixl_bytes_transferred` | Histogram | Bytes per transfer |
| `vllm:nixl_num_descriptors` | Histogram | Descriptor count per transfer |
| `vllm:nixl_num_failed_transfers` | Counter | Cumulative failed transfers |
| `vllm:nixl_num_failed_notifications` | Counter | Cumulative failed notifications |
| `vllm:nixl_num_kv_expired_reqs` | Counter | Requests with expired KV blocks |

## Configuration

Metrics collection is automatic when NixlConnector is active. No additional configuration required.

Key environment variables:
- `VLLM_NIXL_SIDE_CHANNEL_PORT` - Handshake port (default 5600)
- `kv_lease_duration` - Lease duration in seconds (default 30s)

## Interpreting Metrics

### Healthy Indicators
- **Low Avg xfer time** (< 10ms for small models, < 50ms for large)
- **P90 ≈ Avg** - Consistent latency
- **High Throughput** (> 1 GB/s on InfiniBand, > 100 MB/s on TCP)
- **Zero failed transfers/notifications**

### Warning Signs
| Symptom | Likely Cause |
|---------|--------------|
| High P90/Avg ratio | Network stragglers or descriptor fragmentation |
| High post time P90 | Descriptor registration overhead |
| Rising `nixl_num_kv_expired_reqs` | Lease duration too short |
| Non-zero failed transfers | Network issues or memory pressure |

### Prometheus Queries

```promql
# Transfer latency P99
histogram_quantile(0.99, rate(vllm:nixl_xfer_time_seconds_bucket[5m]))

# Throughput over 5m
rate(vllm:nixl_bytes_transferred_sum[5m]) / rate(vllm:nixl_bytes_transferred_count[5m])

# Failure rate
rate(vllm:nixl_num_failed_transfers_total[5m])

# Expired KV blocks (adjust lease if high)
rate(vllm:nixl_num_kv_expired_reqs_total[5m])
```

## Related Documentation

- [NixlConnector Usage Guide](nixl_connector_usage.md)
- [NIXL KV Cache Lease Renewal](nixl_kv_cache_lease.md)
- [NIXL Connector Compatibility](nixl_connector_compatibility.md)
- [Disaggregated Prefill Architecture](disagg_prefill.md)
