# Metrics, histograms and observability

> Community sprint note 020/120 · dependency-free design note · vLLM community working group

## Scope

Latency percentiles, queue depth and cache occupancy are what turn a mystery slowdown into a diagnosis. Cardinality is the usual reason a metrics endpoint melts down.

## Mechanism

1. Counters track requests, tokens and preemptions; histograms capture latency distributions.
2. The scrape endpoint renders exposition text from the in-process registry.
3. Label sets multiply series; request ids or prompts as labels are the classic cardinality bomb.
4. Logs carry the same request id so a slow trace can be followed across components.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `scrape_interval_s` | `15.0` | Prometheus scrape cadence. |
| `histogram_buckets` | `20.0` | Resolution of latency histograms. |
| `label_cardinality` | `500.0` | Distinct label combinations the registry tolerates. |
| `retention_s` | `3600.0` | In-process window used by the dashboard endpoint. |

## Failure modes

- **Unbounded labels, which turn a scrape into a memory event.**
- **Averaging latency instead of reading percentiles.**
- **Emitting metrics only on success, so failures vanish from the dashboards.**

## Instrumentation

Watch `series_count`, `scrape_duration_ms` and `histogram_overflow_rate` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --scrape_interval_s 15.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/metrics-and-observability.py`, its tests in
`community/tests/test_metrics-and-observability.py`, and a micro-benchmark in
`community/benchmarks/bench_metrics-and-observability.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
