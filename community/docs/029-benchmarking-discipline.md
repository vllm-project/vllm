# Benchmarking and profiling discipline

> Community sprint note 029/120 · dependency-free design note · vLLM community working group

## Scope

A benchmark is only useful if it states the workload, the warm-up, the measurement window and the percentile it reports. Most 'regressions' are measurement artefacts.

## Mechanism

1. Warm-up iterations absorb compilation and allocator growth before measurement starts.
2. Each configuration is measured across several rounds so percentiles are meaningful.
3. Arrival patterns matter: closed-loop QPS hides queueing that an open-loop client exposes.
4. Profiles are captured for the slowest p1% of requests, not the median.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `warmup_iters` | `10.0` | Iterations discarded before measurement. |
| `measure_iters` | `100.0` | Measured iterations per round. |
| `rounds` | `5.0` | Repeated rounds used for confidence. |
| `load_ramp_pct` | `80.0` | Share of saturation the benchmark targets. |

## Failure modes

- **Closed-loop clients reported as throughput at fixed latency.**
- **Measuring with a warm prefix cache and comparing against a cold one.**
- **One round, one percentile, one machine.**

## Instrumentation

Watch `tokens_per_second`, `ttft_ms_p99` and `itl_ms_p99` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --warmup_iters 10.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/benchmarking-discipline.py`, its tests in
`community/tests/test_benchmarking-discipline.py`, and a micro-benchmark in
`community/benchmarks/bench_benchmarking-discipline.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
