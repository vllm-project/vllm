# Attention backend selection

> Community sprint note 022/120 · dependency-free design note · vLLM community working group

## Scope

FlashAttention, FlashInfer, Triton and the xformers-style backends differ in head-dim support, page size assumptions and performance envelopes. Selection has to be evidence-based.

## Mechanism

1. Backends register supported head dims, dtypes, block sizes and features such as sliding windows.
2. The engine filters by compatibility, then ranks survivors by measured kernel time.
3. Falling back silently is dangerous: it changes both latency and numerics.
4. Per-shape measurements matter more than published benchmark tables.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `head_dim` | `128.0` | Head width the backend must support. |
| `page_size` | `16.0` | KV block size the kernel expects. |
| `causal_penalty` | `0.05` | Fractional overhead of causal masking in the kernel. |
| `backend_score` | `1.0` | Measured throughput relative to the reference backend. |

## Failure modes

- **Silent fallback to a slow backend because the fast one lacks a feature.**
- **Comparing backends at one shape and generalizing across the serving envelope.**
- **Ignoring numerical differences in long-context prefills.**

## Instrumentation

Watch `kernel_time_ms_p50`, `fallback_events` and `supported_shapes` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --head_dim 128.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/attention-backends.py`, its tests in
`community/tests/test_attention-backends.py`, and a micro-benchmark in
`community/benchmarks/bench_attention-backends.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
