# Disaggregated prefill and decode

> Community sprint note 016/120 · dependency-free design note · vLLM community working group

## Scope

Prefill is compute-bound and decode is bandwidth-bound, so running them on separate pools lets each be sized and scheduled independently. The price is moving KV state between them.

## Mechanism

1. A prefill worker computes KV for the prompt and hands the blocks to a decode worker.
2. The transfer goes over RDMA or NVLink, chunked so decode can start before the full prompt arrives.
3. Decode workers hold long-lived KV and are packed for maximum batch size.
4. Placement aims to co-locate the handoff: same node, same rack, then same fabric.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `kv_transfer_gbps` | `100.0` | Effective bandwidth of the prefill-to-decode path. |
| `kv_mb_per_1k_tokens` | `128.0` | KV footprint per thousand prompt tokens. |
| `handoff_us` | `400.0` | Control-plane latency of one handoff. |
| `prompt_tokens` | `8192.0` | Prompt length that determines transfer volume. |

## Failure modes

- **Measuring TTFT without the transfer, which hides the entire cost of the design.**
- **Co-locating pools on the same GPU and calling it disaggregation.**
- **Ignoring that a failed handoff needs a retry path or the request hangs.**

## Instrumentation

Watch `ttft_ms_p99`, `kv_transfer_time_pct` and `handoff_failure_rate` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --kv_transfer_gbps 100.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/disaggregated-prefill-decode.py`, its tests in
`community/tests/test_disaggregated-prefill-decode.py`, and a micro-benchmark in
`community/benchmarks/bench_disaggregated-prefill-decode.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
