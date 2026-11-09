# KV cache eviction and sliding windows

> Community sprint note 014/120 · dependency-free design note · vLLM community working group

## Scope

Long conversations cannot keep every KV entry. Eviction policies trade exactness for capacity: drop the oldest, keep attention sinks, or recompute on demand.

## Mechanism

1. A window bounds how many recent tokens keep KV state per sequence.
2. Sink positions at the start of the sequence are retained because they absorb a large share of attention mass.
3. Evicted positions are either discarded (approximate) or recomputed when the window slides back over them.
4. The policy runs at block granularity, so the effective window is rounded to block boundaries.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `window_size` | `4096.0` | Retained recent tokens per sequence. |
| `sink_tokens` | `4.0` | Leading tokens pinned regardless of the window. |
| `block_size` | `16.0` | Eviction granularity. |
| `recompute_penalty_ms` | `3.0` | Cost of recomputing one evicted block. |

## Failure modes

- **Evicting sinks and watching quality collapse on summarization tasks.**
- **Reporting memory savings without the recompute cost they imply.**
- **Mixing sequences with different windows in one batch without masking.**

## Instrumentation

Watch `retained_tokens_per_seq`, `recompute_blocks_per_request` and `quality_delta` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --window_size 4096.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/kv-cache-eviction.py`, its tests in
`community/tests/test_kv-cache-eviction.py`, and a micro-benchmark in
`community/benchmarks/bench_kv-cache-eviction.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
