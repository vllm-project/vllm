# Distributed KV transfer protocol

> Community sprint note 017/120 · dependency-free design note · vLLM community working group

## Scope

Moving KV blocks between workers needs chunking, acknowledgement and a retry budget. The protocol details decide whether a slow link degrades throughput or stalls the scheduler.

## Mechanism

1. Blocks are grouped into chunks sized to keep the pipeline full without exhausting pinned buffers.
2. Each chunk carries a sequence id, block ids and a generation counter to detect reuse races.
3. The receiver acknowledges per chunk; missing acks trigger a retransmit from the sender's copy.
4. Backpressure propagates to the scheduler so prefill admission slows before buffers overflow.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `chunk_mb` | `8.0` | Transfer unit size. |
| `links` | `4.0` | Parallel connections per peer. |
| `ack_timeout_ms` | `200.0` | Time before a chunk is considered lost. |
| `retry_budget` | `3.0` | Retransmits before the handoff is failed. |

## Failure modes

- **Unbounded pinned buffers, which look like a memory leak under load.**
- **Ack timeouts shorter than the link's p99 latency, producing duplicate traffic.**
- **No generation counter, so a recycled block id delivers stale KV silently.**

## Instrumentation

Watch `transfer_throughput_gbps`, `retransmit_rate` and `buffer_high_watermark` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --chunk_mb 8.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/distributed-kv-transfer-protocol.py`, its tests in
`community/tests/test_distributed-kv-transfer-protocol.py`, and a micro-benchmark in
`community/benchmarks/bench_distributed-kv-transfer-protocol.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
