# Streaming responses and cancellation

> Community sprint note 019/120 · dependency-free design note · vLLM community working group

## Scope

Clients disconnect. Whether the engine notices within one step or one minute decides how much GPU time is spent generating tokens nobody will read.

## Mechanism

1. The server checks for client disconnects on each streaming write or on a timer.
2. Cancellation marks the sequence for abort; the scheduler retires it at the next step boundary.
3. Already-generated KV blocks are released and, with prefix caching, may remain reusable.
4. Chunk size trades syscall count against detection latency for a cancelled request.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `cancel_poll_ms` | `100.0` | How often the server checks for disconnects. |
| `stream_chunk_tokens` | `1.0` | Tokens accumulated before a chunk is flushed. |
| `flush_us` | `40.0` | Cost of one flush. |
| `buffer_kb` | `64.0` | Per-stream send buffer. |

## Failure modes

- **Cancellation implemented as a flag the scheduler never reads.**
- **Per-token flushes that cost more than the generated token.**
- **Assuming a closed connection means the request stopped consuming GPU.**

## Instrumentation

Watch `wasted_tokens_pct`, `cancel_latency_ms` and `flushes_per_request` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --cancel_poll_ms 100.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/streaming-and-cancellation.py`, its tests in
`community/tests/test_streaming-and-cancellation.py`, and a micro-benchmark in
`community/benchmarks/bench_streaming-and-cancellation.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
