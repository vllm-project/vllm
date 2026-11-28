# Continuous batching at iteration granularity

> Community sprint note 003/120 · dependency-free design note · vLLM community working group

## Scope

Instead of batching whole requests, the engine re-forms the batch every decode step: finished sequences leave, waiting ones join. GPU occupancy tracks the token budget rather than the request arrival pattern.

## Mechanism

1. Each step the scheduler builds a batch from running, swapped and waiting queues under a token budget.
2. Admission is gated by remaining KV blocks, so a large prompt cannot starve decodes already in flight.
3. Sequences that hit stop tokens or max_tokens are retired immediately and their blocks are recycled.
4. Step latency is dominated by the slowest member of the batch, which is why prefill is chunked.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `batch_token_budget` | `8192.0` | Upper bound on tokens processed in one engine step. |
| `max_num_seqs` | `256.0` | Hard cap on concurrently scheduled sequences. |
| `decode_overhead_us` | `120.0` | Per-step scheduling and Python overhead outside the model. |
| `prefill_chunk` | `2048.0` | Prompt tokens admitted per step when mixing prefill and decode. |

## Failure modes

- **A token budget far above the memory budget only moves the queue from the scheduler to the allocator.**
- **max_num_seqs tuned on a 7B model silently OOMs on a 70B model with the same block pool.**
- **Ignoring the per-step overhead makes benchmarks look better than the served latency.**

## Instrumentation

Watch `steps_per_second`, `queue_wait_ms_p99` and `batch_fill_ratio` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --batch_token_budget 8192.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/continuous-batching.py`, its tests in
`community/tests/test_continuous-batching.py`, and a micro-benchmark in
`community/benchmarks/bench_continuous-batching.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
