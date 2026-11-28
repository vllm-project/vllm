# Chunked prefill and step budgeting

> Community sprint note 005/120 · dependency-free design note · vLLM community working group

## Scope

Splitting a long prompt across several engine steps keeps decode latency predictable: no single step is dominated by a 32k-token prefill. The cost is more steps and slightly worse prefill efficiency.

## Mechanism

1. The prompt is cut into chunks of at most the configured size and each chunk becomes a step's prefill work.
2. Partial KV state lives in the normal block pool, so a chunked prefill is indistinguishable from a long one afterwards.
3. Decode sequences scheduled in the same step run alongside the chunk, filling otherwise idle compute.
4. Inter-token latency for already-running requests becomes a function of the chunk size, not the prompt length.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `chunk_tokens` | `2048.0` | Prompt tokens processed per step. |
| `max_batched_tokens` | `8192.0` | Ceiling shared by prefill chunks and decode tokens. |
| `decode_first_ratio` | `0.5` | Share of the budget reserved for decodes before prefill claims the rest. |
| `step_deadline_ms` | `50.0` | Soft deadline used to shrink the next chunk. |

## Failure modes

- **Chunks smaller than the kernel's efficient tile size lose more to launch overhead than they save in latency.**
- **Reserving no decode budget starves interactive traffic behind a single long document.**
- **Chunk boundaries that ignore block alignment break prefix-cache reuse.**

## Instrumentation

Watch `prefill_chunks_per_request`, `itl_ms_p99` and `prefill_tokens_per_second` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --chunk_tokens 2048.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/chunked-prefill.py`, its tests in
`community/tests/test_chunked-prefill.py`, and a micro-benchmark in
`community/benchmarks/bench_chunked-prefill.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
