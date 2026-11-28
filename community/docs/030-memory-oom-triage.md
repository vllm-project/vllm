# Memory profiling and OOM triage

> Community sprint note 030/120 · dependency-free design note · vLLM community working group

## Scope

Serving OOMs are usually accounting errors: weights, activations, graphs and KV competing for the same budget. A profile taken before the crash is worth more than a guess after it.

## Mechanism

1. Startup profiling measures peak allocation per batch size to derive the KV budget.
2. The reserve fraction protects against fragmentation spikes and allocator overhead.
3. Activations peak during prefill for long prompts, not during decode.
4. Fragmentation grows when block sizes mix; a snapshot shows whether the pool or the allocator is short.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `gpu_memory_utilization` | `0.9` | Share of device memory the engine may claim. |
| `reserve_gb` | `4.0` | Unclaimed headroom kept for fragmentation. |
| `peak_mb` | `68000.0` | Measured peak during the profiling pass. |
| `fragmentation_pct` | `6.0` | Observed allocator fragmentation. |

## Failure modes

- **Setting utilisation to 0.99 and blaming vLLM when the allocator spikes.**
- **Profiling with a short prompt and serving 32k contexts.**
- **Ignoring graph pools and adapter memory when computing the KV budget.**

## Instrumentation

Watch `peak_memory_mb`, `fragmentation_pct` and `kv_blocks_available` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --gpu_memory_utilization 0.9 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/memory-oom-triage.py`, its tests in
`community/tests/test_memory-oom-triage.py`, and a micro-benchmark in
`community/benchmarks/bench_memory-oom-triage.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
