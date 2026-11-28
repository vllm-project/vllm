# Serving many LoRA adapters

> Community sprint note 012/120 · dependency-free design note · vLLM community working group

## Scope

Adapters are small enough to keep resident and swap per batch, which turns multi-tenant fine-tuning into a scheduling problem: how many adapters, at what rank, with what swap cost.

## Mechanism

1. Each request names an adapter; the engine groups requests by adapter within a step.
2. Adapter weights live in a bounded GPU cache; a miss costs a host-to-device copy before the step runs.
3. The batched LoRA kernel applies each request's low-rank delta without splitting the batch.
4. Rank and target modules determine the size of the delta and therefore the swap cost.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `max_loras` | `8.0` | Adapters resident on the GPU at once. |
| `rank` | `16.0` | Low-rank width of the adapters. |
| `adapter_cache_mb` | `1024.0` | Host-side cache for adapters not currently resident. |
| `swap_cost_us` | `900.0` | Cost of paging one adapter in. |

## Failure modes

- **Round-robin traffic across more adapters than fit, which turns every step into a swap.**
- **Sizing max_loras without reserving the KV pool the adapters displace.**
- **Assuming rank 8 and rank 64 have the same kernel cost.**

## Instrumentation

Watch `adapter_swap_rate`, `lora_tokens_per_second` and `resident_adapters` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --max_loras 8.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/lora-adapter-serving.py`, its tests in
`community/tests/test_lora-adapter-serving.py`, and a micro-benchmark in
`community/benchmarks/bench_lora-adapter-serving.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
