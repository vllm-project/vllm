# Prefix caching and block reuse

> Community sprint note 004/120 · dependency-free design note · vLLM community working group

## Scope

Identical prompt prefixes hash to the same block sequence, so a second request sharing a system prompt can skip recomputation entirely. The win is proportional to shared prefix length and how quickly the cache turns over.

## Mechanism

1. Each full block of a prompt is hashed together with its parent hash, giving a chain that identifies the prefix.
2. On admission the engine walks the chain and links any matching blocks instead of allocating new ones.
3. Linked blocks are reference counted and evicted with an LRU policy when the pool runs dry.
4. Only block-aligned prefixes hit, so prompts shorter than one block never benefit.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `cache_blocks` | `4096.0` | Share of the pool the prefix cache may occupy. |
| `block_size` | `16.0` | Alignment of a cacheable prefix. |
| `hash_bits` | `64.0` | Width of the block hash; narrow hashes risk collisions on adversarial prompts. |
| `reuse_window_s` | `300.0` | How long an evicted-but-unchanged prefix is expected to come back. |

## Failure modes

- **Caching prompts that carry per-user secrets turns the cache into a cross-tenant information leak.**
- **Cache hits counted before the block chain is verified overstate the reported saving.**
- **A cache that never evicts starves decodes during a burst of unique prompts.**

## Instrumentation

Watch `prefix_cache_hit_rate`, `skipped_prefill_tokens` and `cache_evictions` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --cache_blocks 4096.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/prefix-caching.py`, its tests in
`community/tests/test_prefix-caching.py`, and a micro-benchmark in
`community/benchmarks/bench_prefix-caching.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
