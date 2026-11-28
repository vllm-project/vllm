# PagedAttention and KV memory layout

> Community sprint note 002/120 · dependency-free design note · vLLM community working group

## Scope

Logical KV slots of a sequence are decoupled from physical GPU blocks, so the allocator can hand out memory in fixed-size pages the way an OS does. Block tables turn that indirection into a gather the attention kernel can follow.

## Mechanism

1. The scheduler chunks a prompt and reserves one logical slot per KV entry before any token is decoded.
2. A GPU-side pool of physical blocks is handed out on demand and recorded in the per-sequence block table.
3. The attention kernel gathers K and V through the block table, so non-contiguous blocks behave like a contiguous cache.
4. A block returns to the pool only after the final reference (sequence, fork or prefix-cache entry) is dropped.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `block_size` | `16.0` | Tokens per physical block; smaller pages cut internal fragmentation and raise table overhead. |
| `kv_blocks` | `2048.0` | Size of the GPU block pool derived from the memory-profiling run. |
| `head_dim` | `128.0` | KV head width; multiplies the bytes each slot consumes. |
| `elem_bytes` | `2.0` | Storage width per element (2 for fp16/bf16, 1 for fp8 KV cache). |

## Failure modes

- **Block sizes above 32 waste KV memory on short chat turns and inflate the preemption rate.**
- **Preemption paths that forget to release blocks leak the pool until the engine refuses new work.**
- **Prefix-cache hits pin blocks longer than the request lifetime suggests, so the free-watermark must account for them.**

## Instrumentation

Watch `kv_cache_usage_pct`, `blocks_allocated_per_seq` and `preemption_count` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --block_size 16.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/paged-attention-memory-layout.py`, its tests in
`community/tests/test_paged-attention-memory-layout.py`, and a micro-benchmark in
`community/benchmarks/bench_paged-attention-memory-layout.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
