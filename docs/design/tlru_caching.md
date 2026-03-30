# Tail-Optimized LRU (T-LRU) Cache Eviction

T-LRU is an optional eviction policy that extends vLLM's default [LRU prefix cache](prefix_caching.md)
to reduce **tail (P95/P99) Time-to-First-Token (TTFT)** in multi-turn conversation workloads.
It requires no changes to the serving API and is fully backward-compatible with standard prefix
caching.

## Background

vLLM's default LRU policy maximises average cache hit rate but is *conversation-length blind*: it
treats a single block from a 10 000-token conversation the same as a block from a 100-token
conversation. When many conversations compete for the same KV cache budget, LRU can evict the
history of a long, active conversation while caching blocks from short or finished conversations,
producing avoidable tail-latency spikes.

## Key Idea

For a conversation with history `H` (in blocks) and estimated next-query length `Q̂` (in blocks),
any prefix block **beyond** position `B = max(0, H + Q̂ − ξ)` can be evicted without pushing the
next turn's recomputation cost above the SLA threshold `ξ` (also in blocks). Those blocks are
called **TEL-safe** (Tail-Excess-Latency safe).

T-LRU sets them aside in a dedicated `tel_safe_queue` and evicts them *before* touching any
regular LRU block. This lets the critical prefix (the first `B` blocks of long conversations)
survive longer in the cache, reducing the number of turns that exceed the latency SLA.

The policy is proved optimal under a natural stochastic model of conversation dynamics; see the
companion paper for details.

## Algorithm at a Glance

```
When a request finishes:
  B = max(0, H + Q̂ − ξ)            # minimum blocks needed in cache to meet SLA
  tail blocks (last H − B blocks) → tel_safe_queue   # safe to evict first
  head blocks (first B blocks)    → normal LRU queue  # kept longer

When new blocks are needed:
  1. drain tel_safe_queue first (TEL-safe evictions)
  2. fall back to normal LRU queue
```

Within each queue, standard LRU ordering (oldest-freed first) is preserved.

## Configuration

Enable T-LRU by setting `--tlru-xi-tokens`. It has no effect unless also enabling prefix caching
with `--enable-prefix-caching`.

| CLI flag | Type | Default | Meaning |
|---|---|---|---|
| `--tlru-xi-tokens` | `int \| None` | `None` (disabled) | SLA latency threshold in **tokens**. T-LRU is only active when this is set. Set to your P95 TTFT target converted to tokens (e.g. 200 ms × tokens/ms). |
| `--tlru-qhat-tokens` | `int` | `200` | Estimated next-query length in tokens. Can be tuned to the mean or P75 of observed query lengths in your workload. |

Example:

```bash
vllm serve meta-llama/Llama-3-8B \
  --enable-prefix-caching \
  --tlru-xi-tokens 4096 \
  --tlru-qhat-tokens 200
```

!!! note
    `--tlru-xi-tokens` does **not** affect compilation caching (it is excluded from
    `CacheConfig.compute_hash()`), so changing it between restarts does not invalidate compiled
    graphs.

## Tuning Tips

- **ξ (`--tlru-xi-tokens`)**: Set this to your latency SLA expressed in tokens.
  For example, if your SLA is 200 ms and your model processes ~20 tokens/ms, set `ξ = 4000`.
  When ξ is very small (tight SLA), `B = max(0, H + Q̂ − ξ)` is large, so few blocks are
  TEL-safe and T-LRU behaves like standard LRU.  When ξ is large (loose SLA), `B` approaches
  0, meaning all blocks are TEL-safe and are preferentially evicted before any cached prefix —
  this is intentional: a large SLA means the system can afford to recompute everything and
  should free the cache aggressively.
- **Q̂ (`--tlru-qhat-tokens`)**: Set this to the expected next-query length. The default of 200
  tokens works well for typical chat workloads. If your users tend to send long follow-up queries,
  increase this value to protect more of each conversation's prefix.

## Experimental Results

On the [WildChat](https://huggingface.co/datasets/allenai/WildChat) real conversation dataset,
T-LRU reduces **P95 TTFT by up to 27.4%** compared to standard LRU and closes 25–79% of the gap
to the clairvoyant offline optimum. Results are consistent on ShareGPT. See the paper below for
full tables.

## Further Reading

- Zhang, Moallemi, Peng. *Tail-Optimized Caching for LLM Inference.* **NeurIPS 2025.** https://arxiv.org/abs/2510.15152
- [Automatic Prefix Caching](prefix_caching.md) — vLLM's base prefix caching design.
