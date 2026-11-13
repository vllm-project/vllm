# Hybrid State Checkpoints

Hybrid models interleave full-attention layers with recurrent layers whose state vLLM manages as Mamba state (Mamba, gated delta rule, Kimi delta attention and similar). With prefix caching in Mamba `align` mode, vLLM caches a recurrent state only on the block grid (see [Hybrid KV Cache Manager](hybrid_kv_cache_manager.md#case-3-mamba-models)). This document describes two opt-in extensions, implemented in `vllm/v1/core/hybrid_checkpoint/`. The first keeps more of those states resumable. The second chooses which ones to keep.

| Option | Default | Effect |
| --- | --- | --- |
| `mamba_frontier_checkpoint` | `False` | Caches a finished request's state at its exact end, which may lie inside a block, and at the last block boundary its output crossed. |
| `prefix_cache_eviction` with `prefill_cost_coefficients` | `"lru"`, `None` | With `"gds"`, evicts cached states, each with the attention pages only it needs, by GreedyDual-Size on the prefill time they save. |
| `prefix_cache_leaf_last`, `prefix_cache_leaf_idle_seconds` | `False`, `None` | With `"gds"`, evicts every state that has a deeper cached state on its path before any session's latest state. Optionally, a latest state unused for that long goes with them. |

How to enable them is described in [Automatic Prefix Caching](../features/automatic_prefix_caching.md#resuming-past-the-block-grid).

## Motivation

In `align` mode a request's recurrent state is cached at the block boundaries its prefill reaches. In multi-turn and agentic traffic, each turn's prompt extends the previous turn's prompt and output. This pays two costs:

1. **Resume granularity.** The next turn resumes at the last block boundary the previous prompt reached. It recomputes the rest of that prompt, up to a block, plus every token the previous turn generated. Blocks of hybrid models are large (one page holds a whole recurrent state), often hundreds to thousands of tokens.
2. **Eviction granularity.** The free queue evicts one block at a time, least recently freed first. A recurrent state is usable only together with the attention pages on its path, so evicting one of them wastes the rest. A long session's state also goes as readily as a short one's.

## Design principles

- All logic lives in `vllm/v1/core/hybrid_checkpoint/`. The scheduler, the KV cache manager, the block pool and `MambaManager` call it from narrow hooks that do nothing while the options are off.
- Every option defaults off. Every unsupported combination is refused at startup with a `ValueError` that names the missing requirement.
- Model-specific behaviour is declared on KV cache specs, not by model names (see [Indexer rings](#indexer-rings)).
- Upstream mechanisms are reused: `hash_block_tokens` for keys, the partial-hit copy-on-write for resuming, and the free queue for the eviction order.

## Recovery units

A recovery unit (`units.RecoveryUnit`) is one recurrent state that a request can resume from, cached in every Mamba group:

- a block-boundary state that `align` mode caches itself;
- a decode-boundary state: the running state at the last block boundary a request's output crossed, published when the request finishes;
- a frontier: the state at a finished request's exact processed length, inside a block.

A unit is usable only while all of its blocks remain cached. The block pool reports every key a block loses (`BlockPool.key_removal_listener`). When any block of a frontier is evicted, the frontier's other blocks lose the key too, so a lookup never finds part of a unit.

`units.CheckpointGraph` relates the units. A unit's parent is the nearest unit before it on its prefix path: the one its request resumed from, or the one the request cached before it. When a unit is removed, its children move up to its parent. A unit is a branch point when two cached continuations diverge there, or when a request other than its session's next turn has resumed there.

### Frontier keys

A frontier is cached under a key that is not the hash of its block's token range. `keys.frontier_key` chains the request's block hash at the last hash boundary with the remaining tokens, their extra keys (multimodal inputs, LoRA, cache salt) and the length. It goes through `hash_block_tokens` with a marker that separates frontier keys from block hashes. Equal keys mean equal prefixes of equal length. `BlockPool.cache_block_under_key` caches a block under such a key.

## Frontier checkpoints

When a request is freed after processing `L` tokens, with `L` inside block `k` (block size `B`), `FrontierIndex.publish` caches the following under the frontier key of `L`, all of them or none:

- the attention page holding `[kB, L)` in every full-attention group;
- the running state block in every Mamba group;
- the ring block of an indexer whose compression group is open at `L` (see [Indexer rings](#indexer-rings)).

A later request whose tokens extend those `L` tokens first gets its block-aligned hit as before. `FrontierIndex.extend` then lengthens it to the longest frontier that the request's tokens extend. The request resumes at `L` through the partial-hit copy-on-write, which gives it private copies of the partial page and the state.

A `prefix_match_unit` smaller than the block size gives partial hits on a finer hash unit. Then the page holding `[kB, L)` may already be cached under the request's own key at the last hash-unit boundary inside it. The frontier key joins that key (the block pool keeps several keys per block). Removing the frontier removes only its own key (`BlockPool.uncache_key`), and the page stays cached under the request's key. Mamba state and ring blocks must still be unkeyed.

A chunk that starts inside a Mamba block and runs past its end leaves that block's state column at the chunk's start state. `MambaManager` does not cache such a block under its full-block hash, which would promise the state at the block's end.

### Decode-boundary states

A request's output can cross block boundaries, where `align` mode caches no state. With frontier checkpoints, when a running request's output crosses a block boundary, `HybridCheckpointManager.snapshot_decode_boundary` copies its running state into a scratch block in every Mamba group. The state at the last such boundary is published when the request finishes. A scratch block never preempts a request. It comes from the blocks the scheduled requests left free, and a running request that cannot allocate frees scratch blocks first.

### Session leaves

When a request finishes, the newest state it published becomes its session's leaf. The leaf it resumed from is released, unless that leaf is a branch point, a running request still holds its state, or a waiting request resumes there: its lookup did, or under GreedyDual-Size its hit pins it. A frontier that differs from the request only in its last token, such as a stop token the client replaced, is released too. An aborted request publishes no frontier, because its last state may come from a step or a load that never completed.

### Indexer rings

Some models keep an auxiliary ring beside their attention layers, for example a compressor whose open group of tokens is summarised only once the group is full. A request restored inside an open group needs a copy of the ring. The ring's spec declares how the request continues (`CircularBufferSpec.resume_catchup`):

- `"decode"`: only the decode path reads the ring, so the scheduler feeds the restored request one token per step until the group closes;
- `"prefill"`: the prefill path reads the ring too, so the restored request prefills as usual;
- `None`: the ring cannot be restored inside a group, and frontier checkpoints are refused for the model.

The ring group's manager takes at most one hit per request, the frontier's ring block. It copies that block into the request's own ring when the ring is allocated.

A ring that can hold more than one group declares its group size (`CircularBufferSpec.compression_ratio`). A frontier then keeps the ring only when `L` is inside an open group, and a restored request catches up to the end of that group.

## GreedyDual-Size eviction

With `prefix_cache_eviction="gds"`, evictions follow GreedyDual-Size over recovery units (`eviction.GreedyDualSizeEviction`). Each unit is a state together with the attention pages on its path that no other unit needs. For each unit:

- `gain` is the prefill time it saves: the seconds to prefill from its fallback to its own position. The fallback is the deepest unit before it on its path, or position 0. A unit that lost a page on its path can no longer be hit and has gain 0.
- `size` is the number of blocks that evicting it frees: its state blocks and its exclusive pages.
- Its score is `L + gain / size`. `L` is the score of the last evicted unit at the time the unit was last cached, resumed at or released by a request.

The unit with the lowest score is evicted whole: its state blocks, then its exclusive pages, deepest first. A frontier's page can also be cached under the request's own key at a hash unit finer than the block. Such a page only loses the frontier's key: it is neither freed nor counted in the frontier's size. A cached free block that no unit needs scores `L`.

The free queue stays the block pool's only record of the order. The policy implements `CachedBlockEvictionPolicy`. The block pool tells it about cached blocks as they are freed (`on_blocks_freed`). Before an allocation takes cached blocks, the policy moves the blocks it chooses to the front of the free queue (`prepare_eviction`). Without a policy the pool keeps its LRU order.

With `prefix_cache_leaf_last`, a unit that has a deeper live unit on its path (one holding its key as a page, or a child in the graph) is inner, unless it is a branch point. Every inner unit goes before any leaf. The score becomes `(tier, score)` with dead 0, inner 1 and leaf 2. GreedyDual-Size orders each tier, and `L` rises only when a leaf goes. A session's latest state, where its next turn resumes, thus outlasts the earlier states on its path, which serve only rewrites and forks. With `prefix_cache_leaf_idle_seconds`, a leaf that is not a branch point and was not used for that long ranks as inner.

Units that a waiting request's prefix hit reads are pinned, and so are units a running request holds. The waiting requests checked are the first 256 whose lookups walk at most 262,144 hash units in all, refreshed at most every 0.5 s of wall-clock time. Pinned units are not evicted. When no other unit is left, the pool's order applies with pinned blocks last, those of the waiting queue's back first, so that its head, the next request admitted, keeps its hit longest. A new unit whose score is below every other unit's is released as soon as its blocks are free, while the pool has no uncached free block. Under this policy, leaf replacement no longer releases block-boundary states; the eviction order decides when they go.

### Prefill cost model

`cost.PrefillCostModel` models the time to prefill positions `[a, b)` as

```text
d * (b - a) + e * (b^2 - a^2) / 200000
```

seconds: `d` seconds per token, plus `e` seconds per token per 100,000 tokens of context for the attention work that grows with position. `prefill_cost_coefficients=(d, e)` sets the two coefficients. Every score is a past score plus `gain / size`, so scaling both coefficients by one constant leaves the eviction order unchanged; only `e / d` matters. Fit them by least squares to the model's prefill times of `n` tokens after `c` tokens of context, `d * n + e * n * (2 * c + n) / 200000`, at a few context lengths.

With a hash unit finer than the block, a frontier's fallback is its graph parent or the deepest block-boundary unit before it. It is not a state at a hash-unit boundary inside its own block, so its gain counts from there.

## KV connectors

Frontier checkpoints run with a KV connector whose `kv_role` is `"kv_both"` and whose `kv_load_failure_policy` is `"recompute"`. Under `"fail"`, a request whose load failed would finish with a computed length over blocks the load did not write, where no frontier may be cut.

The scheduler shows a connector only the block-aligned part of a local hit that ends inside a block, and lets a longer external hit replace the rest. A frontier hit is such a hit. When the connector's hit is longer, the frontier's partial page, state and ring block are dropped from the request's hit. The request then resumes at the state the load ends with. This applies to the native CPU offloading connector (`--kv-offloading-size`) like any other connector.

## Speculative decoding

Under speculative decoding a finished request's computed-token count does not say where its exact state is: a verify step writes one state column per draft position, and under asynchronous scheduling one more step of the request may be in flight when it finishes. `speculative.SpeculativeFrontiers` keeps each request's last completed step (its device start, its query length, whether it verified drafts, and the tokens the device sampled and the scheduler kept), which the scheduler records in `update_from_output`. When the request finishes, it resolves where the exact state is:

- with a step in flight, the state after the request's last token, in that step's first column;
- otherwise, the state after the kept tokens, in the column of the last accepted draft.

A state in a later draft column is folded into the step's first column on the GPU before anything reads it. The scheduler sends a `MambaStateGather` in the next step's output (`SchedulerOutput.mamba_state_gathers`), and Model Runner V2 applies it with the `align`-mode pre-copy kernel before any other block operation of that step. Both blocks stay referenced until the step completes. When the scheduler kept fewer tokens than the device sampled, for example at a repetition-detection stop, the request publishes no frontier; `SpeculativeFrontiers.declined` counts the reasons. A stop string is caught by the frontend, which aborts the request, so it publishes nothing either.

A decode-boundary state comes from the block in which a verify step leaves the state at a block boundary its accepted tokens crossed. It is snapshotted when the step's output is processed.

`speculative_config.truncate_acceptance_at_stop` makes the rejection sampler stop accepting at the request's stop: the first sampled token that is its EOS token (unless `ignore_eos`) or one of its stop token ids, or that reaches its length limit. The output is unchanged, since the scheduler drops the tokens after a stop anyway, but the device's state then ends at the request's last token. Frontier checkpoints under speculative decoding require it.

## Metrics

`vllm:prefix_cache_hits` includes the tokens a frontier serves. `vllm:prefix_cache_frontier_hits` is registered only when `mamba_frontier_checkpoint` is on. It counts the part of those hits that lies past the block-aligned hit, which is what frontier checkpoints save. Re-admissions of preempted requests are not counted.

## Requirements

`mamba_frontier_checkpoint` requires:

- prefix caching with `mamba_cache_mode="align"` and Model Runner V2;
- `prefix_cache_retention_interval=0`;
- no `long_prefill_token_threshold` below the block size;
- KV cache groups of full attention, Mamba and indexer rings only, at least one full-attention and one Mamba group, and every ring declaring `resume_catchup` (a sliding-window group, for example, is refused);
- a KV connector, if any, with `kv_role="kv_both"` and `kv_load_failure_policy="recompute"`.

With speculative decoding it also requires:

- `speculative_config.truncate_acceptance_at_stop`;
- no EAGLE block drop (`disable_eagle_block_drop` for drafters that need it, such as MTP and EAGLE);
- pipeline parallel size 1;
- no KV connector;
- indexer rings that only the decode path reads (`resume_catchup="decode"`).

It is refused with ReplaySSM, KV cache events, and decode or prefill context parallelism.

`prefix_cache_eviction="gds"` requires `mamba_frontier_checkpoint` and `prefill_cost_coefficients` with `d > 0` and `e >= 0`. `prefix_cache_leaf_last` requires `prefix_cache_eviction="gds"`. `prefix_cache_leaf_idle_seconds` (positive) requires `prefix_cache_leaf_last`.

## Testing

The tests in `tests/v1/core/hybrid_checkpoint/` run on CPU. All but the graph tests drive the real scheduler and KV cache manager:

- `test_checkpoint_graph.py`: a removed unit's children move to its parent, branch points, and frontier keys that separate tokens, lengths and extra keys.
- `test_frontier_checkpoints.py`: resuming at a frontier, catch-up inside an indexer group, decode-boundary states, evicting one block of a frontier, aborted requests, leaf replacement, a prefix-match unit smaller than the block, a KV connector hit longer than the frontier, and refused configurations.
- `test_gds_eviction.py`: GreedyDual-Size against LRU under memory pressure, a frontier pinned by a waiting request, an exhausted pool giving up the pins of the waiting queue's back first, the pool's consistency after every step of a seeded multi-session run, leaf-last tiers and idle demotion, and refused configurations.
- `test_speculative_frontiers.py`: frontier checkpoints under speculative decoding with an n-gram drafter, under synchronous and asynchronous scheduling, with stops inside accepted drafts, pool pressure, aborts, a prefix-cache reset with a gather pending, and refused configurations. A device stand-in (`spec_device.py`) checks every state read and every cached state against its token position, and must catch injected faults.

Elsewhere, `tests/v1/worker/test_gpu_stop_truncation.py` checks the truncated acceptance against the scheduler's `check_stop` (GPU), `tests/v1/core/test_prefix_caching.py` checks the block pool's key-removal listener and eviction-policy hooks, and `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py` checks that a chunk running past a partially hit block does not cache that block.
