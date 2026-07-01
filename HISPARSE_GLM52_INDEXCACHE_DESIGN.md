# HiSparse for GLM-5.2 indexCache — improvement/simplification design

Implementation-ready spec for exploiting GLM-5.2's index sharing (1 `full` + 3
`shared` layers per group; all 4 attend the *same* 2048 top-k positions). Grounds
the throughput work now that the capacity work is banked (I1 native in vLLM 0.24:
indexer + its GPU KV gated to full+MTP layers; host-resident MLA gives the ~5×).

## Current architecture (per decode step, per layer)

Each sparse-attention impl (`FlashMLASparseImpl` / `FlashInferMLASparseImpl`) owns
its own `HiSparseCoordinator` with an independent per-layer `hot_cache`,
`device_global_indices` (dgi), and `lru_slots`. `swap_in(topk_indices)`:
1. `triton_convert_req_index_to_global_index`: per-req top-k → global token ids.
2. `hisparse_swap_in` kernel (fused): resolve globals vs this layer's hot buffer
   (LRU), classify hit/miss/newest, evict LRU slots for misses, **gather missed
   rows host→device**, update dgi+lru, return `hot_indices`.
3. Attention reads `hot_cache_paged` + `hot_indices`.

Reference semantics: `_swap_in_fallback` (hisparse.py:754). The "plan" per row is
`(hot_indices, dgi', lru', miss_src[], miss_dst[])`.

**Redundancy for GLM-5.2:** within a group the 4 layers get identical
`topk_indices` ⇒ identical `global_indices` ⇒ (given identical dgi/lru state,
which holds because they apply the same plan every step) **identical plan**
(`hot_indices`, evictions, `miss_src`/`miss_dst`). Only the *bytes* moved differ
(each layer's own KV). So steps 1–2's *control* is computed 4× redundantly, and
the PCIe gather of step 2 sits on the compute critical path.

## Target design

### Group-shared state (simplification)
One LRU/dgi state per (request, **group**) instead of per (request, layer). The 4
layers share `device_global_indices` + `lru_slots` (the mapping is identical);
each keeps its own `hot_cache` (the data). ~4× less bookkeeping; removes the "keep
4 copies in sync" invariant entirely.

### Plan-once (#2)
The `full` layer computes the plan once (globals + LRU resolve → `hot_indices`,
dgi/lru update, `miss_src`/`miss_dst`) and publishes it to the group. The 3
`shared` layers reuse `hot_indices` + the group dgi/lru (no `triton_convert`, no
LRU resolve) and only perform their own data move.

### Overlapped group prefetch (#1) — the main throughput lever
Because the full layer's indexer fires 3 layers ahead, the moment the plan is
known we issue the shared layers' `miss_src→miss_dst` host→device gathers on a
**dedicated copy stream**, overlapping the full layer's attention + the shared
layers' MoE/MLP compute. Each shared layer's attention waits on its prefetch
event before reading its `hot_cache`.

**Graph-capture constraint (critical):** the decode KV path runs inside a
`FULL_DECODE_ONLY` CUDA graph. Cross-stream waits during capture raise
"dependency created on uncaptured work in another stream" (the same bug that
forced the host backup inline; see hisparse.py:299-313). Options, in order of
preference:
1. Capture the prefetch copy + event on a stream that is itself captured into the
   graph (side-stream capture is legal if the wait is also captured). Validate
   the copy-stream + event are recorded, not just issued.
2. If (1) proves not capture-safe, gate overlap behind eager decode
   (`VLLM_HISPARSE_ASYNC_PREFETCH=1`) like the existing async-backup toggle, and
   ship plan-once (capture-safe, no stream) as the always-on win.

### Coalesced gather (#3)
With one shared plan, the per-layer gathers become one batched/strided
`index_copy_` (fallback) or one kernel launch over the group's stacked host
caches (kernel path), instead of 4 separate launches.

### All-resident fast path (#4) + positional keying
Re-key the hot buffer **positionally** (SGLang-style) instead of by global slot.
Then `seq_len ≤ device_buffer_size` ⇒ the whole sequence is resident by
construction ⇒ skip swap-in entirely and attend the contiguous hot region. The
current global-slot LRU keying is what removed this fast path; positional keying
restores it *and* is the cleaner base for group-shared residency.

### GLM-5.2-only simplifications
- Delete the unified (prefill+decode) path: RL decode is PD-decode-only, so the
  `_hisparse_host_prefill_cache` gather (flashmla_sparse.py — the thing that
  crushed `num_blocks` in unified-mode benchmarking) can be removed, not just
  avoided.
- Drop the mirror-slots / DSV4 branches.
- Single host knob (`host_pool_gib`); drop the ratio-allocator branch.

## Kernel changes

Split `hisparse_swap_in` into:
- `hisparse_plan`: LRU resolve → `hot_indices`, dgi', lru', `miss_dst`,
  `miss_src` (no data move). Runs once per group on the full layer.
- `hisparse_apply` (gather-only): given `miss_src`/`miss_dst`, copy rows
  host/gpu→hot_cache. Runs per layer (per shared layer, on the copy stream).
The existing fused `hisparse_swap_in` stays as the fallback / non-grouped path.
Keep the Python reference (`_swap_in_fallback`) split the same way for the parity
test.

## Plumbing
- Group id + role: the model already computes `skip_topk` (0.24
  deepseek_v2.py:1029) — `full` = indexer present, `shared` = skip_topk. Plumb a
  `group_id = f(layer_id, index_topk_freq, index_skip_topk_offset)` and a
  group-coordinator registry (per model) so the 4 impls share one state object.
- Coordinator creation moves from per-impl to per-group (created by the full
  layer, referenced by the shared layers via the registry).

## Validation plan (must pass before merge)
1. `pytest tests/v1/attention/test_sparse_mla_backends.py -k hisparse` — extend
   `test_hisparse_kernel_matches_fallback` to cover the plan/apply split
   (bit-exact vs the split reference) + a group-reuse case (shared layers'
   `hot_indices` == full layer's).
2. GLM-5.2-FP8 greedy first-token logprob parity vs the current (0.24 native-I1)
   HiSparse — must stay within the FP8-MoE+EP nondeterminism floor.
3. Capacity unchanged; throughput A/B at concurrency 64/128/256 on a KV-bound PD
   decode topology (the regime SGLang wins) — the earlier A/B was blocked by a
   GLM-5-FP8 topology bind, not a HiSparse defect; set up a genuinely KV-bound
   decode instance.

## Sequencing
1. Positional keying + group-shared state (enables everything; parity-test each).
2. Plan-once (always-on, capture-safe).
3. All-resident fast path (#4).
4. Overlapped prefetch (#1) — resolve the graph-capture question first.
5. Coalesced gather (#3).
6. Delete unified/mirror/DSV4.
Each step: rebuild + parity + logprob before the next.
