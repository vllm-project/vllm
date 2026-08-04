# Hybrid KV Cache Groups

## Summary

Minimal hybrid memory allocator (HMA) design for the multiprocess vLLM
connector. It separates three concepts:

- **Engine KV cache group** — a group defined by the serving engine (vLLM's
  `KVCacheConfig.kv_cache_groups`). Each is one distinct paged-block address
  space; block IDs are only meaningful within one group.
- **`EngineGroupInfo`** — LMCache's engine-neutral, `msgspec`-encoded record of
  one such group (`group_view.py`). A `list[EngineGroupInfo]` is the
  registration contract.
- **`KVLayerGroupInfo`** — the server's runtime transfer-kernel dispatch unit,
  built from the engine group infos + the real tensors (`kv_layer_groups.py`).

vLLM groups layers by cache behavior; LMCache must transfer by physical layout
(kv_size, num_heads, head_size, block_size, dtype) *and* keep distinct engine
block-id spaces separate. So at registration we build engine group infos, and
store/retrieve address those infos directly.

## Goals / Non-Goals

- Keep the ZMQ API engine-neutral; confine vLLM field reads to
  `lmcache.integration.vllm`.
- Registration defines the protocol-visible group order; store/retrieve block
  IDs are indexed by that order.
- Reuse one grouping primitive (`group_layers_by_identity`) on both the vLLM and
  server sides so group order matches.
- **Not** in scope: sliding-window load-plan trimming; HMA on the non-GPU
  transfer path (it rejects multi-group); removing `layout_hints` (still used
  for tensor layout). Per-group block *sizes*, cross-layer KV sharing, and
  DeepSeek-V4-style slot compression (`compress_ratio > 1`, packing several
  logical tokens per physical slot) *are* supported (see Store and retrieve).

## Types

- **`EngineGroupInfo`** (`msgspec.Struct`): `engine_group_id` (which engine
  block group its layers live in; dense from 0) + `layer_indices` +
  `tokens_per_block` (logical tokens covered by one of the group's paged
  chunks, from the engine's KV cache spec `block_size`; `0` = unreported).
  Several infos may share an `engine_group_id` when one engine group is split
  by physical transfer identity. The list order is the protocol-visible group
  order; an empty list means a single non-hybrid group.
- Helpers in `group_view.py` operate on `Sequence[EngineGroupInfo]`:
  `num_engine_groups`, `num_engine_group_infos`, `expand_engine_block_ids`,
  `get_engine_group_indices`.
- **`KVLayerGroupInfo`** (runtime, server-only): layer indices,
  `PageBufferShapeDesc`, dtype, `tokens_per_block` / `slots_per_block`,
  `slots_per_chunk`, `engine_group_idx`. Derived from real tensors — never the
  API contract.

## Data flow

```text
vLLM KVCacheConfig + registered kv_caches
  | integration.vllm.kv_cache_groups.create_engine_group_infos_from_vllm
  v
list[EngineGroupInfo]  --REGISTER_KV_CACHE (msgspec)-->  server msgspec-decode
  | KVLayerGroupsManager validates against real tensors
  v
KVLayerGroupInfo list   --STORE/RETRIEVE block_ids per info-->  transfer kernels
```

## Registration

`create_engine_group_infos_from_vllm` (the only place that reads vLLM `KVCacheConfig`):

1. Discover each layer's Engine KV format from its registered tensor
   (`normalize_and_discover_per_layer_formats`). Detection is per *layout*, not
   per engine group: a single engine group can contain layers whose registered
   KV tensors have different shapes — for example a 5-D key+value cache
   (`[NB, 2, BS, NH, HS]`, `kv_size=2`) alongside a 3-D key-only cache
   (`[NB, BS, HS]`, `kv_size=1`) in one `UniformTypeKVCacheSpecs` group — so each
   distinct layout within a group is detected and reported separately.
   ("5-D"/"3-D" is the tensor rank: the number of dimensions of one layer's
   registered KV tensor.)
2. Map each registered layer to its engine group index; layers absent from
   every group's `layer_names` (cross-layer KV-sharing layers) are tagged
   `EXCLUDED_ENGINE_GROUP` and dropped (see Cross-layer KV sharing).
3. `group_layers_by_identity` splits layers by transfer identity
   `(kv_size, num_heads, head_size, block_size, engine_group_idx, dtype,
   engine_kv_format)` — `engine_group_idx` keeps identically-shaped layers from
   different engine groups in separate infos, and `engine_kv_format` keeps
   different layouts that share one engine group apart (the 5-D key+value vs the
   3-D key-only cache from step 1).
4. Emit one `EngineGroupInfo` per identity; send the list in the
   `REGISTER_KV_CACHE` payload (the message queue encodes it).

## Store and retrieve

vLLM reports block IDs per engine group. The worker adapter re-indexes them to
engine-group-info order with `expand_engine_block_ids(engine_group_infos, block_ids)` (each
info reuses its source engine group's block IDs), so `STORE`/`RETRIEVE` receive
`list[list[int]]` indexed by info order. The server loop is then trivial: for
info `i`, use `gpu_block_ids[i]`.

### Per-group block sizes and compression

There is no single "engine block size". Each group has two per-group
quantities, and everything else is derived from them:

- **`tokens_per_block`** — logical tokens covered by one of the group's paged
  chunks (one block ID). Read from the group's KV cache spec `block_size` in
  `kv_cache_config` at initialization and carried in `EngineGroupInfo`.
  Hybrid models mix values freely (`google/gemma-4-E4B-it`: sliding-window
  groups 32, full-attention groups 16; DeepSeek-V4-Flash: 256/64/8/4).
- **`slots_per_block`** — physical slots in one paged chunk, detected from the
  registered tensors at registration time (the batch-size dimension,
  `shape_desc.bs`). Only available per kernel group.

A group is compressed when `tokens_per_block > slots_per_block` (each physical
slot packs `tokens_per_block // slots_per_block` logical tokens): ordinary
attention has one token per slot, while DeepSeek-V4-Flash's MLA / indexer
caches pack 4 and 128. No `compress_ratio` is stored — wherever a ratio is
needed it is computed inline from these two ground-truth quantities. The
LMCache chunk size must be a multiple of every group's `tokens_per_block`
(validated at connector init and registration).

The scheduler-side connector does all accounting (hit counts, store/retrieve
ranges) in *tokens* — the only unit shared by every group — and slices each
group's block IDs by `token_range / tokens_per_block_g`
(`slice_block_ids_per_group`). The server counts
`blocks_per_chunk = lmcache_tokens_per_chunk // tokens_per_block` per group.

### Cross-layer KV sharing

Some models (e.g. `google/gemma-4-E4B-it`) reuse one layer's KV cache for
another (`kv_caches[layer] = kv_caches[target]`). vLLM lists only cache-*owning*
layers in `kv_cache_groups`; a sharing layer is absent from every group's
`layer_names`. Such a layer's KV physically lives in its target owner's blocks,
so storing/retrieving the owner already covers it. Registration therefore tags
unlisted layers with `EXCLUDED_ENGINE_GROUP` and `group_layers_by_identity`
skips them — they never form their own info. (Placing them in a group would
duplicate work and, when their block size differs from the group they default
into, corrupt the per-group block-id counts.)

**Store is all-or-nothing (fail-closed):** if the block IDs don't fully cover
every chunk for every group (e.g. a caller bug), or a copy fails, the whole
store is skipped and nothing is committed — a later retrieve simply misses and
the engine recomputes. The non-GPU transfer path rejects multi-group transfers
outright.

## Example

vLLM exposes two engine groups — group 0: layers [0,2,4], group 1: [1,3]. If
layers 0–3 share a shape but layer 4 differs, registration produces:

```text
info 0: engine group 0, layers [0, 2]
info 1: engine group 1, layers [1, 3]
info 2: engine group 0, layers [4]
```

Block IDs `{group 0: [10,11], group 1: [20,21]}` are sent as
`[[10,11], [20,21], [10,11]]` (infos 0 and 2 share group 0's IDs).

## Invariants

- The `list[EngineGroupInfo]` order is the protocol-visible group order; callers
  send one block-id list per info.
- vLLM-specific access stays in `lmcache.integration.vllm`; infos carry neutral
  metadata only.
- The server reproduces grouping with the same `group_layers_by_identity`; real
  tensors remain the source of truth for shape/dtype/stride.

## Mamba / linear-attention hybrids

Supported via registration-time tensor re-views (e.g. Qwen3.5 GDN): Mamba
state pairs become opaque page views, and full-attention layers whose logical
block size was inflated for page-size unification are re-viewed at
logical-block granularity. See
[kv-cache-group-edits](kv_cache_group_edits.md) for the design and its
limits (notably: edited groups are byte-opaque — no content-aware processing,
no cross-backend cache sharing).

## Code map

| Area | File |
|---|---|
| Engine group info (IPC type) + helpers | `lmcache/v1/multiprocess/group_view.py` |
| Shared grouping primitive | `lmcache/v1/kv_layer_groups.py` |
| vLLM → `list[EngineGroupInfo]` | `lmcache/integration/vllm/kv_cache_groups.py` |
| Group metadata edits (Mamba, sub-paged attention) | `lmcache/integration/vllm/kv_cache_group_edits.py` |
| Register / store / retrieve | `lmcache/integration/vllm/{lmcache_mp_connector,vllm_multi_process_adapter}.py` |
| Server GPU context / transfer | `lmcache/v1/multiprocess/{gpu_context,modules/lmcache_driven_transfer}.py` |
| ZMQ protocol | `lmcache/v1/multiprocess/protocols/engine.py` |
