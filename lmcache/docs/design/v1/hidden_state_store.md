# HiddenStateStore

`HiddenStateStore` is an LMCache-internal store dedicated to caching per-token
hidden-state tensors next to KV chunks. It exists to support any system where
we may need to draw out intermediate activations from an inference forward
pass (e.g. vLLM-Omni thinker -> talker) where a downstream stage needs the
upstream stage's intermediate activations for the cached prefix and cannot
reconstruct them from KV alone.

## Goals

- Keep KV cache logic and hidden-state cache logic in **separate classes**.
  `LMCacheEngine` owns one instance and exposes it via
  `engine.hidden_state_store`, but does not embed hidden-state state in
  itself.
- Use a **separate pinned CPU memory pool** so that hidden-state allocations
  cannot fragment the KV pool. KV tensors and intermediate activations have
  **heterogeneous shapes** (KV is per-layer `[num_tokens, 2, head_dim]`-style
  while hidden states / other activations are typically
  `[num_tokens, hidden_dim]` and may differ in dtype and per-layer count), so
  packing them into a shared allocator would either waste space or force
  one cache to evict the other under pressure.
- **Reuse** existing chunking and key generation
  (`ChunkedTokenDatabase.process_tokens` -> `CacheEngineKey`) so each hidden
  chunk is keyed identically to its KV counterpart.
- Implement the **coupled-but-asymmetric** eviction rule:
  - KV evicted -> HS for that chunk is dropped on the next access.
  - HS evicted -> KV stays.
  - Restore stops at the first chunk where HS is missing.

## Class layout

```
LMCacheEngine
  - storage_manager       (KV: LocalCPUBackend + LocalDiskBackend + ...)
  - hidden_state_store    (HiddenStateStore, owns its own pinned pool)
       - _allocator       (MixedMemoryAllocator, independent buffer)
       - _chunks: dict[CacheEngineKey, dict[layer_idx, MemoryObj]]
       - _lru:    OrderedDict[CacheEngineKey, None]
```

`HiddenStateStore` does not depend on `StorageManager` for storage. It does
hold a reference to the storage manager (set via `bind_storage_manager()`) so
that on retrieve it can ask "is KV still here for this key?" via
`storage_manager.contains(key)`.

## Public API

Integrators call these methods on **`engine.hidden_state_store`** when it is
not `None` (when `config.enable_hidden_state_cache` is `True`). They are not
methods on `LMCacheEngine`. Callers **must** check for `None` themselves (same
pattern as vLLM-Omni ``OmniGPUModelRunner``: skip store/retrieve when HS
caching is disabled).

`HiddenStateStore`:

- `store_hidden_states(token_ids, hidden_states, *, layer_idx=0) -> int`
  Chunks `token_ids` with the engine's `token_database`, copies the matching
  rows of `hidden_states` into pinned memory under the KV chunk key, returns
  the number of chunks stored.
- `retrieve_hidden_states(token_ids, *, layer_idx=0) -> torch.Tensor | None`
  Walks chunks of `token_ids` in order. Stops at the first chunk where
  either KV is missing (lazy coupled-eviction cleanup) or HS is missing
  (prefix-strict). Returns the contiguous prefix tensor or `None` for a
  full miss.
- `close()`
  Frees the pinned pool.

Multi-layer payloads are handled by invoking `store_hidden_states` once per
distinct `layer_idx` (and `retrieve_hidden_states` per layer on restore).

When `enable_hidden_state_cache` is `False`, `engine.hidden_state_store` is
`None`; callers must omit HS APIs on that worker.

## Eviction (lazy coupled-check)

- HS-only allocator pressure: `HiddenStateStore` evicts its own LRU entry
  and retries; KV is never touched.
- KV evictions are **not** observed eagerly. Instead, on every retrieve we
  ask `storage_manager.contains(key)`. If KV is gone for a chunk we hold HS
  for, we drop that HS entry and stop the prefix there.

This keeps the engine surface tiny (no callbacks, no shared index) while
still satisfying:

- KV evict implies HS evict (next read drops the orphan).
- HS evict does not imply KV evict.
- Restore stops at first missing HS *or* missing KV chunk.

## Configuration

All four fields already exist in `LMCacheEngineConfig` (PR 1330). The pool
size **is** the new pool size; no extra field is required:

| Field                             | Meaning                                                         |
| --------------------------------- | --------------------------------------------------------------- |
| `enable_hidden_state_cache`       | Master toggle. When `False`, `engine.hidden_state_store` is `None`; integrators gate store/retrieve on this field before calling HS APIs. |
| `max_hidden_state_cpu_size` (GB)  | Independent pinned-CPU pool size for the HS allocator.          |
| `hidden_state_layers`             | Optional allowlist of `layer_idx` values accepted on store.     |

Retrieve always stops at the first chunk without KV or without HS for the requested layer (prefix-strict); there is no alternate assembly mode.

## Why lazy eviction (not callbacks)

- No invasive changes to `LocalCPUBackend` or any cache policy.
- Works uniformly for any backend that implements `contains()` (LocalCPU,
  LocalDisk, remote, etc.).
- The cost is one extra `contains()` call per chunk on retrieve, which is a
  cheap dict lookup for the local case.
