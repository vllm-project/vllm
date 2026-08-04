# Encoder Cache (EC) Design

## Overview

The **Encoder Cache (EC)** subsystem persists vLLM multimodal encoder
outputs through LMCache's existing storage backends, so that encoder
work performed for one request can be reused by future requests with
the same multimodal inputs (e.g. an image referenced by the same hash
in two chat completions).

EC is a sibling of, not part of, the KV cache pipeline. It mirrors the
KV engine's layering but with a much narrower contract: each entry is a
single tensor keyed by a single `mm_hash`. There is no token chunking,
no layerwise streaming, and no paged gather/scatter.

---

## Layering

```
        vLLM scheduler / worker
                 │
                 ▼
   ┌──────────────────────────────┐    lmcache/integration/vllm/
   │  LMCacheECConnectorImpl      │    vllm_ec_adapter.py
   │  (vLLM ECConnectorBase glue) │
   └──────────────────────────────┘
                 │
                 ▼
   ┌──────────────────────────────┐    lmcache/v1/ec_engine.py
   │  ECCacheEngine               │
   │  - put(mm_hash, tensor)      │
   │  - get(mm_hash, device)      │
   │  - contains(mm_hash)         │
   └──────────────────────────────┘
                 │
                 ▼
   ┌──────────────────────────────┐    lmcache/v1/storage_backend/
   │  StorageManager              │    storage_manager.py
   │  (existing KV-cache plumbing)│
   └──────────────────────────────┘
                 │
                 ▼
        Local CPU / disk / remote / NIXL backends
```

The connector implements the vLLM `ECConnectorBase` API and is
duplexed: scheduler-side methods (`has_cache_item`,
`update_state_after_alloc`, `build_connector_meta`) and worker-side
methods (`start_load_caches`, `save_caches`) live on the same class
because vLLM's API requires it. The scheduler-only state
(`_mm_hashes_need_loads`) is unused on worker-side instances.

The engine is **transport-agnostic**: it speaks tensors and `mm_hash`
strings. The adapter owns all knowledge of vLLM's `encoder_cache` dict,
producer/consumer roles, and connector metadata.

---

## Cache Key

EC uses the same `CacheEngineKey` type as KV, but with deliberately
different field semantics:

| Field          | KV cache                         | EC cache                              |
|----------------|----------------------------------|---------------------------------------|
| `model_name`   | model identity                   | model identity (same value)           |
| `world_size`   | tensor-parallel world size       | sentinel `1`                          |
| `worker_id`    | tensor-parallel rank             | sentinel `0`                          |
| `chunk_hash`   | hash of token chunk              | `_stable_u64_from_str(mm_hash)`       |
| `dtype`        | KV cache dtype (post-quant)      | encoder output dtype (model dtype)    |
| `request_configs` | per-request config tags       | empty                                 |

### Why sentinel `world_size` / `worker_id`

Encoder outputs are **replicated across tensor-parallel ranks**: every
TP rank computes the same encoder output for a given multimodal input.
If we keyed EC entries by `worker_id` we would store N redundant copies
on shared disk for TP=N. By collapsing to a single logical rank, all
TP processes write to the same on-disk key. Concurrent puts are
idempotent (identical contents).

### Why the dtype is decoupled from `metadata.kv_dtype`

If we used `metadata.kv_dtype` (the KV cache's quantization dtype) for
the EC cache key, changing KV quantization settings (fp16 → fp8) would
silently invalidate every EC entry on disk, even though encoder outputs
have nothing to do with KV quant. The engine therefore takes
`encoder_dtype` explicitly, sourced from `vllm_config.model_config.dtype`.

---

## Engine API

The engine exposes three operations:

```python
def contains(self, mm_hash: str) -> bool: ...
def put(self, mm_hash: str, tensor: torch.Tensor) -> bool: ...
def get(self, mm_hash: str, device: str) -> Optional[torch.Tensor]: ...
```

- `put` returns `True` on successful submission to the storage manager,
  `False` only on transient allocator pressure. It does not enforce
  caller invariants (e.g. tensor presence) — it expects a real tensor.
- `get` returns the tensor on a hit, `None` on a miss. The returned
  tensor never aliases an LMCache-managed buffer; callers can keep it
  indefinitely. The engine takes care of `ref_count_down` on every path,
  including the `mem_obj is not None / mem_obj.tensor is None` case.

This shape was chosen deliberately to keep return values
unambiguous — a `False`/`None` return has exactly one meaning each, in
line with the project's coding-standards rule against multi-meaning
return values.

---

## Configuration

EC engines accept overrides on top of the base LMCache config:

| Source        | Prefix        | Example                                  |
|---------------|---------------|------------------------------------------|
| Environment   | `LMCACHE_EC_` | `LMCACHE_EC_CHUNK_SIZE=1024`             |
| YAML key      | `ec_`         | `ec_local_disk: /tmp/ec-disk`            |
| YAML map      | `ec:` nested  | `ec: { local_disk: /tmp/ec-disk }`       |

Overrides land via `load_ec_engine_config()` in `lmcache/v1/config.py`,
which clones the base `LMCacheEngineConfig` and applies the EC-prefixed
keys. Unknown keys are logged and dropped — EC config is best-effort.

EC can run with no explicit storage configuration: the loader
unconditionally enables ``local_cpu`` and sets ``max_local_cpu_size``
to 1 GiB if it is unset, so the engine always has somewhere to put
data. The disk default is **conditional** — it only applies if the
user has set a ``local_disk`` path. In that case ``max_local_disk_size``
defaults to 64 GiB if not specified. Without an explicit ``local_disk``
path EC entries live in CPU memory only and do not survive process
restart; this is intentional, because picking an on-disk location for
the user could overwrite or fill an unintended directory.

---

## Storage Format

EC tensors are stored under `MemoryFormat.EC_TD` (token, dim). The
StorageManager's allocator produces a pinned-CPU buffer which the
engine's `put` populates with a single `copy_` from the source tensor —
this handles GPU→CPU transfer and dtype casting in one step.

Eviction policy and L1/L2/L3 routing follow the existing StorageManager
defaults: EC entries participate in the same eviction queue as KV
chunks unless an EC-specific override changes the relevant config.

---

## Concurrency Notes

- All TP ranks may call `put` concurrently for the same `mm_hash`. This
  is safe because (a) the on-disk key is identical and (b) the contents
  are identical bytes; the storage backend simply overwrites with the
  same payload.
- `start_load_caches` is called once per scheduler step on the worker
  side; it iterates the metadata's `mm_datas` and calls `engine.get`
  for each, populating vLLM's `encoder_cache` only on hits.
- The scheduler-side `_mm_hashes_need_loads` set is drained on every
  `build_connector_meta` call; it does not persist across steps.

---

## Testing

`tests/v1/test_ec_connector.py` exercises the full save → contains →
load roundtrip through `LMCacheECConnectorImpl` against a real
`StorageManager` backed by a temporary directory. The test depends on
vLLM being importable and uses `pytest.importorskip("vllm")` at module
top so it cleanly skips in environments where vLLM is not installed
(e.g. the unit-test CI image).

---

## Design Decisions

### Separate StorageManager from KV

KV cache and EC cache each construct their own `StorageManager`. This
is **intentional**, not an oversight to be cleaned up:

- KV and EC have very different access patterns. KV is chunked,
  layerwise, and high-throughput; EC is single-tensor, request-scoped,
  and far lower-volume. Mixing them in one allocator pool and
  eviction queue lets hot KV chunks evict cold-but-valuable EC
  entries (or vice versa) in non-obvious ways.
- Resource budgeting becomes auditable: an operator can size local
  CPU and disk pools per workload (`max_local_cpu_size`,
  `max_local_disk_size` for KV; `ec_max_local_cpu_size`,
  `ec_max_local_disk_size` for EC) without one cache cannibalizing
  the other.
- The price is one extra background event-loop thread per process,
  and modest duplication of allocator metadata. Both are negligible
  next to the determinism gain.

If a future workload genuinely benefits from shared pools, the
mechanism would be: pass an externally-constructed `StorageManager`
into `ECCacheEngine.__init__` (DI), instead of having the engine
construct one. Today no caller wants that.

### Role pinned to `"worker"` for the storage manager

vLLM's `ECConnectorBase` multiplexes scheduler-side and worker-side
methods onto a single class with a `role` discriminator. Naively one
would forward that role to `create_lmcache_metadata` so LMCache can
size resources per role. **We deliberately do not** — the EC connector
calls `create_lmcache_metadata(vllm_config, role="worker")` regardless
of the vLLM-side role.

The reason: scheduler-side `has_cache_item` calls `engine.contains()`,
which needs a fully constructed `StorageManager` (including
`LocalCPUBackend`, since `LocalDiskBackend` is layered on top of it).
LMCache's `CreateStorageBackends` short-circuits the CPU backend when
`metadata.role == "scheduler"` and then asserts on it for the disk
backend — so threading the real role aborts startup with an
`AssertionError`. Until LMCache grows a scheduler-friendly storage
path (or EC splits scheduler/worker into separate engines), the
connector keeps the role pinned to `"worker"`.

`_mm_hashes_need_loads` is scheduler-only state; it is initialized on
both roles for simplicity but only mutated/read on the scheduler side.

## Future Work

- **Encoder dtype on metadata.** `LMCacheMetadata` does not yet carry
  an `encoder_dtype` field; the connector currently passes
  `vllm_config.model_config.dtype` directly to `ECCacheEngine.__init__`.
  If more producers (sglang, etc.) gain encoder caches, lifting the
  field onto `LMCacheMetadata` would let the engine become connector-
  agnostic.
- **Public connector-metadata accessor in vLLM.** `start_load_caches`
  reaches into `self._parent._get_connector_metadata()`; once vLLM
  exposes a public method, drop the `# noqa: SLF001`.
