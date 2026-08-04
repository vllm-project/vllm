# LMCache SDK

> Status: out-of-process, CPU-only client for the LMCache multiprocess (MP) server.

## Goal

A small Python surface for moving **KV cache** (and **query**) tensors in and out of a
running LMCache MP server, addressed by **token ids** — so an ML engineer can retrieve a
prefix's KV, edit it, and store it back (e.g. token-dropping inside a KV connector). One
`LMCacheSDKContext` serves either kind, selected by `LMCacheSDKCacheKind` (`KV` / `QUERY`).
Retrieve / store / close are **methods** on the context.

```py
import lmcache.sdk.kvcache as kvcache   # query tensors: lmcache.sdk.qcache

ctx = kvcache.connect(
    url="tcp://localhost:5555",       # ZMQ url
    http_url="http://localhost:9000", # HTTP url for retrieving KV cache shape
    model_name="Qwen/Qwen3-8B",
)
kv = ctx.retrieve(tokens=[1, 2, 3, ...])   # [2, L, hit_tokens, D] or None
# ... edit kv ...
ok = ctx.store(kv=kv, tokens=[4, 5, 6, ...])
ctx.close()
```

See the [token-dropping example](../../../examples/kvcache_sdk/e2e_kv_edit.ipynb).

The model layout must already be registered in the server by a vLLM instance that called
`REGISTER_KV_CACHE`; the SDK reads that layout from `/status` and `/config` to configure itself.

## Architecture

The SDK is a **separate, CPU-only process** from the LMCache server.

```
SDK process                                  LMCache MP server
-----------                                  -----------------
LMCacheSDKContext
  ├ ContiguousTransferWrapper
  │   └ EngineDrivenContext{Shm,Pickle} ──MQ──▶ EngineDrivenTransferModule
  │                                             └ StorageManager (L1 pool, locks, prefetch)
  └ MessageQueueClient ──────────────────ZMQ──▶ LookupModule (LOOKUP / QUERY_PREFETCH_STATUS)
  SharedMemory(name) ◀────────────────────────  L1 POSIX segment (SHM transport only)
```

- **Control plane (ZMQ):** lookup/prefetch, slot reservation, lock release, session end.
- **Data plane:** **SHM** when the server exposes an L1 pool, otherwise **pickle** over the
  MQ — both driven through one `EngineDrivenContext`, so the SDK never branches on transport.
- **`ContiguousTransferWrapper`** ([wrapper/contiguous.py](../../../lmcache/sdk/wrapper/contiguous.py))
  bridges a contiguous `[2, L, T, D]` tensor to the per-chunk `prepare`/`commit` protocol and
  masks the SHM-vs-pickle difference.

## Registration handshake

`connect()` builds the context, then `register_caches()` runs once (for the `QUERY` kind the
model name is suffixed `##query`, so the lookups below use that key):

1. HTTP `/config` → `chunk_size`.
2. HTTP `/status` → the `cache_context_meta` entry for `model_name`: `world_size` and the GPU
   `kv_cache_layout` (`num_layers`, `dtype`, `tokens_per_block`, `engine_kv_format`,
   `engine_kv_concrete_shape`).
3. Decode geometry from the layout: `num_kv_heads` / `block_size` via the format-aware
   `get_num_heads` / `get_block_size` (on a `meta` probe tensor — no allocation), `head_dim`
   from the last dim; assert `block_size == tokens_per_block`.
4. Allocate a **dummy 1-block** HND buffer `{layer.i: zeros([1, 2, NH, BS, HS], cpu)}` — used
   only to register the layout; the data plane never touches it.
5. `create_transfer_context(buffer)` → `EngineDrivenTransferContext.register(...)`, which sends
   `REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT` (server reserves the SHM pool / picks SHM vs pickle).
6. Wrap `transfer_ctx.engine_driven_context` in a `ContiguousTransferWrapper`.

`instance_id` is the SDK process PID.

## Public API

`connect()` lives in `lmcache.sdk.kvcache` (KV) and `lmcache.sdk.qcache` (query); the
`lmcache.sdk.connect(kind, ...)` helper dispatches by `LMCacheSDKCacheKind`. Both return an
`LMCacheSDKContext`, on which retrieve / store / close are **methods**. `LMCacheSDKContext` /
`LMCacheSDKCacheKind` / `LMCacheSDKError` are exported from `lmcache.sdk`.

- **`connect(url, http_url, model_name, timeout=60.0)`** — open the MQ client, fetch config,
  run the handshake; returns an `LMCacheSDKContext`.
- **`ctx.retrieve(tokens, cache_salt="")`** → contiguous CPU `[2, num_layers, hit_tokens,
  hidden_dim]` for the cached prefix, or `None` (empty/sub-chunk input, or nothing cached).
- **`ctx.store(kv, tokens, cache_salt="")`** → `bool`. `kv` is `[2, L, T, D]`; `len(tokens)`
  must equal `T`; both are truncated to whole chunks before storing.
- **`ctx.close()`** — shut down the MQ client and ZMQ context.

## Cache addressing

Both build an `IPCCacheServerKey(model_name, world_size=1, worker_id=0, token_ids, start=0,
end=<chunk-aligned>, request_id, cache_salt)`. The server resolves it to per-chunk `ObjectKey`s;
**cache identity** = token-chunk hashes + `model_name` + `kv_rank(worker_id)` + `cache_salt`.
`request_id` (`store-`/`retrieve-<uuid>`) only keys the per-request session, not cache identity.

- `worker_id=0` is valid because the SDK is `world_size == 1` (one shard per chunk).
- LOOKUP uses the `worker_id=None` (expand-to-all-workers) variant.

## Flows

**store** — validate `len(tokens) == kv.shape[2]`, truncate to whole chunks, then
`transfer_ctx.store(key, instance_id, kv_cpu)`: `prepare_store` returns writable SHM slots
(filled in place) or `None` for pickle (gather all chunks); `commit_store`. Writes use mode
`"new"`, so already-cached chunks are deduplicated.

**retrieve** — Phase 0: `LOOKUP` + poll `QUERY_PREFETCH_STATUS` until non-`None`; if 0 → `None`.
Phase 1: `transfer_ctx.retrieve(key, instance_id)`: `prepare_retrieve` → `torch.cat` the chunk
slots into one contiguous tensor → `commit_retrieve`. `end_session` always runs in `finally`.

## Copy summary

| Flow | SHM | Pickle |
| --- | --- | --- |
| store | 1 (tensor → SHM slot) | ~2 (tensor → chunks, then serialize) |
| retrieve | 1 (slot → contiguous) | ~2 (deserialize, then → contiguous) |

## Constraints & known gaps

- **`world_size == 1` only**, and a **single non-hybrid kernel group** only.
- **Model must be pre-registered** by a vLLM instance (`REGISTER_KV_CACHE`); the SDK reads the
  GPU layout from `/status` and cannot derive it from `model_name` alone.
- **Query (Q) kind** (`kind=LMCacheSDKCacheKind.QUERY`, keyed under `<model>##query`) is served
  by the same context; its layout is registered by the vLLM worker's Q ring (`REGISTER_Q_CACHE`)
  rather than `REGISTER_KV_CACHE`.
