# TensorRT-LLM integration

## Adapter shape

```
lmcache/integration/tensorrt_llm/
├── __init__.py             # Optional-import surface
├── utils.py                # ENGINE_NAME, lmcache_get_config,
│                           # create_trtllm_metadata
├── tensorrt_adapter.py     # In-process — engine in TRT-LLM process
└── tensorrt_mp_adapter.py  # Multi-process — engine in standalone server
```

Both adapters subclass TRT-LLM's `KvCacheConnectorScheduler` and
`KvCacheConnectorWorker`. The TRT-LLM imports are *guarded at module
level only via the package's `__init__`*; nothing in core LMCache
imports the adapter modules. This keeps `pip install lmcache` unaffected
when TRT-LLM is absent.

The TRT-LLM connector preset registry (PR
[NVIDIA/TensorRT-LLM#12626](https://github.com/NVIDIA/TensorRT-LLM/pull/12626))
maps:

| Preset | Module | Scheduler | Worker |
|---|---|---|---|
| `lmcache` | `lmcache.integration.tensorrt_llm.tensorrt_adapter` | `LMCacheKvConnectorScheduler` | `LMCacheKvConnectorWorker` |
| `lmcache-mp` | `lmcache.integration.tensorrt_llm.tensorrt_mp_adapter` | `LMCacheMPKvConnectorScheduler` | `LMCacheMPKvConnectorWorker` |

## Lifecycle

| Stage | TRT-LLM hook | LMCache call |
|---|---|---|
| Init | `worker.register_kv_caches(kv_cache_tensor)` | Build engine via `_get_or_create_engine`; call `gpu_connector.register_kv_caches(kv_cache_tensor)` |
| Before scheduling | `scheduler.get_num_new_matched_tokens(req, num_computed)` | `engine.lookup(tokens)` (in-process) or `LOOKUP` + `QUERY_PREFETCH_STATUS` (MP) |
| Pre-forward | `scheduler.build_connector_meta(scheduler_output)` | `LMCacheConnectorMetadata(loads=..., saves=...)` |
| Forward | `worker.start_load_kv(stream)` | `engine.retrieve(tokens, block_ids)` |
| Forward | `worker.wait_for_save(stream)` | `engine.store(tokens, block_ids)` |

## In-process vs MP

The two modes share the lifecycle but differ in where state lives.

| Aspect | In-process (`lmcache`) | Multi-process (`lmcache-mp`) |
|---|---|---|
| LMCache engine | Singleton inside the TRT-LLM process | Standalone ZMQ server |
| Tensor sharing | Direct (same process) | `RawCudaIPCWrapper` (cudaIpc + cupy DLPack) |
| Lookup | `engine.lookup(tokens)` returns chunk count | `LOOKUP` enqueues prefetch; `QUERY_PREFETCH_STATUS` reads result keyed by `request_id` |
| Configuration | `LMCACHE_CONFIG_FILE` env var | Same; plus `server_url` in connector config (or `LMCACHE_SERVER_URL` env) |
| Failure mode | One process crash takes down both | Engine survives TRT-LLM crash; multiple TRT-LLM instances can share cache |
| Setup cost | None | Run `python -m lmcache.v1.multiprocess.server` |

## Why **not** subclass `VLLMPagedMemGPUConnectorV3`

V3's transfer path is wrong for TRT-LLM:

- **V3 uses the in-process kernel** (`multi_layer_kv_transfer`) with a
  `slot_mapping` of token positions and per-layer pointers. TRT-LLM's
  cross-layer pool is a *single* base pointer and we want to transfer
  by *block ids*, not slot positions.
- **TRT-LLM needs the MP kernel** (`multi_layer_block_kv_transfer`)
  which natively handles single-base-pointer cross-layer with
  `shape_desc.nl` walking layers internally. There is nothing to
  inherit.

`TRTLLMGPUConnector` is therefore a *standalone* `GPUConnectorInterface`
implementation. It also exposes a bespoke
`register_kv_caches(kv_cache_tensor)` method called by the worker once
at init — separate from `to_gpu`/`from_gpu`. The factory in
`lmcache/v1/gpu_connector/__init__.py` constructs it from
`LMCacheMetadata` plus the device, and the adapter wires the pool
tensor in afterwards.

## Forcing real LMCache hits in tests

TRT-LLM has its own GPU block reuse. To verify LMCache contributes the
hit (and not TRT-LLM's reuse), the E2E tests size TRT-LLM's pool tiny
(`KvCacheConfig(max_tokens=512)`) while sending prompts much larger than
512 tokens. The first request fills LMCache and TRT-LLM. The second is
guaranteed-evicted from TRT-LLM's pool and *must* come from LMCache —
which the test asserts via the `lmcache_cached=… new_matched=…` log
line on request 3.
