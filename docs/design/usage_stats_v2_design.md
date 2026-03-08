# Usage Stats V2: Initial Design Sketch

**Status:** Draft  
**Project:** Usage Stats V2  
**Linear Issue:** VLLM-94  
**Depends on:** [Usage Stats V2 PRD](usage_stats_v2_prd.md)

## 1. Design Principles

1. **Additive only:** New fields extend existing schema; no breaking changes
2. **Fail-safe:** Collection errors are logged and skipped; never raise to caller
3. **Async-first:** All collection and transmission off the critical path
4. **Schema versioning:** Include `usage_stats_version` (e.g., `2`) for evolution
5. **Curlable format:** Flat JSON key-value; `curl -X POST -d @payload.json $URL`

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     vLLM Engine (Worker 0)                        │
├─────────────────────────────────────────────────────────────────┤
│  report_usage_stats(vllm_config)                                 │
│       │                                                           │
│       ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  UsageStatsCollector (new)                                   │ │
│  │  - collect_hardware()   → GPU topology, cluster               │ │
│  │  - collect_model()     → HF ID, params, size bucket          │ │
│  │  - collect_config()    → existing + max_model_len             │ │
│  │  - collect_runtime()   → workload buckets (Phase 2)          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│       │                                                           │
│       ▼                                                           │
│  usage_message.report_usage(arch, context, extra_kvs)            │
│       │                                                           │
│       ├──► _write_to_file(usage_stats.json)  [append JSONL]       │
│       └──► _send_to_server(POST stats.vllm.ai)                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Runtime Aggregator (Phase 2, optional)                          │
│  - Subscribes to request completion events                       │
│  - Maintains in-memory bucket counters (prompt, output, batch)    │
│  - Flushed on heartbeat (every 10 min)                          │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Component Design

### 3.1 UsageStatsCollector (New Module)

**Location:** `vllm/usage/usage_collector.py`

**Responsibilities:**
- Gather hardware, model, and config data
- Normalize and validate values
- Return a `dict[str, Any]` of extra key-values to merge with `UsageMessage`

**Interface:**
```python
def collect_usage_stats_extras(vllm_config: VllmConfig) -> dict[str, Any]:
    """Collect additional usage stats. Returns flat KV dict. Never raises."""
    extras = {}
    try:
        extras.update(_collect_hardware_extras())
        extras.update(_collect_model_extras(vllm_config.model_config))
        extras.update(_collect_config_extras(vllm_config))
    except Exception as e:
        logger.debug("Usage stats collection failed: %s", e)
    return extras
```

**Integration point:** `report_usage_stats` in `vllm/v1/utils.py` calls this and merges into `extra_kvs`.

### 3.2 Hardware Collection

**Source:** NVML (pynvml / vllm's pynvml), platform, env vars

| Field | Source | Fallback |
|-------|--------|----------|
| gpu_topology_nvlink_links | `nvmlDeviceGetNvLinkState` count | 0 or null |
| gpu_topology_pcie_gen | `nvmlDeviceGetPcieInfo` → gen | null |
| gpu_topology_pcie_width | `nvmlDeviceGetPcieInfo` → width | null |
| gpu_compute_capability | `nvmlDeviceGetCudaComputeCapability` | null |
| cluster_node_count | `WORLD_SIZE` / `gpu_count` heuristic | 1 |
| cluster_gpus_per_node | `gpu_count` (single node) or env | null |

**NVML availability:** Use try/import; if NVML unavailable (e.g., CPU, TPU), skip GPU topology fields.

**Code sketch:**
```python
def _collect_hardware_extras() -> dict[str, Any]:
    extras = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # NVLink
        try:
            nvlink_count = sum(1 for i in range(pynvml.NVML_NVLINK_MAX_LINKS)
                              if pynvml.nvmlDeviceGetNvLinkState(handle, i) == 1)
            extras["gpu_topology_nvlink_links"] = nvlink_count
        except Exception:
            extras["gpu_topology_nvlink_links"] = 0
        # PCIe
        pcie = pynvml.nvmlDeviceGetPcieInfo(handle)
        extras["gpu_topology_pcie_gen"] = getattr(pcie, "pcieGen", None)
        extras["gpu_topology_pcie_width"] = getattr(pcie, "pcieWidth", None)
        # Compute cap
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        extras["gpu_compute_capability"] = f"{major}.{minor}"
        pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("NVML hardware collection failed: %s", e)
    return extras
```

### 3.3 Model Metadata Collection

**Source:** `ModelConfig`, HuggingFace config

| Field | Source |
|-------|--------|
| model_hf_id | `model_config.model` (normalize HF path) |
| model_num_parameters | `model_config.hf_config` or `get_num_params_from_config` |
| model_size_bucket | Derived from num_parameters (7B, 70B, etc.) |
| model_hidden_size | `model_config.get_hidden_size()` |
| model_num_layers | `model_config.get_num_layers(parallel_config)` |
| model_max_position_embeddings | From `hf_config` or derived |

**Normalization for model_hf_id:**
- If `model` is HF path (e.g., `meta-llama/Llama-3.1-8B`), use as-is
- If local path, use `None` or hash of path (avoid leaking paths)
- Strip query params, revision suffixes for consistency

**Parameter count:** Many HF configs have `num_parameters` or similar; otherwise estimate from `hidden_size * num_layers * 12` (rough transformer formula). Prefer explicit config.

**Code sketch:**
```python
def _collect_model_extras(model_config: ModelConfig) -> dict[str, Any]:
    extras = {}
    try:
        model_path = getattr(model_config, "model", None)
        if model_path and _is_hf_model_id(model_path):
            extras["model_hf_id"] = _normalize_hf_id(model_path)
        # num_parameters from config
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config:
            num_params = getattr(hf_config, "num_parameters", None) or ...
            if num_params:
                extras["model_num_parameters"] = num_params
                extras["model_size_bucket"] = _to_size_bucket(num_params)
        extras["model_hidden_size"] = model_config.get_hidden_size()
        # num_layers needs parallel_config; may need to pass from vllm_config
    except Exception as e:
        logger.debug("Model metadata collection failed: %s", e)
    return extras
```

### 3.4 Config Extensions

**Add to existing extra_kvs:**
- `max_model_len`: `vllm_config.model_config.max_model_len` (after resolution)

### 3.5 Runtime Workload (Phase 2)

**Design:**
- Add `set_runtime_usage_data` calls at request completion (or equivalent hook)
- Maintain global counters: `prompt_tokens_bucket`, `output_tokens_bucket`, `batch_size_bucket`
- Buckets: `[1-64, 65-256, 257-1024, 1025-4096, 4097-16384, 16385+]`
- On heartbeat, include `prompt_tokens_bucket_counts`, `output_tokens_bucket_counts` as JSON string
- Reset counters after each report (or sliding window)

**Privacy:** Only bucket indices; no raw values. Aggregation over 10-min window.

**Hook location:** Scheduler or engine output path; must be low-overhead (atomic increment).

### 3.6 Schema Versioning

Add to every payload:
```json
"usage_stats_version": 2
```

Server can route/store by version for backward compatibility.

## 4. Backward Compatibility

| Aspect | Strategy |
|--------|----------|
| **Payload** | New fields are additive; server ignores unknown keys |
| **File format** | Same JSONL append to `usage_stats.json` |
| **Opt-out** | No change; `VLLM_NO_USAGE_STATS`, `DO_NOT_TRACK`, `do_not_track` file |
| **Server URL** | Same `VLLM_USAGE_STATS_SERVER`; server must accept new fields |
| **Curlable** | Payload remains flat KV; `curl -X POST -H "Content-Type: application/json" -d @payload.json $URL` |

## 5. Error Handling and Tolerance

- **Collection:** Each collector (hardware, model, config) wrapped in try/except; failures logged at debug, return partial data
- **Transmission:** Existing behavior: `requests.exceptions.RequestException` caught, debug log only
- **File write:** Existing behavior: create dirs, append; failures could raise—consider try/except in `_write_to_file` for extra safety
- **No blocking:** `report_usage` already uses daemon thread; no change

## 6. Testing Strategy

1. **Unit tests:** `_collect_hardware_extras`, `_collect_model_extras` with mocked NVML, ModelConfig
2. **Integration:** Run `vllm serve` with usage stats enabled; verify `usage_stats.json` contains new fields
3. **Opt-out:** Verify `VLLM_NO_USAGE_STATS=1` prevents all collection
4. **Curlable:** Generate payload, `curl -X POST -d @payload.json https://httpbin.org/post` to validate format

## 7. Dependencies

- **pynvml:** vLLM already has `vllm/third_party/pynvml.py`; use for NVML
- **No new deps:** All collection uses existing vLLM imports

## 8. Rollout Plan

1. **Phase 1 PR:** Add `UsageStatsCollector`, hardware + model + max_model_len; feature-flagged or always-on for v2
2. **Server:** Update stats.vllm.ai to accept and store new fields (separate effort)
3. **Phase 2:** Add runtime aggregator and workload buckets
4. **Docs:** Update `docs/usage/usage_stats.md` with new fields and example payload

## 9. Open Questions

1. **Model path normalization:** How to handle S3/GCS paths, custom registries?
2. **Cluster detection:** Reliable multi-node detection without NCCL init?
3. **Parameter count:** Fallback when `num_parameters` not in config?
4. **Heartbeat enrichment:** Which counters to add in Phase 1 vs Phase 2?

## 10. Decision Log

| Decision | Rationale |
|----------|-----------|
| Flat KV only | Server constraint; simplifies parsing and curlability |
| NVML for topology | Standard, already in tree; no new deps |
| Bucket-based workload | Privacy; no raw token counts |
| Additive schema | Backward compatibility; no migration |
| Daemon thread | Keeps collection off critical path |
