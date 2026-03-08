# Usage Stats V2: Product Requirements Document

**Status:** Draft  
**Project:** Usage Stats V2  
**Linear Issue:** VLLM-94  
**Last Updated:** 2025-03-08

## Executive Summary

vLLM's current usage stats capture basic GPU and model configuration but lack the depth needed to drive product decisions around feature deprecation, model support prioritization, and optimization investments. This PRD defines requirements for Usage Stats V2—a comprehensive, privacy-preserving telemetry system that enables data-driven development of vLLM while maintaining backward compatibility, low footprint, and robust queryability.

## 1. Goals and Success Criteria

### Primary Goals

1. **Product Intelligence:** Inform which features to deprecate, which models to prioritize/deprecate, and which optimizations to improve
2. **Hardware Understanding:** Capture cluster, topology, and networking information to understand deployment patterns
3. **Model Intelligence:** Beyond architecture class names, capture model metadata (base checkpoints, parameter counts, sizes)
4. **Workload Characterization:** Understand typical context lengths and input-output distributions (anonymized)
5. **Future Direction:** Provide data to guide vLLM's roadmap and competitive positioning

### Success Criteria

- **Backward compatible:** Existing opt-out mechanisms and data format remain functional
- **Low footprint:** Minimal CPU/memory overhead, non-blocking collection
- **Tolerant:** Failures in collection or transmission do not affect inference
- **Queryable:** Data easily curlable as key-value pairs; compatible with data warehouse infrastructure
- **Privacy-preserving:** No sensitive information; workload data sufficiently anonymized

## 2. Current State Analysis

### 2.1 What vLLM Collects Today

**Source:** `vllm/usage/usage_lib.py`, `vllm/v1/utils.py` (report_usage_stats)

| Category | Fields | Notes |
|----------|--------|-------|
| **Environment** | provider, num_cpu, cpu_type, cpu_family_model_stepping, total_memory, architecture, platform | Cloud detection via DMI, env vars |
| **GPU** | gpu_count, gpu_type, gpu_memory_per_device, cuda_runtime | Single device properties; no topology |
| **Model** | model_architecture | Class name only (e.g., LlamaForCausalLM) |
| **vLLM Config** | dtype, block_size, gpu_memory_utilization, kv_cache_memory_bytes, quantization, kv_cache_dtype | |
| **Features** | enable_lora, enable_prefix_caching, enforce_eager, disable_custom_all_reduce | |
| **Parallelism** | tensor_parallel_size, data_parallel_size, pipeline_parallel_size, enable_expert_parallel, all2all_backend, kv_connector | |
| **Metadata** | uuid, log_time, source, context (UsageContext) | |
| **Env Vars** | VLLM_USE_MODELSCOPE, VLLM_USE_FLASHINFER_SAMPLER, etc. | JSON blob |

**Reporting Flow:**
- One-time report at engine init (from v1 workers: gpu_worker, xpu_worker)
- Continuous heartbeat every 10 minutes (uuid, log_time, _GLOBAL_RUNTIME_DATA)
- Writes to `~/.config/vllm/usage_stats.json` (append JSONL)
- POSTs to `https://stats.vllm.ai`

**Opt-out:** `VLLM_NO_USAGE_STATS`, `DO_NOT_TRACK`, `VLLM_DO_NOT_TRACK`, `~/.config/vllm/do_not_track`

### 2.2 Gaps Identified

| Gap | Impact |
|-----|--------|
| **GPU topology** | Cannot distinguish NVLink vs PCIe clusters; affects TP/PP optimization decisions |
| **Cluster info** | Unknown multi-node vs single-node distribution |
| **Networking** | No InfiniBand/RoCE vs Ethernet visibility |
| **Model metadata** | No base model ID, parameter count, or size; hard to prioritize model support |
| **Context length** | No visibility into typical max_model_len or enabled context |
| **Workload distribution** | No input/output token distributions; cannot tune batching/scheduling |
| **Runtime metrics** | Heartbeat has minimal data; no aggregated performance signals |

## 3. Competitive Landscape

### 3.1 Hugging Face Text Generation Inference (TGI)

- **Scope:** Docker-only; startup/shutdown + 15-min heartbeat
- **Model:** model_type, tokenizer_class, revision
- **System:** CPU count/type, memory, architecture, platform
- **GPU:** device name, driver version, memory usage, temperature, utilization, power draw, compute capability, ECC errors
- **Config:** max_batch_prefill_tokens, max_batch_size, max_input_tokens, max_total_tokens, max_concurrent_requests
- **Opt-out:** `--usage-stats=no-stack` (omit stack traces), `--usage-stats=off` (full disable)

### 3.2 NVIDIA TensorRT-LLM

- **Metrics:** Prometheus endpoint (`/prometheus/metrics`)
- **Request-level:** KV cache hit rates, TTFT, TPOT, e2e latency, queue time, finish reasons
- **Aggregate:** Running/waiting requests, prompt/generation token totals, per-iteration stats
- **Model:** Model name and version tracking

### 3.3 Takeaways

- TGI: Rich GPU telemetry (temp, power, ECC); batch/config limits; Docker-gated
- TensorRT-LLM: Production-oriented metrics; Prometheus-native; request-level histograms
- vLLM opportunity: Combine hardware topology (unique), model metadata (HF-style), and workload distributions (anonymized) for product intelligence

## 4. Requirements

### 4.1 Hardware / GPU (Cluster, Topology, Networking)

| Requirement | Priority | Description |
|-------------|----------|-------------|
| H1 | P0 | **GPU topology:** NVLink link count, PCIe gen/width, P2P capability summary |
| H2 | P0 | **Cluster size:** Number of nodes (if detectable), GPUs per node |
| H3 | P1 | **Networking:** InfiniBand/RoCE vs Ethernet (via env or driver hints) |
| H4 | P1 | **Compute capability:** SM version (e.g., 8.0, 9.0) for kernel compatibility |
| H5 | P2 | **Driver/CUDA:** Driver version, CUDA version (already partially present) |

**Implementation hints:**
- NVML: `nvmlDeviceGetNvLinkState`, `nvmlDeviceGetPcieInfo`, `nvmlDeviceGetCudaComputeCapability`
- `nvidia-smi topo -m` (parse or use NVML equivalents)
- Cluster: `NCCL_*` env vars, `RANK`, `WORLD_SIZE` patterns

### 4.2 Model Metadata

| Requirement | Priority | Description |
|-------------|----------|-------------|
| M1 | P0 | **Base model identifier:** HuggingFace model ID or normalized name (e.g., `meta-llama/Llama-3.1-8B`) |
| M2 | P0 | **Parameter count:** Total parameters (from config or loaded model) |
| M3 | P1 | **Model size bucket:** e.g., 7B, 70B, 405B for bucketing |
| M4 | P1 | **Hidden size, num layers:** For architecture variant differentiation |
| M5 | P2 | **Revision/commit:** Model revision if from HF Hub |

**Privacy:** Use model ID from public registries (HF, etc.); avoid custom paths that could leak org names.

### 4.3 Runtime / Workload (Anonymized)

| Requirement | Priority | Description |
|-------------|----------|-------------|
| R1 | P0 | **max_model_len:** Configured context length (already derivable; ensure reported) |
| R2 | P0 | **Input/output distributions:** Histogram buckets (e.g., prompt_tokens: 1-64, 65-256, 257-1K, 1K-4K, 4K+) |
| R3 | P1 | **Batch size distribution:** Concurrent request counts over time |
| R4 | P1 | **Feature usage:** Prefix cache hit rate, speculative decoding usage (if enabled) |
| R5 | P2 | **Latency percentiles:** TTFT, TPOT (aggregated, no per-request) |

**Anonymization rules:**
- No raw token counts; only bucket indices or aggregates
- No request IDs, timestamps finer than minute-level
- No user-identifiable content

### 4.4 Non-Functional

| Requirement | Description |
|-------------|-------------|
| NF1 | **Backward compatibility:** Existing `usage_stats.json` format and server contract; new fields additive |
| NF2 | **Opt-out:** All existing mechanisms remain; no new tracking without opt-out |
| NF3 | **Low footprint:** Collection async, <1% CPU; no blocking on network |
| NF4 | **Tolerance:** Exceptions in collection do not affect inference |
| NF5 | **Curlable:** Payload as flat key-value JSON; easy to inspect and replay |
| NF6 | **Warehouse-ready:** Schema stable; suitable for BigQuery/Snowflake/etc. |

## 5. Data Schema (Proposed Additions)

### 5.1 New Fields (Flat KV)

```
# Hardware
gpu_topology_nvlink_links: int | null      # NVLink links per GPU (0 if PCIe-only)
gpu_topology_pcie_gen: int | null         # e.g., 4, 5
gpu_topology_pcie_width: int | null       # e.g., 16
gpu_compute_capability: str | null        # e.g., "8.0", "9.0"
cluster_node_count: int | null            # 1 if single-node
cluster_gpus_per_node: int | null

# Model
model_hf_id: str | null                   # e.g., "meta-llama/Llama-3.1-8B"
model_num_parameters: int | null          # e.g., 8_000_000_000
model_size_bucket: str | null              # "7B", "70B", "405B"
model_hidden_size: int | null
model_num_layers: int | null
model_max_position_embeddings: int | null

# Runtime (anonymized)
max_model_len: int                         # configured context length
prompt_tokens_bucket_counts: str | null   # JSON: {"1-64": N, "65-256": M, ...}
output_tokens_bucket_counts: str | null   # same
batch_size_bucket_counts: str | null      # concurrent requests over time
prefix_cache_hit_rate_avg: float | null   # 0-1, if prefix caching enabled
```

### 5.2 Heartbeat Enrichment

Continuous heartbeat (every 10 min) could include:
- `uptime_seconds`
- `total_requests_served` (counter)
- `total_prompt_tokens`, `total_output_tokens` (counters)
- `prefix_cache_hits`, `prefix_cache_queries` (if enabled)

## 6. Privacy and Anonymization

### 6.1 Principles

- **Minimize:** Collect only what is needed for stated goals
- **Anonymize:** No PII; workload data in coarse buckets only
- **Transparency:** Document all fields in public docs (as today)
- **User control:** Opt-out must be easy and complete

### 6.2 Sensitive Data (Never Collect)

- Model paths that could reveal org/user (e.g., `/home/company/secret-model`)
- Request content, prompts, or outputs
- IP addresses, hostnames (beyond cloud provider detection)
- Fine-grained timestamps that could correlate with external logs

### 6.3 Workload Anonymization

- **Prompt/output lengths:** Use fixed buckets (e.g., 1-64, 65-256, 257-1024, 1025-4096, 4097-16384, 16385+)
- **Aggregation:** Report counts per bucket over a time window (e.g., 10 min)
- **No linkage:** No request-level data; only aggregates

## 7. Open Source and Product Angles

### 7.1 Open Source

- Public schema and documentation
- Transparent opt-out
- Align with DO_NOT_TRACK and similar conventions
- Publish aggregated reports (e.g., 2024.vllm.ai) to build trust

### 7.2 Product

- Data drives deprecation (low-usage features)
- Model support prioritization (high-usage architectures)
- Optimization ROI (where to invest in kernels, memory, etc.)
- Competitive differentiation (e.g., "vLLM runs on X% of H100 clusters")

### 7.3 Mature PM Considerations

- **Stakeholder trust:** Clear docs, minimal data, strong opt-out
- **Legal:** GDPR/privacy compliance; avoid consent fatigue
- **Competitive:** Don't over-collect; avoid perception of surveillance
- **Technical debt:** Design for schema evolution; version the payload

## 8. Implementation Phases

### Phase 1 (MVP)
- Add model metadata (M1, M2, M3)
- Add max_model_len to initial report
- Add GPU topology (H1, H4) where NVML available
- No runtime workload yet

### Phase 2
- Add cluster info (H2)
- Add workload buckets (R1, R2) with anonymization
- Enrich heartbeat with aggregate counters

### Phase 3
- Add networking hints (H3)
- Add feature usage (R4)
- Refine schema based on warehouse usage

## 9. Appendix: Current Code References

- **Usage lib:** `vllm/usage/usage_lib.py`
- **Report call:** `vllm/v1/utils.py::report_usage_stats`
- **Workers:** `vllm/v1/worker/gpu_worker.py`, `vllm/v1/worker/xpu_worker.py`
- **Model loader:** `vllm/model_executor/model_loader/utils.py::get_architecture_class_name`
- **Platform utils:** `vllm/utils/platform_utils.py::cuda_get_device_properties`
- **NVML bindings:** `vllm/third_party/pynvml.py` (NVLink, PCIe constants)
- **Config:** `vllm/config/model.py` (ModelConfig: max_model_len, get_hidden_size, get_num_layers)
