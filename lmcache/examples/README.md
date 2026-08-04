# LMCache Examples

This directory contains runnable examples organized by use case. The table below describes what each example does, its hardware requirements, and a recommended learning order for infrastructure engineers getting started with LMCache.

> **Single-node vs. multi-node:** Examples marked with NIXL or UCX require a
> high-bandwidth interconnect (NVLink or PCIe Gen4/5). Running them on a
> single machine with two GPUs is simpler than a true multi-node setup, but
> cross-node deployments add network configuration complexity — don't start
> there if you're still learning the basics.

---

## Before You Start

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`kv_cache_calculator/`](kv_cache_calculator/) | Web UI for calculating KV cache size (GB) given model architecture, dtype, and token count. Start here to size GPU memory and cache tiers before running anything. | None (browser) |

---

## Tier 1 — Core Concepts (Single-Node)

Focus: local caching within a single machine. No cross-node networking required.

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`kv_cache_reuse/local_backends/`](kv_cache_reuse/local_backends/) | Offload KV cache from GPU to CPU memory or local disk. Repeated requests with identical prefixes skip prefill entirely. Includes a Rust-based NVMe backend with `io_uring` support (Linux kernel 5.10+ required for `io_uring`). | 1 GPU; optional NVMe for disk path |
| [`online_session/`](online_session/) | Measure TTFT (time-to-first-token) for cold vs. cache-hit requests. Outputs JSONL for plotting. Includes a sweep script across context lengths. | 1 GPU |

---

## Tier 2 — Disaggregated Prefill and Multi-Instance Sharing

Focus: separating prefill from decode, and sharing KV cache across vLLM instances.

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`disagg_prefill_mp/`](disagg_prefill_mp/) | PD disaggregation via the LMCache multiprocess server. P and D exchange KV through the LMCache MP service — no direct NIXL connection required. **Recommended starting point for PD disaggregation.** | 2 GPUs |
| [`kv_cache_reuse/share_across_engines/`](kv_cache_reuse/share_across_engines/) | An SGLang instance computes a prompt KV cache and a vLLM instance reuses it through one LMCache multiprocess server. Includes a pinned `uv` setup and an automated correctness check. | 2 GPUs (80 GB each for the validated Qwen 32B model) |
| [`disagg_prefill/1p1d/`](disagg_prefill/1p1d/) | PD disaggregation with direct NIXL transfer: 1 prefill server + 1 decode server + a FastAPI proxy. Includes a benchmark script with expected latency numbers. High-bandwidth interconnect (NVLink or PCIe Gen4/5) is strongly recommended — without it, KV transfer overhead may negate the gains. | 2 GPUs + [NIXL](https://github.com/ai-dynamo/nixl) |
| [`kv_cache_reuse/share_across_instances/centralized_sharing/`](kv_cache_reuse/share_across_instances/centralized_sharing/) | Two vLLM instances share one LMCache server. A prefix computed by instance A is reused by instance B. ⚠️ Requires `PYTHONHASHSEED=0` in all processes — without this, hashes differ across processes and cache lookups silently miss. | 2 GPUs (same node) |
| [`kv_cache_reuse/share_across_instances/p2p_sharing/`](kv_cache_reuse/share_across_instances/p2p_sharing/) | Two vLLM instances transfer KV directly peer-to-peer using NIXL and UCX RDMA (`UCX_TLS=rc`). | 2 GPUs + NIXL + UCX |
| [`p2p/`](p2p/) | P2P KV cache sharing in multiprocess mode: each node runs an LMCache server, and a server reads a prefix it lacks directly from the peer that holds it over RDMA. A coordinator handles peer discovery. Includes single-node (debug) and multi-node setups and the logs to expect. | 2+ GPUs (single- or multi-node) + RDMA fabric |

---

## Tier 3 — Production Operations and Observability

Focus: deploying, monitoring, and operating LMCache in production.

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`observability/`](observability/) | Full observability stack: OTel → Prometheus + Tempo → Grafana. Pre-provisioned dashboard shows cache hit rate, read/write throughput, and per-request trace waterfalls. Started with `docker compose up`. | 1 GPU + Docker |
| [`multi_process/`](multi_process/) | Kubernetes DaemonSet YAML for deploying the LMCache server as a per-node sidecar (60 GB L1, 4 workers). Includes resource `requests`/`limits` calibrated to the L1 size. | Kubernetes cluster with GPU nodes |
| [`kubernetes/`](kubernetes/) | `health_probe.py`: readiness/liveness probe script for LMCache in Kubernetes. Complements the DaemonSet config in `multi_process/`. | None |
| [`chunk_statistics/`](chunk_statistics/) | Track chunk reuse rate using a memory Bloom filter or an on-disk hash log. Query via REST (`/chunk_statistics/status`). Use this to answer "does my workload benefit from caching?" before committing to a full deployment. | 1+ GPUs |
| [`cache_controller/`](cache_controller/) | REST APIs for cache orchestration: `lookup` (which instance holds a key), `move` (migrate a hot context between instances), `pin` (prevent eviction), `clear`, and `compress`. | 1 GPU |
| [`cache_with_configs/`](cache_with_configs/) and [`cache_interface/`](cache_interface/) | Per-request control: attach tags (`lmcache.tag.*`), set TTL (`lmcache.ttl`), or skip caching entirely (`lmcache.skip_save: true`). Essential for multi-tenant deployments. | 1 GPU |
| [`remote_config_server/`](remote_config_server/) | Flask reference server for centralised LMCache configuration. Workers POST their current config on startup and receive overrides — useful for managing settings across a large fleet. | None |

---

## Tier 4 — Advanced Features and Ecosystem

### Framework Integrations

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`sgl_integration/`](sgl_integration/) | Using LMCache with SGLang instead of vLLM. | 1 GPU |
| [`frontend/`](frontend/) | Streamlit chat UI demo backed by vLLM + LMCache. Uses the ffmpeg man page as a long shared context to demonstrate TTFT reduction across turns. | 1 GPU |
| [`disagg_prefill/xpyd/`](disagg_prefill/xpyd/) | xP + yD topology: multiple prefill servers feeding multiple decode servers. | 3+ GPUs + NIXL |

### Storage Backends and Serialization

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`kv_cache_reuse/remote_backends/`](kv_cache_reuse/remote_backends/) | Remote storage backends: InfiniStore, Mooncake, S3, Valkey, Redis. Each subdirectory has its own README with backend-specific setup. | 1 GPU + respective backend |
| [`serde/fp8/`](serde/fp8/) | Quantize KV cache to fp8 before writing to disk (L2 adapter), then dequantize on prefetch. Halves disk storage requirements. | Hopper / Ada GPU (H100, RTX 40-series) |
| [`redis_lookup/`](redis_lookup/) | Shows the Redis key schema used by LMCache (`model@world_size@worker_id@chunk_hash`) and `redis-cli` commands for inspecting live cache entries. | Redis + 1 GPU |

### Non-Prefix KV Reuse (CacheBlend)

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`blend_kv_v1/`](blend_kv_v1/) | CacheBlend v1: reuse KV cache even when the new prompt is not a prefix of the cached one (e.g., RAG with swapped documents). Requires a small patch to vLLM source. | 1 GPU (experimental) |
| [`blend_kv/`](blend_kv/) | CacheBlend v0 (legacy): same concept using the old `lmcache_vllm` integration. Kept for reference. | 1+ GPUs (legacy) |

### Developer Extensibility

| Example | What it does | Hardware |
|---------|-------------|----------|
| [`lmc_external_l2_adapter/`](lmc_external_l2_adapter/) and [`lmc_external_native_connector/`](lmc_external_native_connector/) | Templates for writing a custom L2 storage adapter (Python) or a native C++ GPU connector plugin. | Depends on implementation |
| [`runtime_plugins/`](runtime_plugins/) and [`mp_runtime_plugins/`](mp_runtime_plugins/) | Sidecar scripts (Python or shell) that run alongside LMCache workers: heartbeats, metric reporters, alert hooks. Filename prefix controls which role (`scheduler`, `worker_0`, `all`) runs the script. | None |
| [`basic_check/`](basic_check/) | CLI tool for verifying storage backend health and generating test keys. Useful in CI and for on-call diagnostics. | Optional GPU |
| [`agents/`](agents/) | Script for analyzing prefix-hash distribution of a prompt dataset. Useful for estimating cache efficiency before deployment. | None |
