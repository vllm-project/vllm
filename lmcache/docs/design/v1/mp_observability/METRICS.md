# LMCache MP Mode Observability Metrics

## Overview

The observability system uses an **EventBus with pub/sub dispatch** and
**OpenTelemetry** for metrics instrumentation.

- **Producers** (`L1Manager`, `StorageManager`, `MPCacheServer`) publish `Event` objects
  to the EventBus.
- **Metrics subscribers** (e.g. `L1MetricsSubscriber`, `L2MetricsSubscriber`) subscribe to
  specific event types and update OTel counters.
- **Logging subscribers** (`MPServerLoggingSubscriber`) log events at debug level.
- **Tracing subscribers** (`MPServerTracingSubscriber`) create OTel spans from START/END pairs.
- **Export** is via OTLP push to an OTel collector (production) or an in-process
  Prometheus `/metrics` endpoint (dev/debug fallback).

All metrics use the `lmcache_mp.` prefix (mp = multiprocess), distinct from the main
engine's `lmcache.` namespace. On Prometheus, `.` is converted to `_` and counters get
a `_total` suffix (e.g., `lmcache_mp.l1_read` with `unit="chunks"` is exposed as
`lmcache_mp_l1_read_chunks_total`).

For implementation guidance on adding new events and subscribers, see [README.md](README.md).

## Global Resource Attributes

Every metric (and span) exported by an MP server carries Resource-level
attributes built at startup:

| Attribute | CLI flag | Source | Applies to |
|---|---|---|---|
| `service.instance.id` | `--instance-id` | `MPServerConfig.instance_id` (defaults to a random UUID v4 at startup; projected onto `ObservabilityConfig.service_instance_id` by `run_cache_server` so telemetry and coordinator membership share one id) | All metrics + spans |

Resource attributes are attached to the `MeterProvider` / `TracerProvider`
in `otel_init.py` and therefore appear on every datapoint exported via
OTLP.  On Prometheus, SDK resource attributes are typically surfaced via
the `target_info` series rather than on each time-series.

Per-metric attributes (e.g. `cache_salt`) remain on the individual
datapoints and are orthogonal to these Resource attributes.

---

## L1 Read Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l1_read` | `lmcache_mp_l1_read_chunks_total` | Counter (attr: `cache_salt`) | `L1_READ_FINISHED` | `+len(keys)` per `cache_salt` |

**What it answers:** How many chunks are being read from L1?

> **Note:** `L1_READ_RESERVED` is published but has no metrics subscriber — key counts
> are recorded only when the read actually completes.

---

## L1 Write Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l1_write` | `lmcache_mp_l1_write_chunks_total` | Counter (attr: `cache_salt`) | `L1_WRITE_FINISHED` | `+len(keys)` per `cache_salt` |
| *(same counter)* | *(same)* | Counter (attr: `cache_salt`) | `L1_WRITE_FINISHED_AND_READ_RESERVED` | `+len(keys)` per `cache_salt` |

**What it answers:** How many chunks are being written to L1?

> **Note:** `L1_WRITE_RESERVED` is published but has no metrics subscriber.
> `L1_WRITE_FINISHED_AND_READ_RESERVED` (atomic write-then-read used by prefetch)
> increments the same write counter.

---

## L1 Eviction Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l1_evicted` | `lmcache_mp_l1_evicted_chunks_total` | Counter (attr: `cache_salt`) | `L1_KEYS_EVICTED` | `+len(keys)` per `cache_salt` |
| `lmcache_mp.l1_eviction_loop_ticks` | `lmcache_mp_l1_eviction_loop_ticks_total` | Counter | `L1_EVICTION_LOOP_TICK` | +1 per loop iteration |
| `lmcache_mp.l1_eviction_loop_triggered` | `lmcache_mp_l1_eviction_loop_triggered_total` | Counter | `L1_EVICTION_LOOP_TICK` | +1 when `triggered=True` |
| `lmcache_mp.l1_usage_ratio` | `lmcache_mp_l1_usage_ratio` | Observable Gauge | (callback on `L1Manager`) | `used / total` at scrape time |

**What it answers:** How aggressively is the eviction controller clearing L1?  Is the eviction loop alive but staying below the watermark, or actively firing?  What is the current L1 fullness?

The two loop counters distinguish "loop is alive" from "eviction fired" — important when debugging short-lived benchmarks (a workload that completes in <1 s never gives the 1Hz polling loop a chance to fire even when usage exceeds the watermark).  `l1_usage_ratio` is registered via :func:`register_gauge` against `L1Manager`, so its value reflects current state at scrape time, not a per-tick sample.

---

## L1 Failure Metrics (LM-291 health monitoring)

Tagged counters covering L1 allocation and read failures. The subscriber
groups keys by `ObjectKey.model_name` to emit a `model_name` OTel
attribute on every data point, enabling per-model Prometheus slicing
(e.g. `lmcache_mp_l1_allocation_failure_total{during="l1_store",model_name="llama-7b"}`).

The ticket-specified `lmcache_instance_id` tag is **deferred** to a
follow-up: threading it through `StorageManager`/`StoreController` would
require a cross-cutting API change out of scope for this PR.

| OTel metric name | Prometheus name | Type | Source event | Calculation | Tags |
|---|---|---|---|---|---|
| `lmcache_mp.l1_allocation_failure` | `lmcache_mp_l1_allocation_failure_chunks_total` | Counter | `L1_ALLOCATION_FAILED` | `+count` per `(during, model_name)` bucket | `during` ∈ {`l1_store`, `l2_prefetch`}, `model_name` |
| `lmcache_mp.l1_read_failure` | `lmcache_mp_l1_read_failure_chunks_total` | Counter | `L1_READ_FAILED` | `+count` per `(during, reason, model_name)` bucket | `during` ∈ {`l2_store`, `l1_retrieve`}, `reason` ∈ {`not_found`, `write_locked`}, `model_name` |

**What it answers:**
- `l1_allocation_failure` — how often is L1 rejecting writes for lack of memory, split by whether the pressure is user stores or L2 prefetch?
- `l1_read_failure` — a **post-lookup anomaly counter**, not a cache-miss counter. Should stay near zero in healthy operation; any non-zero value indicates a lookup/reserve race or unexpected eviction in MP mode.

---

## Timeout Metrics

Cross-component anomaly counter. Incremented once per `LMCacheTimeoutError`
constructed (see `errors.py` and the `TIMEOUT_RAISED` event in
[EVENTS.md](EVENTS.md)), tagged by `exception_type` so operators can alert on
the timeout rate per class.

| OTel metric name | Prometheus name | Type | Source event | Calculation | Tags |
|---|---|---|---|---|---|
| `lmcache_mp.timeouts` | `lmcache_mp_timeouts_total` | Counter | `TIMEOUT_RAISED` | `+1` per event | `exception_type` |

**What it answers:** how often are operations timing out, and of which kind?
Should stay near zero in healthy operation; a rising rate signals an
overloaded or stuck MQ/transfer/adapter path.

---

## L1 Chunk Lifecycle Histograms

Sampled (default 1%) chunk-level lifecycle tracking.  Only sampled chunks
contribute to histograms; counters above always count all events.

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l1_chunk_lifetime` | `lmcache_mp_l1_chunk_lifetime_seconds` | Histogram | `L1_KEYS_EVICTED` | `eviction_time - alloc_time` per sampled chunk |
| `lmcache_mp.l1_chunk_idle_before_evict` | `lmcache_mp_l1_chunk_idle_before_evict_seconds` | Histogram | `L1_KEYS_EVICTED` | `eviction_time - last_access_time` per sampled chunk |
| `lmcache_mp.l1_chunk_reuse_gap` | `lmcache_mp_l1_chunk_reuse_gap_seconds` | Histogram | `L1_READ_FINISHED`, `L1_WRITE_FINISHED`, `L1_WRITE_FINISHED_AND_READ_RESERVED` | Time gap between consecutive touches of the same chunk |
| `lmcache_mp.l1_chunk_evict_reuse_gap` | `lmcache_mp_l1_chunk_evict_reuse_gap_seconds` | Histogram | `L1_KEYS_EVICTED` → `L1_WRITE_FINISHED` | Time from eviction to next reuse (capped at 300 s) |
| `lmcache_mp.real_reuse_gap` | `lmcache_mp_real_reuse_gap_seconds` | Histogram (tagged `cache_salt`) | `SM_READ_PREFETCHED_FINISHED`, `SM_WRITE_FINISHED` | Time gap between a chunk's last access (read or write) and the next read.  Captures **storage cost**.  Emitted only on read events. |
| `lmcache_mp.real_reuse_gap_objects` | `lmcache_mp_real_reuse_gap_objects_chunks` | Histogram (tagged `cache_salt`) | `SM_READ_PREFETCHED_FINISHED`, `SM_WRITE_FINISHED` | Per-`cache_salt` access-counter gap between two reads of the same chunk.  Counter bumps on every read and write of every chunk; histogram emitted only on read events for sampled chunks.  Captures **storage volume**. |

**What it answers:** How long do L1 chunks live? How idle are they before eviction? How quickly are evicted chunks reused?

---


## L2 Store Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l2_store_submitted` | `lmcache_mp_l2_store_submitted_requests_total` | Counter | `L2_STORE_SUBMITTED` | +1 per event |
| `lmcache_mp.l2_store_submitted_objects` | `lmcache_mp_l2_store_submitted_objects_chunks_total` | Counter (attr: `cache_salt`) | `L2_STORE_SUBMITTED` | `+count` per `cache_salt` via `key_count_per_salt` |
| `lmcache_mp.l2_store_completed` | `lmcache_mp_l2_store_completed_requests_total` | Counter (attr: `l2_name`) | `L2_STORE_COMPLETED` | +1 per event |
| `lmcache_mp.l2_store_completed_objects` | `lmcache_mp_l2_store_completed_objects_chunks_total` | Counter (attr: `cache_salt`) | `L2_STORE_COMPLETED` | `+count` per `cache_salt` via `key_count_per_salt` |

**What it answers:** How many chunks are being pushed to L2? What fraction fail?

---

## L2 Prefetch Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l2_prefetch_lookup` | `lmcache_mp_l2_prefetch_lookup_requests_total` | Counter | `L2_PREFETCH_LOOKUP_SUBMITTED` | +1 per event |
| `lmcache_mp.l2_prefetch_lookup_objects` | `lmcache_mp_l2_prefetch_lookup_objects_chunks_total` | Counter (attr: `cache_salt`) | `L2_PREFETCH_LOOKUP_SUBMITTED` | `+count` per `cache_salt` via `key_count_per_salt` |
| `lmcache_mp.l2_prefetch_hit` | `lmcache_mp_l2_prefetch_hit_chunks_total` | Counter | `L2_PREFETCH_LOOKUP_COMPLETED` | `+prefix_hit_count` |
| `lmcache_mp.l2_prefetch_load_submitted` | `lmcache_mp_l2_prefetch_load_submitted_requests_total` | Counter | `L2_PREFETCH_LOAD_SUBMITTED` | `+adapter_count` (per-adapter task count) |
| `lmcache_mp.l2_prefetch_load_submitted_objects` | `lmcache_mp_l2_prefetch_load_submitted_objects_chunks_total` | Counter (attr: `cache_salt`) | `L2_PREFETCH_LOAD_SUBMITTED` | `+count` per `cache_salt` via `key_count_per_salt` |
| `lmcache_mp.l2_prefetch_load_completed` | `lmcache_mp_l2_prefetch_load_completed_chunks_total` | Counter (attr: `cache_salt`) | `L2_PREFETCH_LOAD_COMPLETED` | `+count` per `cache_salt` via `key_count_per_salt` |
| `lmcache_mp.l2_load_completed` | `lmcache_mp_l2_load_completed_requests_total` | Counter (attr: `l2_name`) | `L2_LOAD_TASK_COMPLETED` | +1 per event |

**What it answers:** How effective is L2 prefetching? What is the L2 hit rate?

**Per-backend IOPS.**  `lmcache_mp.l2_store_completed` (attr `l2_name`) counts
completed L1→L2 store tasks; `lmcache_mp.l2_load_completed` (attr `l2_name`)
counts completed per-adapter L2→L1 load tasks.  Derive per-backend ops/sec on
the dashboard with
`rate(lmcache_mp_l2_store_completed_requests_total{l2_name="..."}[1m])`
(and the equivalent for loads).  No separate `*_iops` metric is exported — the
raw counter keeps the window choice in the dashboard.

---

## L2 Eviction Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l2_evicted_objects` | `lmcache_mp_l2_evicted_chunks_total` | Counter (attr: `cache_salt`) | `L2_KEYS_EVICTED` | `+count` per `cache_salt` via `key_count_per_salt` |

**What it answers:** How many chunks are being evicted from L2? Which tenants are losing data?

---

## Lookup Hit-Rate Metrics (L1 + L2 combined)

Token-level counters derived from the `MP_LOOKUP_PREFETCH_END` event.  Their
ratio is the fraction of tokens requested by a lookup that were served from
either L1 or L2.  L0 (GPU prefix cache) is intentionally excluded — it is
vLLM-owned and not observable from LMCache.

Both counters carry `model_name` and `cache_salt` OTel attributes (captured
at lookup time from `IPCCacheServerKey`), enabling per-model and per-tenant
slicing of the hit rate.  `cache_salt` can be high-cardinality; drop it at
scrape time with `metric_relabel_configs` if storage cost matters.

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.lookup_requested` | `lmcache_mp_lookup_requested_tokens_total` | Counter (attrs: `model_name`, `cache_salt`) | `MP_LOOKUP_PREFETCH_END` | `+requested_tokens` |
| `lmcache_mp.lookup_hit` | `lmcache_mp_lookup_hit_tokens_total` | Counter (attrs: `model_name`, `cache_salt`) | `MP_LOOKUP_PREFETCH_END` | `+hit_tokens` |

**What it answers:** What fraction of tokens requested by a lookup were served from cache (L1 or L2)?

```promql
# Aggregate hit rate (all models, all salts):
rate(lmcache_mp_lookup_hit_tokens_total[5m])
/ rate(lmcache_mp_lookup_requested_tokens_total[5m])

# Per-model hit rate:
sum(rate(lmcache_mp_lookup_hit_tokens_total[5m])) by (model_name)
/ sum(rate(lmcache_mp_lookup_requested_tokens_total[5m])) by (model_name)
```

> **Note:** Both counters are driven by the *same* event, so they always
> advance together per completed lookup.  Early-exit lookups (no GPU
> context matches, empty `chunk_hashes`) contribute `0` to both, and
> abandoned lookups (client never polls `query_prefetch_status`)
> contribute to neither.  See
> [L1_L2_HIT_RATE_PLAN.md](L1_L2_HIT_RATE_PLAN.md) for the full rationale.

---

## L2 Failure Metrics (LM-291 health monitoring)

| OTel metric name | Prometheus name | Type | Source event | Calculation | Tags |
|---|---|---|---|---|---|
| `lmcache_mp.l2_prefetch_failure` | `lmcache_mp_l2_prefetch_failure_chunks_total` | Counter | `L2_PREFETCH_FAILED` | `+count` per `(reason, model_name)` bucket | `reason` ∈ {`l1_oom`, `not_found`}, `model_name` |

**What it answers:** For keys L2 reported present at lookup but failed to land in L1: was L1 full (`l1_oom`), or did the adapter fail to produce the data (`not_found`)?

> **Serde failures**: a third `reason=serde_failure` value will be added as an additive, non-breaking extension once the serde PR lands and L2 adapters distinguish deserialization errors from missing objects. No dashboard migration needed when that happens.

> **TTL lock expiration**: `lmcache_mp.l1_ttl_lock_expire` from the ticket is deferred to a follow-up because the current `TTLLock` primitive (native) has no expiration callback; lazy detection requires a C++/Rust-side change.

---

## L0 (GPU) Block Lifecycle Histograms

Sampled (default 1%) GPU KV cache block lifecycle tracking via shadow monitoring
of `MP_VLLM_BLOCK_ALLOCATION` and `MP_VLLM_END_SESSION` events.  Eviction is
detected at reallocation time (when a block is assigned different tokens).

All L0 histograms carry `instance_id` and `model_name` OTel attributes, enabling
per-instance and per-model Prometheus metric slicing (e.g.
`lmcache_mp_l0_block_lifetime_seconds{instance_id="12345",model_name="llama-7b"}`).

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l0_block_lifetime` | `lmcache_mp_l0_block_lifetime_seconds` | Histogram | `MP_VLLM_BLOCK_ALLOCATION` (eviction detected) | `eviction_time - alloc_time` per sampled block |
| `lmcache_mp.l0_block_idle_before_evict` | `lmcache_mp_l0_block_idle_before_evict_seconds` | Histogram | `MP_VLLM_BLOCK_ALLOCATION` (eviction detected) | `eviction_time - last_access_time` per sampled block |
| `lmcache_mp.l0_block_reuse_gap` | `lmcache_mp_l0_block_reuse_gap_seconds` | Histogram | `MP_VLLM_BLOCK_ALLOCATION` (cache hit) | Time gaps between consecutive accesses from access history |

**What it answers:** How long do GPU blocks live before eviction? How idle are they? How frequently are cached blocks reused? Which instance/model is experiencing the most churn?

---

## L0 ↔ L1 Throughput Histograms

Per-request throughput of GPU↔CPU copies via
`L0L1ThroughputSubscriber`. Correlates `MP_{STORE,RETRIEVE}_START` → `MP_{STORE,RETRIEVE}_END`
pairs by `session_id`, computes `total_bytes / (end_ts - start_ts)` in GB/s.
Every request contributes one sample (no sampling).
START/END events fire on the GPU cupy stream (`publish_on_stream`), so
timestamps reflect true GPU-stream copy time — not Python/lock overhead.

All throughput histograms carry `engine_id` (vLLM worker instance id),
`device` (e.g. `"cuda:3"`), and `model_name` OTel attributes, enabling
per-worker, per-device, and per-model slicing in Prometheus (e.g.
`lmcache_mp_l0_l1_store_throughput_gbs{engine_id="0",device="cuda:3",model_name="meta-llama/Llama-3.1-8B"}`).

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l0_l1_store_throughput` | `lmcache_mp_l0_l1_store_throughput_GBs` | Histogram | `MP_STORE_START` → `MP_STORE_END` | `total_bytes / (end_ts - start_ts) / 1e9` per request |
| `lmcache_mp.l0_l1_load_throughput` | `lmcache_mp_l0_l1_load_throughput_GBs` | Histogram | `MP_RETRIEVE_START` → `MP_RETRIEVE_END` | `total_bytes / (end_ts - start_ts) / 1e9` per request |

**What it answers:** What GPU↔CPU throughput is each vLLM worker actually
achieving for KV store/load? Does it match the theoretical PCIe bandwidth?
Are some workers or GPUs underperforming?

---

## L1 ↔ L2 Throughput Histograms

Per-task throughput of L1↔L2 transfers via
`L2ThroughputSubscriber`. The store path correlates `L2_STORE_SUBMITTED` →
`L2_STORE_COMPLETED` by `(adapter_index, task_id)`. The load path
correlates the new per-adapter `L2_LOAD_TASK_SUBMITTED` →
`L2_LOAD_TASK_COMPLETED` events by `(request_id, adapter_index)`; the
pre-existing request-level `L2_PREFETCH_LOAD_*` events aggregate across
adapters and cannot attribute throughput to a specific `l2_name`.
Every task contributes one sample (no sampling).

Unlike the L0↔L1 histograms, these timestamps span **submit → complete**,
so `(end_ts - start_ts)` includes adapter queue, network, and disk I/O
time. Treat the value as *bytes / end-to-end latency*, not raw transfer
rate — useful for comparing adapter types and tracking regressions, not
for validating peak fabric bandwidth.

All throughput histograms carry a single `l2_name` OTel attribute — the
registered adapter type (e.g. `"fs"`, `"nixl_store"`, `"mooncake_store"`)
— enabling per-adapter-type slicing in Prometheus (e.g.
`lmcache_mp_l2_store_throughput_gbs{l2_name="nixl_store"}`).

**Store-path fast-path accounting.** Some adapters skip the write when
a key is already present in the backend, collapsing
`(completed_ts - submitted_ts)` to near-zero while the submitted
`total_bytes` count stays unchanged. To avoid inflated throughput
samples, the `L2StoreResult` returned by `pop_completed_store_tasks()`
carries `bytes_transferred()` covering only the bytes actually written.
The `L2_STORE_COMPLETED` event propagates this value, and the store
throughput subscriber records `bytes_transferred / dt`; when
`bytes_transferred <= 0` (every key fast-pathed) the sample is dropped
entirely. The load path continues to use the submitted `total_bytes`.

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l2_store_throughput` | `lmcache_mp_l2_store_throughput_GBs` | Histogram | `L2_STORE_SUBMITTED` → `L2_STORE_COMPLETED` | `bytes_transferred / (completed_ts - submitted_ts) / 1e9` per task. `bytes_transferred` is read from the `L2_STORE_COMPLETED` event (populated from the `L2StoreResult` returned by `pop_completed_store_tasks()`); samples where `bytes_transferred <= 0` (e.g. duplicate-key fast paths that skip the write) are dropped, so the histogram reflects real work, not submitted-but-skipped bytes. |
| `lmcache_mp.l2_load_throughput` | `lmcache_mp_l2_load_throughput_GBs` | Histogram | `L2_LOAD_TASK_SUBMITTED` → `L2_LOAD_TASK_COMPLETED` | `total_bytes / (completed_ts - submitted_ts) / 1e9` per (request, adapter) pair. The load path still uses submitted `total_bytes`; per-task real-bytes accounting only applies to the store path. |

**What it answers:** What end-to-end throughput is each L2 adapter
delivering? Which backends are keeping up with demand, and which are
queue-bound or I/O-bound?

---

## Engine Counters

Worker-scoped counters tied to what the MP server delivers back to each
vLLM worker.  Labeled by `worker_id` — the vLLM worker instance id,
distinct from any scheduler-scoped id used elsewhere.

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_mp.num_chunks_loaded` | `lmcache_mp_num_chunks_loaded_total` | Counter (attrs: `worker_id`, `model_name`, `cache_salt`) | `MP_RETRIEVE_END` | `+retrieved_count` per event |

**What it answers:** How many LMCache chunks is each vLLM worker loading
from LMCache into its engine?  Compare across workers to spot uneven
demand or underserved ranks.  Slice by `model_name` to see per-model
load volume in multi-model deployments, or by `cache_salt` for per-tenant
attribution (note: `cache_salt` can be high-cardinality — drop it at
scrape time with `metric_relabel_configs` if storage cost matters).

---

## EventBus Self-Monitoring

Health metrics for the EventBus itself. The two gauges are registered
inside `EventBus.__init__` via `register_gauge`; the two observable
counters are registered by `EventBusSelfMetricsSubscriber`. Unlike the
other metrics subscribers, these are not driven by events — they observe
bus state directly via the `EventBus` accessors and report on every OTel
scrape.

| OTel metric name | Prometheus name | Type | Source | Calculation |
|---|---|---|---|---|
| `lmcache_mp.event_bus.queue_depth` | `lmcache_mp_event_bus_queue_depth` | ObservableGauge | `EventBus.queue_depth()` | `len(_queue)` at scrape time |
| `lmcache_mp.event_bus.drain_lag_seconds` | `lmcache_mp_event_bus_drain_lag_seconds` | ObservableGauge | `EventBus.oldest_event_lag_seconds()` | `time.time() - oldest.timestamp`, or `0.0` when empty |
| `lmcache_mp.event_bus.dropped_events_total` | `lmcache_mp_event_bus_dropped_events_total` | ObservableCounter | `EventBus.dropped_events_count()` | cumulative `_discard_count` |
| `lmcache_mp.event_bus.subscriber_exceptions` | `lmcache_mp_event_bus_subscriber_exceptions_total` | ObservableCounter (attr: `subscriber_name`) | `EventBus.subscriber_exception_counts()` | cumulative count per subscriber, incremented when `_drain_all` catches a callback exception |

**What it answers:** Is the EventBus keeping up with publishers? Is anything being dropped? Are any subscriber callbacks raising?

`subscriber_name` is derived from the failing callback: bound methods report their owning class (e.g. `L1MetricsSubscriber`); free functions report `__qualname__`.

---

## MPCacheServer Observable Gauges

These metrics are registered directly via `register_gauge` (pull-based OTel
observable gauges) rather than through the EventBus, because they represent
point-in-time state snapshots that do not correspond to discrete events.

| OTel metric name | Prometheus name | Type | Source | Calculation |
|---|---|---|---|---|
| `lmcache_mp.active_prefetch_jobs` | `lmcache_mp_active_prefetch_jobs` | ObservableGauge | `MPCacheServer._prefetch_jobs` | `len(_prefetch_jobs)` at scrape time |

**What it answers:** How many prefetch jobs are currently in-flight? A sustained high value may indicate slow L2 backends or client-side polling delays.

---

## L1 / L2 State Metrics

Live state of the L1 memory pool, the per-adapter L2 byte usage, and the
in-flight L2 store / prefetch-load queues.  These metrics are useful for
capacity planning, sizing L1, watching L2 fullness, and spotting
backpressure on individual L2 adapters.

All five metrics are OTel `ObservableGauge` instruments registered via the
shared `register_gauge` helper.  At scrape time, OTel invokes the
registered callback, which iterates the controller's live state and
returns one observation per adapter (for `l2_usage_bytes`, one per
configured adapter; for the in-flight gauges, only adapters with work).
Adapters with no in-flight work emit no datapoint for the three
in-flight gauges.

`lmcache_mp.l2_usage_bytes` carries a single `l2_name` attribute — the
adapter's registered type name (e.g. `"fs"`, `"mock"`,
`"nixl_store"`).  The three in-flight metrics carry two attributes
that disambiguate adapters even when more than one is registered with
the same backend type — same shape as the existing
`lmcache_mp.l2_store_completed` counter:

- `l2_name` — the registered adapter type (e.g. `"fs"`, `"mock"`,
  `"nixl_store"`).
- `adapter_index` — position in the `StoreController`/`PrefetchController`
  adapter list.  Distinguishes two adapters of the same type (e.g.
  `fs[0]` and `fs[1]`).

| OTel metric name | Prometheus name | Type | Source of truth | Calculation |
|---|---|---|---|---|
| `lmcache_mp.l1_memory_usage_bytes` | `lmcache_mp_l1_memory_usage_bytes` | ObservableGauge | `L1Manager.get_memory_usage()` | Bytes currently held in L1 at scrape time |
| `lmcache_mp.l2_usage_bytes` | `lmcache_mp_l2_usage_bytes` | ObservableGauge (attr: `l2_name`) | `StorageManager.get_l2_usages()` (calls `L2AdapterInterface.get_usage().total_bytes_used`) | Per-adapter bytes currently held in L2 at scrape time; one observation per configured adapter.  Adapters whose `get_usage()` raises are skipped silently. |
| `lmcache_mp.num_inflight_l2_stores` | `lmcache_mp_num_inflight_l2_stores` | ObservableGauge (attrs: `l2_name`, `adapter_index`) | `StoreController.get_inflight_count_by_adapter()` | Snapshot of in-flight L2 store tasks grouped by adapter |
| `lmcache_mp.num_inflight_l2_loads` | `lmcache_mp_num_inflight_l2_loads` | ObservableGauge (attrs: `l2_name`, `adapter_index`) | `PrefetchController.get_inflight_load_state_by_adapter()` | Per-adapter count from the same snapshot |
| `lmcache_mp.inflight_load_memory_usage_bytes` | `lmcache_mp_inflight_load_memory_usage_bytes` | ObservableGauge (attrs: `l2_name`, `adapter_index`) | `PrefetchController.get_inflight_load_state_by_adapter()` | Per-adapter reserved bytes from the same snapshot |

**What `l1_memory_usage_bytes` answers:** How full is the L1 cache? Helps
size L1 against working set and detect leaks (steadily climbing without
plateauing).

**What `l2_usage_bytes` answers:** How full is each L2 backend? Lets
operators query how much each L2 tier currently holds, decide whether
an adapter needs eviction or purge, and spot per-backend asymmetries
when more than one L2 is configured.  Parallel to `l1_memory_usage_bytes`
on the L2 tier.

**What `num_inflight_l2_stores` answers:** Are L2 stores piling up on a
particular adapter? Sustained non-zero values indicate the adapter cannot
keep up with the L1 → L2 write rate.

**What `num_inflight_l2_loads` answers:** Are L2 → L1 prefetch loads
backing up? Pair with `num_inflight_l2_stores` to see whether read or
write traffic dominates a given backend.

**What `inflight_load_memory_usage_bytes` answers:** How much L1 capacity
is currently *reserved but not yet filled* by in-flight prefetches? Rising
in-flight bytes alongside rising `l1_memory_usage_bytes` is a signal that
prefetch reservations are crowding out cacheable data.

> **Bytes attribution.** A single prefetch request may load from multiple
> adapters.  The byte count is split per-adapter via the request's
> `load_plan` bitmap × per-key `MemoryObj.size` (precomputed at submit
> time and stored on `InFlightPrefetchRequest.load_bytes_by_adapter`) so
> each in-flight byte is attributed to exactly one `(l2_name,
> adapter_index)` pair — sums across adapters are not double-counted.

> **Singleton dispatch.** L1Manager / StoreController / PrefetchController
> are singletons in MP mode.  Each controller registers its gauge once
> (guarded by a class-level `_gauge_registered` flag) and the callback
> dispatches via a class-level `_gauge_target` so the most recently
> constructed instance owns the reported values.  This is invisible in
> production (one instance per process); it matters in tests that create
> multiple controllers.

> **Thread safety.** Callbacks run on the OTel reader thread and read
> state mutated by the controller's background loop thread.  Snapshots
> use `dict.copy()`, which is implemented in C and atomic under the
> CPython GIL — concurrent mutation cannot crash the snapshot, though it
> may briefly see a state that is one mutation stale.  Acceptable for a
> 10-second scrape cadence.

---

## Cache Blending (CB) Metrics

Metrics for Cache Blending operations use the `lmcache_blend.` prefix (distinct from the
MP mode `lmcache_mp.` namespace).  On Prometheus, `.` becomes `_` and counters get
`_total` suffix (e.g., `lmcache_blend_lookup_requests_total`).

### CB Lookup Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_blend.lookup_requests` | `lmcache_blend_lookup_requests_total` | Counter | `CB_LOOKUP_START` | +1 per event |
| `lmcache_blend.lookup_requested_tokens` | `lmcache_blend_lookup_requested_tokens_total` | Counter | `CB_LOOKUP_END` | `+requested_tokens` |
| `lmcache_blend.lookup_hit_tokens` | `lmcache_blend_lookup_hit_tokens_total` | Counter | `CB_LOOKUP_END` | `+hit_tokens` |
| `lmcache_blend.lookup_fingerprint_hits` | `lmcache_blend_lookup_fingerprint_hits_total` | Counter | `CB_LOOKUP_END` | `+fingerprint_hits` |
| `lmcache_blend.lookup_storage_hits` | `lmcache_blend_lookup_storage_hits_total` | Counter | `CB_LOOKUP_END` | `+storage_hits` |
| `lmcache_blend.lookup_stale_chunks` | `lmcache_blend_lookup_stale_chunks_total` | Counter | `CB_LOOKUP_END` | `+stale_chunks` |
| `lmcache_blend.lookup_no_gpu_context_errors` | `lmcache_blend_lookup_no_gpu_context_errors_total` | Counter | `CB_LOOKUP_END` | +1 when `no_gpu_context=True` |

**What it answers:** How often does the CB server receive lookup requests? What fraction of requested tokens are served by blend (token-level hit rate)? What fraction hit the fingerprint table? What fraction are confirmed in storage? How many stale evictions occur?

**Blend token-level hit rate** (numerator and denominator co-emit on the same `CB_LOOKUP_END` event so the ratio is meaningful even under partial-failure paths):

```
rate(lmcache_blend_lookup_hit_tokens_total[5m])
/ rate(lmcache_blend_lookup_requested_tokens_total[5m])
```

### CB Retrieve Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_blend.retrieve_requests` | `lmcache_blend_retrieve_requests_total` | Counter | `CB_RETRIEVE_START` | +1 per event |
| `lmcache_blend.retrieve_chunks` | `lmcache_blend_retrieve_chunks_total` | Counter | `CB_RETRIEVE_START` | `+num_chunks` |
| `lmcache_blend.retrieve_failures` | `lmcache_blend_retrieve_failures_total` | Counter | `CB_RETRIEVE_END` | +1 when `success=False` |

**What it answers:** How often is CB retrieval invoked? How many chunks are retrieved per call? What is the failure rate?

### CB Store Pre-computed Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_blend.store_pre_computed_requests` | `lmcache_blend_store_pre_computed_requests_total` | Counter | `CB_STORE_PRE_COMPUTED_START` | +1 per event |
| `lmcache_blend.store_pre_computed_chunks` | `lmcache_blend_store_pre_computed_chunks_total` | Counter | `CB_STORE_PRE_COMPUTED_END` | `+stored_chunks` |
| `lmcache_blend.store_pre_computed_failures` | `lmcache_blend_store_pre_computed_failures_total` | Counter | `CB_STORE_PRE_COMPUTED_END` | +1 when `success=False` |

**What it answers:** How often is pre-computed CB storage invoked? How many chunks are written? What is the failure rate?

### CB Store Final Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_blend.store_final_requests` | `lmcache_blend_store_final_requests_total` | Counter | `CB_STORE_FINAL_START` | +1 per event |
| `lmcache_blend.store_final_chunks` | `lmcache_blend_store_final_chunks_total` | Counter | `CB_STORE_FINAL_END` | `+stored_chunks` |
| `lmcache_blend.store_final_failures` | `lmcache_blend_store_final_failures_total` | Counter | `CB_STORE_FINAL_END` | +1 when `success=False` |

**What it answers:** How often is final CB storage invoked? How many chunks are committed? What is the failure rate?

### CB Fingerprint Table Metrics

| OTel metric name | Prometheus name | Type | Source event | Calculation |
|---|---|---|---|---|
| `lmcache_blend.fingerprints_registered` | `lmcache_blend_fingerprints_registered_total` | Counter | `CB_FINGERPRINTS_REGISTERED` | `+num_chunks` |
| `lmcache_blend.chunks_evicted` | `lmcache_blend_chunks_evicted_total` | Counter | `CB_CHUNKS_EVICTED` | `+num_chunks` |

**What it answers:** How many chunks are indexed into the fingerprint table? How many stale entries are evicted?

---

For derivations of L1-only / L2-only / blend hit rates from these
counters, and a "what to send when reporting" checklist, see
[DEBUG.md](DEBUG.md).

For event metadata contracts (what keys each `EventType` carries), see
[EVENTS.md](EVENTS.md).
