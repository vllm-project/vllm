// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Scheduler-scoped metrics, exported from the engine's `SchedulerStats`.
//!
//! Each metric group keeps its part of the `SchedulerStats` payload, its
//! Prometheus families, its per-engine handles and its recording together.
//! KV connector telemetry has one module per connector in `kv_connector`.
//! The `SchedulerStats` envelope stays with the engine protocol, which also
//! routes on it; the engine client hands each group its part.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use itertools::Itertools as _;
use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue, LabelValueEncoder};
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};

use crate::buckets::{BucketConfig, BucketFamily, FamilyHistogram};
use crate::{BucketHistogramFamily, F64Gauge, HistogramMetric, U64Counter, U64Gauge};

mod kv_connector;

pub use kv_connector::KvConnectorHandles;
use kv_connector::KvConnectorMetrics;
pub use kv_connector::{
    KvConnectorStats, MooncakeRecord, MooncakeStats, MooncakeTransferStats, MultiConnectorStats,
    NixlStats, SimpleCpuOffloadData, SimpleCpuOffloadStats, UmbpOperationCounts, UmbpStats,
};

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineLabels {
    pub model_name: String,
    pub engine: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct CacheHitSourceLabels {
    pub model_name: String,
    pub engine: u32,
    pub source: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EnginePositionLabels {
    pub model_name: String,
    pub engine: u32,
    pub position: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct WaitingReasonLabels {
    pub model_name: String,
    pub engine: u32,
    pub reason: &'static str,
}

/// Labels for `vllm:engine_sleep_state`: one series per engine and state,
/// `awake`, `weights_offloaded` (sleep level 1) or `discard_all` (level 2).
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineSleepStateLabels {
    pub model_name: String,
    pub engine: u32,
    pub sleep_state: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct KvFetchStageLabels {
    pub model_name: String,
    pub engine: u32,
    pub stage: &'static str,
}

/// Adapter names encoded as a deterministic comma-joined Prometheus label value.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct LoraAdapterNames(pub BTreeSet<String>);

impl EncodeLabelValue for LoraAdapterNames {
    fn encode(&self, encoder: &mut LabelValueEncoder) -> Result<(), std::fmt::Error> {
        EncodeLabelValue::encode(&self.0.iter().join(","), encoder)
    }
}

/// Labels for `vllm:lora_requests_info`. Field order is the exposition order,
/// which matches Python's (sorted by name).
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct LoraInfoLabels {
    /// The engine's `max_loras`, constant for the server's lifetime.
    pub max_lora: u32,
    pub running_lora_adapters: LoraAdapterNames,
    pub waiting_lora_adapters: LoraAdapterNames,
}

/// CUDA graph sample key used for periodic text-log aggregation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CudagraphLogKey {
    pub num_unpadded_tokens: u64,
    pub num_padded_tokens: u64,
    pub num_paddings: u64,
    pub runtime_mode: String,
}

/// Raw scheduler stats accumulated for one periodic text-log interval.
#[derive(Default)]
pub struct SchedulerLogStatsInterval {
    pub spec_num_drafts: u64,
    pub spec_accepted_tokens_per_pos: Vec<u64>,
    pub cudagraph_counts: BTreeMap<CudagraphLogKey, u64>,
}

impl SchedulerLogStatsInterval {
    /// Merge another drained interval into this one.
    pub fn merge(&mut self, other: Self) {
        self.spec_num_drafts += other.spec_num_drafts;

        if self.spec_accepted_tokens_per_pos.len() < other.spec_accepted_tokens_per_pos.len() {
            self.spec_accepted_tokens_per_pos
                .resize(other.spec_accepted_tokens_per_pos.len(), 0);
        }
        for (position, accepted_tokens) in
            other.spec_accepted_tokens_per_pos.into_iter().enumerate()
        {
            self.spec_accepted_tokens_per_pos[position] += accepted_tokens;
        }

        for (key, count) in other.cudagraph_counts {
            *self.cudagraph_counts.entry(key).or_default() += count;
        }
    }
}

/// Internal, non-Prometheus accumulator for periodic text logs that need raw
/// scheduler DTOs.
#[derive(Clone, Default)]
pub struct SchedulerLogStatsAccumulator {
    inner: Arc<Mutex<SchedulerLogStatsInterval>>,
}

impl SchedulerLogStatsAccumulator {
    /// Observe spec-decoding fields needed for per-position text-log rates.
    pub fn observe_spec_decode(&self, num_drafts: u64, accepted_tokens_per_pos: &[u64]) {
        let mut inner = self.inner.lock().expect("scheduler log stats accumulator poisoned");
        inner.spec_num_drafts += num_drafts;

        if inner.spec_accepted_tokens_per_pos.len() < accepted_tokens_per_pos.len() {
            inner.spec_accepted_tokens_per_pos.resize(accepted_tokens_per_pos.len(), 0);
        }
        for (position, accepted_tokens) in accepted_tokens_per_pos.iter().copied().enumerate() {
            inner.spec_accepted_tokens_per_pos[position] += accepted_tokens;
        }
    }

    /// Observe one CUDA graph runtime sample for the interval table.
    pub fn observe_cudagraph(
        &self,
        num_unpadded_tokens: u64,
        num_padded_tokens: u64,
        num_paddings: u64,
        runtime_mode: &str,
    ) {
        let mut inner = self.inner.lock().expect("scheduler log stats accumulator poisoned");
        let key = CudagraphLogKey {
            num_unpadded_tokens,
            num_padded_tokens,
            num_paddings,
            runtime_mode: runtime_mode.to_string(),
        };
        *inner.cudagraph_counts.entry(key).or_default() += 1;
    }

    /// Drain and reset the current text-log interval.
    pub fn drain(&self) -> SchedulerLogStatsInterval {
        let mut inner = self.inner.lock().expect("scheduler log stats accumulator poisoned");
        std::mem::take(&mut *inner)
    }
}

/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/compilation/cuda_graph.py#L28-L33>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CudagraphStats {
    /// Number of real tokens in the captured batch before padding.
    pub num_unpadded_tokens: u64,
    /// Number of padded tokens in the captured batch.
    pub num_padded_tokens: u64,
    /// Number of padding positions added for capture/runtime shape alignment.
    pub num_paddings: u64,
    /// Runtime mode string associated with this CUDA graph sample.
    pub runtime_mode: String,
}

/// Scheduler/batch-scoped Prometheus families exported from `SchedulerStats`.
pub struct SchedulerMetrics {
    pub state: SchedulerStateMetrics,

    /// `vllm:engine_sleep_state`, set by the engine client's sleep and wake
    /// calls rather than from `SchedulerStats`.
    pub engine_sleep_state: Family<EngineSleepStateLabels, U64Gauge>,

    /// `vllm:lora_requests_info`. Value is the emit-time unix timestamp in
    /// seconds.
    pub lora_info: Family<LoraInfoLabels, F64Gauge>,

    /// `vllm:cache_config_info`, one series per engine, always `1`. Labeled
    /// with every `CacheConfig.metrics_info()` field plus `engine`.
    pub cache_config_info: Family<Vec<(String, String)>, F64Gauge>,

    pub prefix_cache: PrefixCacheMetrics,
    pub spec_decode: SpecDecodeMetrics,
    pub perf: PerfMetrics,
    pub kv_cache_residency: KvCacheResidencyMetrics,
    kv_connector: KvConnectorMetrics,

    /// Non-Prometheus interval accumulators for periodic text-log helpers.
    pub log_stats: Family<EngineLabels, SchedulerLogStatsAccumulator>,
}

impl SchedulerMetrics {
    /// Register the scheduler-oriented metric families into the shared
    /// registry.
    pub(crate) fn register(registry: &mut Registry, buckets: &Arc<BucketConfig>) -> Self {
        let state = SchedulerStateMetrics::register(registry);

        let engine_sleep_state = Family::default();
        registry.register(
            "vllm:engine_sleep_state",
            "Engine sleep state; awake = 0 means engine is sleeping; \
             awake = 1 means engine is awake; \
             weights_offloaded = 1 means sleep level 1; \
             discard_all = 1 means sleep level 2.",
            engine_sleep_state.clone(),
        );

        let lora_info = Family::default();
        registry.register(
            "vllm:lora_requests_info",
            "Running stats on lora requests.",
            lora_info.clone(),
        );

        let cache_config_info = Family::default();
        registry.register(
            "vllm:cache_config_info",
            "Information of the LLMEngine CacheConfig",
            cache_config_info.clone(),
        );

        Self {
            state,
            engine_sleep_state,
            lora_info,
            cache_config_info,
            prefix_cache: PrefixCacheMetrics::register(registry),
            spec_decode: SpecDecodeMetrics::register(registry),
            perf: PerfMetrics::register(registry),
            kv_cache_residency: KvCacheResidencyMetrics::register(registry, buckets),
            kv_connector: KvConnectorMetrics::register(registry),
            log_stats: Family::default(),
        }
    }

    /// Resolve all fixed-label scheduler metrics for one engine.
    pub fn resolve_handles(&self, model_name: &str, engine: u32) -> SchedulerStatsHandles {
        let labels = EngineLabels {
            model_name: model_name.to_string(),
            engine,
        };
        SchedulerStatsHandles {
            state: self.state.resolve(&labels),
            engine_sleep_state: EngineSleepStateHandles(SLEEP_STATES.map(|sleep_state| {
                self.engine_sleep_state.get_or_create_owned(&EngineSleepStateLabels {
                    model_name: model_name.to_string(),
                    engine,
                    sleep_state,
                })
            })),
            prefix_cache: self.prefix_cache.resolve(&labels),
            spec_decode: self.spec_decode.resolve(&labels),
            log_stats: self.log_stats.get_or_create_owned(&labels),
            perf: self.perf.resolve(&labels),
            kv_cache_residency: self.kv_cache_residency.resolve(&labels),
            kv_connector: KvConnectorHandles {
                metrics: self.kv_connector.clone(),
                labels,
            },
        }
    }

    /// Export one engine's cache config as `vllm:cache_config_info`, which
    /// Python also creates with its stat logger.
    pub fn record_cache_config_info(&self, engine: u32, info: &BTreeMap<String, String>) {
        let mut labels = info.clone();
        labels.insert("engine".to_string(), engine.to_string());
        self.cache_config_info.get_or_create(&labels.into_iter().collect()).set(1.0);
    }
}

/// Per-engine cached metric handles used while recording `SchedulerStats`,
/// one field per metric group.
pub struct SchedulerStatsHandles {
    pub state: SchedulerStateHandles,
    pub engine_sleep_state: EngineSleepStateHandles,
    pub prefix_cache: PrefixCacheHandles,
    pub spec_decode: SpecDecodeHandles,
    pub perf: PerfHandles,
    pub kv_cache_residency: KvCacheResidencyHandles,
    pub kv_connector: KvConnectorHandles,

    // Non-Prometheus interval accumulator for periodic text-log helpers.
    pub log_stats: SchedulerLogStatsAccumulator,
}

// Scheduler state gauges.

const WAITING_REASON_CAPACITY: &str = "capacity";
const WAITING_REASON_DEFERRED: &str = "deferred";

/// Waiting requests with an async KV load, per load stage. Python sends a
/// `dict[str, int]` keyed by stage, so absent stages default to zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct KvFetchReqsByStage {
    /// Needs an async KV load that has not started.
    pub waiting_to_start: u64,
    /// Load started and not yet reported finished.
    pub in_progress: u64,
    /// Load finished, request not running yet.
    pub completed_waiting: u64,
}

impl KvFetchReqsByStage {
    /// Label values, in the same order as [`Self::counts`].
    pub const STAGES: [&'static str; 3] = ["waiting_to_start", "in_progress", "completed_waiting"];

    pub fn counts(&self) -> [u64; 3] {
        [
            self.waiting_to_start,
            self.in_progress,
            self.completed_waiting,
        ]
    }
}

/// Scheduler state gauges.
pub struct SchedulerStateMetrics {
    pub running: Family<EngineLabels, U64Gauge>,
    pub waiting: Family<EngineLabels, U64Gauge>,
    pub waiting_by_reason: Family<WaitingReasonLabels, U64Gauge>,
    pub kv_fetch_by_stage: Family<KvFetchStageLabels, U64Gauge>,
    pub kv_cache_usage: Family<EngineLabels, F64Gauge>,
}

impl SchedulerStateMetrics {
    fn register(registry: &mut Registry) -> Self {
        let running = Family::default();
        registry.register(
            "vllm:num_requests_running",
            "Number of requests in model execution batches",
            running.clone(),
        );

        let waiting = Family::default();
        registry.register(
            "vllm:num_requests_waiting",
            "Number of requests waiting to be processed",
            waiting.clone(),
        );

        let waiting_by_reason = Family::default();
        registry.register(
            "vllm:num_requests_waiting_by_reason",
            "Number of waiting requests by reason. \
             Reason labels: 'capacity' = waiting for scheduling capacity; \
             'deferred' = deferred by transient constraints (LoRA budget, KV transfer, \
             blocked status). Sum of all reasons equals vllm:num_requests_waiting.",
            waiting_by_reason.clone(),
        );

        let kv_fetch_by_stage = Family::default();
        registry.register(
            "vllm:num_requests_kv_fetch_by_stage",
            "Number of waiting requests by async KV load stage. \
             Stage labels: 'waiting_to_start' = needs an async KV load that has not started; \
             'in_progress' = load started and not yet reported finished; \
             'completed_waiting' = load finished, request not running yet.",
            kv_fetch_by_stage.clone(),
        );

        let kv_cache_usage = Family::default();
        registry.register(
            "vllm:kv_cache_usage_perc",
            "KV-cache usage. 1 means 100 percent usage",
            kv_cache_usage.clone(),
        );

        Self {
            running,
            waiting,
            waiting_by_reason,
            kv_fetch_by_stage,
            kv_cache_usage,
        }
    }

    fn resolve(&self, labels: &EngineLabels) -> SchedulerStateHandles {
        let reason = |reason| WaitingReasonLabels {
            model_name: labels.model_name.clone(),
            engine: labels.engine,
            reason,
        };
        SchedulerStateHandles {
            running: self.running.get_or_create_owned(labels),
            waiting: self.waiting.get_or_create_owned(labels),
            waiting_capacity: self
                .waiting_by_reason
                .get_or_create_owned(&reason(WAITING_REASON_CAPACITY)),
            waiting_deferred: self
                .waiting_by_reason
                .get_or_create_owned(&reason(WAITING_REASON_DEFERRED)),
            kv_fetch_by_stage: KvFetchReqsByStage::STAGES.map(|stage| {
                self.kv_fetch_by_stage.get_or_create_owned(&KvFetchStageLabels {
                    model_name: labels.model_name.clone(),
                    engine: labels.engine,
                    stage,
                })
            }),
            kv_cache_usage: self.kv_cache_usage.get_or_create_owned(labels),
        }
    }
}

pub struct SchedulerStateHandles {
    running: U64Gauge,
    waiting: U64Gauge,
    waiting_capacity: U64Gauge,
    waiting_deferred: U64Gauge,
    /// One handle per `KvFetchReqsByStage::STAGES` entry.
    kv_fetch_by_stage: [U64Gauge; 3],
    kv_cache_usage: F64Gauge,
}

impl SchedulerStateHandles {
    pub fn record(
        &self,
        num_running_reqs: u64,
        num_waiting_reqs: u64,
        num_skipped_waiting_reqs: u64,
        num_kv_fetch_reqs_by_stage: &KvFetchReqsByStage,
        kv_cache_usage: f64,
    ) {
        self.running.set(num_running_reqs);
        self.waiting.set(num_waiting_reqs + num_skipped_waiting_reqs);
        self.waiting_capacity.set(num_waiting_reqs);
        self.waiting_deferred.set(num_skipped_waiting_reqs);
        for (handle, count) in
            self.kv_fetch_by_stage.iter().zip(num_kv_fetch_reqs_by_stage.counts())
        {
            handle.set(count);
        }
        self.kv_cache_usage.set(kv_cache_usage);
    }
}

// Engine sleep state.

/// `sleep_state` label values of `vllm:engine_sleep_state`, in the order of
/// the values [`EngineSleepStateHandles::record`] takes.
pub const SLEEP_STATES: [&str; 3] = ["awake", "weights_offloaded", "discard_all"];

/// One handle per [`SLEEP_STATES`] entry.
pub struct EngineSleepStateHandles([U64Gauge; 3]);

impl EngineSleepStateHandles {
    /// Record the engine's sleep state, one gauge value per [`SLEEP_STATES`]
    /// entry.
    pub fn record(&self, values: [u64; 3]) {
        for (gauge, value) in self.0.iter().zip(values) {
            gauge.set(value);
        }
    }
}

// LoRA info.

/// Exports `vllm:lora_requests_info` as a single series covering all LoRA
/// requests tracked by this client across every engine in the replica.
///
/// The engine's `SchedulerStats` never carries adapter names: the Python
/// frontend fills them in from per-request lifecycle events tracked by
/// `LoRARequestStates` in `vllm/v1/engine/output_processor.py`. The Rust
/// frontend mirrors that, deriving the sets from the request registry.
pub struct LoraInfoExporter {
    /// The engines' `max_loras`, reported as the `max_lora` label.
    max_lora: u32,
    current: Option<LoraInfoLabels>,
}

impl LoraInfoExporter {
    pub fn new(max_lora: u32) -> Self {
        Self {
            max_lora,
            current: None,
        }
    }

    pub fn update(
        &mut self,
        metrics: &SchedulerMetrics,
        running: BTreeSet<String>,
        waiting: BTreeSet<String>,
    ) {
        let next = (!running.is_empty() || !waiting.is_empty()).then_some(LoraInfoLabels {
            max_lora: self.max_lora,
            running_lora_adapters: LoraAdapterNames(running),
            waiting_lora_adapters: LoraAdapterNames(waiting),
        });

        if self.current != next
            && let Some(prev) = &self.current
        {
            metrics.lora_info.remove(prev);
        }

        // Python sets this gauge to the current time on every record.
        if let Some(labels) = &next {
            metrics.lora_info.get_or_create(labels).set(now_unix_secs());
        }

        self.current = next;
    }
}

fn now_unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// Prefix-cache counters, including the connector-backed external cache path.

/// Stores cache hit statistics.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L18-L32>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BaseCacheStats {
    /// Whether the cache was reset.
    pub reset: bool,
    /// The number of requests in this update.
    pub requests: u64,
    /// The number of queries in these requests.
    pub queries: u64,
    /// The number of hits in these requests.
    pub hits: u64,
}

/// Cache-hit tokens per source tier. Python sends a sparse
/// `dict[CacheHitSource, int]`, so absent tiers default to zero, and only
/// non-zero tiers are serialized.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CachedTokensBySource {
    #[serde(skip_serializing_if = "is_zero")]
    pub device: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub host: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub p2p: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub disk: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub external_unspecified: u64,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

impl CachedTokensBySource {
    /// Label values, in the same order as [`Self::counts`].
    pub const SOURCES: [&'static str; 5] =
        ["device", "host", "p2p", "disk", "external_unspecified"];

    pub fn counts(&self) -> [u64; 5] {
        [
            self.device,
            self.host,
            self.p2p,
            self.disk,
            self.external_unspecified,
        ]
    }
}

/// Stores prefix cache hit statistics.
/// - `reset`: Whether `reset_prefix_cache` was invoked.
/// - `queries`: Refers to the number of tokens that were queried.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L114-L143>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PrefixCacheStats {
    /// Embedded base cache counters and reset flag.
    #[serde(flatten)]
    pub base: BaseCacheStats,
    /// The number of previously preempted requests in this update.
    pub preempted_requests: u64,
    /// The `queries` number for preempted requests.
    pub preempted_queries: u64,
    /// The `hits` number for preempted requests.
    pub preempted_hits: u64,
    /// `hits` split by the cache tier that supplied them (connector stats only).
    #[serde(default)]
    pub hits_by_source: CachedTokensBySource,
}

/// Prefix-cache counters, including the connector-backed external cache path.
pub struct PrefixCacheMetrics {
    pub queries: Family<EngineLabels, U64Counter>,
    pub hits: Family<EngineLabels, U64Counter>,
    pub external_queries: Family<EngineLabels, U64Counter>,
    pub external_hits: Family<EngineLabels, U64Counter>,
    pub prompt_tokens_cached_by_source: Family<CacheHitSourceLabels, U64Counter>,
}

impl PrefixCacheMetrics {
    fn register(registry: &mut Registry) -> Self {
        let queries = Family::default();
        registry.register(
            "vllm:prefix_cache_queries",
            "Prefix cache queries, in terms of number of queried tokens",
            queries.clone(),
        );

        let hits = Family::default();
        registry.register(
            "vllm:prefix_cache_hits",
            "Prefix cache hits, in terms of number of cached tokens.",
            hits.clone(),
        );

        let external_queries = Family::default();
        registry.register(
            "vllm:external_prefix_cache_queries",
            "External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens.",
            external_queries.clone(),
        );

        let external_hits = Family::default();

        let prompt_tokens_cached_by_source = Family::default();
        registry.register(
            "vllm:prompt_tokens_cached_by_source",
            "Prefix-cache hit tokens by the cache tier that supplied them: device (local HBM), host (offloaded to DRAM), disk, p2p (transferred from another vLLM instance) or external_unspecified. Counted at admission; sums to vllm:prefix_cache_hits + vllm:external_prefix_cache_hits.",
            prompt_tokens_cached_by_source.clone(),
        );
        registry.register(
            "vllm:external_prefix_cache_hits",
            "External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens.",
            external_hits.clone(),
        );

        Self {
            queries,
            hits,
            external_queries,
            external_hits,
            prompt_tokens_cached_by_source,
        }
    }

    fn resolve(&self, labels: &EngineLabels) -> PrefixCacheHandles {
        PrefixCacheHandles {
            queries: self.queries.get_or_create_owned(labels),
            hits: self.hits.get_or_create_owned(labels),
            external_queries: self.external_queries.get_or_create_owned(labels),
            external_hits: self.external_hits.get_or_create_owned(labels),
            prompt_tokens_cached_by_source: CachedTokensBySource::SOURCES.map(|source| {
                self.prompt_tokens_cached_by_source.get_or_create_owned(&CacheHitSourceLabels {
                    model_name: labels.model_name.clone(),
                    engine: labels.engine,
                    source,
                })
            }),
        }
    }
}

pub struct PrefixCacheHandles {
    queries: U64Counter,
    hits: U64Counter,
    external_queries: U64Counter,
    external_hits: U64Counter,
    /// One handle per `CachedTokensBySource::SOURCES` entry.
    prompt_tokens_cached_by_source: [U64Counter; 5],
}

impl PrefixCacheHandles {
    pub fn record(&self, local: &PrefixCacheStats, connector: Option<&PrefixCacheStats>) {
        self.queries.inc_by(local.base.queries);
        self.hits.inc_by(local.base.hits);
        // Local prefix-cache hits are the `device` tier.
        self.prompt_tokens_cached_by_source[0].inc_by(local.base.hits);

        if let Some(connector) = connector {
            self.external_queries.inc_by(connector.base.queries);
            self.external_hits.inc_by(connector.base.hits);
            let counts = connector.hits_by_source.counts();
            for (handle, count) in self.prompt_tokens_cached_by_source.iter().zip(counts) {
                handle.inc_by(count);
            }
        }
    }
}

// Speculative decoding counters.

/// Per-step iteration decoding stats from scheduler.
///
/// Each scheduler step, statistics on spec decoding performance are aggregated
/// across requests by the scheduler and returned to the frontend in
/// `EngineCoreOutputs -> SchedulerStats`.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/spec_decode/metrics.py#L16-L44>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecDecodingStats {
    /// Configured speculative token count for this scheduler.
    pub num_spec_tokens: u64,
    /// Number of drafted speculative decoding attempts.
    pub num_drafts: u64,
    /// Number of drafted tokens.
    pub num_draft_tokens: u64,
    /// Number of accepted drafted tokens.
    pub num_accepted_tokens: u64,
    /// Accepted drafted tokens counted by draft position.
    pub num_accepted_tokens_per_pos: Vec<u64>,
    /// Drafted tokens counted by draft position.
    #[serde(default)]
    pub num_draft_tokens_per_pos: Vec<u64>,
    /// Number of draft tokens the target model ran a forward pass on.
    pub num_verified_tokens: u64,
    /// Verified draft tokens counted by draft position.
    pub num_verified_tokens_per_pos: Vec<u64>,
}

/// Speculative decoding counters.
pub struct SpecDecodeMetrics {
    pub num_drafts: Family<EngineLabels, U64Counter>,
    pub num_draft_tokens: Family<EngineLabels, U64Counter>,
    pub num_accepted_tokens: Family<EngineLabels, U64Counter>,
    pub num_accepted_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,
    pub num_verified_tokens: Family<EngineLabels, U64Counter>,
    pub num_verified_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,
}

impl SpecDecodeMetrics {
    fn register(registry: &mut Registry) -> Self {
        let num_drafts = Family::default();
        registry.register(
            "vllm:spec_decode_num_drafts",
            "Number of spec decoding drafts.",
            num_drafts.clone(),
        );

        let num_draft_tokens = Family::default();
        registry.register(
            "vllm:spec_decode_num_draft_tokens",
            "Number of draft tokens.",
            num_draft_tokens.clone(),
        );

        let num_accepted_tokens = Family::default();
        registry.register(
            "vllm:spec_decode_num_accepted_tokens",
            "Number of accepted tokens.",
            num_accepted_tokens.clone(),
        );

        let num_accepted_tokens_per_pos = Family::default();
        registry.register(
            "vllm:spec_decode_num_accepted_tokens_per_pos",
            "Accepted tokens per draft position.",
            num_accepted_tokens_per_pos.clone(),
        );

        let num_verified_tokens = Family::default();
        registry.register(
            "vllm:spec_decode_num_verified_tokens",
            "Number of draft tokens verified by the target.",
            num_verified_tokens.clone(),
        );

        let num_verified_tokens_per_pos = Family::default();
        registry.register(
            "vllm:spec_decode_num_verified_tokens_per_pos",
            "Verified draft tokens per draft position.",
            num_verified_tokens_per_pos.clone(),
        );

        Self {
            num_drafts,
            num_draft_tokens,
            num_accepted_tokens,
            num_accepted_tokens_per_pos,
            num_verified_tokens,
            num_verified_tokens_per_pos,
        }
    }

    fn resolve(&self, labels: &EngineLabels) -> SpecDecodeHandles {
        SpecDecodeHandles {
            labels: labels.clone(),
            num_drafts: self.num_drafts.get_or_create_owned(labels),
            num_draft_tokens: self.num_draft_tokens.get_or_create_owned(labels),
            num_accepted_tokens: self.num_accepted_tokens.get_or_create_owned(labels),
            num_accepted_tokens_per_pos: self.num_accepted_tokens_per_pos.clone(),
            num_verified_tokens: self.num_verified_tokens.get_or_create_owned(labels),
            num_verified_tokens_per_pos: self.num_verified_tokens_per_pos.clone(),
        }
    }
}

pub struct SpecDecodeHandles {
    // Base labels of the per-position series.
    labels: EngineLabels,
    num_drafts: U64Counter,
    num_draft_tokens: U64Counter,
    num_accepted_tokens: U64Counter,
    num_accepted_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,
    num_verified_tokens: U64Counter,
    num_verified_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,
}

impl SpecDecodeHandles {
    pub fn record(&self, stats: &SpecDecodingStats) {
        let labels = &self.labels;
        self.num_drafts.inc_by(stats.num_drafts);
        self.num_draft_tokens.inc_by(stats.num_draft_tokens);
        self.num_accepted_tokens.inc_by(stats.num_accepted_tokens);
        self.num_verified_tokens.inc_by(stats.num_verified_tokens);

        for (position, accepted_tokens) in
            stats.num_accepted_tokens_per_pos.iter().copied().enumerate()
        {
            self.num_accepted_tokens_per_pos
                .get_or_create(&EnginePositionLabels {
                    model_name: labels.model_name.clone(),
                    engine: labels.engine,
                    position: position as u32,
                })
                .inc_by(accepted_tokens);
        }
        for (position, verified_tokens) in
            stats.num_verified_tokens_per_pos.iter().copied().enumerate()
        {
            self.num_verified_tokens_per_pos
                .get_or_create(&EnginePositionLabels {
                    model_name: labels.model_name.clone(),
                    engine: labels.engine,
                    position: position as u32,
                })
                .inc_by(verified_tokens);
        }
    }
}

// Per-engine performance / MFU counters.

/// Stats for debugging the metrics calculation.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/perf.py#L46-L55>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DebugPerfStats {
    /// Time spent calculating these stats.
    pub calc_duration: f64,
    /// Number of prefill requests included in the sampled batch.
    pub num_prefill_requests: u64,
    /// Number of decode requests included in the sampled batch.
    pub num_decode_requests: u64,
    /// Optional execution-context breakdown used for debugging.
    pub context_breakdown: Option<BTreeMap<String, u64>>,
    /// Optional per-component FLOPs breakdown.
    pub num_flops_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
    /// Optional per-component memory-read breakdown.
    pub num_read_bytes_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
    /// Optional per-component memory-write breakdown.
    pub num_write_bytes_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
}

/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/perf.py#L58-L63>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerfStats {
    /// Estimated floating point operations per GPU.
    pub num_flops_per_gpu: u64,
    /// Estimated bytes read from memory per GPU.
    pub num_read_bytes_per_gpu: u64,
    /// Estimated bytes written to memory per GPU.
    pub num_write_bytes_per_gpu: u64,
    /// Optional debug-only perf derivation details.
    pub debug_stats: Option<DebugPerfStats>,
}

/// Per-engine performance / MFU counters.
pub struct PerfMetrics {
    pub estimated_flops_per_gpu: Family<EngineLabels, U64Counter>,
    pub estimated_read_bytes_per_gpu: Family<EngineLabels, U64Counter>,
    pub estimated_write_bytes_per_gpu: Family<EngineLabels, U64Counter>,
}

impl PerfMetrics {
    fn register(registry: &mut Registry) -> Self {
        let estimated_flops_per_gpu = Family::default();
        registry.register(
            "vllm:estimated_flops_per_gpu",
            "Estimated number of floating point operations per GPU (for Model Flops Utilization calculations).",
            estimated_flops_per_gpu.clone(),
        );

        let estimated_read_bytes_per_gpu = Family::default();
        registry.register(
            "vllm:estimated_read_bytes_per_gpu",
            "Estimated number of bytes read from memory per GPU (for Model Flops Utilization calculations).",
            estimated_read_bytes_per_gpu.clone(),
        );

        let estimated_write_bytes_per_gpu = Family::default();
        registry.register(
            "vllm:estimated_write_bytes_per_gpu",
            "Estimated number of bytes written to memory per GPU (for Model Flops Utilization calculations).",
            estimated_write_bytes_per_gpu.clone(),
        );

        Self {
            estimated_flops_per_gpu,
            estimated_read_bytes_per_gpu,
            estimated_write_bytes_per_gpu,
        }
    }

    fn resolve(&self, labels: &EngineLabels) -> PerfHandles {
        PerfHandles {
            estimated_flops_per_gpu: self.estimated_flops_per_gpu.get_or_create_owned(labels),
            estimated_read_bytes_per_gpu: self
                .estimated_read_bytes_per_gpu
                .get_or_create_owned(labels),
            estimated_write_bytes_per_gpu: self
                .estimated_write_bytes_per_gpu
                .get_or_create_owned(labels),
        }
    }
}

pub struct PerfHandles {
    estimated_flops_per_gpu: U64Counter,
    estimated_read_bytes_per_gpu: U64Counter,
    estimated_write_bytes_per_gpu: U64Counter,
}

impl PerfHandles {
    pub fn record(&self, stats: &PerfStats) {
        if stats.num_flops_per_gpu != 0
            || stats.num_read_bytes_per_gpu != 0
            || stats.num_write_bytes_per_gpu != 0
        {
            self.estimated_flops_per_gpu.inc_by(stats.num_flops_per_gpu);
            self.estimated_read_bytes_per_gpu.inc_by(stats.num_read_bytes_per_gpu);
            self.estimated_write_bytes_per_gpu.inc_by(stats.num_write_bytes_per_gpu);
        }
    }
}

// Sampled KV-cache residency histograms.

/// Single KV cache block eviction sample.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L161-L167>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KvCacheEvictionEvent {
    /// Lifetime from allocation to eviction.
    pub lifetime_seconds: f64,
    /// Idle time observed before eviction.
    pub idle_seconds: f64,
    /// Time gaps between consecutive accesses before eviction.
    pub reuse_gaps_seconds: Vec<f64>,
}

/// Sampled KV-cache residency histograms.
pub struct KvCacheResidencyMetrics {
    pub kv_block_lifetime_seconds: BucketHistogramFamily,
    pub kv_block_idle_before_evict_seconds: BucketHistogramFamily,
    pub kv_block_reuse_gap_seconds: BucketHistogramFamily,
}

impl KvCacheResidencyMetrics {
    fn register(registry: &mut Registry, buckets: &Arc<BucketConfig>) -> Self {
        let kv_block_lifetime_seconds = Family::new_with_constructor(FamilyHistogram::new(
            BucketFamily::KvCacheResidency,
            buckets,
        ));
        registry.register(
            "vllm:kv_block_lifetime_seconds",
            "Histogram of KV cache block lifetime from allocation to eviction. Sampled metrics (controlled by --kv-cache-metrics-sample).",
            kv_block_lifetime_seconds.clone(),
        );

        let kv_block_idle_before_evict_seconds = Family::new_with_constructor(
            FamilyHistogram::new(BucketFamily::KvCacheResidency, buckets),
        );
        registry.register(
            "vllm:kv_block_idle_before_evict_seconds",
            "Histogram of idle time before KV cache block eviction. Sampled metrics (controlled by --kv-cache-metrics-sample).",
            kv_block_idle_before_evict_seconds.clone(),
        );

        let kv_block_reuse_gap_seconds = Family::new_with_constructor(FamilyHistogram::new(
            BucketFamily::KvCacheResidency,
            buckets,
        ));
        registry.register(
            "vllm:kv_block_reuse_gap_seconds",
            "Histogram of time gaps between consecutive KV cache block accesses. Only the most recent accesses are recorded (ring buffer). Sampled metrics (controlled by --kv-cache-metrics-sample).",
            kv_block_reuse_gap_seconds.clone(),
        );

        Self {
            kv_block_lifetime_seconds,
            kv_block_idle_before_evict_seconds,
            kv_block_reuse_gap_seconds,
        }
    }

    fn resolve(&self, labels: &EngineLabels) -> KvCacheResidencyHandles {
        KvCacheResidencyHandles {
            kv_block_lifetime_seconds: self.kv_block_lifetime_seconds.get_or_create_owned(labels),
            kv_block_idle_before_evict_seconds: self
                .kv_block_idle_before_evict_seconds
                .get_or_create_owned(labels),
            kv_block_reuse_gap_seconds: self.kv_block_reuse_gap_seconds.get_or_create_owned(labels),
        }
    }
}

pub struct KvCacheResidencyHandles {
    kv_block_lifetime_seconds: HistogramMetric,
    kv_block_idle_before_evict_seconds: HistogramMetric,
    kv_block_reuse_gap_seconds: HistogramMetric,
}

impl KvCacheResidencyHandles {
    pub fn record(&self, events: &[KvCacheEvictionEvent]) {
        for event in events {
            self.kv_block_lifetime_seconds.observe(event.lifetime_seconds);
            self.kv_block_idle_before_evict_seconds.observe(event.idle_seconds);
            for reuse_gap_seconds in &event.reuse_gaps_seconds {
                self.kv_block_reuse_gap_seconds.observe(*reuse_gap_seconds);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use expect_test::expect;
    use itertools::Itertools as _;

    use super::{CachedTokensBySource, KvFetchReqsByStage, LoraInfoExporter, PrefixCacheStats};
    use crate::{CudagraphLogKey, EngineLabels, Metrics, SchedulerLogStatsAccumulator};

    #[test]
    fn perf_counters_render_with_a_single_total_suffix() {
        let metrics = Metrics::new();
        let labels = EngineLabels {
            model_name: "model".to_string(),
            engine: 0,
        };

        metrics.scheduler.perf.estimated_flops_per_gpu.get_or_create(&labels).inc();
        metrics.scheduler.perf.estimated_read_bytes_per_gpu.get_or_create(&labels).inc();
        metrics
            .scheduler
            .perf
            .estimated_write_bytes_per_gpu
            .get_or_create(&labels)
            .inc();

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains(
                "vllm:estimated_flops_per_gpu_total{model_name=\"model\",engine=\"0\"} 1"
            )
        );
        assert!(rendered.contains(
            "vllm:estimated_read_bytes_per_gpu_total{model_name=\"model\",engine=\"0\"} 1"
        ));
        assert!(rendered.contains(
            "vllm:estimated_write_bytes_per_gpu_total{model_name=\"model\",engine=\"0\"} 1"
        ));
        assert!(!rendered.contains("vllm:estimated_flops_per_gpu_total_total"));
        assert!(!rendered.contains("vllm:estimated_read_bytes_per_gpu_total_total"));
        assert!(!rendered.contains("vllm:estimated_write_bytes_per_gpu_total_total"));
    }

    #[test]
    fn log_stats_accumulator_drains_interval_data() {
        let accumulator = SchedulerLogStatsAccumulator::default();

        accumulator.observe_spec_decode(2, &[1, 2]);
        accumulator.observe_spec_decode(3, &[3, 4, 5]);
        accumulator.observe_cudagraph(8, 16, 8, "FULL");
        accumulator.observe_cudagraph(8, 16, 8, "FULL");

        let interval = accumulator.drain();

        assert_eq!(interval.spec_num_drafts, 5);
        assert_eq!(interval.spec_accepted_tokens_per_pos, vec![4, 6, 5]);
        assert_eq!(
            interval
                .cudagraph_counts
                .get(&CudagraphLogKey {
                    num_unpadded_tokens: 8,
                    num_padded_tokens: 16,
                    num_paddings: 8,
                    runtime_mode: "FULL".to_string(),
                })
                .copied(),
            Some(2)
        );
        assert_eq!(accumulator.drain().spec_num_drafts, 0);
    }

    #[test]
    fn prefix_cache_stats_hits_by_source_defaults_and_decodes() {
        // Engines without per-source attribution omit the field entirely.
        let legacy = serde_json::json!({
            "reset": false, "requests": 1, "queries": 8, "hits": 4,
            "preempted_requests": 0, "preempted_queries": 0, "preempted_hits": 0
        });
        let wire = rmp_serde::to_vec_named(&legacy).unwrap();
        let stats: PrefixCacheStats = rmp_serde::from_slice(&wire).unwrap();
        assert_eq!(stats.base.hits, 4);
        assert_eq!(stats.hits_by_source, CachedTokensBySource::default());

        let payload = serde_json::json!({
            "reset": false, "requests": 1, "queries": 16, "hits": 12,
            "preempted_requests": 0, "preempted_queries": 0, "preempted_hits": 0,
            "hits_by_source": {"host": 8, "p2p": 4}
        });
        let wire = rmp_serde::to_vec_named(&payload).unwrap();
        let stats: PrefixCacheStats = rmp_serde::from_slice(&wire).unwrap();
        assert_eq!(stats.hits_by_source.counts(), [0, 8, 4, 0, 0]);
        assert_eq!(CachedTokensBySource::SOURCES[1], "host");

        // Serialized as sparsely as Python sends it.
        let encoded: serde_json::Value =
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&stats).unwrap()).unwrap();
        assert_eq!(encoded, payload);
    }

    fn names(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|name| (*name).to_string()).collect()
    }

    /// The `lora_requests_info` series with the non-deterministic timestamp
    /// value replaced by `<ts>`, one line per series.
    fn lora_series(rendered: &str) -> String {
        rendered
            .lines()
            .filter(|l| l.starts_with("vllm:lora_requests_info{"))
            .map(|l| match l.rsplit_once("} ") {
                Some((labels, _value)) => format!("{labels}}} <ts>"),
                None => l.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn lora_info_emits_clears_stale_and_drains() {
        let metrics = Metrics::new();
        let mut exporter = LoraInfoExporter::new(4);

        // No adapters: nothing emitted.
        exporter.update(&metrics.scheduler, names(&[]), names(&[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));

        // Two running (sorted), one waiting.
        exporter.update(&metrics.scheduler, names(&["b", "a"]), names(&["c"]));
        expect![[
            r#"vllm:lora_requests_info{max_lora="4",running_lora_adapters="a,b",waiting_lora_adapters="c"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // "c" gets scheduled and "d" arrives: the stale series is replaced.
        exporter.update(&metrics.scheduler, names(&["a", "b", "c"]), names(&["d"]));
        expect![[
            r#"vllm:lora_requests_info{max_lora="4",running_lora_adapters="a,b,c",waiting_lora_adapters="d"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // Everything but "d" finishes.
        exporter.update(&metrics.scheduler, names(&["d"]), names(&[]));
        expect![[
            r#"vllm:lora_requests_info{max_lora="4",running_lora_adapters="d",waiting_lora_adapters=""} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // All requests done: series removed entirely.
        exporter.update(&metrics.scheduler, names(&[]), names(&[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));
    }

    /// Exports every engine's KV fetch stage gauges from startup and sets them
    /// from each `SchedulerStats` update.
    #[test]
    fn kv_fetch_reqs_by_stage_are_recorded_into_stage_gauges() {
        let metrics = Metrics::new();
        let handles = metrics.scheduler.resolve_handles("model", 0);
        // Series of one family render in hash order, so sort them.
        let kv_fetch_series = || {
            metrics
                .render()
                .unwrap()
                .lines()
                .filter(|line| line.starts_with("vllm:num_requests_kv_fetch_by_stage{"))
                .sorted()
                .join("\n")
        };

        expect![[r#"
            vllm:num_requests_kv_fetch_by_stage{model_name="model",engine="0",stage="completed_waiting"} 0
            vllm:num_requests_kv_fetch_by_stage{model_name="model",engine="0",stage="in_progress"} 0
            vllm:num_requests_kv_fetch_by_stage{model_name="model",engine="0",stage="waiting_to_start"} 0"#]]
        .assert_eq(&kv_fetch_series());

        handles.state.record(
            0,
            0,
            0,
            &KvFetchReqsByStage {
                waiting_to_start: 3,
                in_progress: 1,
                completed_waiting: 2,
            },
            0.0,
        );

        expect![[r#"
            vllm:num_requests_kv_fetch_by_stage{model_name="model",engine="0",stage="completed_waiting"} 2
            vllm:num_requests_kv_fetch_by_stage{model_name="model",engine="0",stage="in_progress"} 1
            vllm:num_requests_kv_fetch_by_stage{model_name="model",engine="0",stage="waiting_to_start"} 3"#]]
        .assert_eq(&kv_fetch_series());
    }
}
