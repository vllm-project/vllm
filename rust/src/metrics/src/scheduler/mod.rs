// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Scheduler-scoped metrics, exported from the engine's `SchedulerStats`.
//!
//! Each metric group has its own module, which keeps its part of the
//! `SchedulerStats` payload, its Prometheus families, its per-engine handles
//! and its recording together. KV connector telemetry has one module per
//! connector in `kv_connector`.
//! The `SchedulerStats` envelope stays with the engine protocol, which also
//! routes on it; the engine client hands each group its part.

use std::sync::Arc;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;

use crate::buckets::BucketConfig;

mod engine_state;
mod kv_cache_residency;
mod kv_connector;
mod log_stats;
mod perf;
mod prefix_cache;
mod spec_decode;
mod state;

pub use engine_state::{
    EngineSleepStateHandles, EngineSleepStateLabels, EngineStateMetrics, LoraAdapterNames,
    LoraInfoExporter, LoraInfoLabels, SLEEP_STATES,
};
pub use kv_cache_residency::{
    KvCacheEvictionEvent, KvCacheResidencyHandles, KvCacheResidencyMetrics,
};
pub use kv_connector::KvConnectorHandles;
use kv_connector::KvConnectorMetrics;
pub use kv_connector::{
    KvConnectorStats, MooncakeRecord, MooncakeStats, MooncakeTransferStats, MultiConnectorStats,
    NixlStats, SimpleCpuOffloadData, SimpleCpuOffloadStats, UmbpOperationCounts, UmbpStats,
};
pub use log_stats::{
    CudagraphLogKey, CudagraphStats, SchedulerLogStatsAccumulator, SchedulerLogStatsInterval,
};
pub use perf::{DebugPerfStats, PerfHandles, PerfMetrics, PerfStats};
pub use prefix_cache::{
    BaseCacheStats, CacheHitSourceLabels, CachedTokensBySource, PrefixCacheHandles,
    PrefixCacheMetrics, PrefixCacheStats,
};
pub use spec_decode::{
    EnginePositionLabels, SpecDecodeHandles, SpecDecodeMetrics, SpecDecodingStats,
};
pub use state::{
    KvFetchReqsByStage, KvFetchStageLabels, SchedulerStateHandles, SchedulerStateMetrics,
    WaitingReasonLabels,
};

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineLabels {
    pub model_name: String,
    pub engine: u32,
}

/// Scheduler/batch-scoped Prometheus families exported from `SchedulerStats`.
pub struct SchedulerMetrics {
    pub state: SchedulerStateMetrics,
    pub engine_state: EngineStateMetrics,
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
        Self {
            state: SchedulerStateMetrics::register(registry),
            engine_state: EngineStateMetrics::register(registry),
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
            engine_sleep_state: self.engine_state.resolve(&labels),
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
