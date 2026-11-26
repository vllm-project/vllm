// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use vllm_metrics::{
    EngineLabels, HistogramMetric, RequestMetrics, SchedulerMetrics, SchedulerStatsHandles,
};

use crate::protocol::stats::SchedulerStats;
use crate::protocol::utility::SleepLevel;
use crate::transport::ConnectedEngine;

/// Engine sleep state exported as `vllm:engine_sleep_state`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EngineSleepState {
    Awake,
    Asleep(SleepLevel),
    /// After `release_kv_cache_memory`, which Python records like a level-0
    /// sleep.
    KvCacheReleased,
}

impl EngineSleepState {
    /// Gauge values for the `awake`, `weights_offloaded` and `discard_all`
    /// series, as Python's `PrometheusStatLogger.record_sleep_state` sets them.
    fn gauge_values(self) -> [u64; 3] {
        match self {
            Self::Awake => [1, 0, 0],
            Self::Asleep(SleepLevel::PauseOnly) | Self::KvCacheReleased => [0, 0, 0],
            Self::Asleep(SleepLevel::OffloadWeights) => [0, 1, 0],
            Self::Asleep(SleepLevel::DiscardAll) => [0, 0, 1],
        }
    }
}

/// Cached output-batch metric handles for one model and engine index.
pub(crate) struct IterationMetricHandles {
    pub iteration_tokens_total: HistogramMetric,
}

impl IterationMetricHandles {
    pub(crate) fn new(metrics: &RequestMetrics, model_name: &str, engine: u32) -> Self {
        let labels = EngineLabels {
            model_name: model_name.to_string(),
            engine,
        };
        Self {
            iteration_tokens_total: metrics.iteration_tokens_total.get_or_create_owned(&labels),
        }
    }
}

/// Cached scheduler-stats metric handles for all engines connected to one
/// frontend client.
pub(crate) struct SchedulerStatsRecorder {
    engines: BTreeMap<u32, SchedulerStatsHandles>,
}

impl SchedulerStatsRecorder {
    /// Resolve the fixed-label metric handles for the connected engines.
    pub(crate) fn new(
        metrics: &SchedulerMetrics,
        model_name: &str,
        engines: &[ConnectedEngine],
    ) -> Self {
        Self::record_cache_config_info(metrics, engines);
        let engines = engines
            .iter()
            .filter_map(|engine| {
                let engine = engine.engine_id.engine_index()?;
                Some((engine, metrics.resolve_handles(model_name, engine)))
            })
            .collect();

        let recorder = Self { engines };
        // Engines start awake, as Python sets when it creates the stat logger.
        recorder.record_sleep_state(EngineSleepState::Awake);
        recorder
    }

    /// Export each engine's cache config as `vllm:cache_config_info`, which
    /// Python also creates with its stat logger.
    fn record_cache_config_info(metrics: &SchedulerMetrics, engines: &[ConnectedEngine]) {
        for engine in engines {
            let (Some(index), Some(info)) = (
                engine.engine_id.engine_index(),
                &engine.ready_response.cache_config_info,
            ) else {
                continue;
            };
            metrics.engine_state.record_cache_config_info(index, info);
        }
    }

    /// Record the sleep state of every connected engine. Sleep and wake are
    /// utility calls on all engines, so the state is not per engine.
    pub(crate) fn record_sleep_state(&self, state: EngineSleepState) {
        let values = state.gauge_values();
        for handles in self.engines.values() {
            handles.engine_sleep_state.record(values);
        }
    }

    /// Record one scheduler-stats payload for the given engine index, handing
    /// each metric group its part.
    pub(crate) fn record(&self, engine_index: u32, stats: &SchedulerStats) {
        let Some(handles) = self.engines.get(&engine_index) else {
            return;
        };

        handles.state.record(
            stats.num_running_reqs,
            stats.num_waiting_reqs,
            stats.num_skipped_waiting_reqs,
            &stats.num_kv_fetch_reqs_by_stage,
            stats.kv_cache_usage,
            stats.num_preempted_reqs,
        );
        handles.prefix_cache.record(
            &stats.prefix_cache_stats,
            stats.connector_prefix_cache_stats.as_ref(),
        );

        if let Some(spec_decoding_stats) = &stats.spec_decoding_stats {
            handles.spec_decode.record(spec_decoding_stats);
            handles.log_stats.observe_spec_decode(
                spec_decoding_stats.num_drafts,
                &spec_decoding_stats.num_accepted_tokens_per_pos,
            );
        }

        if let Some(perf_stats) = &stats.perf_stats {
            handles.perf.record(perf_stats);
        }

        if let Some(cudagraph_stats) = &stats.cudagraph_stats {
            handles.log_stats.observe_cudagraph(
                cudagraph_stats.num_unpadded_tokens,
                cudagraph_stats.num_padded_tokens,
                cudagraph_stats.num_paddings,
                &cudagraph_stats.runtime_mode,
            );
        }

        handles.kv_cache_residency.record(&stats.kv_cache_eviction_events);

        if let Some(kv_connector_stats) = &stats.kv_connector_stats {
            handles.kv_connector.record(kv_connector_stats);
        }
    }
}
