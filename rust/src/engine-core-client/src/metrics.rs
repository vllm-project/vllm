// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::fmt::Write;

use tracing::info;
use vllm_metrics::{
    EngineLabels, HistogramMetric, RequestMetrics, SchedulerMetrics, SchedulerStatsHandles,
};

use crate::protocol::stats::{SchedulerIterationDetails, SchedulerStats};
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
        for record in &stats.iteration_records {
            info!(
                "{}",
                iteration_log_line(engine_index, record, stats.kv_cache_usage)
            );
        }
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

/// One iteration-details line, as Python's `LoggingStatLogger` logs each record
/// of `--enable-logging-iteration-details`.
fn iteration_log_line(
    engine_index: u32,
    details: &SchedulerIterationDetails,
    kv_cache_usage: f64,
) -> String {
    let mut line = format!(
        "Engine {engine_index:03}: Iteration({}): {} context requests, {} context tokens, \
         {} generation requests, {} generation tokens",
        details.iteration_index,
        details.num_ctx_requests,
        details.num_ctx_tokens,
        details.num_generation_requests,
        details.num_generation_tokens,
    );
    if let Some(elapsed_ms) = details.elapsed_ms {
        write!(line, ", iteration elapsed time: {elapsed_ms:.2} ms").unwrap();
    }
    if details.is_dummy {
        line.push_str(" (dummy)");
    }
    write!(line, ", GPU KV cache usage: {:.1}%", kv_cache_usage * 100.0).unwrap();
    if details.num_encoder_inputs > 0 {
        write!(
            line,
            ", encoder inputs: {}, encoder output embeddings: {}",
            details.num_encoder_inputs, details.num_encoder_output_tokens
        )
        .unwrap();
    }
    line
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::{Arc, Mutex};

    use expect_test::expect;
    use vllm_metrics::METRICS;

    use super::SchedulerStatsRecorder;
    use crate::protocol::stats::{SchedulerIterationDetails, SchedulerStats};

    /// Captures formatted log output in memory.
    #[derive(Clone, Default)]
    struct LogBuffer(Arc<Mutex<Vec<u8>>>);

    impl io::Write for LogBuffer {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn scheduler_stats_log_one_line_per_iteration_record_like_python() {
        let record =
            |iteration_index, ctx: (u64, u64), generation: (u64, u64)| SchedulerIterationDetails {
                iteration_index,
                num_ctx_requests: ctx.0,
                num_ctx_tokens: ctx.1,
                num_generation_requests: generation.0,
                num_generation_tokens: generation.1,
                ..Default::default()
            };
        // The first four are the cases of Python's
        // `test_iteration_lines_have_an_elapsed_time_only_when_timed`.
        let stats = SchedulerStats {
            kv_cache_usage: 0.125,
            iteration_records: vec![
                SchedulerIterationDetails {
                    elapsed_ms: Some(1.5),
                    ..record(0, (1, 8), (2, 2))
                },
                record(1, (0, 0), (2, 2)),
                SchedulerIterationDetails {
                    is_dummy: true,
                    ..record(2, (0, 0), (0, 0))
                },
                SchedulerIterationDetails {
                    elapsed_ms: Some(0.0),
                    is_dummy: true,
                    ..record(3, (0, 0), (0, 0))
                },
                SchedulerIterationDetails {
                    num_encoder_inputs: 2,
                    num_encoder_output_tokens: 128,
                    ..record(4, (2, 300), (1, 1))
                },
            ],
            ..Default::default()
        };
        let recorder = SchedulerStatsRecorder {
            engines: [(
                3,
                METRICS.scheduler.resolve_handles("iteration-log-test", 3),
            )]
            .into(),
        };

        let buffer = LogBuffer::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer({
                let buffer = buffer.clone();
                move || buffer.clone()
            })
            .with_ansi(false)
            .without_time()
            .with_target(false)
            .finish();
        tracing::subscriber::with_default(subscriber, || recorder.record(3, &stats));

        let logs = String::from_utf8(buffer.0.lock().unwrap().clone()).unwrap();
        // The formatter pads `INFO` to the width of the longest level name.
        let logs = logs.lines().map(str::trim_start).collect::<Vec<_>>().join("\n");
        expect![[r#"
            INFO Engine 003: Iteration(0): 1 context requests, 8 context tokens, 2 generation requests, 2 generation tokens, iteration elapsed time: 1.50 ms, GPU KV cache usage: 12.5%
            INFO Engine 003: Iteration(1): 0 context requests, 0 context tokens, 2 generation requests, 2 generation tokens, GPU KV cache usage: 12.5%
            INFO Engine 003: Iteration(2): 0 context requests, 0 context tokens, 0 generation requests, 0 generation tokens (dummy), GPU KV cache usage: 12.5%
            INFO Engine 003: Iteration(3): 0 context requests, 0 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 0.00 ms (dummy), GPU KV cache usage: 12.5%
            INFO Engine 003: Iteration(4): 2 context requests, 300 context tokens, 1 generation requests, 1 generation tokens, GPU KV cache usage: 12.5%, encoder inputs: 2, encoder output embeddings: 128"#]].assert_eq(&logs);
    }
}
