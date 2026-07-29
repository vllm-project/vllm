// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

use vllm_metrics::{
    EngineLabels, EnginePositionLabels, F64Gauge, Family, HistogramMetric, LoraAdapterNames,
    LoraInfoLabels, SchedulerLogStatsAccumulator, SchedulerMetrics, U64Counter, U64Gauge,
    WaitingReasonLabels,
};

use crate::protocol::stats::SchedulerStats;
use crate::transport::ConnectedEngine;

const WAITING_REASON_CAPACITY: &str = "capacity";
const WAITING_REASON_DEFERRED: &str = "deferred";

/// Cached scheduler-stats metric handles for all engines connected to one
/// frontend client.
pub(crate) struct SchedulerStatsRecorder {
    engines: BTreeMap<u32, SchedulerStatsHandles>,
}

/// Per-engine cached metric handles used while recording `SchedulerStats`.
struct SchedulerStatsHandles {
    // Base labels reused for dynamic child labels.
    labels: EngineLabels,

    // Scheduler state gauges.
    scheduler_running: U64Gauge,
    scheduler_waiting: U64Gauge,
    scheduler_waiting_capacity: U64Gauge,
    scheduler_waiting_deferred: U64Gauge,
    kv_cache_usage: F64Gauge,

    // Prefix-cache counters, including the connector-backed external cache path.
    prefix_cache_queries: U64Counter,
    prefix_cache_hits: U64Counter,
    external_prefix_cache_queries: U64Counter,
    external_prefix_cache_hits: U64Counter,

    // Speculative decoding counters.
    spec_decode_num_drafts: U64Counter,
    spec_decode_num_draft_tokens: U64Counter,
    spec_decode_num_accepted_tokens: U64Counter,
    spec_decode_num_accepted_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,

    // Per-engine performance / MFU counters.
    estimated_flops_per_gpu: U64Counter,
    estimated_read_bytes_per_gpu: U64Counter,
    estimated_write_bytes_per_gpu: U64Counter,

    // Sampled KV-cache residency histograms.
    kv_block_lifetime_seconds: HistogramMetric,
    kv_block_idle_before_evict_seconds: HistogramMetric,
    kv_block_reuse_gap_seconds: HistogramMetric,

    // Non-Prometheus interval accumulator for periodic text-log helpers.
    log_stats: SchedulerLogStatsAccumulator,
}

impl SchedulerStatsRecorder {
    /// Resolve the fixed-label metric handles for the connected engines.
    pub(crate) fn new(
        metrics: &SchedulerMetrics,
        model_name: &str,
        engines: &[ConnectedEngine],
    ) -> Self {
        let engines = engines
            .iter()
            .filter_map(|engine| {
                let engine = engine.engine_id.engine_index()?;
                Some((
                    engine,
                    resolve_scheduler_stats_handles(metrics, model_name, engine),
                ))
            })
            .collect();

        Self { engines }
    }

    /// Record one scheduler-stats payload for the given engine index.
    pub(crate) fn record(&self, engine_index: u32, stats: &SchedulerStats) {
        if let Some(handles) = self.engines.get(&engine_index) {
            record_scheduler_stats_with_handles(handles, stats);
        }
    }
}

/// Resolve all fixed-label scheduler metrics for one engine.
fn resolve_scheduler_stats_handles(
    metrics: &SchedulerMetrics,
    model_name: &str,
    engine: u32,
) -> SchedulerStatsHandles {
    let labels = EngineLabels {
        model_name: model_name.to_string(),
        engine,
    };
    let capacity = WaitingReasonLabels {
        model_name: model_name.to_string(),
        engine,
        reason: WAITING_REASON_CAPACITY,
    };
    let deferred = WaitingReasonLabels {
        model_name: model_name.to_string(),
        engine,
        reason: WAITING_REASON_DEFERRED,
    };

    SchedulerStatsHandles {
        scheduler_running: metrics.scheduler_running.get_or_create_owned(&labels),
        scheduler_waiting: metrics.scheduler_waiting.get_or_create_owned(&labels),
        scheduler_waiting_capacity: metrics
            .scheduler_waiting_by_reason
            .get_or_create_owned(&capacity),
        scheduler_waiting_deferred: metrics
            .scheduler_waiting_by_reason
            .get_or_create_owned(&deferred),
        kv_cache_usage: metrics.kv_cache_usage.get_or_create_owned(&labels),
        prefix_cache_queries: metrics.prefix_cache_queries.get_or_create_owned(&labels),
        prefix_cache_hits: metrics.prefix_cache_hits.get_or_create_owned(&labels),
        external_prefix_cache_queries: metrics
            .external_prefix_cache_queries
            .get_or_create_owned(&labels),
        external_prefix_cache_hits: metrics.external_prefix_cache_hits.get_or_create_owned(&labels),
        spec_decode_num_drafts: metrics.spec_decode_num_drafts.get_or_create_owned(&labels),
        spec_decode_num_draft_tokens: metrics
            .spec_decode_num_draft_tokens
            .get_or_create_owned(&labels),
        spec_decode_num_accepted_tokens: metrics
            .spec_decode_num_accepted_tokens
            .get_or_create_owned(&labels),
        spec_decode_num_accepted_tokens_per_pos: metrics
            .spec_decode_num_accepted_tokens_per_pos
            .clone(),
        log_stats: metrics.log_stats.get_or_create_owned(&labels),
        estimated_flops_per_gpu: metrics.estimated_flops_per_gpu.get_or_create_owned(&labels),
        estimated_read_bytes_per_gpu: metrics
            .estimated_read_bytes_per_gpu
            .get_or_create_owned(&labels),
        estimated_write_bytes_per_gpu: metrics
            .estimated_write_bytes_per_gpu
            .get_or_create_owned(&labels),
        kv_block_lifetime_seconds: metrics.kv_block_lifetime_seconds.get_or_create_owned(&labels),
        kv_block_idle_before_evict_seconds: metrics
            .kv_block_idle_before_evict_seconds
            .get_or_create_owned(&labels),
        kv_block_reuse_gap_seconds: metrics.kv_block_reuse_gap_seconds.get_or_create_owned(&labels),
        labels,
    }
}

/// Record scheduler-stats values through pre-resolved metric handles.
fn record_scheduler_stats_with_handles(handles: &SchedulerStatsHandles, stats: &SchedulerStats) {
    // Scheduler state gauges.
    handles.scheduler_running.set(stats.num_running_reqs);
    handles
        .scheduler_waiting
        .set(stats.num_waiting_reqs + stats.num_skipped_waiting_reqs);
    handles.scheduler_waiting_capacity.set(stats.num_waiting_reqs);
    handles.scheduler_waiting_deferred.set(stats.num_skipped_waiting_reqs);
    handles.kv_cache_usage.set(stats.kv_cache_usage);

    // Prefix-cache counters, including the connector-backed external cache path.
    handles.prefix_cache_queries.inc_by(stats.prefix_cache_stats.base.queries);
    handles.prefix_cache_hits.inc_by(stats.prefix_cache_stats.base.hits);

    if let Some(connector_prefix_cache_stats) = &stats.connector_prefix_cache_stats {
        handles
            .external_prefix_cache_queries
            .inc_by(connector_prefix_cache_stats.base.queries);
        handles
            .external_prefix_cache_hits
            .inc_by(connector_prefix_cache_stats.base.hits);
    }

    // Speculative decoding counters.
    if let Some(spec_decoding_stats) = &stats.spec_decoding_stats {
        handles.spec_decode_num_drafts.inc_by(spec_decoding_stats.num_drafts);
        handles
            .spec_decode_num_draft_tokens
            .inc_by(spec_decoding_stats.num_draft_tokens);
        handles
            .spec_decode_num_accepted_tokens
            .inc_by(spec_decoding_stats.num_accepted_tokens);
        handles.log_stats.observe_spec_decode(
            spec_decoding_stats.num_drafts,
            &spec_decoding_stats.num_accepted_tokens_per_pos,
        );

        for (position, accepted_tokens) in
            spec_decoding_stats.num_accepted_tokens_per_pos.iter().copied().enumerate()
        {
            handles
                .spec_decode_num_accepted_tokens_per_pos
                .get_or_create(&EnginePositionLabels {
                    model_name: handles.labels.model_name.clone(),
                    engine: handles.labels.engine,
                    position: position as u32,
                })
                .inc_by(accepted_tokens);
        }
    }

    // Per-engine performance / MFU counters.
    if let Some(perf_stats) = &stats.perf_stats
        && (perf_stats.num_flops_per_gpu != 0
            || perf_stats.num_read_bytes_per_gpu != 0
            || perf_stats.num_write_bytes_per_gpu != 0)
    {
        handles.estimated_flops_per_gpu.inc_by(perf_stats.num_flops_per_gpu);
        handles.estimated_read_bytes_per_gpu.inc_by(perf_stats.num_read_bytes_per_gpu);
        handles.estimated_write_bytes_per_gpu.inc_by(perf_stats.num_write_bytes_per_gpu);
    }

    if let Some(cudagraph_stats) = &stats.cudagraph_stats {
        handles.log_stats.observe_cudagraph(
            cudagraph_stats.num_unpadded_tokens,
            cudagraph_stats.num_padded_tokens,
            cudagraph_stats.num_paddings,
            &cudagraph_stats.runtime_mode,
        );
    }

    // Sampled KV-cache residency histograms.
    if !stats.kv_cache_eviction_events.is_empty() {
        for event in &stats.kv_cache_eviction_events {
            handles.kv_block_lifetime_seconds.observe(event.lifetime_seconds);
            handles.kv_block_idle_before_evict_seconds.observe(event.idle_seconds);
            for reuse_gap_seconds in &event.reuse_gaps_seconds {
                handles.kv_block_reuse_gap_seconds.observe(*reuse_gap_seconds);
            }
        }
    }
}

/// Exports `vllm:lora_requests_info` as a single series covering all LoRA
/// requests tracked by this client across every engine in the replica.
///
/// The engine's `SchedulerStats` never carries adapter names: the Python
/// frontend fills them in from per-request lifecycle events tracked by
/// `LoRARequestStates` in `vllm/v1/engine/output_processor.py`. The Rust
/// frontend mirrors that, deriving the sets from the request registry.
#[derive(Default)]
pub(crate) struct LoraInfoExporter {
    current: Option<LoraInfoLabels>,
}

impl LoraInfoExporter {
    pub(crate) fn update(
        &mut self,
        metrics: &SchedulerMetrics,
        running: BTreeSet<String>,
        waiting: BTreeSet<String>,
    ) {
        let next = (!running.is_empty() || !waiting.is_empty()).then_some(LoraInfoLabels {
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use expect_test::expect;
    use vllm_metrics::Metrics;

    use crate::metrics::LoraInfoExporter;

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
        let mut exporter = LoraInfoExporter::default();

        // No adapters: nothing emitted.
        exporter.update(&metrics.scheduler, names(&[]), names(&[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));

        // Two running (sorted), one waiting.
        exporter.update(&metrics.scheduler, names(&["b", "a"]), names(&["c"]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="a,b",waiting_lora_adapters="c"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // "c" gets scheduled and "d" arrives: the stale series is replaced.
        exporter.update(&metrics.scheduler, names(&["a", "b", "c"]), names(&["d"]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="a,b,c",waiting_lora_adapters="d"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // Everything but "d" finishes.
        exporter.update(&metrics.scheduler, names(&["d"]), names(&[]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="d",waiting_lora_adapters=""} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // All requests done: series removed entirely.
        exporter.update(&metrics.scheduler, names(&[]), names(&[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));
    }
}
