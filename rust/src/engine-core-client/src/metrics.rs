// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

use vllm_metrics::{
    EngineLabels, EnginePositionLabels, F64Gauge, Family, HistogramMetric, LoraAdapterNames,
    LoraInfoLabels, METRICS, Metrics, MooncakeOperationCounterFamily,
    MooncakeOperationHistogramFamily, MooncakeOperationLabels, RequestMetrics,
    SchedulerLogStatsAccumulator, SchedulerMetrics, U64Counter, U64Gauge, WaitingReasonLabels,
};

use crate::connector_metrics::{DescriptorDrivenAdapter, observe_opaque_connector_stats};
use crate::protocol::stats::{
    KvConnectorStats, MooncakeStats, MultiConnectorStats, NixlStats, SchedulerStats,
};
use crate::transport::ConnectedEngine;

const WAITING_REASON_CAPACITY: &str = "capacity";
const WAITING_REASON_DEFERRED: &str = "deferred";

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
    /// Descriptor-driven recorder for third-party KV connector stats.
    generic_connector_metrics: DescriptorDrivenAdapter,
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

    // Mooncake store connector telemetry, decoded from `kv_connector_stats`.
    // Kept as `Family` (not pre-resolved) because `operation`/`status` are
    // dynamic per-record labels.
    mooncake_operation_time_seconds: MooncakeOperationHistogramFamily,
    mooncake_operation_total: MooncakeOperationCounterFamily,
    mooncake_operation_keys_total: MooncakeOperationCounterFamily,
    mooncake_operation_bytes_total: MooncakeOperationCounterFamily,
    mooncake_operation_failed_keys_total: MooncakeOperationCounterFamily,

    // NIXL connector telemetry, decoded from `kv_connector_stats`.
    nixl_xfer_time_seconds: HistogramMetric,
    nixl_post_time_seconds: HistogramMetric,
    nixl_bytes_transferred: HistogramMetric,
    nixl_num_descriptors: HistogramMetric,
    nixl_num_failed_transfers: U64Counter,
    nixl_num_failed_notifications: U64Counter,
    nixl_num_kv_expired_reqs: U64Counter,

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
        Self::new_with_metrics(&METRICS, metrics, model_name, engines)
    }

    /// Like [`Self::new`], but allows tests to inject a non-global metrics
    /// registry for descriptor-driven connector families.
    pub(crate) fn new_with_metrics(
        root_metrics: &'static Metrics,
        metrics: &SchedulerMetrics,
        model_name: &str,
        engines: &[ConnectedEngine],
    ) -> Self {
        let engines: BTreeMap<u32, SchedulerStatsHandles> = engines
            .iter()
            .filter_map(|engine| {
                let engine = engine.engine_id.engine_index()?;
                Some((
                    engine,
                    resolve_scheduler_stats_handles(metrics, model_name, engine),
                ))
            })
            .collect();
        let engine_indices: Vec<u32> = engines.keys().copied().collect();

        Self {
            engines,
            generic_connector_metrics: DescriptorDrivenAdapter::new(
                root_metrics,
                model_name,
                &engine_indices,
            ),
        }
    }

    /// Record one scheduler-stats payload for the given engine index.
    pub(crate) fn record(&self, engine_index: u32, stats: &SchedulerStats) {
        if let Some(handles) = self.engines.get(&engine_index) {
            record_scheduler_stats_with_handles(handles, stats, &self.generic_connector_metrics);
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
        mooncake_operation_time_seconds: metrics.mooncake_operation_time_seconds.clone(),
        mooncake_operation_total: metrics.mooncake_operation_total.clone(),
        mooncake_operation_keys_total: metrics.mooncake_operation_keys_total.clone(),
        mooncake_operation_bytes_total: metrics.mooncake_operation_bytes_total.clone(),
        mooncake_operation_failed_keys_total: metrics.mooncake_operation_failed_keys_total.clone(),
        nixl_xfer_time_seconds: metrics.nixl_xfer_time_seconds.get_or_create_owned(&labels),
        nixl_post_time_seconds: metrics.nixl_post_time_seconds.get_or_create_owned(&labels),
        nixl_bytes_transferred: metrics.nixl_bytes_transferred.get_or_create_owned(&labels),
        nixl_num_descriptors: metrics.nixl_num_descriptors.get_or_create_owned(&labels),
        nixl_num_failed_transfers: metrics.nixl_num_failed_transfers.get_or_create_owned(&labels),
        nixl_num_failed_notifications: metrics
            .nixl_num_failed_notifications
            .get_or_create_owned(&labels),
        nixl_num_kv_expired_reqs: metrics.nixl_num_kv_expired_reqs.get_or_create_owned(&labels),
        labels,
    }
}

/// Record scheduler-stats values through pre-resolved metric handles.
fn record_scheduler_stats_with_handles(
    handles: &SchedulerStatsHandles,
    stats: &SchedulerStats,
    generic: &DescriptorDrivenAdapter,
) {
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

    // Connector-specific KV transfer stats. A bare connector reports its own
    // flat payload; MultiConnector reports connector class name -> flat child
    // payload. Unknown / third-party children go through the descriptor-driven
    // generic adapter.
    if let Some(kv_connector_stats) = &stats.kv_connector_stats {
        match kv_connector_stats {
            KvConnectorStats::Nixl(stats) => record_nixl_stats(handles, stats),
            KvConnectorStats::Mooncake(stats) => record_mooncake_stats(handles, stats),
            KvConnectorStats::Multi(stats) => record_multi_connector_stats(handles, stats, generic),
            KvConnectorStats::Other(map) => {
                observe_opaque_connector_stats(
                    generic,
                    &handles.labels.model_name,
                    handles.labels.engine,
                    map,
                );
            }
        }
    }
}

fn record_multi_connector_stats(
    handles: &SchedulerStatsHandles,
    stats: &MultiConnectorStats,
    generic: &DescriptorDrivenAdapter,
) {
    for nixl in [&stats.nixl, &stats.nixl_pull, &stats.nixl_push].into_iter().flatten() {
        record_nixl_stats(handles, nixl);
    }
    if let Some(mooncake) = &stats.mooncake {
        record_mooncake_stats(handles, mooncake);
    }
    if !stats.other.is_empty() {
        observe_opaque_connector_stats(
            generic,
            &handles.labels.model_name,
            handles.labels.engine,
            &stats.other,
        );
    }
}

fn record_mooncake_stats(handles: &SchedulerStatsHandles, stats: &MooncakeStats) {
    for (operation, records) in &stats.0 {
        for record in records {
            let labels = MooncakeOperationLabels {
                model_name: handles.labels.model_name.clone(),
                engine: handles.labels.engine,
                operation: operation.as_str().to_string(),
                status: record.status.as_str().to_string(),
            };
            handles
                .mooncake_operation_time_seconds
                .get_or_create(&labels)
                .observe(record.duration_seconds);
            handles.mooncake_operation_total.get_or_create(&labels).inc();
            handles
                .mooncake_operation_keys_total
                .get_or_create(&labels)
                .inc_by(record.num_keys);
            handles
                .mooncake_operation_bytes_total
                .get_or_create(&labels)
                .inc_by(record.num_bytes);
            handles
                .mooncake_operation_failed_keys_total
                .get_or_create(&labels)
                .inc_by(record.num_failed_keys);
        }
    }
}

fn record_nixl_stats(handles: &SchedulerStatsHandles, stats: &NixlStats) {
    for value in &stats.transfer_duration {
        handles.nixl_xfer_time_seconds.observe(*value);
    }
    for value in &stats.post_duration {
        handles.nixl_post_time_seconds.observe(*value);
    }
    for value in &stats.bytes_transferred {
        handles.nixl_bytes_transferred.observe(*value as f64);
    }
    for value in &stats.num_descriptors {
        handles.nixl_num_descriptors.observe(*value as f64);
    }
    handles
        .nixl_num_failed_transfers
        .inc_by(stats.num_failed_transfers.iter().sum());
    handles
        .nixl_num_failed_notifications
        .inc_by(stats.num_failed_notifications.iter().sum());
    handles.nixl_num_kv_expired_reqs.inc_by(stats.num_kv_expired_reqs.iter().sum());
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
    use std::collections::{BTreeMap, BTreeSet};

    use expect_test::expect;
    use vllm_metrics::Metrics;

    use crate::metrics::LoraInfoExporter;
    use crate::protocol::stats::{
        KvConnectorStats, MooncakeOperation, MooncakeRecord, MooncakeStats, MooncakeStatus,
        MultiConnectorStats, NixlStats, SchedulerStats,
    };

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

    fn nixl_stats() -> NixlStats {
        NixlStats {
            transfer_duration: vec![0.01, 0.02],
            post_duration: vec![0.001, 0.002],
            bytes_transferred: vec![4096, 8192],
            num_descriptors: vec![2, 4],
            num_failed_transfers: vec![],
            num_failed_notifications: vec![],
            num_kv_expired_reqs: vec![1],
        }
    }

    fn mooncake_stats() -> MooncakeStats {
        MooncakeStats(BTreeMap::from([(
            MooncakeOperation::LoadGet,
            vec![MooncakeRecord {
                duration_seconds: 0.05,
                num_keys: 3,
                num_bytes: 1024,
                status: MooncakeStatus::Ok,
                num_failed_keys: 0,
            }],
        )]))
    }

    /// Records one MultiConnector payload into Mooncake and NIXL metrics.
    #[test]
    fn kv_connector_stats_are_recorded_into_mooncake_and_nixl_metrics() {
        let metrics = Metrics::new();
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);
        let generic = crate::connector_metrics::DescriptorDrivenAdapter::empty(Box::leak(
            Box::new(Metrics::new()),
        ));

        let stats = SchedulerStats {
            kv_connector_stats: Some(KvConnectorStats::Multi(Box::new(MultiConnectorStats {
                nixl: Some(nixl_stats()),
                mooncake: Some(mooncake_stats()),
                ..Default::default()
            }))),
            ..Default::default()
        };

        super::record_scheduler_stats_with_handles(&handles, &stats, &generic);

        let rendered = metrics.render().unwrap();
        assert!(rendered.contains(
            "vllm:mooncake_store_operation_total{model_name=\"model\",engine=\"0\",\
             operation=\"load_get\",status=\"ok\"} 1"
        ));
        assert!(rendered.contains(
            "vllm:mooncake_store_operation_keys_total{model_name=\"model\",engine=\"0\",\
             operation=\"load_get\",status=\"ok\"} 3"
        ));
        assert!(rendered.contains(
            "vllm:mooncake_store_operation_bytes_total{model_name=\"model\",engine=\"0\",\
             operation=\"load_get\",status=\"ok\"} 1024"
        ));
        assert!(
            rendered.contains(
                "vllm:nixl_num_kv_expired_reqs_total{model_name=\"model\",engine=\"0\"} 1"
            )
        );
        assert!(
            rendered
                .contains("vllm:nixl_xfer_time_seconds_count{model_name=\"model\",engine=\"0\"} 2")
        );
    }

    /// Encode `kv` inside `SchedulerStats` and decode it on the production path.
    ///
    /// `Other` is only an encoder for a bare msgpack map (the shape Python
    /// emits). Untagged `KvConnectorStats` matches `Multi` before `Other`, so
    /// recording the decoded value is what production does. Child-vs-flat
    /// classification itself is covered in `connector_metrics::dispatch`.
    fn decode_wire_scheduler_stats(kv: KvConnectorStats) -> SchedulerStats {
        let stats = SchedulerStats {
            kv_connector_stats: Some(kv),
            ..Default::default()
        };
        let bytes = rmp_serde::to_vec_named(&stats).expect("encode scheduler stats");
        let decoded: SchedulerStats =
            crate::protocol::decode_msgpack(&bytes).expect("decode scheduler stats");
        match &decoded.kv_connector_stats {
            Some(KvConnectorStats::Multi(_)) => decoded,
            Some(other) => panic!("wire map must decode as Multi, got {other:?}"),
            None => panic!("missing kv_connector_stats"),
        }
    }

    fn descriptor_value(document: serde_json::Value) -> rmpv::Value {
        rmpv::ext::to_value(&document).expect("descriptor to msgpack value")
    }

    /// Flat single-connector payload: counter plus scaled `u64` histogram.
    #[test]
    fn flat_descriptor_payload_decodes_as_multi_and_records_scaled_histogram() {
        use rmpv::Value;

        use crate::connector_metrics::DescriptorDrivenAdapter;
        use crate::connector_metrics::descriptor::METRICS_DESCRIPTOR_KEY;

        let metrics: &'static Metrics = Box::leak(Box::new(Metrics::new()));
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);
        let generic = DescriptorDrivenAdapter::empty(metrics);

        let mut first = BTreeMap::new();
        first.insert(
            METRICS_DESCRIPTOR_KEY.to_string(),
            descriptor_value(serde_json::json!({
                "descriptor_version": 1,
                "connector_id": "FakeConnector",
                "metrics": [
                    {
                        "name": "fake_puts",
                        "type": "counter",
                        "documentation": "Fake put attempts",
                        "samples_path": "put_total",
                        "sample_kind": "inc_by_u64"
                    },
                    {
                        "name": "fake_latency_seconds",
                        "type": "histogram",
                        "documentation": "Fake latency",
                        "buckets": [0.001, 0.01, 0.1, 1.0],
                        "samples_path": "latency_us",
                        "sample_kind": "observe_each_u64_as_f64",
                        "scale": 1e-6
                    }
                ]
            })),
        );
        first.insert("put_total".to_string(), Value::from(3u64));
        first.insert(
            "latency_us".to_string(),
            Value::Array(vec![Value::from(1000u64), Value::from(2000u64)]),
        );
        let decoded = decode_wire_scheduler_stats(KvConnectorStats::Other(first));
        super::record_scheduler_stats_with_handles(&handles, &decoded, &generic);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("fake_puts_total{engine=\"0\",model_name=\"model\"} 3"),
            "missing counter after decoded descriptor tick:\n{rendered}"
        );
        assert!(
            rendered.contains("fake_latency_seconds_count{engine=\"0\",model_name=\"model\"} 2"),
            "missing scaled histogram after decoded descriptor tick:\n{rendered}"
        );
        assert!(
            rendered.contains("fake_latency_seconds_sum{engine=\"0\",model_name=\"model\"} 0.003"),
            "u64 samples must be scaled to seconds:\n{rendered}"
        );
        assert_eq!(generic.registered_ids(), vec!["FakeConnector".to_string()]);

        // Data-only tick, still through msgpack → Multi, not the `Other` variant.
        let mut second = BTreeMap::new();
        second.insert("put_total".to_string(), Value::from(4u64));
        let decoded = decode_wire_scheduler_stats(KvConnectorStats::Other(second));
        super::record_scheduler_stats_with_handles(&handles, &decoded, &generic);
        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("fake_puts_total{engine=\"0\",model_name=\"model\"} 7"),
            "decoded data-only tick should continue observing:\n{rendered}"
        );
    }

    /// Offloading `types`+`data` with empty-tuple keys: gauge, f64 counter, histogram.
    #[test]
    fn offloading_label_tuple_payload_decodes_as_multi_and_records() {
        use rmpv::Value;

        use crate::connector_metrics::DescriptorDrivenAdapter;
        use crate::connector_metrics::descriptor::METRICS_DESCRIPTOR_KEY;

        let metrics: &'static Metrics = Box::leak(Box::new(Metrics::new()));
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);
        let generic = DescriptorDrivenAdapter::empty(metrics);

        let empty_tuple = Value::Array(vec![]);
        let data_map = vec![
            (
                Value::String("vllm:kv_offload_load_bytes".into()),
                Value::Map(vec![(empty_tuple.clone(), Value::from(100u64))]),
            ),
            (
                Value::String("vllm:kv_offload_load_time".into()),
                Value::Map(vec![(empty_tuple.clone(), Value::F64(1.5))]),
            ),
            (
                Value::String("vllm:kv_offload_cpu_cache_usage_perc".into()),
                Value::Map(vec![(empty_tuple.clone(), Value::F64(0.25))]),
            ),
            (
                Value::String("vllm:kv_offload_load_size".into()),
                Value::Map(vec![(
                    empty_tuple,
                    Value::Array(vec![Value::F64(1e6), Value::F64(2e6)]),
                )]),
            ),
        ];
        let mut flat = BTreeMap::new();
        flat.insert(
            METRICS_DESCRIPTOR_KEY.to_string(),
            descriptor_value(serde_json::json!({
                "descriptor_version": 1,
                "connector_id": "OffloadingConnector",
                "metrics": [
                    {
                        "name": "vllm:kv_offload_load_bytes",
                        "type": "counter",
                        "samples_path": "data.vllm:kv_offload_load_bytes",
                        "sample_kind": "inc_by_u64"
                    },
                    {
                        "name": "vllm:kv_offload_load_time",
                        "type": "counter",
                        "samples_path": "data.vllm:kv_offload_load_time",
                        "sample_kind": "inc_by_f64"
                    },
                    {
                        "name": "vllm:kv_offload_cpu_cache_usage_perc",
                        "type": "gauge",
                        "samples_path": "data.vllm:kv_offload_cpu_cache_usage_perc",
                        "sample_kind": "set_f64"
                    },
                    {
                        "name": "vllm:kv_offload_load_size",
                        "type": "histogram",
                        "buckets": [1e6, 5e6, 10e6],
                        "samples_path": "data.vllm:kv_offload_load_size",
                        "sample_kind": "observe_each_f64"
                    }
                ]
            })),
        );
        flat.insert("types".to_string(), Value::Map(vec![]));
        flat.insert("data".to_string(), Value::Map(data_map));

        let decoded = decode_wire_scheduler_stats(KvConnectorStats::Other(flat));
        super::record_scheduler_stats_with_handles(&handles, &decoded, &generic);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains(
                "vllm:kv_offload_load_bytes_total{engine=\"0\",model_name=\"model\"} 100"
            ),
            "missing load_bytes in:\n{rendered}"
        );
        assert!(
            rendered
                .contains("vllm:kv_offload_load_time_total{engine=\"0\",model_name=\"model\"} 1.5"),
            "missing load_time in:\n{rendered}"
        );
        assert!(
            rendered.contains(
                "vllm:kv_offload_cpu_cache_usage_perc{engine=\"0\",model_name=\"model\"} 0.25"
            ),
            "missing gauge in:\n{rendered}"
        );
        assert!(
            rendered
                .contains("vllm:kv_offload_load_size_count{engine=\"0\",model_name=\"model\"} 2"),
            "missing histogram in:\n{rendered}"
        );
    }

    /// MultiConnector child map decodes as `Multi` and records each child.
    #[test]
    fn multi_connector_payload_decodes_and_records_children() {
        use rmpv::Value;

        use crate::connector_metrics::DescriptorDrivenAdapter;
        use crate::connector_metrics::descriptor::METRICS_DESCRIPTOR_KEY;

        let metrics: &'static Metrics = Box::leak(Box::new(Metrics::new()));
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);
        let generic = DescriptorDrivenAdapter::empty(metrics);

        let example_payload = Value::Map(vec![
            (
                Value::String(METRICS_DESCRIPTOR_KEY.into()),
                descriptor_value(serde_json::json!({
                    "descriptor_version": 1,
                    "connector_id": "ExampleConnector",
                    "metrics": [{
                        "name": "example_put",
                        "type": "counter",
                        "samples_path": "put_total",
                        "sample_kind": "inc_by_u64"
                    }]
                })),
            ),
            (Value::String("put_total".into()), Value::from(7u64)),
        ]);
        let mut children = BTreeMap::new();
        children.insert("ExampleConnector".to_string(), example_payload);

        let decoded = decode_wire_scheduler_stats(KvConnectorStats::Other(children));
        super::record_scheduler_stats_with_handles(&handles, &decoded, &generic);

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains("example_put_total{engine=\"0\",model_name=\"model\"} 7"),
            "missing nested example connector in:\n{rendered}"
        );
        assert_eq!(
            generic.registered_ids(),
            vec!["ExampleConnector".to_string()]
        );
    }

    /// Flat stats without ``_metrics_descriptor`` decode as `Multi` and are dropped.
    #[test]
    fn flat_connector_without_payload_descriptor_is_not_recorded() {
        use rmpv::Value;

        use crate::connector_metrics::DescriptorDrivenAdapter;

        let metrics: &'static Metrics = Box::leak(Box::new(Metrics::new()));
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);
        let generic = DescriptorDrivenAdapter::empty(metrics);

        let mut other = BTreeMap::new();
        other.insert("put_total".to_string(), Value::from(5u64));
        other.insert("bytes_put".to_string(), Value::from(100u64));

        let decoded = decode_wire_scheduler_stats(KvConnectorStats::Other(other));
        super::record_scheduler_stats_with_handles(&handles, &decoded, &generic);

        let rendered = metrics.render().unwrap();
        assert!(
            !rendered.contains("put_total{"),
            "unregistered flat fields must not become Prom series:\n{rendered}"
        );
        assert!(generic.registered_ids().is_empty());
    }
}
