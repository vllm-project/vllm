// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

use rmpv::Value;
use vllm_metrics::{
    EngineLabels, EnginePositionLabels, F64Gauge, Family, HistogramMetric, LoraAdapterNames,
    LoraInfoLabels, MooncakeOperationCounterFamily, MooncakeOperationHistogramFamily,
    MooncakeOperationLabels, SchedulerLogStatsAccumulator, SchedulerMetrics, U64Counter, U64Gauge,
    WaitingReasonLabels,
};

use crate::protocol::OpaqueValue;
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

    // Connector-specific KV transfer stats. The payload is a
    // `{connector_class_name: {"data": <connector-specific shape>}}` map
    // (see `MultiKVConnectorStats`/`KVConnectorStats` on the Python side);
    // we only know how to decode the two connectors we run in production.
    if let Some(kv_connector_stats) = &stats.kv_connector_stats {
        for (connector_id, value) in kv_connector_stats {
            let Some(data) = map_get(value, "data") else {
                continue;
            };
            match connector_id.as_str() {
                "MooncakeStoreConnector" => record_mooncake_stats(handles, data),
                "NixlPullConnector" | "NixlPushConnector" => record_nixl_stats(handles, data),
                _ => {}
            }
        }
    }
}

/// Look up a string key in a msgpack map value, e.g. `{"data": ...}`.
fn map_get<'a>(value: &'a OpaqueValue, key: &str) -> Option<&'a OpaqueValue> {
    value.as_map()?.iter().find(|(k, _)| k.as_str() == Some(key)).map(|(_, v)| v)
}

/// Best-effort numeric coercion: msgpack may encode the same logical number
/// as a float or an integer depending on the Python value that produced it.
fn value_as_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|v| v as f64))
        .or_else(|| value.as_u64().map(|v| v as f64))
}

fn value_as_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().map(|v| v.max(0) as u64))
        .or_else(|| value.as_f64().map(|v| v as u64))
}

/// Decode one Mooncake store connector stats blob:
/// `{operation: [{"duration_seconds", "num_keys", "num_bytes", "status",
/// "num_failed_keys"}, ...]}`. Mirrors `MooncakeStorePromMetrics.observe()`
/// on the Python side.
fn record_mooncake_stats(handles: &SchedulerStatsHandles, data: &OpaqueValue) {
    let Some(operations) = data.as_map() else {
        return;
    };
    for (operation_key, records) in operations {
        let (Some(operation), Some(records)) = (operation_key.as_str(), records.as_array()) else {
            continue;
        };
        for record in records {
            let Some(fields) = record.as_map() else {
                continue;
            };
            let get =
                |key: &str| fields.iter().find(|(k, _)| k.as_str() == Some(key)).map(|(_, v)| v);

            let duration_seconds = get("duration_seconds").and_then(value_as_f64).unwrap_or(0.0);
            let num_keys = get("num_keys").and_then(value_as_u64).unwrap_or(0);
            let num_bytes = get("num_bytes").and_then(value_as_u64).unwrap_or(0);
            let num_failed_keys = get("num_failed_keys").and_then(value_as_u64).unwrap_or(0);
            let status = get("status").and_then(Value::as_str).unwrap_or("ok");

            let labels = MooncakeOperationLabels {
                model_name: handles.labels.model_name.clone(),
                engine: handles.labels.engine,
                operation: operation.to_string(),
                status: status.to_string(),
            };
            handles
                .mooncake_operation_time_seconds
                .get_or_create(&labels)
                .observe(duration_seconds);
            handles.mooncake_operation_total.get_or_create(&labels).inc();
            handles.mooncake_operation_keys_total.get_or_create(&labels).inc_by(num_keys);
            handles.mooncake_operation_bytes_total.get_or_create(&labels).inc_by(num_bytes);
            handles
                .mooncake_operation_failed_keys_total
                .get_or_create(&labels)
                .inc_by(num_failed_keys);
        }
    }
}

/// Decode one NIXL connector stats blob:
/// `{"transfer_duration"/"post_duration"/"bytes_transferred"/
/// "num_descriptors": [number, ...], "num_failed_transfers"/
/// "num_failed_notifications"/"num_kv_expired_reqs": [1, ...]}`. Mirrors
/// `NixlPromMetrics.observe()` on the Python side (each list is one
/// observation per transfer/event since the last flush).
fn record_nixl_stats(handles: &SchedulerStatsHandles, data: &OpaqueValue) {
    let list_len = |key: &str| -> u64 {
        map_get(data, key)
            .and_then(Value::as_array)
            .map(|values| values.len() as u64)
            .unwrap_or(0)
    };
    let list_values = |key: &str| -> Vec<f64> {
        map_get(data, key)
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(value_as_f64).collect())
            .unwrap_or_default()
    };

    for value in list_values("transfer_duration") {
        handles.nixl_xfer_time_seconds.observe(value);
    }
    for value in list_values("post_duration") {
        handles.nixl_post_time_seconds.observe(value);
    }
    for value in list_values("bytes_transferred") {
        handles.nixl_bytes_transferred.observe(value);
    }
    for value in list_values("num_descriptors") {
        handles.nixl_num_descriptors.observe(value);
    }
    handles.nixl_num_failed_transfers.inc_by(list_len("num_failed_transfers"));
    handles
        .nixl_num_failed_notifications
        .inc_by(list_len("num_failed_notifications"));
    handles.nixl_num_kv_expired_reqs.inc_by(list_len("num_kv_expired_reqs"));
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
    use crate::protocol::stats::SchedulerStats;

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

    fn msgpack_map(entries: Vec<(&str, rmpv::Value)>) -> rmpv::Value {
        rmpv::Value::Map(
            entries.into_iter().map(|(k, v)| (rmpv::Value::String(k.into()), v)).collect(),
        )
    }

    /// Builds a `kv_connector_stats` payload shaped like the one
    /// `MultiKVConnectorStats` sends over the wire: one Mooncake store
    /// "load_get" record and one NIXL transfer sample.
    #[test]
    fn kv_connector_stats_are_decoded_into_mooncake_and_nixl_metrics() {
        let metrics = Metrics::new();
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);

        let mooncake_record = msgpack_map(vec![
            ("duration_seconds", rmpv::Value::F64(0.05)),
            ("num_keys", rmpv::Value::from(3i64)),
            ("num_bytes", rmpv::Value::from(1024i64)),
            ("status", rmpv::Value::String("ok".into())),
            ("num_failed_keys", rmpv::Value::from(0i64)),
        ]);
        let mooncake_data = msgpack_map(vec![(
            "data",
            msgpack_map(vec![(
                "load_get",
                rmpv::Value::Array(vec![mooncake_record]),
            )]),
        )]);

        let nixl_data = msgpack_map(vec![(
            "data",
            msgpack_map(vec![
                (
                    "transfer_duration",
                    rmpv::Value::Array(vec![rmpv::Value::F64(0.01), rmpv::Value::F64(0.02)]),
                ),
                (
                    "post_duration",
                    rmpv::Value::Array(vec![rmpv::Value::F64(0.001)]),
                ),
                (
                    "bytes_transferred",
                    rmpv::Value::Array(vec![rmpv::Value::from(4096i64)]),
                ),
                (
                    "num_descriptors",
                    rmpv::Value::Array(vec![rmpv::Value::from(2i64)]),
                ),
                ("num_failed_transfers", rmpv::Value::Array(vec![])),
                ("num_failed_notifications", rmpv::Value::Array(vec![])),
                (
                    "num_kv_expired_reqs",
                    rmpv::Value::Array(vec![rmpv::Value::from(1i64)]),
                ),
            ]),
        )]);

        let mut kv_connector_stats = BTreeMap::new();
        kv_connector_stats.insert("MooncakeStoreConnector".to_string(), mooncake_data);
        kv_connector_stats.insert("NixlPullConnector".to_string(), nixl_data);

        let stats = SchedulerStats {
            kv_connector_stats: Some(kv_connector_stats),
            ..Default::default()
        };

        super::record_scheduler_stats_with_handles(&handles, &stats);

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
}
