// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use itertools::Itertools as _;
use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue, LabelValueEncoder};
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

use crate::{F64Gauge, HistogramFamily, U64Counter, U64Gauge};

const KV_CACHE_RESIDENCY_BUCKETS: [f64; 21] = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0,
    120.0, 300.0, 600.0, 1200.0, 1800.0,
];

fn kv_block_lifetime_histogram() -> Histogram {
    Histogram::new(KV_CACHE_RESIDENCY_BUCKETS.iter().copied())
}

fn kv_block_idle_before_evict_histogram() -> Histogram {
    Histogram::new(KV_CACHE_RESIDENCY_BUCKETS.iter().copied())
}

fn kv_block_reuse_gap_histogram() -> Histogram {
    Histogram::new(KV_CACHE_RESIDENCY_BUCKETS.iter().copied())
}

// Buckets copied verbatim from the Python `MooncakeStorePromMetrics` /
// `NixlPromMetrics` classes so dashboards built against either frontend are
// interchangeable.
const MOONCAKE_OPERATION_TIME_BUCKETS: [f64; 15] = [
    1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 7.5e-1, 1.0, 1.5, 2.0, 3.0, 4.0,
];
const NIXL_XFER_TIME_BUCKETS: [f64; 12] = [
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 5.0,
];
const NIXL_POST_TIME_BUCKETS: [f64; 13] = [
    0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 5.0,
];
// Uniform 2KB to 8GB range: 2**(10+i) for i in range(1, 25, 2).
const NIXL_BYTES_TRANSFERRED_BUCKETS: [f64; 12] = [
    2048.0,
    8192.0,
    32768.0,
    131072.0,
    524288.0,
    2097152.0,
    8388608.0,
    33554432.0,
    134217728.0,
    536870912.0,
    2147483648.0,
    8589934592.0,
];
const NIXL_NUM_DESCRIPTORS_BUCKETS: [f64; 14] = [
    10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 200.0, 400.0, 1000.0, 2000.0, 4000.0, 10000.0, 20000.0,
    50000.0,
];

fn mooncake_operation_time_histogram() -> Histogram {
    Histogram::new(MOONCAKE_OPERATION_TIME_BUCKETS.iter().copied())
}

fn nixl_xfer_time_histogram() -> Histogram {
    Histogram::new(NIXL_XFER_TIME_BUCKETS.iter().copied())
}

fn nixl_post_time_histogram() -> Histogram {
    Histogram::new(NIXL_POST_TIME_BUCKETS.iter().copied())
}

fn nixl_bytes_transferred_histogram() -> Histogram {
    Histogram::new(NIXL_BYTES_TRANSFERRED_BUCKETS.iter().copied())
}

fn nixl_num_descriptors_histogram() -> Histogram {
    Histogram::new(NIXL_NUM_DESCRIPTORS_BUCKETS.iter().copied())
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineLabels {
    pub model_name: String,
    pub engine: u32,
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

/// Labels for per-operation Mooncake store connector telemetry. `operation`
/// and `status` are dynamic (not known ahead of time), unlike the other
/// fixed-label families in this module.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct MooncakeOperationLabels {
    pub model_name: String,
    pub engine: u32,
    pub operation: String,
    pub status: String,
}

/// Family type aliases for the Mooncake connector telemetry, exposed so
/// `engine-core-client` can hold onto the `Family` directly (rather than a
/// pre-resolved handle) since `operation`/`status` vary per record.
pub type MooncakeOperationHistogramFamily =
    Family<MooncakeOperationLabels, Histogram, fn() -> Histogram>;
pub type MooncakeOperationCounterFamily = Family<MooncakeOperationLabels, U64Counter>;

/// Adapter names encoded as a deterministic comma-joined Prometheus label value.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct LoraAdapterNames(pub BTreeSet<String>);

impl EncodeLabelValue for LoraAdapterNames {
    fn encode(&self, encoder: &mut LabelValueEncoder) -> Result<(), std::fmt::Error> {
        EncodeLabelValue::encode(&self.0.iter().join(","), encoder)
    }
}

/// Labels for `vllm:lora_requests_info`.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct LoraInfoLabels {
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

/// Scheduler/batch-scoped Prometheus families exported from `SchedulerStats`.
pub struct SchedulerMetrics {
    // Scheduler state gauges.
    pub scheduler_running: Family<EngineLabels, U64Gauge>,
    pub scheduler_waiting: Family<EngineLabels, U64Gauge>,
    pub scheduler_waiting_by_reason: Family<WaitingReasonLabels, U64Gauge>,
    pub kv_cache_usage: Family<EngineLabels, F64Gauge>,

    /// `vllm:lora_requests_info`. Value is the emit-time unix timestamp in
    /// seconds.
    pub lora_info: Family<LoraInfoLabels, F64Gauge>,

    // Prefix-cache counters, including the connector-backed external cache path.
    pub prefix_cache_queries: Family<EngineLabels, U64Counter>,
    pub prefix_cache_hits: Family<EngineLabels, U64Counter>,
    pub external_prefix_cache_queries: Family<EngineLabels, U64Counter>,
    pub external_prefix_cache_hits: Family<EngineLabels, U64Counter>,

    // Speculative decoding counters.
    pub spec_decode_num_drafts: Family<EngineLabels, U64Counter>,
    pub spec_decode_num_draft_tokens: Family<EngineLabels, U64Counter>,
    pub spec_decode_num_accepted_tokens: Family<EngineLabels, U64Counter>,
    pub spec_decode_num_accepted_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,

    // Per-engine performance / MFU counters.
    pub estimated_flops_per_gpu: Family<EngineLabels, U64Counter>,
    pub estimated_read_bytes_per_gpu: Family<EngineLabels, U64Counter>,
    pub estimated_write_bytes_per_gpu: Family<EngineLabels, U64Counter>,

    // Sampled KV-cache residency histograms.
    pub kv_block_lifetime_seconds: HistogramFamily,
    pub kv_block_idle_before_evict_seconds: HistogramFamily,
    pub kv_block_reuse_gap_seconds: HistogramFamily,

    // Mooncake store connector telemetry. Mirrors `MooncakeStorePromMetrics`.
    pub mooncake_operation_time_seconds: MooncakeOperationHistogramFamily,
    pub mooncake_operation_total: MooncakeOperationCounterFamily,
    pub mooncake_operation_keys_total: MooncakeOperationCounterFamily,
    pub mooncake_operation_bytes_total: MooncakeOperationCounterFamily,
    pub mooncake_operation_failed_keys_total: MooncakeOperationCounterFamily,

    // NIXL connector telemetry. Mirrors `NixlPromMetrics`.
    pub nixl_xfer_time_seconds: HistogramFamily,
    pub nixl_post_time_seconds: HistogramFamily,
    pub nixl_bytes_transferred: HistogramFamily,
    pub nixl_num_descriptors: HistogramFamily,
    pub nixl_num_failed_transfers: Family<EngineLabels, U64Counter>,
    pub nixl_num_failed_notifications: Family<EngineLabels, U64Counter>,
    pub nixl_num_kv_expired_reqs: Family<EngineLabels, U64Counter>,

    /// Non-Prometheus interval accumulators for periodic text-log helpers.
    pub log_stats: Family<EngineLabels, SchedulerLogStatsAccumulator>,
}

impl SchedulerMetrics {
    /// Register the scheduler-oriented metric families into the shared
    /// registry.
    pub(crate) fn register(registry: &mut Registry) -> Self {
        // Scheduler state gauges.
        let scheduler_running = Family::default();
        registry.register(
            "vllm:num_requests_running",
            "Number of requests in model execution batches",
            scheduler_running.clone(),
        );

        let scheduler_waiting = Family::default();
        registry.register(
            "vllm:num_requests_waiting",
            "Number of requests waiting to be processed",
            scheduler_waiting.clone(),
        );

        let scheduler_waiting_by_reason = Family::default();
        registry.register(
            "vllm:num_requests_waiting_by_reason",
            "Number of waiting requests by reason. \
             Reason labels: 'capacity' = waiting for scheduling capacity; \
             'deferred' = deferred by transient constraints (LoRA budget, KV transfer, \
             blocked status). Sum of all reasons equals vllm:num_requests_waiting.",
            scheduler_waiting_by_reason.clone(),
        );

        let kv_cache_usage = Family::default();
        registry.register(
            "vllm:kv_cache_usage_perc",
            "KV-cache usage. 1 means 100 percent usage",
            kv_cache_usage.clone(),
        );

        let lora_info = Family::default();
        registry.register(
            "vllm:lora_requests_info",
            "Running stats on lora requests.",
            lora_info.clone(),
        );

        // Prefix-cache counters, including the connector-backed external cache path.
        let prefix_cache_queries = Family::default();
        registry.register(
            "vllm:prefix_cache_queries",
            "Prefix cache queries, in terms of number of queried tokens",
            prefix_cache_queries.clone(),
        );

        let prefix_cache_hits = Family::default();
        registry.register(
            "vllm:prefix_cache_hits",
            "Prefix cache hits, in terms of number of cached tokens.",
            prefix_cache_hits.clone(),
        );

        let external_prefix_cache_queries = Family::default();
        registry.register(
            "vllm:external_prefix_cache_queries",
            "External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens.",
            external_prefix_cache_queries.clone(),
        );

        let external_prefix_cache_hits = Family::default();
        registry.register(
            "vllm:external_prefix_cache_hits",
            "External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens.",
            external_prefix_cache_hits.clone(),
        );

        // Speculative decoding counters.
        let spec_decode_num_drafts = Family::default();
        registry.register(
            "vllm:spec_decode_num_drafts",
            "Number of spec decoding drafts.",
            spec_decode_num_drafts.clone(),
        );

        let spec_decode_num_draft_tokens = Family::default();
        registry.register(
            "vllm:spec_decode_num_draft_tokens",
            "Number of draft tokens.",
            spec_decode_num_draft_tokens.clone(),
        );

        let spec_decode_num_accepted_tokens = Family::default();
        registry.register(
            "vllm:spec_decode_num_accepted_tokens",
            "Number of accepted tokens.",
            spec_decode_num_accepted_tokens.clone(),
        );

        let spec_decode_num_accepted_tokens_per_pos = Family::default();
        registry.register(
            "vllm:spec_decode_num_accepted_tokens_per_pos",
            "Accepted tokens per draft position.",
            spec_decode_num_accepted_tokens_per_pos.clone(),
        );

        // Per-engine performance / MFU counters.
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

        // Sampled KV-cache residency histograms.
        let kv_block_lifetime_seconds =
            Family::new_with_constructor(kv_block_lifetime_histogram as fn() -> Histogram);
        registry.register(
            "vllm:kv_block_lifetime_seconds",
            "Histogram of KV cache block lifetime from allocation to eviction. Sampled metrics (controlled by --kv-cache-metrics-sample).",
            kv_block_lifetime_seconds.clone(),
        );

        let kv_block_idle_before_evict_seconds =
            Family::new_with_constructor(kv_block_idle_before_evict_histogram as fn() -> Histogram);
        registry.register(
            "vllm:kv_block_idle_before_evict_seconds",
            "Histogram of idle time before KV cache block eviction. Sampled metrics (controlled by --kv-cache-metrics-sample).",
            kv_block_idle_before_evict_seconds.clone(),
        );

        let kv_block_reuse_gap_seconds =
            Family::new_with_constructor(kv_block_reuse_gap_histogram as fn() -> Histogram);
        registry.register(
            "vllm:kv_block_reuse_gap_seconds",
            "Histogram of time gaps between consecutive KV cache block accesses. Only the most recent accesses are recorded (ring buffer). Sampled metrics (controlled by --kv-cache-metrics-sample).",
            kv_block_reuse_gap_seconds.clone(),
        );

        // Mooncake store connector telemetry.
        let mooncake_operation_time_seconds =
            Family::new_with_constructor(mooncake_operation_time_histogram as fn() -> Histogram);
        registry.register(
            "vllm:mooncake_store_operation_time_seconds",
            "Histogram of Mooncake store communication time.",
            mooncake_operation_time_seconds.clone(),
        );

        // NOTE: registration names below omit the `_total` suffix that the
        // Python metric names carry (e.g. `vllm:mooncake_store_operation_total`)
        // because this crate's `prometheus-client` (unlike Python's, which
        // strips a user-supplied `_total` before re-adding it) appends
        // `_total` unconditionally for counters; see the note in `lib.rs`.
        let mooncake_operation_total = Family::default();
        registry.register(
            "vllm:mooncake_store_operation",
            "Number of Mooncake store communication operations.",
            mooncake_operation_total.clone(),
        );

        let mooncake_operation_keys_total = Family::default();
        registry.register(
            "vllm:mooncake_store_operation_keys",
            "Number of Mooncake store keys touched by operations.",
            mooncake_operation_keys_total.clone(),
        );

        let mooncake_operation_bytes_total = Family::default();
        registry.register(
            "vllm:mooncake_store_operation_bytes",
            "Number of bytes transferred by Mooncake store operations.",
            mooncake_operation_bytes_total.clone(),
        );

        let mooncake_operation_failed_keys_total = Family::default();
        registry.register(
            "vllm:mooncake_store_operation_failed_keys",
            "Number of Mooncake store keys that failed in operations.",
            mooncake_operation_failed_keys_total.clone(),
        );

        // NIXL connector telemetry.
        let nixl_xfer_time_seconds =
            Family::new_with_constructor(nixl_xfer_time_histogram as fn() -> Histogram);
        registry.register(
            "vllm:nixl_xfer_time_seconds",
            "Histogram of transfer duration for NIXL KV Cache transfers.",
            nixl_xfer_time_seconds.clone(),
        );

        let nixl_post_time_seconds =
            Family::new_with_constructor(nixl_post_time_histogram as fn() -> Histogram);
        registry.register(
            "vllm:nixl_post_time_seconds",
            "Histogram of transfer post time for NIXL KV Cache transfers.",
            nixl_post_time_seconds.clone(),
        );

        let nixl_bytes_transferred =
            Family::new_with_constructor(nixl_bytes_transferred_histogram as fn() -> Histogram);
        registry.register(
            "vllm:nixl_bytes_transferred",
            "Histogram of bytes transferred per NIXL KV Cache transfers.",
            nixl_bytes_transferred.clone(),
        );

        let nixl_num_descriptors =
            Family::new_with_constructor(nixl_num_descriptors_histogram as fn() -> Histogram);
        registry.register(
            "vllm:nixl_num_descriptors",
            "Histogram of number of descriptors per NIXL KV Cache transfers.",
            nixl_num_descriptors.clone(),
        );

        let nixl_num_failed_transfers = Family::default();
        registry.register(
            "vllm:nixl_num_failed_transfers",
            "Number of failed NIXL KV Cache transfers.",
            nixl_num_failed_transfers.clone(),
        );

        let nixl_num_failed_notifications = Family::default();
        registry.register(
            "vllm:nixl_num_failed_notifications",
            "Number of failed NIXL KV Cache notifications.",
            nixl_num_failed_notifications.clone(),
        );

        let nixl_num_kv_expired_reqs = Family::default();
        registry.register(
            "vllm:nixl_num_kv_expired_reqs",
            "Number of requests that had their KV expire. \
             NOTE: This metric is tracked on the P instance.",
            nixl_num_kv_expired_reqs.clone(),
        );

        Self {
            scheduler_running,
            scheduler_waiting,
            scheduler_waiting_by_reason,
            kv_cache_usage,
            lora_info,
            prefix_cache_queries,
            prefix_cache_hits,
            external_prefix_cache_queries,
            external_prefix_cache_hits,
            spec_decode_num_drafts,
            spec_decode_num_draft_tokens,
            spec_decode_num_accepted_tokens,
            spec_decode_num_accepted_tokens_per_pos,
            estimated_flops_per_gpu,
            estimated_read_bytes_per_gpu,
            estimated_write_bytes_per_gpu,
            kv_block_lifetime_seconds,
            kv_block_idle_before_evict_seconds,
            kv_block_reuse_gap_seconds,
            mooncake_operation_time_seconds,
            mooncake_operation_total,
            mooncake_operation_keys_total,
            mooncake_operation_bytes_total,
            mooncake_operation_failed_keys_total,
            nixl_xfer_time_seconds,
            nixl_post_time_seconds,
            nixl_bytes_transferred,
            nixl_num_descriptors,
            nixl_num_failed_transfers,
            nixl_num_failed_notifications,
            nixl_num_kv_expired_reqs,
            log_stats: Family::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{CudagraphLogKey, EngineLabels, Metrics, SchedulerLogStatsAccumulator};

    #[test]
    fn perf_counters_render_with_a_single_total_suffix() {
        let metrics = Metrics::new();
        let labels = EngineLabels {
            model_name: "model".to_string(),
            engine: 0,
        };

        metrics.scheduler.estimated_flops_per_gpu.get_or_create(&labels).inc();
        metrics.scheduler.estimated_read_bytes_per_gpu.get_or_create(&labels).inc();
        metrics.scheduler.estimated_write_bytes_per_gpu.get_or_create(&labels).inc();

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
}
