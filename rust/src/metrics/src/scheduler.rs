use prometheus_client::encoding::EncodeLabelSet;
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

/// Scheduler/batch-scoped Prometheus families exported from `SchedulerStats`.
pub struct SchedulerMetrics {
    // Scheduler state gauges.
    pub scheduler_running: Family<EngineLabels, U64Gauge>,
    pub scheduler_waiting: Family<EngineLabels, U64Gauge>,
    pub kv_cache_usage: Family<EngineLabels, F64Gauge>,

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
    pub estimated_flops_per_gpu_total: Family<EngineLabels, U64Counter>,
    pub estimated_read_bytes_per_gpu_total: Family<EngineLabels, U64Counter>,
    pub estimated_write_bytes_per_gpu_total: Family<EngineLabels, U64Counter>,

    // Sampled KV-cache residency histograms.
    pub kv_block_lifetime_seconds: HistogramFamily,
    pub kv_block_idle_before_evict_seconds: HistogramFamily,
    pub kv_block_reuse_gap_seconds: HistogramFamily,
}

impl SchedulerMetrics {
    /// Register the scheduler-oriented metric families into the shared registry.
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

        let kv_cache_usage = Family::default();
        registry.register(
            "vllm:kv_cache_usage_perc",
            "KV-cache usage. 1 means 100 percent usage",
            kv_cache_usage.clone(),
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
        let estimated_flops_per_gpu_total = Family::default();
        registry.register(
            "vllm:estimated_flops_per_gpu_total",
            "Estimated number of floating point operations per GPU (for Model Flops Utilization calculations).",
            estimated_flops_per_gpu_total.clone(),
        );

        let estimated_read_bytes_per_gpu_total = Family::default();
        registry.register(
            "vllm:estimated_read_bytes_per_gpu_total",
            "Estimated number of bytes read from memory per GPU (for Model Flops Utilization calculations).",
            estimated_read_bytes_per_gpu_total.clone(),
        );

        let estimated_write_bytes_per_gpu_total = Family::default();
        registry.register(
            "vllm:estimated_write_bytes_per_gpu_total",
            "Estimated number of bytes written to memory per GPU (for Model Flops Utilization calculations).",
            estimated_write_bytes_per_gpu_total.clone(),
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

        Self {
            scheduler_running,
            scheduler_waiting,
            kv_cache_usage,
            prefix_cache_queries,
            prefix_cache_hits,
            external_prefix_cache_queries,
            external_prefix_cache_hits,
            spec_decode_num_drafts,
            spec_decode_num_draft_tokens,
            spec_decode_num_accepted_tokens,
            spec_decode_num_accepted_tokens_per_pos,
            estimated_flops_per_gpu_total,
            estimated_read_bytes_per_gpu_total,
            estimated_write_bytes_per_gpu_total,
            kv_block_lifetime_seconds,
            kv_block_idle_before_evict_seconds,
            kv_block_reuse_gap_seconds,
        }
    }
}
