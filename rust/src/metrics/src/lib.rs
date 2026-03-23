use std::fmt;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;
use prometheus_client_derive_encode::EncodeLabelSet;

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
pub struct FinishReasonLabels {
    pub model_name: String,
    pub engine: u32,
    pub finish_reason: &'static str,
}

type U64Counter = Counter<u64, AtomicU64>;
type U64Gauge = Gauge<u64, AtomicU64>;
type F64Gauge = Gauge<f64, AtomicU64>;
type HistogramFamily = Family<EngineLabels, Histogram, fn() -> Histogram>;
type FinishReasonCounterFamily = Family<FinishReasonLabels, U64Counter>;

const KV_CACHE_RESIDENCY_BUCKETS: [f64; 21] = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0,
    120.0, 300.0, 600.0, 1200.0, 1800.0,
];
const TTFT_BUCKETS: [f64; 22] = [
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
];
const ITL_BUCKETS: [f64; 19] = [
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
    40.0, 80.0,
];
const REQUEST_LATENCY_BUCKETS: [f64; 21] = [
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0,
    480.0, 960.0, 1920.0, 7680.0,
];
const REQUEST_PARAMS_N_BUCKETS: [f64; 5] = [1.0, 2.0, 5.0, 10.0, 20.0];

fn build_1_2_5_buckets(max_value: u32) -> Vec<f64> {
    let mut buckets = Vec::new();
    let mut exponent = 0;
    loop {
        for mantissa in [1_u32, 2, 5] {
            let value = mantissa * 10_u32.pow(exponent);
            if value <= max_value {
                buckets.push(value as f64);
            } else {
                if buckets.last().copied() != Some(max_value as f64) {
                    buckets.push(max_value as f64);
                }
                return buckets;
            }
        }
        exponent += 1;
    }
}

fn kv_block_lifetime_histogram() -> Histogram {
    Histogram::new(KV_CACHE_RESIDENCY_BUCKETS.iter().copied())
}

fn kv_block_idle_before_evict_histogram() -> Histogram {
    Histogram::new(KV_CACHE_RESIDENCY_BUCKETS.iter().copied())
}

fn kv_block_reuse_gap_histogram() -> Histogram {
    Histogram::new(KV_CACHE_RESIDENCY_BUCKETS.iter().copied())
}

fn time_to_first_token_histogram() -> Histogram {
    Histogram::new(TTFT_BUCKETS.iter().copied())
}

fn inter_token_latency_histogram() -> Histogram {
    Histogram::new(ITL_BUCKETS.iter().copied())
}

fn request_time_per_output_token_histogram() -> Histogram {
    Histogram::new(ITL_BUCKETS.iter().copied())
}

fn request_latency_histogram() -> Histogram {
    Histogram::new(REQUEST_LATENCY_BUCKETS.iter().copied())
}

fn request_count_histogram() -> Histogram {
    Histogram::new(build_1_2_5_buckets(131_072))
}

fn request_params_n_histogram() -> Histogram {
    Histogram::new(REQUEST_PARAMS_N_BUCKETS.iter().copied())
}

/// Shared Prometheus registry for frontend metrics.
///
/// This currently owns the scheduler-stats-backed subset of frontend metrics. More metric
/// families can be registered here over time as the Rust frontend closes the remaining
/// observability gap.
pub struct Metrics {
    registry: Registry,

    pub scheduler_running: Family<EngineLabels, U64Gauge>,
    pub scheduler_waiting: Family<EngineLabels, U64Gauge>,
    pub kv_cache_usage: Family<EngineLabels, F64Gauge>,
    pub prefix_cache_queries: Family<EngineLabels, U64Counter>,
    pub prefix_cache_hits: Family<EngineLabels, U64Counter>,
    pub external_prefix_cache_queries: Family<EngineLabels, U64Counter>,
    pub external_prefix_cache_hits: Family<EngineLabels, U64Counter>,
    pub spec_decode_num_drafts: Family<EngineLabels, U64Counter>,
    pub spec_decode_num_draft_tokens: Family<EngineLabels, U64Counter>,
    pub spec_decode_num_accepted_tokens: Family<EngineLabels, U64Counter>,
    pub spec_decode_num_accepted_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,
    pub estimated_flops_per_gpu_total: Family<EngineLabels, U64Counter>,
    pub estimated_read_bytes_per_gpu_total: Family<EngineLabels, U64Counter>,
    pub estimated_write_bytes_per_gpu_total: Family<EngineLabels, U64Counter>,
    pub kv_block_lifetime_seconds: HistogramFamily,
    pub kv_block_idle_before_evict_seconds: HistogramFamily,
    pub kv_block_reuse_gap_seconds: HistogramFamily,

    pub request_success: FinishReasonCounterFamily,
    pub request_prompt_tokens: HistogramFamily,
    pub request_generation_tokens: HistogramFamily,
    pub request_max_num_generation_tokens: HistogramFamily,
    pub request_params_max_tokens: HistogramFamily,
    pub request_params_n: HistogramFamily,
    pub request_prefill_kv_computed_tokens: HistogramFamily,
    pub time_to_first_token_seconds: HistogramFamily,
    pub inter_token_latency_seconds: HistogramFamily,
    pub e2e_request_latency_seconds: HistogramFamily,
    pub request_queue_time_seconds: HistogramFamily,
    pub request_prefill_time_seconds: HistogramFamily,
    pub request_decode_time_seconds: HistogramFamily,
    pub request_inference_time_seconds: HistogramFamily,
    pub request_time_per_output_token_seconds: HistogramFamily,
}

impl Metrics {
    /// Construct a new metrics registry.
    pub fn new() -> Self {
        let mut registry = Registry::default();

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

        // Request lifecycle counters and histograms.
        let request_success = Family::default();
        registry.register(
            "vllm:request_success",
            "Count of successfully processed requests.",
            request_success.clone(),
        );

        let request_prompt_tokens =
            Family::new_with_constructor(request_count_histogram as fn() -> Histogram);
        registry.register(
            "vllm:request_prompt_tokens",
            "Number of prefill tokens processed.",
            request_prompt_tokens.clone(),
        );

        let request_generation_tokens =
            Family::new_with_constructor(request_count_histogram as fn() -> Histogram);
        registry.register(
            "vllm:request_generation_tokens",
            "Number of generation tokens processed.",
            request_generation_tokens.clone(),
        );

        let request_max_num_generation_tokens =
            Family::new_with_constructor(request_count_histogram as fn() -> Histogram);
        registry.register(
            "vllm:request_max_num_generation_tokens",
            "Histogram of maximum number of requested generation tokens.",
            request_max_num_generation_tokens.clone(),
        );

        let request_params_max_tokens =
            Family::new_with_constructor(request_count_histogram as fn() -> Histogram);
        registry.register(
            "vllm:request_params_max_tokens",
            "Histogram of the max_tokens request parameter.",
            request_params_max_tokens.clone(),
        );

        let request_params_n =
            Family::new_with_constructor(request_params_n_histogram as fn() -> Histogram);
        registry.register(
            "vllm:request_params_n",
            "Histogram of the n request parameter.",
            request_params_n.clone(),
        );

        let request_prefill_kv_computed_tokens =
            Family::new_with_constructor(request_count_histogram as fn() -> Histogram);
        registry.register(
            "vllm:request_prefill_kv_computed_tokens",
            "Histogram of new KV tokens computed during prefill (excluding cached tokens).",
            request_prefill_kv_computed_tokens.clone(),
        );

        let time_to_first_token_seconds =
            Family::new_with_constructor(time_to_first_token_histogram as fn() -> Histogram);
        registry.register(
            "vllm:time_to_first_token_seconds",
            "Histogram of time to first token in seconds.",
            time_to_first_token_seconds.clone(),
        );

        let inter_token_latency_seconds =
            Family::new_with_constructor(inter_token_latency_histogram as fn() -> Histogram);
        registry.register(
            "vllm:inter_token_latency_seconds",
            "Histogram of inter-token latency in seconds.",
            inter_token_latency_seconds.clone(),
        );

        let e2e_request_latency_seconds =
            Family::new_with_constructor(request_latency_histogram as fn() -> Histogram);
        registry.register(
            "vllm:e2e_request_latency_seconds",
            "Histogram of e2e request latency in seconds.",
            e2e_request_latency_seconds.clone(),
        );

        let request_queue_time_seconds =
            Family::new_with_constructor(request_latency_histogram as fn() -> Histogram);
        registry.register(
            "vllm:request_queue_time_seconds",
            "Histogram of time spent in WAITING phase for request.",
            request_queue_time_seconds.clone(),
        );

        let request_prefill_time_seconds =
            Family::new_with_constructor(request_latency_histogram as fn() -> Histogram);
        registry.register(
            "vllm:request_prefill_time_seconds",
            "Histogram of time spent in PREFILL phase for request.",
            request_prefill_time_seconds.clone(),
        );

        let request_decode_time_seconds =
            Family::new_with_constructor(request_latency_histogram as fn() -> Histogram);
        registry.register(
            "vllm:request_decode_time_seconds",
            "Histogram of time spent in DECODE phase for request.",
            request_decode_time_seconds.clone(),
        );

        let request_inference_time_seconds =
            Family::new_with_constructor(request_latency_histogram as fn() -> Histogram);
        registry.register(
            "vllm:request_inference_time_seconds",
            "Histogram of time spent in RUNNING phase for request.",
            request_inference_time_seconds.clone(),
        );

        let request_time_per_output_token_seconds = Family::new_with_constructor(
            request_time_per_output_token_histogram as fn() -> Histogram,
        );
        registry.register(
            "vllm:request_time_per_output_token_seconds",
            "Histogram of time_per_output_token_seconds per request.",
            request_time_per_output_token_seconds.clone(),
        );

        Self {
            registry,
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
            request_success,
            request_prompt_tokens,
            request_generation_tokens,
            request_max_num_generation_tokens,
            request_params_max_tokens,
            request_params_n,
            request_prefill_kv_computed_tokens,
            time_to_first_token_seconds,
            inter_token_latency_seconds,
            e2e_request_latency_seconds,
            request_queue_time_seconds,
            request_prefill_time_seconds,
            request_decode_time_seconds,
            request_inference_time_seconds,
            request_time_per_output_token_seconds,
        }
    }

    /// Render the current metrics registry into Prometheus/OpenMetrics text format.
    pub fn render(&self) -> Result<String, fmt::Error> {
        let mut output = String::new();
        encode(&mut output, &self.registry)?;
        Ok(output)
    }

    /// Return the registry owned by this metrics object.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-global metrics registry shared by the frontend crates.
pub static METRICS: LazyLock<Metrics> = LazyLock::new(Metrics::new);
