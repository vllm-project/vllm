use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

use crate::{FinishReasonCounterFamily, HistogramFamily};

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

/// Request-lifecycle Prometheus families exported from the `llm` layer.
pub struct RequestMetrics {
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

impl RequestMetrics {
    /// Register the request-oriented metric families into the shared registry.
    pub(crate) fn register(registry: &mut Registry) -> Self {
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
}
