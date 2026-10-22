// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;

use crate::buckets::{BucketConfig, BucketFamily, FamilyHistogram};
use crate::{BucketHistogramFamily, EngineLabels, U64Counter};

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct FinishedReasonLabels {
    pub model_name: String,
    pub engine: u32,
    pub finished_reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct PromptTokenSourceLabels {
    pub model_name: String,
    pub engine: u32,
    pub source: &'static str,
}

pub(crate) type FinishedReasonCounterFamily = Family<FinishedReasonLabels, U64Counter>;
pub(crate) type PromptTokenSourceCounterFamily = Family<PromptTokenSourceLabels, U64Counter>;
/// Request and output-batch Prometheus families.
pub struct RequestMetrics {
    /// Computed prompt and generated tokens per nonempty engine output batch,
    /// recorded by the engine-core client for requests tracked by this frontend.
    pub iteration_tokens_total: BucketHistogramFamily,

    // Request-derived counters.
    pub num_preemptions: Family<EngineLabels, U64Counter>,
    pub prompt_tokens: Family<EngineLabels, U64Counter>,
    pub prompt_tokens_by_source: PromptTokenSourceCounterFamily,
    pub prompt_tokens_cached: Family<EngineLabels, U64Counter>,
    pub generation_tokens: Family<EngineLabels, U64Counter>,
    pub corrupted_requests: Family<EngineLabels, U64Counter>,

    // Request lifecycle counters and histograms.
    pub request_success: FinishedReasonCounterFamily,
    pub request_prompt_tokens: BucketHistogramFamily,
    pub request_generation_tokens: BucketHistogramFamily,
    pub request_max_num_generation_tokens: BucketHistogramFamily,
    pub request_params_max_tokens: BucketHistogramFamily,
    pub request_params_n: BucketHistogramFamily,
    pub request_num_preemptions: BucketHistogramFamily,
    pub request_prefill_kv_computed_tokens: BucketHistogramFamily,
    pub time_to_first_token_seconds: BucketHistogramFamily,
    pub inter_token_latency_seconds: BucketHistogramFamily,
    pub e2e_request_latency_seconds: BucketHistogramFamily,
    pub request_queue_time_seconds: BucketHistogramFamily,
    pub request_prefill_time_seconds: BucketHistogramFamily,
    pub request_decode_time_seconds: BucketHistogramFamily,
    pub request_inference_time_seconds: BucketHistogramFamily,
    pub request_time_per_output_token_seconds: BucketHistogramFamily,

    /// Bucket settings shared with the histogram constructors.
    buckets: Arc<BucketConfig>,
}

impl RequestMetrics {
    /// Register the request-oriented metric families into the shared registry.
    pub(crate) fn register(registry: &mut Registry, buckets: &Arc<BucketConfig>) -> Self {
        let histogram =
            |family| Family::new_with_constructor(FamilyHistogram::new(family, buckets));

        let iteration_tokens_total = histogram(BucketFamily::IterationTokens);
        registry.register(
            "vllm:iteration_tokens_total",
            "Histogram of computed prompt and generated tokens per nonempty engine output batch.",
            iteration_tokens_total.clone(),
        );

        // Request-derived counters.
        let num_preemptions = Family::default();
        registry.register(
            "vllm:num_preemptions",
            "Cumulative number of preemption events.",
            num_preemptions.clone(),
        );

        let prompt_tokens = Family::default();
        registry.register(
            "vllm:prompt_tokens",
            "Number of prefill tokens processed.",
            prompt_tokens.clone(),
        );

        let prompt_tokens_by_source = Family::default();
        registry.register(
            "vllm:prompt_tokens_by_source",
            "Number of prompt tokens by source.",
            prompt_tokens_by_source.clone(),
        );

        let prompt_tokens_cached = Family::default();
        registry.register(
            "vllm:prompt_tokens_cached",
            "Number of prompt tokens with prefix cache hits.",
            prompt_tokens_cached.clone(),
        );

        let generation_tokens = Family::default();
        registry.register(
            "vllm:generation_tokens",
            "Number of generation tokens processed.",
            generation_tokens.clone(),
        );

        let corrupted_requests = Family::default();
        registry.register(
            "vllm:corrupted_requests",
            "Corrupted requests, in terms of total number of requests with NaNs in logits.",
            corrupted_requests.clone(),
        );

        // Request lifecycle counters and histograms.
        let request_success = Family::default();
        registry.register(
            "vllm:request_success",
            "Count of successfully processed requests.",
            request_success.clone(),
        );

        let request_prompt_tokens = histogram(BucketFamily::RequestTokens);
        registry.register(
            "vllm:request_prompt_tokens",
            "Number of prefill tokens processed.",
            request_prompt_tokens.clone(),
        );

        let request_generation_tokens = histogram(BucketFamily::RequestTokens);
        registry.register(
            "vllm:request_generation_tokens",
            "Number of generation tokens processed.",
            request_generation_tokens.clone(),
        );

        let request_max_num_generation_tokens = histogram(BucketFamily::RequestTokens);
        registry.register(
            "vllm:request_max_num_generation_tokens",
            "Histogram of maximum number of requested generation tokens.",
            request_max_num_generation_tokens.clone(),
        );

        let request_params_max_tokens = histogram(BucketFamily::RequestTokens);
        registry.register(
            "vllm:request_params_max_tokens",
            "Histogram of the max_tokens request parameter.",
            request_params_max_tokens.clone(),
        );

        let request_params_n = histogram(BucketFamily::RequestParamsN);
        registry.register(
            "vllm:request_params_n",
            "Histogram of the n request parameter.",
            request_params_n.clone(),
        );

        let request_num_preemptions = histogram(BucketFamily::RequestNumPreemptions);
        registry.register(
            "vllm:request_num_preemptions",
            "Histogram of the number of times a request was preempted.",
            request_num_preemptions.clone(),
        );

        let request_prefill_kv_computed_tokens = histogram(BucketFamily::RequestTokens);
        registry.register(
            "vllm:request_prefill_kv_computed_tokens",
            "Histogram of new KV tokens computed during prefill (excluding cached tokens).",
            request_prefill_kv_computed_tokens.clone(),
        );

        let time_to_first_token_seconds = histogram(BucketFamily::TimeToFirstToken);
        registry.register(
            "vllm:time_to_first_token_seconds",
            "Histogram of time to first token in seconds.",
            time_to_first_token_seconds.clone(),
        );

        let inter_token_latency_seconds = histogram(BucketFamily::InterTokenLatency);
        registry.register(
            "vllm:inter_token_latency_seconds",
            "Histogram of inter-token latency in seconds.",
            inter_token_latency_seconds.clone(),
        );

        let e2e_request_latency_seconds = histogram(BucketFamily::RequestLatency);
        registry.register(
            "vllm:e2e_request_latency_seconds",
            "Histogram of e2e request latency in seconds.",
            e2e_request_latency_seconds.clone(),
        );

        let request_queue_time_seconds = histogram(BucketFamily::RequestLatency);
        registry.register(
            "vllm:request_queue_time_seconds",
            "Histogram of time spent in WAITING phase for request.",
            request_queue_time_seconds.clone(),
        );

        let request_prefill_time_seconds = histogram(BucketFamily::RequestLatency);
        registry.register(
            "vllm:request_prefill_time_seconds",
            "Histogram of time spent in PREFILL phase for request.",
            request_prefill_time_seconds.clone(),
        );

        let request_decode_time_seconds = histogram(BucketFamily::RequestLatency);
        registry.register(
            "vllm:request_decode_time_seconds",
            "Histogram of time spent in DECODE phase for request.",
            request_decode_time_seconds.clone(),
        );

        let request_inference_time_seconds = histogram(BucketFamily::RequestLatency);
        registry.register(
            "vllm:request_inference_time_seconds",
            "Histogram of time spent in RUNNING phase for request.",
            request_inference_time_seconds.clone(),
        );

        let request_time_per_output_token_seconds = histogram(BucketFamily::InterTokenLatency);
        registry.register(
            "vllm:request_time_per_output_token_seconds",
            "Histogram of time_per_output_token_seconds per request.",
            request_time_per_output_token_seconds.clone(),
        );

        Self {
            iteration_tokens_total,
            num_preemptions,
            prompt_tokens,
            prompt_tokens_by_source,
            prompt_tokens_cached,
            generation_tokens,
            corrupted_requests,
            request_success,
            request_prompt_tokens,
            request_generation_tokens,
            request_max_num_generation_tokens,
            request_params_max_tokens,
            request_params_n,
            request_num_preemptions,
            request_prefill_kv_computed_tokens,
            time_to_first_token_seconds,
            inter_token_latency_seconds,
            e2e_request_latency_seconds,
            request_queue_time_seconds,
            request_prefill_time_seconds,
            request_decode_time_seconds,
            request_inference_time_seconds,
            request_time_per_output_token_seconds,
            buckets: buckets.clone(),
        }
    }

    /// Cap the request token-count histogram buckets at the engine-reported
    /// `max_model_len`.
    ///
    /// Must be called before the first request is recorded. The registry is
    /// process-global and serves one model, so only the first call takes
    /// effect.
    pub fn set_max_model_len(&self, max_model_len: u32) {
        self.buckets.set_max_model_len(max_model_len);
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use itertools::Itertools;
    use prometheus_client::encoding::text::encode;

    use super::*;

    /// Request token-count histograms use Python's 1-2-5 buckets capped at
    /// `max_model_len`, with no extra bound at `max_model_len` itself.
    #[test]
    fn request_token_buckets_are_capped_at_max_model_len() {
        let mut registry = Registry::default();
        let metrics = RequestMetrics::register(&mut registry, &Arc::default());
        metrics.set_max_model_len(4096);
        metrics.request_prompt_tokens.get_or_create_owned(&EngineLabels {
            model_name: "model".to_string(),
            engine: 0,
        });

        let mut output = String::new();
        encode(&mut output, &registry).unwrap();
        let bounds = output
            .lines()
            .filter(|line| line.starts_with("vllm:request_prompt_tokens_bucket"))
            .filter_map(|line| line.split_once("le=\"")?.1.split_once('"'))
            .map(|(bound, _)| bound)
            .join(" ");
        expect!["1.0 2.0 5.0 10.0 20.0 50.0 100.0 200.0 500.0 1000.0 2000.0 +Inf"]
            .assert_eq(&bounds);
    }
}
