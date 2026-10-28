// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::Serialize;
use vllm_engine_core_client::protocol::output::RequestSpecDecodeMetrics;
use vllm_llm::RequestTimestamps;

/// Per-request response timing fields matching Python's `PerRequestMetrics`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct PerRequestMetrics {
    pub time_to_first_token_ms: Option<f64>,
    pub generation_time_ms: Option<f64>,
    pub queue_time_ms: Option<f64>,
    pub mean_itl_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub speculative_decoding: Option<SpeculativeDecodingMetrics>,
}

impl PerRequestMetrics {
    pub fn from_timestamps(times: RequestTimestamps, num_generation_tokens: usize) -> Self {
        let duration = |end: f64, start: f64| (end > 0.0 && start > 0.0).then_some(end - start);
        let decode = duration(times.last_token_ts, times.first_token_ts);
        let inference = duration(times.last_token_ts, times.scheduled_ts);

        Self {
            time_to_first_token_ms: duration(times.first_token_ts, times.scheduled_ts)
                .map(|seconds| seconds * 1000.0),
            generation_time_ms: decode.map(|seconds| seconds * 1000.0),
            queue_time_ms: duration(times.scheduled_ts, times.queued_ts)
                .map(|seconds| seconds * 1000.0),
            mean_itl_ms: decode
                .filter(|_| num_generation_tokens > 1)
                .map(|seconds| seconds * 1000.0 / (num_generation_tokens - 1) as f64),
            tokens_per_second: inference
                .filter(|seconds| *seconds > 0.0)
                .map(|seconds| num_generation_tokens as f64 / seconds),
            speculative_decoding: None,
        }
    }

    pub fn from_request(
        include_timing: bool,
        times: RequestTimestamps,
        num_generation_tokens: usize,
        spec_decode_metrics: Option<RequestSpecDecodeMetrics>,
    ) -> Option<Self> {
        let speculative_decoding = spec_decode_metrics.map(Into::into);
        (include_timing || speculative_decoding.is_some()).then(|| {
            let mut metrics = if include_timing {
                Self::from_timestamps(times, num_generation_tokens)
            } else {
                Self::default()
            };
            metrics.speculative_decoding = speculative_decoding;
            metrics
        })
    }
}

/// Streaming form of [`PerRequestMetrics`].
///
/// Python serializes the final usage chunk with recursive `exclude_none=True`,
/// so unset fields are omitted instead of serialized as `null`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StreamPerRequestMetrics {
    pub time_to_first_token_ms: Option<f64>,
    pub generation_time_ms: Option<f64>,
    pub queue_time_ms: Option<f64>,
    pub mean_itl_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub speculative_decoding: Option<StreamingSpeculativeDecodingMetrics>,
}

impl From<PerRequestMetrics> for StreamPerRequestMetrics {
    fn from(metrics: PerRequestMetrics) -> Self {
        let PerRequestMetrics {
            time_to_first_token_ms,
            generation_time_ms,
            queue_time_ms,
            mean_itl_ms,
            tokens_per_second,
            speculative_decoding,
        } = metrics;
        Self {
            time_to_first_token_ms,
            generation_time_ms,
            queue_time_ms,
            mean_itl_ms,
            tokens_per_second,
            speculative_decoding: speculative_decoding.map(Into::into),
        }
    }
}

/// Mirrors the Python vLLM `SpeculativeDecodingMetrics` class.
///
/// Derived from the raw engine accumulator the same way as Python
/// `RequestSpecDecodeMetrics.to_dict`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeculativeDecodingMetrics {
    pub mean_acceptance_length: f64,
    pub draft_acceptance_rate: f64,
    pub acceptance_histogram: Vec<u64>,
    pub num_spec_steps: u64,
    pub num_accepted_draft_tokens: u64,
    pub num_draft_tokens: u64,
    pub num_spec_tokens: u64,
    pub per_step_accepted: Option<Vec<u64>>,
    pub per_step_drafted: Option<Vec<u64>>,
}

impl From<RequestSpecDecodeMetrics> for SpeculativeDecodingMetrics {
    fn from(raw: RequestSpecDecodeMetrics) -> Self {
        let num_spec_steps: u64 = raw.histogram.iter().sum();
        let num_accepted_draft_tokens: u64 =
            (0u64..).zip(&raw.histogram).map(|(accepted, count)| accepted * count).sum();
        let ratio = |num: u64, den: u64| {
            if den == 0 {
                0.0
            } else {
                num as f64 / den as f64
            }
        };
        let detailed = !raw.per_step_accepted.is_empty();
        Self {
            mean_acceptance_length: if num_spec_steps == 0 {
                1.0
            } else {
                1.0 + ratio(num_accepted_draft_tokens, num_spec_steps)
            },
            draft_acceptance_rate: ratio(num_accepted_draft_tokens, raw.num_draft_tokens),
            acceptance_histogram: raw.histogram,
            num_spec_steps,
            num_accepted_draft_tokens,
            num_draft_tokens: raw.num_draft_tokens,
            num_spec_tokens: raw.num_spec_tokens,
            per_step_accepted: detailed.then_some(raw.per_step_accepted),
            per_step_drafted: detailed.then_some(raw.per_step_drafted),
        }
    }
}

/// Streaming form omits detailed fields when summary metrics are requested.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StreamingSpeculativeDecodingMetrics {
    pub mean_acceptance_length: f64,
    pub draft_acceptance_rate: f64,
    pub acceptance_histogram: Vec<u64>,
    pub num_spec_steps: u64,
    pub num_accepted_draft_tokens: u64,
    pub num_draft_tokens: u64,
    pub num_spec_tokens: u64,
    pub per_step_accepted: Option<Vec<u64>>,
    pub per_step_drafted: Option<Vec<u64>>,
}

impl From<SpeculativeDecodingMetrics> for StreamingSpeculativeDecodingMetrics {
    fn from(metrics: SpeculativeDecodingMetrics) -> Self {
        Self {
            mean_acceptance_length: metrics.mean_acceptance_length,
            draft_acceptance_rate: metrics.draft_acceptance_rate,
            acceptance_histogram: metrics.acceptance_histogram,
            num_spec_steps: metrics.num_spec_steps,
            num_accepted_draft_tokens: metrics.num_accepted_draft_tokens,
            num_draft_tokens: metrics.num_draft_tokens,
            num_spec_tokens: metrics.num_spec_tokens,
            per_step_accepted: metrics.per_step_accepted,
            per_step_drafted: metrics.per_step_drafted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_timing_formulas_and_missing_fields() {
        let times = RequestTimestamps {
            queued_ts: 10.0,
            scheduled_ts: 10.2,
            first_token_ts: 10.5,
            last_token_ts: 11.0,
        };
        let metrics = PerRequestMetrics::from_timestamps(times, 6);
        assert!((metrics.time_to_first_token_ms.unwrap() - 300.0).abs() < 1e-9);
        assert_eq!(metrics.generation_time_ms, Some(500.0));
        assert!((metrics.queue_time_ms.unwrap() - 200.0).abs() < 1e-9);
        assert_eq!(metrics.mean_itl_ms, Some(100.0));
        assert!((metrics.tokens_per_second.unwrap() - 7.5).abs() < 1e-9);

        let single_token = PerRequestMetrics::from_timestamps(times, 1);
        assert_eq!(single_token.mean_itl_ms, None);

        let missing = PerRequestMetrics::from_timestamps(RequestTimestamps::default(), 0);
        assert_eq!(missing.time_to_first_token_ms, None);
        assert_eq!(missing.generation_time_ms, None);
        assert_eq!(missing.queue_time_ms, None);
        assert_eq!(missing.mean_itl_ms, None);
        assert_eq!(missing.tokens_per_second, None);
        assert_eq!(missing.speculative_decoding, None);
    }

    #[test]
    fn stream_serialization_omits_unset_fields() {
        let metrics = PerRequestMetrics::from_timestamps(
            RequestTimestamps {
                queued_ts: 10.0,
                scheduled_ts: 10.2,
                first_token_ts: 10.5,
                last_token_ts: 10.5,
            },
            1,
        );
        let value = serde_json::to_value(StreamPerRequestMetrics::from(metrics)).unwrap();
        let metrics = value.as_object().unwrap();
        assert_eq!(metrics.len(), 4);
        assert!(metrics.contains_key("time_to_first_token_ms"));
        assert!(metrics.contains_key("generation_time_ms"));
        assert!(metrics.contains_key("queue_time_ms"));
        assert!(metrics.contains_key("tokens_per_second"));
        assert!(!metrics.contains_key("mean_itl_ms"));
    }
}
