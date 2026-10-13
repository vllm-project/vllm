// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::Serialize;
use vllm_llm::RequestTimestamps;

/// Per-request response timing fields matching Python's `PerRequestMetrics`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerRequestMetrics {
    pub time_to_first_token_ms: Option<f64>,
    pub generation_time_ms: Option<f64>,
    pub queue_time_ms: Option<f64>,
    pub mean_itl_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
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
        }
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
}

impl From<PerRequestMetrics> for StreamPerRequestMetrics {
    fn from(metrics: PerRequestMetrics) -> Self {
        let PerRequestMetrics {
            time_to_first_token_ms,
            generation_time_ms,
            queue_time_ms,
            mean_itl_ms,
            tokens_per_second,
        } = metrics;
        Self {
            time_to_first_token_ms,
            generation_time_ms,
            queue_time_ms,
            mean_itl_ms,
            tokens_per_second,
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
