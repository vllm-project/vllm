// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
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

pub fn per_request_metrics(
    enabled: bool,
    times: RequestTimestamps,
    num_generation_tokens: usize,
) -> Option<PerRequestMetrics> {
    enabled.then(|| PerRequestMetrics::from_timestamps(times, num_generation_tokens))
}

/// Serialize stream metrics like Pydantic's recursive `exclude_none=True`.
pub fn serialize_stream_metrics<S>(
    metrics: &Option<PerRequestMetrics>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let Some(metrics) = metrics else {
        return serializer.serialize_none();
    };
    let field_count = [
        metrics.time_to_first_token_ms,
        metrics.generation_time_ms,
        metrics.queue_time_ms,
        metrics.mean_itl_ms,
        metrics.tokens_per_second,
    ]
    .into_iter()
    .flatten()
    .count();
    let mut map = serializer.serialize_map(Some(field_count))?;
    if let Some(value) = metrics.time_to_first_token_ms {
        map.serialize_entry("time_to_first_token_ms", &value)?;
    }
    if let Some(value) = metrics.generation_time_ms {
        map.serialize_entry("generation_time_ms", &value)?;
    }
    if let Some(value) = metrics.queue_time_ms {
        map.serialize_entry("queue_time_ms", &value)?;
    }
    if let Some(value) = metrics.mean_itl_ms {
        map.serialize_entry("mean_itl_ms", &value)?;
    }
    if let Some(value) = metrics.tokens_per_second {
        map.serialize_entry("tokens_per_second", &value)?;
    }
    map.end()
}

#[cfg(test)]
mod tests {
    use serde::Serialize;

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
        #[derive(Serialize)]
        struct Chunk {
            #[serde(serialize_with = "serialize_stream_metrics")]
            metrics: Option<PerRequestMetrics>,
        }

        let metrics = PerRequestMetrics::from_timestamps(
            RequestTimestamps {
                queued_ts: 10.0,
                scheduled_ts: 10.2,
                first_token_ts: 10.5,
                last_token_ts: 10.5,
            },
            1,
        );
        let value = serde_json::to_value(Chunk {
            metrics: Some(metrics),
        })
        .unwrap();
        let metrics = value["metrics"].as_object().unwrap();
        assert_eq!(metrics.len(), 4);
        assert!(metrics.contains_key("time_to_first_token_ms"));
        assert!(metrics.contains_key("generation_time_ms"));
        assert!(metrics.contains_key("queue_time_ms"));
        assert!(metrics.contains_key("tokens_per_second"));
        assert!(!metrics.contains_key("mean_itl_ms"));
    }
}
