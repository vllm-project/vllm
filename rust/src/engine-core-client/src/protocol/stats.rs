// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};
// The parts of `SchedulerStats` that only metrics read are defined next to
// the metric families they feed.
pub use vllm_metrics::{
    BaseCacheStats, CachedTokensBySource, CudagraphStats, DebugPerfStats, KvCacheEvictionEvent,
    KvConnectorStats, KvFetchReqsByStage, MooncakeRecord, MooncakeStats, MooncakeTransferStats,
    MultiConnectorStats, NixlStats, OffloadingSample, OffloadingStats, PerfStats, PrefixCacheStats,
    SimpleCpuOffloadData, SimpleCpuOffloadStats, SpecDecodingStats, UmbpOperationCounts, UmbpStats,
};

use super::OpaqueValue;

/// Breakdown of a scheduled prefill computation.
///
/// Python models this as a plain `@dataclass`, so it is serialized by msgspec
/// as a map (named fields) rather than in the array-like form used by
/// `EngineCoreOutput` itself.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/d3af8c18317c0dc008d42e4367fbb9045cfb7bf6/vllm/v1/metrics/stats.py#L242-L273>
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillStats {
    /// Total number of tokens to be prefilled.
    #[serde(default)]
    pub num_prompt_tokens: u32,
    /// Tokens to be prefilled locally (actual compute work).
    #[serde(default)]
    pub num_computed_tokens: u32,
    /// Tokens to be prefilled without actual compute work.
    #[serde(default)]
    pub num_cached_tokens: u32,
    /// Tokens to be prefilled from local prefix cache.
    #[serde(default)]
    pub num_local_cached_tokens: u32,
    /// Tokens to be prefilled from external KV transfer.
    #[serde(default)]
    pub num_external_cached_tokens: u32,
    /// Prompt tokens newly admitted into the local prefix cache.
    #[serde(default)]
    pub num_cache_creation_tokens: u32,
}

/// Scheduler-side details for one engine iteration, which the frontend logs
/// one line each.
///
/// `scheduled_workload` is not decoded: Python hands it only to custom stat
/// loggers.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f9d1824a770f01e681fef8adb4b4bbe506e080aa/vllm/v1/metrics/stats.py#L198-L213>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulerIterationDetails {
    pub iteration_index: u64,
    pub num_ctx_requests: u64,
    pub num_ctx_tokens: u64,
    pub num_generation_requests: u64,
    pub num_generation_tokens: u64,
    /// GPU time, measured by the output worker; `None` where it does not time.
    #[serde(default)]
    pub elapsed_ms: Option<f64>,
    #[serde(default)]
    pub num_encoder_inputs: u64,
    #[serde(default)]
    pub num_encoder_output_tokens: u64,
    #[serde(default)]
    pub is_dummy: bool,
}

/// Stats associated with the scheduler.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/906a8c15d0beded2cca8b357877266c631340f74/vllm/v1/metrics/stats.py#L170-L196>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Number of requests in model execution batches.
    pub num_running_reqs: u64,
    /// Number of waiting requests not deferred.
    pub num_waiting_reqs: u64,
    /// Number of waiting requests deferred by transient constraints or a blocked status.
    #[serde(default)]
    pub num_skipped_waiting_reqs: u64,
    /// Requests preempted since the previous stats, counted at preemption time.
    #[serde(default)]
    pub num_preempted_reqs: u64,
    /// Waiting requests with an async KV load, by load stage.
    #[serde(default)]
    pub num_kv_fetch_reqs_by_stage: KvFetchReqsByStage,
    /// Internal DP load-balancing step counter.
    pub step_counter: u64,
    /// Internal DP load-balancing wave number.
    pub current_wave: u64,
    /// KV-cache usage. `1.0` means 100% usage.
    pub kv_cache_usage: f64,
    /// Local prefix cache statistics.
    pub prefix_cache_stats: PrefixCacheStats,
    /// External connector prefix cache statistics, when configured.
    pub connector_prefix_cache_stats: Option<PrefixCacheStats>,
    /// Sampled KV cache eviction events for residency metrics.
    pub kv_cache_eviction_events: Vec<KvCacheEvictionEvent>,
    /// Speculative decoding scheduler stats, when enabled.
    pub spec_decoding_stats: Option<SpecDecodingStats>,
    /// Connector-specific KV transfer stats.
    pub kv_connector_stats: Option<KvConnectorStats>,
    /// Connector-specific encoder cache transfer stats, passed through as the
    /// engine's connector reports them.
    #[serde(default)]
    pub ec_connector_stats: Option<OpaqueValue>,
    /// Iterations to report, with `--enable-logging-iteration-details`: timed on
    /// the GPU by the output worker, or untimed from EngineCore.
    #[serde(default)]
    pub iteration_records: Vec<SchedulerIterationDetails>,
    /// CUDA graph runtime stats when graph metrics are enabled.
    pub cudagraph_stats: Option<CudagraphStats>,
    /// Estimated MFU/performance stats, when enabled.
    pub perf_stats: Option<PerfStats>,
}

#[cfg(test)]
mod tests {
    use super::{KvFetchReqsByStage, SchedulerIterationDetails, SchedulerStats};

    #[test]
    fn scheduler_stats_kv_fetch_reqs_by_stage_defaults_and_decodes() {
        let mut payload = serde_json::json!({
            "num_running_reqs": 1, "num_waiting_reqs": 2,
            "step_counter": 0, "current_wave": 0, "kv_cache_usage": 0.5,
            "prefix_cache_stats": {
                "reset": false, "requests": 0, "queries": 0, "hits": 0,
                "preempted_requests": 0, "preempted_queries": 0, "preempted_hits": 0
            },
            "kv_cache_eviction_events": []
        });

        // Engines before vllm-project/vllm#58874 omit the field entirely.
        let wire = rmp_serde::to_vec_named(&payload).unwrap();
        let stats: SchedulerStats = rmp_serde::from_slice(&wire).unwrap();
        assert_eq!(
            stats.num_kv_fetch_reqs_by_stage,
            KvFetchReqsByStage::default()
        );

        payload["num_kv_fetch_reqs_by_stage"] =
            serde_json::json!({"waiting_to_start": 3, "in_progress": 1});
        let wire = rmp_serde::to_vec_named(&payload).unwrap();
        let stats: SchedulerStats = rmp_serde::from_slice(&wire).unwrap();
        assert_eq!(stats.num_kv_fetch_reqs_by_stage.counts(), [3, 1, 0]);
    }

    #[test]
    fn scheduler_stats_decodes_iteration_records() {
        let payload = serde_json::json!({
            "num_running_reqs": 1, "num_waiting_reqs": 0,
            "step_counter": 0, "current_wave": 0, "kv_cache_usage": 0.25,
            "prefix_cache_stats": {
                "reset": false, "requests": 0, "queries": 0, "hits": 0,
                "preempted_requests": 0, "preempted_queries": 0, "preempted_hits": 0
            },
            "kv_cache_eviction_events": [],
            "iteration_records": [
                {
                    "iteration_index": 7, "num_ctx_requests": 1, "num_ctx_tokens": 8,
                    "num_generation_requests": 2, "num_generation_tokens": 2,
                    "elapsed_ms": 1.5, "num_encoder_inputs": 1,
                    "num_encoder_output_tokens": 64, "is_dummy": false,
                    "scheduled_workload": {"num_prefill_tokens": 8}
                },
                {
                    "iteration_index": 8, "num_ctx_requests": 0, "num_ctx_tokens": 0,
                    "num_generation_requests": 0, "num_generation_tokens": 0,
                    "elapsed_ms": null, "num_encoder_inputs": 0,
                    "num_encoder_output_tokens": 0, "is_dummy": true,
                    "scheduled_workload": null
                }
            ],
            "queued_workload": {"num_waiting_tokens": 3}
        });

        let wire = rmp_serde::to_vec_named(&payload).unwrap();
        let stats: SchedulerStats = rmp_serde::from_slice(&wire).unwrap();
        assert_eq!(
            stats.iteration_records,
            [
                SchedulerIterationDetails {
                    iteration_index: 7,
                    num_ctx_requests: 1,
                    num_ctx_tokens: 8,
                    num_generation_requests: 2,
                    num_generation_tokens: 2,
                    elapsed_ms: Some(1.5),
                    num_encoder_inputs: 1,
                    num_encoder_output_tokens: 64,
                    is_dummy: false,
                },
                SchedulerIterationDetails {
                    iteration_index: 8,
                    is_dummy: true,
                    ..Default::default()
                },
            ]
        );
    }
}
