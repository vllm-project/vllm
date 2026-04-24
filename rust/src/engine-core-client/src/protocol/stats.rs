use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::protocol::OpaqueValue;

/// Stores cache hit statistics.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L18-L32>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BaseCacheStats {
    /// Whether the cache was reset.
    pub reset: bool,
    /// The number of requests in this update.
    pub requests: u64,
    /// The number of queries in these requests.
    pub queries: u64,
    /// The number of hits in these requests.
    pub hits: u64,
}

/// Stores prefix cache hit statistics.
/// - `reset`: Whether `reset_prefix_cache` was invoked.
/// - `queries`: Refers to the number of tokens that were queried.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L114-L143>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PrefixCacheStats {
    /// Embedded base cache counters and reset flag.
    #[serde(flatten)]
    pub base: BaseCacheStats,
    /// The number of previously preempted requests in this update.
    pub preempted_requests: u64,
    /// The `queries` number for preempted requests.
    pub preempted_queries: u64,
    /// The `hits` number for preempted requests.
    pub preempted_hits: u64,
}

/// Single KV cache block eviction sample.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L161-L167>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KvCacheEvictionEvent {
    /// Lifetime from allocation to eviction.
    pub lifetime_seconds: f64,
    /// Idle time observed before eviction.
    pub idle_seconds: f64,
    /// Time gaps between consecutive accesses before eviction.
    pub reuse_gaps_seconds: Vec<f64>,
}

/// Per-step iteration decoding stats from scheduler.
///
/// Each scheduler step, statistics on spec decoding performance are aggregated
/// across requests by the scheduler and returned to the frontend in
/// `EngineCoreOutputs -> SchedulerStats`.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/spec_decode/metrics.py#L16-L44>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecDecodingStats {
    /// Configured speculative token count for this scheduler.
    pub num_spec_tokens: u64,
    /// Number of drafted speculative decoding attempts.
    pub num_drafts: u64,
    /// Number of drafted tokens.
    pub num_draft_tokens: u64,
    /// Number of accepted drafted tokens.
    pub num_accepted_tokens: u64,
    /// Accepted drafted tokens counted by draft position.
    pub num_accepted_tokens_per_pos: Vec<u64>,
}

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
}

/// Stats for debugging the metrics calculation.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/perf.py#L46-L55>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DebugPerfStats {
    /// Time spent calculating these stats.
    pub calc_duration: f64,
    /// Number of prefill requests included in the sampled batch.
    pub num_prefill_requests: u64,
    /// Number of decode requests included in the sampled batch.
    pub num_decode_requests: u64,
    /// Optional execution-context breakdown used for debugging.
    pub context_breakdown: Option<BTreeMap<String, u64>>,
    /// Optional per-component FLOPs breakdown.
    pub num_flops_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
    /// Optional per-component memory-read breakdown.
    pub num_read_bytes_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
    /// Optional per-component memory-write breakdown.
    pub num_write_bytes_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
}

/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/perf.py#L58-L63>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerfStats {
    /// Estimated floating point operations per GPU.
    pub num_flops_per_gpu: u64,
    /// Estimated bytes read from memory per GPU.
    pub num_read_bytes_per_gpu: u64,
    /// Estimated bytes written to memory per GPU.
    pub num_write_bytes_per_gpu: u64,
    /// Optional debug-only perf derivation details.
    pub debug_stats: Option<DebugPerfStats>,
}

/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/compilation/cuda_graph.py#L28-L33>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CudagraphStat {
    /// Number of real tokens in the captured batch before padding.
    pub num_unpadded_tokens: u64,
    /// Number of padded tokens in the captured batch.
    pub num_padded_tokens: u64,
    /// Number of padding positions added for capture/runtime shape alignment.
    pub num_paddings: u64,
    /// Runtime mode string associated with this CUDA graph sample.
    pub runtime_mode: String,
}

/// Stats associated with the scheduler.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L170-L197>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Number of requests in model execution batches.
    pub num_running_reqs: u64,
    /// Number of requests waiting to be processed.
    pub num_waiting_reqs: u64,
    /// Internal DP load-balancing step counter.
    pub step_counter: u64,
    /// Internal DP load-balancing wave number.
    pub current_wave: u64,
    /// KV-cache usage. `1.0` means 100% usage.
    pub kv_cache_usage: f64,
    /// Encoder cache usage fraction.
    pub encoder_cache_usage: f64,
    /// Local prefix cache statistics.
    pub prefix_cache_stats: PrefixCacheStats,
    /// External connector prefix cache statistics, when configured.
    pub connector_prefix_cache_stats: Option<PrefixCacheStats>,
    /// Sampled KV cache eviction events for residency metrics.
    pub kv_cache_eviction_events: Vec<KvCacheEvictionEvent>,
    /// Speculative decoding scheduler stats, when enabled.
    pub spec_decoding_stats: Option<SpecDecodingStats>,
    /// Connector-specific KV transfer stats, kept opaque for now.
    pub kv_connector_stats: Option<BTreeMap<String, OpaqueValue>>,
    /// Waiting request counts per LoRA adapter.
    pub waiting_lora_adapters: BTreeMap<String, u64>,
    /// Running request counts per LoRA adapter.
    pub running_lora_adapters: BTreeMap<String, u64>,
    /// CUDA graph runtime stats when graph metrics are enabled.
    pub cudagraph_stats: Option<CudagraphStat>,
    /// Estimated MFU/performance stats, when enabled.
    pub perf_stats: Option<PerfStats>,
}
