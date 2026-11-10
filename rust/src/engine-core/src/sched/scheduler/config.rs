// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The scalars `Scheduler.__init__` derives from `VllmConfig`.

use crate::kv_cache::{KvCacheError, MambaCacheMode};
use crate::sched::request_queue::SchedulingPolicy;
use crate::sched::stats::SpecDecodeMetricsLevel;

/// Which Python scheduler class the scheduler behaves as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulerClass {
    /// `Scheduler`.
    Sync,
    /// `AsyncScheduler`: output placeholders are advanced at schedule time.
    Async {
        /// Steps between same-request decodes: `pipeline_parallel_size`,
        /// or 1 on XPU without `VLLM_XPU_PP_MICROBATCH`.
        decode_stagger: u64,
    },
}

/// `KVTransferConfig` as the scheduler reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvTransferConfig {
    pub is_kv_producer: bool,
    pub is_kv_consumer: bool,
    /// `kv_load_failure_policy == "recompute"`.
    pub recompute_kv_load_failures: bool,
}

/// The configuration of a [`super::Scheduler`]: every value `Scheduler`
/// reads from `VllmConfig` and its constructor arguments, resolved by the
/// caller (the binding layer reads them from the Python config objects).
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    // --- `scheduler_config` ---
    pub max_num_seqs: usize,
    pub max_num_active_seqs: Option<usize>,
    pub max_num_batched_tokens: usize,
    pub max_num_scheduled_tokens: Option<usize>,
    pub long_prefill_token_threshold: usize,
    pub long_prefill_token_threshold_adaptive: bool,
    pub enable_chunked_prefill: bool,
    pub disable_chunked_mm_input: bool,
    pub policy: SchedulingPolicy,
    pub async_scheduling: bool,
    pub scheduler_reserve_full_isl: bool,
    pub cache_aware_admission_window: usize,
    pub cache_aware_admission_threshold: f64,
    pub watermark: f64,
    pub max_num_inflight_kv_loads: Option<usize>,
    /// `max_async_kv_loads_per_step`; Python treats 0 like unset.
    pub max_async_kv_loads_per_step: Option<usize>,

    // --- `model_config` ---
    pub max_model_len: usize,
    /// 0 for pooling and diffusion models, which do not append a sampled
    /// token after prefill; 1 otherwise.
    pub num_sampled_tokens_per_step: usize,
    pub is_encoder_decoder: bool,
    /// `VllmConfig.is_mm_encoder_only`.
    pub is_mm_encoder_only: bool,
    pub uses_mrope: bool,
    pub return_sampling_mask: bool,

    // --- `cache_config` and constructor arguments ---
    /// The `block_size` constructor argument (the scheduler block size).
    pub block_size: usize,
    /// The `hash_block_size` constructor argument; defaults to `block_size`.
    pub hash_block_size: Option<usize>,
    /// `cache_config.block_size`, which the Mamba split aligns to.
    pub cache_block_size: usize,
    /// `cache_config.num_gpu_blocks`.
    pub num_gpu_blocks: usize,
    pub enable_prefix_caching: bool,
    pub mamba_cache_mode: MambaCacheMode,
    pub enable_mamba_shared_prefix_checkpoint: bool,

    // --- `parallel_config` and derived ---
    pub pipeline_parallel_size: usize,
    pub decode_context_parallel_size: usize,
    /// `VllmConfig.max_concurrent_batches`.
    pub max_concurrent_batches: usize,
    /// `VllmConfig.max_in_flight_tokens`.
    pub max_in_flight_tokens: Option<usize>,

    // --- speculative decoding ---
    /// `VllmConfig.num_speculative_tokens`.
    pub num_spec_tokens: usize,
    /// `VllmConfig.num_lookahead_tokens`.
    pub num_lookahead_tokens: usize,
    /// `VllmConfig.num_prefill_lookahead_tokens`.
    pub num_prefill_lookahead_tokens: usize,
    /// `VllmConfig.num_hash_lookahead_tokens`.
    pub num_hash_lookahead_tokens: usize,
    /// `speculative_config.use_eagle_block_drop()` (false without a
    /// speculative config). Cleared when lookahead hashes are used.
    pub use_eagle_block_drop: bool,
    /// `speculative_config.max_num_new_slots_for_drafting` (0 without one).
    pub draft_slots: usize,
    /// `build_dynamic_sd_schedule_lookup(...)` when
    /// `num_speculative_tokens_per_batch_size` is set.
    pub dynamic_sd_lookup: Option<Vec<usize>>,

    // --- multi-modal ---
    /// `MultiModalBudget.encoder_compute_budget` (0 without multi-modal
    /// inputs).
    pub max_num_encoder_input_tokens: usize,
    /// `MultiModalBudget.encoder_cache_size` (0 without multi-modal inputs).
    pub encoder_cache_size: usize,

    // --- LoRA ---
    /// `lora_config.max_loras`, `None` without a LoRA config.
    pub max_loras: Option<usize>,

    // --- KV transfer ---
    pub kv_transfer: Option<KvTransferConfig>,

    // --- observability ---
    pub log_stats: bool,
    /// `(log_stats and enable_logging_iteration_details) or profiler is not
    /// None`.
    pub enable_iteration_details: bool,
    /// `kv_cache_metrics_sample` when `kv_cache_metrics` is enabled.
    pub kv_cache_metrics_sample: Option<f64>,
    pub spec_decode_metrics_level: SpecDecodeMetricsLevel,
    pub enable_kv_cache_events: bool,
    /// `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`.
    pub kv_events_use_int_block_hashes: bool,
    /// Whether a `watermark_config` is set.
    pub report_watermarking: bool,
    pub include_finished_set: bool,
    pub scheduler_class: SchedulerClass,

    // --- not supported: must stay unset ---
    /// Whether an `ec_transfer_config` is set.
    pub ec_transfer: bool,
    /// `aux_output_config.enabled`.
    pub aux_output: bool,
    /// Whether `ec_manager_config` names a custom encoder cache manager.
    pub custom_encoder_cache_manager: bool,
}

impl SchedulerConfig {
    /// `create_scheduler()` defaults of `tests/v1/core/utils.py`.
    pub fn for_test(block_size: usize, max_model_len: usize) -> Self {
        Self {
            max_num_seqs: 16,
            max_num_active_seqs: None,
            max_num_batched_tokens: 8192,
            max_num_scheduled_tokens: None,
            long_prefill_token_threshold: 0,
            long_prefill_token_threshold_adaptive: false,
            enable_chunked_prefill: true,
            disable_chunked_mm_input: false,
            policy: SchedulingPolicy::Fcfs,
            async_scheduling: false,
            scheduler_reserve_full_isl: true,
            cache_aware_admission_window: 0,
            cache_aware_admission_threshold: 0.5,
            watermark: 0.0,
            max_num_inflight_kv_loads: None,
            max_async_kv_loads_per_step: None,
            max_model_len,
            num_sampled_tokens_per_step: 1,
            is_encoder_decoder: false,
            is_mm_encoder_only: false,
            uses_mrope: false,
            return_sampling_mask: false,
            block_size,
            hash_block_size: None,
            cache_block_size: block_size,
            num_gpu_blocks: 10000,
            enable_prefix_caching: false,
            mamba_cache_mode: MambaCacheMode::None,
            enable_mamba_shared_prefix_checkpoint: false,
            pipeline_parallel_size: 1,
            decode_context_parallel_size: 1,
            max_concurrent_batches: 1,
            max_in_flight_tokens: None,
            num_spec_tokens: 0,
            num_lookahead_tokens: 0,
            num_prefill_lookahead_tokens: 0,
            num_hash_lookahead_tokens: 0,
            use_eagle_block_drop: false,
            draft_slots: 0,
            dynamic_sd_lookup: None,
            max_num_encoder_input_tokens: 0,
            encoder_cache_size: 0,
            max_loras: None,
            kv_transfer: None,
            log_stats: false,
            enable_iteration_details: false,
            kv_cache_metrics_sample: None,
            spec_decode_metrics_level: SpecDecodeMetricsLevel::None,
            enable_kv_cache_events: false,
            kv_events_use_int_block_hashes: false,
            report_watermarking: false,
            include_finished_set: false,
            scheduler_class: SchedulerClass::Sync,
            ec_transfer: false,
            aux_output: false,
            custom_encoder_cache_manager: false,
        }
    }
}

/// Errors constructing a [`super::Scheduler`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SchedulerError {
    /// The configuration is invalid (`ValueError` / `AssertionError` at
    /// construction).
    #[error("{message}")]
    InvalidConfig { message: String },
    /// The configuration enables a feature this port does not implement.
    #[error("{message}")]
    Unsupported { message: String },
    #[error(transparent)]
    KvCache(#[from] KvCacheError),
}
