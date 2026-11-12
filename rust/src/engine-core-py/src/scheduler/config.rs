// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The Rust `SchedulerConfig` from the dict of resolved values that
//! `vllm.v1.core.sched.rust_scheduler` derives from `VllmConfig`, as
//! `Scheduler.__init__` does.

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use vllm_engine_core::kv_cache::MambaCacheMode;
use vllm_engine_core::sched::request_queue::SchedulingPolicy;
use vllm_engine_core::sched::scheduler::{KvTransferConfig, SchedulerClass, SchedulerConfig};
use vllm_engine_core::sched::stats::SpecDecodeMetricsLevel;

struct Values<'a, 'py>(&'a Bound<'py, PyDict>);

impl<'py> Values<'_, 'py> {
    fn get<T: for<'a> FromPyObject<'a, 'py>>(&self, key: &str) -> PyResult<T> {
        let value = self
            .0
            .get_item(key)?
            .ok_or_else(|| PyKeyError::new_err(format!("scheduler config is missing {key:?}")))?;
        value.extract().map_err(Into::into)
    }
}

pub fn scheduler_config(config: &Bound<'_, PyDict>) -> PyResult<SchedulerConfig> {
    let v = Values(config);
    let policy: String = v.get("policy")?;
    let policy = match policy.as_str() {
        "fcfs" => SchedulingPolicy::Fcfs,
        "priority" => SchedulingPolicy::Priority,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown scheduling policy: {policy}"
            )));
        }
    };
    let mamba_cache_mode: String = v.get("mamba_cache_mode")?;
    let mamba_cache_mode = match mamba_cache_mode.as_str() {
        "none" => MambaCacheMode::None,
        "all" => MambaCacheMode::All,
        "align" => MambaCacheMode::Align,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown mamba_cache_mode: {mamba_cache_mode}"
            )));
        }
    };
    let spec_decode_metrics_level: String = v.get("spec_decode_metrics_level")?;
    let spec_decode_metrics_level = match spec_decode_metrics_level.as_str() {
        "none" => SpecDecodeMetricsLevel::None,
        "summary" => SpecDecodeMetricsLevel::Summary,
        "detailed" => SpecDecodeMetricsLevel::Detailed,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown per_request_spec_decode_metrics: {spec_decode_metrics_level}"
            )));
        }
    };
    let kv_transfer: Option<(bool, bool, bool)> = v.get("kv_transfer")?;
    let decode_stagger: Option<u64> = v.get("decode_stagger")?;
    Ok(SchedulerConfig {
        max_num_seqs: v.get("max_num_seqs")?,
        max_num_active_seqs: v.get("max_num_active_seqs")?,
        max_num_batched_tokens: v.get("max_num_batched_tokens")?,
        max_num_scheduled_tokens: v.get("max_num_scheduled_tokens")?,
        long_prefill_token_threshold: v.get("long_prefill_token_threshold")?,
        long_prefill_token_threshold_adaptive: v.get("long_prefill_token_threshold_adaptive")?,
        enable_chunked_prefill: v.get("enable_chunked_prefill")?,
        disable_chunked_mm_input: v.get("disable_chunked_mm_input")?,
        policy,
        async_scheduling: v.get("async_scheduling")?,
        scheduler_reserve_full_isl: v.get("scheduler_reserve_full_isl")?,
        cache_aware_admission_window: v.get("cache_aware_admission_window")?,
        cache_aware_admission_threshold: v.get("cache_aware_admission_threshold")?,
        watermark: v.get("watermark")?,
        max_num_inflight_kv_loads: v.get("max_num_inflight_kv_loads")?,
        max_async_kv_loads_per_step: v.get("max_async_kv_loads_per_step")?,
        max_model_len: v.get("max_model_len")?,
        num_sampled_tokens_per_step: v.get("num_sampled_tokens_per_step")?,
        is_encoder_decoder: v.get("is_encoder_decoder")?,
        is_mm_encoder_only: v.get("is_mm_encoder_only")?,
        uses_mrope: v.get("uses_mrope")?,
        return_sampling_mask: v.get("return_sampling_mask")?,
        block_size: v.get("block_size")?,
        hash_block_size: v.get("hash_block_size")?,
        cache_block_size: v.get("cache_block_size")?,
        num_gpu_blocks: v.get("num_gpu_blocks")?,
        enable_prefix_caching: v.get("enable_prefix_caching")?,
        mamba_cache_mode,
        enable_mamba_shared_prefix_checkpoint: v.get("enable_mamba_shared_prefix_checkpoint")?,
        pipeline_parallel_size: v.get("pipeline_parallel_size")?,
        decode_context_parallel_size: v.get("decode_context_parallel_size")?,
        max_concurrent_batches: v.get("max_concurrent_batches")?,
        max_in_flight_tokens: v.get("max_in_flight_tokens")?,
        num_spec_tokens: v.get("num_spec_tokens")?,
        num_lookahead_tokens: v.get("num_lookahead_tokens")?,
        num_prefill_lookahead_tokens: v.get("num_prefill_lookahead_tokens")?,
        num_hash_lookahead_tokens: v.get("num_hash_lookahead_tokens")?,
        use_eagle_block_drop: v.get("use_eagle_block_drop")?,
        draft_slots: v.get("draft_slots")?,
        dynamic_sd_lookup: v.get("dynamic_sd_lookup")?,
        max_num_encoder_input_tokens: v.get("max_num_encoder_input_tokens")?,
        encoder_cache_size: v.get("encoder_cache_size")?,
        max_loras: v.get("max_loras")?,
        kv_transfer: kv_transfer.map(
            |(is_kv_producer, is_kv_consumer, recompute_kv_load_failures)| KvTransferConfig {
                is_kv_producer,
                is_kv_consumer,
                recompute_kv_load_failures,
            },
        ),
        log_stats: v.get("log_stats")?,
        enable_iteration_details: v.get("enable_iteration_details")?,
        kv_cache_metrics_sample: v.get("kv_cache_metrics_sample")?,
        spec_decode_metrics_level,
        enable_kv_cache_events: v.get("enable_kv_cache_events")?,
        kv_events_use_int_block_hashes: v.get("kv_events_use_int_block_hashes")?,
        report_watermarking: v.get("report_watermarking")?,
        include_finished_set: v.get("include_finished_set")?,
        scheduler_class: match decode_stagger {
            None => SchedulerClass::Sync,
            Some(decode_stagger) => SchedulerClass::Async { decode_stagger },
        },
        ec_transfer: v.get("ec_transfer")?,
        aux_output: v.get("aux_output")?,
        custom_encoder_cache_manager: v.get("custom_encoder_cache_manager")?,
    })
}
