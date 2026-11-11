// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The scheduler's outputs as the frontend's wire types.
//!
//! Fields whose payloads stay with the binding layer (logprobs, sampling
//! masks, prompt logprobs, routed experts, and the connectors' opaque params)
//! are left unset here for the binding to fill in.

use vllm_engine_core::kv_cache::{KvCacheEvictionEvent, PrefixCacheStats};
use vllm_engine_core::sched::request::{
    EngineCoreEvent, EngineCoreEventType, FinishReason, PrefillStats, StopReason,
};
use vllm_engine_core::sched::scheduler::{EngineCoreOutput, RequestError};
use vllm_engine_core::sched::stats::{
    ConnectorPrefixCacheStats, RequestSpecDecodeMetrics, SchedulerStats, SpecDecodingStats,
};
use vllm_engine_core_client::protocol::output as wire;
use vllm_engine_core_client::protocol::stats as wire_stats;

fn finish_reason(reason: FinishReason) -> wire::EngineCoreFinishReason {
    match reason {
        FinishReason::Stop => wire::EngineCoreFinishReason::Stop,
        FinishReason::Length => wire::EngineCoreFinishReason::Length,
        FinishReason::Abort => wire::EngineCoreFinishReason::Abort,
        FinishReason::Error => wire::EngineCoreFinishReason::Error,
        FinishReason::Repetition => wire::EngineCoreFinishReason::Repetition,
    }
}

fn stop_reason(reason: &StopReason) -> wire::StopReason {
    match reason {
        StopReason::TokenId(token) => wire::StopReason::TokenId(*token as u32),
        StopReason::Str(text) => wire::StopReason::Text(text.clone()),
    }
}

fn event(event: &EngineCoreEvent) -> wire::EngineCoreEvent {
    wire::EngineCoreEvent {
        r#type: match event.r#type {
            EngineCoreEventType::Queued => wire::EngineCoreEventType::Queued,
            EngineCoreEventType::Scheduled => wire::EngineCoreEventType::Scheduled,
            EngineCoreEventType::Preempted => wire::EngineCoreEventType::Preempted,
        },
        timestamp: event.timestamp,
    }
}

fn prefill_stats(stats: &PrefillStats) -> wire_stats::PrefillStats {
    wire_stats::PrefillStats {
        num_prompt_tokens: stats.num_prompt_tokens as u32,
        num_computed_tokens: stats.num_computed_tokens as u32,
        num_cached_tokens: stats.num_cached_tokens as u32,
        num_local_cached_tokens: stats.num_local_cached_tokens as u32,
        num_external_cached_tokens: stats.num_external_cached_tokens as u32,
        num_cache_creation_tokens: stats.num_cache_creation_tokens as u32,
    }
}

fn spec_decode_metrics(metrics: &RequestSpecDecodeMetrics) -> wire::RequestSpecDecodeMetrics {
    let u64s = |values: &[usize]| values.iter().map(|&v| v as u64).collect();
    wire::RequestSpecDecodeMetrics {
        num_spec_tokens: metrics.num_spec_tokens as u64,
        histogram: u64s(&metrics.histogram),
        num_draft_tokens: metrics.num_draft_tokens as u64,
        per_step_accepted: u64s(&metrics.per_step_accepted),
        per_step_drafted: u64s(&metrics.per_step_drafted),
    }
}

fn request_error(error: &RequestError) -> wire::RequestError {
    wire::RequestError {
        code: error.code.clone(),
        message: error.message.clone(),
        retryable: false,
    }
}

/// The output of one request, without the payloads the binding layer holds
/// (see the module docs).
pub fn engine_core_output(output: &EngineCoreOutput) -> wire::EngineCoreOutput {
    wire::EngineCoreOutput {
        request_id: output.request_id.clone(),
        new_token_ids: output.new_token_ids.iter().map(|&t| t as u32).collect(),
        finish_reason: output.finish_reason.map(finish_reason),
        stop_reason: output.stop_reason.as_ref().map(stop_reason),
        events: output.events.as_ref().map(|events| events.iter().map(event).collect()),
        prefill_stats: output.prefill_stats.as_ref().map(|s| Box::new(prefill_stats(s))),
        num_nans_in_logits: output.num_nans_in_logits as u32,
        spec_decode_metrics: (output.spec_decode_metrics.as_ref())
            .map(|m| Box::new(spec_decode_metrics(m))),
        error: output.error.as_ref().map(|e| Box::new(request_error(e))),
        watermarking: output.watermarking,
        ..Default::default()
    }
}

fn prefix_cache_stats(stats: &PrefixCacheStats) -> wire_stats::PrefixCacheStats {
    wire_stats::PrefixCacheStats {
        base: wire_stats::BaseCacheStats {
            reset: stats.reset,
            requests: stats.requests as u64,
            queries: stats.queries as u64,
            hits: stats.hits as u64,
        },
        preempted_requests: stats.preempted_requests as u64,
        preempted_queries: stats.preempted_queries as u64,
        preempted_hits: stats.preempted_hits as u64,
        hits_by_source: Default::default(),
    }
}

fn connector_prefix_cache_stats(stats: &ConnectorPrefixCacheStats) -> wire_stats::PrefixCacheStats {
    let mut converted = prefix_cache_stats(&stats.stats);
    let by_source = &mut converted.hits_by_source;
    for (source, &hits) in &stats.hits_by_source {
        // The frontend reads these sources only; Python's `CacheHitSource`
        // has no others.
        let count = match source.as_str() {
            "device" => &mut by_source.device,
            "host" => &mut by_source.host,
            "p2p" => &mut by_source.p2p,
            "disk" => &mut by_source.disk,
            "external_unspecified" => &mut by_source.external_unspecified,
            _ => continue,
        };
        *count += hits as u64;
    }
    converted
}

fn kv_cache_eviction_event(event: &KvCacheEvictionEvent) -> wire_stats::KvCacheEvictionEvent {
    wire_stats::KvCacheEvictionEvent {
        lifetime_seconds: event.lifetime_seconds,
        idle_seconds: event.idle_seconds,
        reuse_gaps_seconds: event.reuse_gaps_seconds.clone(),
    }
}

fn spec_decoding_stats(stats: &SpecDecodingStats) -> wire_stats::SpecDecodingStats {
    wire_stats::SpecDecodingStats {
        num_spec_tokens: stats.num_spec_tokens as u64,
        num_drafts: stats.num_drafts as u64,
        num_draft_tokens: stats.num_draft_tokens as u64,
        num_accepted_tokens: stats.num_accepted_tokens as u64,
        num_accepted_tokens_per_pos: (stats.num_accepted_tokens_per_pos.iter())
            .map(|&n| n as u64)
            .collect(),
    }
}

/// The scheduler's stats, without the ones Python objects hold
/// (`kv_connector_stats`, `cudagraph_stats`, `perf_stats`). `step_counter`
/// and `current_wave` are set only on the stats DP engines publish to the
/// coordinator.
pub fn scheduler_stats(stats: &SchedulerStats) -> wire_stats::SchedulerStats {
    let fetch = &stats.num_kv_fetch_reqs_by_stage;
    wire_stats::SchedulerStats {
        num_running_reqs: stats.num_running_reqs as u64,
        num_waiting_reqs: stats.num_waiting_reqs as u64,
        num_skipped_waiting_reqs: stats.num_skipped_waiting_reqs as u64,
        num_kv_fetch_reqs_by_stage: wire_stats::KvFetchReqsByStage {
            waiting_to_start: fetch.waiting_to_start as u64,
            in_progress: fetch.in_progress as u64,
            completed_waiting: fetch.completed_waiting as u64,
        },
        step_counter: 0,
        current_wave: 0,
        kv_cache_usage: stats.kv_cache_usage,
        prefix_cache_stats: prefix_cache_stats(&stats.prefix_cache_stats),
        connector_prefix_cache_stats: (stats.connector_prefix_cache_stats.as_ref())
            .map(connector_prefix_cache_stats),
        kv_cache_eviction_events: (stats.kv_cache_eviction_events.iter())
            .map(kv_cache_eviction_event)
            .collect(),
        spec_decoding_stats: stats.spec_decoding_stats.as_ref().map(spec_decoding_stats),
        kv_connector_stats: None,
        cudagraph_stats: None,
        perf_stats: None,
    }
}
