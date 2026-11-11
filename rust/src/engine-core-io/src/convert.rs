// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The scheduler's outputs as the frontend's wire types.
//!
//! What the scheduler reports per request is already in the wire types; this
//! fills them into an output, and converts the scheduler's own accumulators
//! (the prefix cache stats). Fields whose payloads stay with the binding
//! layer (logprobs, sampling masks, prompt logprobs, routed experts, and the
//! connectors' opaque params) are left unset for the binding to fill in.

use vllm_engine_core::kv_cache::PrefixCacheStats;
use vllm_engine_core::sched::scheduler::EngineCoreOutput;
use vllm_engine_core::sched::stats::{ConnectorPrefixCacheStats, SchedulerStats};
use vllm_engine_core_client::protocol::output as wire;
use vllm_engine_core_client::protocol::stats as wire_stats;

/// The output of one request, without the payloads the binding layer holds
/// (see the module docs).
pub fn engine_core_output(output: &EngineCoreOutput) -> wire::EngineCoreOutput {
    wire::EngineCoreOutput {
        request_id: output.request_id.clone(),
        new_token_ids: output.new_token_ids.iter().map(|&t| t as u32).collect(),
        finish_reason: output.finish_reason,
        stop_reason: output.stop_reason.clone(),
        events: output.events.clone(),
        prefill_stats: output.prefill_stats.map(Box::new),
        num_nans_in_logits: output.num_nans_in_logits as u32,
        spec_decode_metrics: output.spec_decode_metrics.clone().map(Box::new),
        error: output.error.clone().map(Box::new),
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
        kv_cache_eviction_events: stats.kv_cache_eviction_events.clone(),
        spec_decoding_stats: stats.spec_decoding_stats.clone(),
        kv_connector_stats: None,
        cudagraph_stats: None,
        perf_stats: None,
    }
}
