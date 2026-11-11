// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The scheduler's outputs as the frontend's wire types.
//!
//! What the scheduler reports per request is already in the wire types; this
//! fills them into an output, and converts the scheduler's own accumulators
//! (the prefix cache stats). Fields whose payloads stay with the binding
//! layer (logprobs, sampling masks, prompt logprobs, routed experts, and the
//! connectors' opaque params) are left unset for the binding to fill in.

use vllm_engine_core_protocol::output as wire;
use vllm_engine_core_protocol::stats as wire_stats;

use super::scheduler::EngineCoreOutput;
use super::stats::{ConnectorPrefixCacheStats, SchedulerStats};
use crate::kv_cache::PrefixCacheStats;

impl EngineCoreOutput {
    /// The output of this request, without the payloads the binding layer
    /// holds (see the module docs).
    pub fn to_wire(&self) -> wire::EngineCoreOutput {
        wire::EngineCoreOutput {
            request_id: self.request_id.clone(),
            new_token_ids: self.new_token_ids.iter().map(|&t| t as u32).collect(),
            finish_reason: self.finish_reason,
            stop_reason: self.stop_reason.clone(),
            events: self.events.clone(),
            prefill_stats: self.prefill_stats.map(Box::new),
            num_nans_in_logits: self.num_nans_in_logits as u32,
            spec_decode_metrics: self.spec_decode_metrics.clone().map(Box::new),
            error: self.error.clone().map(Box::new),
            watermarking: self.watermarking,
            ..Default::default()
        }
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

impl SchedulerStats {
    /// The scheduler's stats, without the ones Python objects hold
    /// (`kv_connector_stats`, `ec_connector_stats`, `cudagraph_stats`,
    /// `perf_stats`) or the engine adds (`iteration_details`). `step_counter`
    /// and `current_wave` are set only on the stats DP engines publish to the
    /// coordinator.
    pub fn to_wire(&self) -> wire_stats::SchedulerStats {
        let fetch = &self.num_kv_fetch_reqs_by_stage;
        wire_stats::SchedulerStats {
            num_running_reqs: self.num_running_reqs as u64,
            num_waiting_reqs: self.num_waiting_reqs as u64,
            num_skipped_waiting_reqs: self.num_skipped_waiting_reqs as u64,
            num_kv_fetch_reqs_by_stage: wire_stats::KvFetchReqsByStage {
                waiting_to_start: fetch.waiting_to_start as u64,
                in_progress: fetch.in_progress as u64,
                completed_waiting: fetch.completed_waiting as u64,
            },
            step_counter: 0,
            current_wave: 0,
            kv_cache_usage: self.kv_cache_usage,
            prefix_cache_stats: prefix_cache_stats(&self.prefix_cache_stats),
            connector_prefix_cache_stats: (self.connector_prefix_cache_stats.as_ref())
                .map(connector_prefix_cache_stats),
            kv_cache_eviction_events: self.kv_cache_eviction_events.clone(),
            spec_decoding_stats: self.spec_decoding_stats.clone(),
            kv_connector_stats: None,
            ec_connector_stats: None,
            iteration_details: None,
            cudagraph_stats: None,
            perf_stats: None,
        }
    }
}
