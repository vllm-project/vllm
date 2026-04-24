use vllm_metrics::{EngineLabels, EnginePositionLabels, SchedulerMetrics, WaitingReasonLabels};

use crate::protocol::stats::SchedulerStats;

const WAITING_REASON_CAPACITY: &str = "capacity";
const WAITING_REASON_DEFERRED: &str = "deferred";

/// Record the scheduler-stats-backed metrics for one engine at one point in time.
pub(crate) fn record_scheduler_stats(
    metrics: &SchedulerMetrics,
    model_name: impl Into<String>,
    engine: u32,
    stats: &SchedulerStats,
) {
    let model_name = model_name.into();
    let labels = EngineLabels {
        model_name: model_name.clone(),
        engine,
    };

    // Scheduler state gauges.
    metrics
        .scheduler_running
        .get_or_create(&labels)
        .set(stats.num_running_reqs);
    metrics
        .scheduler_waiting
        .get_or_create(&labels)
        .set(stats.num_waiting_reqs + stats.num_skipped_waiting_reqs);
    metrics
        .scheduler_waiting_by_reason
        .get_or_create(&WaitingReasonLabels {
            model_name: model_name.clone(),
            engine,
            reason: WAITING_REASON_CAPACITY,
        })
        .set(stats.num_waiting_reqs);
    metrics
        .scheduler_waiting_by_reason
        .get_or_create(&WaitingReasonLabels {
            model_name: model_name.clone(),
            engine,
            reason: WAITING_REASON_DEFERRED,
        })
        .set(stats.num_skipped_waiting_reqs);
    metrics
        .kv_cache_usage
        .get_or_create(&labels)
        .set(stats.kv_cache_usage);

    // Prefix-cache counters, including the connector-backed external cache path.
    metrics
        .prefix_cache_queries
        .get_or_create(&labels)
        .inc_by(stats.prefix_cache_stats.base.queries);
    metrics
        .prefix_cache_hits
        .get_or_create(&labels)
        .inc_by(stats.prefix_cache_stats.base.hits);

    if let Some(connector_prefix_cache_stats) = &stats.connector_prefix_cache_stats {
        metrics
            .external_prefix_cache_queries
            .get_or_create(&labels)
            .inc_by(connector_prefix_cache_stats.base.queries);
        metrics
            .external_prefix_cache_hits
            .get_or_create(&labels)
            .inc_by(connector_prefix_cache_stats.base.hits);
    }

    // Speculative decoding counters.
    if let Some(spec_decoding_stats) = &stats.spec_decoding_stats {
        metrics
            .spec_decode_num_drafts
            .get_or_create(&labels)
            .inc_by(spec_decoding_stats.num_drafts);
        metrics
            .spec_decode_num_draft_tokens
            .get_or_create(&labels)
            .inc_by(spec_decoding_stats.num_draft_tokens);
        metrics
            .spec_decode_num_accepted_tokens
            .get_or_create(&labels)
            .inc_by(spec_decoding_stats.num_accepted_tokens);

        for (position, accepted_tokens) in spec_decoding_stats
            .num_accepted_tokens_per_pos
            .iter()
            .copied()
            .enumerate()
        {
            metrics
                .spec_decode_num_accepted_tokens_per_pos
                .get_or_create(&EnginePositionLabels {
                    model_name: model_name.clone(),
                    engine,
                    position: position as u32,
                })
                .inc_by(accepted_tokens);
        }
    }

    // Per-engine performance / MFU counters.
    if let Some(perf_stats) = &stats.perf_stats
        && (perf_stats.num_flops_per_gpu != 0
            || perf_stats.num_read_bytes_per_gpu != 0
            || perf_stats.num_write_bytes_per_gpu != 0)
    {
        metrics
            .estimated_flops_per_gpu
            .get_or_create(&labels)
            .inc_by(perf_stats.num_flops_per_gpu);
        metrics
            .estimated_read_bytes_per_gpu
            .get_or_create(&labels)
            .inc_by(perf_stats.num_read_bytes_per_gpu);
        metrics
            .estimated_write_bytes_per_gpu
            .get_or_create(&labels)
            .inc_by(perf_stats.num_write_bytes_per_gpu);
    }

    // Sampled KV-cache residency histograms.
    if !stats.kv_cache_eviction_events.is_empty() {
        let kv_block_lifetime_seconds = metrics.kv_block_lifetime_seconds.get_or_create(&labels);
        let kv_block_idle_before_evict_seconds = metrics
            .kv_block_idle_before_evict_seconds
            .get_or_create(&labels);
        let kv_block_reuse_gap_seconds = metrics.kv_block_reuse_gap_seconds.get_or_create(&labels);

        for event in &stats.kv_cache_eviction_events {
            kv_block_lifetime_seconds.observe(event.lifetime_seconds);
            kv_block_idle_before_evict_seconds.observe(event.idle_seconds);
            for reuse_gap_seconds in &event.reuse_gaps_seconds {
                kv_block_reuse_gap_seconds.observe(*reuse_gap_seconds);
            }
        }
    }
}
