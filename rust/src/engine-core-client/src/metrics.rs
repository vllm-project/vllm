use vllm_metrics::{EngineLabels, Metrics};

use crate::protocol::stats::SchedulerStats;

/// Record the scheduler gauges for one engine at one point in time.
pub(crate) fn record_scheduler_stats(
    metrics: &Metrics,
    model_name: impl Into<String>,
    engine: u32,
    stats: &SchedulerStats,
) {
    let labels = EngineLabels {
        model_name: model_name.into(),
        engine,
    };
    metrics
        .scheduler_running
        .get_or_create(&labels)
        .set(stats.num_running_reqs);
    metrics
        .scheduler_waiting
        .get_or_create(&labels)
        .set(stats.num_waiting_reqs);
    metrics
        .kv_cache_usage
        .get_or_create(&labels)
        .set(stats.kv_cache_usage);
}
