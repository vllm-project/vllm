use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

use vllm_metrics::{
    EngineLabels, EnginePositionLabels, LoraInfoLabels, SchedulerMetrics, WaitingReasonLabels,
};

use crate::protocol::stats::SchedulerStats;

const WAITING_REASON_CAPACITY: &str = "capacity";
const WAITING_REASON_DEFERRED: &str = "deferred";

/// Record the scheduler-stats-backed metrics for one engine at one point in
/// time.
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
    metrics.scheduler_running.get_or_create(&labels).set(stats.num_running_reqs);
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
    metrics.kv_cache_usage.get_or_create(&labels).set(stats.kv_cache_usage);

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

        for (position, accepted_tokens) in
            spec_decoding_stats.num_accepted_tokens_per_pos.iter().copied().enumerate()
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
        let kv_block_idle_before_evict_seconds =
            metrics.kv_block_idle_before_evict_seconds.get_or_create(&labels);
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

/// Exports `vllm:lora_requests_info` as a single series whose adapter labels are
/// the union across all engines in the replica.
#[derive(Default)]
pub(crate) struct LoraInfoExporter {
    per_engine: BTreeMap<u32, EngineLoraAdapters>,
    current: Option<LoraInfoLabels>,
}

#[derive(Default)]
struct EngineLoraAdapters {
    running: Vec<String>,
    waiting: Vec<String>,
}

impl LoraInfoExporter {
    pub(crate) fn record(
        &mut self,
        metrics: &SchedulerMetrics,
        engine: u32,
        stats: &SchedulerStats,
    ) {
        let active =
            !stats.running_lora_adapters.is_empty() || !stats.waiting_lora_adapters.is_empty();

        if !active && !self.per_engine.contains_key(&engine) {
            return;
        }

        if active {
            self.per_engine.insert(
                engine,
                EngineLoraAdapters {
                    running: stats.running_lora_adapters.keys().cloned().collect(),
                    waiting: stats.waiting_lora_adapters.keys().cloned().collect(),
                },
            );
        } else {
            self.per_engine.remove(&engine);
        }

        self.refresh(metrics);
    }

    fn refresh(&mut self, metrics: &SchedulerMetrics) {
        let mut running = BTreeSet::new();
        let mut waiting = BTreeSet::new();
        for adapters in self.per_engine.values() {
            running.extend(adapters.running.iter().map(String::as_str));
            waiting.extend(adapters.waiting.iter().map(String::as_str));
        }

        let next = (!running.is_empty() || !waiting.is_empty()).then(|| LoraInfoLabels {
            running_lora_adapters: join_names(&running),
            waiting_lora_adapters: join_names(&waiting),
        });

        if self.current != next
            && let Some(prev) = &self.current
        {
            metrics.lora_info.remove(prev);
        }

        // Python sets this gauge to the current time on every record.
        if let Some(labels) = &next {
            metrics.lora_info.get_or_create(labels).set(now_unix_secs());
        }

        self.current = next;
    }
}

fn join_names(names: &BTreeSet<&str>) -> String {
    let mut out = String::new();
    for name in names {
        if !out.is_empty() {
            out.push(',');
        }
        out.push_str(name);
    }
    out
}

fn now_unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use vllm_metrics::Metrics;

    use super::*;

    fn stats_with(running: &[&str], waiting: &[&str]) -> SchedulerStats {
        let mut stats = SchedulerStats::default();
        for name in running {
            stats.running_lora_adapters.insert((*name).to_string(), 1);
        }
        for name in waiting {
            stats.waiting_lora_adapters.insert((*name).to_string(), 1);
        }
        stats
    }

    /// The `lora_requests_info` series with the non-deterministic timestamp
    /// value replaced by `<ts>`, one line per series.
    fn lora_series(rendered: &str) -> String {
        rendered
            .lines()
            .filter(|l| l.starts_with("vllm:lora_requests_info{"))
            .map(|l| match l.rsplit_once("} ") {
                Some((labels, _value)) => format!("{labels}}} <ts>"),
                None => l.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn lora_info_unions_engines_clears_stale_and_drains() {
        let metrics = Metrics::new();
        let mut exporter = LoraInfoExporter::default();

        // No adapters: nothing emitted.
        exporter.record(&metrics.scheduler, 0, &stats_with(&[], &[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));

        // Engine 0: two running (sorted), one waiting.
        exporter.record(&metrics.scheduler, 0, &stats_with(&["b", "a"], &["c"]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="a,b",waiting_lora_adapters="c"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // Engine 1 adds "d": series is the union across engines.
        exporter.record(&metrics.scheduler, 1, &stats_with(&["d"], &[]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="a,b,d",waiting_lora_adapters="c"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // Engine 0 drains: only engine 1's "d" remains.
        exporter.record(&metrics.scheduler, 0, &stats_with(&[], &[]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="d",waiting_lora_adapters=""} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // Engine 1 drains: series removed entirely.
        exporter.record(&metrics.scheduler, 1, &stats_with(&[], &[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));
    }
}
