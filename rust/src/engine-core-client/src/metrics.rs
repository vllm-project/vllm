use std::collections::{BTreeSet, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

use vllm_metrics::{
    EngineLabels, EnginePositionLabels, LoraAdapterNames, LoraInfoLabels, LoraLoadedLabels,
    LoraLoadedLevel, SchedulerMetrics, WaitingReasonLabels,
};

use crate::protocol::notifications::LoraLoadEvent;
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

/// Exports `vllm:lora_requests_info` as a single series covering all LoRA
/// requests tracked by this client across every engine in the replica.
///
/// The engine's `SchedulerStats` never carries adapter names: the Python
/// frontend fills them in from per-request lifecycle events tracked by
/// `LoRARequestStates` in `vllm/v1/engine/output_processor.py`. The Rust
/// frontend mirrors that, deriving the sets from the request registry.
#[derive(Default)]
pub(crate) struct LoraInfoExporter {
    current: Option<LoraInfoLabels>,
}

impl LoraInfoExporter {
    pub(crate) fn update(
        &mut self,
        metrics: &SchedulerMetrics,
        running: BTreeSet<String>,
        waiting: BTreeSet<String>,
    ) {
        let next = (!running.is_empty() || !waiting.is_empty()).then_some(LoraInfoLabels {
            running_lora_adapters: LoraAdapterNames(running),
            waiting_lora_adapters: LoraAdapterNames(waiting),
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

/// Exports the loaded-LoRA-adapter gauges from
/// `EngineNotification::LoraLoadEvent` events.
///
/// Each event carries the complete current state for one engine, so the
/// exporter diffs against the previously emitted series to drop adapters
/// that were evicted since the last event.
#[derive(Default)]
pub(crate) struct LoraLoadedExporter {
    current: HashMap<u32, BTreeSet<LoraLoadedLabels>>,
}

impl LoraLoadedExporter {
    pub(crate) fn update(
        &mut self,
        metrics: &SchedulerMetrics,
        model_name: impl Into<String>,
        engine: u32,
        event: &LoraLoadEvent,
    ) {
        let model_name = model_name.into();

        let engine_labels = EngineLabels {
            model_name: model_name.clone(),
            engine,
        };
        metrics
            .lora_gpu_adapters
            .get_or_create(&engine_labels)
            .set(event.gpu_adapters.len() as u64);
        metrics
            .lora_cpu_adapters
            .get_or_create(&engine_labels)
            .set(event.cpu_adapters.len() as u64);

        let gpu: BTreeSet<&str> = event.gpu_adapters.iter().map(String::as_str).collect();
        let pinned: BTreeSet<&str> = event.pinned_adapters.iter().map(String::as_str).collect();
        let next: BTreeSet<LoraLoadedLabels> = event
            .cpu_adapters
            .iter()
            .map(|adapter_name| LoraLoadedLabels {
                model_name: model_name.clone(),
                engine,
                adapter_name: adapter_name.clone(),
                level: if gpu.contains(adapter_name.as_str()) {
                    LoraLoadedLevel::Gpu
                } else {
                    LoraLoadedLevel::Cpu
                },
                pinned: pinned.contains(adapter_name.as_str()),
            })
            .collect();

        let prev = self.current.entry(engine).or_default();
        for stale in prev.difference(&next) {
            metrics.lora_adapter_loaded.remove(stale);
        }
        for labels in &next {
            metrics.lora_adapter_loaded.get_or_create(labels).set(1);
        }
        *prev = next;
    }
}

fn now_unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use expect_test::expect;
    use vllm_metrics::Metrics;

    use crate::metrics::LoraInfoExporter;

    fn names(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|name| (*name).to_string()).collect()
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
    fn lora_loaded_emits_levels_and_clears_stale() {
        use crate::metrics::LoraLoadedExporter;
        use crate::protocol::notifications::LoraLoadEvent;

        let metrics = Metrics::new();
        let mut exporter = LoraLoadedExporter::default();

        // Family iteration order is not deterministic; sort for snapshots.
        let loaded_series = |rendered: &str| {
            let mut lines = rendered
                .lines()
                .filter(|l| l.starts_with("vllm:lora_adapter_loaded{"))
                .collect::<Vec<_>>();
            lines.sort_unstable();
            lines.join("\n")
        };

        // "a" active on GPU and pinned, "b" only in the CPU cache.
        exporter.update(
            &metrics.scheduler,
            "model",
            0,
            &LoraLoadEvent {
                gpu_adapters: vec!["a".to_string()],
                cpu_adapters: vec!["a".to_string(), "b".to_string()],
                pinned_adapters: vec!["a".to_string()],
            },
        );
        let rendered = metrics.render().unwrap();
        expect![[r#"
            vllm:lora_adapter_loaded{model_name="model",engine="0",adapter_name="a",level="gpu",pinned="true"} 1
            vllm:lora_adapter_loaded{model_name="model",engine="0",adapter_name="b",level="cpu",pinned="false"} 1"#]]
        .assert_eq(&loaded_series(&rendered));
        assert!(
            rendered
                .contains(r#"vllm:num_gpu_loaded_lora_adapters{model_name="model",engine="0"} 1"#)
        );
        assert!(
            rendered
                .contains(r#"vllm:num_cpu_loaded_lora_adapters{model_name="model",engine="0"} 2"#)
        );

        // "a" evicted from GPU to CPU (and unpinned), "b" evicted entirely:
        // the stale series must disappear.
        exporter.update(
            &metrics.scheduler,
            "model",
            0,
            &LoraLoadEvent {
                gpu_adapters: vec![],
                cpu_adapters: vec!["a".to_string()],
                pinned_adapters: vec![],
            },
        );
        let rendered = metrics.render().unwrap();
        expect![[r#"
            vllm:lora_adapter_loaded{model_name="model",engine="0",adapter_name="a",level="cpu",pinned="false"} 1"#]]
        .assert_eq(&loaded_series(&rendered));
        assert!(
            rendered
                .contains(r#"vllm:num_gpu_loaded_lora_adapters{model_name="model",engine="0"} 0"#)
        );
        assert!(
            rendered
                .contains(r#"vllm:num_cpu_loaded_lora_adapters{model_name="model",engine="0"} 1"#)
        );
    }

    #[test]
    fn lora_info_emits_clears_stale_and_drains() {
        let metrics = Metrics::new();
        let mut exporter = LoraInfoExporter::default();

        // No adapters: nothing emitted.
        exporter.update(&metrics.scheduler, names(&[]), names(&[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));

        // Two running (sorted), one waiting.
        exporter.update(&metrics.scheduler, names(&["b", "a"]), names(&["c"]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="a,b",waiting_lora_adapters="c"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // "c" gets scheduled and "d" arrives: the stale series is replaced.
        exporter.update(&metrics.scheduler, names(&["a", "b", "c"]), names(&["d"]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="a,b,c",waiting_lora_adapters="d"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // Everything but "d" finishes.
        exporter.update(&metrics.scheduler, names(&["d"]), names(&[]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="d",waiting_lora_adapters=""} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // All requests done: series removed entirely.
        exporter.update(&metrics.scheduler, names(&[]), names(&[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));
    }
}
