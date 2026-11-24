// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Per-engine state that does not come from `SchedulerStats`: the sleep
//! state, LoRA info and cache config info.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

use itertools::Itertools as _;
use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue, LabelValueEncoder};
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;

use super::SchedulerMetrics;
use crate::{EngineLabels, F64Gauge, U64Gauge};

/// Labels for `vllm:engine_sleep_state`: one series per engine and state,
/// `awake`, `weights_offloaded` (sleep level 1) or `discard_all` (level 2).
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineSleepStateLabels {
    pub model_name: String,
    pub engine: u32,
    pub sleep_state: &'static str,
}

/// Adapter names encoded as a deterministic comma-joined Prometheus label value.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct LoraAdapterNames(pub BTreeSet<String>);

impl EncodeLabelValue for LoraAdapterNames {
    fn encode(&self, encoder: &mut LabelValueEncoder) -> Result<(), std::fmt::Error> {
        EncodeLabelValue::encode(&self.0.iter().join(","), encoder)
    }
}

/// Labels for `vllm:lora_requests_info`. Field order is the exposition order,
/// which matches Python's (sorted by name).
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct LoraInfoLabels {
    /// The engine's `max_loras`, constant for the server's lifetime.
    pub max_lora: u32,
    pub running_lora_adapters: LoraAdapterNames,
    pub waiting_lora_adapters: LoraAdapterNames,
}

/// Per-engine state that does not come from `SchedulerStats`.
pub struct EngineStateMetrics {
    /// `vllm:engine_sleep_state`, set by the engine client's sleep and wake
    /// calls rather than from `SchedulerStats`.
    pub engine_sleep_state: Family<EngineSleepStateLabels, U64Gauge>,

    /// `vllm:lora_requests_info`. Value is the emit-time unix timestamp in
    /// seconds.
    pub lora_info: Family<LoraInfoLabels, F64Gauge>,

    /// `vllm:cache_config_info`, one series per engine, always `1`. Labeled
    /// with every `CacheConfig.metrics_info()` field plus `engine`.
    pub cache_config_info: Family<Vec<(String, String)>, F64Gauge>,
}

impl EngineStateMetrics {
    pub(super) fn register(registry: &mut Registry) -> Self {
        let engine_sleep_state = Family::default();
        registry.register(
            "vllm:engine_sleep_state",
            "Engine sleep state; awake = 0 means engine is sleeping; \
             awake = 1 means engine is awake; \
             weights_offloaded = 1 means sleep level 1; \
             discard_all = 1 means sleep level 2.",
            engine_sleep_state.clone(),
        );

        let lora_info = Family::default();
        registry.register(
            "vllm:lora_requests_info",
            "Running stats on lora requests.",
            lora_info.clone(),
        );

        let cache_config_info = Family::default();
        registry.register(
            "vllm:cache_config_info",
            "Information of the LLMEngine CacheConfig",
            cache_config_info.clone(),
        );

        Self {
            engine_sleep_state,
            lora_info,
            cache_config_info,
        }
    }

    pub(super) fn resolve(&self, labels: &EngineLabels) -> EngineSleepStateHandles {
        EngineSleepStateHandles(SLEEP_STATES.map(|sleep_state| {
            self.engine_sleep_state.get_or_create_owned(&EngineSleepStateLabels {
                model_name: labels.model_name.clone(),
                engine: labels.engine,
                sleep_state,
            })
        }))
    }

    /// Export one engine's cache config as `vllm:cache_config_info`, which
    /// Python also creates with its stat logger.
    pub fn record_cache_config_info(&self, engine: u32, info: &BTreeMap<String, String>) {
        let mut labels = info.clone();
        labels.insert("engine".to_string(), engine.to_string());
        self.cache_config_info.get_or_create(&labels.into_iter().collect()).set(1.0);
    }
}

/// `sleep_state` label values of `vllm:engine_sleep_state`, in the order of
/// the values [`EngineSleepStateHandles::record`] takes.
pub const SLEEP_STATES: [&str; 3] = ["awake", "weights_offloaded", "discard_all"];

/// One handle per [`SLEEP_STATES`] entry.
pub struct EngineSleepStateHandles([U64Gauge; 3]);

impl EngineSleepStateHandles {
    /// Record the engine's sleep state, one gauge value per [`SLEEP_STATES`]
    /// entry.
    pub fn record(&self, values: [u64; 3]) {
        for (gauge, value) in self.0.iter().zip(values) {
            gauge.set(value);
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
pub struct LoraInfoExporter {
    /// The engines' `max_loras`, reported as the `max_lora` label.
    max_lora: u32,
    current: Option<LoraInfoLabels>,
}

impl LoraInfoExporter {
    pub fn new(max_lora: u32) -> Self {
        Self {
            max_lora,
            current: None,
        }
    }

    pub fn update(
        &mut self,
        metrics: &SchedulerMetrics,
        running: BTreeSet<String>,
        waiting: BTreeSet<String>,
    ) {
        let next = (!running.is_empty() || !waiting.is_empty()).then_some(LoraInfoLabels {
            max_lora: self.max_lora,
            running_lora_adapters: LoraAdapterNames(running),
            waiting_lora_adapters: LoraAdapterNames(waiting),
        });

        if self.current != next
            && let Some(prev) = &self.current
        {
            metrics.engine_state.lora_info.remove(prev);
        }

        // Python sets this gauge to the current time on every record.
        if let Some(labels) = &next {
            metrics.engine_state.lora_info.get_or_create(labels).set(now_unix_secs());
        }

        self.current = next;
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

    use super::LoraInfoExporter;
    use crate::Metrics;

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
    fn lora_info_emits_clears_stale_and_drains() {
        let metrics = Metrics::new();
        let mut exporter = LoraInfoExporter::new(4);

        // No adapters: nothing emitted.
        exporter.update(&metrics.scheduler, names(&[]), names(&[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));

        // Two running (sorted), one waiting.
        exporter.update(&metrics.scheduler, names(&["b", "a"]), names(&["c"]));
        expect![[
            r#"vllm:lora_requests_info{max_lora="4",running_lora_adapters="a,b",waiting_lora_adapters="c"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // "c" gets scheduled and "d" arrives: the stale series is replaced.
        exporter.update(&metrics.scheduler, names(&["a", "b", "c"]), names(&["d"]));
        expect![[
            r#"vllm:lora_requests_info{max_lora="4",running_lora_adapters="a,b,c",waiting_lora_adapters="d"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // Everything but "d" finishes.
        exporter.update(&metrics.scheduler, names(&["d"]), names(&[]));
        expect![[
            r#"vllm:lora_requests_info{max_lora="4",running_lora_adapters="d",waiting_lora_adapters=""} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // All requests done: series removed entirely.
        exporter.update(&metrics.scheduler, names(&[]), names(&[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));
    }
}
