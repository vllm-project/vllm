// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};

use crate::{EngineLabels, U64Counter, U64Gauge};

/// SimpleCPUOffload connector telemetry.
///
/// Unlike the NIXL, Mooncake and UMBP payloads (one field per counter or
/// sample list), Python sends the offloading connectors' self-describing
/// envelope (also `OffloadingConnectorStats`): `{"types": {name: kind},
/// "data": {name: {label_values: value}}}`, keyed by Prometheus metric name
/// and then by the label values as a tuple (`()` when unlabeled). The kind
/// of each metric is fixed by its name, so only `data` is read, and the
/// fixed-size array keys check each metric's label count when decoding.
///
/// Original Python definition: `SimpleCPUOffloadStats` in
/// `vllm/v1/simple_kv_offload/metrics.py`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SimpleCpuOffloadStats {
    pub data: SimpleCpuOffloadData,
}

/// SimpleCPUOffload samples of one interval, keyed by metric name and then
/// by the metric's label values (a Python tuple, empty when unlabeled).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SimpleCpuOffloadData {
    /// Store admission counter increments by `outcome`.
    #[serde(default, rename = "vllm:simple_kv_offload_save_outcomes_total")]
    pub save_outcomes: BTreeMap<[String; 1], u64>,
    /// Counter increment of blocks loaded back to GPU.
    #[serde(default, rename = "vllm:simple_kv_offload_load_blocks_total")]
    pub load_blocks: BTreeMap<[String; 0], u64>,
    /// Pinned offload-pool blocks. Python sets it on every interval, which
    /// also tells this payload apart from other connectors' `types`/`data`
    /// payloads.
    #[serde(rename = "vllm:simple_kv_offload_used_blocks")]
    pub used_blocks: BTreeMap<[String; 0], u64>,
    /// Offload blocks in pending stores.
    #[serde(default, rename = "vllm:simple_kv_offload_pending_store_blocks")]
    pub pending_store_blocks: BTreeMap<[String; 0], u64>,
    /// Deployment facts as labels (`backend`, `page_cache`, `lazy_offload`,
    /// `capacity_blocks`); the value is always 1.
    #[serde(default, rename = "vllm:simple_kv_offload_info")]
    pub info: BTreeMap<[String; 4], u64>,
}

/// Labels for `vllm:simple_kv_offload_save_outcomes`.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct SimpleCpuOffloadOutcomeLabels {
    model_name: String,
    engine: u32,
    outcome: String,
}

/// Labels for `vllm:simple_kv_offload_info`, whose label values are the
/// connector's deployment facts.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct SimpleCpuOffloadInfoLabels {
    model_name: String,
    engine: u32,
    backend: String,
    page_cache: String,
    lazy_offload: String,
    capacity_blocks: String,
}

/// SimpleCPUOffload connector telemetry. Mirrors `SimpleCPUOffloadPromMetrics`.
#[derive(Clone)]
pub(super) struct SimpleCpuOffloadMetrics {
    save_outcomes_total: Family<SimpleCpuOffloadOutcomeLabels, U64Counter>,
    load_blocks_total: Family<EngineLabels, U64Counter>,
    used_blocks: Family<EngineLabels, U64Gauge>,
    pending_store_blocks: Family<EngineLabels, U64Gauge>,
    info: Family<SimpleCpuOffloadInfoLabels, U64Gauge>,
}

impl SimpleCpuOffloadMetrics {
    pub(super) fn register(registry: &mut Registry) -> Self {
        let save_outcomes_total = Family::default();
        registry.register(
            "vllm:simple_kv_offload_save_outcomes",
            "Store admission decisions in SimpleCPUOffloadConnector, by outcome. Outcomes \
             classify eager-mode boundary hand-off stores; lazy-mode stores are not classified",
            save_outcomes_total.clone(),
        );

        let load_blocks_total = Family::default();
        registry.register(
            "vllm:simple_kv_offload_load_blocks",
            "KV blocks restored to GPU from the offload pool, counted when a load finishes. This \
             is prefill work avoided through offload cache hits",
            load_blocks_total.clone(),
        );

        let used_blocks = Family::default();
        registry.register(
            "vllm:simple_kv_offload_used_blocks",
            "Offload-pool blocks currently pinned by in-flight transfers or cache hits (capacity \
             minus free). Evictable cached blocks are not counted; capacity_blocks is on \
             vllm:simple_kv_offload_info",
            used_blocks.clone(),
        );

        let pending_store_blocks = Family::default();
        registry.register(
            "vllm:simple_kv_offload_pending_store_blocks",
            "Offload blocks in pending stores: queued for dispatch, issued to workers, or \
             awaiting release after a cache reset. A persistently growing value indicates a stuck \
             transfer",
            pending_store_blocks.clone(),
        );

        let info = Family::default();
        registry.register(
            "vllm:simple_kv_offload_info",
            "SimpleCPUOffloadConnector deployment facts. Value is always 1. backend is cpu or \
             disk; page_cache and lazy_offload are true/false; capacity_blocks is the offload \
             pool size",
            info.clone(),
        );

        Self {
            save_outcomes_total,
            load_blocks_total,
            used_blocks,
            pending_store_blocks,
            info,
        }
    }

    pub(super) fn record(&self, labels: &EngineLabels, stats: &SimpleCpuOffloadStats) {
        let data = &stats.data;
        for ([outcome], value) in &data.save_outcomes {
            let outcome_labels = SimpleCpuOffloadOutcomeLabels {
                model_name: labels.model_name.clone(),
                engine: labels.engine,
                outcome: outcome.clone(),
            };
            self.save_outcomes_total.get_or_create(&outcome_labels).inc_by(*value);
        }
        for value in data.load_blocks.values() {
            self.load_blocks_total.get_or_create(labels).inc_by(*value);
        }
        for value in data.used_blocks.values() {
            self.used_blocks.get_or_create(labels).set(*value);
        }
        for value in data.pending_store_blocks.values() {
            self.pending_store_blocks.get_or_create(labels).set(*value);
        }
        for ([backend, page_cache, lazy_offload, capacity_blocks], value) in &data.info {
            let info_labels = SimpleCpuOffloadInfoLabels {
                model_name: labels.model_name.clone(),
                engine: labels.engine,
                backend: backend.clone(),
                page_cache: page_cache.clone(),
                lazy_offload: lazy_offload.clone(),
                capacity_blocks: capacity_blocks.clone(),
            };
            self.info.get_or_create(&info_labels).set(*value);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use expect_test::expect;
    use itertools::Itertools as _;

    use super::{SimpleCpuOffloadData, SimpleCpuOffloadStats};
    use crate::{KvConnectorStats, Metrics, MultiConnectorStats};

    #[test]
    fn simple_cpu_offload_stats_are_recorded_into_simple_kv_offload_metrics() {
        let metrics = Metrics::new();
        let handles = metrics.scheduler.resolve_handles("model", 0);
        // Series of one family render in hash order, so sort them.
        let simple_kv_offload_series = || {
            metrics
                .render()
                .unwrap()
                .lines()
                .filter(|line| line.starts_with("vllm:simple_kv_offload_"))
                .sorted()
                .join("\n")
        };

        // Like Python, no series before the connector reports.
        expect![[""]].assert_eq(&simple_kv_offload_series());

        let interval = |stored, used_blocks| SimpleCpuOffloadStats {
            data: SimpleCpuOffloadData {
                save_outcomes: BTreeMap::from([
                    (["stored".to_string()], stored),
                    (["dropped_cpu_full".to_string()], 2),
                ]),
                load_blocks: BTreeMap::from([([], 4)]),
                used_blocks: BTreeMap::from([([], used_blocks)]),
                pending_store_blocks: BTreeMap::from([([], 2)]),
                info: BTreeMap::from([(["cpu", "false", "true", "64"].map(String::from), 1)]),
            },
        };
        // Once as the only connector, once as a `MultiConnector` child:
        // counters add up, gauges keep the latest value.
        for kv_connector_stats in [
            KvConnectorStats::SimpleCpuOffload(interval(3, 9)),
            KvConnectorStats::Multi(Box::new(MultiConnectorStats {
                simple_cpu_offload: Some(interval(1, 5)),
                ..Default::default()
            })),
        ] {
            handles.kv_connector.record(&kv_connector_stats);
        }

        expect![[r#"
            vllm:simple_kv_offload_info{model_name="model",engine="0",backend="cpu",page_cache="false",lazy_offload="true",capacity_blocks="64"} 1
            vllm:simple_kv_offload_load_blocks_total{model_name="model",engine="0"} 8
            vllm:simple_kv_offload_pending_store_blocks{model_name="model",engine="0"} 2
            vllm:simple_kv_offload_save_outcomes_total{model_name="model",engine="0",outcome="dropped_cpu_full"} 4
            vllm:simple_kv_offload_save_outcomes_total{model_name="model",engine="0",outcome="stored"} 4
            vllm:simple_kv_offload_used_blocks{model_name="model",engine="0"} 5"#]].assert_eq(&simple_kv_offload_series());
    }
}
