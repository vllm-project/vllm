// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};

use crate::{EngineLabels, U64Counter};

/// Interval counters of one UMBP store transfer operation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UmbpOperationCounts {
    /// Objects submitted.
    pub submitted: u64,
    /// Objects completed.
    pub completed: u64,
    /// Objects failed.
    pub failed: u64,
    /// Bytes in successfully completed transfer ranges.
    pub num_bytes: u64,
}

/// UMBP store connector telemetry grouped by operation name.
///
/// Python records any operation name and exports it as a label, so names
/// stay strings here, as in [`MooncakeStats`]. The values must still be
/// operation counters, which keeps other connectors' payloads from
/// decoding as UMBP.
///
/// Original Python definition: `UMBPStoreConnectorStats` in
/// `vllm/distributed/kv_transfer/kv_connector/v1/umbp/stats.py`.
///
/// [`MooncakeStats`]: super::MooncakeStats
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UmbpStats(pub BTreeMap<String, UmbpOperationCounts>);

/// Labels for per-operation UMBP store connector telemetry. `operation` is
/// dynamic, as in Mooncake store telemetry.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct UmbpOperationLabels {
    model_name: String,
    engine: u32,
    operation: String,
}

type UmbpOperationCounterFamily = Family<UmbpOperationLabels, U64Counter>;

/// UMBP store connector telemetry. Mirrors `UMBPStorePromMetrics`.
#[derive(Clone)]
pub(super) struct UmbpMetrics {
    transfer_submitted_total: UmbpOperationCounterFamily,
    transfer_completed_total: UmbpOperationCounterFamily,
    transfer_failed_total: UmbpOperationCounterFamily,
    transfer_bytes_total: UmbpOperationCounterFamily,
}

impl UmbpMetrics {
    pub(super) fn register(registry: &mut Registry) -> Self {
        let transfer_submitted_total = Family::default();
        registry.register(
            "vllm:umbp_transfer_submitted",
            "Number of UMBP objects submitted",
            transfer_submitted_total.clone(),
        );

        let transfer_completed_total = Family::default();
        registry.register(
            "vllm:umbp_transfer_completed",
            "Number of UMBP objects completed",
            transfer_completed_total.clone(),
        );

        let transfer_failed_total = Family::default();
        registry.register(
            "vllm:umbp_transfer_failed",
            "Number of UMBP objects failed",
            transfer_failed_total.clone(),
        );

        let transfer_bytes_total = Family::default();
        registry.register(
            "vllm:umbp_transfer_bytes",
            "Bytes in successfully completed UMBP transfer ranges",
            transfer_bytes_total.clone(),
        );

        Self {
            transfer_submitted_total,
            transfer_completed_total,
            transfer_failed_total,
            transfer_bytes_total,
        }
    }

    pub(super) fn record(&self, labels: &EngineLabels, stats: &UmbpStats) {
        for (operation, counts) in &stats.0 {
            let labels = UmbpOperationLabels {
                model_name: labels.model_name.clone(),
                engine: labels.engine,
                operation: operation.clone(),
            };
            self.transfer_submitted_total.get_or_create(&labels).inc_by(counts.submitted);
            self.transfer_completed_total.get_or_create(&labels).inc_by(counts.completed);
            self.transfer_failed_total.get_or_create(&labels).inc_by(counts.failed);
            self.transfer_bytes_total.get_or_create(&labels).inc_by(counts.num_bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use itertools::Itertools as _;

    use crate::{KvConnectorStats, Metrics};

    /// Records a bare `UMBPStoreConnector` payload, in the shape Python's
    /// `UMBPStoreConnectorStats` sends, into the `vllm:umbp_transfer_*`
    /// counters, one series per reported operation.
    #[test]
    fn umbp_stats_are_recorded_into_umbp_transfer_metrics() {
        let metrics = Metrics::new();
        let handles = metrics.scheduler.resolve_handles("model", 0);
        // Series of one family render in hash order, so sort them.
        let umbp_series = || {
            metrics
                .render()
                .unwrap()
                .lines()
                .filter(|line| line.starts_with("vllm:umbp_transfer_"))
                .sorted()
                .join("\n")
        };

        // Like Python, no series before the connector reports an operation.
        expect![[""]].assert_eq(&umbp_series());

        let kv_connector_stats: KvConnectorStats = serde_json::from_value(serde_json::json!({
            "load": {"submitted": 3, "completed": 2, "failed": 1, "num_bytes": 8192},
            "store": {"submitted": 1, "completed": 0, "failed": 0, "num_bytes": 0},
        }))
        .unwrap();
        handles.kv_connector.record(&kv_connector_stats);
        handles.kv_connector.record(&kv_connector_stats);

        expect![[r#"
            vllm:umbp_transfer_bytes_total{model_name="model",engine="0",operation="load"} 16384
            vllm:umbp_transfer_bytes_total{model_name="model",engine="0",operation="store"} 0
            vllm:umbp_transfer_completed_total{model_name="model",engine="0",operation="load"} 4
            vllm:umbp_transfer_completed_total{model_name="model",engine="0",operation="store"} 0
            vllm:umbp_transfer_failed_total{model_name="model",engine="0",operation="load"} 2
            vllm:umbp_transfer_failed_total{model_name="model",engine="0",operation="store"} 0
            vllm:umbp_transfer_submitted_total{model_name="model",engine="0",operation="load"} 6
            vllm:umbp_transfer_submitted_total{model_name="model",engine="0",operation="store"} 2"#]]
        .assert_eq(&umbp_series());
    }
}
