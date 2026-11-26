// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};

use super::nixl::{bytes_transferred_histogram, num_descriptors_histogram, xfer_time_histogram};
use crate::{EngineLabels, Histogram, HistogramFamily, U64Counter};

/// Mooncake connector transfer telemetry.
///
/// Original Python definition: `MooncakeKVConnectorStats` in
/// `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/stats.py`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MooncakeTransferStats {
    /// Time spent transferring each successful request.
    pub transfer_duration: Vec<f64>,
    /// Bytes transferred by each successful request.
    pub bytes_transferred: Vec<u64>,
    /// Descriptor count for each successful request.
    pub num_descriptors: Vec<u64>,
    /// Failure counter increments collected since the previous update.
    pub num_failed_transfers: Vec<u64>,
    /// Receive failure counter increments collected since the previous update.
    pub num_failed_recvs: Vec<u64>,
    /// Expired-request counter increments collected since the previous update.
    pub num_kv_expired_reqs: Vec<u64>,
}

/// Mooncake connector transfer telemetry. Mirrors `MooncakePromMetrics`.
#[derive(Clone)]
pub(super) struct MooncakeTransferMetrics {
    xfer_time_seconds: HistogramFamily,
    bytes_transferred: HistogramFamily,
    num_descriptors: HistogramFamily,
    num_failed_transfers: Family<EngineLabels, U64Counter>,
    num_failed_recvs: Family<EngineLabels, U64Counter>,
    num_kv_expired_reqs: Family<EngineLabels, U64Counter>,
}

impl MooncakeTransferMetrics {
    pub(super) fn register(registry: &mut Registry) -> Self {
        // Python's `MooncakePromMetrics` uses the same bucket edges as
        // `NixlPromMetrics`.
        let xfer_time_seconds =
            Family::new_with_constructor(xfer_time_histogram as fn() -> Histogram);
        registry.register(
            "vllm:mooncake_xfer_time_seconds",
            "Histogram of transfer duration for Mooncake KV Cache transfers.",
            xfer_time_seconds.clone(),
        );

        let bytes_transferred =
            Family::new_with_constructor(bytes_transferred_histogram as fn() -> Histogram);
        registry.register(
            "vllm:mooncake_bytes_transferred",
            "Histogram of bytes transferred per Mooncake KV Cache transfer.",
            bytes_transferred.clone(),
        );

        let num_descriptors =
            Family::new_with_constructor(num_descriptors_histogram as fn() -> Histogram);
        registry.register(
            "vllm:mooncake_num_descriptors",
            "Histogram of number of descriptors per Mooncake KV Cache transfer.",
            num_descriptors.clone(),
        );

        let num_failed_transfers = Family::default();
        registry.register(
            "vllm:mooncake_num_failed_transfers",
            "Number of failed Mooncake KV Cache transfers.",
            num_failed_transfers.clone(),
        );

        let num_failed_recvs = Family::default();
        registry.register(
            "vllm:mooncake_num_failed_recvs",
            "Number of failed Mooncake KV Cache receives.",
            num_failed_recvs.clone(),
        );

        let num_kv_expired_reqs = Family::default();
        registry.register(
            "vllm:mooncake_num_kv_expired_reqs",
            "Number of requests that had their KV expire. \
             NOTE: This metric is tracked on the P instance.",
            num_kv_expired_reqs.clone(),
        );

        Self {
            xfer_time_seconds,
            bytes_transferred,
            num_descriptors,
            num_failed_transfers,
            num_failed_recvs,
            num_kv_expired_reqs,
        }
    }

    pub(super) fn record(&self, labels: &EngineLabels, stats: &MooncakeTransferStats) {
        let xfer_time_seconds = self.xfer_time_seconds.get_or_create(labels);
        for value in &stats.transfer_duration {
            xfer_time_seconds.observe(*value);
        }
        let bytes_transferred = self.bytes_transferred.get_or_create(labels);
        for value in &stats.bytes_transferred {
            bytes_transferred.observe(*value as f64);
        }
        let num_descriptors = self.num_descriptors.get_or_create(labels);
        for value in &stats.num_descriptors {
            num_descriptors.observe(*value as f64);
        }
        self.num_failed_transfers
            .get_or_create(labels)
            .inc_by(stats.num_failed_transfers.iter().sum());
        self.num_failed_recvs
            .get_or_create(labels)
            .inc_by(stats.num_failed_recvs.iter().sum());
        self.num_kv_expired_reqs
            .get_or_create(labels)
            .inc_by(stats.num_kv_expired_reqs.iter().sum());
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::{KvConnectorStats, Metrics};

    /// Records a bare `MooncakeConnector` payload, in the shape Python's
    /// `MooncakeKVConnectorStats` sends, into the `vllm:mooncake_*` transfer
    /// metrics.
    #[test]
    fn mooncake_transfer_stats_are_recorded_into_mooncake_metrics() {
        let metrics = Metrics::new();
        let handles = metrics.scheduler.resolve_handles("model", 0);

        let kv_connector_stats: KvConnectorStats = serde_json::from_value(serde_json::json!({
            "transfer_duration": [0.01, 0.02],
            "bytes_transferred": [4096, 8192],
            "num_descriptors": [2, 4],
            "num_failed_transfers": [1, 1],
            "num_failed_recvs": [1],
            "num_kv_expired_reqs": [],
        }))
        .unwrap();

        handles.kv_connector.record(&kv_connector_stats);

        let rendered = metrics.render().unwrap();
        let mooncake_transfer_series = rendered
            .lines()
            .filter(|line| {
                line.starts_with("vllm:mooncake_")
                    && !line.starts_with("vllm:mooncake_store_")
                    && !line.contains("_bucket{")
            })
            .collect::<Vec<_>>()
            .join("\n");
        expect![[r#"
            vllm:mooncake_xfer_time_seconds_sum{model_name="model",engine="0"} 0.03
            vllm:mooncake_xfer_time_seconds_count{model_name="model",engine="0"} 2
            vllm:mooncake_bytes_transferred_sum{model_name="model",engine="0"} 12288.0
            vllm:mooncake_bytes_transferred_count{model_name="model",engine="0"} 2
            vllm:mooncake_num_descriptors_sum{model_name="model",engine="0"} 6.0
            vllm:mooncake_num_descriptors_count{model_name="model",engine="0"} 2
            vllm:mooncake_num_failed_transfers_total{model_name="model",engine="0"} 2
            vllm:mooncake_num_failed_recvs_total{model_name="model",engine="0"} 1
            vllm:mooncake_num_kv_expired_reqs_total{model_name="model",engine="0"} 0"#]]
        .assert_eq(&mooncake_transfer_series);
    }
}
