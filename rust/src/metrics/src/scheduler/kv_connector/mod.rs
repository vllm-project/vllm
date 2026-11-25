// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! KV connector telemetry carried by `SchedulerStats`: one module per
//! connector, each holding its payload decoder, its Prometheus families and
//! their recording. Connector series appear once the connector first reports;
//! Python builds only the configured connector's metrics, and creates their
//! per-engine series at startup.

use std::collections::BTreeMap;

use prometheus_client::registry::Registry;
use rmpv::Value as OpaqueValue;
use serde::{Deserialize, Serialize};

use crate::EngineLabels;

mod mooncake;
mod mooncake_store;
mod nixl;
mod offloading;
mod simple_cpu_offload;
mod umbp;

use mooncake::MooncakeTransferMetrics;
pub use mooncake::MooncakeTransferStats;
use mooncake_store::MooncakeStoreMetrics;
pub use mooncake_store::{MooncakeRecord, MooncakeStats};
use nixl::NixlMetrics;
pub use nixl::NixlStats;
use offloading::OffloadingMetrics;
pub use offloading::{OffloadingSample, OffloadingStats};
use simple_cpu_offload::SimpleCpuOffloadMetrics;
pub use simple_cpu_offload::{SimpleCpuOffloadData, SimpleCpuOffloadStats};
use umbp::UmbpMetrics;
pub use umbp::{UmbpOperationCounts, UmbpStats};

/// Telemetry emitted by a `MultiConnector` with a flat child payload map.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MultiConnectorStats {
    #[serde(rename = "NixlConnector")]
    pub nixl: Option<NixlStats>,
    #[serde(rename = "NixlPullConnector")]
    pub nixl_pull: Option<NixlStats>,
    #[serde(rename = "NixlPushConnector")]
    pub nixl_push: Option<NixlStats>,
    #[serde(rename = "MooncakeConnector")]
    pub mooncake_transfer: Option<MooncakeTransferStats>,
    #[serde(rename = "MooncakeStoreConnector")]
    pub mooncake: Option<MooncakeStats>,
    #[serde(rename = "UMBPStoreConnector")]
    pub umbp: Option<UmbpStats>,
    #[serde(rename = "SimpleCPUOffloadConnector")]
    pub simple_cpu_offload: Option<SimpleCpuOffloadStats>,
    #[serde(rename = "OffloadingConnector")]
    pub offloading: Option<OffloadingStats>,
    /// Child connector payloads without a Rust telemetry implementation.
    #[serde(flatten)]
    pub other: BTreeMap<String, OpaqueValue>,
}

/// Connector-specific scheduler telemetry.
///
/// Variants are tried in order. `Multi` accepts any map, so every
/// connector payload with its own variant must come before it.
///
/// The typed payloads ignore unknown fields: Python adds counters to them
/// over time, and a strict decode would drop every metric of that
/// connector, and of its `MultiConnector` siblings, until Rust caught up.
/// The variants stay distinct through the fields each one requires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KvConnectorStats {
    Nixl(NixlStats),
    MooncakeTransfer(MooncakeTransferStats),
    Mooncake(MooncakeStats),
    Umbp(UmbpStats),
    SimpleCpuOffload(SimpleCpuOffloadStats),
    /// After `SimpleCpuOffload`, whose payloads share its envelope.
    Offloading(OffloadingStats),
    Multi(Box<MultiConnectorStats>),
    Other(BTreeMap<String, OpaqueValue>),
}

/// Prometheus families of every connector with a Rust telemetry
/// implementation.
#[derive(Clone)]
pub(super) struct KvConnectorMetrics {
    mooncake_store: MooncakeStoreMetrics,
    mooncake_transfer: MooncakeTransferMetrics,
    nixl: NixlMetrics,
    umbp: UmbpMetrics,
    simple_cpu_offload: SimpleCpuOffloadMetrics,
    offloading: OffloadingMetrics,
}

/// One engine's KV connector recording. The families stay unresolved, so a
/// connector's series appear once it first reports.
pub struct KvConnectorHandles {
    pub(super) metrics: KvConnectorMetrics,
    pub(super) labels: EngineLabels,
}

impl KvConnectorHandles {
    pub fn record(&self, stats: &KvConnectorStats) {
        self.metrics.record(&self.labels, stats);
    }
}

impl KvConnectorMetrics {
    pub(super) fn register(registry: &mut Registry) -> Self {
        Self {
            mooncake_store: MooncakeStoreMetrics::register(registry),
            mooncake_transfer: MooncakeTransferMetrics::register(registry),
            nixl: NixlMetrics::register(registry),
            umbp: UmbpMetrics::register(registry),
            simple_cpu_offload: SimpleCpuOffloadMetrics::register(registry),
            offloading: OffloadingMetrics::register(registry),
        }
    }

    /// Record one connector payload. A bare connector reports its own flat
    /// payload; MultiConnector reports connector class name -> flat child
    /// payload.
    fn record(&self, labels: &EngineLabels, stats: &KvConnectorStats) {
        match stats {
            KvConnectorStats::Nixl(stats) => self.nixl.record(labels, stats),
            KvConnectorStats::MooncakeTransfer(stats) => {
                self.mooncake_transfer.record(labels, stats)
            }
            KvConnectorStats::Mooncake(stats) => self.mooncake_store.record(labels, stats),
            KvConnectorStats::Umbp(stats) => self.umbp.record(labels, stats),
            KvConnectorStats::SimpleCpuOffload(stats) => {
                self.simple_cpu_offload.record(labels, stats)
            }
            KvConnectorStats::Offloading(stats) => self.offloading.record(labels, stats),
            KvConnectorStats::Multi(stats) => self.record_multi(labels, stats),
            KvConnectorStats::Other(_) => {}
        }
    }

    fn record_multi(&self, labels: &EngineLabels, stats: &MultiConnectorStats) {
        for nixl in [&stats.nixl, &stats.nixl_pull, &stats.nixl_push].into_iter().flatten() {
            self.nixl.record(labels, nixl);
        }
        if let Some(mooncake_transfer) = &stats.mooncake_transfer {
            self.mooncake_transfer.record(labels, mooncake_transfer);
        }
        if let Some(mooncake) = &stats.mooncake {
            self.mooncake_store.record(labels, mooncake);
        }
        if let Some(umbp) = &stats.umbp {
            self.umbp.record(labels, umbp);
        }
        if let Some(simple_cpu_offload) = &stats.simple_cpu_offload {
            self.simple_cpu_offload.record(labels, simple_cpu_offload);
        }
        if let Some(offloading) = &stats.offloading {
            self.offloading.record(labels, offloading);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use expect_test::expect;
    use itertools::Itertools as _;

    use super::{
        KvConnectorStats, MooncakeRecord, MooncakeStats, MooncakeTransferStats,
        MultiConnectorStats, NixlStats,
    };
    use crate::Metrics;

    #[test]
    fn mooncake_stats_keep_unknown_operations_and_statuses() {
        let decode = |payload: serde_json::Value| -> KvConnectorStats {
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&payload).unwrap()).unwrap()
        };
        let record = |status: &str| {
            serde_json::json!({
                "duration_seconds": 0.5, "num_keys": 4, "num_bytes": 0, "status": status,
                "num_failed_keys": 1
            })
        };
        let nixl = serde_json::json!({
            "transfer_duration": [0.5], "post_duration": [0.1], "bytes_transferred": [4096],
            "num_descriptors": [2], "num_failed_transfers": [], "num_failed_notifications": [],
            "num_kv_expired_reqs": [], "num_future_counter": [1]
        });
        // Names no Rust code mentions, next to the `load_get` and `ok` of today.
        let store = serde_json::json!({
            "load_get": [record("ok")],
            "future_operation": [record("future_status")],
        });

        let KvConnectorStats::Mooncake(stats) = decode(store.clone()) else {
            panic!("expected Mooncake store stats");
        };
        assert_eq!(
            stats.0.keys().collect::<Vec<_>>(),
            ["future_operation", "load_get"]
        );
        assert_eq!(stats.0["future_operation"][0].status, "future_status");
        assert!(matches!(
            decode(serde_json::json!({"NixlConnector": nixl, "MooncakeStoreConnector": store})),
            KvConnectorStats::Multi(stats) if stats.nixl.is_some()
                && stats.mooncake.as_ref().is_some_and(|m| m.0.len() == 2)
        ));
    }

    #[test]
    fn kv_connector_stats_ignore_unknown_fields() {
        let decode = |payload: serde_json::Value| -> KvConnectorStats {
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&payload).unwrap()).unwrap()
        };
        let nixl = serde_json::json!({
            "transfer_duration": [0.5], "post_duration": [0.1], "bytes_transferred": [4096],
            "num_descriptors": [2], "num_failed_transfers": [], "num_failed_notifications": [],
            "num_kv_expired_reqs": [], "num_future_counter": [1]
        });
        let mooncake_transfer = serde_json::json!({
            "transfer_duration": [0.5], "bytes_transferred": [4096], "num_descriptors": [2],
            "num_failed_transfers": [1], "num_failed_recvs": [], "num_kv_expired_reqs": [],
            "num_future_counter": [1]
        });

        assert!(matches!(
            decode(nixl.clone()),
            KvConnectorStats::Nixl(stats) if stats.bytes_transferred == [4096]
        ));
        assert!(matches!(
            decode(mooncake_transfer.clone()),
            KvConnectorStats::MooncakeTransfer(stats) if stats.num_failed_transfers == [1]
        ));
        let umbp = serde_json::json!({
            "load": {"submitted": 2, "completed": 1, "failed": 1, "num_bytes": 4096,
                     "future_counter": 1},
            "future_operation": {"submitted": 1, "completed": 1, "failed": 0, "num_bytes": 0}
        });
        assert!(matches!(
            decode(umbp.clone()),
            KvConnectorStats::Umbp(stats)
                if stats.0["load"].num_bytes == 4096
                    && stats.0.contains_key("future_operation")
        ));
        // A newer child payload must not drop its siblings.
        assert!(matches!(
            decode(serde_json::json!({
                "NixlConnector": nixl,
                "MooncakeConnector": mooncake_transfer,
                "UMBPStoreConnector": umbp,
            })),
            KvConnectorStats::Multi(stats)
                if stats.nixl.is_some() && stats.mooncake_transfer.is_some()
                    && stats.umbp.is_some()
        ));
    }

    fn nixl_stats() -> NixlStats {
        NixlStats {
            transfer_duration: vec![0.01, 0.02],
            post_duration: vec![0.001, 0.002],
            bytes_transferred: vec![4096, 8192],
            num_descriptors: vec![2, 4],
            num_failed_transfers: vec![],
            num_failed_notifications: vec![1],
            num_failed_handshakes: vec![1],
            num_kv_expired_reqs: vec![1],
            num_notifications_after_expiry: vec![2],
        }
    }

    fn mooncake_stats() -> MooncakeStats {
        MooncakeStats(BTreeMap::from([(
            "load_get".to_owned(),
            vec![MooncakeRecord {
                duration_seconds: 0.05,
                num_keys: 3,
                num_bytes: 1024,
                status: "ok".to_owned(),
                num_failed_keys: 0,
            }],
        )]))
    }

    /// No NIXL or Mooncake series appear before the connector first reports,
    /// and then only the reporting connector's. Python builds only the
    /// configured connector's metrics, and creates its per-engine series at
    /// startup.
    #[test]
    fn kv_connector_series_appear_on_first_report() {
        let metrics = Metrics::new();
        let handles = metrics.scheduler.resolve_handles("model", 0);
        let connector_series = || {
            metrics
                .render()
                .unwrap()
                .lines()
                .filter(|line| {
                    (line.starts_with("vllm:nixl_") || line.starts_with("vllm:mooncake_"))
                        && !line.contains("_bucket{")
                })
                .sorted()
                .join("\n")
        };

        expect![[""]].assert_eq(&connector_series());

        let stats = KvConnectorStats::Nixl(nixl_stats());
        handles.kv_connector.record(&stats);

        expect![[r#"
            vllm:nixl_bytes_transferred_count{model_name="model",engine="0"} 2
            vllm:nixl_bytes_transferred_sum{model_name="model",engine="0"} 12288.0
            vllm:nixl_num_descriptors_count{model_name="model",engine="0"} 2
            vllm:nixl_num_descriptors_sum{model_name="model",engine="0"} 6.0
            vllm:nixl_num_failed_notifications_total{model_name="model",engine="0"} 1
            vllm:nixl_num_failed_transfers_total{model_name="model",engine="0"} 2
            vllm:nixl_num_kv_expired_reqs_total{model_name="model",engine="0"} 1
            vllm:nixl_num_notifications_after_expiry_total{model_name="model",engine="0"} 2
            vllm:nixl_post_time_seconds_count{model_name="model",engine="0"} 2
            vllm:nixl_post_time_seconds_sum{model_name="model",engine="0"} 0.003
            vllm:nixl_xfer_time_seconds_count{model_name="model",engine="0"} 2
            vllm:nixl_xfer_time_seconds_sum{model_name="model",engine="0"} 0.03"#]]
        .assert_eq(&connector_series());
    }

    /// Records one MultiConnector payload into Mooncake and NIXL metrics.
    #[test]
    fn kv_connector_stats_are_recorded_into_mooncake_and_nixl_metrics() {
        let metrics = Metrics::new();
        let handles = metrics.scheduler.resolve_handles("model", 0);

        let stats = KvConnectorStats::Multi(Box::new(MultiConnectorStats {
            nixl: Some(nixl_stats()),
            mooncake_transfer: Some(MooncakeTransferStats {
                transfer_duration: vec![0.03],
                bytes_transferred: vec![2048],
                num_descriptors: vec![1],
                num_failed_recvs: vec![1],
                ..Default::default()
            }),
            mooncake: Some(mooncake_stats()),
            ..Default::default()
        }));

        handles.kv_connector.record(&stats);

        let rendered = metrics.render().unwrap();
        assert!(rendered.contains(
            "vllm:mooncake_store_operation_total{model_name=\"model\",engine=\"0\",\
             operation=\"load_get\",status=\"ok\"} 1"
        ));
        assert!(rendered.contains(
            "vllm:mooncake_store_operation_keys_total{model_name=\"model\",engine=\"0\",\
             operation=\"load_get\",status=\"ok\"} 3"
        ));
        assert!(rendered.contains(
            "vllm:mooncake_store_operation_bytes_total{model_name=\"model\",engine=\"0\",\
             operation=\"load_get\",status=\"ok\"} 1024"
        ));
        assert!(
            rendered.contains(
                "vllm:nixl_num_kv_expired_reqs_total{model_name=\"model\",engine=\"0\"} 1"
            )
        );
        assert!(
            rendered.contains(
                "vllm:nixl_num_failed_transfers_total{model_name=\"model\",engine=\"0\"} 2"
            )
        );
        assert!(rendered.contains(
            "vllm:nixl_num_notifications_after_expiry_total{model_name=\"model\",engine=\"0\"} 2"
        ));
        assert!(
            rendered
                .contains("vllm:nixl_xfer_time_seconds_count{model_name=\"model\",engine=\"0\"} 2")
        );
        assert!(rendered.contains(
            "vllm:mooncake_xfer_time_seconds_count{model_name=\"model\",engine=\"0\"} 1"
        ));
        assert!(
            rendered.contains(
                "vllm:mooncake_num_failed_recvs_total{model_name=\"model\",engine=\"0\"} 1"
            )
        );
    }
}
