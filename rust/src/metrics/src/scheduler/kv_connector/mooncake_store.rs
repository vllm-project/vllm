// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};

use crate::{EngineLabels, Histogram, U64Counter};

/// One Mooncake store operation sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MooncakeRecord {
    pub duration_seconds: f64,
    pub num_keys: u64,
    pub num_bytes: u64,
    /// `ok`, `error` or `partial_failure` today; passed through as the
    /// `status` label, like the operation name.
    pub status: String,
    pub num_failed_keys: u64,
}

/// Mooncake store telemetry grouped by operation name.
///
/// Python records any operation name and status string and exports them
/// as labels, so they stay strings here: a new operation is exported like
/// Python without a Rust change, instead of failing this payload and,
/// through the untagged [`KvConnectorStats`], every typed sibling of a
/// `MultiConnector`. The values must still be operation records, which
/// keeps other connectors' payloads from decoding as Mooncake.
///
/// [`KvConnectorStats`]: super::KvConnectorStats
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MooncakeStats(pub BTreeMap<String, Vec<MooncakeRecord>>);

// Buckets copied verbatim from the Python `MooncakeStorePromMetrics` class so
// dashboards built against either frontend are interchangeable.
const OPERATION_TIME_BUCKETS: [f64; 15] = [
    1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 7.5e-1, 1.0, 1.5, 2.0, 3.0, 4.0,
];

fn operation_time_histogram() -> Histogram {
    Histogram::new(OPERATION_TIME_BUCKETS.iter().copied())
}

/// Labels for per-operation Mooncake store connector telemetry. `operation`
/// and `status` are dynamic (not known ahead of time).
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct MooncakeOperationLabels {
    model_name: String,
    engine: u32,
    operation: String,
    status: String,
}

type MooncakeOperationCounterFamily = Family<MooncakeOperationLabels, U64Counter>;

/// Mooncake store connector telemetry. Mirrors `MooncakeStorePromMetrics`.
#[derive(Clone)]
pub(super) struct MooncakeStoreMetrics {
    operation_time_seconds: Family<MooncakeOperationLabels, Histogram, fn() -> Histogram>,
    operation_total: MooncakeOperationCounterFamily,
    operation_keys_total: MooncakeOperationCounterFamily,
    operation_bytes_total: MooncakeOperationCounterFamily,
    operation_failed_keys_total: MooncakeOperationCounterFamily,
}

impl MooncakeStoreMetrics {
    pub(super) fn register(registry: &mut Registry) -> Self {
        let operation_time_seconds =
            Family::new_with_constructor(operation_time_histogram as fn() -> Histogram);
        registry.register(
            "vllm:mooncake_store_operation_time_seconds",
            "Histogram of Mooncake store communication time.",
            operation_time_seconds.clone(),
        );

        // NOTE: registration names below omit the `_total` suffix that the
        // Python metric names carry (e.g. `vllm:mooncake_store_operation_total`)
        // because this crate's `prometheus-client` (unlike Python's, which
        // strips a user-supplied `_total` before re-adding it) appends
        // `_total` unconditionally for counters; see the note in `lib.rs`.
        let operation_total = Family::default();
        registry.register(
            "vllm:mooncake_store_operation",
            "Number of Mooncake store communication operations.",
            operation_total.clone(),
        );

        let operation_keys_total = Family::default();
        registry.register(
            "vllm:mooncake_store_operation_keys",
            "Number of Mooncake store keys touched by operations.",
            operation_keys_total.clone(),
        );

        let operation_bytes_total = Family::default();
        registry.register(
            "vllm:mooncake_store_operation_bytes",
            "Number of bytes transferred by Mooncake store operations.",
            operation_bytes_total.clone(),
        );

        let operation_failed_keys_total = Family::default();
        registry.register(
            "vllm:mooncake_store_operation_failed_keys",
            "Number of Mooncake store keys that failed in operations.",
            operation_failed_keys_total.clone(),
        );

        Self {
            operation_time_seconds,
            operation_total,
            operation_keys_total,
            operation_bytes_total,
            operation_failed_keys_total,
        }
    }

    pub(super) fn record(&self, labels: &EngineLabels, stats: &MooncakeStats) {
        for (operation, records) in &stats.0 {
            for record in records {
                let labels = MooncakeOperationLabels {
                    model_name: labels.model_name.clone(),
                    engine: labels.engine,
                    operation: operation.clone(),
                    status: record.status.clone(),
                };
                self.operation_time_seconds
                    .get_or_create(&labels)
                    .observe(record.duration_seconds);
                self.operation_total.get_or_create(&labels).inc();
                self.operation_keys_total.get_or_create(&labels).inc_by(record.num_keys);
                self.operation_bytes_total.get_or_create(&labels).inc_by(record.num_bytes);
                self.operation_failed_keys_total
                    .get_or_create(&labels)
                    .inc_by(record.num_failed_keys);
            }
        }
    }
}
