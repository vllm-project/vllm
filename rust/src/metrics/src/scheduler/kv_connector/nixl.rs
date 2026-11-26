// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};

use crate::{EngineLabels, Histogram, HistogramFamily, U64Counter};

/// NIXL connector transfer telemetry.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NixlStats {
    /// Time spent transferring each successful request.
    pub transfer_duration: Vec<f64>,
    /// Time spent posting each successful transfer.
    pub post_duration: Vec<f64>,
    /// Bytes transferred by each successful request.
    pub bytes_transferred: Vec<u64>,
    /// Descriptor count for each successful request.
    pub num_descriptors: Vec<u64>,
    /// Failure counter increments collected since the previous update.
    pub num_failed_transfers: Vec<u64>,
    /// Notification failure counter increments collected since the previous update.
    pub num_failed_notifications: Vec<u64>,
    /// Handshake failure counter increments collected since the previous update.
    #[serde(default)]
    pub num_failed_handshakes: Vec<u64>,
    /// Expired-request counter increments collected since the previous update.
    pub num_kv_expired_reqs: Vec<u64>,
    /// Late notification counter increments collected since the previous update.
    #[serde(default)]
    pub num_notifications_after_expiry: Vec<u64>,
    /// Declined expired-read counter increments collected since the previous update.
    #[serde(default)]
    pub num_declined_expired_reads: Vec<u64>,
    /// Push-mode registration-expiry counter increments collected since the previous update.
    #[serde(default)]
    pub num_registration_expired: Vec<u64>,
}

// Buckets copied verbatim from the Python `NixlPromMetrics` class so
// dashboards built against either frontend are interchangeable.
const XFER_TIME_BUCKETS: [f64; 12] = [
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 5.0,
];
const POST_TIME_BUCKETS: [f64; 13] = [
    0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 5.0,
];
// Uniform 2KB to 8GB range: 2**(10+i) for i in range(1, 25, 2).
const BYTES_TRANSFERRED_BUCKETS: [f64; 12] = [
    2048.0,
    8192.0,
    32768.0,
    131072.0,
    524288.0,
    2097152.0,
    8388608.0,
    33554432.0,
    134217728.0,
    536870912.0,
    2147483648.0,
    8589934592.0,
];
const NUM_DESCRIPTORS_BUCKETS: [f64; 14] = [
    10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 200.0, 400.0, 1000.0, 2000.0, 4000.0, 10000.0, 20000.0,
    50000.0,
];

pub(super) fn xfer_time_histogram() -> Histogram {
    Histogram::new(XFER_TIME_BUCKETS.iter().copied())
}

fn post_time_histogram() -> Histogram {
    Histogram::new(POST_TIME_BUCKETS.iter().copied())
}

pub(super) fn bytes_transferred_histogram() -> Histogram {
    Histogram::new(BYTES_TRANSFERRED_BUCKETS.iter().copied())
}

pub(super) fn num_descriptors_histogram() -> Histogram {
    Histogram::new(NUM_DESCRIPTORS_BUCKETS.iter().copied())
}

/// NIXL connector telemetry. Mirrors `NixlPromMetrics`.
#[derive(Clone)]
pub(super) struct NixlMetrics {
    xfer_time_seconds: HistogramFamily,
    post_time_seconds: HistogramFamily,
    bytes_transferred: HistogramFamily,
    num_descriptors: HistogramFamily,
    num_failed_transfers: Family<EngineLabels, U64Counter>,
    num_failed_notifications: Family<EngineLabels, U64Counter>,
    num_kv_expired_reqs: Family<EngineLabels, U64Counter>,
    num_notifications_after_expiry: Family<EngineLabels, U64Counter>,
    num_declined_expired_reads: Family<EngineLabels, U64Counter>,
    num_registration_expired: Family<EngineLabels, U64Counter>,
}

impl NixlMetrics {
    pub(super) fn register(registry: &mut Registry) -> Self {
        let xfer_time_seconds =
            Family::new_with_constructor(xfer_time_histogram as fn() -> Histogram);
        registry.register(
            "vllm:nixl_xfer_time_seconds",
            "Histogram of transfer duration for NIXL KV Cache transfers.",
            xfer_time_seconds.clone(),
        );

        let post_time_seconds =
            Family::new_with_constructor(post_time_histogram as fn() -> Histogram);
        registry.register(
            "vllm:nixl_post_time_seconds",
            "Histogram of transfer post time for NIXL KV Cache transfers.",
            post_time_seconds.clone(),
        );

        let bytes_transferred =
            Family::new_with_constructor(bytes_transferred_histogram as fn() -> Histogram);
        registry.register(
            "vllm:nixl_bytes_transferred",
            "Histogram of bytes transferred per NIXL KV Cache transfers.",
            bytes_transferred.clone(),
        );

        let num_descriptors =
            Family::new_with_constructor(num_descriptors_histogram as fn() -> Histogram);
        registry.register(
            "vllm:nixl_num_descriptors",
            "Histogram of number of descriptors per NIXL KV Cache transfers.",
            num_descriptors.clone(),
        );

        let num_failed_transfers = Family::default();
        registry.register(
            "vllm:nixl_num_failed_transfers",
            "Number of failed NIXL KV Cache transfers, including handshake and notification \
             failures. NOTE: KV expiry is tracked separately in \
             vllm:nixl_num_kv_expired_reqs.",
            num_failed_transfers.clone(),
        );

        let num_failed_notifications = Family::default();
        registry.register(
            "vllm:nixl_num_failed_notifications",
            "Number of failed NIXL KV Cache notifications. Retained for compatibility; \
             these failures are also included in vllm:nixl_num_failed_transfers.",
            num_failed_notifications.clone(),
        );

        let num_kv_expired_reqs = Family::default();
        registry.register(
            "vllm:nixl_num_kv_expired_reqs",
            "Number of requests that had their KV expire. \
             NOTE: This metric is tracked on the P instance.",
            num_kv_expired_reqs.clone(),
        );

        let num_notifications_after_expiry = Family::default();
        registry.register(
            "vllm:nixl_num_notifications_after_expiry",
            "Number of completion notifications for requests that were no longer tracked, \
             usually because their KV lease expired. The KV blocks may have been reused \
             before the transfer finished. Counted per notification (one per remote rank).",
            num_notifications_after_expiry.clone(),
        );

        let num_declined_expired_reads = Family::default();
        registry.register(
            "vllm:nixl_num_declined_expired_reads",
            "Number of remote KV reads declined because the remote had released the blocks \
             (lease expired) before the read. Each one is a request that follows \
             kv_load_failure_policy. NOTE: This metric is tracked on the reading (consumer) \
             instance",
            num_declined_expired_reads.clone(),
        );

        let num_registration_expired = Family::default();
        registry.register(
            "vllm:nixl_num_registration_expired",
            "Number of push-mode D-side registrations whose watchdog deadline passed \
             without a push completion. NOTE: This metric is tracked on the D instance.",
            num_registration_expired.clone(),
        );

        Self {
            xfer_time_seconds,
            post_time_seconds,
            bytes_transferred,
            num_descriptors,
            num_failed_transfers,
            num_failed_notifications,
            num_kv_expired_reqs,
            num_notifications_after_expiry,
            num_declined_expired_reads,
            num_registration_expired,
        }
    }

    pub(super) fn record(&self, labels: &EngineLabels, stats: &NixlStats) {
        let xfer_time_seconds = self.xfer_time_seconds.get_or_create(labels);
        for value in &stats.transfer_duration {
            xfer_time_seconds.observe(*value);
        }
        let post_time_seconds = self.post_time_seconds.get_or_create(labels);
        for value in &stats.post_duration {
            post_time_seconds.observe(*value);
        }
        let bytes_transferred = self.bytes_transferred.get_or_create(labels);
        for value in &stats.bytes_transferred {
            bytes_transferred.observe(*value as f64);
        }
        let num_descriptors = self.num_descriptors.get_or_create(labels);
        for value in &stats.num_descriptors {
            num_descriptors.observe(*value as f64);
        }
        self.num_failed_transfers.get_or_create(labels).inc_by(
            stats
                .num_failed_transfers
                .iter()
                .chain(&stats.num_failed_handshakes)
                .chain(&stats.num_failed_notifications)
                .sum(),
        );
        self.num_failed_notifications
            .get_or_create(labels)
            .inc_by(stats.num_failed_notifications.iter().sum());
        self.num_kv_expired_reqs
            .get_or_create(labels)
            .inc_by(stats.num_kv_expired_reqs.iter().sum());
        self.num_notifications_after_expiry
            .get_or_create(labels)
            .inc_by(stats.num_notifications_after_expiry.iter().sum());
        self.num_declined_expired_reads
            .get_or_create(labels)
            .inc_by(stats.num_declined_expired_reads.iter().sum());
        self.num_registration_expired
            .get_or_create(labels)
            .inc_by(stats.num_registration_expired.iter().sum());
    }
}
