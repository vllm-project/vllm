// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Sampled KV-cache residency histograms.

use std::sync::Arc;

use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};

use crate::buckets::{BucketConfig, BucketFamily, FamilyHistogram};
use crate::{BucketHistogramFamily, EngineLabels, HistogramMetric};

/// Single KV cache block eviction sample.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L161-L167>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KvCacheEvictionEvent {
    /// Lifetime from allocation to eviction.
    pub lifetime_seconds: f64,
    /// Idle time observed before eviction.
    pub idle_seconds: f64,
    /// Time gaps between consecutive accesses before eviction.
    pub reuse_gaps_seconds: Vec<f64>,
}

/// Sampled KV-cache residency histograms.
pub struct KvCacheResidencyMetrics {
    pub kv_block_lifetime_seconds: BucketHistogramFamily,
    pub kv_block_idle_before_evict_seconds: BucketHistogramFamily,
    pub kv_block_reuse_gap_seconds: BucketHistogramFamily,
}

impl KvCacheResidencyMetrics {
    pub(super) fn register(registry: &mut Registry, buckets: &Arc<BucketConfig>) -> Self {
        let kv_block_lifetime_seconds = Family::new_with_constructor(FamilyHistogram::new(
            BucketFamily::KvCacheResidency,
            buckets,
        ));
        registry.register(
            "vllm:kv_block_lifetime_seconds",
            "Histogram of KV cache block lifetime from allocation to eviction. Sampled metrics (controlled by --kv-cache-metrics-sample).",
            kv_block_lifetime_seconds.clone(),
        );

        let kv_block_idle_before_evict_seconds = Family::new_with_constructor(
            FamilyHistogram::new(BucketFamily::KvCacheResidency, buckets),
        );
        registry.register(
            "vllm:kv_block_idle_before_evict_seconds",
            "Histogram of idle time before KV cache block eviction. Sampled metrics (controlled by --kv-cache-metrics-sample).",
            kv_block_idle_before_evict_seconds.clone(),
        );

        let kv_block_reuse_gap_seconds = Family::new_with_constructor(FamilyHistogram::new(
            BucketFamily::KvCacheResidency,
            buckets,
        ));
        registry.register(
            "vllm:kv_block_reuse_gap_seconds",
            "Histogram of time gaps between consecutive KV cache block accesses. Only the most recent accesses are recorded (ring buffer). Sampled metrics (controlled by --kv-cache-metrics-sample).",
            kv_block_reuse_gap_seconds.clone(),
        );

        Self {
            kv_block_lifetime_seconds,
            kv_block_idle_before_evict_seconds,
            kv_block_reuse_gap_seconds,
        }
    }

    pub(super) fn resolve(&self, labels: &EngineLabels) -> KvCacheResidencyHandles {
        KvCacheResidencyHandles {
            kv_block_lifetime_seconds: self.kv_block_lifetime_seconds.get_or_create_owned(labels),
            kv_block_idle_before_evict_seconds: self
                .kv_block_idle_before_evict_seconds
                .get_or_create_owned(labels),
            kv_block_reuse_gap_seconds: self.kv_block_reuse_gap_seconds.get_or_create_owned(labels),
        }
    }
}

pub struct KvCacheResidencyHandles {
    kv_block_lifetime_seconds: HistogramMetric,
    kv_block_idle_before_evict_seconds: HistogramMetric,
    kv_block_reuse_gap_seconds: HistogramMetric,
}

impl KvCacheResidencyHandles {
    pub fn record(&self, events: &[KvCacheEvictionEvent]) {
        for event in events {
            self.kv_block_lifetime_seconds.observe(event.lifetime_seconds);
            self.kv_block_idle_before_evict_seconds.observe(event.idle_seconds);
            for reuse_gap_seconds in &event.reuse_gaps_seconds {
                self.kv_block_reuse_gap_seconds.observe(*reuse_gap_seconds);
            }
        }
    }
}
