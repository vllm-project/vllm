// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Prefix-cache counters, including the connector-backed external cache path.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};

use crate::{EngineLabels, U64Counter};

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct CacheHitSourceLabels {
    pub model_name: String,
    pub engine: u32,
    pub source: &'static str,
}

/// Stores cache hit statistics.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L18-L32>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BaseCacheStats {
    /// Whether the cache was reset.
    pub reset: bool,
    /// The number of requests in this update.
    pub requests: u64,
    /// The number of queries in these requests.
    pub queries: u64,
    /// The number of hits in these requests.
    pub hits: u64,
}

/// Cache-hit tokens per source tier. Python sends a sparse
/// `dict[CacheHitSource, int]`, so absent tiers default to zero, and only
/// non-zero tiers are serialized.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CachedTokensBySource {
    #[serde(skip_serializing_if = "is_zero")]
    pub device: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub host: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub p2p: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub disk: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub external_unspecified: u64,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

impl CachedTokensBySource {
    /// Label values, in the same order as [`Self::counts`].
    pub const SOURCES: [&'static str; 5] =
        ["device", "host", "p2p", "disk", "external_unspecified"];

    pub fn counts(&self) -> [u64; 5] {
        [
            self.device,
            self.host,
            self.p2p,
            self.disk,
            self.external_unspecified,
        ]
    }
}

/// Stores prefix cache hit statistics.
/// - `reset`: Whether `reset_prefix_cache` was invoked.
/// - `queries`: Refers to the number of tokens that were queried.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L114-L143>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PrefixCacheStats {
    /// Embedded base cache counters and reset flag.
    #[serde(flatten)]
    pub base: BaseCacheStats,
    /// The number of previously preempted requests in this update.
    pub preempted_requests: u64,
    /// The `queries` number for preempted requests.
    pub preempted_queries: u64,
    /// The `hits` number for preempted requests.
    pub preempted_hits: u64,
    /// `hits` split by the cache tier that supplied them (connector stats only).
    #[serde(default)]
    pub hits_by_source: CachedTokensBySource,
}

/// Prefix-cache counters, including the connector-backed external cache path.
pub struct PrefixCacheMetrics {
    pub queries: Family<EngineLabels, U64Counter>,
    pub hits: Family<EngineLabels, U64Counter>,
    pub external_queries: Family<EngineLabels, U64Counter>,
    pub external_hits: Family<EngineLabels, U64Counter>,
    pub prompt_tokens_cached_by_source: Family<CacheHitSourceLabels, U64Counter>,
}

impl PrefixCacheMetrics {
    pub(super) fn register(registry: &mut Registry) -> Self {
        let queries = Family::default();
        registry.register(
            "vllm:prefix_cache_queries",
            "Prefix cache queries, in terms of number of queried tokens",
            queries.clone(),
        );

        let hits = Family::default();
        registry.register(
            "vllm:prefix_cache_hits",
            "Prefix cache hits, in terms of number of cached tokens.",
            hits.clone(),
        );

        let external_queries = Family::default();
        registry.register(
            "vllm:external_prefix_cache_queries",
            "External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens.",
            external_queries.clone(),
        );

        let external_hits = Family::default();

        let prompt_tokens_cached_by_source = Family::default();
        registry.register(
            "vllm:prompt_tokens_cached_by_source",
            "Prefix-cache hit tokens by the cache tier that supplied them: device (local HBM), host (offloaded to DRAM), disk, p2p (transferred from another vLLM instance) or external_unspecified. Counted at admission; sums to vllm:prefix_cache_hits + vllm:external_prefix_cache_hits.",
            prompt_tokens_cached_by_source.clone(),
        );
        registry.register(
            "vllm:external_prefix_cache_hits",
            "External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens.",
            external_hits.clone(),
        );

        Self {
            queries,
            hits,
            external_queries,
            external_hits,
            prompt_tokens_cached_by_source,
        }
    }

    pub(super) fn resolve(&self, labels: &EngineLabels) -> PrefixCacheHandles {
        PrefixCacheHandles {
            queries: self.queries.get_or_create_owned(labels),
            hits: self.hits.get_or_create_owned(labels),
            external_queries: self.external_queries.get_or_create_owned(labels),
            external_hits: self.external_hits.get_or_create_owned(labels),
            prompt_tokens_cached_by_source: CachedTokensBySource::SOURCES.map(|source| {
                self.prompt_tokens_cached_by_source.get_or_create_owned(&CacheHitSourceLabels {
                    model_name: labels.model_name.clone(),
                    engine: labels.engine,
                    source,
                })
            }),
        }
    }
}

pub struct PrefixCacheHandles {
    queries: U64Counter,
    hits: U64Counter,
    external_queries: U64Counter,
    external_hits: U64Counter,
    /// One handle per `CachedTokensBySource::SOURCES` entry.
    prompt_tokens_cached_by_source: [U64Counter; 5],
}

impl PrefixCacheHandles {
    pub fn record(&self, local: &PrefixCacheStats, connector: Option<&PrefixCacheStats>) {
        self.queries.inc_by(local.base.queries);
        self.hits.inc_by(local.base.hits);
        // Local prefix-cache hits are the `device` tier.
        self.prompt_tokens_cached_by_source[0].inc_by(local.base.hits);

        if let Some(connector) = connector {
            self.external_queries.inc_by(connector.base.queries);
            self.external_hits.inc_by(connector.base.hits);
            let counts = connector.hits_by_source.counts();
            for (handle, count) in self.prompt_tokens_cached_by_source.iter().zip(counts) {
                handle.inc_by(count);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CachedTokensBySource, PrefixCacheStats};

    #[test]
    fn prefix_cache_stats_hits_by_source_defaults_and_decodes() {
        // Engines without per-source attribution omit the field entirely.
        let legacy = serde_json::json!({
            "reset": false, "requests": 1, "queries": 8, "hits": 4,
            "preempted_requests": 0, "preempted_queries": 0, "preempted_hits": 0
        });
        let wire = rmp_serde::to_vec_named(&legacy).unwrap();
        let stats: PrefixCacheStats = rmp_serde::from_slice(&wire).unwrap();
        assert_eq!(stats.base.hits, 4);
        assert_eq!(stats.hits_by_source, CachedTokensBySource::default());

        let payload = serde_json::json!({
            "reset": false, "requests": 1, "queries": 16, "hits": 12,
            "preempted_requests": 0, "preempted_queries": 0, "preempted_hits": 0,
            "hits_by_source": {"host": 8, "p2p": 4}
        });
        let wire = rmp_serde::to_vec_named(&payload).unwrap();
        let stats: PrefixCacheStats = rmp_serde::from_slice(&wire).unwrap();
        assert_eq!(stats.hits_by_source.counts(), [0, 8, 4, 0, 0]);
        assert_eq!(CachedTokensBySource::SOURCES[1], "host");

        // Serialized as sparsely as Python sends it.
        let encoded: serde_json::Value =
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&stats).unwrap()).unwrap();
        assert_eq!(encoded, payload);
    }
}
