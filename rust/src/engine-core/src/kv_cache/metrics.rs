// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `kv_cache_metrics.py` and the `PrefixCacheStats` of `vllm/v1/metrics/stats.py`.

use std::collections::VecDeque;
use std::time::Instant;

use rustc_hash::FxHashMap;

use super::block::BlockId;

/// A sampled block's residency when it was evicted (`KVCacheEvictionEvent`).
// The wire type of a sampled eviction, which the scheduler reports.
pub use vllm_engine_core_protocol::stats::KvCacheEvictionEvent;

/// Tracks lifecycle metrics for a single KV cache block.
#[derive(Clone, Debug)]
struct BlockMetricsState {
    birth_time: Instant,
    last_access: Instant,
    /// Bounded to prevent unbounded growth if a block is accessed many times.
    access_history: VecDeque<Instant>,
}

const ACCESS_HISTORY_LEN: usize = 4;

impl BlockMetricsState {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            birth_time: now,
            last_access: now,
            access_history: VecDeque::with_capacity(ACCESS_HISTORY_LEN),
        }
    }

    fn record_access(&mut self) {
        let now = Instant::now();
        self.last_access = now;
        if self.access_history.len() == ACCESS_HISTORY_LEN {
            self.access_history.pop_front();
        }
        self.access_history.push_back(now);
    }

    fn reuse_gaps_seconds(&self) -> Vec<f64> {
        if self.access_history.len() < 2 {
            return Vec::new();
        }
        self.access_history
            .iter()
            .zip(self.access_history.iter().skip(1))
            .map(|(a, b)| (*b - *a).as_secs_f64())
            .collect()
    }
}

/// Collects KV cache residency metrics with sampling.
///
/// Sampling is random (as in Python), so it never affects allocation.
#[derive(Clone, Debug)]
pub struct KvCacheMetricsCollector {
    pub sample_rate: f64,
    block_metrics: FxHashMap<BlockId, BlockMetricsState>,
    eviction_events: Vec<KvCacheEvictionEvent>,
    rng_state: u64,
}

impl KvCacheMetricsCollector {
    pub fn new(sample_rate: f64) -> Self {
        assert!(
            0.0 < sample_rate && sample_rate <= 1.0,
            "sample_rate must be in (0, 1.0], got {sample_rate}"
        );
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9e37_79b9_7f4a_7c15);
        Self {
            sample_rate,
            block_metrics: FxHashMap::default(),
            eviction_events: Vec::new(),
            rng_state: seed | 1,
        }
    }

    /// Uniform sample in [0, 1) from xorshift64*.
    fn random(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        (x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn should_sample_block(&mut self) -> bool {
        self.random() < self.sample_rate
    }

    pub fn on_block_allocated(&mut self, block: BlockId) {
        if self.should_sample_block() {
            self.block_metrics.insert(block, BlockMetricsState::new());
        }
    }

    pub fn on_block_accessed(&mut self, block: BlockId) {
        if let Some(metrics) = self.block_metrics.get_mut(&block) {
            metrics.record_access();
        }
    }

    pub fn on_block_evicted(&mut self, block: BlockId) {
        let Some(metrics) = self.block_metrics.remove(&block) else {
            return;
        };
        let now = Instant::now();
        self.eviction_events.push(KvCacheEvictionEvent {
            lifetime_seconds: (now - metrics.birth_time).as_secs_f64(),
            idle_seconds: (now - metrics.last_access).as_secs_f64(),
            reuse_gaps_seconds: metrics.reuse_gaps_seconds(),
        });
    }

    /// Clear all state on cache reset.
    pub fn reset(&mut self) {
        self.block_metrics.clear();
        self.eviction_events.clear();
    }

    pub fn drain_events(&mut self) -> Vec<KvCacheEvictionEvent> {
        std::mem::take(&mut self.eviction_events)
    }
}

/// Stores prefix cache hit statistics.
/// - `reset`: Whether `reset_prefix_cache` was invoked.
/// - `queries`: Refers to the number of tokens that were queried.
// TODO: `hits_by_source` is only filled for connector stats, never by the KV
// cache manager.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    /// Whether the cache was reset.
    pub reset: bool,
    /// The number of requests in this update.
    pub requests: usize,
    /// The number of queries in these requests.
    pub queries: usize,
    /// The number of hits in these requests.
    pub hits: usize,
    /// The number of previously preempted requests in this update.
    pub preempted_requests: usize,
    /// The `queries` number for preempted requests.
    pub preempted_queries: usize,
    /// The `hits` number for preempted requests.
    pub preempted_hits: usize,
}

impl PrefixCacheStats {
    /// Aggregate request information into the stats.
    pub fn record(&mut self, num_tokens: usize, num_hits: usize, preempted: bool) {
        if preempted {
            // Previously preempted request
            self.preempted_requests += 1;
            self.preempted_queries += num_tokens;
            self.preempted_hits += num_hits;
            return;
        }
        // New request
        self.requests += 1;
        self.queries += num_tokens;
        self.hits += num_hits;
    }
}
