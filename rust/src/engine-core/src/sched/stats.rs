// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The scheduler's stats records: ports of `SchedulerStats`,
//! `RequestSpecDecodeMetrics` (`vllm/v1/metrics/stats.py`) and
//! `SpecDecodingStats` (`vllm/v1/spec_decode/metrics.py`).
//!
//! Stats the binding layer fills in from Python objects are not mirrored:
//! `kv_connector_stats` (worker-side stats aggregated with the scheduler-side
//! ones the connector reports), `ec_connector_stats`, `cudagraph_stats`,
//! `perf_stats`, the DP fields (`step_counter`, `current_wave`) and the LoRA
//! adapter counts, which the engine core sets.

use indexmap::IndexMap;

use crate::kv_cache::{KvCacheEvictionEvent, PrefixCacheStats};

/// Stage of a request's async KV load (`KV_FETCH_STAGES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvFetchStage {
    WaitingToStart,
    InProgress,
    CompletedWaiting,
}

impl KvFetchStage {
    /// The stage name, the key of `num_kv_fetch_reqs_by_stage`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WaitingToStart => "waiting_to_start",
            Self::InProgress => "in_progress",
            Self::CompletedWaiting => "completed_waiting",
        }
    }
}

/// Number of requests with an async KV load, by stage
/// (`dict.fromkeys(KV_FETCH_STAGES, 0)`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvFetchCounts {
    pub waiting_to_start: usize,
    pub in_progress: usize,
    pub completed_waiting: usize,
}

impl KvFetchCounts {
    pub(crate) fn get_mut(&mut self, stage: KvFetchStage) -> &mut usize {
        match stage {
            KvFetchStage::WaitingToStart => &mut self.waiting_to_start,
            KvFetchStage::InProgress => &mut self.in_progress,
            KvFetchStage::CompletedWaiting => &mut self.completed_waiting,
        }
    }

    /// The counts in `KV_FETCH_STAGES` order.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, usize)> {
        [
            (KvFetchStage::WaitingToStart, self.waiting_to_start),
            (KvFetchStage::InProgress, self.in_progress),
            (KvFetchStage::CompletedWaiting, self.completed_waiting),
        ]
        .into_iter()
        .map(|(stage, n)| (stage.as_str(), n))
    }
}

/// Prefix cache stats of a KV connector: [`PrefixCacheStats`] plus
/// `hits_by_source`, which only connector stats fill.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConnectorPrefixCacheStats {
    pub stats: PrefixCacheStats,
    /// `hits` split by the cache tier that supplied them, keyed by the
    /// `CacheHitSource` value, in first-seen order.
    pub hits_by_source: IndexMap<String, usize>,
}

/// `CacheHitSource.EXTERNAL_UNSPECIFIED.value`.
pub const CACHE_HIT_SOURCE_EXTERNAL_UNSPECIFIED: &str = "external_unspecified";

impl ConnectorPrefixCacheStats {
    /// Aggregate request information into the stats.
    pub fn record(
        &mut self,
        num_tokens: usize,
        num_hits: usize,
        preempted: bool,
        hits_by_source: Option<Vec<(String, usize)>>,
    ) {
        self.stats.record(num_tokens, num_hits, preempted);
        if preempted {
            return;
        }
        let Some(mut hits_by_source) = hits_by_source else {
            return;
        };
        if hits_by_source.iter().map(|(_, n)| n).sum::<usize>() != num_hits {
            // TODO: Python logs a warning once.
            hits_by_source = vec![(CACHE_HIT_SOURCE_EXTERNAL_UNSPECIFIED.to_owned(), num_hits)];
        }
        for (source, num_source_hits) in hits_by_source {
            if num_source_hits > 0 {
                *self.hits_by_source.entry(source).or_default() += num_source_hits;
            }
        }
    }
}

/// Per-step iteration decoding stats from scheduler.
///
/// Each scheduler step, statistics on spec decoding performance are
/// aggregated across requests by the scheduler and returned to the
/// frontend in EngineCoreOutputs->SchedulerStats.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecDecodingStats {
    pub num_spec_tokens: usize,
    pub num_drafts: usize,
    pub num_draft_tokens: usize,
    pub num_accepted_tokens: usize,
    pub num_accepted_tokens_per_pos: Vec<usize>,
    pub num_draft_tokens_per_pos: Vec<usize>,
}

impl SpecDecodingStats {
    pub fn new(num_spec_tokens: usize) -> Self {
        Self {
            num_spec_tokens,
            num_drafts: 0,
            num_draft_tokens: 0,
            num_accepted_tokens: 0,
            num_accepted_tokens_per_pos: vec![0; num_spec_tokens],
            num_draft_tokens_per_pos: vec![0; num_spec_tokens],
        }
    }

    pub fn observe_draft(&mut self, num_draft_tokens: usize, num_accepted_tokens: usize) {
        self.num_drafts += 1;
        self.num_draft_tokens += num_draft_tokens;
        self.num_accepted_tokens += num_accepted_tokens;
        assert!(num_accepted_tokens <= self.num_spec_tokens);
        for i in 0..num_accepted_tokens {
            self.num_accepted_tokens_per_pos[i] += 1;
        }
        for i in 0..num_draft_tokens {
            self.num_draft_tokens_per_pos[i] += 1;
        }
    }
}

/// `ObservabilityConfig.per_request_spec_decode_metrics`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpecDecodeMetricsLevel {
    #[default]
    None,
    Summary,
    Detailed,
}

/// Per-output-sequence speculative-decoding statistics accumulator.
///
/// Accumulates, over one sequence's verify steps, a histogram of accepted
/// draft-token counts (`j`, draft-only) and the total number of proposed
/// draft tokens. When `detailed` is requested it also records the ordered
/// per-step accepted/proposed sequences (`summary` omits them).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestSpecDecodeMetrics {
    /// Configured `num_speculative_tokens` (the max `k`); also the
    /// histogram's upper bound.
    pub num_spec_tokens: usize,
    /// Dense counts indexed by accepted draft tokens `j` (length
    /// `num_spec_tokens + 1`).
    pub histogram: Vec<usize>,
    /// Total proposed draft tokens, after the grammar-invalidated
    /// (`num_invalid_spec_tokens`) adjustment.
    pub num_draft_tokens: usize,
    /// Ordered accepted-draft count per verify step (`detailed` only; empty
    /// otherwise).
    pub per_step_accepted: Vec<usize>,
    /// Ordered proposed-draft count per verify step (`detailed` only; empty
    /// otherwise).
    pub per_step_drafted: Vec<usize>,
}

impl RequestSpecDecodeMetrics {
    pub fn new(num_spec_tokens: usize) -> Self {
        Self {
            num_spec_tokens,
            histogram: vec![0; num_spec_tokens + 1],
            num_draft_tokens: 0,
            per_step_accepted: Vec::new(),
            per_step_drafted: Vec::new(),
        }
    }

    pub fn observe(&mut self, num_draft_tokens: usize, num_accepted: usize, detailed: bool) {
        self.histogram[num_accepted] += 1;
        self.num_draft_tokens += num_draft_tokens;
        if detailed {
            self.per_step_accepted.push(num_accepted);
            self.per_step_drafted.push(num_draft_tokens);
        }
    }
}

/// Stats associated with the scheduler.
#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerStats {
    pub num_running_reqs: usize,
    /// waiting requests not deferred
    pub num_waiting_reqs: usize,
    /// waiting requests deferred by transient constraints or a blocked status
    pub num_skipped_waiting_reqs: usize,
    pub num_kv_fetch_reqs_by_stage: KvFetchCounts,
    pub kv_cache_usage: f64,
    pub prefix_cache_stats: PrefixCacheStats,
    pub connector_prefix_cache_stats: Option<ConnectorPrefixCacheStats>,
    pub kv_cache_eviction_events: Vec<KvCacheEvictionEvent>,
    pub spec_decoding_stats: Option<SpecDecodingStats>,
}
