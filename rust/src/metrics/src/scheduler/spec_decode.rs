// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Speculative decoding counters.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};

use crate::{EngineLabels, U64Counter};

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EnginePositionLabels {
    pub model_name: String,
    pub engine: u32,
    pub position: u32,
}

/// Per-step iteration decoding stats from scheduler.
///
/// Each scheduler step, statistics on spec decoding performance are aggregated
/// across requests by the scheduler and returned to the frontend in
/// `EngineCoreOutputs -> SchedulerStats`.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/spec_decode/metrics.py#L16-L44>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecDecodingStats {
    /// Configured speculative token count for this scheduler.
    pub num_spec_tokens: u64,
    /// Number of drafted speculative decoding attempts.
    pub num_drafts: u64,
    /// Number of drafted tokens.
    pub num_draft_tokens: u64,
    /// Number of accepted drafted tokens.
    pub num_accepted_tokens: u64,
    /// Accepted drafted tokens counted by draft position.
    pub num_accepted_tokens_per_pos: Vec<u64>,
    /// Drafted tokens counted by draft position.
    #[serde(default)]
    pub num_draft_tokens_per_pos: Vec<u64>,
    /// Number of draft tokens the target model ran a forward pass on.
    #[serde(default)]
    pub num_verified_tokens: u64,
    /// Verified draft tokens counted by draft position.
    #[serde(default)]
    pub num_verified_tokens_per_pos: Vec<u64>,
}

/// Speculative decoding counters.
pub struct SpecDecodeMetrics {
    pub num_drafts: Family<EngineLabels, U64Counter>,
    pub num_draft_tokens: Family<EngineLabels, U64Counter>,
    pub num_accepted_tokens: Family<EngineLabels, U64Counter>,
    pub num_accepted_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,
    pub num_verified_tokens: Family<EngineLabels, U64Counter>,
    pub num_verified_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,
}

impl SpecDecodeMetrics {
    pub(super) fn register(registry: &mut Registry) -> Self {
        let num_drafts = Family::default();
        registry.register(
            "vllm:spec_decode_num_drafts",
            "Number of spec decoding drafts.",
            num_drafts.clone(),
        );

        let num_draft_tokens = Family::default();
        registry.register(
            "vllm:spec_decode_num_draft_tokens",
            "Number of draft tokens.",
            num_draft_tokens.clone(),
        );

        let num_accepted_tokens = Family::default();
        registry.register(
            "vllm:spec_decode_num_accepted_tokens",
            "Number of accepted tokens.",
            num_accepted_tokens.clone(),
        );

        let num_accepted_tokens_per_pos = Family::default();
        registry.register(
            "vllm:spec_decode_num_accepted_tokens_per_pos",
            "Accepted tokens per draft position.",
            num_accepted_tokens_per_pos.clone(),
        );

        let num_verified_tokens = Family::default();
        registry.register(
            "vllm:spec_decode_num_verified_tokens",
            "Number of draft tokens verified by the target.",
            num_verified_tokens.clone(),
        );

        let num_verified_tokens_per_pos = Family::default();
        registry.register(
            "vllm:spec_decode_num_verified_tokens_per_pos",
            "Verified draft tokens per draft position.",
            num_verified_tokens_per_pos.clone(),
        );

        Self {
            num_drafts,
            num_draft_tokens,
            num_accepted_tokens,
            num_accepted_tokens_per_pos,
            num_verified_tokens,
            num_verified_tokens_per_pos,
        }
    }

    pub(super) fn resolve(&self, labels: &EngineLabels) -> SpecDecodeHandles {
        SpecDecodeHandles {
            labels: labels.clone(),
            num_drafts: self.num_drafts.get_or_create_owned(labels),
            num_draft_tokens: self.num_draft_tokens.get_or_create_owned(labels),
            num_accepted_tokens: self.num_accepted_tokens.get_or_create_owned(labels),
            num_accepted_tokens_per_pos: self.num_accepted_tokens_per_pos.clone(),
            num_verified_tokens: self.num_verified_tokens.get_or_create_owned(labels),
            num_verified_tokens_per_pos: self.num_verified_tokens_per_pos.clone(),
        }
    }
}

pub struct SpecDecodeHandles {
    // Base labels of the per-position series.
    labels: EngineLabels,
    num_drafts: U64Counter,
    num_draft_tokens: U64Counter,
    num_accepted_tokens: U64Counter,
    num_accepted_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,
    num_verified_tokens: U64Counter,
    num_verified_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,
}

impl SpecDecodeHandles {
    pub fn record(&self, stats: &SpecDecodingStats) {
        let labels = &self.labels;
        self.num_drafts.inc_by(stats.num_drafts);
        self.num_draft_tokens.inc_by(stats.num_draft_tokens);
        self.num_accepted_tokens.inc_by(stats.num_accepted_tokens);
        self.num_verified_tokens.inc_by(stats.num_verified_tokens);

        for (position, accepted_tokens) in
            stats.num_accepted_tokens_per_pos.iter().copied().enumerate()
        {
            self.num_accepted_tokens_per_pos
                .get_or_create(&EnginePositionLabels {
                    model_name: labels.model_name.clone(),
                    engine: labels.engine,
                    position: position as u32,
                })
                .inc_by(accepted_tokens);
        }
        for (position, verified_tokens) in
            stats.num_verified_tokens_per_pos.iter().copied().enumerate()
        {
            self.num_verified_tokens_per_pos
                .get_or_create(&EnginePositionLabels {
                    model_name: labels.model_name.clone(),
                    engine: labels.engine,
                    position: position as u32,
                })
                .inc_by(verified_tokens);
        }
    }
}
