// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::Serialize;
use vllm_engine_core_client::protocol::output::RequestSpecDecodeMetrics;

/// Mirrors the Python vLLM `SpeculativeDecodingMetrics` class.
///
/// Derived from the raw engine accumulator the same way as Python
/// `RequestSpecDecodeMetrics.to_dict`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct SpeculativeDecodingMetrics {
    pub mean_acceptance_length: f64,
    pub draft_acceptance_rate: f64,
    pub acceptance_histogram: Vec<u64>,
    pub num_spec_steps: u64,
    pub num_accepted_draft_tokens: u64,
    pub num_draft_tokens: u64,
    pub num_spec_tokens: u64,
    pub per_step_accepted: Option<Vec<u64>>,
    pub per_step_drafted: Option<Vec<u64>>,
}

impl From<RequestSpecDecodeMetrics> for SpeculativeDecodingMetrics {
    fn from(raw: RequestSpecDecodeMetrics) -> Self {
        let num_spec_steps: u64 = raw.histogram.iter().sum();
        let num_accepted_draft_tokens: u64 =
            (0u64..).zip(&raw.histogram).map(|(accepted, count)| accepted * count).sum();
        let ratio = |num: u64, den: u64| {
            if den == 0 {
                0.0
            } else {
                num as f64 / den as f64
            }
        };
        let detailed = !raw.per_step_accepted.is_empty();
        Self {
            mean_acceptance_length: if num_spec_steps == 0 {
                1.0
            } else {
                1.0 + ratio(num_accepted_draft_tokens, num_spec_steps)
            },
            draft_acceptance_rate: ratio(num_accepted_draft_tokens, raw.num_draft_tokens),
            acceptance_histogram: raw.histogram,
            num_spec_steps,
            num_accepted_draft_tokens,
            num_draft_tokens: raw.num_draft_tokens,
            num_spec_tokens: raw.num_spec_tokens,
            per_step_accepted: detailed.then_some(raw.per_step_accepted),
            per_step_drafted: detailed.then_some(raw.per_step_drafted),
        }
    }
}

/// Streaming form omits detailed fields when summary metrics are requested.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct StreamingSpeculativeDecodingMetrics {
    pub mean_acceptance_length: f64,
    pub draft_acceptance_rate: f64,
    pub acceptance_histogram: Vec<u64>,
    pub num_spec_steps: u64,
    pub num_accepted_draft_tokens: u64,
    pub num_draft_tokens: u64,
    pub num_spec_tokens: u64,
    pub per_step_accepted: Option<Vec<u64>>,
    pub per_step_drafted: Option<Vec<u64>>,
}

impl From<SpeculativeDecodingMetrics> for StreamingSpeculativeDecodingMetrics {
    fn from(metrics: SpeculativeDecodingMetrics) -> Self {
        Self {
            mean_acceptance_length: metrics.mean_acceptance_length,
            draft_acceptance_rate: metrics.draft_acceptance_rate,
            acceptance_histogram: metrics.acceptance_histogram,
            num_spec_steps: metrics.num_spec_steps,
            num_accepted_draft_tokens: metrics.num_accepted_draft_tokens,
            num_draft_tokens: metrics.num_draft_tokens,
            num_spec_tokens: metrics.num_spec_tokens,
            per_step_accepted: metrics.per_step_accepted,
            per_step_drafted: metrics.per_step_drafted,
        }
    }
}
