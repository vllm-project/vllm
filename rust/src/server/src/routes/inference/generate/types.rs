// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use llm_multimodal::MediaContentPart;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::Validate;
use vllm_engine_core_client::protocol::output::RequestSpecDecodeMetrics;
use vllm_text::SamplingParams;

use crate::routes::openai::utils::types::{ChatLogProbs, Normalizable, StreamOptions, Usage};

/// Sampling parameters for the token-in/token-out generate API.
///
/// Wraps [`SamplingParams`] to additionally capture `n`, which the shared
/// northbound type intentionally omits (parallel sampling is handled by
/// higher layers, and the Rust frontend does not implement it). Capturing it
/// here lets validation reject `n > 1` explicitly instead of silently
/// dropping the key and returning a single choice.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerateSamplingParams {
    /// Number of output sequences to generate. Only `1` is supported.
    pub n: Option<u32>,
    /// The supported sampling parameters, lowered to the engine.
    #[serde(flatten)]
    pub inner: SamplingParams,
}

/// vLLM-compatible request type for the token-in/token-out generate API.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct GenerateRequest {
    pub request_id: Option<String>,
    pub model: Option<String>,
    pub token_ids: Vec<u32>,
    pub sampling_params: GenerateSamplingParams,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub cache_salt: Option<String>,
    #[serde(default)]
    pub priority: i32,
    pub kv_transfer_params: Option<HashMap<String, Value>>,
    pub ec_transfer_params: Option<HashMap<String, Value>>,
    /// Raw multimodal input; server resolves media. Mutually exclusive with `features`.
    pub content_parts: Option<Vec<MediaContentPart>>,
    pub return_token_ids: Option<bool>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Normalizable for GenerateRequest {}

/// Mirrors the Python vLLM `GenerateResponseChoice` class.
///
/// Do not skip serializing `None` fields here: non-streaming response types
/// should serialize `None` as explicit `null`.
#[derive(Debug, Clone, Serialize)]
pub(super) struct GenerateResponseChoice {
    pub index: u32,
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>,
    pub token_ids: Vec<u32>,
}

/// Mirrors the Python vLLM `GenerateResponseStreamChoice` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub(super) struct GenerateResponseStreamChoice {
    pub index: u32,
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>,
    pub token_ids: Vec<u32>,
}

/// Mirrors the Python vLLM `GenerateStreamResponse` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub(super) struct GenerateStreamResponse {
    pub request_id: String,
    pub choices: Vec<GenerateResponseStreamChoice>,
    pub usage: Option<Usage>,
    pub prompt_token_ids: Option<Vec<u32>>,
    pub mm_placeholders: Option<MultiModalPlaceholders>,
    pub metrics: Option<PerRequestMetrics<StreamingSpeculativeDecodingMetrics>>,
}

/// Mirrors the Python vLLM `GenerateResponse` class.
#[derive(Debug, Clone, Serialize)]
pub(super) struct GenerateResponse {
    pub request_id: String,
    pub choices: Vec<GenerateResponseChoice>,
    pub prompt_logprobs: Option<Vec<Option<HashMap<u32, GenerateLogprob>>>>,
    pub prompt_token_ids: Option<Vec<u32>>,
    pub mm_placeholders: Option<MultiModalPlaceholders>,
    pub kv_transfer_params: Option<Value>,
    pub ec_transfer_params: Option<Value>,
    pub metrics: Option<PerRequestMetrics<SpeculativeDecodingMetrics>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct PerRequestMetrics<T> {
    pub speculative_decoding: T,
}

/// Mirrors the Python vLLM `SpeculativeDecodingMetrics` class.
///
/// Derived from the raw engine accumulator the same way as Python
/// `RequestSpecDecodeMetrics.to_dict`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct SpeculativeDecodingMetrics {
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
pub(super) struct StreamingSpeculativeDecodingMetrics {
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

pub(super) type MultiModalPlaceholders = HashMap<String, Vec<PlaceholderRangeInfo>>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct PlaceholderRangeInfo {
    pub offset: usize,
    pub length: usize,
}

/// Mirrors the Python vLLM `Logprob` class used in prompt-logprobs payloads.
#[derive(Debug, Clone, Serialize)]
pub(super) struct GenerateLogprob {
    pub logprob: f32,
    pub rank: Option<u32>,
    pub decoded_token: Option<String>,
}
