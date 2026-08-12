// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use llm_multimodal::MediaContentPart;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::Validate;
use vllm_text::SamplingParams;

use crate::routes::openai::utils::types::{
    ChatLogProbs, Normalizable, StreamOptions, StringOrArray, Usage,
};

/// Sampling parameters accepted by `/generate`.
///
/// Python's `GenerateRequest.sampling_params` is the full `vllm.SamplingParams`,
/// so the detokenization-side options travel inside the same object as the
/// sampler-side ones. `vllm_text::SamplingParams` stops at the sampler
/// boundary, so the decode-side half is split back out here and lowered into
/// `TextDecodeOptions`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(try_from = "Map<String, Value>")]
pub struct GenerateSamplingParams {
    #[serde(flatten)]
    pub sampling: SamplingParams,
    pub stop: Option<StringOrArray>,
    pub include_stop_str_in_output: bool,
    pub skip_special_tokens: bool,
    /// Accepted for Python parity. The Rust frontend always runs the shared
    /// detokenizer, so this only gates `stop`.
    pub detokenize: bool,
}

impl Default for GenerateSamplingParams {
    fn default() -> Self {
        Self {
            sampling: SamplingParams::default(),
            stop: None,
            include_stop_str_in_output: false,
            skip_special_tokens: true,
            detokenize: true,
        }
    }
}

impl TryFrom<Map<String, Value>> for GenerateSamplingParams {
    type Error = serde_json::Error;

    /// `#[serde(flatten)]` is used on the serialize side only. On deserialize it
    /// routes the whole object through serde's buffered `Content`, whose map
    /// keys arrive as strings, so a flattened `SamplingParams` would reject the
    /// integer keys of `logit_bias`. Splitting the object by hand keeps those
    /// keys on the plain deserializer.
    fn try_from(mut object: Map<String, Value>) -> Result<Self, Self::Error> {
        let defaults = Self::default();
        Ok(Self {
            stop: take_field::<Option<StringOrArray>>(&mut object, "stop")?.flatten(),
            include_stop_str_in_output: take_field(&mut object, "include_stop_str_in_output")?
                .unwrap_or(defaults.include_stop_str_in_output),
            skip_special_tokens: take_field(&mut object, "skip_special_tokens")?
                .unwrap_or(defaults.skip_special_tokens),
            detokenize: take_field(&mut object, "detokenize")?.unwrap_or(defaults.detokenize),
            sampling: serde_json::from_value(Value::Object(object))?,
        })
    }
}

/// Remove one optional field from a JSON object, tagging errors with its key.
fn take_field<T: serde::de::DeserializeOwned>(
    object: &mut Map<String, Value>,
    key: &str,
) -> Result<Option<T>, serde_json::Error> {
    use serde::de::Error as _;

    object
        .remove(key)
        .map(|value| {
            serde_json::from_value(value)
                .map_err(|error| serde_json::Error::custom(format!("`{key}`: {error}")))
        })
        .transpose()
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
}

/// Mirrors the Python vLLM `GenerateResponse` class.
#[derive(Debug, Clone, Serialize)]
pub(super) struct GenerateResponse {
    pub request_id: String,
    pub choices: Vec<GenerateResponseChoice>,
    pub prompt_logprobs: Option<Vec<Option<HashMap<u32, GenerateLogprob>>>>,
    pub kv_transfer_params: Option<Value>,
    pub ec_transfer_params: Option<Value>,
}

/// Mirrors the Python vLLM `Logprob` class used in prompt-logprobs payloads.
#[derive(Debug, Clone, Serialize)]
pub(super) struct GenerateLogprob {
    pub logprob: f32,
    pub rank: Option<u32>,
    pub decoded_token: Option<String>,
}
