// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use llm_multimodal::MediaContentPart;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::Validate;
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
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Normalizable for GenerateRequest {}

/// Mirrors the Python vLLM `GenerateResponseChoice` class.
///
/// Do not skip serializing `None` fields here: non-streaming response types
/// should serialize `None` as explicit `null`.
///
/// Also deserialized by the derender endpoints. A JSON `null` `token_ids`
/// fails deserialization (400); the derender handler additionally rejects
/// missing/empty `token_ids` with Python's "empty or null token_ids" error.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct GenerateResponseChoice {
    pub index: u32,
    pub logprobs: Option<ChatLogProbs>,
    // Per OpenAI spec the Python default is "stop".
    #[serde(default = "default_finish_reason")]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub token_ids: Vec<u32>,
}

fn default_finish_reason() -> Option<String> {
    Some("stop".to_string())
}

/// Mirrors the Python vLLM `GenerateResponseStreamChoice` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct GenerateResponseStreamChoice {
    pub index: u32,
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub token_ids: Vec<u32>,
}

/// Mirrors the Python vLLM `GenerateStreamResponse` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct GenerateStreamResponse {
    #[serde(default)]
    pub request_id: String,
    pub choices: Vec<GenerateResponseStreamChoice>,
    pub usage: Option<Usage>,
}

/// Engine-wire prompt logprobs: one candidate map per prompt position.
pub(crate) type PromptLogprobMaps = Vec<Option<HashMap<u32, GenerateLogprob>>>;

/// Mirrors the Python vLLM `GenerateResponse` class.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct GenerateResponse {
    #[serde(default)]
    pub request_id: String,
    pub choices: Vec<GenerateResponseChoice>,
    #[serde(default, deserialize_with = "deserialize_prompt_logprob_maps")]
    pub prompt_logprobs: Option<PromptLogprobMaps>,
    pub kv_transfer_params: Option<Value>,
    pub ec_transfer_params: Option<Value>,
}

/// Deserialize prompt-logprob position maps with integer keys.
///
/// Serde's untagged-union buffering (used by the derender request unions)
/// cannot drive `HashMap<u32, _>`'s key parsing, so accept string keys and
/// parse them explicitly.
fn deserialize_prompt_logprob_maps<'de, D>(
    deserializer: D,
) -> Result<Option<PromptLogprobMaps>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = Option::<Vec<Option<HashMap<String, GenerateLogprob>>>>::deserialize(deserializer)?;
    raw.map(|positions| {
        positions
            .into_iter()
            .map(|position| {
                position
                    .map(|candidates| {
                        candidates
                            .into_iter()
                            .map(|(key, value)| {
                                key.parse::<u32>()
                                    .map(|token_id| (token_id, value))
                                    .map_err(serde::de::Error::custom)
                            })
                            .collect::<Result<HashMap<_, _>, _>>()
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .transpose()
}

/// Mirrors the Python vLLM `Logprob` class used in prompt-logprobs payloads.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct GenerateLogprob {
    pub logprob: f32,
    pub rank: Option<u32>,
    pub decoded_token: Option<String>,
}
