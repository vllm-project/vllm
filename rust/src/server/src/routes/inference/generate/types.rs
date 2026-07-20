// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::Validate;
use vllm_text::SamplingParams;

use crate::routes::openai::utils::types::{ChatLogProbs, Normalizable, StreamOptions, Usage};

/// vLLM-compatible request type for the token-in/token-out generate API.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct GenerateRequest {
    pub request_id: Option<String>,
    pub model: Option<String>,
    pub token_ids: Vec<u32>,
    pub sampling_params: SamplingParams,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub cache_salt: Option<String>,
    #[serde(default)]
    pub priority: i32,
    pub kv_transfer_params: Option<HashMap<String, Value>>,
    pub ec_transfer_params: Option<HashMap<String, Value>>,
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
