// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::routes::openai::utils::types::{Normalizable, Usage, default_true};

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub(crate) enum EmbeddingInput {
    TokenIds(Vec<u32>),
    TokenIdBatch(Vec<Vec<u32>>),
    Text(String),
    TextBatch(Vec<String>),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum EncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum TruncationSide {
    Left,
    #[default]
    Right,
}

#[derive(Debug, Clone, Deserialize, Validate)]
pub(crate) struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: EncodingFormat,
    pub dimensions: Option<u32>,
    pub use_activation: Option<bool>,
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
    pub truncate_prompt_tokens: Option<i64>,
    pub truncation_side: Option<TruncationSide>,
    pub request_id: Option<String>,
    pub priority: Option<i32>,
    pub cache_salt: Option<String>,
    pub embed_dtype: Option<String>,
    pub endianness: Option<String>,
}

impl Normalizable for EmbeddingRequest {}

#[derive(Debug, Clone, Serialize)]
pub(super) struct EmbeddingResponseData {
    pub object: &'static str,
    pub index: usize,
    pub embedding: EmbeddingData,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(super) enum EmbeddingData {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct EmbeddingResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub data: Vec<EmbeddingResponseData>,
    pub usage: Usage,
}
