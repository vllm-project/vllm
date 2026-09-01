// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};
use validator::Validate;
use vllm_text::{Prompt, TruncationSide};

use crate::routes::openai::utils::types::{Normalizable, Usage, default_true};

/// Text or token-ID input accepted by the OpenAI embeddings endpoint.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub(crate) enum EmbeddingInput {
    TokenIds(Vec<u32>),
    TokenIdBatch(Vec<Vec<u32>>),
    Text(String),
    TextBatch(Vec<String>),
}

impl EmbeddingInput {
    pub(super) fn into_prompts(self) -> Vec<Prompt> {
        match self {
            Self::TokenIds(token_ids) => vec![Prompt::TokenIds(token_ids)],
            Self::TokenIdBatch(batch) => batch.into_iter().map(Prompt::TokenIds).collect(),
            Self::Text(text) => vec![Prompt::Text(text)],
            Self::TextBatch(batch) => batch.into_iter().map(Prompt::Text).collect(),
        }
    }
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
pub(crate) enum Endianness {
    #[default]
    Native,
    Big,
    Little,
}

/// OpenAI-compatible embeddings request.
///
/// Mirrors the completion-input variant of Python vLLM's `EmbeddingRequest`.
#[derive(Debug, Clone, Deserialize, Validate)]
pub(crate) struct EmbeddingRequest {
    /// ID of the model to use. An omitted or empty value selects the default.
    pub model: Option<String>,
    /// One text/token-ID prompt or a batch of homogeneous prompts.
    pub input: EmbeddingInput,
    /// Response encoding. `float` returns JSON numbers; `base64` returns the
    /// requested float32 byte representation.
    #[serde(default)]
    pub encoding_format: EncodingFormat,
    /// Reduce the dimensions of embeddings for models that support
    /// Matryoshka representation. `None` lets engine-core resolve the model
    /// default.
    pub dimensions: Option<u32>,
    /// Whether to apply activation to pooler outputs. `None` uses the pooler's
    /// model-aware default.
    pub use_activation: Option<bool>,
    /// Whether tokenization adds special tokens such as BOS.
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
    /// Number of prompt tokens to retain. `-1` uses the available input
    /// budget; `None` leaves the prompt untruncated.
    pub truncate_prompt_tokens: Option<i64>,
    /// Side from which excess prompt tokens are discarded.
    pub truncation_side: Option<TruncationSide>,
    /// Request ID used throughout inference and returned in the response.
    pub request_id: Option<String>,
    /// Scheduling priority; lower values are handled earlier.
    pub priority: Option<i32>,
    /// Random salt used to isolate prefix-cache entries across users.
    pub cache_salt: Option<String>,
    /// Numeric type used for encoded output. The Rust endpoint currently
    /// accepts only `float32`.
    pub embed_dtype: Option<String>,
    /// Byte order used for base64-encoded output.
    #[serde(default)]
    pub endianness: Endianness,
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
