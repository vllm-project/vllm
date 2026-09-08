// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use crate::error::{ApiError, bail_invalid_request};
use crate::routes::openai::utils::types::Normalizable;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use validator::Validate;
use vllm_llm::PoolingTask;
use vllm_text::{Prompt, TruncationSide};

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum Input {
    Text(String),
    TextBatch(Vec<String>),
    TokenIds(Vec<u32>),
    TokenIdBatch(Vec<Vec<u32>>),
}

impl Input {
    pub(super) fn into_prompts(self) -> Vec<Prompt> {
        match self {
            Self::Text(text) => vec![Prompt::Text(text)],
            Self::TextBatch(batch) => batch.into_iter().map(Prompt::Text).collect(),
            Self::TokenIds(token_ids) => vec![Prompt::TokenIds(token_ids)],
            Self::TokenIdBatch(batch) => batch.into_iter().map(Prompt::TokenIds).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum EncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Debug, Clone, Deserialize, Validate)]
pub(crate) struct PoolingRequest {
    pub model: Option<String>,
    pub input: Input,
    pub task: Option<PoolingTask>,
    pub dimensions: Option<u32>,
    pub use_activation: Option<bool>,
    #[serde(default)]
    pub encoding_format: EncodingFormat,
    pub add_special_tokens: Option<bool>,
    pub truncate_prompt_tokens: Option<i64>,
    pub truncation_side: Option<TruncationSide>,
    pub request_id: Option<String>,
    #[serde(default)]
    pub priority: i32,
    pub cache_salt: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Normalizable for PoolingRequest {}

impl PoolingRequest {
    pub(super) fn validate_options(&self) -> Result<(), ApiError> {
        if self.dimensions == Some(0) {
            bail_invalid_request!(param = "dimensions", "dimensions must be positive");
        }
        // TODO: support chat/multimodal inputs, padding and additional output dtypes.
        for (key, value) in &self.extra {
            let supported = match key.as_str() {
                "user" => true,
                "embed_dtype" => value == "float32",
                "endianness" => {
                    value == "little" || (value == "native" && cfg!(target_endian = "little"))
                }
                "padding" => value.is_null() || value == "do_not_pad",
                _ => false,
            };
            if !supported {
                bail_invalid_request!("pooling parameter `{key}` is not supported");
            }
        }
        Ok(())
    }
}
