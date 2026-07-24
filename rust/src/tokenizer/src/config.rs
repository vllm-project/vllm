// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Hugging Face `tokenizer_config.json` artifact types.
//!
//! These types describe the repository file format independently of the
//! tokenizer backend selected to consume the model artifacts.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport as _;

use crate::Result;

/// Minimal subset of `tokenizer_config.json` needed by chat/EOS handling.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct HfTokenizerConfig {
    /// Named special tokens exposed to chat templates and EOS handling.
    #[serde(flatten)]
    pub special_tokens: HfSpecialTokens,
    /// Default chat template embedded in the tokenizer config.
    pub chat_template: Option<String>,
    /// The `tokenizer_class` field from HuggingFace tokenizer configs. Some
    /// tiktoken-based models (e.g. DeepSeek, Kimi K2) set this to a value
    /// containing "Tiktoken" which can be used as a hint for backend
    /// selection.
    pub tokenizer_class: Option<String>,
}

/// Hugging Face named special tokens may be serialized as a string or an
/// object carrying the token content.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum NamedSpecialToken {
    /// Token serialized directly as text.
    Text(String),
    /// Token serialized as an object with a `content` field.
    WithContent {
        /// Token text.
        content: String,
    },
}

impl Serialize for NamedSpecialToken {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl From<NamedSpecialToken> for String {
    fn from(value: NamedSpecialToken) -> Self {
        match value {
            NamedSpecialToken::Text(string) => string,
            NamedSpecialToken::WithContent { content } => content,
        }
    }
}

impl NamedSpecialToken {
    /// Return the token text for either supported config representation.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Text(value) => value,
            Self::WithContent { content } => content,
        }
    }
}

/// Minimal set of special-token entries needed by chat/EOS handling.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct HfSpecialTokens {
    /// Beginning-of-sequence token.
    pub bos_token: Option<NamedSpecialToken>,
    /// End-of-sequence token.
    pub eos_token: Option<NamedSpecialToken>,
    /// Unknown-token marker.
    pub unk_token: Option<NamedSpecialToken>,
    /// Padding token.
    pub pad_token: Option<NamedSpecialToken>,
}

impl HfSpecialTokens {
    /// Returns true if we don't discover any special tokens in the config.
    pub fn is_empty(&self) -> bool {
        self.bos_token.is_none()
            && self.eos_token.is_none()
            && self.unk_token.is_none()
            && self.pad_token.is_none()
    }
}

/// Load the tokenizer-side metadata if a config file is present.
pub fn load_tokenizer_config(path: Option<&Path>) -> Result<HfTokenizerConfig> {
    let Some(path) = path else {
        return Ok(HfTokenizerConfig::default());
    };
    let content = fs::read_to_string(path).map_err(|error| {
        tokenizer_error!("failed to read {}: {}", path.display(), error.as_report())
    })?;
    serde_json::from_str(&content).map_err(|error| {
        tokenizer_error!("failed to parse {}: {}", path.display(), error.as_report())
    })
}
