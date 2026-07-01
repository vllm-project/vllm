use std::path::Path;

use serde::Deserialize;

use crate::error::Result;
use crate::json::read_json_file;

/// Minimal subset of `tokenizer_config.json` needed by tokenizer selection.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub(crate) struct TokenizerConfig {
    /// The `tokenizer_class` field from HuggingFace tokenizer configs. Some
    /// tiktoken-based models (e.g. DeepSeek, Kimi K2) set this to a value
    /// containing "Tiktoken" which can be used as a hint for backend
    /// selection.
    pub tokenizer_class: Option<String>,
}

pub(crate) fn load_tokenizer_config(path: Option<&Path>) -> Result<TokenizerConfig> {
    read_json_file(path)
}
