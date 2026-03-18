use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

/// Minimal subset of `tokenizer_config.json` needed by chat/EOS handling.
#[derive(Debug, Default, Deserialize)]
pub(super) struct HfTokenizerConfig {
    #[serde(default)]
    pub eos_token: Option<NamedSpecialToken>,
}

/// Hugging Face named special tokens may be serialized as a string or an
/// object carrying the token content.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum NamedSpecialToken {
    Text(String),
    WithContent { content: String },
}

impl NamedSpecialToken {
    pub(super) fn as_str(&self) -> &str {
        match self {
            Self::Text(value) => value,
            Self::WithContent { content } => content,
        }
    }
}

/// Minimal subset of `generation_config.json` used to recover extra EOS ids.
#[derive(Debug, Default, Deserialize)]
pub(super) struct GenerationConfig {
    #[serde(default)]
    pub eos_token_id: Option<OneOrManyTokenIds>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
}

/// HF generation configs allow either one EOS id or a list of EOS ids.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum OneOrManyTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl OneOrManyTokenIds {
    pub(super) fn into_set(self) -> BTreeSet<u32> {
        match self {
            Self::One(id) => BTreeSet::from([id]),
            Self::Many(ids) => ids.into_iter().collect(),
        }
    }
}

/// Load the tokenizer-side EOS metadata if a config file is present.
pub(super) fn load_tokenizer_config(path: Option<&Path>) -> Result<HfTokenizerConfig> {
    read_json_file(path)
}

/// Load the generation-side EOS metadata if a config file is present.
pub(super) fn load_generation_config(path: Option<&Path>) -> Result<GenerationConfig> {
    read_json_file(path)
}

fn read_json_file<T>(path: Option<&Path>) -> Result<T>
where
    T: for<'de> Deserialize<'de> + Default,
{
    let Some(path) = path else {
        return Ok(T::default());
    };
    let content = fs::read_to_string(path)
        .map_err(|error| Error::Tokenizer(format!("failed to read {}: {error}", path.display())))?;
    serde_json::from_str(&content)
        .map_err(|error| Error::Tokenizer(format!("failed to parse {}: {error}", path.display())))
}
