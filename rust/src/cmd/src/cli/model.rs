// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Model, tokenizer, and config source options mirroring Python's `model`,
//! `--revision`, `--tokenizer`, `--tokenizer-revision`, `--hf-config-path`,
//! `--skip-tokenizer-init`, and `--hf-token`, shared by the OpenAI frontend and
//! the render server.

use std::str::FromStr;

use clap::Args;
use clap::builder::TypedValueParser as _;
use serde::Deserialize;
use vllm_text::backend::hf::{HfToken, ModelFileSources, ModelSource};

use super::unsupported::Fixed;

/// The model and the sources of its tokenizer and config files, mirroring
/// Python's `ModelConfig`.
#[derive(Clone, Debug, Args, PartialEq, Eq, Deserialize)]
pub struct ModelArgs {
    /// Model identifier or local model directory used for backend loading and
    /// public model ID.
    #[arg(value_name = "MODEL")]
    #[serde(rename = "model_tag")]
    pub id: String,

    /// Model revision on the Hugging Face Hub (branch, tag, or commit SHA).
    #[arg(long)]
    #[serde(default)]
    pub revision: Option<String>,

    /// Name or path of the Hugging Face tokenizer to use. If unspecified, model
    /// name or path will be used.
    #[arg(long)]
    #[serde(default)]
    pub tokenizer: Option<String>,

    /// The specific revision to use for the tokenizer on the Hugging Face Hub.
    /// It can be a branch name, a tag name, or a commit id. If unspecified,
    /// `--revision` will be used.
    #[arg(long)]
    #[serde(default)]
    pub tokenizer_revision: Option<String>,

    /// Name or path of the Hugging Face config to use. If unspecified, model
    /// name or path will be used.
    #[arg(long)]
    #[serde(default)]
    pub hf_config_path: Option<String>,

    /// Skip initialization of tokenizer and detokenizer. Expects valid
    /// `prompt_token_ids` and `None` for prompt from the input. The generated
    /// output will contain token ids.
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    #[serde(default)]
    pub skip_tokenizer_init: bool,

    /// The token to use as HTTP bearer authorization for remote files. If
    /// `True`, will use the token generated when running `hf auth login`
    /// (stored in `~/.cache/huggingface/token`).
    #[arg(long, default_missing_value = "true", num_args = 0..=1)]
    #[serde(default)]
    pub hf_token: Option<HfTokenArg>,

    /// Download models and tokenizers from ModelScope instead of the Hugging
    /// Face Hub. Like Python, it is set only through `VLLM_USE_MODELSCOPE`.
    /// The Rust frontend fetches model files from the Hub only, so only off is
    /// accepted.
    #[arg(
        long,
        hide = true,
        env = "VLLM_USE_MODELSCOPE",
        value_parser = super::parse_truthy_flag.try_map(Fixed::<false>::try_from)
    )]
    #[serde(default)]
    pub use_modelscope: Option<Fixed<false>>,
}

/// `--hf-token`: Python's `bool | str`, where a bare flag (`true`) means the
/// stored login token and any other value is the token itself.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(try_from = "HfTokenJson")]
pub enum HfTokenArg {
    /// Use `HF_TOKEN` or the stored login token, as without the flag.
    Login,
    /// Use this token.
    Token(HfToken),
}

impl From<String> for HfTokenArg {
    fn from(value: String) -> Self {
        match value.as_str() {
            "true" => Self::Login,
            _ => Self::Token(HfToken(value)),
        }
    }
}

impl FromStr for HfTokenArg {
    type Err = std::convert::Infallible;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(value.to_string().into())
    }
}

/// The `--args-json` form of `--hf-token`: `true` or a token string.
#[derive(Deserialize)]
#[serde(untagged)]
enum HfTokenJson {
    Flag(bool),
    Token(String),
}

impl TryFrom<HfTokenJson> for HfTokenArg {
    type Error = &'static str;

    fn try_from(value: HfTokenJson) -> Result<Self, Self::Error> {
        match value {
            HfTokenJson::Flag(true) => Ok(Self::Login),
            HfTokenJson::Flag(false) => Err("hf_token=false is not supported"),
            HfTokenJson::Token(token) => Ok(token.into()),
        }
    }
}

impl ModelArgs {
    /// Where each group of model files is read from. Like Python, the tokenizer
    /// defaults to the model and `--revision`, and the config repo is read at
    /// `--revision`.
    pub fn file_sources(&self) -> ModelFileSources {
        let tokenizer =
            (self.tokenizer.is_some() || self.tokenizer_revision.is_some()).then(|| ModelSource {
                id: self.tokenizer.clone().unwrap_or_else(|| self.id.clone()),
                revision: self.tokenizer_revision.clone().or_else(|| self.revision.clone()),
            });
        let hf_config = self.hf_config_path.clone().map(|id| ModelSource {
            id,
            revision: self.revision.clone(),
        });

        ModelFileSources {
            model: ModelSource {
                id: self.id.clone(),
                revision: self.revision.clone(),
            },
            tokenizer,
            hf_config,
            skip_tokenizer_init: self.skip_tokenizer_init,
            hf_token: match &self.hf_token {
                Some(HfTokenArg::Token(token)) => Some(token.clone()),
                Some(HfTokenArg::Login) | None => None,
            },
        }
    }

    /// Forward the revision and source options to the managed Python engine,
    /// which loads the same model, tokenizer, and config. With
    /// `--skip-tokenizer-init` the engine skips its tokenizer too, which it
    /// would otherwise load for structured outputs.
    pub fn python_args(&self) -> Vec<String> {
        [
            ("--revision", &self.revision),
            ("--tokenizer", &self.tokenizer),
            ("--tokenizer-revision", &self.tokenizer_revision),
            ("--hf-config-path", &self.hf_config_path),
        ]
        .into_iter()
        .filter_map(|(flag, value)| Some([flag.to_string(), value.clone()?]))
        .flatten()
        .chain(self.skip_tokenizer_init.then(|| "--skip-tokenizer-init".to_string()))
        .chain(self.hf_token.iter().flat_map(|hf_token| match hf_token {
            HfTokenArg::Login => vec!["--hf-token".to_string()],
            HfTokenArg::Token(HfToken(token)) => vec!["--hf-token".to_string(), token.clone()],
        }))
        .collect()
    }
}
