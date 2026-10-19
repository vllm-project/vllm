// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Tokenizer and config source options mirroring Python's `--tokenizer`,
//! `--tokenizer-revision`, and `--hf-config-path`, shared by the OpenAI
//! frontend and the render server.

use clap::Args;
use serde::Deserialize;
use vllm_text::backend::hf::ModelSource;

/// Tokenizer and config sources that differ from the model, mirroring Python's
/// `ModelConfig`.
#[derive(Clone, Debug, Default, Args, PartialEq, Eq, Deserialize)]
pub struct ModelSourceArgs {
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
}

impl ModelSourceArgs {
    /// Source of the tokenizer files and the chat template, or `None` when it
    /// is the model at `revision`.
    pub fn tokenizer_source(&self, model: &str, revision: Option<&str>) -> Option<ModelSource> {
        if self.tokenizer.is_none() && self.tokenizer_revision.is_none() {
            return None;
        }
        Some(ModelSource {
            id: self.tokenizer.clone().unwrap_or_else(|| model.to_string()),
            revision: self.tokenizer_revision.clone().or_else(|| revision.map(str::to_string)),
        })
    }

    /// Source of `config.json` and `generation_config.json`, or `None` when it
    /// is the model. Like Python, the config repo is read at the model `revision`.
    pub fn hf_config_source(&self, revision: Option<&str>) -> Option<ModelSource> {
        self.hf_config_path.clone().map(|id| ModelSource {
            id,
            revision: revision.map(str::to_string),
        })
    }

    /// Forward these options to the managed Python engine, which loads the same
    /// tokenizer and config.
    pub fn python_args(&self) -> Vec<String> {
        [
            ("--tokenizer", &self.tokenizer),
            ("--tokenizer-revision", &self.tokenizer_revision),
            ("--hf-config-path", &self.hf_config_path),
        ]
        .into_iter()
        .filter_map(|(flag, value)| Some([flag.to_string(), value.clone()?]))
        .flatten()
        .collect()
    }
}
