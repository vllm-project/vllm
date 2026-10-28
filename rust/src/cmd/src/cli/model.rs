// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Model, tokenizer, and config source options mirroring Python's `model`,
//! `--revision`, `--tokenizer`, `--tokenizer-revision`, and `--hf-config-path`,
//! shared by the OpenAI frontend and the render server.

use clap::Args;
use serde::Deserialize;
use vllm_text::backend::hf::{ModelFileSources, ModelSource};

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
        }
    }

    /// Forward the revision and source options to the managed Python engine,
    /// which loads the same model, tokenizer, and config.
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
        .collect()
    }
}
