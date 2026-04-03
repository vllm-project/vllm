mod config;
mod model_files;

use std::collections::BTreeSet;
use std::sync::Arc;

use tracing::info;

use self::config::{GenerationConfig, ModelConfig, load_generation_config, load_model_config};
pub use self::config::{
    HfSpecialTokens, HfTokenizerConfig, NamedSpecialToken, load_tokenizer_config,
};
pub use self::model_files::{ResolvedModelFiles, TokenizerSource};
use crate::backend::{SamplingHints, TextBackend};
use crate::error::Result;
use crate::tokenizers::{DynTokenizer, HuggingFaceTokenizer, TekkenTokenizer, TiktokenTokenizer};

fn load_tokenizer(tokenizer: &TokenizerSource) -> Result<DynTokenizer> {
    match tokenizer {
        TokenizerSource::HuggingFace(path) => Ok(Arc::new(HuggingFaceTokenizer::new(path)?)),
        TokenizerSource::Tiktoken(path) => Ok(Arc::new(TiktokenTokenizer::new(path)?)),
        TokenizerSource::Tekken(path) => Ok(Arc::new(TekkenTokenizer::new(path)?)),
    }
}

/// [`TextBackend`] implementation built on Hugging Face model files.
pub struct HfTextBackend {
    model_id: String,
    files: ResolvedModelFiles,
    tokenizer: DynTokenizer,
    /// Primary EOS handled by engine-core's dedicated EOS path.
    primary_eos_token_id: Option<u32>,
    /// Additional EOS ids that should flow through stop-token handling.
    extra_eos_token_ids: BTreeSet<u32>,
    /// Generation-config for sampling defaults that may be inherited when the user does not
    /// explicitly override them.
    generation_config: GenerationConfig,
    /// Model config (`config.json`).
    model_config: ModelConfig,
}

impl HfTextBackend {
    /// Load the text backend with the given model id.
    pub async fn from_model(model_id: &str) -> Result<Self> {
        let files = ResolvedModelFiles::new(model_id).await?;
        Self::from_resolved_model_files(files, model_id.to_string())
    }

    /// Load the text backend from resolved Hugging Face model files.
    pub fn from_resolved_model_files(files: ResolvedModelFiles, model_id: String) -> Result<Self> {
        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let tokenizer = load_tokenizer(&files.tokenizer)?;
        let primary_eos_token_id = tokenizer_config
            .special_tokens
            .eos_token
            .as_ref()
            .and_then(|token| tokenizer.token_to_id(token.as_str()));

        let model_config = load_model_config(files.config_path.as_deref())?;
        let generation_config = load_generation_config(files.generation_config_path.as_deref())?;
        let mut extra_eos_token_ids = generation_config
            .eos_token_id
            .clone()
            .map(|value| value.into_set())
            .unwrap_or_default();
        if let Some(primary_eos_token_id) = primary_eos_token_id {
            extra_eos_token_ids.remove(&primary_eos_token_id);
        }

        info!(
            model_id,
            "loaded text backend with Hugging Face model files"
        );

        Ok(Self {
            model_id,
            files,
            tokenizer,
            primary_eos_token_id,
            extra_eos_token_ids,
            generation_config,
            model_config,
        })
    }

    /// Expose the resolved model files for use by the chat backend to load the chat template.
    pub fn resolved_model_files(&self) -> &ResolvedModelFiles {
        &self.files
    }
}

impl TextBackend for HfTextBackend {
    fn tokenizer(&self) -> DynTokenizer {
        self.tokenizer.clone()
    }

    fn is_moe(&self) -> bool {
        self.model_config.is_moe()
    }

    fn model_id(&self) -> Option<&str> {
        Some(&self.model_id)
    }

    fn sampling_hints(&self) -> Result<SamplingHints> {
        Ok(SamplingHints {
            primary_eos_token_id: self.primary_eos_token_id,
            extra_eos_token_ids: self.extra_eos_token_ids.clone(),
            default_temperature: self.generation_config.temperature,
            default_top_p: self.generation_config.top_p,
            default_top_k: self.generation_config.top_k,
            default_min_p: self.generation_config.min_p,
            default_repetition_penalty: self.generation_config.repetition_penalty,
            default_max_tokens: self.generation_config.max_new_tokens,
            max_model_len: self.model_config.effective_max_position_embeddings(),
        })
    }
}
