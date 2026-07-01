mod config;

use std::collections::BTreeSet;
use std::sync::Arc;

use tracing::info;
use vllm_tokenizer::{DynTokenizer, HuggingFaceTokenizer, TekkenTokenizer, TiktokenTokenizer};

use self::config::{GenerationConfig, load_generation_config};
pub use self::config::{
    ModelConfig, NamedSpecialToken, SpecialTokens, TokenizerConfig, load_model_config,
    load_tokenizer_config,
};
use crate::backend::{SamplingHints, TextBackend};
use crate::error::Result;
pub use vllm_model_files::{ResolvedModelFiles, TokenizerSource};

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
    /// Generation-config for sampling defaults that may be inherited when the
    /// user does not explicitly override them.
    generation_config: GenerationConfig,
    /// Model vocabulary size from the selected text config.
    model_vocab_size: usize,
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
        let model_vocab_size = model_config.vocab_size()? as usize;
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
            model_vocab_size,
            model_config,
        })
    }

    /// Expose the resolved model files for use by the chat backend to load the
    /// chat template.
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

    fn model_vocab_size(&self) -> usize {
        self.model_vocab_size
    }

    fn model_id(&self) -> &str {
        &self.model_id
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
        })
    }
}

#[cfg(test)]
mod tests {
    use vllm_tokenizer::{TiktokenTokenizer, Tokenizer};

    use super::{ResolvedModelFiles, TokenizerSource};

    #[tokio::test]
    #[ignore = "too slow for CI and requires network access to Hugging Face"]
    async fn tiktoken_real_kimi_k25_tokenizer_files_load_and_handle_special_tokens() {
        let files = ResolvedModelFiles::new("moonshotai/Kimi-K2.5")
            .await
            .expect("resolve real Kimi K2.5 model files");

        let tokenizer_path = match &files.tokenizer {
            TokenizerSource::Tiktoken(path) => path.clone(),
            other => panic!("expected tiktoken tokenizer source, got {other:?}"),
        };

        for backend in [
            TiktokenTokenizer::new_riptoken(&tokenizer_path).expect("load riptoken backend"),
            TiktokenTokenizer::new_tiktoken_rs(&tokenizer_path).expect("load tiktoken-rs backend"),
        ] {
            let think_id = backend.token_to_id("<think>").expect("resolve <think>");
            let end_think_id = backend.token_to_id("</think>").expect("resolve </think>");
            let tool_section_id = backend
                .token_to_id("<|tool_calls_section_begin|>")
                .expect("resolve tool call section marker");
            let contraction_heavy_text =
                "I'm sure it's fine, but I can't say I'd trust that it's what we'd ship.";
            let contraction_heavy_ids = backend.encode(contraction_heavy_text, false).unwrap();

            assert_eq!(
                (think_id, end_think_id, tool_section_id),
                (163606, 163607, 163595)
            );
            assert_eq!(backend.decode(&[think_id], true).unwrap(), "<think>");
            assert_eq!(backend.decode(&[end_think_id], true).unwrap(), "</think>");
            assert_eq!(
                backend.decode(&[tool_section_id], true).unwrap(),
                "<|tool_calls_section_begin|>"
            );

            // This demonstrates that we're using Kimi's custom BPE pattern.
            // With CL100K this will be 23 tokens instead.
            assert_eq!(
                contraction_heavy_ids,
                vec![
                    17172, 3287, 4643, 8201, 11, 996, 374, 8971, 3637, 20020, 8173, 473, 4643,
                    1573, 56229, 13922, 13,
                ]
            );
            assert_eq!(contraction_heavy_ids.len(), 17);
            assert_eq!(
                backend.decode(&contraction_heavy_ids, false).unwrap(),
                contraction_heavy_text
            );

            // Special-looking text that is not actually registered should fail gracefully.
            assert_eq!(backend.token_to_id("◁think▷"), None);
            assert_eq!(backend.token_to_id("<|definitely_not_registered|>"), None);
        }
    }
}
