mod config;
mod model_files;

use std::collections::BTreeSet;
use std::sync::Arc;

use tracing::info;
use vllm_tokenizer::{
    CacheConfig, CachedTokenizer, DynTokenizer, HuggingFaceTokenizer, TekkenTokenizer,
    TiktokenTokenizer,
};

use self::config::{GenerationConfig, load_generation_config};
pub use self::config::{
    HfSpecialTokens, HfTokenizerConfig, ModelConfig, NamedSpecialToken, load_model_config,
    load_tokenizer_config,
};
pub use self::model_files::{ResolvedModelFiles, TokenizerSource};
use crate::backend::{SamplingHints, TextBackend};
use crate::error::Result;

/// Environment variable to enable the tokenizer encode cache.
///
/// Set `VLLM_RS_ENABLE_TOKENIZER_CACHE=1` to wrap the tokenizer with a
/// DashMap-based L0 whole-string cache (inspired by `llm-tokenizer`).
/// Leave unset or set to `0` to use the original uncached tokenizer.
///
/// Optional tuning knobs (only effective when the cache is enabled):
/// - `VLLM_RS_TOKENIZER_CACHE_SIZE`: max entries (default 10 000)
/// - `VLLM_RS_TOKENIZER_CACHE_MAX_KEY_BYTES`: strings longer than this
///   bypass the cache entirely to avoid overhead on long unique prompts
///   (default 2048).
const ENABLE_TOKENIZER_CACHE_ENV: &str = "VLLM_RS_ENABLE_TOKENIZER_CACHE";
const TOKENIZER_CACHE_SIZE_ENV: &str = "VLLM_RS_TOKENIZER_CACHE_SIZE";
const TOKENIZER_CACHE_MAX_KEY_BYTES_ENV: &str = "VLLM_RS_TOKENIZER_CACHE_MAX_KEY_BYTES";

fn load_tokenizer(tokenizer: &TokenizerSource) -> Result<DynTokenizer> {
    let enable_cache = std::env::var_os(ENABLE_TOKENIZER_CACHE_ENV)
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);

    if enable_cache {
        let max_entries: usize = std::env::var(TOKENIZER_CACHE_SIZE_ENV)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_000);
        let max_key_bytes: usize = std::env::var(TOKENIZER_CACHE_MAX_KEY_BYTES_ENV)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);
        let cache_config = CacheConfig {
            enable_l0: true,
            l0_max_entries: max_entries,
            l0_max_key_bytes: max_key_bytes,
        };
        tracing::info!(
            max_entries,
            max_key_bytes,
            "tokenizer encode cache enabled (set by {ENABLE_TOKENIZER_CACHE_ENV})"
        );
        match tokenizer {
            TokenizerSource::HuggingFace(path) => Ok(Arc::new(CachedTokenizer::new(
                HuggingFaceTokenizer::new(path)?,
                cache_config,
            ))),
            TokenizerSource::Tiktoken(path) => Ok(Arc::new(CachedTokenizer::new(
                TiktokenTokenizer::new(path)?,
                cache_config,
            ))),
            TokenizerSource::Tekken(path) => Ok(Arc::new(CachedTokenizer::new(
                TekkenTokenizer::new(path)?,
                cache_config,
            ))),
        }
    } else {
        match tokenizer {
            TokenizerSource::HuggingFace(path) => Ok(Arc::new(HuggingFaceTokenizer::new(path)?)),
            TokenizerSource::Tiktoken(path) => Ok(Arc::new(TiktokenTokenizer::new(path)?)),
            TokenizerSource::Tekken(path) => Ok(Arc::new(TekkenTokenizer::new(path)?)),
        }
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
            max_model_len: self.model_config.max_position_embeddings(),
        })
    }
}
