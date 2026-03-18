mod config;
mod model_files;

use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use fastokens::Tokenizer as FastokensTokenizer;
use thiserror_ext::AsReport;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::info;

use self::config::{load_generation_config, load_tokenizer_config};
use self::model_files::{ResolvedModelFiles, resolve_model_files};
use crate::backend::{ChatBackend, SamplingHints};
use crate::backends::hf::config::GenerationConfig;
use crate::error::{Error, Result};
use crate::request::ChatRequest;
use crate::template::ChatTemplate;

/// Set this environment variable to `1` to disable `fastokens` and fall back to the HuggingFace
/// `tokenizers` crate.
const DISABLE_FASTOKENS_ENV: &str = "VLLM_RS_DISABLE_FASTOKENS";

/// Tokenizer implementation that can be either HuggingFace `tokenizers` or `fastokens`.
enum TokenizerImpl {
    Hf(Box<HfTokenizer>),
    Fastokens(Box<FastokensTokenizer>),
}

/// Returns `true` when the user has opted out of `fastokens` via [`DISABLE_FASTOKENS_ENV`].
fn fastokens_disabled() -> bool {
    std::env::var(DISABLE_FASTOKENS_ENV).is_ok_and(|v| v == "1")
}

impl TokenizerImpl {
    fn from_file(path: &Path) -> Result<Self> {
        if fastokens_disabled() {
            info!("loading tokenizer with HuggingFace tokenizers");
            let t = HfTokenizer::from_file(path).map_err(|error| {
                Error::Tokenizer(format!("failed to load tokenizer: {}", error.as_report()))
            })?;
            Ok(Self::Hf(Box::new(t)))
        } else {
            info!("loading tokenizer with fastokens");
            let t = FastokensTokenizer::from_file(path).map_err(|error| {
                Error::Tokenizer(format!("failed to load tokenizer: {}", error.as_report()))
            })?;
            Ok(Self::Fastokens(Box::new(t)))
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::Hf(t) => {
                let encoding = t.encode(text, false).map_err(|error| {
                    Error::Tokenizer(format!("encoding failed: {}", error.as_report()))
                })?;
                Ok(encoding.get_ids().to_vec())
            }
            Self::Fastokens(t) => t.encode_with_special_tokens(text, false).map_err(|error| {
                Error::Tokenizer(format!("encoding failed: {}", error.as_report()))
            }),
        }
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self {
            Self::Hf(t) => t.decode(token_ids, skip_special_tokens).map_err(|error| {
                Error::Tokenizer(format!("decoding failed: {}", error.as_report()))
            }),
            Self::Fastokens(t) => t.decode(token_ids, skip_special_tokens).map_err(|error| {
                Error::Tokenizer(format!("decoding failed: {}", error.as_report()))
            }),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::Hf(t) => t.token_to_id(token),
            Self::Fastokens(t) => t.token_to_id(token),
        }
    }
}

/// [`ChatBackend`] implementation built on HuggingFace model files, with a configurable tokenizer
/// engine (either HuggingFace `tokenizers` or `fastokens`).
#[derive(Clone)]
pub struct HfChatBackend {
    inner: Arc<HfChatBackendInner>,
}

struct HfChatBackendInner {
    model_id: String,
    tokenizer: TokenizerImpl,
    chat_template: ChatTemplate,
    /// Primary EOS handled by engine-core's dedicated EOS path.
    primary_eos_token_id: Option<u32>,
    /// Additional EOS ids that should flow through stop-token handling.
    extra_eos_token_ids: BTreeSet<u32>,
    /// Generation-config for sampling defaults that may be inherited when the user does not
    /// explicitly override them.
    generation_config: GenerationConfig,
}

impl fmt::Debug for HfChatBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HfChatBackend").finish_non_exhaustive()
    }
}

impl HfChatBackend {
    /// Load one Hugging Face model tokenizer plus adjacent chat/EOS metadata.
    ///
    /// By default the faster `fastokens` crate is used for tokenization. Set
    /// `VLLM_RS_DISABLE_FASTOKENS=1` to fall back to the HuggingFace `tokenizers` crate.
    pub async fn from_model(model_id: &str) -> Result<Self> {
        let files = resolve_model_files(model_id).await?;
        Self::from_resolved_model_files(files, model_id.to_string())
    }

    fn from_resolved_model_files(files: ResolvedModelFiles, model_id: String) -> Result<Self> {
        let tokenizer = TokenizerImpl::from_file(&files.tokenizer_path)?;

        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;

        let chat_template = ChatTemplate::load(
            files.tokenizer_config_path.as_deref(),
            files.chat_template_path.as_deref(),
        )?;

        let primary_eos_token_id = tokenizer_config
            .eos_token
            .as_ref()
            .and_then(|token| tokenizer.token_to_id(token.as_str()));

        let generation_config = load_generation_config(files.generation_config_path.as_deref())?;
        let mut extra_eos_token_ids = generation_config
            .eos_token_id
            .clone()
            .map(|value| value.into_set())
            .unwrap_or_default();
        if let Some(primary_eos_token_id) = primary_eos_token_id {
            extra_eos_token_ids.remove(&primary_eos_token_id);
        }

        Ok(Self {
            inner: Arc::new(HfChatBackendInner {
                model_id,
                tokenizer,
                chat_template,
                primary_eos_token_id,
                extra_eos_token_ids,
                generation_config,
            }),
        })
    }
}

impl ChatBackend for HfChatBackend {
    fn apply_chat_template(&self, request: &ChatRequest) -> Result<String> {
        self.inner.chat_template.apply_chat_template(request)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.tokenizer.encode(text)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.tokenizer.decode(token_ids, skip_special_tokens)
    }

    fn model_id(&self) -> Option<&str> {
        Some(&self.inner.model_id)
    }

    fn sampling_hints(&self) -> Result<SamplingHints> {
        Ok(SamplingHints {
            primary_eos_token_id: self.inner.primary_eos_token_id,
            extra_eos_token_ids: self.inner.extra_eos_token_ids.clone(),
            default_temperature: self.inner.generation_config.temperature,
            default_top_p: self.inner.generation_config.top_p,
            default_top_k: self.inner.generation_config.top_k,
            default_min_p: self.inner.generation_config.min_p,
            default_repetition_penalty: self.inner.generation_config.repetition_penalty,
            default_max_tokens: self.inner.generation_config.max_new_tokens,
        })
    }
}
