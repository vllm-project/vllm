mod config;
mod model_files;

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use tokenizers::Tokenizer as HfTokenizer;

use self::config::{load_generation_config, load_tokenizer_config};
use self::model_files::{ResolvedModelFiles, resolve_model_files};
use crate::backend::{ChatBackend, SamplingHints};
use crate::error::{Error, Result};
use crate::request::ChatRequest;
use crate::template::ChatTemplate;

/// [`ChatBackend`] implementation built directly on the Rust Hugging Face stack.
#[derive(Clone)]
pub struct HfChatBackend {
    inner: Arc<HfChatBackendInner>,
}

struct HfChatBackendInner {
    tokenizer: HfTokenizer,
    chat_template: ChatTemplate,
    /// Primary EOS handled by engine-core's dedicated EOS path.
    primary_eos_token_id: Option<u32>,
    /// Additional EOS ids that should flow through stop-token handling.
    extra_eos_token_ids: BTreeSet<u32>,
}

impl fmt::Debug for HfChatBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HfChatBackend").finish_non_exhaustive()
    }
}

impl HfChatBackend {
    /// Load one Hugging Face model tokenizer plus adjacent chat/EOS metadata.
    pub async fn from_model(model_id: &str) -> Result<Self> {
        let files = resolve_model_files(model_id).await?;
        Self::from_resolved_model_files(files)
    }

    fn from_resolved_model_files(files: ResolvedModelFiles) -> Result<Self> {
        let tokenizer = HfTokenizer::from_file(&files.tokenizer_path)
            .map_err(|error| Error::Tokenizer(format!("failed to load tokenizer: {error}")))?;

        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;

        let chat_template = ChatTemplate::load(
            files.tokenizer_config_path.as_deref(),
            files.chat_template_path.as_deref(),
        )?;

        let primary_eos_token_id = tokenizer_config
            .eos_token
            .as_ref()
            .and_then(|token| tokenizer.token_to_id(token.as_str()));

        let mut extra_eos_token_ids =
            load_generation_config(files.generation_config_path.as_deref())?
                .eos_token_id
                .map(|value| value.into_set())
                .unwrap_or_default();
        if let Some(primary_eos_token_id) = primary_eos_token_id {
            extra_eos_token_ids.remove(&primary_eos_token_id);
        }

        Ok(Self {
            inner: Arc::new(HfChatBackendInner {
                tokenizer,
                chat_template,
                primary_eos_token_id,
                extra_eos_token_ids,
            }),
        })
    }
}

impl ChatBackend for HfChatBackend {
    fn apply_chat_template(&self, request: &ChatRequest) -> Result<String> {
        self.inner.chat_template.apply_chat_template(request)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .tokenizer
            .encode(text, false)
            .map_err(|error| Error::Tokenizer(format!("encoding failed: {error}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| Error::Tokenizer(format!("decoding failed: {error}")))
    }

    fn sampling_hints(&self) -> Result<SamplingHints> {
        Ok(SamplingHints {
            primary_eos_token_id: self.inner.primary_eos_token_id,
            extra_eos_token_ids: self.inner.extra_eos_token_ids.clone(),
        })
    }
}
