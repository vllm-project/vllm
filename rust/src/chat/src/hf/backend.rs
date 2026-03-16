use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use smg_tokenizer::chat_template::{
    ChatTemplateParams, ChatTemplateState, load_chat_template_from_config,
    load_chat_template_from_file,
};
use thiserror_ext::AsReport as _;
use tokenizers::Tokenizer;

use super::config::{load_generation_config, load_tokenizer_config};
use super::model_files::{ResolvedModelFiles, resolve_model_files};
use crate::backend::{ChatBackend, SamplingHints};
use crate::error::{Error, Result};
use crate::request::ChatRequest;
use crate::template::{merged_template_kwargs, template_messages_to_json};

/// [`ChatBackend`] implementation built directly on the Rust Hugging Face stack.
#[derive(Clone)]
pub struct HfChatBackend {
    inner: Arc<HfChatBackendInner>,
}

struct HfChatBackendInner {
    tokenizer: Tokenizer,
    chat_template: ChatTemplateState,
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
        let tokenizer = Tokenizer::from_file(&files.tokenizer_path)
            .map_err(|error| Error::Tokenizer(format!("failed to load tokenizer: {error}")))?;

        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;

        let chat_template = load_chat_template_state(
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

    fn apply_chat_template_inner(
        &self,
        request: &ChatRequest,
        template_kwargs: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> Result<String> {
        let messages = template_messages_to_json(
            &request.messages,
            self.inner.chat_template.content_format(),
        )?;
        let merged_template_kwargs = merged_template_kwargs(request, template_kwargs);

        self.inner
            .chat_template
            .apply(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: request.chat_options.add_generation_prompt,
                    tools: None,
                    documents: None,
                    template_kwargs: Some(&merged_template_kwargs),
                },
            )
            .map_err(|error| Error::Tokenizer(error.to_report_string()))
    }
}

impl ChatBackend for HfChatBackend {
    fn apply_chat_template(&self, request: &ChatRequest) -> Result<String> {
        match self.apply_chat_template_inner(request, Some(&request.chat_options.template_kwargs)) {
            Ok(prompt) => Ok(prompt),
            Err(Error::Tokenizer(message))
                if message.contains("tokenizer.chat_template is not set") =>
            {
                Err(Error::MissingChatTemplate)
            }
            Err(error) => Err(error),
        }
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

fn load_chat_template_state(
    tokenizer_config_path: Option<&std::path::Path>,
    chat_template_path: Option<&std::path::Path>,
) -> Result<ChatTemplateState> {
    // Match the usual HF/SMG precedence: tokenizer_config first, then any
    // adjacent dedicated chat template file.
    let mut chat_template = tokenizer_config_path
        .and_then(|path| path.to_str())
        .map(load_chat_template_from_config)
        .transpose()
        .map_err(|error| Error::Tokenizer(error.to_report_string()))?
        .flatten();

    if let Some(chat_template_path) = chat_template_path {
        chat_template = load_chat_template_from_file(
            chat_template_path
                .to_str()
                .expect("chat template path should be valid UTF-8"),
        )
        .map_err(|error| Error::Tokenizer(error.to_report_string()))?;
    }

    ChatTemplateState::new(chat_template)
        .map_err(|error| Error::Tokenizer(error.to_report_string()))
}
