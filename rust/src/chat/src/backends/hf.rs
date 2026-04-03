use std::sync::Arc;

use smg_tokenizer::SpecialTokens;
use smg_tokenizer::chat_template::load_chat_template_from_file;
use thiserror_ext::AsReport as _;
use tracing::info;
use vllm_text::DynTextBackend;
use vllm_text::backends::hf::{
    HfSpecialTokens, HfTextBackend, ResolvedModelFiles, load_tokenizer_config,
};

use crate::backend::{ChatBackend, DynChatBackend};
use crate::backends::LoadedModelBackends;
use crate::error::{Error, Result};
use crate::request::ChatRequest;
use crate::template::ChatTemplate;

/// [`ChatBackend`] implementation built on Hugging Face model files.
pub struct HfChatBackend {
    chat_template: ChatTemplate,
}

impl HfChatBackend {
    /// Load the chat backend with the given model id.
    pub async fn from_model(model_id: &str) -> Result<Self> {
        let files = ResolvedModelFiles::new(model_id).await?;
        Self::from_resolved_model_files(files, model_id.to_string())
    }

    /// Load the chat backend from resolved Hugging Face model files.
    pub fn from_resolved_model_files(files: ResolvedModelFiles, model_id: String) -> Result<Self> {
        let tokenizer_config = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let special_tokens = to_chat_template_special_tokens(tokenizer_config.special_tokens);

        // Match the usual HF precedence: tokenizer_config first, then any
        // adjacent dedicated chat template file.
        let mut template = tokenizer_config.chat_template;
        if let Some(chat_template_path) = files.chat_template_path.as_deref() {
            template = load_chat_template_from_file(
                chat_template_path
                    .to_str()
                    .expect("chat template path should be valid UTF-8"),
            )
            .map_err(|error| Error::ChatTemplate(error.to_report_string()))?;
        }
        let chat_template = ChatTemplate::new(template, special_tokens)?;

        info!(
            model_id,
            "loaded chat backend with Hugging Face model files"
        );
        Ok(Self { chat_template })
    }
}

impl ChatBackend for HfChatBackend {
    fn apply_chat_template(&self, request: &ChatRequest) -> Result<String> {
        self.chat_template.apply_chat_template(request)
    }
}

/// Load the Hugging Face text and chat backends for the given model id.
pub(super) async fn load_model_backends(model_id: &str) -> Result<LoadedModelBackends> {
    let files = ResolvedModelFiles::new(model_id).await?;
    let text_backend: DynTextBackend = Arc::new(HfTextBackend::from_resolved_model_files(
        files.clone(),
        model_id.to_string(),
    )?);
    let chat_backend: DynChatBackend = Arc::new(HfChatBackend::from_resolved_model_files(
        files,
        model_id.to_string(),
    )?);

    Ok(LoadedModelBackends {
        text_backend,
        chat_backend,
    })
}

fn to_chat_template_special_tokens(tokens: HfSpecialTokens) -> Option<SpecialTokens> {
    if tokens.is_empty() {
        return None;
    }

    Some(SpecialTokens {
        bos_token: tokens.bos_token.map(Into::into),
        eos_token: tokens.eos_token.map(Into::into),
        unk_token: tokens.unk_token.map(Into::into),
        pad_token: tokens.pad_token.map(Into::into),
        ..Default::default()
    })
}
