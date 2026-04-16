use std::sync::Arc;

use thiserror_ext::AsReport as _;
use tracing::{info, warn};
use vllm_text::DynTextBackend;
use vllm_text::backends::hf::{
    HfTextBackend, HfTokenizerConfig, ResolvedModelFiles, load_tokenizer_config,
};

use crate::backend::{ChatBackend, DynChatBackend};
use crate::backends::{LoadModelBackendsOptions, LoadedModelBackends};
use crate::error::{Error, Result};
use crate::renderers::DynChatRenderer;
use crate::renderers::hf::{HfChatRenderer, load_chat_template, resolve_chat_template};

/// [`ChatBackend`] implementation built on Hugging Face model files.
pub struct HfChatBackend {
    chat_renderer: DynChatRenderer,
}

impl HfChatBackend {
    /// Load the chat backend with the given model id.
    pub async fn from_model(model_id: &str) -> Result<Self> {
        let files = ResolvedModelFiles::new(model_id).await?;
        Self::from_resolved_model_files(files, model_id.to_string(), Default::default())
    }

    /// Load the chat backend from resolved Hugging Face model files.
    pub fn from_resolved_model_files(
        files: ResolvedModelFiles,
        model_id: String,
        options: LoadModelBackendsOptions,
    ) -> Result<Self> {
        let HfTokenizerConfig {
            special_tokens,
            chat_template,
            ..
        } = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let mut template = chat_template;
        let special_tokens = (!special_tokens.is_empty()).then_some(special_tokens);

        if let Some(configured_template) = options.chat_template.as_deref() {
            template = Some(
                resolve_chat_template(configured_template)
                    .map_err(|error| Error::ChatTemplate(error.to_report_string()))?,
            );
            info!("using configured chat template override");
        } else if let Some(chat_template_path) = files.chat_template_path.as_deref() {
            // If independent chat template file(s) exist and contain non-empty content, they take
            // priority over template entries in the tokenizer config
            let file_template = load_chat_template(chat_template_path)
                .map_err(|error| Error::ChatTemplate(error.to_report_string()))?;

            if file_template.as_ref().is_some_and(|t| !t.trim().is_empty()) {
                info!(
                    path = %chat_template_path.display(),
                    "loaded dedicated chat template file, overriding tokenizer_config chat_template"
                );
                template = file_template;
            } else {
                warn!(
                    path = %chat_template_path.display(),
                    "ignoring empty dedicated chat template file and falling back to tokenizer_config chat_template"
                );
            }
        }
        let chat_renderer: DynChatRenderer = Arc::new(HfChatRenderer::new(
            template,
            options.default_chat_template_kwargs,
            options.chat_template_content_format,
            special_tokens,
        )?);

        info!(
            model_id,
            "loaded chat backend with Hugging Face model files"
        );
        Ok(Self { chat_renderer })
    }
}

impl ChatBackend for HfChatBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        self.chat_renderer.clone()
    }
}

/// Load the Hugging Face text and chat backends for the given model id.
pub(super) async fn load_model_backends(
    model_id: &str,
    options: LoadModelBackendsOptions,
) -> Result<LoadedModelBackends> {
    let files = ResolvedModelFiles::new(model_id).await?;
    let text_backend: DynTextBackend = Arc::new(HfTextBackend::from_resolved_model_files(
        files.clone(),
        model_id.to_string(),
    )?);
    let chat_backend: DynChatBackend = Arc::new(HfChatBackend::from_resolved_model_files(
        files,
        model_id.to_string(),
        options,
    )?);

    Ok(LoadedModelBackends {
        text_backend,
        chat_backend,
    })
}
