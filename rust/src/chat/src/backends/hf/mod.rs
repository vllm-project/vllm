use std::fmt;
use std::sync::Arc;

use tracing::info;
use vllm_text::backends::hf::HfTextBackend;
use vllm_text::{SamplingHints, TextBackend};

use crate::backend::ChatBackend;
use crate::error::Result;
use crate::request::ChatRequest;
use crate::template::ChatTemplate;

/// Chat-facing backend built from the text backend plus chat-template rendering.
#[derive(Clone)]
pub struct HfChatBackend {
    inner: Arc<HfChatBackendInner>,
}

struct HfChatBackendInner {
    text_backend: HfTextBackend,
    chat_template: ChatTemplate,
}

impl fmt::Debug for HfChatBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HfChatBackend").finish_non_exhaustive()
    }
}

impl HfChatBackend {
    /// Load one Hugging Face model plus its chat template and wrap the text backend.
    pub async fn from_model(model_id: &str) -> Result<Self> {
        let text_backend = HfTextBackend::from_model(model_id).await?;
        Self::from_text_backend(text_backend)
    }

    /// Build the chat wrapper around an already loaded text backend.
    ///
    /// The text backend stays responsible for tokenizer/model loading; this wrapper only adds
    /// chat-template rendering and chat-specific request semantics.
    pub fn from_text_backend(text_backend: HfTextBackend) -> Result<Self> {
        let files = text_backend.resolved_model_files();
        let chat_template = ChatTemplate::load(
            files.tokenizer_config_path.as_deref(),
            files.chat_template_path.as_deref(),
        )?;

        info!(
            model_id = text_backend.model_id(),
            "loaded chat backend from text backend"
        );

        Ok(Self {
            inner: Arc::new(HfChatBackendInner {
                text_backend,
                chat_template,
            }),
        })
    }
}

/// Delegate text encoding/decoding to the text backend.
impl TextBackend for HfChatBackend {
    fn encode(&self, text: &str) -> vllm_text::Result<Vec<u32>> {
        self.inner.text_backend.encode(text)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> vllm_text::Result<String> {
        self.inner
            .text_backend
            .decode(token_ids, skip_special_tokens)
    }

    fn model_id(&self) -> Option<&str> {
        self.inner.text_backend.model_id()
    }

    fn sampling_hints(&self) -> vllm_text::Result<SamplingHints> {
        self.inner.text_backend.sampling_hints()
    }
}

impl ChatBackend for HfChatBackend {
    fn apply_chat_template(&self, request: &ChatRequest) -> Result<String> {
        self.inner.chat_template.apply_chat_template(request)
    }
}
