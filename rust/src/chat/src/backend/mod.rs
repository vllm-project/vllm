use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use vllm_text::{DynTextBackend, TextBackend};

use crate::error::Result;
use crate::multimodal::MultimodalModelInfo;
use crate::output::DynChatOutputProcessor;
use crate::renderer::DynChatRenderer;
use crate::request::ChatRequest;
use crate::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection};

pub mod hf;

/// Options for creating a new chat output processor.
pub struct NewChatOutputProcessorOptions<'a> {
    pub tool_call_parser: &'a ParserSelection,
    pub reasoning_parser: &'a ParserSelection,
}

/// Minimal prompt-processing backend needed by `vllm-chat`.
pub trait ChatBackend: Send + Sync {
    /// Return the renderer used for chat-prompt construction.
    fn chat_renderer(&self) -> DynChatRenderer;

    /// Return model files/config needed for request-scoped multimodal
    /// preprocessing, if supported.
    fn multimodal_model_info(&self) -> Option<&MultimodalModelInfo> {
        None
    }

    /// Create a request-scoped output processor after request-level adjustments
    /// are applied.
    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions<'_>,
    ) -> Result<DynChatOutputProcessor>;
}

/// Shared trait-object form of [`ChatBackend`].
pub type DynChatBackend = Arc<dyn ChatBackend>;

/// Convenience trait for backends that can serve both raw text generation and
/// chat templating.
///
/// This is mainly useful in tests and small examples, where one mock/backend
/// often implements both sides and callers want `ChatLlm` to wire the shared
/// object into `TextLlm` automatically.
pub trait ChatTextBackend: ChatBackend + TextBackend {}

impl<T> ChatTextBackend for T where T: ChatBackend + TextBackend + ?Sized {}

/// Shared trait-object form of [`ChatTextBackend`].
pub type DynChatTextBackend = Arc<dyn ChatTextBackend>;

/// Frontend-side chat backend loading options.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LoadModelBackendsOptions {
    /// Which chat renderer implementation to use.
    pub renderer: RendererSelection,
    /// How to serialize `message.content` when rendering the chat template.
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    /// Optional server-default chat template override, provided either as an
    /// inline template or as a path to a template file.
    pub chat_template: Option<String>,
    /// Optional server-default keyword arguments merged into every
    /// chat-template render before request-level `chat_template_kwargs`.
    pub default_chat_template_kwargs: HashMap<String, Value>,
}

/// Shared backends loaded from a model id.
pub struct LoadedModelBackends {
    pub text_backend: DynTextBackend,
    pub chat_backend: DynChatBackend,
}

/// Load text and chat backends for the given model id.
pub async fn load_model_backends(
    model_id: &str,
    options: LoadModelBackendsOptions,
) -> Result<LoadedModelBackends> {
    // Currently, we only have HuggingFace backends.
    hf::load_model_backends(model_id, options).await
}
