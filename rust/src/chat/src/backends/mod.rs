use std::collections::HashMap;

use serde_json::Value;
use vllm_text::DynTextBackend;

use crate::error::Result;
use crate::{ChatTemplateContentFormatOption, DynChatBackend};

pub mod hf;

/// Frontend-side chat backend loading options.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LoadModelBackendsOptions {
    /// How to serialize `message.content` when rendering the chat template.
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    /// Optional server-default chat template override, provided either as an inline template or
    /// as a path to a template file.
    pub chat_template: Option<String>,
    /// Optional server-default keyword arguments merged into every chat-template render before
    /// request-level `chat_template_kwargs`.
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
