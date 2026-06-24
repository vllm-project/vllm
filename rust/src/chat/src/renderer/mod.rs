use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Value, json};
use vllm_text::Prompt;

use crate::error::Result;
use crate::request::{ChatRequest, ReasoningEffort};

pub mod deepseek_v32;
pub mod deepseek_v4;
pub mod hf;
mod selection;

pub use deepseek_v4::DeepSeekV4ChatRenderer;
pub use deepseek_v32::DeepSeekV32ChatRenderer;
pub use selection::RendererSelection;

/// Rendered chat prompt submitted to the text backend.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedPrompt {
    /// The rendered prompt, either as text or already tokenized.
    pub prompt: Prompt,
    /// Effective chat-template kwargs visible to the renderer after applying
    /// server defaults, request overrides, and typed reasoning controls.
    pub effective_template_kwargs: HashMap<String, Value>,
}

/// Minimal chat-prompt renderer used by `vllm-chat`.
pub trait ChatRenderer: Send + Sync {
    /// Render one chat request into the text prompt submitted to the text
    /// backend.
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt>;
}

/// Shared trait-object form of [`ChatRenderer`].
pub type DynChatRenderer = Arc<dyn ChatRenderer>;

/// Extract the effective chat-template kwargs visible to the renderer from the request,
/// using the provided defaults as the base.
pub(crate) fn effective_template_kwargs(
    default_template_kwargs: &HashMap<String, Value>,
    request: &ChatRequest,
) -> HashMap<String, Value> {
    let mut kwargs = default_template_kwargs.clone();
    kwargs.extend(request.chat_options.template_kwargs.clone());

    if let Some(reasoning_effort) = request.chat_options.reasoning_effort {
        kwargs.insert(
            "reasoning_effort".to_string(),
            Value::String(reasoning_effort.as_str().to_string()),
        );
        if !request.chat_options.template_kwargs.contains_key("enable_thinking") {
            kwargs.insert(
                "enable_thinking".to_string(),
                json!(reasoning_effort != ReasoningEffort::None),
            );
        }
    }

    kwargs
}

/// Extract the effective chat-template kwargs visible to the renderer from the request.
pub(crate) fn request_template_kwargs(request: &ChatRequest) -> HashMap<String, Value> {
    effective_template_kwargs(&HashMap::new(), request)
}
