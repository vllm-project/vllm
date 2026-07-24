// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Engine-independent chat prompt renderers.
//!
//! The crate accepts borrowed chat-domain inputs and produces text or token-ID
//! prompt artifacts. Serving requests, sampling policy, multimodal
//! preprocessing, parser state, and engine transport remain in consumer
//! crates.

use std::collections::HashMap;
use std::sync::Arc;

use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

mod deepseek_v32;
mod deepseek_v4;
mod error;
pub mod harmony;
pub mod hf;
mod inkling;
mod request;
mod selection;
#[cfg(test)]
mod test_utils;
#[cfg(test)]
pub(crate) use test_utils::TestRenderRequest;

pub use deepseek_v4::DeepSeekV4ChatRenderer;
pub use deepseek_v32::DeepSeekV32ChatRenderer;
pub use error::{Error, Result};
pub use harmony::HarmonyChatRenderer;
pub use inkling::InklingChatRenderer;
pub use request::RenderRequest;
pub use selection::RendererSelection;
pub use vllm_chat_types::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRole,
    ChatToolChoice, GenerationPromptMode, ImageDetail, ReasoningEffort, Tool,
};

/// Engine-independent prompt content produced by a chat renderer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, EnumAsInner)]
#[serde(untagged)]
pub enum RenderedPromptContent {
    /// Untokenized prompt text.
    Text(String),
    /// Prompt token IDs produced by a format-specific renderer.
    TokenIds(Vec<u32>),
}

/// Rendered chat prompt plus effective template metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedPrompt {
    /// Engine-independent prompt content.
    pub content: RenderedPromptContent,
    /// Effective chat-template kwargs visible to the renderer after applying
    /// server defaults, request overrides, and typed reasoning controls.
    pub effective_template_kwargs: HashMap<String, Value>,
}

/// Synchronous chat-prompt renderer.
pub trait ChatRenderer: Send + Sync {
    /// Render one borrowed chat request into text or token IDs.
    fn render(&self, request: RenderRequest<'_>) -> Result<RenderedPrompt>;
}

/// Shared trait-object form of [`ChatRenderer`].
pub type DynChatRenderer = Arc<dyn ChatRenderer>;

/// Extract the effective chat-template kwargs visible to the renderer from the request,
/// using the provided defaults as the base.
pub(crate) fn effective_template_kwargs(
    default_template_kwargs: &HashMap<String, Value>,
    request: &RenderRequest<'_>,
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
pub(crate) fn request_template_kwargs(request: &RenderRequest<'_>) -> HashMap<String, Value> {
    effective_template_kwargs(&HashMap::new(), request)
}
