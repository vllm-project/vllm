// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Value, json};
use vllm_text::Prompt;

use crate::error::Result;
use crate::request::{ChatContent, ChatMessage, ChatRequest, ReasoningEffort};

mod deepseek;
pub mod deepseek_v32;
pub mod deepseek_v4;
pub mod deepseek_v41;
pub mod harmony;
pub mod hf;
mod inkling;
mod kimi_k3;
mod selection;
#[cfg(test)]
mod test_utils;

pub use deepseek_v4::DeepSeekV4ChatRenderer;
pub use deepseek_v32::DeepSeekV32ChatRenderer;
pub use deepseek_v41::DeepSeekV41ChatRenderer;
pub use harmony::HarmonyChatRenderer;
pub use inkling::InklingChatRenderer;
pub use kimi_k3::KimiK3ChatRenderer;
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

/// Location of one multimodal content part in the source chat request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaPartSource {
    message_index: usize,
    content_part_index: usize,
}

impl MediaPartSource {
    /// Identify one content part by its message and part indices.
    pub const fn new(message_index: usize, content_part_index: usize) -> Self {
        Self {
            message_index,
            content_part_index,
        }
    }

    /// Return the zero-based message index in the source request.
    pub const fn message_index(self) -> usize {
        self.message_index
    }

    /// Return the zero-based content-part index within the source message.
    pub const fn content_part_index(self) -> usize {
        self.content_part_index
    }
}

/// Minimal chat-prompt renderer used by `vllm-chat`.
pub trait ChatRenderer: Send + Sync {
    /// Render one chat request into the text prompt submitted to the text
    /// backend.
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt>;

    /// Render one request and return media sources in placeholder order.
    ///
    /// The default is suitable for renderers that preserve message and content
    /// order. Renderers that reorder or omit multimodal content must override
    /// this method and derive the media order while rendering.
    fn render_with_media_order(
        &self,
        request: &ChatRequest,
    ) -> Result<(RenderedPrompt, Vec<MediaPartSource>)> {
        let rendered = self.render(request)?;
        Ok((rendered, request_media_order(request)))
    }
}

/// Shared trait-object form of [`ChatRenderer`].
pub type DynChatRenderer = Arc<dyn ChatRenderer>;

fn request_media_order(request: &ChatRequest) -> Vec<MediaPartSource> {
    let mut media_order = Vec::new();
    for (message_index, message) in request.messages.iter().enumerate() {
        let content = match message {
            ChatMessage::System { content }
            | ChatMessage::Developer { content, .. }
            | ChatMessage::User { content }
            | ChatMessage::ToolResponse { content, .. } => content,
            ChatMessage::Assistant { .. } => continue,
        };
        record_content_media(&mut media_order, message_index, content);
    }
    media_order
}

fn record_content_media(
    media_order: &mut Vec<MediaPartSource>,
    message_index: usize,
    content: &ChatContent,
) {
    let ChatContent::Parts(parts) = content else {
        return;
    };
    media_order.extend(
        parts
            .iter()
            .enumerate()
            .filter(|(_, part)| part.is_multimodal())
            .map(|(content_part_index, _)| MediaPartSource::new(message_index, content_part_index)),
    );
}

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
