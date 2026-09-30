// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use vllm_text::Prompt;

use crate::error::Result;
use crate::request::ChatRequest;

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

/// Location of one multimodal content part in the source chat request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaPartSource {
    /// Zero-based message index in the source request.
    pub message_index: usize,
    /// Zero-based content-part index within the source message.
    pub content_part_index: usize,
}

/// Rendered chat prompt submitted to the text backend.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedPrompt {
    /// The rendered prompt, either as text or already tokenized.
    pub prompt: Prompt,
    /// Media sources in rendered placeholder order.
    /// `None` uses the message and content-part order in the request (e.g. Jinja).
    /// `Some` uses exactly the listed sources; an empty list omits all media.
    pub media_order: Option<Vec<MediaPartSource>>,
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
