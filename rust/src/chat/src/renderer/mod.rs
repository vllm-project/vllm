use std::sync::Arc;

use crate::error::Result;
use crate::request::ChatRequest;

pub mod deepseek_v32;
pub mod hf;
mod selection;

pub use deepseek_v32::DeepSeekV32ChatRenderer;
pub use selection::RendererSelection;

/// Rendered chat prompt submitted to the text backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub prompt: String,
}

/// Minimal chat-prompt renderer used by `vllm-chat`.
pub trait ChatRenderer: Send + Sync {
    /// Render one chat request into the text prompt submitted to the text backend.
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt>;
}

/// Shared trait-object form of [`ChatRenderer`].
pub type DynChatRenderer = Arc<dyn ChatRenderer>;
