use std::sync::Arc;

pub use vllm_text::SamplingHints;
use vllm_text::TextBackend;

use crate::error::Result;
use crate::request::ChatRequest;

/// Minimal prompt-processing backend needed by `vllm-chat`.
pub trait ChatBackend: TextBackend + Send + Sync {
    /// Apply the chat template and return the rendered text prompt.
    fn apply_chat_template(&self, request: &ChatRequest) -> Result<String>;
}

/// Shared trait-object form of [`ChatBackend`].
pub type DynChatBackend = Arc<dyn ChatBackend>;
