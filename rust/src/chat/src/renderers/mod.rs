use std::sync::Arc;

use crate::error::Result;
use crate::request::ChatRequest;

pub mod hf;

/// Minimal chat-prompt renderer used by `vllm-chat`.
pub trait ChatRenderer: Send + Sync {
    /// Render one chat request into the text prompt submitted to the text backend.
    fn render(&self, request: &ChatRequest) -> Result<String>;
}

/// Shared trait-object form of [`ChatRenderer`].
pub type DynChatRenderer = Arc<dyn ChatRenderer>;
