use std::sync::Arc;

use crate::error::Result;
use crate::request::ChatRequest;

/// Minimal prompt-processing backend needed by `vllm-chat`.
pub trait ChatBackend: Send + Sync {
    /// Apply the chat template and return the rendered text prompt.
    fn apply_chat_template(&self, request: &ChatRequest) -> Result<String>;

    /// Encode one prompt string into token IDs.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;

    /// Decode one cumulative token sequence into text.
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;
}

/// Shared trait-object form of [`ChatBackend`].
pub type DynChatBackend = Arc<dyn ChatBackend>;
