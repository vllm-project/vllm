use std::sync::Arc;

pub use vllm_text::SamplingHints;
use vllm_text::TextBackend;

use crate::error::Result;
use crate::request::ChatRequest;

/// Minimal prompt-processing backend needed by `vllm-chat`.
pub trait ChatBackend: Send + Sync {
    /// Apply the chat template and return the rendered text prompt.
    fn apply_chat_template(&self, request: &ChatRequest) -> Result<String>;
}

/// Shared trait-object form of [`ChatBackend`].
pub type DynChatBackend = Arc<dyn ChatBackend>;

/// Convenience trait for backends that can serve both raw text generation and chat templating.
///
/// This is mainly useful in tests and small examples, where one mock/backend often implements
/// both sides and callers want `ChatLlm` to wire the shared object into `TextLlm` automatically.
pub trait ChatTextBackend: ChatBackend + TextBackend {}

impl<T> ChatTextBackend for T where T: ChatBackend + TextBackend + ?Sized {}

/// Shared trait-object form of [`ChatTextBackend`].
pub type DynChatTextBackend = Arc<dyn ChatTextBackend>;
