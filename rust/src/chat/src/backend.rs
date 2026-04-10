use std::sync::Arc;

pub use vllm_text::SamplingHints;
use vllm_text::TextBackend;

use crate::renderers::DynChatRenderer;

/// Minimal prompt-processing backend needed by `vllm-chat`.
pub trait ChatBackend: Send + Sync {
    /// Return the renderer used for chat-prompt construction.
    fn chat_renderer(&self) -> DynChatRenderer;
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
