use std::sync::Arc;

use crate::error::Result;
use crate::request::ChatRequest;

pub mod hf;

/// Stream-local reasoning-parser initialization hints derived during rendering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReasoningParserInit {
    /// Mark that reasoning has already started (e.g. `<think>` was injected in the prefill).
    ///
    /// Called when the chat template injects `<think>` in the generation prompt,
    /// so the parser should treat output as reasoning from the start without
    /// waiting for a `<think>` tag in the generated output.
    pub mark_reasoning_started: bool,
    /// Mark that the `<think>` start token was already consumed (in the prefill).
    ///
    /// Prevents the streaming parser from trying to find and strip `<think>`
    /// from the model output when the template already included it.
    pub mark_think_start_stripped: bool,
}

/// Rendered chat prompt plus parser initialization hints for the response stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPrompt {
    /// The rendered text prompt submitted to the text backend.
    pub prompt: String,
    /// Stream-local parser initialization derived from the effective chat template.
    pub reasoning_parser_init: ReasoningParserInit,
}

/// Minimal chat-prompt renderer used by `vllm-chat`.
pub trait ChatRenderer: Send + Sync {
    /// Render one chat request into the text prompt submitted to the text backend.
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt>;
}

/// Shared trait-object form of [`ChatRenderer`].
pub type DynChatRenderer = Arc<dyn ChatRenderer>;
