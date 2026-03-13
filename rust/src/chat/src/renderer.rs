use std::sync::Arc;

use crate::error::Result;
use crate::request::ChatRequest;

/// Result of rendering a chat request into a prompt consumable by the LLM layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderedPrompt {
    /// A plain-text prompt that still needs tokenization.
    Text { prompt: String },
    /// A pre-tokenized prompt for future renderers that can skip text tokenization.
    Tokens {
        prompt_token_ids: Vec<u32>,
        prompt_text: Option<String>,
    },
}

/// Render a chat request into a prompt.
pub trait ChatRenderer: Send + Sync {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt>;
}

/// Shared trait-object form of [`ChatRenderer`].
pub type DynChatRenderer = Arc<dyn ChatRenderer>;
