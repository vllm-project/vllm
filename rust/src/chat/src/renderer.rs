use std::sync::Arc;

use crate::error::Result;
use crate::request::ChatRequest;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderedPrompt {
    Text {
        prompt: String,
    },
    Tokens {
        prompt_token_ids: Vec<u32>,
        prompt_text: Option<String>,
    },
}

pub trait ChatRenderer: Send + Sync {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt>;
}

pub type DynChatRenderer = Arc<dyn ChatRenderer>;
