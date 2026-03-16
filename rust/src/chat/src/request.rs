use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use vllm_engine_core_client::protocol::SamplingParams;

use crate::error::{Error, Result};

/// Role label for one text-only chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// One text-only chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Semantic role used by the chat template.
    pub role: ChatRole,
    /// Plain-text message content.
    pub content: String,
}

/// Chat-template-related request options.
///
/// These are the small subset of chat controls that currently affect prompt rendering in
/// `vllm-chat`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatOptions {
    /// If true, ask the chat template to append a generation prompt for the assistant.
    pub add_generation_prompt: bool,
    /// If true, expose `continue_final_message` to the chat template so the final assistant
    /// message can be left open-ended for continuation instead of starting a new assistant turn.
    pub continue_final_message: bool,

    /// Additional keyword arguments exposed to the chat template.
    pub template_kwargs: HashMap<String, Value>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            continue_final_message: false,
            template_kwargs: HashMap::new(),
        }
    }
}

/// One text-only chat request ready to be rendered into a prompt and lowered into a generate
/// request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Stable caller-supplied request ID.
    pub request_id: String,
    /// Ordered chat history to render.
    pub messages: Vec<ChatMessage>,
    /// Southbound sampling parameters forwarded to `vllm_llm`.
    pub sampling_params: SamplingParams,
    /// Chat-specific rendering options.
    pub chat_options: ChatOptions,
}

impl ChatRequest {
    /// Validate basic request invariants before rendering.
    pub fn validate(&self) -> Result<()> {
        if self.messages.is_empty() {
            return Err(Error::EmptyMessages);
        }
        if self.chat_options.add_generation_prompt && self.chat_options.continue_final_message {
            return Err(Error::ConflictingGenerationPromptMode);
        }
        Ok(())
    }
}

impl ChatRole {
    /// Return the chat-template role string used by the current text-only renderer.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}
