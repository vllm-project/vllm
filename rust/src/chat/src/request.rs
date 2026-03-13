use std::collections::BTreeMap;

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
    ///
    /// This mirrors the meaning of vLLM/OpenAI-server-side `add_generation_prompt`.
    pub add_generation_prompt: bool,
    /// Reserved for vLLM parity, but not wired through the current SMG-backed renderer yet.
    ///
    /// The conflict with `add_generation_prompt` is still validated so the public API does not
    /// drift from the intended upstream semantics.
    pub continue_final_message: bool,
    /// Additional keyword arguments exposed to the chat template.
    pub template_kwargs: BTreeMap<String, Value>,
}

impl ChatOptions {
    /// Create options with the recommended defaults for normal chat generation.
    pub fn with_defaults() -> Self {
        Self {
            add_generation_prompt: true,
            ..Self::default()
        }
    }
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            continue_final_message: false,
            template_kwargs: BTreeMap::new(),
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
    /// Optional cache salt forwarded to engine-core.
    pub cache_salt: Option<String>,
    /// Optional tracing headers forwarded to engine-core.
    pub trace_headers: Option<BTreeMap<String, String>>,
    /// Optional request priority forwarded to engine-core.
    pub priority: i32,
    /// Optional target data-parallel rank forwarded to engine-core.
    pub data_parallel_rank: Option<u32>,
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
