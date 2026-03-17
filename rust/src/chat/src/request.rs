use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};

/// Role label for one text-only chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// One text-only chat content part in OpenAI-style block format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    /// One plain-text content block.
    Text { text: String },
}

impl ChatContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub(crate) fn as_text(&self) -> &str {
        match self {
            Self::Text { text } => text,
        }
    }
}

/// Text-only chat content.
///
/// This supports either a simple string or an OpenAI-style list of text blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    /// Simple text content.
    Text(String),
    /// OpenAI-style text blocks.
    Parts(Vec<ChatContentPart>),
}

impl ChatContent {
    /// Flatten the text content into one plain string without adding separators.
    // TODO: this method will be truly fallible once we add non-text content parts.
    pub fn try_flatten_to_text(&self) -> Result<String> {
        Ok(match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts.iter().map(ChatContentPart::as_text).collect(),
        })
    }
}

impl From<String> for ChatContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for ChatContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<Vec<ChatContentPart>> for ChatContent {
    fn from(value: Vec<ChatContentPart>) -> Self {
        Self::Parts(value)
    }
}

/// One text-only chat message.
///
/// Original Python API reference:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/entrypoints/chat_utils.py#L309-L333>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Semantic role used by the chat template.
    pub role: ChatRole,
    /// Plain-text message content, either as a raw string or OpenAI-style text blocks.
    pub content: ChatContent,
}

impl ChatMessage {
    /// Construct one chat message from any supported text-only content shape.
    pub fn new(role: ChatRole, content: impl Into<ChatContent>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    /// Construct one chat message with plain string content.
    pub fn text(role: ChatRole, text: impl Into<String>) -> Self {
        Self::new(role, text.into())
    }
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

/// User-facing sampling parameters accepted by `vllm-chat`.
///
/// This intentionally keeps only the subset that the current Rust chat layer
/// supports as northbound request semantics. Engine-core-specific normalized
/// fields are derived later during lowering.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L155-L291>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserSamplingParams {
    /// Controls randomness. Lower values are more deterministic; zero means
    /// greedy sampling. `None` means no explicit user override.
    pub temperature: Option<f32>,
    /// Cumulative probability threshold for nucleus sampling.
    pub top_p: Option<f32>,
    /// Maximum number of top tokens to consider. `Some(0)` means all tokens.
    pub top_k: Option<i32>,
    /// Maximum number of tokens to generate. `None` means no explicit user override.
    pub max_tokens: Option<u32>,
    /// Minimum number of tokens to generate before EOS or stop-token handling.
    pub min_tokens: Option<u32>,
    /// If true, keep the terminal stop token in the decoded output text.
    ///
    /// This currently affects token-based stop handling only; string stop
    /// sequences are still out of scope for the minimal Rust chat layer.
    pub include_stop_str_in_output: bool,
    /// Explicit stop token IDs provided by the caller. `None` means no explicit user override.
    pub stop_token_ids: Option<Vec<u32>>,
    /// If true, do not stop on the model's primary EOS token.
    pub ignore_eos: bool,
    /// If true, special tokens are skipped during incremental detokenization.
    pub skip_special_tokens: bool,
}

impl Default for UserSamplingParams {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            min_tokens: None,
            include_stop_str_in_output: false,
            stop_token_ids: None,
            ignore_eos: false,
            skip_special_tokens: true,
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
    /// User-facing sampling parameters accepted by `vllm-chat`.
    pub sampling_params: UserSamplingParams,
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
    /// Return the chat-template role string used by the current text-only chat backend.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ChatContent, ChatContentPart};

    #[test]
    fn chat_content_deserializes_from_raw_string() {
        let content: ChatContent = serde_json::from_value(json!("hello")).unwrap();
        assert_eq!(content, ChatContent::Text("hello".to_string()));
    }

    #[test]
    fn chat_content_deserializes_from_openai_text_blocks() {
        let content: ChatContent =
            serde_json::from_value(json!([{ "type": "text", "text": "hello" }])).unwrap();
        assert_eq!(
            content,
            ChatContent::Parts(vec![ChatContentPart::text("hello")])
        );
    }

    #[test]
    fn chat_content_from_string_like_values_builds_text() {
        assert_eq!(
            ChatContent::from("hello"),
            ChatContent::Text("hello".to_string())
        );
        assert_eq!(
            ChatContent::from("hello".to_string()),
            ChatContent::Text("hello".to_string())
        );
    }

    #[test]
    fn chat_content_try_flattens_text_parts_without_separators() {
        let content = ChatContent::Parts(vec![
            ChatContentPart::text("hello"),
            ChatContentPart::text(" world"),
        ]);
        assert_eq!(content.try_flatten_to_text().unwrap(), "hello world");
    }
}
