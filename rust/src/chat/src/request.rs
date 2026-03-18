use std::collections::HashMap;

use openai_protocol::common::{Function as OpenAiFunction, Tool as OpenAiTool};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::AssistantMessageExt;
use crate::error::{Error, Result};
use crate::event::{AssistantContentBlock, AssistantMessage};

/// Role label for one text-only chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    ToolResponse,
}

/// One text-only chat content part in OpenAI-style block format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    /// One plain-text content block.
    Text { text: String },
    // Image...
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
    /// OpenAI-style blocks.
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

/// One chat message.
///
/// Original Python API reference:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/entrypoints/chat_utils.py#L309-L333>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ChatMessage {
    /// System message content.
    System { content: ChatContent },
    /// User message content.
    User { content: ChatContent },
    /// Assistant history content assembled from structured assistant blocks.
    Assistant { content: Vec<AssistantContentBlock> },
    /// Tool response content associated with one prior assistant tool call.
    ToolResponse {
        content: ChatContent,
        tool_call_id: String,
    },
}

impl ChatMessage {
    /// Construct one chat message with plain string content.
    pub fn text(role: ChatRole, text: impl Into<String>) -> Self {
        let content: String = text.into();

        match role {
            ChatRole::System => Self::system(content),
            ChatRole::User => Self::user(content),
            ChatRole::Assistant => Self::assistant_text(content),
            ChatRole::ToolResponse => {
                panic!(
                    "tool response messages require a tool_call_id; \
                     use ChatMessage::tool_response() instead"
                )
            }
        }
    }

    /// Construct one chat message with system role.
    pub fn system(content: impl Into<ChatContent>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    /// Construct one chat message with user role.
    pub fn user(content: impl Into<ChatContent>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    /// Construct one chat message with assistant role and plain string content.
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: vec![AssistantContentBlock::Text { text: text.into() }],
        }
    }

    /// Construct one chat message with assistant role and structured content blocks.
    pub fn assistant_blocks(content: Vec<AssistantContentBlock>) -> Self {
        Self::Assistant { content }
    }

    /// Construct one tool-role message.
    pub fn tool_response(content: impl Into<ChatContent>, tool_call_id: impl Into<String>) -> Self {
        Self::ToolResponse {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
        }
    }

    /// Return the chat role of this message.
    pub fn role(&self) -> ChatRole {
        match self {
            Self::System { .. } => ChatRole::System,
            Self::User { .. } => ChatRole::User,
            Self::Assistant { .. } => ChatRole::Assistant,
            Self::ToolResponse { .. } => ChatRole::ToolResponse,
        }
    }

    /// Concatenate the visible text carried by this message.
    pub fn text_content(&self) -> Result<String> {
        match self {
            Self::System { content }
            | Self::User { content }
            | Self::ToolResponse { content, .. } => content.try_flatten_to_text(),
            Self::Assistant { content } => Ok(content.text()),
        }
    }

    /// Concatenate assistant reasoning text when present.
    pub fn reasoning_content(&self) -> Option<String> {
        match self {
            Self::Assistant { content } => content.reasoning(),
            Self::System { .. } | Self::User { .. } | Self::ToolResponse { .. } => None,
        }
    }
}

impl From<AssistantMessage> for ChatMessage {
    fn from(value: AssistantMessage) -> Self {
        Self::Assistant {
            content: value.content,
        }
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

/// One function-style tool made available to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
}

impl ChatTool {
    /// Used internally for template rendering and passed to `tool-parser` crate.
    pub(crate) fn to_openai_tool(&self) -> OpenAiTool {
        OpenAiTool {
            tool_type: "function".to_string(),
            function: OpenAiFunction {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.parameters.clone(),
                strict: self.strict,
            },
        }
    }

    pub(crate) fn to_template_value(&self) -> Value {
        serde_json::to_value(self.to_openai_tool()).expect("tool definition must serialize")
    }
}

/// Tool-choice semantics supported by `vllm-chat`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatToolChoice {
    Auto,
    #[default]
    None,
}

/// One chat request ready to be rendered into a prompt and lowered into a generate request.
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
    /// Function tools made available to the model for this request.
    pub tools: Vec<ChatTool>,
    /// Tool-choice behavior for this request.
    pub tool_choice: ChatToolChoice,
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

    /// Return the list of tools in the shape that can be passed to the chat template, based on the
    /// tool choice and tool list.
    pub(crate) fn template_tools(&self) -> Option<Vec<Value>> {
        self.tool_parsing_enabled()
            .then(|| self.tools.iter().map(ChatTool::to_template_value).collect())
    }

    /// Return the list of tools in the shape that can be passed to the `tool-parser` crate, based
    /// on the tool choice and tool list.
    pub(crate) fn parser_tools(&self) -> Option<Vec<OpenAiTool>> {
        self.tool_parsing_enabled()
            .then(|| self.tools.iter().map(ChatTool::to_openai_tool).collect())
    }

    /// Return true if this request should enable tool parsing based on the tool choice and tool
    /// list.
    pub(crate) fn tool_parsing_enabled(&self) -> bool {
        matches!(self.tool_choice, ChatToolChoice::Auto) && !self.tools.is_empty()
    }

    /// Return the number of tool calls in the message history.
    pub(crate) fn history_tool_call_count(&self) -> usize {
        self.messages
            .iter()
            .filter_map(|message| match message {
                ChatMessage::Assistant { content } => Some(content.tool_calls().count()),
                ChatMessage::System { .. }
                | ChatMessage::User { .. }
                | ChatMessage::ToolResponse { .. } => None,
            })
            .sum()
    }
}

impl ChatRole {
    /// Return the chat-template role string used by the current text-only chat backend.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::ToolResponse => "tool_response",
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ChatContent, ChatContentPart, ChatMessage, ChatRole};
    use crate::event::AssistantContentBlock;

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

    #[test]
    fn assistant_message_collects_visible_and_reasoning_text() {
        let message = ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::Reasoning {
                text: "inner".to_string(),
            },
            AssistantContentBlock::Text {
                text: "outer".to_string(),
            },
        ]);

        assert_eq!(message.role(), ChatRole::Assistant);
        assert_eq!(message.text_content().unwrap(), "outer");
        assert_eq!(message.reasoning_content().as_deref(), Some("inner"));
    }
}
