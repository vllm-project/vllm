use std::collections::HashMap;

use openai_protocol::common::{Function as OpenAiFunction, Tool as OpenAiTool};
use serde::{Deserialize, Serialize};
use serde_json::Value;
pub use vllm_text::SamplingParams;
use vllm_text::TextDecodeOptions;

use crate::AssistantMessageExt;
use crate::error::{Error, Result};
use crate::event::{AssistantContentBlock, AssistantMessage};

/// Role label for one text-only chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    Developer,
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

    /// Return whether flattening this chat content would produce an empty string.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Parts(parts) => parts.iter().all(|part| part.as_text().is_empty()),
        }
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
    /// Developer message content plus optional message-local tools.
    Developer {
        content: ChatContent,
        tools: Option<Vec<ChatTool>>,
    },
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
            ChatRole::Developer => Self::developer(content, None),
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

    /// Construct one chat message with developer role.
    pub fn developer(content: impl Into<ChatContent>, tools: Option<Vec<ChatTool>>) -> Self {
        Self::Developer {
            content: content.into(),
            tools,
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
            Self::Developer { .. } => ChatRole::Developer,
            Self::User { .. } => ChatRole::User,
            Self::Assistant { .. } => ChatRole::Assistant,
            Self::ToolResponse { .. } => ChatRole::ToolResponse,
        }
    }

    /// Concatenate the visible text carried by this message.
    pub fn text_content(&self) -> Result<String> {
        match self {
            Self::System { content }
            | Self::Developer { content, .. }
            | Self::User { content }
            | Self::ToolResponse { content, .. } => content.try_flatten_to_text(),
            Self::Assistant { content } => Ok(content.text()),
        }
    }

    /// Concatenate assistant reasoning text when present.
    pub fn reasoning_content(&self) -> Option<String> {
        match self {
            Self::Assistant { content } => content.reasoning(),
            Self::System { .. }
            | Self::Developer { .. }
            | Self::User { .. }
            | Self::ToolResponse { .. } => None,
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

/// Controls how prompt rendering should end after the existing chat history.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationPromptMode {
    /// Append a generation prompt for a new assistant turn.
    ///
    /// Equivalent to `add_generation_prompt = true` and `continue_final_message = false`.
    #[default]
    StartNewAssistant,
    /// Leave the final assistant message open so generation continues it.
    ///
    /// Equivalent to `add_generation_prompt = false` and `continue_final_message = true`.
    ContinueFinalAssistant,
    /// Render the existing chat history without adding any trailing generation prompt.
    ///
    /// Equivalent to `add_generation_prompt = false` and `continue_final_message = false`.
    NoGenerationPrompt,
}

/// Chat-template-related request options.
///
/// These are the small subset of chat controls that currently affect prompt rendering in
/// `vllm-chat`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatOptions {
    /// Controls whether rendering starts a new assistant turn, continues the final assistant
    /// message, or emits no trailing generation prompt at all.
    pub generation_prompt_mode: GenerationPromptMode,

    /// Per-request Jinja chat template override. When set, this template is used instead of the
    /// model's default chat template.
    pub chat_template: Option<String>,

    /// Additional keyword arguments exposed to the chat template.
    pub template_kwargs: HashMap<String, Value>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            generation_prompt_mode: GenerationPromptMode::StartNewAssistant,
            chat_template: None,
            template_kwargs: HashMap::new(),
        }
    }
}

impl ChatOptions {
    /// Whether to add a generation prompt for a new assistant turn after the existing chat history.
    pub fn add_generation_prompt(&self) -> bool {
        matches!(
            self.generation_prompt_mode,
            GenerationPromptMode::StartNewAssistant
        )
    }

    /// Whether to leave the final assistant message open so generation continues it.
    pub fn continue_final_message(&self) -> bool {
        matches!(
            self.generation_prompt_mode,
            GenerationPromptMode::ContinueFinalAssistant
        )
    }
}

/// One function-style tool made available to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    pub sampling_params: SamplingParams,
    /// Chat-specific rendering options.
    pub chat_options: ChatOptions,
    /// Function tools made available to the model for this request.
    pub tools: Vec<ChatTool>,
    /// Tool-choice behavior for this request.
    pub tool_choice: ChatToolChoice,
    /// Text decode options for incremental detokenization.
    pub decode_options: TextDecodeOptions,
    /// Whether to emit intermediate northbound content deltas before the terminal result.
    ///
    /// If `false`, callers only observe the terminal accumulated assistant output. If `true`,
    /// callers may receive zero or more incremental content events before the final terminal one.
    pub intermediate: bool,
    /// Request scheduling priority (lower means earlier handling; default 0).
    pub priority: i32,
    /// Documents for RAG (retrieval-augmented generation), passed to the chat template.
    pub documents: Option<Vec<Value>>,
    /// Salt for prefix cache isolation in multi-user environments.
    pub cache_salt: Option<String>,
    /// Whether to add special tokens (e.g. BOS) during prompt tokenization.
    pub add_special_tokens: bool,
    /// Override data parallel rank.
    #[serde(default)]
    pub data_parallel_rank: Option<u32>,
}

impl ChatRequest {
    /// Return one minimal valid request fixture for tests.
    pub fn for_test() -> Self {
        Self {
            request_id: "test-request".to_string(),
            messages: vec![ChatMessage::text(ChatRole::User, "test")],
            sampling_params: SamplingParams::default(),
            chat_options: ChatOptions::default(),
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            decode_options: TextDecodeOptions::default(),
            intermediate: true,
            priority: 0,
            documents: None,
            cache_salt: None,
            add_special_tokens: false,
            data_parallel_rank: None,
        }
    }

    /// Validate basic request invariants before rendering.
    pub fn validate(&self) -> Result<()> {
        if self.messages.is_empty() {
            return Err(Error::EmptyMessages);
        }
        match (
            self.chat_options.generation_prompt_mode,
            self.messages.last().map(ChatMessage::role),
        ) {
            (GenerationPromptMode::ContinueFinalAssistant, Some(ChatRole::Assistant)) => {}
            (GenerationPromptMode::ContinueFinalAssistant, _) => {
                return Err(Error::ContinueFinalAssistantWithoutFinalAssistant);
            }
            (GenerationPromptMode::NoGenerationPrompt, _)
            | (GenerationPromptMode::StartNewAssistant, _) => {}
        }
        Ok(())
    }

    /// Return true if this request should enable tool parsing based on the tool choice and tool
    /// list.
    pub(crate) fn tool_parsing_enabled(&self) -> bool {
        matches!(self.tool_choice, ChatToolChoice::Auto) && !self.tools.is_empty()
    }

    /// Return the request-level thinking toggle when explicitly requested.
    ///
    /// We currently accept the two request kwargs `thinking` and `enable_thinking`. Both must be
    /// booleans when present. If both are present, they must have the same value. If neither key
    /// is provided, return `None`.
    pub(crate) fn enable_thinking(&self) -> Result<Option<bool>> {
        let thinking = self.parse_template_bool("thinking")?;
        let enable_thinking = self.parse_template_bool("enable_thinking")?;

        match (thinking, enable_thinking) {
            (None, None) => Ok(None),
            (Some(thinking), Some(enable_thinking)) if thinking != enable_thinking => {
                Err(Error::ChatTemplate(
                    "template kwargs `thinking` and `enable_thinking` must match when both are set"
                        .to_string(),
                ))
            }
            (Some(thinking), _) => Ok(Some(thinking)),
            (None, Some(enable_thinking)) => Ok(Some(enable_thinking)),
        }
    }

    fn parse_template_bool(&self, key: &str) -> Result<Option<bool>> {
        match self.chat_options.template_kwargs.get(key) {
            None => Ok(None),
            Some(Value::Bool(value)) => Ok(Some(*value)),
            Some(other) => Err(Error::ChatTemplate(format!(
                "template kwarg `{key}` must be a boolean, got {other}"
            ))),
        }
    }
}

impl ChatRole {
    /// Return the chat-template role string used by the current text-only chat backend.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::ToolResponse => "tool_response",
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, to_value};

    use super::{ChatContent, ChatContentPart, ChatMessage, ChatRequest, ChatRole, ChatTool};
    use crate::Error;
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

    #[test]
    fn developer_message_round_trips_through_serde() {
        let message = ChatMessage::developer(
            "hello",
            Some(vec![ChatTool {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }),
                strict: Some(true),
            }]),
        );

        let value = to_value(&message).unwrap();
        let decoded: ChatMessage = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn enable_thinking_is_none_when_no_kwargs_are_present() {
        let request = ChatRequest::for_test();
        assert_eq!(request.enable_thinking().unwrap(), None);
    }

    #[test]
    fn enable_thinking_accepts_matching_duplicate_kwargs() {
        let mut request = ChatRequest::for_test();
        request
            .chat_options
            .template_kwargs
            .insert("thinking".to_string(), json!(true));
        request
            .chat_options
            .template_kwargs
            .insert("enable_thinking".to_string(), json!(true));

        assert_eq!(request.enable_thinking().unwrap(), Some(true));
    }

    #[test]
    fn enable_thinking_rejects_non_boolean_kwargs() {
        let mut request = ChatRequest::for_test();
        request
            .chat_options
            .template_kwargs
            .insert("thinking".to_string(), json!("yes"));

        assert!(matches!(
            request.enable_thinking(),
            Err(Error::ChatTemplate(message))
                if message.contains("`thinking` must be a boolean")
        ));
    }

    #[test]
    fn enable_thinking_rejects_conflicting_duplicate_kwargs() {
        let mut request = ChatRequest::for_test();
        request
            .chat_options
            .template_kwargs
            .insert("thinking".to_string(), json!(false));
        request
            .chat_options
            .template_kwargs
            .insert("enable_thinking".to_string(), json!(true));

        assert!(matches!(
            request.enable_thinking(),
            Err(Error::ChatTemplate(message))
                if message.contains("`thinking` and `enable_thinking` must match")
        ));
    }
}
