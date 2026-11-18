// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::Deserialize;
use validator::Validate;

use crate::routes::openai::utils::types::validate_top_p_value;

#[derive(Debug, Deserialize, Validate)]
pub(in crate::routes) struct MessagesRequest {
    pub(super) model: String,
    pub(super) messages: Vec<InputMessage>,
    pub(super) system: Option<TextContent>,
    #[validate(range(min = 1))]
    pub(super) max_tokens: Option<u32>,
    #[validate(range(min = 0.0, max = 1.0))]
    pub(super) temperature: Option<f32>,
    #[validate(custom(function = "validate_top_p_value"))]
    pub(super) top_p: Option<f32>,
    pub(super) top_k: Option<u32>,
    pub(super) stop_sequences: Option<Vec<String>>,
    pub(super) stream: Option<bool>,
    pub(super) tools: Option<Vec<ToolDefinition>>,
    pub(super) tool_choice: Option<AnthropicToolChoice>,
    pub(super) thinking: Option<ThinkingConfig>,
    pub(super) output_config: Option<OutputConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub(super) enum InputMessage {
    System { content: TextContent },
    User { content: UserContent },
    Assistant { content: AssistantContent },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum TextContent {
    Text(String),
    Blocks(Vec<TextBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(super) enum TextBlock {
    Text { text: String },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum AssistantContent {
    Text(String),
    Blocks(Vec<AssistantInputBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum AssistantInputBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    // Opaque reasoning cannot be rendered, but clients replay the full response.
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum UserContent {
    Text(String),
    Blocks(Vec<UserInputBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum UserInputBlock {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    ToolResult {
        tool_use_id: String,
        content: Option<ToolResultContent>,
        is_error: Option<bool>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(super) enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(super) enum ToolResultBlock {
    Text { text: String },
    Image { source: ImageSource },
}

#[derive(Debug, Deserialize)]
pub(super) struct ToolDefinition {
    pub(super) name: String,
    pub(super) description: Option<String>,
    pub(super) input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(super) enum AnthropicToolChoice {
    Auto {
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: String,
        disable_parallel_tool_use: Option<bool>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum ThinkingConfig {
    Enabled {
        budget_tokens: Option<i64>,
        // Like Python, always return reasoning and retain it in history.
        display: Option<ThinkingDisplay>,
    },
    Disabled,
    Adaptive {
        display: Option<ThinkingDisplay>,
    },
    BetweenTools,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum ThinkingDisplay {
    Summarized,
    Omitted,
    Updates,
}

#[derive(Debug, Deserialize)]
pub(super) struct OutputConfig {
    pub(super) effort: Option<OutputEffort>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum OutputEffort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

#[derive(serde::Serialize)]
pub(super) struct MessagesResponse {
    pub(super) id: String,
    #[serde(rename = "type")]
    pub(super) message_type: MessageType,
    pub(super) role: &'static str,
    pub(super) content: Vec<OutputContentBlock>,
    pub(super) model: String,
    pub(super) stop_reason: MessagesStopReason,
    pub(super) stop_sequence: Option<String>,
    pub(super) usage: MessagesUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) kv_transfer_params: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ec_transfer_params: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MessageType {
    Message,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MessagesStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
}

impl MessagesStopReason {
    pub(super) fn as_str(&self) -> &'static str {
        match self {
            Self::EndTurn => "end_turn",
            Self::MaxTokens => "max_tokens",
            Self::StopSequence => "stop_sequence",
            Self::ToolUse => "tool_use",
        }
    }
}

#[derive(serde::Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(super) enum OutputContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(serde::Serialize)]
pub(super) struct MessagesUsage {
    pub(super) input_tokens: usize,
    pub(super) output_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cache_read_input_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cache_creation_input_tokens: Option<usize>,
}

#[derive(serde::Serialize)]
pub(super) struct CountTokensResponse {
    pub(super) input_tokens: usize,
    pub(super) context_management: ContextManagement,
}

#[derive(serde::Serialize)]
pub(super) struct ContextManagement {
    pub(super) original_input_tokens: usize,
}

#[derive(Debug, serde::Serialize)]
pub(super) struct ErrorResponse {
    #[serde(rename = "type")]
    pub(super) kind: ErrorResponseType,
    pub(super) error: ErrorDetail,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ErrorResponseType {
    Error,
}

#[derive(Debug, serde::Serialize)]
pub(super) struct ErrorDetail {
    #[serde(rename = "type")]
    pub(super) kind: ErrorType,
    pub(super) message: String,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ErrorType {
    InvalidRequestError,
    NotFoundError,
    ConflictError,
    OverloadedError,
    ApiError,
}
