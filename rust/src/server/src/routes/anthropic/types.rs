// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

use serde::de::{IgnoredAny, SeqAccess, Visitor, value::SeqAccessDeserializer};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};

#[derive(Debug, Deserialize)]
pub(super) struct MessagesInput {
    pub(super) model: String,
    pub(super) messages: Vec<InputMessage>,
    pub(super) system: Option<TextContent>,
    pub(super) thinking: Option<ThinkingConfig>,
    pub(super) tools: Option<Vec<ClientTool>>,
    pub(super) tool_choice: Option<ToolChoice>,
    pub(super) context_management: Option<ContextManagementInput>,
}

#[derive(Debug, Deserialize)]
pub(in crate::routes) struct MessagesRequest {
    #[serde(flatten)]
    pub(super) input: MessagesInput,
    pub(super) max_tokens: u32,
    pub(super) temperature: Option<f32>,
    pub(super) top_p: Option<f32>,
    pub(super) top_k: Option<u32>,
    pub(super) stop_sequences: Option<Vec<String>>,
    pub(super) stream: Option<bool>,
    pub(super) output_config: Option<OutputConfig>,
    pub(super) cache_salt: Option<String>,
    pub(super) output_format: Option<IgnoredAny>,
    pub(super) kv_transfer_params: Option<IgnoredAny>,
    pub(super) ec_transfer_params: Option<IgnoredAny>,
}

#[derive(Debug, Deserialize)]
pub(in crate::routes) struct CountTokensRequest {
    #[serde(flatten)]
    pub(super) input: MessagesInput,
    pub(super) output_config: Option<OutputConfig>,
    pub(super) chat_template_kwargs: Option<HashMap<String, Value>>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ContextManagementInput {
    pub(super) edits: Option<Vec<ContextEdit>>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum ContextEdit {
    #[serde(rename = "clear_thinking_20251015")]
    ClearThinking { keep: Option<Value> },
    #[serde(other)]
    Unsupported,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase", deny_unknown_fields)]
pub(super) enum InputMessage {
    System { content: TextContent },
    User { content: UserContent },
    Assistant { content: AssistantContent },
}

/// Anthropic accepts either a plain string or typed blocks for message content.
/// Dispatch by JSON shape to preserve the error from an invalid block.
#[derive(Debug)]
pub(super) enum Content<T> {
    Text(String),
    Blocks(Vec<T>),
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Content<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ContentVisitor<T>(PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for ContentVisitor<T> {
            type Value = Content<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a string or a list of content blocks")
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(Content::Text(value.to_owned()))
            }

            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(Content::Text(value))
            }

            fn visit_seq<S: SeqAccess<'de>>(self, seq: S) -> Result<Self::Value, S::Error> {
                Vec::deserialize(SeqAccessDeserializer::new(seq)).map(Content::Blocks)
            }
        }

        deserializer.deserialize_any(ContentVisitor(PhantomData))
    }
}

pub(super) type TextContent = Content<TextBlock>;
pub(super) type UserContent = Content<UserInputBlock>;
pub(super) type AssistantContent = Content<AssistantInputBlock>;
pub(super) type ToolResultContent = Content<ToolResultBlock>;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub(super) enum TextBlock {
    Text {
        text: String,
        #[expect(dead_code, reason = "Prefix caching is automatic")]
        cache_control: Option<IgnoredAny>,
    },
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
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum ImageSource {
    Url {
        url: String,
    },
    Base64 {
        media_type: ImageMediaType,
        data: String,
    },
}

#[derive(Debug, Deserialize)]
pub(super) enum ImageMediaType {
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[serde(rename = "image/png")]
    Png,
    #[serde(rename = "image/gif")]
    Gif,
    #[serde(rename = "image/webp")]
    Webp,
}

impl ImageMediaType {
    pub(super) fn as_str(&self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum ToolResultBlock {
    Text { text: String },
    Image { source: ImageSource },
    ToolReference { tool_name: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum AssistantInputBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        #[expect(dead_code, reason = "Thinking signatures are ignored on replay")]
        signature: Option<String>,
    },
    // Opaque reasoning cannot be rendered, but clients replay the full response.
    RedactedThinking {
        #[expect(dead_code, reason = "Redacted thinking is ignored on replay")]
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Map<String, Value>,
    },
}

#[derive(Debug, Deserialize)]
pub(super) struct ClientTool {
    #[serde(rename = "type")]
    pub(super) kind: Option<ClientToolKind>,
    pub(super) name: String,
    pub(super) description: Option<String>,
    pub(super) input_schema: Option<Map<String, Value>>,
    pub(super) strict: Option<bool>,
    pub(super) defer_loading: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ClientToolKind {
    Custom,
    #[serde(other)]
    Unsupported,
}

#[derive(Debug, Deserialize)]
pub(super) struct ToolChoice {
    #[serde(flatten)]
    pub(super) kind: ToolChoiceKind,
    #[serde(default)]
    pub(super) disable_parallel_tool_use: bool,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum ToolChoiceKind {
    Auto,
    Any,
    None,
    Tool { name: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum ThinkingConfig {
    Enabled {
        budget_tokens: Option<i64>,
        // Like Python, always return reasoning and retain it in history.
        #[expect(dead_code, reason = "Display hints do not change reasoning output")]
        display: Option<ThinkingDisplay>,
    },
    Disabled,
    Adaptive {
        #[expect(dead_code, reason = "Display hints do not change reasoning output")]
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

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OutputConfig {
    pub(super) effort: Option<OutputEffort>,
    pub(super) format: Option<OutputFormat>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum OutputFormat {
    JsonSchema { schema: Map<String, Value> },
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
#[serde(tag = "type", rename = "message")]
pub(super) struct MessagesResponse {
    pub(super) id: String,
    pub(super) role: &'static str,
    pub(super) content: Vec<OutputContentBlock>,
    pub(super) model: String,
    pub(super) stop_reason: Option<MessagesStopReason>,
    pub(super) stop_sequence: Option<String>,
    pub(super) usage: MessagesUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) kv_transfer_params: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ec_transfer_params: Option<serde_json::Value>,
}

#[derive(Clone, Copy, serde::Serialize)]
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
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum OutputContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Map<String, Value>,
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
#[serde(tag = "type", rename = "error")]
pub(super) struct ErrorResponse {
    pub(super) error: ErrorDetail,
}

#[derive(Debug, serde::Serialize)]
pub(super) struct ErrorDetail {
    #[serde(rename = "type")]
    pub(super) kind: ErrorType,
    pub(super) message: String,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
#[expect(
    clippy::enum_variant_names,
    reason = "Variant names match the wire values"
)]
pub(super) enum ErrorType {
    InvalidRequestError,
    NotFoundError,
    OverloadedError,
    ApiError,
}
