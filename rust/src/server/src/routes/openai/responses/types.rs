// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! OpenAI Responses API request/response wire types for the Rust frontend.
//!
//! Request types mirror the Python vLLM `ResponsesRequest` class in
//! `vllm/entrypoints/openai/responses/protocol.py`; output item and event
//! types mirror the `openai.types.responses` SDK shapes emitted by the Python
//! frontend.

use std::collections::HashMap;
use std::fmt;

use crate::routes::openai::utils::metrics::PerRequestMetrics;
use crate::routes::openai::utils::types::{
    Function, FunctionChoice, InputAudio, ReasoningEffort, StringOrArray, ToolChoiceValue,
    ToolReference, default_true, deserialize_request_top_k,
};
use llm_multimodal::ImageDetail;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

/// Responses API `input` field: either a plain string (single user message)
/// or a list of input/output items.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponsesInput {
    /// Simple text input; rendered as one user message.
    Text(String),
    /// Ordered list of input/output items (messages, function calls,
    /// function call outputs, reasoning items, ...).
    Items(Vec<ResponseInputItem>),
}

impl<'de> Deserialize<'de> for ResponsesInput {
    /// Deserialize by JSON type rather than `#[serde(untagged)]`, which would
    /// replace a precise per-item error with a generic one.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct InputVisitor;

        impl<'de> Visitor<'de> for InputVisitor {
            type Value = ResponsesInput;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a string or a list of input items")
            }

            fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
                Ok(ResponsesInput::Text(text.to_owned()))
            }

            fn visit_string<E: de::Error>(self, text: String) -> Result<Self::Value, E> {
                Ok(ResponsesInput::Text(text))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut items = Vec::with_capacity(seq.size_hint().unwrap_or_default());
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(ResponsesInput::Items(items))
            }
        }

        deserializer.deserialize_any(InputVisitor)
    }
}

/// Responses API reasoning configuration.
///
/// Mirrors the `Reasoning` shared type from the OpenAI SDK. This frontend
/// supports `effort`. `summary` is accepted for compatibility but does not
/// cause reasoning summaries to be generated.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponsesReasoning {
    #[serde(default)]
    pub effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub summary: Option<ReasoningSummary>,
}

/// Responses API `reasoning.summary` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

/// Responses API `text.format` variants.
///
/// Mirrors the `ResponseTextConfig.format` union from the OpenAI SDK. Note
/// the `json_schema` variant is flat here (`type` next to `name`/`schema`),
/// unlike the chat-completions shape where the schema is nested under a
/// `json_schema` key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTextFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        #[serde(default)]
        description: Option<String>,
        schema: Value,
        #[serde(default)]
        strict: Option<bool>,
    },
}

/// Responses API `text` configuration.
///
/// This frontend supports `format`. `verbosity` is accepted for compatibility
/// but does not control generation separately from the model prompt.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponseTextConfig {
    #[serde(default)]
    pub format: Option<ResponseTextFormat>,
    #[serde(default)]
    pub verbosity: Option<TextVerbosity>,
}

/// Responses API `text.verbosity` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextVerbosity {
    Low,
    Medium,
    High,
}

/// Responses API `truncation` strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseTruncation {
    Auto,
    #[default]
    Disabled,
}

/// Responses API `service_tier` values. Python echoes the requested tier;
/// it does not affect scheduling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    #[default]
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
}

/// One Responses API tool.
///
/// The function tool is the chat-completions [`Function`] flattened next to
/// `type`. Built-in tool types (web search, code interpreter, MCP, ...) are
/// rejected during deserialization because this frontend has no built-in
/// tool executor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTool {
    Function(Function),
    /// Function tools grouped under a namespace. The model sees each one as
    /// `<namespace>__<name>`; see `tools.rs`.
    Namespace(NamespaceTool),
}

/// Responses API `namespace` tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamespaceTool {
    /// The namespace name used in tool calls.
    pub name: String,
    /// A description of the namespace. Only echoed: like Python, the model
    /// sees the flattened function tools alone.
    pub description: String,
    pub tools: Vec<NamespacedTool>,
}

/// One tool inside a [`NamespaceTool`]. Custom tools are rejected during
/// deserialization, as they are at the top level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NamespacedTool {
    Function(Function),
}

/// Responses API `tool_choice` field.
///
/// Mirrors the chat-completions [`ToolChoice`](crate::routes::openai::utils::types::ToolChoice),
/// except that object choices are flat: `{"type": "function", "name": ...}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseToolChoice {
    /// `none` / `auto` / `required`.
    Value(ToolChoiceValue),
    Object(ResponseToolChoiceObject),
}

/// Object form of [`ResponseToolChoice`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseToolChoiceObject {
    Function(FunctionChoice),
    /// TODO: support `allowed_tools` tool choices.
    AllowedTools {
        mode: String,
        tools: Vec<ToolReference>,
    },
}

/// Responses API request body accepted by the Rust frontend.
///
/// Mirrors the Python vLLM `ResponsesRequest` class. Unknown tool,
/// tool-choice, and input-item types fail deserialization; store-dependent
/// features are validated in `validate.rs`/`convert.rs`.
///
/// `background`, `store=true` retention, and `previous_response_id` require
/// server-side state, which this stateless frontend does not provide; see
/// `validate.rs`. `max_tool_calls` limits built-in tool calls only, so with
/// function tools alone it is accepted and echoed.
/// `reasoning.encrypted_content`, `reasoning.summary`, and `text.verbosity`
/// are accepted for Codex/Python compatibility, without generating encrypted
/// state, summaries, or applying a verbosity policy. Other `include` values
/// that this frontend cannot produce are rejected.
#[derive(Debug, Clone, Deserialize, Validate)]
pub struct ResponsesRequest {
    /// The model ID served by this frontend. Optional; defaults to the
    /// served model when omitted.
    #[serde(default)]
    pub model: Option<String>,
    /// Request input: a string or an ordered item list.
    pub input: ResponsesInput,
    /// System-level instructions prepended to the conversation.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Function tools available to the model.
    #[serde(default)]
    pub tools: Vec<ResponsesTool>,
    /// Tool selection behavior.
    #[serde(default)]
    pub tool_choice: Option<ResponseToolChoice>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub max_tool_calls: Option<u32>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// Harmony-only stateless replay history. The Rust frontend does not
    /// implement Harmony rendering, so this is retained solely to reject the
    /// field instead of silently dropping model-visible history.
    #[serde(default)]
    pub previous_input_messages: Option<Vec<Value>>,
    #[serde(default)]
    pub prompt: Option<Value>,
    #[serde(default)]
    pub reasoning: Option<ResponsesReasoning>,
    /// Whether to include reasoning content in the response. When false,
    /// reasoning tokens are still generated but excluded from the final
    /// output items.
    #[serde(default = "default_true")]
    pub include_reasoning: bool,
    #[serde(default)]
    pub service_tier: Option<ServiceTier>,
    #[serde(default)]
    pub background: Option<bool>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Whether to apply the engine's configured watermark to generated text.
    /// `None` requests it like `true`, matching Chat Completions.
    #[serde(default)]
    pub watermarking: Option<bool>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_request_top_k")]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub top_logprobs: Option<i32>,
    #[serde(default)]
    pub text: Option<ResponseTextConfig>,
    #[serde(default)]
    pub truncation: Option<ResponseTruncation>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub include: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stop: Option<StringOrArray>,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,
    #[serde(default = "default_true")]
    pub spaces_between_special_tokens: bool,
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    #[serde(default)]
    pub min_tokens: Option<u32>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(default)]
    pub stop_token_ids: Option<Vec<u32>>,

    // Extra request parameters shared with the other vLLM endpoints.
    /// Caller-supplied request ID used to derive the response and engine IDs.
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub priority: Option<i32>,
    #[serde(default)]
    #[validate(length(min = 1))]
    pub cache_salt: Option<String>,
    #[serde(default)]
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub media_io_kwargs: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub mm_processor_kwargs: Option<HashMap<String, Value>>,
    /// Return the model's input and output messages on the response.
    #[serde(default)]
    pub enable_response_messages: bool,
    #[serde(default)]
    pub structured_outputs: Option<Value>,
    #[serde(default)]
    pub kv_transfer_params: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub ec_transfer_params: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub vllm_xargs: Option<HashMap<String, Value>>,
}

impl crate::routes::openai::utils::types::Normalizable for ResponsesRequest {}

/// One typed input/output item of [`ResponsesInput::Items`].
///
/// Variants cover the supported Responses API history surface: messages,
/// function calls, function call outputs, and reasoning items. Other item
/// types (`custom_tool_call`, `mcp_call`, ...) fail deserialization;
/// `item_reference` is kept to report that it requires the response store.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", remote = "Self")]
pub enum ResponseInputItem {
    /// Chat-style message item (`EasyInputMessageParam` or
    /// `ResponseOutputMessage` shapes both land here).
    Message(ResponseInputMessage),
    FunctionCall(ResponseInputFunctionCall),
    FunctionCallOutput(ResponseInputFunctionCallOutput),
    Reasoning(ResponseInputReasoning),
    ItemReference {},
}

impl<'de> Deserialize<'de> for ResponseInputItem {
    /// Insert `type: "message"` when a message-shaped item omits it, mirroring
    /// the Python `input_item_parsing` validator.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut item = serde_json::Map::deserialize(deserializer)?;
        if !item.contains_key("type") && item.contains_key("role") {
            item.insert("type".to_owned(), Value::from("message"));
        }
        // Calls the derived (`remote = "Self"`) implementation.
        ResponseInputItem::deserialize(Value::Object(item)).map_err(de::Error::custom)
    }
}

/// Role of one message input item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseInputRole {
    User,
    System,
    Developer,
    Assistant,
}

/// One message-shaped input item.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseInputMessage {
    pub role: ResponseInputRole,
    pub content: ResponseMessageContent,
    /// `in_progress`/`incomplete` on the final assistant item requests a
    /// partial-completion continuation (see `should_continue_final_message`
    /// in the Python frontend).
    #[serde(default)]
    pub status: Option<ResponseItemStatus>,
}

/// Message content: either a plain string or a list of typed parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseMessageContent {
    Text(String),
    Parts(Vec<ResponseInputContentPart>),
}

/// One message content part.
///
/// `output_text`/`refusal` appear on assistant history items echoed back by
/// clients; the rest appear on user/system/developer inputs. File inputs are
/// rejected during conversion.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputContentPart {
    InputText {
        text: String,
    },
    InputImage {
        #[serde(default)]
        image_url: Option<String>,
        #[serde(default)]
        detail: Option<ImageDetail>,
        #[serde(default)]
        file_id: Option<String>,
    },
    InputAudio {
        input_audio: InputAudio,
    },
    InputFile {
        #[serde(flatten)]
        extra: serde_json::Map<String, Value>,
    },
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Option<Value>,
    },
    Refusal {
        refusal: String,
    },
}

/// One `function_call` input item (assistant tool invocation in history).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseInputFunctionCall {
    pub call_id: String,
    pub name: String,
    /// Namespace of a function from a `namespace` tool.
    #[serde(default)]
    pub namespace: Option<String>,
    pub arguments: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<ResponseItemStatus>,
}

/// One `function_call_output` input item (tool result in history).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseInputFunctionCallOutput {
    pub call_id: String,
    pub output: ResponseMessageContent,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<ResponseItemStatus>,
}

/// One `reasoning` input item (assistant reasoning in history).
///
/// `content` carries the full reasoning text; `summary` is used as a
/// fallback when `content` is absent, with a warning (mirroring the Python
/// converter). `encrypted_content` requires a response store and is
/// rejected.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseInputReasoning {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub summary: Option<Vec<TextPart>>,
    #[serde(default)]
    pub content: Option<Vec<TextPart>>,
    #[serde(default)]
    pub encrypted_content: Option<String>,
    #[serde(default)]
    pub status: Option<ResponseItemStatus>,
}

/// One plain-text part used by reasoning items.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextPart {
    ReasoningText { text: String },
    SummaryText { text: String },
}

impl TextPart {
    /// Construct one `reasoning_text` part with the given text.
    pub fn reasoning_text(text: String) -> Self {
        Self::ReasoningText { text }
    }

    /// Return the carried text.
    pub fn text(&self) -> &str {
        match self {
            Self::ReasoningText { text } | Self::SummaryText { text } => text,
        }
    }
}

/// Status of a response object or one of its output items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseItemStatus {
    Queued,
    InProgress,
    Incomplete,
    Failed,
    Cancelling,
    Cancelled,
    Completed,
}

/// One `output_text` content part of an output message item.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputContentPart {
    OutputText {
        text: String,
        annotations: Vec<Value>,
        /// Empty on `response.content_part.added`, like Python; omitted
        /// elsewhere.
        ///
        /// TODO: populated when `include=message.output_text.logprobs` is
        /// implemented.
        logprobs: Option<Vec<Value>>,
    },
    Refusal {
        refusal: String,
    },
}

/// One item in the `output` array of a response (or streamed through
/// `response.output_item.*` events).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputItem {
    Message {
        id: String,
        role: AssistantRole,
        status: ResponseItemStatus,
        content: Vec<ResponseOutputContentPart>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        /// The function name, without its namespace.
        name: String,
        /// Namespace of a function from a `namespace` tool.
        #[serde(default)]
        namespace: Option<String>,
        arguments: String,
        #[serde(default)]
        status: Option<ResponseItemStatus>,
    },
    Reasoning {
        id: String,
        summary: Vec<TextPart>,
        #[serde(default)]
        content: Option<Vec<TextPart>>,
        #[serde(default)]
        status: Option<ResponseItemStatus>,
    },
}

impl ResponseOutputItem {
    /// Mark a truncated output item incomplete, matching Python.
    pub(super) fn mark_incomplete(&mut self) {
        match self {
            Self::Message { status, .. } => *status = ResponseItemStatus::Incomplete,
            Self::FunctionCall { status, .. } | Self::Reasoning { status, .. } => {
                *status = Some(ResponseItemStatus::Incomplete)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssistantRole;

impl serde::Serialize for AssistantRole {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str("assistant")
    }
}

/// Usage block of a completed response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: usize,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens: usize,
    pub output_tokens_details: OutputTokensDetails,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputTokensDetails {
    pub cached_tokens: usize,
    /// Prompt tokens newly written to the prefix cache.
    pub cache_write_tokens: usize,
    /// vLLM extension: per-turn token counts, populated only by multi-turn
    /// builtin-tool execution. Always empty in this single-turn frontend.
    pub input_tokens_per_turn: Vec<usize>,
    pub cached_tokens_per_turn: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Reasoning tokens reported by the shared chat output processor.
    pub reasoning_tokens: usize,
    pub tool_output_tokens: usize,
    /// vLLM extension: per-turn token counts, populated only by multi-turn
    /// builtin-tool execution. Always empty in this single-turn frontend.
    pub output_tokens_per_turn: Vec<usize>,
    pub tool_output_tokens_per_turn: Vec<usize>,
}

/// `incomplete_details` block set when generation stopped early.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IncompleteDetails {
    /// Currently only `max_output_tokens` is produced; `content_filter` is
    /// not implemented (same as the Python frontend).
    pub reason: String,
}

/// The top-level response object returned by non-streaming requests and
/// carried by lifecycle streaming events.
///
/// Mirrors the Python vLLM `ResponsesResponse` class, which serializes unset
/// optional fields as `null` rather than omitting them. Sampling fields use
/// explicit request values or the model defaults resolved by the handler.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: ResponseObject,
    pub created_at: u64,
    pub status: ResponseItemStatus,
    #[serde(default)]
    pub error: Option<ResponseError>,
    pub background: bool,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    pub instructions: Option<String>,
    pub max_output_tokens: u32,
    #[serde(default)]
    pub max_tool_calls: Option<u32>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    pub parallel_tool_calls: bool,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub prompt: Option<Value>,
    #[serde(default)]
    pub reasoning: Option<ResponsesReasoning>,
    pub service_tier: ServiceTier,
    pub temperature: f32,
    #[serde(default)]
    pub text: Option<ResponseTextConfig>,
    /// Echoed resolved tool choice.
    pub tool_choice: ResponseToolChoice,
    /// Echoed request tools.
    pub tools: Vec<ResponsesTool>,
    pub top_p: f32,
    #[serde(default)]
    pub top_logprobs: Option<i32>,
    pub truncation: ResponseTruncation,
    #[serde(default)]
    pub usage: Option<ResponseUsage>,
    #[serde(default)]
    pub user: Option<String>,
    /// vLLM-specific per-request metrics. Omitted unless enabled server-side.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<PerRequestMetrics>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    #[serde(default)]
    pub kv_transfer_params: Option<Value>,
    #[serde(default)]
    pub ec_transfer_params: Option<Value>,
}

/// The failure detail carried by a failed Responses API response.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct ResponseObject;

impl serde::Serialize for ResponseObject {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str("response")
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::json;

    use super::*;

    #[test]
    fn request_deserializes_minimal_string_input() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hello",
        }))
        .unwrap();
        assert!(matches!(request.input, ResponsesInput::Text(text) if text == "hello"));
        assert!(request.include_reasoning);
        assert_eq!(request.watermarking, None);
        assert_eq!(request.model.as_deref(), Some("test-model"));
    }

    #[test]
    fn request_normalizes_top_k_disable_values() {
        for (top_k, expected) in [
            (json!(-1), Some(0)),
            (json!(0), Some(0)),
            (json!(20), Some(20)),
        ] {
            let request: ResponsesRequest = serde_json::from_value(json!({
                "input": "hello",
                "top_k": top_k,
            }))
            .expect("deserialize top_k");

            assert_eq!(request.top_k, expected);
        }
    }

    #[test]
    fn request_rejects_empty_cache_salt() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "input": "hello",
            "cache_salt": "",
        }))
        .expect("deserialize request");

        assert!(request.validate().is_err());
    }

    #[test]
    fn request_deserializes_typed_input_items() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": [
                {"role": "user", "content": "hi"},
                {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "42"},
                {"type": "reasoning", "id": "rs_1", "content": [{"type": "reasoning_text", "text": "..."}]},
            ],
        }))
        .unwrap();
        let ResponsesInput::Items(items) = request.input else {
            panic!("expected item list");
        };
        assert!(matches!(
            &items[..],
            [
                ResponseInputItem::Message(ResponseInputMessage {
                    role: ResponseInputRole::User,
                    ..
                }),
                ResponseInputItem::FunctionCall(_),
                ResponseInputItem::FunctionCallOutput(_),
                ResponseInputItem::Reasoning(_),
            ]
        ));
    }

    #[test]
    fn request_reports_unsupported_input_item_type() {
        let error = serde_json::from_value::<ResponsesRequest>(json!({
            "input": [
                {"role": "user", "content": "hi"},
                {"type": "custom_tool_call", "call_id": "call_1", "name": "f", "input": ""},
            ],
        }))
        .unwrap_err();

        assert!(error.to_string().contains("unknown variant `custom_tool_call`"));
    }

    #[test]
    fn response_serializes_without_none_fields() {
        let response = ResponsesResponse {
            id: "resp_1".to_string(),
            object: ResponseObject,
            created_at: 1,
            status: ResponseItemStatus::Completed,
            error: None,
            background: false,
            incomplete_details: None,
            instructions: Some("be terse".to_string()),
            max_output_tokens: 16,
            max_tool_calls: None,
            metadata: None,
            model: "test-model".to_string(),
            output: vec![ResponseOutputItem::Message {
                id: "msg_1".to_string(),
                role: AssistantRole,
                status: ResponseItemStatus::Completed,
                content: vec![ResponseOutputContentPart::OutputText {
                    text: "hi".to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
            }],
            parallel_tool_calls: true,
            previous_response_id: None,
            prompt: None,
            reasoning: None,
            service_tier: ServiceTier::Auto,
            temperature: 1.0,
            text: None,
            tool_choice: ResponseToolChoice::Value(ToolChoiceValue::Auto),
            tools: vec![],
            top_p: 1.0,
            top_logprobs: None,
            truncation: ResponseTruncation::Disabled,
            usage: Some(ResponseUsage {
                input_tokens: 3,
                input_tokens_details: InputTokensDetails {
                    cached_tokens: 1,
                    cache_write_tokens: 2,
                    input_tokens_per_turn: vec![],
                    cached_tokens_per_turn: vec![],
                },
                output_tokens: 2,
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: 0,
                    tool_output_tokens: 0,
                    output_tokens_per_turn: vec![],
                    tool_output_tokens_per_turn: vec![],
                },
                total_tokens: 5,
            }),
            user: None,
            metrics: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            kv_transfer_params: None,
            ec_transfer_params: None,
        };

        expect![[r#"
            {
              "id": "resp_1",
              "object": "response",
              "created_at": 1,
              "status": "completed",
              "error": null,
              "background": false,
              "incomplete_details": null,
              "instructions": "be terse",
              "max_output_tokens": 16,
              "max_tool_calls": null,
              "metadata": null,
              "model": "test-model",
              "output": [
                {
                  "type": "message",
                  "id": "msg_1",
                  "role": "assistant",
                  "status": "completed",
                  "content": [
                    {
                      "type": "output_text",
                      "text": "hi",
                      "annotations": []
                    }
                  ]
                }
              ],
              "parallel_tool_calls": true,
              "previous_response_id": null,
              "prompt": null,
              "reasoning": null,
              "service_tier": "auto",
              "temperature": 1.0,
              "text": null,
              "tool_choice": "auto",
              "tools": [],
              "top_p": 1.0,
              "top_logprobs": null,
              "truncation": "disabled",
              "usage": {
                "input_tokens": 3,
                "input_tokens_details": {
                  "cached_tokens": 1,
                  "cache_write_tokens": 2,
                  "input_tokens_per_turn": [],
                  "cached_tokens_per_turn": []
                },
                "output_tokens": 2,
                "output_tokens_details": {
                  "reasoning_tokens": 0,
                  "tool_output_tokens": 0,
                  "output_tokens_per_turn": [],
                  "tool_output_tokens_per_turn": []
                },
                "total_tokens": 5
              },
              "user": null,
              "presence_penalty": 0.0,
              "frequency_penalty": 0.0,
              "kv_transfer_params": null,
              "ec_transfer_params": null
            }"#]]
        .assert_eq(&serde_json::to_string_pretty(&response).unwrap());
    }
}
