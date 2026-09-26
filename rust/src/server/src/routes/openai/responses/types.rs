// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! OpenAI Responses API request/response wire types for the Rust frontend.
//!
//! Request types mirror the Python vLLM `ResponsesRequest` class in
//! `vllm/entrypoints/openai/responses/protocol.py`; output item and event
//! types mirror the `openai.types.responses` SDK shapes emitted by the Python
//! frontend.

use std::collections::HashMap;

use llm_multimodal::ImageDetail;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;
use vllm_chat::ReasoningEffort;

use crate::routes::openai::utils::types::StringOrArray;

/// Responses API `input` field: either a plain string (single user message)
/// or a list of input/output items.
///
/// Items stay as raw JSON values at this layer so that conversion in
/// `convert.rs` can normalize legacy shapes (e.g. type-less messages) and
/// report precise per-item errors, mirroring the Python `input_item_parsing`
/// validator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    /// Simple text input; rendered as one user message.
    Text(String),
    /// Ordered list of input/output items (messages, function calls,
    /// function call outputs, reasoning items, ...).
    Items(Vec<Value>),
}

/// Responses API reasoning configuration.
///
/// Mirrors the `Reasoning` shared type from the OpenAI SDK. `summary` is
/// accepted but currently ignored; reasoning summaries are not generated.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponsesReasoning {
    #[serde(default)]
    pub effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub summary: Option<String>,
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
/// `verbosity` is accepted for compatibility but currently ignored.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponseTextConfig {
    #[serde(default)]
    pub format: Option<ResponseTextFormat>,
    #[serde(default)]
    pub verbosity: Option<String>,
}

/// Responses API `tool_choice` field.
///
/// The string form covers `none`/`auto`/`required`; the object form is kept
/// as raw JSON so `convert.rs` can distinguish supported
/// (`{"type": "function", "name": ...}`) from unsupported object choices and
/// report precise errors.
///
/// TODO: support `allowed_tools` object choices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseToolChoice {
    /// `none` / `auto` / `required`.
    Mode(String),
    /// Object form, e.g. `{"type": "function", "name": "..."}`.
    Object(Value),
}

/// Responses API request body accepted by the Rust frontend.
///
/// Mirrors the Python vLLM `ResponsesRequest` class; unsupported built-in
/// tool types and store-dependent features are validated in
/// `validate.rs`/`convert.rs` rather than at deserialization time so errors
/// can carry the reporting `param`.
///
/// TODO: `background`, `store=true` retention, `previous_response_id`, and
/// `max_tool_calls` require a server-side response store, which the Rust
/// frontend does not have yet; see `validate.rs` for the enforced behavior.
/// TODO: `include=message.output_text.logprobs` is accepted but output
/// logprobs are not emitted yet.
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
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
    /// Function tools available to the model. Built-in tool types
    /// (web search, code interpreter, MCP, ...) are not supported and are
    /// rejected during conversion.
    #[serde(default)]
    pub tools: Vec<Value>,
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
    pub metadata: Option<Value>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
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
    pub service_tier: Option<String>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub background: Option<bool>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub top_logprobs: Option<i32>,
    #[serde(default)]
    pub text: Option<ResponseTextConfig>,
    #[serde(default)]
    pub truncation: Option<String>,
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
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    #[serde(default)]
    pub min_tokens: Option<u32>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(default)]
    pub stop_token_ids: Option<Vec<u32>>,

    // Extra request parameters shared with the other vLLM endpoints.
    /// Caller-supplied request ID; also used as the response ID.
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub priority: Option<i32>,
    #[serde(default)]
    pub cache_salt: Option<String>,
    #[serde(default)]
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub structured_outputs: Option<Value>,
    #[serde(default)]
    pub kv_transfer_params: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub ec_transfer_params: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub vllm_xargs: Option<HashMap<String, Value>>,
}

fn default_true() -> bool {
    true
}

impl crate::routes::openai::utils::types::Normalizable for ResponsesRequest {}

/// One typed input/output item, parsed from the raw JSON in
/// [`ResponsesInput::Items`] during conversion.
///
/// Variants cover the supported Responses API history surface: messages,
/// function calls, function call outputs, and reasoning items. Item types
/// requiring server-side state or unsupported modalities (`item_reference`,
/// `custom_tool_call`, `mcp_call`, file inputs, ...) are rejected with
/// explicit errors by the converter.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum ResponseInputItem {
    /// Chat-style message item (`EasyInputMessageParam` or
    /// `ResponseOutputMessage` shapes both land here; `convert.rs` inserts
    /// `type: "message"` when absent, mirroring the Python validator).
    Message(ResponseInputMessage),
    FunctionCall(ResponseInputFunctionCall),
    FunctionCallOutput(ResponseInputFunctionCallOutput),
    Reasoning(ResponseInputReasoning),
}

/// One message-shaped input item.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct ResponseInputMessage {
    pub role: String,
    pub content: ResponseMessageContent,
    /// `in_progress`/`incomplete` on the final assistant item requests a
    /// partial-completion continuation (see `should_continue_final_message`
    /// in the Python frontend).
    #[serde(default)]
    pub status: Option<String>,
}

/// Message content: either a plain string or a list of typed parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub(super) enum ResponseMessageContent {
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
pub(super) enum ResponseInputContentPart {
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
        data: String,
        #[serde(default)]
        format: Option<String>,
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
pub(super) struct ResponseInputFunctionCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
}

/// One `function_call_output` input item (tool result in history).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct ResponseInputFunctionCallOutput {
    pub call_id: String,
    pub output: ResponseMessageContent,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
}

/// One `reasoning` input item (assistant reasoning in history).
///
/// `content` carries the full reasoning text; `summary` is used as a
/// fallback when `content` is absent, with a warning (mirroring the Python
/// converter). `encrypted_content` requires a response store and is
/// rejected.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct ResponseInputReasoning {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub summary: Option<Vec<TextPart>>,
    #[serde(default)]
    pub content: Option<Vec<TextPart>>,
    #[serde(default)]
    pub encrypted_content: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
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
        /// TODO: populated when `include=message.output_text.logprobs` is
        /// implemented.
        #[serde(skip)]
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
        name: String,
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
    /// vLLM extension: per-turn token counts, populated only by multi-turn
    /// builtin-tool execution. Always empty in this single-turn frontend.
    pub input_tokens_per_turn: Vec<usize>,
    pub cached_tokens_per_turn: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// TODO: count reasoning tokens from the parser for non-streaming
    /// parity with the Python frontend. Currently always 0.
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
/// Mirrors the Python vLLM `ResponsesResponse` class. Sampling fields are
/// echoed from the request with OpenAI defaults (1.0) applied; note the
/// Python frontend echoes engine-resolved values instead when the model
/// provides generation defaults.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: ResponseObject,
    pub created_at: u64,
    pub status: ResponseItemStatus,
    pub background: bool,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub max_tool_calls: Option<u32>,
    #[serde(default)]
    pub metadata: Option<Value>,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    pub parallel_tool_calls: bool,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub prompt: Option<Value>,
    #[serde(default)]
    pub reasoning: Option<ResponsesReasoning>,
    pub service_tier: String,
    pub temperature: f32,
    #[serde(default)]
    pub text: Option<ResponseTextConfig>,
    /// Echoed resolved tool choice. Raw JSON so named function choices round
    /// trip verbatim.
    pub tool_choice: Value,
    /// Echoed request tools (raw JSON, as received).
    pub tools: Vec<Value>,
    pub top_p: f32,
    #[serde(default)]
    pub top_logprobs: Option<i32>,
    pub truncation: String,
    #[serde(default)]
    pub usage: Option<ResponseUsage>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub kv_transfer_params: Option<Value>,
    #[serde(default)]
    pub ec_transfer_params: Option<Value>,
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
        assert_eq!(request.model.as_deref(), Some("test-model"));
    }

    #[test]
    fn request_deserializes_item_list_input_as_raw_values() {
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
        match request.input {
            ResponsesInput::Items(items) => {
                assert_eq!(items.len(), 4);
                assert_eq!(items[0]["role"], "user");
                assert_eq!(items[1]["type"], "function_call");
            }
            other => panic!("expected item list, got {other:?}"),
        }
    }

    #[test]
    fn response_serializes_without_none_fields() {
        let response = ResponsesResponse {
            id: "resp_1".to_string(),
            object: ResponseObject,
            created_at: 1,
            status: ResponseItemStatus::Completed,
            background: false,
            incomplete_details: None,
            instructions: Some("be terse".to_string()),
            max_output_tokens: None,
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
            service_tier: "auto".to_string(),
            temperature: 1.0,
            text: None,
            tool_choice: json!("auto"),
            tools: vec![],
            top_p: 1.0,
            top_logprobs: None,
            truncation: "disabled".to_string(),
            usage: Some(ResponseUsage {
                input_tokens: 3,
                input_tokens_details: InputTokensDetails {
                    cached_tokens: 1,
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
            presence_penalty: None,
            frequency_penalty: None,
            kv_transfer_params: None,
            ec_transfer_params: None,
        };

        expect![[r#"
            {
              "id": "resp_1",
              "object": "response",
              "created_at": 1,
              "status": "completed",
              "background": false,
              "instructions": "be terse",
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
              "service_tier": "auto",
              "temperature": 1.0,
              "tool_choice": "auto",
              "tools": [],
              "top_p": 1.0,
              "truncation": "disabled",
              "usage": {
                "input_tokens": 3,
                "input_tokens_details": {
                  "cached_tokens": 1,
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
              }
            }"#]]
        .assert_eq(&serde_json::to_string_pretty(&response).unwrap());
    }
}
