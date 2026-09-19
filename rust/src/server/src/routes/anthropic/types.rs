// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;
use vllm_chat::ReasoningEffort;

use crate::routes::openai::utils::types::Normalizable;

// ============================================================================
// Requests
// ============================================================================
// Oracle-mirrored types first: each carries the exact name of a Python vLLM
// protocol class and maps to it 1:1. Helper shapes follow in their own
// section.

/// Request body for `POST /v1/messages`.
///
/// Mirrors the Python vLLM `AnthropicMessagesRequest` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
#[allow(dead_code)] // constructed only by the /v1/messages serving path (PR 2 of 3, #47753)
pub struct AnthropicMessagesRequest {
    /// ID of the model to use.
    #[validate(length(min = 1, message = "model is required"))]
    pub model: String,

    /// Ordered conversation history.
    #[validate(length(min = 1, message = "messages must not be empty"))]
    pub messages: Vec<AnthropicMessage>,

    /// Maximum number of tokens to generate. Required by the Anthropic API.
    #[validate(range(min = 1))]
    pub max_tokens: u32,

    /// System prompt: a plain string or a list of text blocks.
    pub system: Option<SystemPrompt>,

    /// Accepted and unused, matching Python.
    pub metadata: Option<HashMap<String, Value>>,

    /// Sequences where the API will stop generating further tokens.
    pub stop_sequences: Option<Vec<String>>,

    /// If set, server-sent events are streamed as they become available.
    #[serde(default)]
    pub stream: bool,

    /// Sampling temperature.
    pub temperature: Option<f32>,

    /// Nucleus sampling parameter.
    pub top_p: Option<f32>,

    /// Top-k sampling parameter.
    pub top_k: Option<u32>,

    /// Client-side tools the model may call.
    #[validate(custom(function = "validate_tools_input_schema"))]
    pub tools: Option<Vec<AnthropicTool>>,

    /// Controls which (if any) tool is called by the model.
    pub tool_choice: Option<AnthropicToolChoice>,

    /// Output configuration: structured-output format and reasoning effort.
    pub output_config: Option<AnthropicOutputConfig>,

    // -------- vLLM protocol extensions, parity with the Python endpoint ----
    /// Salt for prefix cache isolation in multi-user environments.
    #[validate(length(min = 1))]
    pub cache_salt: Option<String>,

    /// Additional keyword args passed to the chat template renderer.
    pub chat_template_kwargs: Option<HashMap<String, Value>>,

    /// KV transfer parameters for disaggregated serving.
    pub kv_transfer_params: Option<HashMap<String, Value>>,

    /// Encoder cache transfer parameters for disaggregated serving.
    pub ec_transfer_params: Option<HashMap<String, Value>>,
}

/// Request body for `POST /v1/messages/count_tokens`: the prompt-shaping
/// subset of [`AnthropicMessagesRequest`] (no sampling parameters).
///
/// Mirrors the Python vLLM `AnthropicCountTokensRequest` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct AnthropicCountTokensRequest {
    #[validate(length(min = 1, message = "model is required"))]
    pub model: String,
    #[validate(length(min = 1, message = "messages must not be empty"))]
    pub messages: Vec<AnthropicMessage>,
    pub system: Option<SystemPrompt>,
    #[validate(custom(function = "validate_tools_input_schema"))]
    pub tools: Option<Vec<AnthropicTool>>,
    pub tool_choice: Option<AnthropicToolChoice>,
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
}

/// Validates every tool's `input_schema` is a JSON object, matching Python's
/// `validate_input_schema` ("input_schema must be a dictionary").
fn validate_tools_input_schema(tools: &[AnthropicTool]) -> Result<(), validator::ValidationError> {
    if tools.iter().any(|tool| !tool.input_schema.is_object()) {
        return Err(validator::ValidationError::new(
            "input_schema must be an object",
        ));
    }
    Ok(())
}

/// One conversation message.
///
/// Mirrors the Python vLLM `AnthropicMessage` class.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessage {
    pub role: AnthropicRole,
    pub content: MessageContent,
}

/// Typed content blocks accepted on input.
///
/// Mirrors the Python vLLM `AnthropicContentBlock` class, which models every
/// block kind as one all-optional record and coerces missing fields to empty
/// strings; the fields required here are the ones the Anthropic spec marks
/// required, so malformed blocks fail with 400 instead.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    Thinking {
        thinking: String,
        /// Python fabricates a random `uuid4` hex here; whether to copy that
        /// or omit the signature stays open on #47753.
        signature: Option<String>,
    },
    /// Accepted and dropped (opaque), matching Python.
    RedactedThinking {
        data: String,
    },
    /// `id` and `input` are optional on input, matching Python; a missing ID
    /// is fabricated in `convert.rs` using the UUID form (avoiding Python's
    /// timestamp-collision bug, #35667).
    ToolUse {
        id: Option<String>,
        name: String,
        input: Option<Value>,
    },
    ToolResult {
        tool_use_id: String,
        content: Option<ToolResultContent>,
        is_error: Option<bool>,
    },
    ToolReference {
        tool_name: String,
    },
}

/// One client-side tool definition.
///
/// Mirrors the Python vLLM `AnthropicTool` class; `defer_loading` is accepted
/// and dropped, since the Rust chat layer has no analog for it.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
    pub strict: Option<bool>,
}

/// Anthropic tool choice: `auto` / `any` / `tool` / `none`.
///
/// Mirrors the Python vLLM `AnthropicToolChoice` class; the tagged enum makes
/// `name` structurally required for `tool`, which Python enforces with a
/// model validator instead.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AnthropicToolChoice {
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
    None,
}

/// Output configuration carried by `output_config`.
///
/// Mirrors the Python vLLM `AnthropicOutputConfig` class.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(dead_code)] // constructed only by the /v1/messages serving path (PR 2 of 3, #47753)
pub struct AnthropicOutputConfig {
    pub format: Option<OutputFormat>,
    /// Lowered to `chat_options.reasoning_effort`.
    pub effort: Option<Effort>,
}

// ---------------------------------------------------------------------------
// Request helper shapes
// ---------------------------------------------------------------------------
// Shapes the Python protocol module inlines (unions, Literals, untyped
// dicts), or named shapes it restructures; each doc comment points at the
// form it lifts.

/// `system` accepts either a plain string or a list of text blocks.
///
/// Lifts the inline `str | list` union on the Python request's `system`
/// field. Python types the block list as `AnthropicContentBlock` and ignores
/// non-text blocks at conversion time; only text blocks are meaningful in
/// `system`, so the Rust type requires that shape (non-text system blocks
/// fail with 400).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemTextBlock>),
}

/// One text block inside a block-form system prompt: the deliberate
/// tightening of the oracle's all-purpose content block for the system
/// position (see [`SystemPrompt`]).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SystemTextBlock {
    /// Always `"text"`.
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
}

/// Message roles accepted on input. `system` covers inline system messages
/// (#44283); see `convert.rs` for how the merge flag is chosen.
///
/// Lifts Python's inline `Literal["user", "assistant", "system"]` annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    User,
    Assistant,
    System,
}

/// Message content: shorthand string or a list of typed blocks.
///
/// Lifts the inline `str | list[AnthropicContentBlock]` union on the Python
/// message's `content` field.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// Anthropic image source.
///
/// Lifts the source dict the oracle handles untyped. Kept loose (all fields
/// optional) because the oracle's default semantics depend on missing
/// fields: a missing `type` is treated as base64 and a missing media type
/// defaults to `image/jpeg` (see `_convert_image_source_to_url`).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: Option<String>,
    pub media_type: Option<String>,
    pub data: Option<String>,
    pub url: Option<String>,
}

/// `tool_result.content` accepts a shorthand string or nested blocks.
///
/// Lifts the inline `str | list` union the oracle reads off the tool_result
/// block.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// Structured-output format selector.
///
/// Restructures the Python vLLM `AnthropicJsonOutputFormat` class as a
/// tagged enum. Python names the attribute `json_schema` but takes `schema`
/// on the wire via a pydantic alias (alias-only, so the wire name matches
/// ours exactly); Python defaults `type` and tolerates a missing schema as a
/// silent no-op, both required here per the Anthropic spec.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(dead_code)] // constructed only by the /v1/messages serving path (PR 2 of 3, #47753)
pub enum OutputFormat {
    JsonSchema { schema: Value },
}

/// Reasoning effort accepted by the Anthropic endpoint.
///
/// Lifts the effort Literal the Python oracle inlines on its output-config
/// field (`"low" | "medium" | "high" | "xhigh" | "max"`): a strict subset of
/// the internal [`ReasoningEffort`], which additionally accepts `none` and
/// `minimal` — both rejected here with 400, matching Python.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl From<Effort> for ReasoningEffort {
    fn from(effort: Effort) -> Self {
        match effort {
            Effort::Low => Self::Low,
            Effort::Medium => Self::Medium,
            Effort::High => Self::High,
            Effort::XHigh => Self::XHigh,
            Effort::Max => Self::Max,
        }
    }
}

// ============================================================================
// Responses
// ============================================================================
// Oracle-mirrored types first; response-side helper shapes follow.

/// Non-streaming response body for `POST /v1/messages`.
///
/// Mirrors the Python vLLM `AnthropicMessagesResponse` class. Optional fields
/// serialize as explicit `null`, matching the real Anthropic wire format (the
/// Python route dumps with `exclude_none=True` and omits them instead); the
/// usage cache fields are the exception (see [`AnthropicUsage`]).
#[derive(Debug, Clone, Serialize)]
#[allow(dead_code)] // constructed only by the /v1/messages serving path (PR 2 of 3, #47753)
pub(super) struct AnthropicMessagesResponse {
    pub id: String,
    /// Always `"message"`.
    #[serde(rename = "type")]
    pub response_type: &'static str,
    /// Always `"assistant"`.
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ResponseContentBlock>,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
    // vLLM protocol extensions, matching the Python response fields.
    pub kv_transfer_params: Option<Value>,
    pub ec_transfer_params: Option<Value>,
}

/// Usage block. Mirrors the Python vLLM `AnthropicUsage` class; the two
/// cache fields are omitted entirely (not `null`) unless
/// `--enable-prompt-tokens-details` is set, matching Python's
/// `_build_anthropic_usage`.
#[derive(Debug, Clone, Serialize)]
pub(super) struct AnthropicUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
}

/// Response body for `POST /v1/messages/count_tokens`.
///
/// Mirrors the Python vLLM `AnthropicCountTokensResponse` class, including
/// its `context_management` envelope for response-shape parity.
#[derive(Debug, Clone, Serialize)]
pub(super) struct AnthropicCountTokensResponse {
    pub input_tokens: u64,
    pub context_management: AnthropicContextManagement,
}

/// Mirrors the Python vLLM `AnthropicContextManagement` class.
#[derive(Debug, Clone, Serialize)]
pub(super) struct AnthropicContextManagement {
    pub original_input_tokens: u64,
}

// ---------------------------------------------------------------------------
// Response helper shapes
// ---------------------------------------------------------------------------
// Python reuses its all-purpose request records on output; the response side
// emits dedicated strict shapes instead, since every field is under our
// control.

/// Content blocks emitted in responses, preserving thinking/text/tool_use
/// ordering end to end (one of the fidelity wins over Python, #47753).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(dead_code)] // constructed only by the /v1/messages serving path (PR 2 of 3, #47753)
pub(super) enum ResponseContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        /// Whether to fabricate a signature is an open question on #47753.
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        /// Always present on the wire (`{}` when the model emitted no
        /// arguments), unlike the tolerant request-side variant.
        input: Value,
    },
}

/// Anthropic stop reasons, lifting the stop-reason string literals the
/// oracle emits inline. `Abort` / `Repetition` finish reasons have no analog
/// and map to a `null` stop_reason pending the open question on #47753
/// (matching Python's handling of unknown finish reasons).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)] // constructed only by the /v1/messages serving path (PR 2 of 3, #47753)
pub(super) enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
}

// ============================================================================
// Errors
// ============================================================================

/// Anthropic error envelope: `{"type": "error", "error": {"type", "message"}}`.
///
/// Mirrors the Python vLLM `AnthropicErrorResponse` class.
#[derive(Debug, Clone, Serialize)]
pub(super) struct AnthropicErrorResponse {
    /// Always `"error"`.
    #[serde(rename = "type")]
    pub response_type: &'static str,
    pub error: AnthropicError,
}

/// Mirrors the Python vLLM `AnthropicError` class.
#[derive(Debug, Clone, Serialize)]
pub(super) struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ---- trait impls required by ValidatedJson (see tokenize/types.rs) ----
impl Normalizable for AnthropicMessagesRequest {} // default no-op normalize()
impl Normalizable for AnthropicCountTokensRequest {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use validator::Validate as _;

    use super::*;

    #[test]
    fn deserialize_minimal_request() {
        let req: AnthropicMessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":16,
                "messages":[{"role":"user","content":"hello"}]}"#,
        )
        .unwrap();
        req.validate().unwrap();
        assert_eq!(req.max_tokens, 16);
        assert!(!req.stream);
        assert!(matches!(req.messages[0].role, AnthropicRole::User));
        assert!(matches!(req.messages[0].content, MessageContent::Text(_)));
    }

    #[test]
    fn validate_rejects_zero_max_tokens_and_empty_messages() {
        let req: AnthropicMessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":0,
                "messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert!(req.validate().is_err());

        let req: AnthropicMessagesRequest =
            serde_json::from_str(r#"{"model":"m","max_tokens":1,"messages":[]}"#).unwrap();
        assert!(req.validate().is_err());

        // Empty model is rejected, matching Python's `validate_model`.
        let req: AnthropicMessagesRequest = serde_json::from_str(
            r#"{"model":"","max_tokens":1,
                "messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert!(req.validate().is_err());

        let req: AnthropicCountTokensRequest =
            serde_json::from_str(r#"{"model":"","messages":[{"role":"user","content":"hi"}]}"#)
                .unwrap();
        assert!(req.validate().is_err());

        // Non-object input_schema is rejected, matching Python's
        // `validate_input_schema`.
        let req: AnthropicMessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":1,
                "messages":[{"role":"user","content":"hi"}],
                "tools":[{"name":"t","input_schema":"not-an-object"}]}"#,
        )
        .unwrap();
        assert!(req.validate().is_err());
    }

    #[test]
    fn unknown_fields_ignored() {
        // Claude Code attaches cache_control to blocks and other extension
        // fields at the top level; both must deserialize harmlessly, matching
        // the pydantic behavior of the Python endpoint.
        let req: AnthropicMessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":1,"unknown_top_level":true,
                "messages":[{"role":"user","content":[
                  {"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}
                ]}]}"#,
        )
        .unwrap();
        let MessageContent::Blocks(blocks) = &req.messages[0].content else {
            panic!("expected block content");
        };
        assert!(matches!(blocks[0], AnthropicContentBlock::Text { .. }));
    }

    #[test]
    fn system_accepts_string_and_blocks() {
        let string_form: AnthropicMessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":1,"system":"be brief",
                "messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert!(matches!(string_form.system, Some(SystemPrompt::Text(_))));

        let block_form: AnthropicMessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":1,
                "system":[{"type":"text","text":"be brief"}],
                "messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        let Some(SystemPrompt::Blocks(blocks)) = &block_form.system else {
            panic!("expected block-form system prompt");
        };
        assert_eq!(blocks[0].text, "be brief");
    }

    #[test]
    fn tool_result_accepts_string_and_block_content() {
        let msg: AnthropicMessage = serde_json::from_str(
            r#"{"role":"user","content":[
                {"type":"tool_result","tool_use_id":"toolu_1","content":"ok"},
                {"type":"tool_result","tool_use_id":"toolu_2",
                 "content":[{"type":"text","text":"42"}],"is_error":false}
            ]}"#,
        )
        .unwrap();
        let MessageContent::Blocks(blocks) = &msg.content else {
            panic!("expected block content");
        };
        assert!(matches!(
            &blocks[0],
            AnthropicContentBlock::ToolResult { content: Some(ToolResultContent::Text(t)), .. } if t == "ok"
        ));
        assert!(matches!(
            &blocks[1],
            AnthropicContentBlock::ToolResult {
                content: Some(ToolResultContent::Blocks(b)),
                is_error: Some(false),
                ..
            } if b.len() == 1
        ));
    }

    #[test]
    fn assistant_history_round_trips_thinking_and_tool_use() {
        let msg: AnthropicMessage = serde_json::from_str(
            r#"{"role":"assistant","content":[
                {"type":"thinking","thinking":"hm","signature":"sig"},
                {"type":"text","text":"calling a tool"},
                {"type":"tool_use","id":"toolu_1","name":"get_weather",
                 "input":{"city":"Paris"}},
                {"type":"redacted_thinking","data":"opaque"}
            ]}"#,
        )
        .unwrap();
        let MessageContent::Blocks(blocks) = &msg.content else {
            panic!("expected block content");
        };
        assert!(matches!(&blocks[0],
            AnthropicContentBlock::Thinking { signature: Some(s), .. } if s == "sig"));
        assert!(matches!(&blocks[2],
            AnthropicContentBlock::ToolUse { id: Some(id), input: Some(input), .. }
                if id == "toolu_1" && input["city"] == "Paris"));
        assert!(matches!(
            &blocks[3],
            AnthropicContentBlock::RedactedThinking { .. }
        ));
    }

    #[test]
    fn tool_choice_variants_deserialize() {
        let parse = |s: &str| serde_json::from_str::<AnthropicToolChoice>(s).unwrap();
        assert!(matches!(
            parse(r#"{"type":"auto"}"#),
            AnthropicToolChoice::Auto { .. }
        ));
        assert!(matches!(
            parse(r#"{"type":"any","disable_parallel_tool_use":true}"#),
            AnthropicToolChoice::Any {
                disable_parallel_tool_use: Some(true)
            }
        ));
        assert!(matches!(
            parse(r#"{"type":"tool","name":"get_weather"}"#),
            AnthropicToolChoice::Tool { name, .. } if name == "get_weather"
        ));
        assert!(matches!(
            parse(r#"{"type":"none"}"#),
            AnthropicToolChoice::None
        ));
    }

    #[test]
    fn response_serializes_nulls_explicitly_and_omits_cache_fields() {
        // Anthropic wire format: stop_sequence is an explicit null; the usage
        // cache fields are omitted entirely unless prompt-token details are
        // enabled.
        let response: AnthropicMessagesResponse = AnthropicMessagesResponse {
            id: "msg_1".to_string(),
            response_type: "message",
            role: "assistant",
            model: "m".to_string(),
            content: vec![ResponseContentBlock::Text {
                text: "hi".to_string(),
            }],
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 3,
                output_tokens: 1,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
            kv_transfer_params: None,
            ec_transfer_params: None,
        };
        let json: Value = serde_json::to_value(&response).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["stop_reason"], "end_turn");
        assert!(json.as_object().unwrap().contains_key("stop_sequence"));
        assert!(json["stop_sequence"].is_null());
        let usage = json["usage"].as_object().unwrap();
        assert!(!usage.contains_key("cache_read_input_tokens"));

        let error: AnthropicErrorResponse = AnthropicErrorResponse {
            response_type: "error",
            error: AnthropicError {
                error_type: "invalid_request_error".to_string(),
                message: "boom".to_string(),
            },
        };
        let json: Value = serde_json::to_value(&error).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn validate_rejects_empty_cache_salt() {
        let req: AnthropicMessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":1,"cache_salt":"",
                "messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert!(req.validate().is_err());

        let req: AnthropicMessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":1,"cache_salt":"tenant-abc",
                "messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        req.validate().unwrap();
    }

    #[test]
    fn tool_strict_and_tool_reference_deserialize() {
        let tool: AnthropicTool = serde_json::from_str(
            r#"{"name":"t","input_schema":{"type":"object"},
                "strict":true,"defer_loading":false}"#,
        )
        .unwrap();
        assert_eq!(tool.strict, Some(true));

        let block: AnthropicContentBlock =
            serde_json::from_str(r#"{"type":"tool_reference","tool_name":"web_search"}"#).unwrap();
        assert!(matches!(
            block,
            AnthropicContentBlock::ToolReference { tool_name } if tool_name == "web_search"
        ));
    }

    #[test]
    fn output_config_effort_matches_python_literal() {
        let parse = |s: &str| {
            serde_json::from_str::<AnthropicOutputConfig>(&format!(r#"{{"effort":"{s}"}}"#))
        };
        for accepted in ["low", "medium", "high", "xhigh", "max"] {
            parse(accepted).unwrap();
        }
        // Internal `ReasoningEffort` levels the Anthropic endpoint rejects,
        // matching Python's effort Literal.
        assert!(parse("none").is_err());
        assert!(parse("minimal").is_err());
    }
}
