// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Conversion between the OpenAI Responses API surface and the internal
//! `vllm-chat` request types.
//!
//! Message lowering mirrors the Python frontend's `construct_input_messages`
//! and `_construct_message_from_response_item` in
//! `vllm/entrypoints/openai/responses/utils.py`: reasoning, message, and
//! function-call items belonging to one assistant turn are merged into a
//! single assistant chat message, and function-call output items become
//! tool-response messages.

use serde_json::Value;
use tracing::warn;
use uuid::Uuid;
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatMessage,
    ChatOptions, ChatRequest, ChatTool, ChatToolChoice, GenerationPromptMode, ReasoningEffort,
    ResolvedToolContext,
};
use vllm_text::SamplingParams;
use vllm_text::output::TextDecodeOptions;

use super::types::{
    AssistantRole, IncompleteDetails, InputTokensDetails, OutputTokensDetails,
    ResponseInputContentPart, ResponseInputItem, ResponseInputReasoning, ResponseItemStatus,
    ResponseMessageContent, ResponseObject, ResponseOutputContentPart, ResponseOutputItem,
    ResponseTextConfig, ResponseTextFormat, ResponseUsage, ResponsesInput, ResponsesReasoning,
    ResponsesRequest, ResponsesResponse, TextPart,
};
use super::validate;
use crate::error::{ApiError, bail_invalid_request};
use crate::lora::LoraModelResolution;
use crate::routes::openai::utils::structured_outputs::{
    JsonSchemaFormat, ResponseFormat, convert_from_response_format,
};
use crate::utils::{
    ResolvedRequestContext, convert_logit_bias, merge_ec_transfer_params, merge_kv_transfer_params,
    resolve_session_id,
};

/// Response payload metadata echoed back on the final response object and
/// carried by lifecycle streaming events.
pub(crate) struct ResponseMeta {
    pub model: String,
    pub instructions: Option<String>,
    pub metadata: Option<Value>,
    pub tools: Vec<Value>,
    pub tool_choice: Value,
    pub parallel_tool_calls: bool,
    pub max_output_tokens: Option<u32>,
    pub max_tool_calls: Option<u32>,
    pub previous_response_id: Option<String>,
    pub reasoning: Option<ResponsesReasoning>,
    pub service_tier: String,
    pub temperature: f32,
    pub top_p: f32,
    pub top_logprobs: Option<i32>,
    pub text: Option<ResponseTextConfig>,
    pub truncation: String,
    pub user: Option<String>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub include_reasoning: bool,
}

/// Lowered responses request plus the response metadata carried by every SSE
/// lifecycle event.
pub(crate) struct PreparedResponsesRequest {
    pub request_id: String,
    pub meta: ResponseMeta,
    pub chat_request: ChatRequest,
}

/// Validate and lower one Responses API request into the internal chat
/// format.
///
/// `lora_resolution.model_names` must be non-empty; the first entry is used as
/// the base `model` field in the response when no LoRA adapter is selected.
pub(crate) fn prepare_responses_request(
    request: ResponsesRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
) -> Result<PreparedResponsesRequest, ApiError> {
    validate::validate_request_compat(&request)?;
    validate_model(&request, lora_resolution)?;

    let request_id = request
        .request_id
        .clone()
        .unwrap_or_else(|| format!("resp_{}", Uuid::new_v4().simple()));

    let ResponsesRequest {
        model: _,
        input,
        instructions,
        tools,
        tool_choice,
        parallel_tool_calls,
        max_output_tokens,
        max_tool_calls,
        metadata,
        previous_response_id,
        prompt: _,
        reasoning,
        include_reasoning,
        service_tier,
        store: _,
        background: _,
        stream,
        temperature,
        top_p,
        top_k,
        top_logprobs,
        text,
        truncation,
        user,
        include: _,
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
        seed,
        stop,
        ignore_eos,
        skip_special_tokens,
        include_stop_str_in_output,
        min_tokens,
        logit_bias,
        stop_token_ids,
        request_id: _,
        session_id,
        priority,
        cache_salt,
        chat_template_kwargs,
        structured_outputs,
        kv_transfer_params,
        ec_transfer_params,
        vllm_xargs,
    } = request;

    let response_model = lora_resolution
        .lora_request
        .as_ref()
        .map(|request| request.lora_name.clone())
        .unwrap_or_else(|| lora_resolution.model_names.first().cloned().unwrap_or_default());

    let (messages, continue_final) = convert_input(&instructions, input)?;
    let generation_prompt_mode = if continue_final {
        GenerationPromptMode::ContinueFinalAssistant
    } else {
        GenerationPromptMode::StartNewAssistant
    };

    let converted_tools = convert_tools(&tools)?;
    let requested_tool_choice = normalize_tool_choice(tool_choice, converted_tools.is_empty())?;
    let tool_context = ResolvedToolContext::new(
        &messages,
        converted_tools,
        requested_tool_choice,
        parallel_tool_calls.unwrap_or(true),
    )
    .map_err(|error| {
        ApiError::invalid_request(
            format!("failed to resolve request tools: {error}"),
            Some("tools"),
        )
    })?;

    let tool_choice_echo = echo_tool_choice(&tool_context);

    let reasoning_effort = reasoning.as_ref().and_then(|reasoning| reasoning.effort);

    // When reasoning is requested, activate thinking for models whose chat
    // templates require explicit opt-in; mirrors `build_chat_params` in the
    // Python frontend.
    let mut template_kwargs = chat_template_kwargs.unwrap_or_default();
    if let Some(effort) = reasoning_effort
        && !template_kwargs.contains_key("enable_thinking")
    {
        template_kwargs.insert(
            "enable_thinking".to_string(),
            Value::Bool(!matches!(effort, ReasoningEffort::None)),
        );
    }

    // Map the flat Responses `text.format` shape onto the nested
    // chat-completions `response_format` shape before conversion.
    let response_format =
        text.as_ref().and_then(|text| text.format.as_ref()).map(|format| match format {
            ResponseTextFormat::Text => ResponseFormat::Text,
            ResponseTextFormat::JsonObject => ResponseFormat::JsonObject,
            ResponseTextFormat::JsonSchema {
                name,
                description,
                schema,
                strict,
            } => ResponseFormat::JsonSchema {
                json_schema: JsonSchemaFormat {
                    name: name.clone(),
                    description: description.clone(),
                    schema: schema.clone(),
                    strict: *strict,
                },
            },
        });
    if structured_outputs.is_some() && response_format.is_some() {
        bail_invalid_request!(
            param = "structured_outputs",
            "Cannot specify both structured_outputs and text.format"
        );
    }
    let structured_outputs =
        convert_from_response_format(response_format.as_ref(), &structured_outputs)?;
    let response_format_value = response_format
        .as_ref()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|error| {
            ApiError::invalid_request(
                format!("failed to serialize text.format: {error}"),
                Some("text"),
            )
        })?;

    let session_id = resolve_session_id(&ctx, session_id.as_deref(), vllm_xargs.as_ref());

    let chat_request = ChatRequest {
        request_id: request_id.clone(),
        messages,
        sampling_params: SamplingParams {
            temperature,
            top_p,
            top_k,
            seed,
            max_tokens: max_output_tokens,
            min_tokens,
            thinking_token_budget: None,
            logprobs: None,
            prompt_logprobs: None,
            min_p: None,
            frequency_penalty,
            presence_penalty,
            repetition_penalty,
            repetition_detection: None,
            stop_token_ids,
            ignore_eos,
            logit_bias: convert_logit_bias(logit_bias)?,
            allowed_token_ids: None,
            bad_words: None,
            logprob_token_ids: None,
            structured_outputs,
            skip_reading_prefix_cache: None,
            vllm_xargs: merge_kv_transfer_params(
                merge_ec_transfer_params(vllm_xargs, ec_transfer_params.as_ref()),
                kv_transfer_params.as_ref(),
            ),
        },
        chat_options: ChatOptions {
            generation_prompt_mode,
            chat_template: None,
            reasoning_effort,
            response_format: response_format_value,
            template_kwargs,
        },
        tool_context,
        decode_options: TextDecodeOptions {
            skip_special_tokens,
            include_stop_str_in_output,
            stop_strings: stop.map(|stop| stop.into_vec()),
            min_tokens: min_tokens.unwrap_or(0),
        },
        intermediate: stream,
        priority: ctx.priority.or(priority).unwrap_or(0),
        documents: None,
        cache_salt,
        add_special_tokens: false,
        data_parallel_rank: ctx.data_parallel_rank,
        session_id,
        lora_request: lora_resolution.lora_request.clone(),
    };

    let meta = ResponseMeta {
        model: response_model,
        instructions,
        metadata,
        tools,
        tool_choice: tool_choice_echo,
        parallel_tool_calls: parallel_tool_calls.unwrap_or(true),
        max_output_tokens,
        max_tool_calls,
        previous_response_id,
        reasoning,
        service_tier: service_tier.unwrap_or_else(|| "auto".to_string()),
        temperature: temperature.unwrap_or(1.0),
        top_p: top_p.unwrap_or(1.0),
        top_logprobs,
        text,
        truncation: truncation.unwrap_or_else(|| "disabled".to_string()),
        user,
        presence_penalty,
        frequency_penalty,
        include_reasoning,
    };

    Ok(PreparedResponsesRequest {
        request_id,
        meta,
        chat_request,
    })
}

/// Check the requested model against the served models, if one was given.
fn validate_model(
    request: &ResponsesRequest,
    lora_resolution: &LoraModelResolution,
) -> Result<(), ApiError> {
    match &request.model {
        None => Ok(()),
        Some(model) if lora_resolution.model_names.iter().any(|name| name == model) => Ok(()),
        Some(model) => Err(ApiError::model_not_found(model.clone())),
    }
}

/// Convert Responses API tools into internal chat tools.
///
/// Only `function` tools are supported; built-in tool types
/// (`web_search_preview`, `code_interpreter`, `image_generation`, `mcp`, ...)
/// are rejected because this frontend has no builtin-tool executor.
fn convert_tools(tools: &[Value]) -> Result<Vec<ChatTool>, ApiError> {
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let tool_type = tool.get("type").and_then(Value::as_str).unwrap_or_default();
            if tool_type != "function" {
                bail_invalid_request!(
                    param = "tools",
                    "Only function tools are supported by this frontend; got tool type \
                     '{tool_type}' at index {index}."
                );
            }
            serde_json::from_value(tool.clone()).map_err(|error| {
                ApiError::invalid_request(
                    format!("failed to parse function tool at index {index}: {error}"),
                    Some("tools"),
                )
            })
        })
        .collect()
}

/// Resolve the requested tool choice, mirroring the Python
/// `check_tool_usage` validator: without tools, named function choices are
/// errors; with tools, named choices must exist (enforced by
/// [`ResolvedToolContext`]).
fn normalize_tool_choice(
    tool_choice: Option<super::types::ResponseToolChoice>,
    tools_empty: bool,
) -> Result<Option<ChatToolChoice>, ApiError> {
    use super::types::ResponseToolChoice as Choice;

    match tool_choice {
        None => Ok(None),
        Some(Choice::Mode(mode)) => match mode.as_str() {
            "none" => Ok(Some(ChatToolChoice::None)),
            "auto" => Ok(Some(ChatToolChoice::Auto)),
            "required" => Ok(Some(ChatToolChoice::Required)),
            // Unreachable: unknown modes are rejected by
            // validate_request_compat.
            _ => unreachable!(),
        },
        Some(Choice::Object(value))
            if value.get("type").and_then(Value::as_str) == Some("function") =>
        {
            if tools_empty {
                bail_invalid_request!(
                    param = "tool_choice",
                    "Tool choice 'function' not found in 'tools' parameter."
                );
            }
            let Some(name) = value.get("name").and_then(Value::as_str) else {
                bail_invalid_request!(
                    param = "tool_choice",
                    "Function tool choice requires a 'name' field."
                );
            };
            Ok(Some(ChatToolChoice::Function {
                name: name.to_string(),
            }))
        }
        Some(Choice::Object(value)) => {
            let tool_type = value.get("type").and_then(Value::as_str).unwrap_or("<missing type>");
            bail_invalid_request!(
                param = "tool_choice",
                "Tool choice type '{tool_type}' is not supported by this frontend."
            );
        }
    }
}

/// Echoed tool choice in the response, after resolution: without tools the
/// resolved choice is always `none` (Python parity).
fn echo_tool_choice(tool_context: &ResolvedToolContext) -> Value {
    match &tool_context.tool_choice {
        ChatToolChoice::None => Value::String("none".to_string()),
        ChatToolChoice::Auto => Value::String("auto".to_string()),
        ChatToolChoice::Required => Value::String("required".to_string()),
        ChatToolChoice::Function { name } => {
            serde_json::json!({"type": "function", "name": name})
        }
    }
}

/// Convert the request input into chat messages.
///
/// Returns the messages and whether generation should continue the final
/// assistant message (Anthropic-style partial completion), mirroring
/// `should_continue_final_message` in the Python frontend.
fn convert_input(
    instructions: &Option<String>,
    input: ResponsesInput,
) -> Result<(Vec<ChatMessage>, bool), ApiError> {
    let mut messages = Vec::new();
    if let Some(instructions) = instructions
        && !instructions.is_empty()
    {
        messages.push(ChatMessage::system(instructions.clone()));
    }

    let items = match input {
        ResponsesInput::Text(text) => {
            messages.push(ChatMessage::user(text));
            return Ok((messages, false));
        }
        ResponsesInput::Items(items) => items,
    };

    let continue_final = should_continue_final_message(&items);

    let mut assistant = AssistantTurn::default();
    for (index, raw) in items.iter().enumerate() {
        match parse_input_item(index, raw)? {
            ResponseInputItem::Message(message) => match message.role.as_str() {
                "system" | "developer" | "user" => {
                    assistant.flush(&mut messages);
                    let content =
                        convert_content_parts(message.role.as_str(), message.content, true)?;
                    messages.push(match message.role.as_str() {
                        "system" => ChatMessage::system(content),
                        "developer" => ChatMessage::developer(content, None),
                        _ => ChatMessage::user(content),
                    });
                }
                "assistant" => {
                    let content = convert_content_parts("assistant", message.content, false)?;
                    let text = content.try_flatten_to_text().map_err(|_| {
                        ApiError::invalid_request(
                            "assistant input items only support text content".to_string(),
                            Some("input"),
                        )
                    })?;
                    assistant.push_text(&mut messages, text);
                }
                role => {
                    bail_invalid_request!(
                        param = "input",
                        "unsupported role '{role}' in message input item"
                    );
                }
            },
            ResponseInputItem::FunctionCall(call) => {
                assistant.push_block(AssistantContentBlock::ToolCall(AssistantToolCall {
                    id: call.call_id,
                    name: call.name,
                    arguments: call.arguments,
                }));
            }
            ResponseInputItem::FunctionCallOutput(output) => {
                assistant.flush(&mut messages);
                messages.push(ChatMessage::tool_response(
                    output.output.flatten_text()?,
                    output.call_id,
                ));
            }
            ResponseInputItem::Reasoning(reasoning) => {
                if reasoning.encrypted_content.is_some() {
                    bail_invalid_request!(
                        param = "input",
                        "Encrypted reasoning content is not supported by this frontend."
                    );
                }
                let text = reasoning_content_string(&reasoning);
                if !text.is_empty() {
                    assistant.push_reasoning(&mut messages, text);
                }
            }
        }
    }
    assistant.flush(&mut messages);

    Ok((messages, continue_final))
}

/// Parse one raw input item value into a typed input item, inserting
/// `type: "message"` when a message-shaped item omits it (mirroring the
/// Python `input_item_parsing` model validator).
fn parse_input_item(index: usize, raw: &Value) -> Result<ResponseInputItem, ApiError> {
    let Some(object) = raw.as_object() else {
        bail_invalid_request!(
            param = "input",
            "input item at index {index} must be an object"
        );
    };

    let item_type = object.get("type").and_then(Value::as_str);
    let effective_type = match (item_type, object.get("role").and_then(Value::as_str)) {
        (None, Some(_)) => "message",
        (item_type, _) => item_type.unwrap_or_default(),
    };
    match effective_type {
        "message" | "function_call" | "function_call_output" | "reasoning" => {}
        "item_reference" => {
            bail_invalid_request!(
                param = "previous_response_id",
                "item_reference input items require the Responses API store, which is \
                 not supported by this frontend."
            );
        }
        "" => {
            bail_invalid_request!(
                param = "input",
                "input item at index {index} is missing required field 'type' (or 'role' \
                 for message items)."
            );
        }
        other => {
            bail_invalid_request!(
                param = "input",
                "input item at index {index} has unsupported type '{other}'."
            );
        }
    }

    let mut value = raw.clone();
    if item_type.is_none()
        && let Some(object) = value.as_object_mut()
    {
        object.insert("type".to_string(), Value::String("message".to_string()));
    }
    serde_json::from_value(value).map_err(|error| {
        ApiError::invalid_request(
            format!("failed to parse input item at index {index}: {error}"),
            Some("input"),
        )
    })
}

/// Convert message content into internal chat content.
fn convert_content_parts(
    role: &str,
    content: ResponseMessageContent,
    allow_multimodal: bool,
) -> Result<ChatContent, ApiError> {
    match content {
        ResponseMessageContent::Text(text) => Ok(ChatContent::Text(text)),
        ResponseMessageContent::Parts(parts) => {
            let mut converted = Vec::with_capacity(parts.len());
            for part in parts {
                match part {
                    ResponseInputContentPart::InputText { text }
                    | ResponseInputContentPart::OutputText { text, .. } => {
                        converted.push(ChatContentPart::text(text));
                    }
                    ResponseInputContentPart::Refusal { refusal } => {
                        // Refusals carry refusable-answer text only; keep it
                        // so templates still see the turn.
                        converted.push(ChatContentPart::text(refusal));
                    }
                    ResponseInputContentPart::InputImage {
                        image_url,
                        detail,
                        file_id,
                    } => {
                        if !allow_multimodal {
                            bail_invalid_request!(
                                param = "input",
                                "input_image parts are not supported for role '{role}'."
                            );
                        }
                        if file_id.is_some() {
                            bail_invalid_request!(
                                param = "input",
                                "input_image parts with file_id are not supported; pass \
                                 an image_url instead."
                            );
                        }
                        let Some(image_url) = image_url else {
                            bail_invalid_request!(
                                param = "input",
                                "input_image parts require 'image_url'."
                            );
                        };
                        converted.push(ChatContentPart::ImageUrl {
                            image_url,
                            detail,
                            uuid: None,
                        });
                    }
                    ResponseInputContentPart::InputAudio { data, format } => {
                        if !allow_multimodal {
                            bail_invalid_request!(
                                param = "input",
                                "input_audio parts are not supported for role '{role}'."
                            );
                        }
                        converted.push(ChatContentPart::InputAudio {
                            data,
                            format,
                            uuid: None,
                        });
                    }
                    ResponseInputContentPart::InputFile { .. } => {
                        bail_invalid_request!(
                            param = "input",
                            "input_file parts are not supported by this frontend."
                        );
                    }
                }
            }
            Ok(ChatContent::Parts(converted))
        }
    }
}

/// Accumulates content blocks across consecutive input items belonging to one
/// assistant turn, flushing at turn boundaries.
///
/// Merging rules mirror `_construct_message_from_response_item` in the Python
/// frontend: a text item starts a new assistant turn when the pending turn
/// already has visible text, a reasoning item starts a new turn when the
/// pending turn already has reasoning, and function calls always merge.
#[derive(Default)]
struct AssistantTurn {
    blocks: Vec<AssistantContentBlock>,
}

impl AssistantTurn {
    fn push_text(&mut self, messages: &mut Vec<ChatMessage>, text: String) {
        if self
            .blocks
            .iter()
            .any(|block| matches!(block, AssistantContentBlock::Text { .. }))
        {
            self.flush(messages);
        }
        self.push_block(AssistantContentBlock::Text { text });
    }

    fn push_reasoning(&mut self, messages: &mut Vec<ChatMessage>, text: String) {
        if self
            .blocks
            .iter()
            .any(|block| matches!(block, AssistantContentBlock::Reasoning { .. }))
        {
            self.flush(messages);
        }
        self.push_block(AssistantContentBlock::Reasoning { text });
    }

    fn push_block(&mut self, block: AssistantContentBlock) {
        self.blocks.push(block);
    }

    /// Flush the pending assistant turn into the message list if non-empty.
    fn flush(&mut self, messages: &mut Vec<ChatMessage>) {
        if self.blocks.is_empty() {
            return;
        }
        messages.push(ChatMessage::assistant_blocks(std::mem::take(
            &mut self.blocks,
        )));
    }
}

/// Extract the reasoning text from one reasoning input item, mirroring the
/// Python converter: prefer full `content`, fall back to the first summary
/// text with a warning.
fn reasoning_content_string(reasoning: &ResponseInputReasoning) -> String {
    if let Some(content) = &reasoning.content
        && let Some(first) = content.first()
    {
        return first.text().to_string();
    }
    if let Some(summary) = &reasoning.summary
        && let Some(first) = summary.first()
    {
        warn!(
            item_id = reasoning.id.as_deref().unwrap_or_default(),
            "Using summary text as reasoning content; use content instead"
        );
        return first.text().to_string();
    }
    String::new()
}

impl ResponseMessageContent {
    /// Flatten content into one plain string without separators, rejecting
    /// non-text parts (tool outputs support text only in this frontend).
    fn flatten_text(&self) -> Result<String, ApiError> {
        match self {
            Self::Text(text) => Ok(text.clone()),
            Self::Parts(parts) => {
                let mut flattened = String::new();
                for part in parts {
                    match part {
                        ResponseInputContentPart::InputText { text }
                        | ResponseInputContentPart::OutputText { text, .. } => {
                            flattened.push_str(text)
                        }
                        other => {
                            bail_invalid_request!(
                                param = "input",
                                "function_call_output only supports text content parts; got \
                                 part type '{}'",
                                part_type_name(other)
                            );
                        }
                    }
                }
                Ok(flattened)
            }
        }
    }
}

/// Build the response `output` items from the collected assistant message.
///
/// Mirrors `build_response_output_items` in the Python frontend: reasoning
/// comes first when present and enabled, then visible text, then tool calls —
/// except items here follow the structured block order, which preserves
/// interleaved text/tool-call ordering when the parser yields it.
pub(crate) fn build_output_items(
    message: &vllm_chat::AssistantMessage,
    include_reasoning: bool,
) -> Vec<ResponseOutputItem> {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContentBlock::Reasoning { text } if include_reasoning => {
                Some(ResponseOutputItem::Reasoning {
                    id: format!("rs_{}", Uuid::new_v4().simple()),
                    summary: vec![],
                    content: Some(vec![TextPart::reasoning_text(text.clone())]),
                    status: Some(ResponseItemStatus::Completed),
                })
            }
            AssistantContentBlock::Reasoning { .. } => None,
            AssistantContentBlock::Text { text } if !text.is_empty() => {
                Some(ResponseOutputItem::Message {
                    id: format!("msg_{}", Uuid::new_v4().simple()),
                    role: AssistantRole,
                    status: ResponseItemStatus::Completed,
                    content: vec![ResponseOutputContentPart::OutputText {
                        text: text.clone(),
                        annotations: vec![],
                        logprobs: None,
                    }],
                })
            }
            AssistantContentBlock::Text { .. } => None,
            AssistantContentBlock::ToolCall(call) => Some(ResponseOutputItem::FunctionCall {
                id: format!("fc_{}", Uuid::new_v4().simple()),
                call_id: tool_call_id(call),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                status: Some(ResponseItemStatus::Completed),
            }),
        })
        .collect()
}

/// Pick the wire `call_id` for one parsed tool call, generating one when the
/// parser did not assign an ID (Python generates `make_tool_call_id`).
fn tool_call_id(call: &AssistantToolCall) -> String {
    if call.id.is_empty() {
        format!("call_{}", Uuid::new_v4().simple())
    } else {
        call.id.clone()
    }
}

/// Build the `usage` block of a completed response.
pub(crate) fn build_usage(usage: &vllm_llm::TokenUsage) -> ResponseUsage {
    ResponseUsage {
        input_tokens: usage.prompt_token_count,
        input_tokens_details: InputTokensDetails {
            cached_tokens: usage.cached_token_count,
            input_tokens_per_turn: vec![],
            cached_tokens_per_turn: vec![],
        },
        output_tokens: usage.output_token_count,
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: 0,
            tool_output_tokens: 0,
            output_tokens_per_turn: vec![],
            tool_output_tokens_per_turn: vec![],
        },
        total_tokens: usage.prompt_token_count + usage.output_token_count,
    }
}

/// Build the top-level response object for non-streaming responses and the
/// terminal `response.completed` event.
pub(crate) fn build_response(
    meta: &ResponseMeta,
    request_id: &str,
    created_at: u64,
    output: Vec<ResponseOutputItem>,
    status: ResponseItemStatus,
    usage: Option<ResponseUsage>,
    kv_transfer_params: Option<Value>,
    ec_transfer_params: Option<Value>,
) -> ResponsesResponse {
    let incomplete_details = match status {
        ResponseItemStatus::Incomplete => Some(IncompleteDetails {
            reason: "max_output_tokens".to_string(),
        }),
        _ => None,
    };
    ResponsesResponse {
        id: request_id.to_string(),
        object: ResponseObject,
        created_at,
        status,
        background: false,
        incomplete_details,
        instructions: meta.instructions.clone(),
        max_output_tokens: meta.max_output_tokens,
        max_tool_calls: meta.max_tool_calls,
        metadata: meta.metadata.clone(),
        model: meta.model.clone(),
        output,
        parallel_tool_calls: meta.parallel_tool_calls,
        previous_response_id: meta.previous_response_id.clone(),
        prompt: None,
        reasoning: meta.reasoning.clone(),
        service_tier: meta.service_tier.clone(),
        temperature: meta.temperature,
        text: meta.text.clone(),
        tool_choice: meta.tool_choice.clone(),
        tools: meta.tools.clone(),
        top_p: meta.top_p,
        top_logprobs: meta.top_logprobs,
        truncation: meta.truncation.clone(),
        usage,
        user: meta.user.clone(),
        presence_penalty: meta.presence_penalty,
        frequency_penalty: meta.frequency_penalty,
        kv_transfer_params,
        ec_transfer_params,
    }
}

/// Return the wire `type` name of one input content part for error messages.
fn part_type_name(part: &ResponseInputContentPart) -> &'static str {
    match part {
        ResponseInputContentPart::InputText { .. } => "input_text",
        ResponseInputContentPart::InputImage { .. } => "input_image",
        ResponseInputContentPart::InputAudio { .. } => "input_audio",
        ResponseInputContentPart::InputFile { .. } => "input_file",
        ResponseInputContentPart::OutputText { .. } => "output_text",
        ResponseInputContentPart::Refusal { .. } => "refusal",
    }
}

/// Determine whether the final input item is a partial assistant message or
/// reasoning item that generation should continue.
///
/// Mirrors `should_continue_final_message` in the Python frontend.
fn should_continue_final_message(items: &[Value]) -> bool {
    let Some(last) = items.last() else {
        return false;
    };
    let status = last.get("status").and_then(Value::as_str);
    if !matches!(status, Some("in_progress" | "incomplete")) {
        return false;
    }
    match last.get("type").and_then(Value::as_str) {
        Some("reasoning") => true,
        Some("message") => last.get("role").and_then(Value::as_str) == Some("assistant"),
        // Type-less items with a role are messages.
        None => last.get("role").and_then(Value::as_str) == Some("assistant"),
        _ => false,
    }
}
