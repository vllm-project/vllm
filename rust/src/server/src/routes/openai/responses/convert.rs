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

use itertools::Itertools as _;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::warn;
use uuid::Uuid;
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatMessage,
    ChatOptions, ChatRequest, ChatTool, ChatToolChoice, Error as ChatError, GenerationPromptMode,
    ResolvedToolContext,
};
use vllm_text::SamplingParams;
use vllm_text::output::TextDecodeOptions;

use super::types::{
    AssistantRole, IncompleteDetails, InputTokensDetails, OutputTokensDetails,
    ResponseInputContentPart, ResponseInputItem, ResponseInputReasoning, ResponseInputRole,
    ResponseItemStatus, ResponseMessageContent, ResponseObject, ResponseOutputContentPart,
    ResponseOutputItem, ResponseTextConfig, ResponseTextFormat, ResponseToolChoice,
    ResponseToolChoiceObject, ResponseTruncation, ResponseUsage, ResponsesInput,
    ResponsesReasoning, ResponsesRequest, ResponsesResponse, ResponsesTool, ServiceTier, TextPart,
};
use super::validate;
use crate::error::{ApiError, bail_invalid_request};
use crate::lora::LoraModelResolution;
use crate::routes::openai::utils::resolve_generation_prompt_truncation;
use crate::routes::openai::utils::structured_outputs::{
    JsonSchemaFormat, ResponseFormat, convert_from_response_format,
};
use crate::routes::openai::utils::types::{
    FunctionChoice, ReasoningEffort, ToolChoiceValue, ToolReference,
};
use crate::utils::{
    ResolvedRequestContext, convert_logit_bias, merge_ec_transfer_params, merge_kv_transfer_params,
    resolve_session_id,
};

/// Response payload metadata echoed back on the final response object and
/// carried by lifecycle streaming events.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResponseMeta {
    pub model: String,
    pub instructions: Option<String>,
    pub metadata: Option<std::collections::HashMap<String, String>>,
    pub tools: Vec<ResponsesTool>,
    pub tool_choice: ResponseToolChoice,
    pub parallel_tool_calls: bool,
    pub max_output_tokens: Option<u32>,
    pub max_tool_calls: Option<u32>,
    pub previous_response_id: Option<String>,
    pub reasoning: Option<ResponsesReasoning>,
    pub service_tier: ServiceTier,
    pub temperature: f32,
    pub top_p: f32,
    pub top_logprobs: Option<i32>,
    pub text: Option<ResponseTextConfig>,
    pub truncation: ResponseTruncation,
    pub user: Option<String>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub include_reasoning: bool,
}

/// Lowered responses request plus the response metadata carried by every SSE
/// lifecycle event.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedResponsesRequest {
    pub request_id: String,
    pub meta: ResponseMeta,
    pub chat_request: ChatRequest,
}

/// Validate and lower one Responses API request into the internal chat
/// format.
pub(crate) fn prepare_responses_request(
    request: ResponsesRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
) -> Result<PreparedResponsesRequest, ApiError> {
    validate::validate_request_compat(&request)?;
    lora_resolution.model_names.check(request.model.as_deref())?;

    let request_id = format!("resp_{}", ctx.request_id);

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
        previous_input_messages: _,
        prompt: _,
        reasoning,
        include_reasoning,
        service_tier,
        background: _,
        stream,
        temperature,
        watermarking,
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

    let response_model = lora_resolution.response_model();

    let (messages, continue_final) = convert_input(&instructions, input)?;
    let generation_prompt_mode = if continue_final {
        GenerationPromptMode::ContinueFinalAssistant
    } else {
        GenerationPromptMode::StartNewAssistant
    };

    let converted_tools = tools
        .iter()
        .map(|ResponsesTool::Function(function)| ChatTool::from(function.clone()))
        .collect::<Vec<_>>();
    let requested_tool_choice = normalize_tool_choice(tool_choice, converted_tools.is_empty())?;
    let tool_context = ResolvedToolContext::new(
        &messages,
        converted_tools,
        requested_tool_choice,
        parallel_tool_calls.unwrap_or(true),
    )
    .map_err(|error| {
        let param = match &error {
            ChatError::ToolChoiceRequiresTools | ChatError::ToolChoiceFunctionNotFound { .. } => {
                "tool_choice"
            }
            _ => "tools",
        };
        ApiError::invalid_request(
            format!("failed to resolve request tools: {}", error.as_report()),
            Some(param),
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
    let truncation = truncation.unwrap_or_default();
    let prompt_truncation = response_prompt_truncation(truncation)?;

    let chat_request = ChatRequest {
        request_id: request_id.clone(),
        messages,
        sampling_params: SamplingParams {
            temperature,
            watermarking,
            top_p,
            top_k,
            seed,
            max_tokens: max_output_tokens,
            min_tokens,
            thinking_token_budget: None,
            logprobs: None,
            prompt_logprobs: None,
            prompt_logprob_token_ids: None,
            prompt_logprob_start: None,
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
            ..SamplingParams::default()
        },
        chat_options: ChatOptions {
            generation_prompt_mode,
            chat_template: None,
            reasoning_effort: reasoning_effort.map(|effort| effort.as_str().into()),
            response_format: response_format_value,
            template_kwargs,
        },
        tool_context,
        decode_options: TextDecodeOptions {
            skip_special_tokens,
            include_stop_str_in_output,
            stop_strings: stop.map(|stop| stop.into_vec()),
            min_tokens: min_tokens.unwrap_or(0),
            detokenize: true,
        },
        intermediate: stream.unwrap_or(false),
        prompt_truncation,
        priority: ctx.priority.or(priority).unwrap_or(0),
        documents: None,
        cache_salt,
        add_special_tokens: false,
        data_parallel_rank: ctx.data_parallel_rank,
        session_id,
        lora_request: lora_resolution.lora_request.clone(),
        return_prompt_text: false,
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
        service_tier: service_tier.unwrap_or_default(),
        temperature: temperature.unwrap_or(1.0),
        top_p: top_p.unwrap_or(1.0),
        top_logprobs,
        text,
        truncation,
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

/// Map Responses API truncation onto the shared chat prompt policy.
///
/// `auto` retains the newest prompt tokens that fit after output-token
/// reservation, matching the Python Responses frontend.
fn response_prompt_truncation(
    truncation: ResponseTruncation,
) -> Result<Option<vllm_text::PromptTruncation>, ApiError> {
    if truncation == ResponseTruncation::Disabled {
        return Ok(None);
    }

    resolve_generation_prompt_truncation(Some(-1), None).map_err(|error| {
        ApiError::invalid_request(
            format!("invalid automatic prompt truncation: {error}"),
            Some("truncation"),
        )
    })
}

/// Resolve the requested tool choice, mirroring the Python
/// `check_tool_usage` validator: without tools, `auto` is normalized to
/// `none` and named function choices are errors; with tools, named choices
/// must exist (enforced by
/// [`ResolvedToolContext`]).
fn normalize_tool_choice(
    tool_choice: Option<ResponseToolChoice>,
    tools_empty: bool,
) -> Result<Option<ChatToolChoice>, ApiError> {
    let Some(tool_choice) = tool_choice else {
        return Ok(None);
    };
    Ok(Some(match tool_choice {
        ResponseToolChoice::Value(ToolChoiceValue::None) => ChatToolChoice::None,
        ResponseToolChoice::Value(ToolChoiceValue::Auto) if tools_empty => ChatToolChoice::None,
        ResponseToolChoice::Value(ToolChoiceValue::Auto) => ChatToolChoice::Auto,
        ResponseToolChoice::Value(ToolChoiceValue::Required) => ChatToolChoice::Required,
        ResponseToolChoice::Object(ResponseToolChoiceObject::Function(FunctionChoice { name })) => {
            if tools_empty {
                bail_invalid_request!(
                    param = "tool_choice",
                    "Tool choice 'function' not found in 'tools' parameter."
                );
            }
            ChatToolChoice::Function { name }
        }
        ResponseToolChoice::Object(ResponseToolChoiceObject::AllowedTools { tools, .. }) => {
            bail_invalid_request!(
                param = "tool_choice",
                "allowed_tools tool_choice is not supported yet: {}.",
                tools.iter().map(ToolReference::identifier).join(", ")
            );
        }
    }))
}

/// Echoed tool choice in the response, after resolution: without tools the
/// resolved choice is always `none` (Python parity).
fn echo_tool_choice(tool_context: &ResolvedToolContext) -> ResponseToolChoice {
    match &tool_context.tool_choice {
        ChatToolChoice::None => ResponseToolChoice::Value(ToolChoiceValue::None),
        ChatToolChoice::Auto => ResponseToolChoice::Value(ToolChoiceValue::Auto),
        ChatToolChoice::Required => ResponseToolChoice::Value(ToolChoiceValue::Required),
        ChatToolChoice::Function { name } => {
            ResponseToolChoice::Object(ResponseToolChoiceObject::Function(FunctionChoice {
                name: name.clone(),
            }))
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
    for item in items {
        match item {
            ResponseInputItem::Message(message) => match message.role {
                role @ (ResponseInputRole::System
                | ResponseInputRole::Developer
                | ResponseInputRole::User) => {
                    assistant.flush(&mut messages);
                    let content = convert_content_parts(message.content, true)?;
                    messages.push(match role {
                        ResponseInputRole::System => ChatMessage::system(content),
                        ResponseInputRole::Developer => ChatMessage::developer(content, None),
                        _ => ChatMessage::user(content),
                    });
                }
                ResponseInputRole::Assistant => {
                    let content = convert_content_parts(message.content, false)?;
                    let text = content.try_flatten_to_text().map_err(|_| {
                        ApiError::invalid_request(
                            "assistant input items only support text content".to_string(),
                            Some("input"),
                        )
                    })?;
                    assistant.push_text(&mut messages, text);
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
            ResponseInputItem::ItemReference {} => {
                bail_invalid_request!(
                    param = "input",
                    "item_reference input items require the Responses API store, which is \
                     not supported by this frontend."
                );
            }
        }
    }
    assistant.flush(&mut messages);
    if messages.is_empty() {
        bail_invalid_request!(param = "input", "input must contain at least one message");
    }
    if continue_final && !matches!(messages.last(), Some(ChatMessage::Assistant { .. })) {
        bail_invalid_request!(
            param = "input",
            "continuing a final assistant item requires non-empty assistant content"
        );
    }

    Ok((messages, continue_final))
}

/// Convert message content into internal chat content.
fn convert_content_parts(
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
                                "input_image parts are not supported in assistant messages."
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
                    ResponseInputContentPart::InputAudio { input_audio } => {
                        if !allow_multimodal {
                            bail_invalid_request!(
                                param = "input",
                                "input_audio parts are not supported in assistant messages."
                            );
                        }
                        let Some(format) = input_audio.format else {
                            bail_invalid_request!(
                                param = "input",
                                "input_audio parts require 'input_audio.format'."
                            );
                        };
                        converted.push(ChatContentPart::InputAudio {
                            data: input_audio.data,
                            format: Some(format),
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
    finish_reason: &vllm_chat::FinishReason,
) -> Vec<ResponseOutputItem> {
    let mut items: Vec<_> = message
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
                call_id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                status: Some(ResponseItemStatus::Completed),
            }),
        })
        .collect();
    if matches!(finish_reason, vllm_chat::FinishReason::Length)
        && let Some(last) = items.last_mut()
    {
        last.mark_incomplete();
    }
    items
}

/// Build the `usage` block of a completed response.
pub(crate) fn build_usage(usage: &vllm_chat::ChatTokenUsage) -> ResponseUsage {
    ResponseUsage {
        input_tokens: usage.prompt_token_count,
        input_tokens_details: InputTokensDetails {
            cached_tokens: usage.prompt_cache_usage.cached_token_count,
            input_tokens_per_turn: vec![],
            cached_tokens_per_turn: vec![],
        },
        output_tokens: usage.output_token_count,
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: usage.reasoning_tokens,
            tool_output_tokens: 0,
            output_tokens_per_turn: vec![],
            tool_output_tokens_per_turn: vec![],
        },
        total_tokens: usage.prompt_token_count + usage.output_token_count,
    }
}

/// Build the top-level response object for non-streaming responses and the
/// terminal streaming event.
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
        error: None,
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
        service_tier: meta.service_tier,
        temperature: meta.temperature,
        text: meta.text.clone(),
        tool_choice: meta.tool_choice.clone(),
        tools: meta.tools.clone(),
        top_p: meta.top_p,
        top_logprobs: meta.top_logprobs,
        truncation: meta.truncation,
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
fn should_continue_final_message(items: &[ResponseInputItem]) -> bool {
    let partial = |status: Option<ResponseItemStatus>| {
        matches!(
            status,
            Some(ResponseItemStatus::InProgress | ResponseItemStatus::Incomplete)
        )
    };
    match items.last() {
        Some(ResponseInputItem::Reasoning(reasoning)) => partial(reasoning.status),
        Some(ResponseInputItem::Message(message)) => {
            message.role == ResponseInputRole::Assistant && partial(message.status)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vllm_chat::ChatTokenUsage;
    use vllm_llm::TokenUsage;

    use super::*;

    #[test]
    fn auto_tool_choice_without_tools_normalizes_to_none() {
        let tool_choice = serde_json::from_value(json!("auto")).unwrap();

        assert_eq!(
            normalize_tool_choice(Some(tool_choice), true).unwrap(),
            Some(ChatToolChoice::None)
        );
    }

    #[test]
    fn responses_function_tool_defaults_parameters_to_null() {
        let ResponsesTool::Function(function) = serde_json::from_value(json!({
            "type": "function",
            "name": "lookup",
        }))
        .expect("deserialize function tool");

        assert_eq!(ChatTool::from(function).parameters, Value::Null);
    }

    #[test]
    fn usage_preserves_reasoning_and_cached_token_counts() {
        let usage = build_usage(&ChatTokenUsage {
            engine: TokenUsage {
                prompt_token_count: 5,
                output_token_count: 7,
                prompt_cache_usage: vllm_llm::PromptCacheUsage {
                    cached_token_count: 3,
                    cache_creation_token_count: 2,
                },
            },
            reasoning_tokens: 2,
        });

        assert_eq!(usage.input_tokens_details.cached_tokens, 3);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 2);
        assert_eq!(usage.total_tokens, 12);
    }

    #[test]
    fn auto_truncation_uses_the_shared_input_budget_policy() {
        assert!(response_prompt_truncation(ResponseTruncation::Disabled).unwrap().is_none());
        assert!(response_prompt_truncation(ResponseTruncation::Auto).unwrap().is_some());
    }

    #[test]
    fn input_audio_uses_the_nested_responses_wire_shape() {
        let content = serde_json::from_value(json!([
            {
                "type": "input_audio",
                "input_audio": {
                    "data": "dGVzdA==",
                    "format": "wav"
                }
            }
        ]))
        .expect("deserialize Responses audio content");

        let converted =
            convert_content_parts(content, true).expect("convert Responses audio content");
        assert_eq!(
            converted,
            ChatContent::Parts(vec![ChatContentPart::InputAudio {
                data: "dGVzdA==".to_string(),
                format: Some("wav".to_string()),
                uuid: None,
            }])
        );
    }
}
