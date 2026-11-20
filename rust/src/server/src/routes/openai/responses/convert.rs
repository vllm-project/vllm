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
use thiserror_ext::AsReport as _;
use tracing::warn;
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatMessage,
    ChatOptions, ChatRequest, ChatToolChoice, Error as ChatError, GenerationPromptMode,
    ResolvedToolContext,
};
use vllm_text::output::TextDecodeOptions;
use vllm_text::{EffectiveSampling, SamplingParams};

use super::tools::{ResponsesToolNames, ToolIdentity};
use super::types::{
    InputTokensDetails, OutputTokensDetails, ResponseEchoFields, ResponseInputContentPart,
    ResponseInputItem, ResponseInputReasoning, ResponseInputRole, ResponseItemStatus,
    ResponseMessageContent, ResponseObject, ResponseTextConfig, ResponseTextFormat,
    ResponseToolChoice, ResponseToolChoiceObject, ResponseTruncation, ResponseUsage,
    ResponsesInput, ResponsesReasoning, ResponsesRequest, ResponsesResponse, ResponsesTool,
    ServiceTier,
};
use super::validate;
use crate::error::{ApiError, bail_invalid_request};
use crate::lora::LoraModelResolution;
use crate::routes::openai::utils::resolve_generation_prompt_truncation;
use crate::routes::openai::utils::structured_outputs::{
    JsonSchemaFormat, ResponseFormat, convert_from_response_format,
};
use crate::routes::openai::utils::types::{FunctionChoice, ToolChoiceValue, ToolReference};
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
    /// Request fields echoed back unchanged.
    pub echo: ResponseEchoFields,
    pub tools: Vec<ResponsesTool>,
    pub tool_choice: ResponseToolChoice,
    pub parallel_tool_calls: bool,
    pub previous_response_id: Option<String>,
    pub reasoning: Option<ResponsesReasoning>,
    pub service_tier: ServiceTier,
    pub top_logprobs: Option<i32>,
    pub text: Option<ResponseTextConfig>,
    pub truncation: ResponseTruncation,
}

/// Lowered responses request plus the response metadata carried by every SSE
/// lifecycle event.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedResponsesRequest {
    pub request_id: String,
    pub meta: ResponseMeta,
    /// Whether reasoning items appear in the output.
    pub include_reasoning: bool,
    /// Maps called tool names back to namespaced functions in the output.
    pub tool_names: ResponsesToolNames,
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
        echo,
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
        min_p,
        top_logprobs,
        text,
        truncation,
        include: _,
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
        seed,
        stop,
        ignore_eos,
        skip_special_tokens,
        spaces_between_special_tokens: _,
        include_stop_str_in_output,
        min_tokens,
        logit_bias,
        stop_token_ids,
        thinking_token_budget,
        request_id: _,
        session_id,
        priority,
        cache_salt,
        chat_template_kwargs,
        media_io_kwargs: _,
        mm_processor_kwargs: _,
        enable_response_messages: _,
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

    let (converted_tools, tool_names) = ResponsesToolNames::convert(&tools);
    let requested_tool_choice =
        normalize_tool_choice(tool_choice, &tools, &tool_names, converted_tools.is_empty())?;
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
            thinking_token_budget,
            logprobs: None,
            prompt_logprobs: None,
            prompt_logprob_token_ids: None,
            prompt_logprob_start: None,
            min_p,
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
            template_kwargs: chat_template_kwargs.unwrap_or_default(),
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
        echo,
        tools,
        tool_choice: tool_choice_echo,
        parallel_tool_calls: parallel_tool_calls.unwrap_or(true),
        previous_response_id,
        reasoning,
        service_tier: service_tier.unwrap_or_default(),
        top_logprobs,
        text,
        truncation,
    };

    Ok(PreparedResponsesRequest {
        request_id,
        meta,
        include_reasoning,
        tool_names,
        chat_request,
    })
}

/// Map Responses API truncation onto the shared chat prompt policy.
///
/// `auto` keeps the newest `max_model_len - 1` prompt tokens, as
/// `truncate_prompt_tokens=-1` does in the Python Responses frontend; the
/// output-token limit is then clamped to the remaining context.
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
/// must exist and resolve to the name the model sees (see
/// [`ResponsesToolNames::resolve_choice`]).
fn normalize_tool_choice(
    tool_choice: Option<ResponseToolChoice>,
    tools: &[ResponsesTool],
    tool_names: &ResponsesToolNames,
    tools_empty: bool,
) -> Result<Option<ChatToolChoice>, ApiError> {
    let Some(tool_choice) = tool_choice else {
        return Ok(None);
    };
    Ok(Some(match tool_choice {
        ResponseToolChoice::Value(ToolChoiceValue::None) => ChatToolChoice::None,
        ResponseToolChoice::Value(ToolChoiceValue::Auto) => ChatToolChoice::Auto,
        ResponseToolChoice::Value(ToolChoiceValue::Required) => ChatToolChoice::Required,
        ResponseToolChoice::Object(ResponseToolChoiceObject::Function(FunctionChoice { name })) => {
            if tools_empty {
                bail_invalid_request!(
                    param = "tool_choice",
                    "Tool choice 'function' not found in 'tools' parameter."
                );
            }
            ChatToolChoice::Function {
                name: tool_names.resolve_choice(name, tools)?,
            }
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
                    let content = convert_content_parts(message.content)?;
                    messages.push(match role {
                        ResponseInputRole::System => ChatMessage::system(content),
                        ResponseInputRole::Developer => ChatMessage::developer(content, None),
                        _ => ChatMessage::user(content),
                    });
                }
                ResponseInputRole::Assistant => {
                    assistant.push_text(&mut messages, assistant_text_blocks(message.content)?);
                }
            },
            ResponseInputItem::FunctionCall(call) => {
                let name = match &call.namespace {
                    Some(namespace) => ToolIdentity {
                        namespace: Some(namespace),
                        name: &call.name,
                    }
                    .flat_name()
                    .into_owned(),
                    None => call.name,
                };
                assistant.push_block(AssistantContentBlock::ToolCall(AssistantToolCall {
                    id: call.call_id,
                    name,
                    arguments: call.arguments,
                }));
            }
            ResponseInputItem::FunctionCallOutput(output) => {
                assistant.flush(&mut messages);
                messages.push(ChatMessage::tool_response(
                    output.output.into_text_content()?,
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
fn convert_content_parts(content: ResponseMessageContent) -> Result<ChatContent, ApiError> {
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
    /// Push the text blocks of one assistant message item.
    fn push_text(&mut self, messages: &mut Vec<ChatMessage>, blocks: Vec<AssistantContentBlock>) {
        if self
            .blocks
            .iter()
            .any(|block| matches!(block, AssistantContentBlock::Text { .. }))
        {
            self.flush(messages);
        }
        self.blocks.extend(blocks);
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

/// Lower the content of an assistant message item into one text block per
/// part, as chat completions does, so the renderer joins them. Python's
/// converter keeps only the first part.
fn assistant_text_blocks(
    content: ResponseMessageContent,
) -> Result<Vec<AssistantContentBlock>, ApiError> {
    let texts = match content {
        ResponseMessageContent::Text(text) => vec![text],
        ResponseMessageContent::Parts(parts) => parts
            .into_iter()
            .map(|part| -> Result<String, ApiError> {
                match part {
                    ResponseInputContentPart::InputText { text }
                    | ResponseInputContentPart::OutputText { text, .. }
                    // Refusals are answer text; keep them so templates still see the turn.
                    | ResponseInputContentPart::Refusal { refusal: text } => Ok(text),
                    other => bail_invalid_request!(
                        param = "input",
                        "assistant input items only support text content; got part type '{}'",
                        part_type_name(&other)
                    ),
                }
            })
            .try_collect()?,
    };
    // An item with no parts still makes an assistant turn, as before.
    if texts.is_empty() {
        return Ok(vec![AssistantContentBlock::Text {
            text: String::new(),
        }]);
    }
    Ok(texts.into_iter().map(|text| AssistantContentBlock::Text { text }).collect())
}

impl ResponseMessageContent {
    /// Lower `function_call_output` content, keeping its text parts so the
    /// renderer joins them as for a chat tool message; non-text parts are
    /// rejected (tool outputs support text only in this frontend).
    fn into_text_content(self) -> Result<ChatContent, ApiError> {
        match self {
            Self::Text(text) => Ok(ChatContent::Text(text)),
            Self::Parts(parts) => parts
                .into_iter()
                .map(|part| -> Result<ChatContentPart, ApiError> {
                    match part {
                        ResponseInputContentPart::InputText { text }
                        | ResponseInputContentPart::OutputText { text, .. } => {
                            Ok(ChatContentPart::text(text))
                        }
                        other => bail_invalid_request!(
                            param = "input",
                            "function_call_output only supports text content parts; got part type \
                         '{}'",
                            part_type_name(&other)
                        ),
                    }
                })
                .try_collect()
                .map(ChatContent::Parts),
        }
    }
}

/// Build the `usage` block of a completed response.
pub(crate) fn build_usage(usage: &vllm_chat::ChatTokenUsage) -> ResponseUsage {
    ResponseUsage {
        input_tokens: usage.prompt_token_count,
        input_tokens_details: InputTokensDetails {
            cached_tokens: usage.prompt_cache_usage.cached_token_count,
            cache_write_tokens: usage.prompt_cache_usage.cache_creation_token_count,
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

/// Build the in-progress response object of one request. The lifecycle
/// events and the final response all start from it.
pub(crate) fn build_response(
    ResponseMeta {
        model,
        instructions,
        echo,
        tools,
        tool_choice,
        parallel_tool_calls,
        previous_response_id,
        reasoning,
        service_tier,
        top_logprobs,
        text,
        truncation,
    }: ResponseMeta,
    sampling: EffectiveSampling,
    request_id: String,
    created_at: u64,
) -> ResponsesResponse {
    ResponsesResponse {
        id: request_id,
        object: ResponseObject,
        created_at,
        status: ResponseItemStatus::InProgress,
        error: None,
        background: false,
        incomplete_details: None,
        instructions,
        max_output_tokens: sampling.max_tokens,
        echo,
        model,
        output: vec![],
        parallel_tool_calls,
        previous_response_id,
        prompt: None,
        reasoning,
        service_tier,
        temperature: sampling.temperature,
        text,
        tool_choice,
        tools,
        top_p: sampling.top_p,
        top_logprobs,
        truncation,
        usage: None,
        metrics: None,
        presence_penalty: sampling.presence_penalty,
        frequency_penalty: sampling.frequency_penalty,
        kv_transfer_params: None,
        ec_transfer_params: None,
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
    use serde_json::{Value, json};
    use vllm_chat::{ChatTokenUsage, ChatTool};
    use vllm_llm::TokenUsage;

    use super::*;
    use crate::model_names::ServedModelNames;

    #[test]
    fn responses_function_tool_defaults_parameters_to_null() {
        let Ok(ResponsesTool::Function(function)) = serde_json::from_value(json!({
            "type": "function",
            "name": "lookup",
        })) else {
            panic!("deserialize function tool");
        };

        assert_eq!(ChatTool::from(function).parameters, Value::Null);
    }

    #[test]
    fn namespace_tools_reach_the_model_by_their_flat_names() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "input": [
                {"role": "user", "content": "Who is Ada?"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_contact",
                    "namespace": "crm",
                    "arguments": "{\"name\":\"Ada\"}"
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "found"}
            ],
            "tools": [{
                "type": "namespace",
                "name": "crm",
                "description": "CRM tools.",
                "tools": [{"type": "function", "name": "get_contact"}]
            }],
            "tool_choice": {"type": "function", "name": "get_contact"}
        }))
        .expect("deserialize request");
        let served = LoraModelResolution {
            model_names: ServedModelNames::new(vec!["test-model".to_string()]),
            lora_request: None,
        };

        let prepared = prepare_responses_request(request, &served, Default::default())
            .expect("prepare request");

        let tool_context = &prepared.chat_request.tool_context;
        let names = tool_context.initial_tools.iter().map(|tool| tool.name.as_str());
        assert_eq!(names.collect::<Vec<_>>(), ["crm__get_contact"]);
        let flat_choice = ChatToolChoice::Function {
            name: "crm__get_contact".to_string(),
        };
        assert_eq!(tool_context.tool_choice, flat_choice);
        assert_eq!(
            prepared.meta.tool_choice,
            ResponseToolChoice::Object(ResponseToolChoiceObject::Function(FunctionChoice {
                name: "crm__get_contact".to_string(),
            }))
        );
        let ChatMessage::Assistant { content } = &prepared.chat_request.messages[1] else {
            panic!("assistant turn: {:?}", prepared.chat_request.messages);
        };
        assert!(
            matches!(
                content.as_slice(),
                [AssistantContentBlock::ToolCall(call)] if call.name == "crm__get_contact"
            ),
            "{content:?}"
        );
    }

    #[test]
    fn usage_preserves_reasoning_and_prompt_cache_token_counts() {
        let usage = build_usage(&ChatTokenUsage {
            engine: TokenUsage {
                prompt_token_count: 5,
                output_token_count: 7,
                prompt_cache_usage: vllm_llm::PromptCacheUsage {
                    cached_token_count: 3,
                    cache_creation_token_count: 4,
                },
            },
            reasoning_tokens: 2,
        });

        assert_eq!(usage.input_tokens_details.cached_tokens, 3);
        assert_eq!(usage.input_tokens_details.cache_write_tokens, 4);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 2);
        assert_eq!(usage.total_tokens, 12);
    }

    #[test]
    fn auto_truncation_uses_the_shared_input_budget_policy() {
        assert!(response_prompt_truncation(ResponseTruncation::Disabled).unwrap().is_none());
        assert_eq!(
            response_prompt_truncation(ResponseTruncation::Auto).unwrap(),
            Some(vllm_text::PromptTruncation {
                limit: vllm_text::PromptTruncationLimit::InputBudget,
                side: vllm_text::TruncationSide::Left,
            })
        );
    }

    #[test]
    fn text_parts_reach_the_renderer_as_parts() {
        let input = serde_json::from_value(json!([
            {"role": "user", "content": "What is the weather?"},
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Let me check."},
                    {"type": "output_text", "text": "One moment."}
                ]
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    {"type": "input_text", "text": "Sunny"},
                    {"type": "input_text", "text": "25C"}
                ]
            }
        ]))
        .expect("deserialize Responses input");

        let (messages, _) = convert_input(&None, input).expect("convert Responses input");
        assert_eq!(
            messages[1..],
            [
                ChatMessage::assistant_blocks(vec![
                    AssistantContentBlock::Text {
                        text: "Let me check.".to_string()
                    },
                    AssistantContentBlock::Text {
                        text: "One moment.".to_string()
                    },
                    AssistantContentBlock::ToolCall(AssistantToolCall {
                        id: "call_1".to_string(),
                        name: "get_weather".to_string(),
                        arguments: "{}".to_string(),
                    }),
                ]),
                ChatMessage::tool_response(
                    ChatContent::Parts(vec![
                        ChatContentPart::text("Sunny"),
                        ChatContentPart::text("25C"),
                    ]),
                    "call_1",
                ),
            ]
        );
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

        let converted = convert_content_parts(content).expect("convert Responses audio content");
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
