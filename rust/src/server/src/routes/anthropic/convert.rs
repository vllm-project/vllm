// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use validator::Validate;
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatMessage,
    ChatOptions, ChatRequest, ChatTool, ChatToolChoice, EffortValue, FinishReason,
    GenerationPromptMode, ResolvedToolContext, SamplingParams,
};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_llm::TokenUsage;

use super::types::{
    AnthropicToolChoice, AssistantContent, AssistantInputBlock, ImageSource, InputMessage,
    MessagesRequest, MessagesStopReason, MessagesUsage, OutputContentBlock, OutputEffort,
    TextBlock, TextContent, ThinkingConfig, ToolResultContent, UserContent, UserInputBlock,
};
use crate::error::{ApiError, chat_submit_error, invalid_request, server_error};
use crate::lora::LoraModelResolution;
use crate::routes::openai::utils::types::{Normalizable, StringOrArray, validate_stop};
use crate::utils::ResolvedRequestContext;

impl Normalizable for MessagesRequest {
    fn normalize(&mut self) {
        if let Some(system @ TextContent::Blocks(_)) = &mut self.system {
            system.strip_billing_headers();
        }
        self.messages.retain_mut(|message| match message {
            InputMessage::System { content } => content.strip_billing_headers(),
            _ => true,
        });
    }
}

/// Append one Anthropic input message as one or more chat messages: user
/// tool-result blocks each become their own tool-response message.
fn push_input_message(message: InputMessage, out: &mut Vec<ChatMessage>) {
    match message {
        // The renderer handles inline instructions without moving them to the front.
        InputMessage::System { content } => {
            out.push(ChatMessage::developer(text_content_into(content), None))
        }
        InputMessage::User { content } => {
            let blocks = match content {
                UserContent::Text(text) => {
                    out.push(ChatMessage::user(ChatContent::Text(text)));
                    return;
                }
                UserContent::Blocks(blocks) => blocks,
            };
            let mut parts: Vec<ChatContentPart> = Vec::new();
            for block in blocks {
                match block {
                    UserInputBlock::Text { text } => parts.push(ChatContentPart::Text { text }),
                    UserInputBlock::Image { source } => {
                        let image_url = match source {
                            ImageSource::Base64 { media_type, data } => {
                                format!("data:{media_type};base64,{data}")
                            }
                            ImageSource::Url { url } => url,
                        };
                        parts.push(ChatContentPart::ImageUrl {
                            image_url,
                            detail: None,
                            uuid: None,
                        });
                    }
                    UserInputBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        // Tool results may carry image blocks (e.g. Claude Code
                        // reading a PNG); images become multimodal parts on the
                        // tool-response message, text blocks stay text.
                        let mut tr_text: Vec<String> = Vec::new();
                        let mut tr_parts: Vec<ChatContentPart> = Vec::new();
                        match content {
                            None => {}
                            Some(ToolResultContent::Text(text)) => tr_text.push(text),
                            Some(ToolResultContent::Blocks(blocks)) => {
                                for block in blocks {
                                    match block {
                                        super::types::ToolResultBlock::Text { text } => {
                                            tr_text.push(text)
                                        }
                                        super::types::ToolResultBlock::Image { source } => {
                                            let image_url = match source {
                                                ImageSource::Base64 { media_type, data } => {
                                                    format!("data:{media_type};base64,{data}")
                                                }
                                                ImageSource::Url { url } => url,
                                            };
                                            tr_parts.push(ChatContentPart::ImageUrl {
                                                image_url,
                                                detail: None,
                                                uuid: None,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                        let content = if tr_parts.is_empty() {
                            ChatContent::Text(tr_text.join("\n"))
                        } else {
                            if !tr_text.is_empty() {
                                tr_parts.insert(
                                    0,
                                    ChatContentPart::Text {
                                        text: tr_text.join("\n"),
                                    },
                                );
                            }
                            ChatContent::Parts(tr_parts)
                        };
                        out.push(ChatMessage::tool_response(content, tool_use_id));
                    }
                }
            }
            if !parts.is_empty() {
                if parts.len() == 1 {
                    if let ChatContentPart::Text { text } = parts.pop().unwrap() {
                        out.push(ChatMessage::user(ChatContent::Text(text)));
                        return;
                    }
                    unreachable!()
                }
                out.push(ChatMessage::user(ChatContent::Parts(parts)));
            }
        }
        InputMessage::Assistant { content } => match content {
            AssistantContent::Text(text) => out.push(ChatMessage::assistant_text(text)),
            AssistantContent::Blocks(blocks) => out.push(ChatMessage::assistant_blocks(
                blocks
                    .into_iter()
                    .filter_map(|block| match block {
                        AssistantInputBlock::Text { text } => {
                            Some(AssistantContentBlock::Text { text })
                        }
                        AssistantInputBlock::Thinking { thinking, .. } => {
                            Some(AssistantContentBlock::Reasoning { text: thinking })
                        }
                        AssistantInputBlock::RedactedThinking { .. } => None,
                        AssistantInputBlock::ToolUse { id, name, input } => {
                            Some(AssistantContentBlock::ToolCall(AssistantToolCall {
                                id,
                                name,
                                arguments: input.to_string(),
                            }))
                        }
                    })
                    .collect(),
            )),
        },
    }
}

/// Shared text-content conversion for system-style payloads.
fn text_content_into(content: TextContent) -> ChatContent {
    content.into_content()
}

impl From<OutputEffort> for EffortValue {
    fn from(effort: OutputEffort) -> Self {
        Self::from(match effort {
            OutputEffort::Low => "low",
            OutputEffort::Medium => "medium",
            OutputEffort::High => "high",
            OutputEffort::XHigh => "xhigh",
            OutputEffort::Max => "max",
        })
    }
}

impl TextContent {
    /// Filter system text blocks and return whether to keep the content.
    fn strip_billing_headers(&mut self) -> bool {
        fn keep_system_text(text: &str) -> bool {
            !text.is_empty() && !text.starts_with("x-anthropic-billing-header")
        }

        match self {
            Self::Text(text) => keep_system_text(text),
            Self::Blocks(blocks) => {
                blocks.retain(|TextBlock::Text { text }| keep_system_text(text));
                !blocks.is_empty()
            }
        }
    }

    fn into_content(self) -> ChatContent {
        match self {
            Self::Text(text) => ChatContent::Text(text),
            Self::Blocks(blocks) => {
                let mut parts: Vec<_> = blocks
                    .into_iter()
                    .map(|TextBlock::Text { text }| ChatContentPart::Text { text })
                    .collect();
                if parts.len() == 1 {
                    let ChatContentPart::Text { text } = parts.pop().unwrap() else {
                        unreachable!()
                    };
                    ChatContent::Text(text)
                } else {
                    ChatContent::Parts(parts)
                }
            }
        }
    }
}

pub(super) fn prepare_messages_request(
    mut body: MessagesRequest,
    model: &LoraModelResolution,
    ctx: ResolvedRequestContext,
) -> Result<ChatRequest, ApiError> {
    body.normalize();
    body.validate().map_err(|error| invalid_request!("{error}"))?;
    if body.model.is_empty() || body.messages.is_empty() {
        return Err(invalid_request!("model and messages must be non-empty"));
    }
    model.model_names.check(Some(&body.model))?;
    let mut reasoning_effort =
        body.output_config.and_then(|config| config.effort).map(EffortValue::from);
    let thinking_token_budget = match body.thinking {
        Some(ThinkingConfig::Disabled | ThinkingConfig::BetweenTools) => {
            // Like Python, between-tools thinking starts with thinking disabled.
            reasoning_effort = Some(EffortValue::from("none"));
            None
        }
        Some(ThinkingConfig::Enabled { budget_tokens, .. }) => {
            // Absent budget means "engine default": skip the budget entirely.
            let budget = budget_tokens.unwrap_or(0);
            if budget != 0 && budget < 1024 {
                return Err(invalid_request!(
                    param = "thinking.budget_tokens",
                    "thinking.budget_tokens must be at least 1024"
                ));
            }
            (budget != 0).then_some(budget)
        }
        Some(ThinkingConfig::Adaptive { .. }) | None => None,
    };
    let stop_strings = body
        .stop_sequences
        .map(|stops| {
            let stops = StringOrArray::Array(stops);
            validate_stop(&stops).map_err(|error| invalid_request!("{error}"))?;
            Ok::<_, ApiError>(stops.into_vec())
        })
        .transpose()?;
    let mut messages = Vec::new();
    if let Some(system) = body.system {
        let text = match system {
            TextContent::Text(text) => text,
            TextContent::Blocks(blocks) => {
                blocks.into_iter().map(|TextBlock::Text { text }| text).collect()
            }
        };
        if !text.is_empty() {
            messages.push(ChatMessage::system(text));
        }
    }
    for message in body.messages {
        push_input_message(message, &mut messages);
    }
    let continuation = matches!(messages.last(), Some(ChatMessage::Assistant { content })
        if content.iter().any(|b| matches!(b, AssistantContentBlock::Text { text } if !text.is_empty())));
    let chat_tools: Vec<ChatTool> = body
        .tools
        .unwrap_or_default()
        .into_iter()
        .map(|tool| ChatTool {
            name: tool.name,
            description: tool.description,
            parameters: tool.input_schema,
            strict: None,
            defer_loading: None,
        })
        .collect();
    let tool_choice = match (&body.tool_choice, chat_tools.is_empty()) {
        (Some(AnthropicToolChoice::Auto { .. }), false) => Some(ChatToolChoice::Auto),
        (Some(AnthropicToolChoice::Any { .. }), false) => Some(ChatToolChoice::Required),
        (Some(AnthropicToolChoice::Tool { name, .. }), false) => {
            Some(ChatToolChoice::Function { name: name.clone() })
        }
        (None, false) => Some(ChatToolChoice::Auto),
        _ => None,
    };
    let parallel_tool_calls = !matches!(
        &body.tool_choice,
        Some(AnthropicToolChoice::Auto {
            disable_parallel_tool_use: Some(true)
        }) | Some(AnthropicToolChoice::Any {
            disable_parallel_tool_use: Some(true)
        }) | Some(AnthropicToolChoice::Tool {
            disable_parallel_tool_use: Some(true),
            ..
        })
    );
    let tool_context =
        ResolvedToolContext::new(&messages, chat_tools, tool_choice, parallel_tool_calls)
            .map_err(|error| chat_submit_error("invalid Messages tool context", error))?;
    let request = ChatRequest {
        request_id: format!("msg_{}", ctx.request_id),
        messages,
        sampling_params: SamplingParams {
            max_tokens: body.max_tokens,
            temperature: body.temperature,
            top_p: body.top_p,
            top_k: body.top_k,
            thinking_token_budget,
            ..Default::default()
        },
        chat_options: ChatOptions {
            reasoning_effort,
            generation_prompt_mode: if continuation {
                GenerationPromptMode::ContinueFinalAssistant
            } else {
                GenerationPromptMode::StartNewAssistant
            },
            ..Default::default()
        },
        tool_context,
        decode_options: vllm_text::TextDecodeOptions {
            stop_strings,
            ..Default::default()
        },
        intermediate: false,
        prompt_truncation: None,
        priority: ctx.priority.unwrap_or(0),
        documents: None,
        cache_salt: None,
        add_special_tokens: false,
        data_parallel_rank: ctx.data_parallel_rank,
        session_id: ctx.session_id,
        lora_request: model.lora_request.clone(),
        return_prompt_text: false,
    };
    request
        .validate()
        .map_err(|error| chat_submit_error("invalid Messages request", error))?;
    Ok(request)
}

pub(super) fn convert_output_blocks(
    blocks: Vec<AssistantContentBlock>,
) -> Result<Vec<OutputContentBlock>, ApiError> {
    let mut content = blocks
        .into_iter()
        .map(|block| match block {
            AssistantContentBlock::Text { text } => Ok(OutputContentBlock::Text { text }),
            AssistantContentBlock::Reasoning { text } => Ok(OutputContentBlock::Thinking {
                thinking: text,
                signature: uuid::Uuid::new_v4().simple().to_string(),
            }),
            AssistantContentBlock::ToolCall(call) => Ok(OutputContentBlock::ToolUse {
                id: call.id,
                name: call.name,
                input: serde_json::from_str(&call.arguments)
                    .unwrap_or(serde_json::Value::String(call.arguments)),
            }),
        })
        .collect::<Result<Vec<_>, ApiError>>()?;
    if content.is_empty() {
        content.push(OutputContentBlock::Text {
            text: String::new(),
        });
    }
    Ok(content)
}

pub(super) fn stop_reason(
    finish: &FinishReason,
) -> Result<(MessagesStopReason, Option<&str>), ApiError> {
    match finish {
        FinishReason::Error(error) => Err(crate::error::engine_error(error.as_deref())),
        FinishReason::Length => Ok((MessagesStopReason::MaxTokens, None)),
        FinishReason::Stop(Some(StopReason::Text(text))) => {
            Ok((MessagesStopReason::StopSequence, Some(text)))
        }
        FinishReason::Stop(_) => Ok((MessagesStopReason::EndTurn, None)),
        _ => Err(server_error!("Generation did not complete successfully")),
    }
}

pub(super) fn usage(usage: TokenUsage, include_cache: bool) -> MessagesUsage {
    let cached = usage.prompt_cache_usage.cached_token_count;
    let created = usage.prompt_cache_usage.cache_creation_token_count;
    MessagesUsage {
        input_tokens: if include_cache {
            usage.prompt_token_count.saturating_sub(cached).saturating_sub(created)
        } else {
            usage.prompt_token_count
        },
        output_tokens: usage.output_token_count,
        cache_read_input_tokens: include_cache.then_some(cached),
        cache_creation_input_tokens: include_cache.then_some(created),
    }
}
