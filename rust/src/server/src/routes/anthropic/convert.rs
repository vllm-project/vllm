// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use validator::Validate;
use vllm_chat::{
    AssistantContentBlock, ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest,
    EffortValue, FinishReason, GenerationPromptMode, ResolvedToolContext, SamplingParams,
};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_llm::TokenUsage;

use super::types::{
    AssistantContent, AssistantInputBlock, InputMessage, MessagesRequest, MessagesStopReason,
    MessagesUsage, OutputContentBlock, OutputEffort, TextBlock, TextContent, ThinkingConfig,
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

impl From<InputMessage> for ChatMessage {
    fn from(message: InputMessage) -> Self {
        match message {
            // The renderer handles inline instructions without moving them to the front.
            InputMessage::System { content } => Self::developer(content.into_content(), None),
            InputMessage::User { content } => Self::user(content.into_content()),
            InputMessage::Assistant { content } => match content {
                AssistantContent::Text(text) => Self::assistant_text(text),
                AssistantContent::Blocks(blocks) => Self::assistant_blocks(
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
                        })
                        .collect(),
                ),
            },
        }
    }
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
    if body.stream == Some(true) {
        return Err(invalid_request!("Streaming Messages is not supported yet"));
    }
    let mut reasoning_effort =
        body.output_config.and_then(|config| config.effort).map(EffortValue::from);
    let thinking_token_budget = match body.thinking {
        Some(ThinkingConfig::Disabled | ThinkingConfig::BetweenTools) => {
            // Like Python, between-tools thinking starts with thinking disabled.
            reasoning_effort = Some(EffortValue::from("none"));
            None
        }
        Some(ThinkingConfig::Enabled { budget_tokens, .. }) => {
            if budget_tokens < 1024 {
                return Err(invalid_request!(
                    param = "thinking.budget_tokens",
                    "thinking.budget_tokens must be at least 1024"
                ));
            }
            Some(budget_tokens)
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
    messages.extend(body.messages.into_iter().map(ChatMessage::from));
    let continuation = matches!(messages.last(), Some(ChatMessage::Assistant { content })
        if content.iter().any(|b| matches!(b, AssistantContentBlock::Text { text } if !text.is_empty())));
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
        tool_context: ResolvedToolContext::default(),
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
            AssistantContentBlock::ToolCall(_) => {
                Err(invalid_request!("Tool output is not supported yet"))
            }
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
