// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::Deserialize;
use validator::Validate;

use crate::config::{ApiServerOptions, OutputLogConfig};
use crate::error::{ApiError, chat_submit_error, invalid_request, server_error, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::routes::openai::utils::types::{
    Normalizable, StringOrArray, validate_stop, validate_top_p_value,
};
use crate::routes::openai::utils::validate_max_output_tokens;
use crate::routes::output_log::LoggedOutput;
use crate::utils::ResolvedRequestContext;
use tracing_futures::Instrument as _;
use vllm_chat::{
    AssistantContentBlock, AssistantMessageExt as _, ChatContent, ChatContentPart, ChatLlm,
    ChatMessage, ChatOptions, ChatRequest, FinishReason, GenerationPromptMode, ResolvedToolContext,
    SamplingParams,
};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_llm::TokenUsage;

#[derive(Debug, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub(crate) struct MessagesRequest {
    pub(crate) model: String,
    messages: Vec<Message>,
    system: Option<TextContent>,
    #[validate(range(min = 1))]
    max_tokens: Option<u32>,
    #[validate(range(min = 0.0, max = 1.0))]
    temperature: Option<f32>,
    #[validate(custom(function = "validate_top_p_value"))]
    top_p: Option<f32>,
    top_k: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    stream: Option<bool>,
}

impl Normalizable for MessagesRequest {
    fn normalize(&mut self) {
        if let Some(TextContent::Blocks(blocks)) = &mut self.system {
            blocks
                .retain(|TextBlock::Text { text }| !text.starts_with("x-anthropic-billing-header"));
        }
    }
}

#[derive(Debug, Deserialize, serde::Serialize)]
#[serde(tag = "role", rename_all = "lowercase", deny_unknown_fields)]
enum Message {
    User { content: TextContent },
    Assistant { content: TextContent },
}

impl From<Message> for ChatMessage {
    fn from(message: Message) -> Self {
        match message {
            Message::User { content } => Self::user(content.into_content()),
            Message::Assistant { content } => match content {
                TextContent::Text(text) => Self::assistant_text(text),
                TextContent::Blocks(blocks) => Self::assistant_blocks(
                    blocks
                        .into_iter()
                        .map(|TextBlock::Text { text }| AssistantContentBlock::Text { text })
                        .collect(),
                ),
            },
        }
    }
}

#[derive(Debug, Deserialize, serde::Serialize)]
#[serde(untagged)]
enum TextContent {
    Text(String),
    Blocks(Vec<TextBlock>),
}

#[derive(Debug, Deserialize, serde::Serialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum TextBlock {
    Text { text: String },
}

impl TextContent {
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

impl MessagesRequest {
    fn into_chat_request(
        mut self,
        model: &LoraModelResolution,
        ctx: ResolvedRequestContext,
    ) -> Result<ChatRequest, ApiError> {
        self.normalize();
        self.validate().map_err(|error| invalid_request!("{error}"))?;
        if self.model.is_empty() || self.messages.is_empty() {
            return Err(invalid_request!("model and messages must be non-empty"));
        }
        model.model_names.check(Some(&self.model))?;
        if self.stream == Some(true) {
            return Err(invalid_request!("Streaming Messages is not supported yet"));
        }
        let stop_strings = self
            .stop_sequences
            .map(|stops| {
                let stops = StringOrArray::Array(stops);
                validate_stop(&stops).map_err(|error| invalid_request!("{error}"))?;
                Ok::<_, ApiError>(stops.into_vec())
            })
            .transpose()?;
        let mut messages = Vec::new();
        if let Some(system) = self.system {
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
        messages.extend(self.messages.into_iter().map(ChatMessage::from));
        let continuation = matches!(messages.last(), Some(ChatMessage::Assistant { content })
            if content.iter().any(|b| matches!(b, AssistantContentBlock::Text { text } if !text.is_empty())));
        let request = ChatRequest {
            request_id: format!("msg_{}", ctx.request_id),
            messages,
            sampling_params: SamplingParams {
                max_tokens: self.max_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                ..Default::default()
            },
            chat_options: ChatOptions {
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
}

#[derive(serde::Serialize)]
pub(crate) struct CountTokensResponse {
    input_tokens: usize,
    context_management: ContextManagement,
}
#[derive(serde::Serialize)]
struct ContextManagement {
    original_input_tokens: usize,
}

/// Anthropic request handling over the application's shared chat inference.
pub(crate) struct MessagesLlm<'a> {
    chat: &'a ChatLlm,
    enable_log_requests: bool,
    log_outputs: OutputLogConfig,
    enable_prompt_tokens_details: bool,
}

impl<'a> MessagesLlm<'a> {
    pub(crate) fn new(chat: &'a ChatLlm, options: ApiServerOptions) -> Self {
        Self {
            chat,
            enable_log_requests: options.enable_log_requests,
            log_outputs: options.log_outputs,
            enable_prompt_tokens_details: options.enable_prompt_tokens_details,
        }
    }
    pub(crate) async fn count_tokens(
        &self,
        body: MessagesRequest,
        model: &LoraModelResolution,
        context: ResolvedRequestContext,
    ) -> Result<CountTokensResponse, ApiError> {
        let request = body.into_chat_request(model, context)?;
        let text = self
            .chat
            .request_processor()
            .prepare_for_tokenization(request)
            .await
            .map_err(|error| chat_submit_error("count_tokens failed", error))?;
        let tokens = self
            .chat
            .text()
            .request_processor()
            .tokenize(text)
            .map_err(|error| text_submit_error("count_tokens failed", error))?;
        Ok(CountTokensResponse {
            input_tokens: tokens.len(),
            context_management: ContextManagement {
                original_input_tokens: tokens.len(),
            },
        })
    }
    pub(crate) async fn create(
        &self,
        body: MessagesRequest,
        model: &LoraModelResolution,
        context: ResolvedRequestContext,
    ) -> Result<MessagesResponse, ApiError> {
        if body.max_tokens.is_none() {
            return Err(invalid_request!(
                param = "max_tokens",
                "max_tokens is required for generation"
            ));
        }
        let request = body.into_chat_request(model, context)?;
        validate_max_output_tokens(
            "max_tokens",
            request.sampling_params.max_tokens,
            self.chat.text().llm().info().max_model_len,
        )?;
        let request_id = request.request_id.clone();
        let span = tracing::info_span!("messages", request_id = %request_id, engine_request_id = tracing::field::Empty);
        let response = async {
        let response = self.chat.chat(request).await
            .map_err(|error| chat_submit_error("failed to submit Messages request", error))?
            .collect_message().await
            .map_err(|error| chat_submit_error("failed to collect Messages response", error))?;
        let (reason, sequence) = stop_reason(&response.finish_reason)?;
        if response.message.has_reasoning() || response.message.has_tool_calls() {
            return Err(invalid_request!("Reasoning and tool output are not supported yet"));
        }
        let text = response.message.text();
        if self.enable_log_requests {
            tracing::info!(model = %model.response_model(), prompt_tokens = response.usage.prompt_token_count,
                output_tokens = response.usage.output_token_count, finish_reason = reason, "Messages finished");
        }
        if self.log_outputs.logs_complete() && !text.is_empty() {
            self.log_outputs.log(LoggedOutput::Response, &text, Some(&response.token_ids), Some(reason));
        }
        Ok(MessagesResponse {
            id: request_id, message_type: "message", message: Message::Assistant { content: TextContent::Blocks(vec![TextBlock::Text { text }]) }, model: model.response_model(),
            stop_reason: reason, stop_sequence: sequence.map(str::to_owned),
            usage: usage(response.usage.engine, self.enable_prompt_tokens_details),
            kv_transfer_params: response.kv_transfer_params, ec_transfer_params: response.ec_transfer_params,
        })
    }.instrument(span).await?;
        Ok(response)
    }
}

#[derive(serde::Serialize)]
pub(crate) struct MessagesResponse {
    id: String,
    #[serde(rename = "type")]
    message_type: &'static str,
    #[serde(flatten)]
    message: Message,
    model: String,
    stop_reason: &'static str,
    stop_sequence: Option<String>,
    usage: MessagesUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_transfer_params: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ec_transfer_params: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
struct MessagesUsage {
    input_tokens: usize,
    output_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_read_input_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_creation_input_tokens: Option<usize>,
}

fn stop_reason(finish: &FinishReason) -> Result<(&'static str, Option<&str>), ApiError> {
    match finish {
        FinishReason::Error(error) => Err(crate::error::engine_error(error.as_deref())),
        FinishReason::Length => Ok(("max_tokens", None)),
        FinishReason::Stop(Some(StopReason::Text(text))) => Ok(("stop_sequence", Some(text))),
        FinishReason::Stop(_) => Ok(("end_turn", None)),
        _ => Err(server_error!("Generation did not complete successfully")),
    }
}

fn usage(usage: TokenUsage, include_cache: bool) -> MessagesUsage {
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
