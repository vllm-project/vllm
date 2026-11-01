// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::json;
use validator::Validate;

use crate::error::{ApiError, chat_submit_error, invalid_request, server_error, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::routes::openai::utils::types::{
    Normalizable, StringOrArray, validate_stop, validate_top_p_value,
};
use crate::routes::openai::utils::validate_max_output_tokens;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::output_log::LoggedOutput;
use crate::state::AppState;
use crate::utils::{ResolvedRequestContext, resolve_request_context};
use tracing_futures::Instrument as _;
use vllm_chat::{
    AssistantContentBlock, AssistantMessageExt as _, ChatContent, ChatContentPart, ChatMessage,
    ChatOptions, ChatRequest, FinishReason, GenerationPromptMode, ResolvedToolContext,
    SamplingParams,
};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_llm::TokenUsage;

#[derive(Debug, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct CountTokensRequest {
    model: String,
    messages: Vec<Message>,
    system: Option<TextContent>,
}

impl Normalizable for CountTokensRequest {}

#[derive(Debug, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase", deny_unknown_fields)]
enum Message {
    User { content: TextContent },
    Assistant { content: TextContent },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TextContent {
    Text(String),
    Blocks(Vec<TextBlock>),
}

#[derive(Debug, Deserialize)]
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

    fn into_text(self, strip_billing: bool) -> String {
        match self {
            Self::Text(text) => text,
            Self::Blocks(blocks) => blocks
                .into_iter()
                .map(|TextBlock::Text { text }| text)
                .filter(|text| !strip_billing || !text.starts_with("x-anthropic-billing-header"))
                .collect(),
        }
    }
}

impl CountTokensRequest {
    fn into_chat(
        self,
        model: &LoraModelResolution,
        ctx: ResolvedRequestContext,
    ) -> Result<ChatRequest, ApiError> {
        if self.model.is_empty() || self.messages.is_empty() {
            return Err(invalid_request!("model and messages must be non-empty"));
        }
        model.model_names.check(Some(&self.model))?;
        let mut messages = Vec::new();
        if let Some(system) = self.system {
            let text = system.into_text(true);
            if !text.is_empty() {
                messages.push(ChatMessage::system(text));
            }
        }
        for message in self.messages {
            messages.push(match message {
                Message::User { content } => ChatMessage::user(content.into_content()),
                Message::Assistant { content } => {
                    ChatMessage::assistant_text(content.into_text(false))
                }
            });
        }
        let continuation = matches!(messages.last(), Some(ChatMessage::Assistant { content })
            if content.iter().any(|b| matches!(b, AssistantContentBlock::Text { text } if !text.is_empty())));
        Ok(ChatRequest {
            request_id: format!("msg_{}", ctx.request_id),
            messages,
            sampling_params: SamplingParams::default(),
            chat_options: ChatOptions {
                generation_prompt_mode: if continuation {
                    GenerationPromptMode::ContinueFinalAssistant
                } else {
                    GenerationPromptMode::StartNewAssistant
                },
                ..Default::default()
            },
            tool_context: ResolvedToolContext::default(),
            decode_options: Default::default(),
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
        })
    }
}

pub struct AnthropicError(ApiError);

impl From<ApiError> for AnthropicError {
    fn from(error: ApiError) -> Self {
        Self(error)
    }
}

impl IntoResponse for AnthropicError {
    fn into_response(self) -> Response {
        let status = self.0.status_code();
        let kind = match status.as_u16() {
            400 => "invalid_request_error",
            404 => "not_found_error",
            409 => "conflict_error",
            503 => "overloaded_error",
            _ => "api_error",
        };
        (
            status,
            Json(json!({"type": "error", "error": {
                "type": kind, "message": self.0.to_error_response().error.message,
            }})),
        )
            .into_response()
    }
}

pub async fn count_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<ValidatedJson<CountTokensRequest>, ApiError>,
) -> Result<Json<serde_json::Value>, AnthropicError> {
    let body = body?.0;
    let model = state.resolve_model_with_loras(Some(&body.model)).await;
    let request = body.into_chat(&model, resolve_request_context(&headers, None))?;
    let text = state
        .chat
        .request_processor()
        .prepare_for_tokenization(request)
        .await
        .map_err(|error| chat_submit_error("count_tokens failed", error))?;
    let tokens = state
        .chat
        .text()
        .request_processor()
        .tokenize(text)
        .map_err(|error| text_submit_error("count_tokens failed", error))?;
    Ok(Json(json!({"input_tokens": tokens.len(),
        "context_management": {"original_input_tokens": tokens.len()}})))
}

#[derive(Debug, Deserialize, Validate)]
pub struct MessagesRequest {
    #[serde(flatten)]
    input: CountTokensRequest,
    #[validate(range(min = 1))]
    max_tokens: u32,
    #[validate(range(min = 0.0, max = 2.0))]
    temperature: Option<f32>,
    #[validate(custom(function = "validate_top_p_value"))]
    top_p: Option<f32>,
    top_k: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    stream: Option<bool>,
}

impl Normalizable for MessagesRequest {}

impl MessagesRequest {
    fn into_chat(
        self,
        model: &LoraModelResolution,
        ctx: ResolvedRequestContext,
    ) -> Result<ChatRequest, ApiError> {
        if self.stream == Some(true) {
            return Err(invalid_request!("Streaming Messages is not supported yet"));
        }
        let mut request = self.input.into_chat(model, ctx)?;
        request.sampling_params.max_tokens = Some(self.max_tokens);
        request.sampling_params.temperature = self.temperature;
        request.sampling_params.top_p = self.top_p;
        request.sampling_params.top_k = self.top_k;
        if let Some(stops) = self.stop_sequences {
            let stops = StringOrArray::Array(stops);
            validate_stop(&stops).map_err(|error| invalid_request!("{error}"))?;
            request.decode_options.stop_strings = Some(stops.into_vec());
        }
        request
            .validate()
            .map_err(|error| chat_submit_error("invalid Messages request", error))?;
        Ok(request)
    }
}

pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<ValidatedJson<MessagesRequest>, ApiError>,
) -> Result<Response, AnthropicError> {
    let body = body?.0;
    let model = state.resolve_model_with_loras(Some(&body.input.model)).await;
    let request = body.into_chat(&model, resolve_request_context(&headers, None))?;
    validate_max_output_tokens(
        "max_tokens",
        request.sampling_params.max_tokens,
        state.chat.engine_core_client().max_model_len(),
    )?;
    let request_id = request.request_id.clone();
    let span = tracing::info_span!("messages", request_id = %request_id, engine_request_id = tracing::field::Empty);
    let response = async {
        let response = state.chat.chat(request).await
            .map_err(|error| chat_submit_error("failed to submit Messages request", error))?
            .collect_message().await
            .map_err(|error| chat_submit_error("failed to collect Messages response", error))?;
        let (reason, sequence) = stop_reason(&response.finish_reason)?;
        if response.message.has_reasoning() || response.message.has_tool_calls() {
            return Err(invalid_request!("Reasoning and tool output are not supported yet"));
        }
        let text = response.message.text();
        let options = state.api_server_options;
        if options.enable_log_requests {
            tracing::info!(model = %model.response_model(), prompt_tokens = response.usage.prompt_token_count,
                output_tokens = response.usage.output_token_count, finish_reason = reason, "Messages finished");
        }
        if options.log_outputs.logs_complete() && !text.is_empty() {
            options.log_outputs.log(LoggedOutput::Response, &text, Some(&response.token_ids), Some(reason));
        }
        Ok(MessagesResponse {
            id: request_id, message_type: "message", role: "assistant", model: model.response_model(),
            content: vec![ResponseBlock::Text { text }], stop_reason: reason, stop_sequence: sequence.map(str::to_owned),
            usage: usage(response.usage.engine, options.enable_prompt_tokens_details),
            kv_transfer_params: response.kv_transfer_params, ec_transfer_params: response.ec_transfer_params,
        })
    }.instrument(span).await?;
    Ok(Json(response).into_response())
}

#[derive(serde::Serialize)]
struct MessagesResponse {
    id: String,
    #[serde(rename = "type")]
    message_type: &'static str,
    role: &'static str,
    model: String,
    content: Vec<ResponseBlock>,
    stop_reason: &'static str,
    stop_sequence: Option<String>,
    usage: MessagesUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_transfer_params: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ec_transfer_params: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseBlock {
    Text { text: String },
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
