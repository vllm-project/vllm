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
use crate::routes::openai::chat_completions::{
    ChatCompletionOutput, ChatCompletionRequest, create_chat_completion, lower_chat_request,
};
use crate::routes::openai::utils::types::{ChatMessage, ContentPart, MessageContent, Normalizable};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::resolve_request_context;

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
    fn into_content(self) -> MessageContent {
        match self {
            Self::Text(text) => MessageContent::Text(text),
            Self::Blocks(blocks) => {
                let mut parts: Vec<_> = blocks
                    .into_iter()
                    .map(|TextBlock::Text { text }| ContentPart::Text { text })
                    .collect();
                if parts.len() == 1 {
                    let ContentPart::Text { text } = parts.pop().unwrap() else {
                        unreachable!()
                    };
                    MessageContent::Text(text)
                } else {
                    MessageContent::Parts(parts)
                }
            }
        }
    }

    fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Blocks(blocks) => {
                blocks.into_iter().map(|TextBlock::Text { text }| text).collect()
            }
        }
    }
}

impl CountTokensRequest {
    fn into_chat(self) -> Result<ChatCompletionRequest, ApiError> {
        if self.model.is_empty() || self.messages.is_empty() {
            return Err(invalid_request!("model and messages must be non-empty"));
        }
        let mut messages = Vec::new();
        if let Some(system) = self.system {
            let text = system.into_text();
            if !text.is_empty() {
                messages.push(ChatMessage::System {
                    content: MessageContent::Text(text),
                    name: None,
                });
            }
        }
        for message in self.messages {
            messages.push(match message {
                Message::User { content } => ChatMessage::User {
                    content: content.into_content(),
                    name: None,
                },
                Message::Assistant { content } => ChatMessage::Assistant {
                    content: Some(content.into_content()),
                    name: None,
                    reasoning: None,
                    tool_calls: None,
                },
            });
        }
        let mut request = ChatCompletionRequest {
            model: Some(self.model),
            messages,
            ..Default::default()
        };
        request.normalize();
        request.validate().map_err(|error| invalid_request!("{error}"))?;
        Ok(request)
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
    let request = body?.0.into_chat()?;
    let model = state.resolve_model_with_loras(request.model.as_deref()).await;
    let request = lower_chat_request(request, &model, resolve_request_context(&headers, None))?;
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
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    stream: Option<bool>,
}

impl Normalizable for MessagesRequest {}

impl MessagesRequest {
    fn into_chat(self) -> Result<ChatCompletionRequest, ApiError> {
        if self.stream == Some(true) {
            return Err(invalid_request!("Streaming Messages is not supported yet"));
        }
        let mut request = self.input.into_chat()?;
        request.max_completion_tokens = Some(self.max_tokens);
        request.temperature = self.temperature;
        request.top_p = self.top_p;
        request.top_k = self.top_k;
        request.stop = self
            .stop_sequences
            .map(crate::routes::openai::utils::types::StringOrArray::Array);
        request.validate().map_err(|error| invalid_request!("{error}"))?;
        Ok(request)
    }
}

pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<ValidatedJson<MessagesRequest>, ApiError>,
) -> Result<Response, AnthropicError> {
    let request = body?.0.into_chat()?;
    let ChatCompletionOutput::Full(response) =
        create_chat_completion(state, headers, request).await?
    else {
        unreachable!("streaming rejected during request conversion");
    };
    let choice = &response.choices[0];
    let (reason, sequence) =
        stop_reason(choice.finish_reason.as_deref(), choice.stop_reason.as_ref())?;
    if choice.message.reasoning.is_some() || !choice.message.tool_calls.is_empty() {
        return Err(invalid_request!("Reasoning and tool output are not supported yet").into());
    }
    let mut message = json!({
        "id": response.id, "type": "message", "role": "assistant", "model": response.model,
        "content": [{"type": "text", "text": choice.message.content.as_deref().unwrap_or("")}],
        "stop_reason": reason, "stop_sequence": sequence,
        "usage": usage(response.usage.as_ref()),
    });
    if let Some(params) = response.kv_transfer_params {
        message["kv_transfer_params"] = params;
    }
    if let Some(params) = response.ec_transfer_params {
        message["ec_transfer_params"] = params;
    }
    Ok(Json(message).into_response())
}

fn stop_reason<'a>(
    finish: Option<&str>,
    stop: Option<&'a serde_json::Value>,
) -> Result<(&'static str, Option<&'a str>), ApiError> {
    match finish {
        Some("length") => Ok(("max_tokens", None)),
        Some("stop") => match stop.and_then(serde_json::Value::as_str) {
            Some(text) => Ok(("stop_sequence", Some(text))),
            None => Ok(("end_turn", None)),
        },
        _ => Err(server_error!("Generation did not complete successfully")),
    }
}

fn usage(usage: Option<&crate::routes::openai::utils::types::Usage>) -> serde_json::Value {
    let Some(usage) = usage else {
        return json!({"input_tokens": 0, "output_tokens": 0});
    };
    let mut result = json!({"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens.unwrap_or(0)});
    if let Some(details) = &usage.prompt_tokens_details {
        let cached = details.cached_tokens;
        let created = details.created_cache_tokens;
        result["input_tokens"] =
            json!(usage.prompt_tokens.saturating_sub(cached).saturating_sub(created));
        result["cache_read_input_tokens"] = json!(cached);
        result["cache_creation_input_tokens"] = json!(created);
    }
    result
}
