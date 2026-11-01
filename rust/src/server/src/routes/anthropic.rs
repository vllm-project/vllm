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

use crate::error::{ApiError, chat_submit_error, invalid_request, text_submit_error};
use crate::routes::openai::chat_completions::{ChatCompletionRequest, lower_chat_request};
use crate::routes::openai::utils::types::{ChatMessage, MessageContent, Normalizable};
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
            messages.push(ChatMessage::System {
                content: MessageContent::Text(system.into_text()),
                name: None,
            });
        }
        for message in self.messages {
            messages.push(match message {
                Message::User { content } => ChatMessage::User {
                    content: MessageContent::Text(content.into_text()),
                    name: None,
                },
                Message::Assistant { content } => ChatMessage::Assistant {
                    content: Some(MessageContent::Text(content.into_text())),
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
