// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use axum::Json;
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value};
use vllm_chat::{AssistantContentBlock, AssistantMessageExt as _, FinishReason};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_llm::TokenUsage;

use super::types::{
    ErrorDetail, ErrorResponse, ErrorType, MessagesStopReason, MessagesUsage, OutputContentBlock,
};
use crate::error::{ApiError, server_error};

pub(in crate::routes) struct AnthropicError(ApiError);

impl From<ApiError> for AnthropicError {
    fn from(error: ApiError) -> Self {
        Self(error)
    }
}

impl IntoResponse for AnthropicError {
    fn into_response(self) -> Response {
        (self.0.status_code(), Json(self.into_body())).into_response()
    }
}

impl AnthropicError {
    pub(super) fn into_body(self) -> ErrorResponse {
        let error = self.0;
        let kind = match error {
            ApiError::InvalidRequest { .. }
            | ApiError::JsonParseError { .. }
            | ApiError::InvalidQuery { .. }
            | ApiError::Conflict { .. } => ErrorType::InvalidRequestError,
            ApiError::ModelNotFound { .. } | ApiError::NotFound { .. } => ErrorType::NotFoundError,
            ApiError::ServiceUnavailable { .. } => ErrorType::OverloadedError,
            ApiError::ServerError { .. } => ErrorType::ApiError,
        };
        ErrorResponse {
            error: ErrorDetail {
                kind,
                message: error.message(),
            },
        }
    }
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
                signature: thinking_signature(),
            }),
            AssistantContentBlock::ToolCall(call) => Ok(OutputContentBlock::ToolUse {
                input: parse_tool_input(&call.arguments)
                    .ok_or_else(|| server_error!("Model returned invalid tool arguments"))?,
                id: call.id,
                name: call.name,
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

pub(super) fn thinking_signature() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn parse_tool_input(arguments: &str) -> Option<Map<String, Value>> {
    serde_json::from_str(arguments).ok()
}

pub(super) fn stop_reason<'a>(
    finish: &'a FinishReason,
    blocks: &[AssistantContentBlock],
) -> Result<(Option<MessagesStopReason>, Option<&'a str>), ApiError> {
    match finish {
        FinishReason::Error(error) => Err(crate::error::engine_error(error.as_deref())),
        FinishReason::Length => Ok((Some(MessagesStopReason::MaxTokens), None)),
        FinishReason::Stop(stop) => {
            if let Some(StopReason::Text(text)) = stop {
                let mut tools = blocks.tool_calls().peekable();
                if tools.peek().is_none()
                    || !tools.all(|call| parse_tool_input(&call.arguments).is_some())
                {
                    return Ok((Some(MessagesStopReason::StopSequence), Some(text)));
                }
            }
            let reason = if blocks.has_tool_calls() {
                MessagesStopReason::ToolUse
            } else {
                MessagesStopReason::EndTurn
            };
            Ok((Some(reason), None))
        }
        FinishReason::Abort | FinishReason::Repetition(_) => Ok((None, None)),
    }
}

pub(super) fn convert_usage(usage: TokenUsage, include_cache: bool) -> MessagesUsage {
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
