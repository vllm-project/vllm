// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod convert;
mod types;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use tracing_futures::Instrument as _;
use vllm_chat::AssistantMessageExt as _;

use self::convert::{convert_output_blocks, prepare_messages_request, stop_reason, usage};
use self::types::{
    ContextManagement, CountTokensResponse, ErrorDetail, ErrorResponse, ErrorResponseType,
    ErrorType, MessageType, MessagesRequest, MessagesResponse,
};
use crate::error::{ApiError, chat_submit_error, invalid_request, text_submit_error};
use crate::routes::openai::utils::validate_max_output_tokens;
use crate::routes::output_log::LoggedOutput;
use crate::state::AppState;
use crate::utils::resolve_request_context;

pub(super) struct AnthropicError(ApiError);

impl From<ApiError> for AnthropicError {
    fn from(error: ApiError) -> Self {
        Self(error)
    }
}

impl IntoResponse for AnthropicError {
    fn into_response(self) -> Response {
        let error = self.0;
        let kind = match error {
            ApiError::InvalidRequest { .. }
            | ApiError::JsonParseError { .. }
            | ApiError::InvalidQuery { .. } => ErrorType::InvalidRequestError,
            ApiError::ModelNotFound { .. } | ApiError::NotFound { .. } => ErrorType::NotFoundError,
            ApiError::Conflict { .. } => ErrorType::ConflictError,
            ApiError::ServiceUnavailable { .. } => ErrorType::OverloadedError,
            ApiError::ServerError { .. } => ErrorType::ApiError,
        };
        let body = ErrorResponse {
            kind: ErrorResponseType::Error,
            error: ErrorDetail {
                kind,
                message: error.message(),
            },
        };
        (error.status_code(), Json(body)).into_response()
    }
}

pub(super) async fn count_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<MessagesRequest>, JsonRejection>,
) -> Result<impl IntoResponse, AnthropicError> {
    let Json(body) = body.map_err(ApiError::from)?;
    let model = state.resolve_model_with_loras(Some(&body.model)).await;
    let ctx = resolve_request_context(&headers, None)?;
    let request = prepare_messages_request(body, &model, ctx)?;
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
    Ok(Json(CountTokensResponse {
        input_tokens: tokens.len(),
        context_management: ContextManagement {
            original_input_tokens: tokens.len(),
        },
    }))
}

/// Anthropic request handling over the application's shared chat inference.
pub(super) async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<MessagesRequest>, JsonRejection>,
) -> Result<impl IntoResponse, AnthropicError> {
    let Json(body) = body.map_err(ApiError::from)?;
    let model = state.resolve_model_with_loras(Some(&body.model)).await;
    let options = state.api_server_options;
    if body.max_tokens.is_none() {
        return Err(invalid_request!(
            param = "max_tokens",
            "max_tokens is required for generation"
        )
        .into());
    }
    let ctx = resolve_request_context(&headers, None)?;
    let request = prepare_messages_request(body, &model, ctx)?;
    validate_max_output_tokens(
        "max_tokens",
        request.sampling_params.max_tokens,
        state.chat.text().llm().info().max_model_len,
    )?;
    let request_id = request.request_id.clone();
    let span = tracing::info_span!("messages", request_id = %request_id, engine_request_id = tracing::field::Empty);
    let response = async {
        let response = state
            .chat
            .chat(request)
            .await
            .map_err(|error| chat_submit_error("failed to submit Messages request", error))?
            .collect_message()
            .await
            .map_err(|error| chat_submit_error("failed to collect Messages response", error))?;
        let (reason, sequence) = stop_reason(&response.finish_reason)?;
        let text = response.message.text();
        let content = convert_output_blocks(response.message.content)?;
        if options.enable_log_requests {
            tracing::info!(
                model = %model.response_model(),
                prompt_tokens = response.usage.prompt_token_count,
                output_tokens = response.usage.output_token_count,
                finish_reason = reason.as_str(),
                "Messages finished"
            );
        }
        if options.log_outputs.logs_complete() && !text.is_empty() {
            options.log_outputs.log(
                LoggedOutput::Response,
                &text,
                Some(&response.token_ids),
                Some(reason.as_str()),
            );
        }
        Ok::<_, ApiError>(MessagesResponse {
            id: request_id,
            message_type: MessageType::Message,
            role: "assistant",
            content,
            model: model.response_model(),
            stop_reason: reason,
            stop_sequence: sequence.map(str::to_owned),
            usage: usage(response.usage.engine, options.enable_prompt_tokens_details),
            kv_transfer_params: response.kv_transfer_params,
            ec_transfer_params: response.ec_transfer_params,
        })
    }
    .instrument(span)
    .await?;
    Ok(Json(response))
}
