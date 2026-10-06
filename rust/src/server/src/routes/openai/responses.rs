// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! OpenAI Responses API handler for the Rust frontend.
//!
//! Implements the stateless subset of the Python frontend's
//! `vllm/entrypoints/openai/responses/serving.py`: single-turn generation
//! from full conversation replays, non-streaming, with
//! reasoning items and function tool calls. Store-dependent features
//! (`background`, `previous_response_id`, response retrieval/cancellation)
//! require a server-side response store that this frontend does not have;
//! see `validate.rs` for the enforced behavior.

pub mod types;

mod convert;
mod validate;

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use thiserror_ext::AsReport as _;
use tracing::info;
use tracing_futures::Instrument as _;
use vllm_chat::{ChatEventStream, FinishReason};

use self::convert::{ResponseMeta, build_response, build_usage, prepare_responses_request};
use self::types::{ResponseItemStatus, ResponsesRequest, ResponsesResponse};
use crate::config::ApiServerOptions;
use crate::error::{ApiError, chat_submit_error, invalid_request, server_error};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

/// Create one response (`POST /v1/responses`).
pub async fn create_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ResponsesRequest>,
) -> Response {
    // TODO: Add Responses streaming support.
    if body.stream {
        return invalid_request!(
            param = "stream",
            "streaming Responses are not supported yet"
        )
        .into_response();
    }
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state
        .resolve_model_with_loras(body.model.as_deref().filter(|model| !model.is_empty()))
        .await;

    let prepared = match prepare_responses_request(body, &lora_resolution, request_context) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    let request_span = tracing::info_span!(
        "responses",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let created_at = unix_timestamp();
    let api_server_options = state.api_server_options;

    let chat_stream =
        match state.chat.chat(prepared.chat_request).instrument(request_span.clone()).await {
            Ok(stream) => stream,
            Err(error) => {
                return chat_submit_error("failed to submit responses request", error)
                    .into_response();
            }
        };

    let response = match collect_responses(
        chat_stream,
        &prepared.meta,
        &prepared.request_id,
        created_at,
        &api_server_options,
    )
    .instrument(request_span)
    .await
    {
        Ok(response) => response,
        Err(error) => return error.into_response(),
    };
    Json(response).into_response()
}

/// Retrieve one response (`GET /v1/responses/{response_id}`).
///
/// Nothing is ever stored in this frontend, so every ID is unknown.
pub async fn retrieve_response(Path(response_id): Path<String>) -> Response {
    ApiError::response_not_found(response_id).into_response()
}

/// Cancel one response (`POST /v1/responses/{response_id}/cancel`).
///
/// Nothing is ever stored in this frontend, so every ID is unknown.
pub async fn cancel_response(Path(response_id): Path<String>) -> Response {
    ApiError::response_not_found(response_id).into_response()
}

/// Collect one non-streaming response from the chat event stream.
async fn collect_responses(
    stream: ChatEventStream,
    meta: &ResponseMeta,
    request_id: &str,
    created_at: u64,
    ApiServerOptions {
        enable_log_requests,
        ..
    }: &ApiServerOptions,
) -> Result<ResponsesResponse, ApiError> {
    let collected = stream.collect_message().await.map_err(|error| {
        server_error!(
            "failed to collect responses result: {}",
            error.to_report_string()
        )
    })?;
    let vllm_chat::CollectedAssistantMessage {
        message,
        usage,
        finish_reason,
        kv_transfer_params,
        ec_transfer_params,
        ..
    } = collected;

    if matches!(finish_reason, FinishReason::Error) {
        return Err(server_error!(
            "responses generation failed with a retryable internal error"
        ));
    }
    let status = response_status(&finish_reason);

    if *enable_log_requests {
        info!(
            model = %meta.model,
            prompt_tokens = usage.prompt_token_count,
            output_tokens = usage.output_token_count,
            finish_reason = finish_reason.as_str(),
            "responses finished"
        );
    }

    Ok(build_response(
        meta,
        request_id,
        created_at,
        convert::build_output_items(&message, meta.include_reasoning),
        status,
        Some(build_usage(&usage)),
        kv_transfer_params,
        ec_transfer_params,
    ))
}

/// Map the internal finish reason onto the response status.
fn response_status(finish_reason: &FinishReason) -> ResponseItemStatus {
    match finish_reason {
        FinishReason::Length => ResponseItemStatus::Incomplete,
        FinishReason::Abort => ResponseItemStatus::Cancelled,
        FinishReason::Error => ResponseItemStatus::Failed,
        FinishReason::Stop(_) | FinishReason::Repetition(_) => ResponseItemStatus::Completed,
    }
}
