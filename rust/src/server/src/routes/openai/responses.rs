// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! OpenAI Responses API handler for the Rust frontend.
//!
//! Implements the stateless subset of the Python frontend's
//! `vllm/entrypoints/openai/responses/serving.py`: single-turn generation
//! from full conversation replays, streaming and non-streaming, with
//! reasoning items and function tool calls. Store-dependent features
//! (`background`, `previous_response_id`, response retrieval/cancellation)
//! require a server-side response store that this frontend does not have;
//! see `validate.rs` for the enforced behavior.

pub mod types;

mod convert;
mod streaming;
mod validate;

use std::convert::Infallible;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::Json;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tracing::{error, info, trace};
use tracing_futures::Instrument as _;
use vllm_chat::{ChatEvent, ChatEventStream, ChatEventStreamTrait, FinishReason};
use vllm_llm::TokenUsage;

use self::convert::{ResponseMeta, build_response, build_usage, prepare_responses_request};
use self::streaming::{OutputItemStreamer, ResponseStreamEvent, response_lifecycle_event};
use self::types::{ResponseItemStatus, ResponsesRequest, ResponsesResponse};
use crate::config::ApiServerOptions;
use crate::error::{ApiError, chat_submit_error, server_error};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

/// Create one response (`POST /v1/responses`).
pub async fn create_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ResponsesRequest>,
) -> Response {
    let stream_requested = body.stream;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(body.model.as_deref()).await;

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

    let chat_stream =
        match state.chat.chat(prepared.chat_request).instrument(request_span.clone()).await {
            Ok(stream) => stream,
            Err(error) => {
                return chat_submit_error("failed to submit responses request", error)
                    .into_response();
            }
        };

    if stream_requested {
        let event_stream =
            responses_event_stream(chat_stream, prepared.meta, prepared.request_id, created_at);
        let sse_stream = responses_sse_stream(event_stream).instrument(request_span);
        Sse::new(sse_stream).into_response()
    } else {
        let response = match collect_responses(
            chat_stream,
            &prepared.meta,
            &prepared.request_id,
            created_at,
            &state.api_server_options,
        )
        .instrument(request_span)
        .await
        {
            Ok(response) => response,
            Err(error) => return error.into_response(),
        };
        Json(response).into_response()
    }
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

/// Terminal event metadata captured from the internal `Done` chat event.
struct TerminalOutput {
    message: vllm_chat::AssistantMessage,
    usage: TokenUsage,
    finish_reason: FinishReason,
    kv_transfer_params: Option<serde_json::Value>,
    ec_transfer_params: Option<serde_json::Value>,
}

/// Convert one chat event stream into Responses API SSE events.
///
/// Emits `response.created`/`response.in_progress` upfront, item events as
/// generation proceeds, and one terminal `response.completed` or
/// `response.failed` event. Mid-stream errors are reported through
/// `response.failed` so the transport stream itself stays infallible.
#[try_stream]
async fn responses_event_stream(
    mut stream: impl ChatEventStreamTrait + Unpin,
    meta: ResponseMeta,
    request_id: String,
    created_at: u64,
    mut y: TryYielder<ResponseStreamEvent, Infallible>,
) -> Result<(), Infallible> {
    let mut items = OutputItemStreamer::new(meta.include_reasoning);

    let initial = build_response(
        &meta,
        &request_id,
        created_at,
        vec![],
        ResponseItemStatus::InProgress,
        None,
        None,
        None,
    );
    y.yield_ok(response_lifecycle_event("response.created", &initial)).await;
    y.yield_ok(response_lifecycle_event("response.in_progress", &initial)).await;

    let mut terminal: Option<TerminalOutput> = None;
    while let Some(next) = stream.next().await {
        match next {
            Ok(ChatEvent::Done {
                message,
                usage,
                finish_reason,
                kv_transfer_params,
                ec_transfer_params,
            }) => {
                terminal = Some(TerminalOutput {
                    message,
                    usage,
                    finish_reason,
                    kv_transfer_params,
                    ec_transfer_params,
                });
                break;
            }
            Ok(event) => {
                for event in items.on_event(&event) {
                    y.yield_ok(event).await;
                }
            }
            Err(error) => {
                error!(error = %error.as_report(), "responses stream failed");
                emit_failed(&mut y, &meta, &request_id, created_at).await;
                return Ok(());
            }
        }
    }

    for event in items.on_stream_end() {
        y.yield_ok(event).await;
    }

    let Some(terminal) = terminal else {
        error!("responses stream ended before the terminal done event");
        emit_failed(&mut y, &meta, &request_id, created_at).await;
        return Ok(());
    };
    let TerminalOutput {
        message,
        usage,
        finish_reason,
        kv_transfer_params,
        ec_transfer_params,
    } = terminal;

    if matches!(finish_reason, FinishReason::Error) {
        emit_failed(&mut y, &meta, &request_id, created_at).await;
        return Ok(());
    }

    let final_response = build_response(
        &meta,
        &request_id,
        created_at,
        items.final_output_items(&message),
        response_status(&finish_reason),
        Some(build_usage(&usage)),
        kv_transfer_params,
        ec_transfer_params,
    );
    y.yield_ok(response_lifecycle_event(
        "response.completed",
        &final_response,
    ))
    .await;
    Ok(())
}

/// Emit one terminal `response.failed` event.
async fn emit_failed(
    y: &mut TryYielder<ResponseStreamEvent, Infallible>,
    meta: &ResponseMeta,
    request_id: &str,
    created_at: u64,
) {
    let failed = build_response(
        meta,
        request_id,
        created_at,
        vec![],
        ResponseItemStatus::Failed,
        None,
        None,
        None,
    );
    y.yield_ok(response_lifecycle_event("response.failed", &failed)).await;
}

/// Convert Responses API SSE events into transport-level SSE events,
/// assigning `sequence_number` globally and emitting both `event:` and
/// `data:` lines like the Python frontend.
#[try_stream]
async fn responses_sse_stream(
    stream: impl Stream<Item = Result<ResponseStreamEvent, Infallible>>,
    mut y: TryYielder<Event, Infallible>,
) -> Result<(), Infallible> {
    pin_mut!(stream);
    let mut sequence = 0u64;
    while let Some(next) = stream.next().await {
        let event = match next {
            Ok(event) => event,
            Err(error) => match error {},
        };
        let data = event.to_json(sequence);
        trace!(payload = %data, "responses emitting event");
        y.yield_ok(Event::default().event(event.event_type()).data(data)).await;
        sequence += 1;
    }
    Ok(())
}
