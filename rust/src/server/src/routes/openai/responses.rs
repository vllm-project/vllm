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
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tracing::{error, info, trace};
use tracing_futures::Instrument as _;
use vllm_chat::{ChatEvent, ChatEventStream, ChatEventStreamTrait, ChatTokenUsage, FinishReason};

use self::convert::{ResponseMeta, build_response, build_usage, prepare_responses_request};
use self::streaming::{OutputItemStreamer, ResponseStreamEvent, response_lifecycle_event};
use self::types::{ResponseError, ResponseItemStatus, ResponsesRequest, ResponsesResponse};
use crate::config::ApiServerOptions;
use crate::error::{ApiError, chat_submit_error, server_error, text_submit_error};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{resolve_request_context, sse_response, unix_timestamp};

/// Create one response (`POST /v1/responses`).
pub async fn create_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(mut body): ValidatedJson<ResponsesRequest>,
) -> Response {
    let stream_requested = body.stream.unwrap_or(false);
    let sampling_hints = match state.chat.text().request_processor().sampling_hints() {
        Ok(hints) => hints,
        Err(error) => {
            return text_submit_error("failed to resolve sampling defaults", error).into_response();
        }
    };
    body.temperature = body.temperature.or(sampling_hints.default_temperature);
    body.top_p = body.top_p.or(sampling_hints.default_top_p);
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

    if stream_requested {
        let event_stream =
            responses_event_stream(chat_stream, prepared.meta, prepared.request_id, created_at);
        let sse_stream = responses_sse_stream(event_stream).instrument(request_span);
        sse_response(sse_stream, api_server_options.sse_keep_alive_interval)
    } else {
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
    usage: ChatTokenUsage,
    finish_reason: FinishReason,
    kv_transfer_params: Option<serde_json::Value>,
    ec_transfer_params: Option<serde_json::Value>,
}

/// Convert one chat event stream into Responses API SSE events.
///
/// Emits `response.created`/`response.in_progress` upfront, item events as
/// generation proceeds, and one terminal lifecycle event. A length-limited
/// response emits `response.incomplete`; an aborted response uses
/// `response.completed` with a `cancelled` response status because the
/// Responses streaming schema has no separate cancellation event. Mid-stream
/// errors are reported through `response.failed` so the transport stream
/// itself stays infallible.
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
                for event in items.on_stream_end() {
                    y.yield_ok(event).await;
                }
                emit_failed(
                    &mut y,
                    &meta,
                    &request_id,
                    created_at,
                    items.completed_output_items(),
                    "The response stream failed before generation completed.",
                    None,
                )
                .await;
                return Ok(());
            }
        }
    }

    for event in items.on_stream_end() {
        y.yield_ok(event).await;
    }

    let Some(terminal) = terminal else {
        error!("responses stream ended before the terminal done event");
        emit_failed(
            &mut y,
            &meta,
            &request_id,
            created_at,
            items.completed_output_items(),
            "The response stream ended before generation completed.",
            None,
        )
        .await;
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
        emit_failed(
            &mut y,
            &meta,
            &request_id,
            created_at,
            items.final_output_items(&message),
            "The model failed to generate a response.",
            Some(&usage),
        )
        .await;
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
        terminal_event_type(&finish_reason),
        &final_response,
    ))
    .await;
    Ok(())
}

/// Return the Responses lifecycle event associated with a terminal finish.
fn terminal_event_type(finish_reason: &FinishReason) -> &'static str {
    match finish_reason {
        FinishReason::Length => "response.incomplete",
        // The Responses streaming API exposes no `response.cancelled` event.
        // Preserve the cancellation outcome on the response object itself.
        FinishReason::Abort | FinishReason::Stop(_) | FinishReason::Repetition(_) => {
            "response.completed"
        }
        // Error finishes are emitted above with an error payload.
        FinishReason::Error => "response.failed",
    }
}

/// Emit one terminal `response.failed` event.
async fn emit_failed(
    y: &mut TryYielder<ResponseStreamEvent, Infallible>,
    meta: &ResponseMeta,
    request_id: &str,
    created_at: u64,
    output: Vec<self::types::ResponseOutputItem>,
    message: &str,
    usage: Option<&ChatTokenUsage>,
) {
    let mut failed = build_response(
        meta,
        request_id,
        created_at,
        output,
        ResponseItemStatus::Failed,
        usage.map(build_usage),
        None,
        None,
    );
    failed.error = Some(ResponseError {
        code: "server_error".to_string(),
        message: message.to_string(),
    });
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
        let event_type = event.event_type();
        let data = event.into_json(sequence);
        trace!(payload = %data, "responses emitting event");
        y.yield_ok(Event::default().event(event_type).data(data)).await;
        sequence += 1;
    }
    Ok(())
}
