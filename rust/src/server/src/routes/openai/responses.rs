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
mod tools;
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
use vllm_llm::RequestTimestamps;

use self::convert::{build_response, build_usage, prepare_responses_request};
use self::streaming::{OutputItemStreamer, ResponseStreamEvent, SequencedEvent};
use self::types::{
    IncompleteDetails, ResponseError, ResponseItemStatus, ResponseOutputItem, ResponsesRequest,
    ResponsesResponse,
};
use crate::config::ApiServerOptions;
use crate::error::{ApiError, chat_submit_error, engine_error, server_error};
use crate::routes::openai::utils::metrics::PerRequestMetrics;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{resolve_request_context, sse_response, unix_timestamp};

/// Create one response (`POST /v1/responses`).
pub async fn create_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ResponsesRequest>,
) -> Response {
    let stream_requested = body.stream.unwrap_or(false);
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
    // Echo the values the engine uses, as Python does, rather than the request's.
    let response = build_response(
        prepared.meta,
        chat_stream.sampling(),
        prepared.request_id,
        created_at,
    );

    let items = OutputItemStreamer::new(prepared.include_reasoning, prepared.tool_names);
    if stream_requested {
        let event_stream = responses_event_stream(
            chat_stream,
            response,
            items,
            api_server_options.enable_per_request_metrics,
        );
        let sse_stream = responses_sse_stream(event_stream).instrument(request_span);
        sse_response(sse_stream, api_server_options.sse_keep_alive_interval)
    } else {
        match collect_responses(chat_stream, response, items, &api_server_options)
            .instrument(request_span)
            .await
        {
            Ok(response) => Json(response).into_response(),
            Err(error) => error.into_response(),
        }
    }
}

/// Retrieve one response (`GET /v1/responses/{response_id}`).
///
/// Nothing is ever stored in this frontend, so every ID is unknown.
pub async fn retrieve_response(Path(response_id): Path<String>) -> Response {
    response_not_found(&response_id).into_response()
}

/// Cancel one response (`POST /v1/responses/{response_id}/cancel`).
///
/// Nothing is ever stored in this frontend, so every ID is unknown.
pub async fn cancel_response(Path(response_id): Path<String>) -> Response {
    response_not_found(&response_id).into_response()
}

/// The error for a response ID, which is always unknown without a store.
fn response_not_found(response_id: &str) -> ApiError {
    ApiError::not_found(format!("Response with id '{response_id}' not found."))
}

/// Collect one non-streaming response by driving the streaming state machine
/// and keeping its terminal response.
///
/// Unlike Chat Completions, which folds through `collect_message`, the
/// Responses terminal event carries the complete response, so `output` comes
/// from the same item rules in both modes. Errors still map to HTTP statuses
/// instead of a `failed` response.
async fn collect_responses(
    mut stream: ChatEventStream,
    response: ResponsesResponse,
    mut items: OutputItemStreamer,
    ApiServerOptions {
        enable_log_requests,
        enable_per_request_metrics,
        ..
    }: &ApiServerOptions,
) -> Result<ResponsesResponse, ApiError> {
    let terminal = loop {
        let Some(event) = stream.next().await else {
            return Err(server_error!(
                "responses stream ended before generation completed"
            ));
        };
        let event = event.map_err(|error| {
            server_error!(
                "failed to collect responses result: {}",
                error.to_report_string()
            )
        })?;
        // Only the final output matters here; the item events are discarded.
        items.on_event(&event);
        if let Some(terminal) = TerminalOutput::from_event(event) {
            break terminal;
        }
    };

    if let FinishReason::Error(error) = &terminal.finish_reason {
        return Err(engine_error(error.as_deref()));
    }
    items.on_stream_end(Some(&terminal.finish_reason));

    if *enable_log_requests {
        info!(
            model = %response.model,
            prompt_tokens = terminal.usage.prompt_token_count,
            output_tokens = terminal.usage.output_token_count,
            finish_reason = terminal.finish_reason.as_str(),
            "responses finished"
        );
    }

    Ok(completed_response(
        response,
        items.into_output_items(),
        terminal,
        *enable_per_request_metrics,
    ))
}

/// Map the internal finish reason onto the response status.
fn response_status(finish_reason: &FinishReason) -> ResponseItemStatus {
    match finish_reason {
        FinishReason::Length => ResponseItemStatus::Incomplete,
        FinishReason::Abort => ResponseItemStatus::Cancelled,
        FinishReason::Error(_) => ResponseItemStatus::Failed,
        FinishReason::Stop(_) | FinishReason::Repetition(_) => ResponseItemStatus::Completed,
    }
}

/// Terminal event metadata captured from the internal `Done` chat event.
struct TerminalOutput {
    usage: ChatTokenUsage,
    finish_reason: FinishReason,
    kv_transfer_params: Option<serde_json::Value>,
    ec_transfer_params: Option<serde_json::Value>,
    timestamps: RequestTimestamps,
}

impl TerminalOutput {
    /// Take the terminal metadata from a `Done` event.
    fn from_event(event: ChatEvent) -> Option<Self> {
        match event {
            ChatEvent::Done {
                // The output items already carry the parser's final blocks.
                message: _,
                usage,
                finish_reason,
                kv_transfer_params,
                ec_transfer_params,
                timestamps,
                ..
            } => Some(Self {
                usage,
                finish_reason,
                kv_transfer_params: kv_transfer_params.map(|params| *params),
                ec_transfer_params: ec_transfer_params.map(|params| *params),
                timestamps,
            }),
            _ => None,
        }
    }
}

/// Finish the in-progress response of a generation that ended normally.
fn completed_response(
    response: ResponsesResponse,
    output: Vec<ResponseOutputItem>,
    TerminalOutput {
        usage,
        finish_reason,
        kv_transfer_params,
        ec_transfer_params,
        timestamps,
    }: TerminalOutput,
    enable_per_request_metrics: bool,
) -> ResponsesResponse {
    let status = response_status(&finish_reason);
    ResponsesResponse {
        status,
        incomplete_details: (status == ResponseItemStatus::Incomplete).then(|| IncompleteDetails {
            reason: "max_output_tokens".to_string(),
        }),
        output,
        usage: Some(build_usage(&usage)),
        metrics: enable_per_request_metrics
            .then(|| PerRequestMetrics::from_timestamps(timestamps, usage.output_token_count)),
        kv_transfer_params,
        ec_transfer_params,
        ..response
    }
}

/// Fail the in-progress response with a server error.
fn failed_response(
    response: ResponsesResponse,
    output: Vec<ResponseOutputItem>,
    message: &str,
    usage: Option<&ChatTokenUsage>,
) -> ResponsesResponse {
    ResponsesResponse {
        status: ResponseItemStatus::Failed,
        error: Some(ResponseError {
            code: "server_error".to_string(),
            message: message.to_string(),
        }),
        output,
        usage: usage.map(build_usage),
        ..response
    }
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
    response: ResponsesResponse,
    mut items: OutputItemStreamer,
    enable_per_request_metrics: bool,
    mut y: TryYielder<ResponseStreamEvent, Infallible>,
) -> Result<(), Infallible> {
    let initial = Arc::new(response);
    y.yield_ok(ResponseStreamEvent::Created {
        response: Arc::clone(&initial),
    })
    .await;
    y.yield_ok(ResponseStreamEvent::InProgress {
        response: Arc::clone(&initial),
    })
    .await;
    let response = Arc::unwrap_or_clone(initial);

    let mut terminal = None;
    while let Some(next) = stream.next().await {
        match next {
            Ok(event) => {
                for event in items.on_event(&event) {
                    y.yield_ok(event).await;
                }
                if let Some(output) = TerminalOutput::from_event(event) {
                    terminal = Some(output);
                    break;
                }
            }
            Err(error) => {
                error!(error = %error.as_report(), "responses stream failed");
                for event in items.on_stream_end(None) {
                    y.yield_ok(event).await;
                }
                let failed = failed_response(
                    response,
                    items.into_output_items(),
                    "The response stream failed before generation completed.",
                    None,
                );
                y.yield_ok(ResponseStreamEvent::Failed {
                    response: Arc::new(failed),
                })
                .await;
                return Ok(());
            }
        }
    }

    for event in items.on_stream_end(terminal.as_ref().map(|output| &output.finish_reason)) {
        y.yield_ok(event).await;
    }
    let output = items.into_output_items();

    let final_response = match terminal {
        None => {
            error!("responses stream ended before the terminal done event");
            failed_response(
                response,
                output,
                "The response stream ended before generation completed.",
                None,
            )
        }
        Some(terminal) if matches!(terminal.finish_reason, FinishReason::Error(_)) => {
            failed_response(
                response,
                output,
                "The model failed to generate a response.",
                Some(&terminal.usage),
            )
        }
        Some(terminal) => {
            completed_response(response, output, terminal, enable_per_request_metrics)
        }
    };
    y.yield_ok(terminal_event(Arc::new(final_response))).await;
    Ok(())
}

/// Build the Responses lifecycle event that carries one terminal response.
fn terminal_event(response: Arc<ResponsesResponse>) -> ResponseStreamEvent {
    match response.status {
        ResponseItemStatus::Incomplete => ResponseStreamEvent::Incomplete { response },
        ResponseItemStatus::Failed => ResponseStreamEvent::Failed { response },
        // The Responses streaming API exposes no `response.cancelled` event.
        // Preserve the cancellation outcome on the response object itself.
        ResponseItemStatus::Completed | ResponseItemStatus::Cancelled => {
            ResponseStreamEvent::Completed { response }
        }
        ResponseItemStatus::Queued
        | ResponseItemStatus::InProgress
        | ResponseItemStatus::Cancelling => {
            unreachable!("terminal responses are completed or failed")
        }
    }
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
        trace!(?event, "responses emitting event");
        let sse_event = Event::default()
            .event(event.event_type())
            .json_data(SequencedEvent::new(&event, sequence))
            .expect("Responses stream event must serialize to JSON");
        y.yield_ok(sse_event).await;
        sequence += 1;
    }
    Ok(())
}
