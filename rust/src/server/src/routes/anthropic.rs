// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod convert;
mod types;

use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use futures::StreamExt as _;
use serde_json::json;
use std::convert::Infallible;
use tracing_futures::Instrument as _;
use vllm_chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessageExt as _, ChatEvent,
    ChatEventStream, ChatEventStreamTrait, ChatRequest,
};

use self::convert::{convert_output_blocks, prepare_messages_request, stop_reason, usage};
use self::types::{
    ContextManagement, CountTokensResponse, ErrorDetail, ErrorResponse, ErrorResponseType,
    ErrorType, MessageType, MessagesRequest, MessagesResponse,
};
use crate::error::{ApiError, chat_submit_error, invalid_request, text_submit_error};
use crate::routes::openai::utils::validate_max_output_tokens;
use crate::routes::output_log::LoggedOutput;
use crate::state::AppState;
use crate::utils::{resolve_request_context, sse_response};

fn error_body(error: ApiError) -> ErrorResponse {
    let kind = match error {
        ApiError::InvalidRequest { .. }
        | ApiError::JsonParseError { .. }
        | ApiError::InvalidQuery { .. } => ErrorType::InvalidRequestError,
        ApiError::ModelNotFound { .. } | ApiError::NotFound { .. } => ErrorType::NotFoundError,
        ApiError::Conflict { .. } => ErrorType::ConflictError,
        ApiError::ServiceUnavailable { .. } => ErrorType::OverloadedError,
        ApiError::ServerError { .. } => ErrorType::ApiError,
    };
    ErrorResponse {
        kind: ErrorResponseType::Error,
        error: ErrorDetail {
            kind,
            message: error.message(),
        },
    }
}

fn error_response(error: ApiError) -> Response {
    (error.status_code(), Json(error_body(error))).into_response()
}

pub(super) async fn count_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(body) => body.0,
        Err(error) => {
            tracing::error!("DEBUG_400 messages: {}", error);
            return error_response(error.into());
        }
    };
    let model = state.resolve_model_with_loras(Some(&body.model)).await;
    let request = match resolve_request_context(&headers, None) {
        Ok(ctx) => match prepare_messages_request(body, &model, ctx) {
            Ok(request) => request,
            Err(error) => return error_response(error),
        },
        Err(error) => return error_response(error),
    };
    let retry = request.chat_options.reasoning_effort.is_some().then(|| request.clone());
    let text = match state.chat.request_processor().prepare_for_tokenization(request).await {
        Ok(text) => text,
        Err(error) => {
            if let (true, Some(mut request)) =
                (error.to_string().contains("reasoning effort"), retry)
            {
                request.chat_options.reasoning_effort = None;
                match state.chat.request_processor().prepare_for_tokenization(request).await {
                    Ok(text) => text,
                    Err(error) => {
                        return error_response(chat_submit_error("count_tokens failed", error));
                    }
                }
            } else {
                return error_response(chat_submit_error("count_tokens failed", error));
            }
        }
    };
    let tokens = match state.chat.text().request_processor().tokenize(text) {
        Ok(tokens) => tokens,
        Err(error) => return error_response(text_submit_error("count_tokens failed", error)),
    };
    Json(CountTokensResponse {
        input_tokens: tokens.len(),
        context_management: ContextManagement {
            original_input_tokens: tokens.len(),
        },
    })
    .into_response()
}

/// Submits a chat request, retrying once with the engine-default reasoning
/// effort when the model's chat template rejects the requested one. Stock
/// Anthropic clients hardcode their default effort (Claude Code sends
/// `high`), which templates with a restricted effort set refuse at render
/// time; without this fallback such models are unusable out of the box.
async fn submit_chat_with_effort_fallback(
    state: &crate::state::AppState,
    request: ChatRequest,
) -> Result<vllm_chat::ChatEventStream, vllm_chat::Error> {
    let retry = request.chat_options.reasoning_effort.is_some().then(|| request.clone());
    match state.chat.chat(request).await {
        Ok(stream) => Ok(stream),
        Err(error) => {
            if let (true, Some(mut request)) =
                (error.to_string().contains("reasoning effort"), retry)
            {
                request.chat_options.reasoning_effort = None;
                tracing::info!("retrying /v1/messages with engine-default reasoning effort");
                state.chat.chat(request).await
            } else {
                Err(error)
            }
        }
    }
}

/// Anthropic request handling over the application's shared chat inference.
pub(super) async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(body) => body.0,
        Err(error) => {
            tracing::error!("DEBUG_400 messages: {}", error);
            return error_response(error.into());
        }
    };
    let model = state.resolve_model_with_loras(Some(&body.model)).await;
    let options = state.api_server_options;
    let streaming = body.stream == Some(true);
    if body.max_tokens.is_none() {
        return error_response(invalid_request!(
            param = "max_tokens",
            "max_tokens is required for generation"
        ));
    }
    let request = match resolve_request_context(&headers, None) {
        Ok(ctx) => match prepare_messages_request(body, &model, ctx) {
            Ok(request) => request,
            Err(error) => return error_response(error),
        },
        Err(error) => return error_response(error),
    };
    if let Err(error) = validate_max_output_tokens(
        "max_tokens",
        request.sampling_params.max_tokens,
        state.chat.text().llm().info().max_model_len,
    ) {
        return error_response(error);
    }
    let request_id = request.request_id.clone();
    if streaming {
        let span = tracing::info_span!(
            "messages_stream",
            request_id = %request_id,
            engine_request_id = tracing::field::Empty
        );
        let response = async {
            let stream = submit_chat_with_effort_fallback(&state, request)
                .await
                .map_err(|error| chat_submit_error("failed to submit Messages request", error))?;
            Ok::<_, ApiError>(sse_response(
                anthropic_sse_stream(stream, request_id, model.response_model()).instrument(span),
                options.sse_keep_alive_interval,
            ))
        }
        .await;
        return match response {
            Ok(response) => response,
            Err(error) => error_response(error),
        };
    }
    let span = tracing::info_span!("messages", request_id = %request_id, engine_request_id = tracing::field::Empty);
    let response = async {
        let response = submit_chat_with_effort_fallback(&state, request)
            .await
            .map_err(|error| chat_submit_error("failed to submit Messages request", error))?
            .collect_message()
            .await
            .map_err(|error| chat_submit_error("failed to collect Messages response", error))?;
        let (mut reason, sequence) = stop_reason(&response.finish_reason)?;
        let text = response.message.text();
        let content = convert_output_blocks(response.message.content)?;
        if content.iter().any(|block| {
            matches!(
                block,
                crate::routes::anthropic::types::OutputContentBlock::ToolUse { .. }
            )
        }) {
            reason = crate::routes::anthropic::types::MessagesStopReason::ToolUse;
        }
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
    .await;
    match response {
        Ok(response) => Json(response).into_response(),
        Err(error) => error_response(error),
    }
}

/// Convert the internal chat event stream into Anthropic Messages SSE events.
#[try_stream]
async fn anthropic_sse_stream(
    mut stream: impl ChatEventStreamTrait + Unpin,
    request_id: String,
    response_model: String,
    mut y: TryYielder<Event, Infallible>,
) -> Result<(), Infallible> {
    fn sse(kind: &str, payload: &serde_json::Value) -> Event {
        Event::default()
            .event(kind)
            .data(serde_json::to_string(payload).expect("SSE payload serializes"))
    }

    let mut saw_tool_use = false;
    let mut input_tokens = 0usize;

    fn error_event(error: &ApiError) -> Event {
        Event::default().event("error").data(
            serde_json::to_string(&serde_json::json!({
                "type": "error",
                "error": {"type": "api_error", "message": error.message()}
            }))
            .expect("SSE error payload serializes"),
        )
    }

    while let Some(next) = stream.next().await {
        let event = match next {
            Ok(event) => event,
            Err(error) => {
                let error = chat_submit_error("Messages stream failed", error);
                y.yield_ok(error_event(&error)).await;
                return Ok(());
            }
        };
        match event {
            ChatEvent::Start {
                prompt_token_ids, ..
            } => {
                input_tokens = prompt_token_ids.len();
                y.yield_ok(sse(
                    "message_start",
                    &json!({"type": "message_start",
                        "message": {
                            "id": request_id,
                            "type": "message",
                            "role": "assistant",
                            "model": response_model,
                            "content": [],
                            "stop_reason": null,
                            "stop_sequence": null,
                            "usage": {"input_tokens": prompt_token_ids.len(), "output_tokens": 1}
                        }
                    }),
                ))
                .await;
            }
            ChatEvent::BlockStart { index, kind } => {
                let content_block = match kind {
                    AssistantBlockKind::Reasoning => json!({"type": "thinking", "thinking": ""}),
                    AssistantBlockKind::Text => json!({"type": "text", "text": ""}),
                    AssistantBlockKind::ToolCall => continue,
                };
                y.yield_ok(sse("content_block_start", &json!({"type": "content_block_start", "index": index, "content_block": content_block
                })))
                .await;
            }
            ChatEvent::BlockDelta {
                index, kind, delta, ..
            } => {
                let delta = match kind {
                    AssistantBlockKind::Reasoning => {
                        json!({"type": "thinking_delta", "thinking": delta})
                    }
                    AssistantBlockKind::Text => json!({"type": "text_delta", "text": delta}),
                    AssistantBlockKind::ToolCall => continue,
                };
                y.yield_ok(sse(
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": index, "delta": delta
                    }),
                ))
                .await;
            }
            ChatEvent::BlockEnd { index, block } => {
                if matches!(block, AssistantContentBlock::Reasoning { .. }) {
                    y.yield_ok(sse(
                        "content_block_delta",
                        &json!({"type": "content_block_delta", "index": index,
                            "delta": {"type": "signature_delta",
                                      "signature": uuid::Uuid::new_v4().simple().to_string()}
                        }),
                    ))
                    .await;
                }
                y.yield_ok(sse(
                    "content_block_stop",
                    &json!({"type": "content_block_stop", "index": index
                    }),
                ))
                .await;
            }
            ChatEvent::LogprobsDelta { .. } => continue,
            ChatEvent::ToolCallStart { index, id, name } => {
                saw_tool_use = true;
                y.yield_ok(sse(
                    "content_block_start",
                    &json!({
                        "type": "content_block_start", "index": index,
                        "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}
                    }),
                ))
                .await;
            }
            ChatEvent::ToolCallArgumentsDelta { index, delta } => {
                y.yield_ok(sse(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": delta}
                    }),
                ))
                .await;
            }
            ChatEvent::ToolCallEnd { index, .. } => {
                y.yield_ok(sse(
                    "content_block_stop",
                    &json!({
                        "type": "content_block_stop", "index": index
                    }),
                ))
                .await;
            }
            ChatEvent::Done {
                usage,
                finish_reason,
                ..
            } => {
                let (mut reason, sequence) = match stop_reason(&finish_reason) {
                    Ok(pair) => pair,
                    Err(error) => {
                        y.yield_ok(error_event(&error)).await;
                        return Ok(());
                    }
                };
                if saw_tool_use {
                    reason = crate::routes::anthropic::types::MessagesStopReason::ToolUse;
                }
                y.yield_ok(sse("message_delta", &json!({"type": "message_delta",
                    "delta": {"stop_reason": reason.as_str(), "stop_sequence": sequence},
                    "usage": {
                        "output_tokens": usage.engine.output_token_count,
                        "input_tokens": input_tokens,
                        "cache_read_input_tokens": usage.engine.prompt_cache_usage.cached_token_count,
                        "cache_creation_input_tokens": usage.engine.prompt_cache_usage.cache_creation_token_count
                    }
                })))
                .await;
                y.yield_ok(sse(
                    "message_stop",
                    &json!({"type": "message_stop"
                    }),
                ))
                .await;
                break;
            }
        }
    }
    Ok(())
}
