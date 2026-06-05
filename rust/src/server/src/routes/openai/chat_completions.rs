pub mod convert;
mod types;
mod validate;

use std::convert::Infallible;
use std::pin::Pin;
use std::result::Result;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::future::try_join_all;
use futures::stream::SelectAll;
use futures::{Stream, StreamExt as _, pin_mut};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::{debug, error, info, trace};
use tracing_futures::Instrument as _;
use vllm_chat::{
    AssistantBlockKind, AssistantMessageExt as _, ChatEvent, ChatEventStream, ChatEventStreamTrait,
    CollectedAssistantMessage, FinishReason,
};
use vllm_engine_core_client::protocol::StopReason;

use crate::error::{ApiError, bail_server_error, server_error};
use crate::routes::openai::chat_completions::convert::prepare_chat_request;
use crate::routes::openai::chat_completions::types::{
    AssistantRole, ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStreamChoice, ChatCompletionStreamResponse,
    ChatMessageDelta,
};
use crate::routes::openai::utils::logprobs::{
    decoded_logprobs_to_openai_chat, decoded_prompt_logprobs_to_maps,
};
use crate::routes::openai::utils::types::{
    ChatLogProbs, FunctionCallDelta, FunctionCallResponse, ToolCall, ToolCallDelta, Usage,
};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

type ChatCompletionEventStream =
    Pin<Box<dyn Stream<Item = Result<ChatCompletionStreamEvent, ApiError>> + Send>>;

#[derive(Debug, Clone, Copy)]
struct ChoiceUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug)]
struct ChatChoiceOutput {
    choice: ChatCompletionChoice,
    usage: ChoiceUsage,
    prompt_logprobs: Option<Vec<Option<std::collections::HashMap<String, f32>>>>,
    prompt_token_ids: Option<Vec<u32>>,
    kv_transfer_params: Option<Value>,
}

#[derive(Debug)]
enum ChatCompletionStreamEvent {
    Chunk(ChatCompletionStreamResponse),
    Finished(ChoiceUsage),
}

/// Validate one chat completion request and proxy it into the shared
/// `vllm-chat` stack.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    let stream = body.stream;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(Some(&body.model)).await;

    let prepared = match prepare_chat_request(body, &lora_resolution, request_context) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    let request_span = tracing::info_span!(
        "chat_completions",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let created = unix_timestamp();
    let log_request = state.enable_log_requests;

    if stream {
        let mut event_streams: SelectAll<ChatCompletionEventStream> = SelectAll::new();
        for choice_index in 0..prepared.n {
            let chat_request = match child_chat_request(
                &prepared.chat_request,
                &prepared.request_id,
                choice_index,
                prepared.n,
            ) {
                Ok(request) => request,
                Err(error) => return error.into_response(),
            };
            let chat_stream =
                match state.chat.chat(chat_request).instrument(request_span.clone()).await {
                    Ok(stream) => stream,
                    Err(error) => {
                        return server_error!(
                            "failed to submit chat request: {}",
                            error.to_report_string()
                        )
                        .into_response();
                    }
                };
            event_streams.push(Box::pin(chat_completion_choice_event_stream(
                chat_stream,
                prepared.request_id.clone(),
                prepared.response_model.clone(),
                created,
                choice_index,
                log_request,
                prepared.requested_logprobs,
                prepared.echo.clone(),
                prepared.return_token_ids,
                prepared.return_tokens_as_token_ids,
            )));
        }

        let chunk_stream = chat_completion_parallel_chunk_stream(
            event_streams,
            prepared.request_id,
            prepared.response_model,
            created,
            prepared.include_usage,
        );
        let sse_stream = chat_completion_sse_stream(chunk_stream).instrument(request_span);

        Sse::new(sse_stream).into_response()
    } else {
        let mut choice_futures = Vec::with_capacity(prepared.n as usize);
        for choice_index in 0..prepared.n {
            let chat_request = match child_chat_request(
                &prepared.chat_request,
                &prepared.request_id,
                choice_index,
                prepared.n,
            ) {
                Ok(request) => request,
                Err(error) => return error.into_response(),
            };
            let chat_stream =
                match state.chat.chat(chat_request).instrument(request_span.clone()).await {
                    Ok(stream) => stream,
                    Err(error) => {
                        return server_error!(
                            "failed to submit chat request: {}",
                            error.to_report_string()
                        )
                        .into_response();
                    }
                };
            choice_futures.push(collect_chat_completion_choice(
                chat_stream,
                choice_index,
                prepared.requested_logprobs,
                prepared.include_prompt_logprobs,
                prepared.echo.clone(),
                prepared.return_token_ids,
                prepared.return_tokens_as_token_ids,
            ));
        }

        let choices = match try_join_all(choice_futures).instrument(request_span.clone()).await {
            Ok(choices) => choices,
            Err(error) => return error.into_response(),
        };

        let response = chat_completion_response_from_choices(
            prepared.request_id,
            prepared.response_model,
            created,
            choices,
        );

        if log_request {
            let usage = response.usage.as_ref();
            info!(
                parent: &request_span,
                model = %response.model,
                prompt_tokens = usage.map_or(0, |u| u.prompt_tokens),
                output_tokens = usage.and_then(|u| u.completion_tokens).unwrap_or(0),
                finish_reason = response.choices.first().and_then(|c| c.finish_reason.as_deref()).unwrap_or("unknown"),
                "chat completion finished"
            );
        }

        Json(response).into_response()
    }
}

fn child_chat_request(
    base: &vllm_chat::ChatRequest,
    parent_request_id: &str,
    choice_index: u32,
    choice_count: u32,
) -> Result<vllm_chat::ChatRequest, ApiError> {
    let mut request = base.clone();
    if choice_count > 1 {
        request.request_id = format!("{choice_index}_{parent_request_id}");
    } else {
        request.request_id = parent_request_id.to_string();
    }
    if let Some(seed) = request.sampling_params.seed {
        request.sampling_params.seed =
            Some(seed.checked_add(i64::from(choice_index)).ok_or_else(|| {
                ApiError::invalid_request(
                    "`seed + choice index` must fit within a signed 64-bit integer.".to_string(),
                    Some("seed"),
                )
            })?);
    }
    Ok(request)
}

async fn collect_chat_completion_choice(
    stream: ChatEventStream,
    choice_index: u32,
    requested_logprobs: bool,
    include_prompt_logprobs: bool,
    echo: Option<String>,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
) -> Result<ChatChoiceOutput, ApiError> {
    let collected = stream.collect_message().await.map_err(|error| {
        server_error!(
            "failed to collect chat completion response: {}",
            error.to_report_string()
        )
    })?;
    let CollectedAssistantMessage {
        message,
        prompt_token_count,
        prompt_token_ids,
        prompt_logprobs,
        logprobs,
        token_ids,
        output_token_count,
        finish_reason,
        kv_transfer_params,
    } = collected;
    let stop_reason = finish_reason.as_stop_reason().map(stop_reason_to_json);
    let saw_tool_calls = message.tool_calls().next().is_some();
    let finish_reason = chat_finish_reason_to_openai(&finish_reason, saw_tool_calls)?.to_string();
    let tool_calls = message
        .tool_calls()
        .map(|call| ToolCall {
            id: call.id.clone(),
            tool_type: "function".to_string(),
            function: FunctionCallResponse {
                name: call.name.clone(),
                arguments: Some(call.arguments.clone()),
            },
        })
        .collect::<Vec<_>>();
    let logprobs = if requested_logprobs {
        Some(decoded_logprobs_to_openai_chat(
            logprobs.as_ref().ok_or_else(|| {
                server_error!("chat response requested logprobs but generation returned none")
            })?,
            return_tokens_as_token_ids,
        )?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        Some(decoded_prompt_logprobs_to_maps(
            prompt_logprobs.as_ref().ok_or_else(|| {
                server_error!(
                    "chat response requested prompt_logprobs but generation returned none"
                )
            })?,
            return_tokens_as_token_ids,
        ))
    } else {
        None
    };
    Ok(ChatChoiceOutput {
        choice: ChatCompletionChoice {
            index: choice_index,
            message: ChatCompletionMessage {
                role: AssistantRole,
                content: match &echo {
                    Some(prefix) => Some(format!("{prefix}{}", message.text())),
                    None => Some(message.text()).filter(|t| !t.is_empty()),
                },
                tool_calls: Some(tool_calls).filter(|calls| !calls.is_empty()),
                reasoning: message.reasoning(),
            },
            logprobs,
            finish_reason: Some(finish_reason),
            stop_reason,
            token_ids: return_token_ids.then_some(token_ids),
        },
        usage: ChoiceUsage {
            prompt_tokens: prompt_token_count as u32,
            completion_tokens: output_token_count as u32,
        },
        prompt_logprobs,
        prompt_token_ids: return_token_ids.then(|| prompt_token_ids.to_vec()),
        kv_transfer_params,
    })
}

fn chat_completion_response_from_choices(
    request_id: String,
    response_model: String,
    created: u64,
    outputs: Vec<ChatChoiceOutput>,
) -> ChatCompletionResponse {
    let prompt_tokens = outputs.first().map_or(0, |output| output.usage.prompt_tokens);
    let completion_tokens = outputs.iter().map(|output| output.usage.completion_tokens).sum();
    let prompt_logprobs = outputs.iter().find_map(|output| output.prompt_logprobs.clone());
    let prompt_token_ids = outputs.iter().find_map(|output| output.prompt_token_ids.clone());
    let kv_transfer_params = outputs.iter().find_map(|output| output.kv_transfer_params.clone());

    ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: response_model,
        choices: outputs.into_iter().map(|output| output.choice).collect(),
        usage: Some(Usage::from_counts(prompt_tokens, completion_tokens)),
        system_fingerprint: None,
        prompt_logprobs,
        prompt_token_ids,
        kv_transfer_params,
    }
}

/// Convert one internal chat event stream into OpenAI chat-completion chunks.
#[try_stream]
async fn chat_completion_chunk_stream(
    stream: impl ChatEventStreamTrait + Unpin,
    request_id: String,
    response_model: String,
    created: u64,
    choice_index: u32,
    log_request: bool,
    include_usage: bool,
    requested_logprobs: bool,
    echo: Option<String>,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
    mut y: TryYielder<ChatCompletionStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    let events = chat_completion_choice_event_stream(
        stream,
        request_id.clone(),
        response_model.clone(),
        created,
        choice_index,
        log_request,
        requested_logprobs,
        echo,
        return_token_ids,
        return_tokens_as_token_ids,
    );
    pin_mut!(events);

    while let Some(next) = events.next().await {
        match next? {
            ChatCompletionStreamEvent::Chunk(chunk) => y.yield_ok(chunk).await,
            ChatCompletionStreamEvent::Finished(usage) => {
                if include_usage {
                    y.yield_ok(usage_chunk(
                        &request_id,
                        &response_model,
                        created,
                        Usage::from_counts(usage.prompt_tokens, usage.completion_tokens),
                    ))
                    .await;
                }
            }
        }
    }

    Ok(())
}

#[try_stream]
async fn chat_completion_parallel_chunk_stream(
    stream: impl Stream<Item = Result<ChatCompletionStreamEvent, ApiError>>,
    request_id: String,
    response_model: String,
    created: u64,
    include_usage: bool,
    mut y: TryYielder<ChatCompletionStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut prompt_tokens = None;
    let mut completion_tokens = 0_u32;

    while let Some(next) = stream.next().await {
        match next? {
            ChatCompletionStreamEvent::Chunk(chunk) => y.yield_ok(chunk).await,
            ChatCompletionStreamEvent::Finished(usage) => {
                prompt_tokens.get_or_insert(usage.prompt_tokens);
                completion_tokens = completion_tokens.saturating_add(usage.completion_tokens);
            }
        }
    }

    if include_usage {
        y.yield_ok(usage_chunk(
            &request_id,
            &response_model,
            created,
            Usage::from_counts(prompt_tokens.unwrap_or(0), completion_tokens),
        ))
        .await;
    }

    Ok(())
}

/// Convert one internal chat event stream into indexed OpenAI chat-completion events.
#[try_stream]
async fn chat_completion_choice_event_stream(
    mut stream: impl ChatEventStreamTrait + Unpin,
    request_id: String,
    response_model: String,
    created: u64,
    choice_index: u32,
    log_request: bool,
    requested_logprobs: bool,
    echo: Option<String>,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
    mut y: TryYielder<ChatCompletionStreamEvent, ApiError>,
) -> Result<(), ApiError> {
    let mut saw_tool_calls = false;

    // If the client requested logprobs or token_ids, we need to buffer chunks until
    // we receive the separate `LogprobsDelta` event, so that we can emit one
    // combined chunk with both the semantic delta and its per-update metadata.
    let mut pending_chunk =
        (requested_logprobs || return_token_ids).then(PendingChatChunk::default);

    while let Some(next) = stream.next().await {
        match next {
            Ok(ChatEvent::Start {
                prompt_token_ids, ..
            }) => {
                let mut chunk = start_chunk(&request_id, &response_model, created, choice_index);
                if return_token_ids {
                    chunk.prompt_token_ids = Some(prompt_token_ids.to_vec());
                }
                y.yield_ok(ChatCompletionStreamEvent::Chunk(chunk)).await;
                // When echo=true, emit the last assistant message content as a delta chunk.
                if let Some(echo_text) = &echo {
                    y.yield_ok(ChatCompletionStreamEvent::Chunk(block_delta_chunk(
                        &request_id,
                        &response_model,
                        created,
                        choice_index,
                        AssistantBlockKind::Text,
                        echo_text.clone(),
                    )))
                    .await;
                }
            }
            Ok(ChatEvent::BlockDelta { kind, delta, .. }) => {
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.push_block_delta(kind, delta);
                } else {
                    y.yield_ok(ChatCompletionStreamEvent::Chunk(block_delta_chunk(
                        &request_id,
                        &response_model,
                        created,
                        choice_index,
                        kind,
                        delta,
                    )))
                    .await;
                }
            }
            Ok(ChatEvent::LogprobsDelta {
                logprobs,
                token_ids,
            }) => {
                let openai_logprobs = logprobs
                    .as_ref()
                    .map(|lp| decoded_logprobs_to_openai_chat(lp, return_tokens_as_token_ids))
                    .transpose()?;
                let openai_token_ids =
                    return_token_ids.then_some(token_ids).filter(|t| !t.is_empty());
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.logprobs = openai_logprobs;
                    pending_chunk.token_ids = openai_token_ids;
                    if let Some(chunk) = pending_chunk.take_chunk(
                        &request_id,
                        &response_model,
                        created,
                        choice_index,
                    ) {
                        y.yield_ok(ChatCompletionStreamEvent::Chunk(chunk)).await;
                    }
                } else if let Some(logprobs) = openai_logprobs {
                    y.yield_ok(ChatCompletionStreamEvent::Chunk(logprobs_only_chunk(
                        &request_id,
                        &response_model,
                        created,
                        choice_index,
                        logprobs,
                    )))
                    .await;
                }
            }
            Ok(ChatEvent::BlockStart { kind, .. }) => {
                debug!(?kind, "starting new block");
            }
            Ok(ChatEvent::BlockEnd { .. }) => {
                debug!("ending current block");
            }
            Ok(ChatEvent::ToolCallStart { index, id, name }) => {
                let tool_index = index as u32;
                saw_tool_calls = true;
                debug!(
                    tool_call_id = %id,
                    tool_call_name = %name,
                    "starting new tool call"
                );
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.push_tool_call_start(tool_index, id, name);
                } else {
                    y.yield_ok(ChatCompletionStreamEvent::Chunk(tool_call_start_chunk(
                        &request_id,
                        &response_model,
                        created,
                        choice_index,
                        tool_index,
                        id,
                        name,
                    )))
                    .await;
                }
            }
            Ok(ChatEvent::ToolCallArgumentsDelta { index, delta }) => {
                let tool_index = index as u32;
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.push_tool_call_arguments(tool_index, delta);
                } else {
                    y.yield_ok(ChatCompletionStreamEvent::Chunk(tool_call_arguments_chunk(
                        &request_id,
                        &response_model,
                        created,
                        choice_index,
                        tool_index,
                        delta,
                    )))
                    .await;
                }
            }
            Ok(ChatEvent::ToolCallEnd { .. }) => {
                debug!("ending current tool call");
            }
            Ok(ChatEvent::Done {
                prompt_token_count,
                finish_reason,
                output_token_count,
                ..
            }) => {
                if log_request {
                    info!(
                        stream = true,
                        model = %response_model,
                        prompt_tokens = prompt_token_count,
                        output_tokens = output_token_count,
                        finish_reason = finish_reason.as_str(),
                        "chat completion finished"
                    );
                }

                if let Some(pending_chunk) = pending_chunk.as_mut()
                    && let Some(chunk) = pending_chunk.take_chunk(
                        &request_id,
                        &response_model,
                        created,
                        choice_index,
                    )
                {
                    y.yield_ok(ChatCompletionStreamEvent::Chunk(chunk)).await;
                }

                match final_chunk(
                    &request_id,
                    &response_model,
                    created,
                    choice_index,
                    finish_reason,
                    saw_tool_calls,
                ) {
                    Ok(chunk) => y.yield_ok(ChatCompletionStreamEvent::Chunk(chunk)).await,
                    Err(error) => {
                        error!(
                            error = %error.to_error_response().error.message,
                            "invalid terminal finish reason"
                        );
                        return Err(error);
                    }
                }

                y.yield_ok(ChatCompletionStreamEvent::Finished(ChoiceUsage {
                    prompt_tokens: prompt_token_count as u32,
                    completion_tokens: output_token_count as u32,
                }))
                .await;

                return Ok(());
            }
            Err(error) => {
                error!(
                    error = %error.as_report(),
                    "chat stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }
    Ok(())
}

fn usage_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    usage: Usage,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.usage = Some(usage);
    chunk
}

/// One in-flight chat-completions SSE chunk being assembled at the route layer.
///
/// `vllm-chat` emits semantic chat events first and `LogprobsDelta` separately,
/// because one decoded update may be rewritten into multiple chat events.
/// The OpenAI chat API, though, wants one streamed chunk to optionally carry
/// both the delta and its logprobs.
///
/// This small buffer accumulates the semantic delta first, then attaches the
/// following `LogprobsDelta` and flushes one combined chunk. It relies on the
/// current `vllm-chat` invariant that all semantic events from one decoded
/// update are emitted before that update's `LogprobsDelta`.
#[derive(Debug, Default)]
struct PendingChatChunk {
    /// The currently buffered OpenAI delta payload assembled from one or more
    /// chat semantic events belonging to the same decoded update.
    delta: ChatMessageDelta,
    /// The token-aligned logprobs for that same decoded update.
    logprobs: Option<ChatLogProbs>,
    /// Per-update output token IDs for the same decoded update.
    token_ids: Option<Vec<u32>>,
}

impl PendingChatChunk {
    /// Append one assistant text/reasoning block delta to the buffered OpenAI
    /// delta payload.
    fn push_block_delta(&mut self, kind: AssistantBlockKind, delta: String) {
        match kind {
            AssistantBlockKind::Text => append_delta_text(&mut self.delta.content, delta),
            AssistantBlockKind::Reasoning => append_delta_text(&mut self.delta.reasoning, delta),
            AssistantBlockKind::ToolCall => {
                unreachable!("tool calls must flow through dedicated tool-call chunks")
            }
        }
    }

    /// Append the OpenAI tool-call-start representation to the buffered delta.
    fn push_tool_call_start(&mut self, index: u32, id: String, name: String) {
        self.delta.tool_calls.get_or_insert_with(Vec::new).push(ToolCallDelta {
            index,
            id: Some(id),
            tool_type: Some("function".to_string()),
            function: Some(FunctionCallDelta {
                name: Some(name),
                arguments: None,
            }),
        });
    }

    /// Append one incremental tool-call arguments update to the buffered delta.
    fn push_tool_call_arguments(&mut self, index: u32, delta: String) {
        self.delta.tool_calls.get_or_insert_with(Vec::new).push(ToolCallDelta {
            index,
            id: None,
            tool_type: None,
            function: Some(FunctionCallDelta {
                name: None,
                arguments: Some(delta),
            }),
        });
    }

    /// Finalize the currently buffered SSE chunk, if it contains either a
    /// semantic delta or a logprobs payload.
    ///
    /// This may produce:
    /// - a combined delta + logprobs chunk
    /// - a delta-only chunk
    /// - a logprobs-only chunk
    ///
    /// The logprobs-only case is intentional: token-level metadata in one
    /// decoded update is correlated with the same update boundary, not
    /// necessarily with a visible/chat-semantic delta.
    fn take_chunk(
        &mut self,
        request_id: &str,
        response_model: &str,
        created: u64,
        choice_index: u32,
    ) -> Option<ChatCompletionStreamResponse> {
        let has_delta = self.delta.content.is_some()
            || self.delta.reasoning.is_some()
            || self.delta.tool_calls.is_some();
        let logprobs = self.logprobs.take();
        let token_ids = self.token_ids.take();
        if !has_delta && logprobs.is_none() && token_ids.is_none() {
            return None;
        }

        let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
        chunk.choices.push(ChatCompletionStreamChoice {
            index: choice_index,
            delta: self.take_delta(),
            logprobs,
            token_ids,
            ..Default::default()
        });
        Some(chunk)
    }

    /// Take the currently buffered OpenAI delta payload and leave this pending
    /// chunk empty for the next decoded update.
    fn take_delta(&mut self) -> ChatMessageDelta {
        ChatMessageDelta {
            role: self.delta.role.take(),
            content: self.delta.content.take(),
            tool_calls: self.delta.tool_calls.take(),
            reasoning: self.delta.reasoning.take(),
        }
    }
}

/// Append one text fragment to an optional OpenAI delta string field.
fn append_delta_text(slot: &mut Option<String>, delta: String) {
    match slot {
        Some(existing) => existing.push_str(&delta),
        None => *slot = Some(delta),
    }
}

/// Convert one chunk stream into OpenAI-style SSE events.
///
/// OpenAI-style streaming errors are encoded as ordinary `data: {"error": ...}`
/// events followed by `data: [DONE]`, so the transport stream itself stays
/// infallible even when generation fails after the HTTP response has started.
#[try_stream]
async fn chat_completion_sse_stream(
    stream: impl Stream<Item = Result<ChatCompletionStreamResponse, ApiError>>,
    mut y: TryYielder<Event, Infallible>,
) -> Result<(), Infallible> {
    pin_mut!(stream);

    while let Some(next) = stream.next().await {
        match next {
            Ok(chunk) => y.yield_ok(to_sse_event(&chunk)).await,
            Err(error) => {
                y.yield_ok(to_error_sse_event(&error)).await;
                break;
            }
        }
    }

    y.yield_ok(done_sse_event()).await;
    Ok(())
}

/// Serialize one OpenAI chunk payload into one SSE `data:` event.
fn to_sse_event(chunk: &ChatCompletionStreamResponse) -> Event {
    let payload =
        serde_json::to_string(chunk).expect("ChatCompletionStreamResponse must serialize to JSON");
    trace!(payload, "chat completion emitting chunk");
    Event::default().data(payload)
}

/// Serialize one OpenAI error payload into one SSE `data:` event.
fn to_error_sse_event(error: &ApiError) -> Event {
    let payload = serde_json::to_string(&error.to_error_response())
        .expect("ErrorResponse must serialize to JSON");
    trace!(payload, "chat completion emitting error");
    Event::default().data(payload)
}

/// Build the terminal OpenAI SSE sentinel event.
fn done_sse_event() -> Event {
    trace!("chat completion emitting done");
    Event::default().data("[DONE]")
}

/// Build the initial assistant-role SSE chunk required by the OpenAI streaming
/// protocol.
fn start_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    choice_index: u32,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        index: choice_index,
        delta: ChatMessageDelta {
            role: Some(AssistantRole),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

/// Build one content-delta SSE chunk from one internal assistant block delta.
fn block_delta_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    choice_index: u32,
    kind: AssistantBlockKind,
    delta: String,
) -> ChatCompletionStreamResponse {
    let delta = match kind {
        AssistantBlockKind::Text => ChatMessageDelta {
            content: Some(delta),
            ..Default::default()
        },
        AssistantBlockKind::Reasoning => ChatMessageDelta {
            reasoning: Some(delta),
            ..Default::default()
        },
        AssistantBlockKind::ToolCall => {
            unreachable!("tool calls must flow through dedicated tool-call chunks")
        }
    };

    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        index: choice_index,
        delta,
        ..Default::default()
    });
    chunk
}

fn tool_call_start_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    choice_index: u32,
    tool_index: u32,
    id: String,
    name: String,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        index: choice_index,
        delta: ChatMessageDelta {
            tool_calls: Some(vec![ToolCallDelta {
                index: tool_index,
                id: Some(id),
                tool_type: Some("function".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some(name),
                    arguments: None,
                }),
            }]),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

fn tool_call_arguments_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    choice_index: u32,
    tool_index: u32,
    delta: String,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        index: choice_index,
        delta: ChatMessageDelta {
            tool_calls: Some(vec![ToolCallDelta {
                index: tool_index,
                id: None,
                tool_type: None,
                function: Some(FunctionCallDelta {
                    name: None,
                    arguments: Some(delta),
                }),
            }]),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

fn logprobs_only_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    choice_index: u32,
    logprobs: ChatLogProbs,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        index: choice_index,
        logprobs: Some(logprobs),
        ..Default::default()
    });
    chunk
}

/// Build the terminal SSE chunk carrying the OpenAI finish reason.
fn final_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    choice_index: u32,
    finish_reason: FinishReason,
    saw_tool_calls: bool,
) -> Result<ChatCompletionStreamResponse, ApiError> {
    let stop_reason = finish_reason.as_stop_reason().map(stop_reason_to_json);
    let finish_reason = chat_finish_reason_to_openai(&finish_reason, saw_tool_calls)?;

    debug!(
        finish_reason = %finish_reason,
        stop_reason = ?stop_reason,
        "chat stream finished"
    );

    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        index: choice_index,
        finish_reason: Some(finish_reason.to_string()),
        stop_reason,
        ..Default::default()
    });
    Ok(chunk)
}

fn chat_finish_reason_to_openai(
    finish_reason: &FinishReason,
    saw_tool_calls: bool,
) -> Result<&'static str, ApiError> {
    match finish_reason {
        FinishReason::Stop(_) if saw_tool_calls => Ok("tool_calls"),
        FinishReason::Stop(_) => Ok("stop"),
        FinishReason::Length => Ok("length"),
        FinishReason::Abort => Ok("abort"),
        FinishReason::Repetition => Ok("stop"),
        FinishReason::Error => {
            bail_server_error!("Internal server error");
        }
    }
}

/// Convert one internal stop reason into the OpenAI-compatible `stop_reason`
/// JSON shape.
fn stop_reason_to_json(stop_reason: &StopReason) -> Value {
    serde_json::to_value(stop_reason).expect("StopReason must serialize to JSON")
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt as _, stream};
    use serde_json::json;
    use vllm_chat::{AssistantBlockKind, AssistantToolCall, ChatEvent, FinishReason};
    use vllm_engine_core_client::protocol::StopReason;
    use vllm_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob};

    use super::{block_delta_chunk, chat_completion_chunk_stream, final_chunk};

    #[test]
    fn text_chunk_uses_content_only_delta() {
        let chunk = block_delta_chunk(
            "chatcmpl-1",
            "model",
            1,
            0,
            AssistantBlockKind::Text,
            "hello".to_string(),
        );
        assert_eq!(chunk.choices[0].delta.role, None);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
        assert_eq!(chunk.choices[0].delta.reasoning, None);
    }

    #[test]
    fn reasoning_chunk_uses_reasoning_only_delta() {
        let chunk = block_delta_chunk(
            "chatcmpl-1",
            "model",
            1,
            0,
            AssistantBlockKind::Reasoning,
            "thinking".to_string(),
        );
        assert_eq!(chunk.choices[0].delta.role, None);
        assert_eq!(chunk.choices[0].delta.content, None);
        assert_eq!(
            chunk.choices[0].delta.reasoning.as_deref(),
            Some("thinking")
        );
    }

    #[test]
    fn final_chunk_maps_stop_finish_reason_and_stop_reason() {
        let chunk = final_chunk(
            "chatcmpl-1",
            "model",
            1,
            0,
            FinishReason::Stop(Some(StopReason::Text("stop".to_string()))),
            false,
        )
        .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunk.choices[0].stop_reason, Some(json!("stop")));
    }

    #[test]
    fn final_chunk_maps_length_finish_reason() {
        let chunk = final_chunk("chatcmpl-1", "model", 1, 0, FinishReason::Length, false)
            .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
        assert_eq!(chunk.choices[0].stop_reason, None);
    }

    #[test]
    fn final_chunk_maps_abort_finish_reason() {
        let chunk = final_chunk("chatcmpl-1", "model", 1, 0, FinishReason::Abort, false)
            .expect("abort is a valid finish reason");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("abort"));
        assert_eq!(chunk.choices[0].stop_reason, None);
    }

    #[test]
    fn final_chunk_rejects_error_finish_reason() {
        assert!(final_chunk("chatcmpl-1", "model", 1, 0, FinishReason::Error, false).is_err());
    }

    #[test]
    fn final_chunk_maps_stop_to_tool_calls_when_tool_calls_were_streamed() {
        let chunk = final_chunk("chatcmpl-1", "model", 1, 0, FinishReason::stop_eos(), true)
            .expect("finish reason is valid");

        assert_eq!(
            chunk.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[tokio::test]
    async fn chunk_stream_coalesces_text_delta_with_logprobs() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::BlockStart {
                index: 0,
                kind: AssistantBlockKind::Text,
            }),
            Ok(ChatEvent::BlockDelta {
                index: 0,
                kind: AssistantBlockKind::Text,
                delta: "hi".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "hi".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![],
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                prompt_token_count: 1,
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            0,
            false,
            false,
            true,
            None,
            false,
            false,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("hi"));
        let logprobs = chunks[1].choices[0].logprobs.as_ref().expect("logprobs");
        let content = logprobs.content.as_ref().expect("logprobs content");
        assert_eq!(content[0].token, "hi");
    }

    #[tokio::test]
    async fn chunk_stream_coalesces_reasoning_delta_with_logprobs() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::BlockStart {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
            }),
            Ok(ChatEvent::BlockDelta {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
                delta: "think".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "think".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![],
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                prompt_token_count: 1,
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            0,
            false,
            false,
            true,
            None,
            false,
            false,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks[1].choices[0].delta.reasoning.as_deref(),
            Some("think")
        );
        assert!(chunks[1].choices[0].logprobs.is_some());
    }

    #[tokio::test]
    async fn chunk_stream_preserves_tool_call_index_and_omits_id_from_arguments_delta() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::ToolCallStart {
                index: 3,
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
            }),
            Ok(ChatEvent::ToolCallArgumentsDelta {
                index: 3,
                delta: r#"{"city":"Paris"}"#.to_string(),
            }),
            Ok(ChatEvent::ToolCallEnd {
                index: 3,
                call: AssistantToolCall {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"Paris"}"#.to_string(),
                },
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                prompt_token_count: 1,
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            0,
            false,
            false,
            false,
            None,
            false,
            false,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(
            chunks[1].choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
            3
        );
        assert_eq!(
            chunks[1].choices[0].delta.tool_calls.as_ref().unwrap()[0].id,
            Some("call_1".to_string())
        );
        assert_eq!(
            chunks[2].choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
            3
        );
        assert_eq!(
            chunks[2].choices[0].delta.tool_calls.as_ref().unwrap()[0].id,
            None
        );
    }
}
