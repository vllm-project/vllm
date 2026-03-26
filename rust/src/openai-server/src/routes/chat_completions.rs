pub mod convert;
mod types;
mod validate;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use openai_protocol::chat::{
    ChatChoice, ChatCompletionMessage, ChatCompletionStreamResponse, ChatMessageDelta,
    ChatStreamChoice,
};
use openai_protocol::common::{
    ChatLogProbs, FunctionCallDelta, FunctionCallResponse, ToolCall, ToolCallDelta, Usage,
};
use openai_protocol::validated::ValidatedJson;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::{debug, error, info};
use vllm_chat::{
    AssistantBlockKind, AssistantMessageExt as _, ChatEvent, ChatEventStream, ChatEventStreamTrait,
};
use vllm_engine_core_client::protocol::{FinishReason, StopReason};

use crate::error::{ApiError, bail_server_error, server_error};
use crate::routes::chat_completions::convert::prepare_chat_request;
use crate::routes::chat_completions::types::{ChatCompletionRequest, ChatCompletionResponse};
use crate::routes::utils::logprobs::{
    decoded_logprobs_to_openai_chat, decoded_prompt_logprobs_to_maps,
};
use crate::routes::utils::unix_timestamp;
use crate::state::AppState;

/// Validate one chat completion request and proxy it into the shared `vllm-chat` stack.
pub(super) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    let prepared = match prepare_chat_request(&body, &state.model_id) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };

    let response_id = prepared.response_id.clone();
    let response_model = prepared.response_model.clone();
    let created = unix_timestamp();
    info!(
        request_id = %response_id,
        model = %response_model,
        stream = body.stream,
        "chat completion"
    );

    let chat_stream = match state.chat.chat(prepared.chat_request).await {
        Ok(stream) => stream,
        Err(error) => {
            return server_error!(
                "failed to submit chat request: {}",
                error.to_report_string()
            )
            .into_response();
        }
    };

    if body.stream {
        let chunk_stream = chat_completion_chunk_stream(
            chat_stream,
            response_id,
            response_model,
            created,
            prepared.include_usage,
            prepared.requested_logprobs,
        );
        let sse_stream = chat_completion_sse_stream(chunk_stream);

        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let response = match collect_chat_completion(
            chat_stream,
            response_id,
            response_model,
            created,
            prepared.requested_logprobs,
            prepared.include_prompt_logprobs,
        )
        .await
        {
            Ok(response) => response,
            Err(error) => return error.into_response(),
        };

        Json(response).into_response()
    }
}

async fn collect_chat_completion(
    stream: ChatEventStream,
    response_id: String,
    response_model: String,
    created: u64,
    requested_logprobs: bool,
    include_prompt_logprobs: bool,
) -> Result<ChatCompletionResponse, ApiError> {
    let collected = stream.collect_message().await.map_err(|error| {
        server_error!(
            "failed to collect chat completion response: {}",
            error.to_report_string()
        )
    })?;
    let finish_reason = collected
        .finish_reason
        .ok_or_else(|| server_error!("chat stream terminated without a terminal finish reason"))?;
    let saw_tool_calls = collected.message.tool_calls().next().is_some();
    let finish_reason = chat_finish_reason_to_openai(finish_reason, saw_tool_calls)?.to_string();
    let matched_stop = collected.stop_reason.map(stop_reason_to_json);
    let tool_calls = collected
        .message
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
            collected.logprobs.as_ref().ok_or_else(|| {
                server_error!("chat response requested logprobs but generation returned none")
            })?,
        )?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        Some(decoded_prompt_logprobs_to_maps(
            collected.prompt_logprobs.as_ref().ok_or_else(|| {
                server_error!(
                    "chat response requested prompt_logprobs but generation returned none"
                )
            })?,
        ))
    } else {
        None
    };
    let usage = Usage::from_counts(
        collected.prompt_token_count as u32,
        collected.token_ids.len() as u32,
    );

    Ok(ChatCompletionResponse {
        id: response_id,
        object: "chat.completion".to_string(),
        created,
        model: response_model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: Some(collected.message.text()).filter(|text| !text.is_empty()),
                tool_calls: Some(tool_calls).filter(|calls| !calls.is_empty()),
                reasoning_content: collected.message.reasoning(),
            },
            logprobs,
            finish_reason: Some(finish_reason),
            matched_stop,
            hidden_states: None,
        }],
        usage: Some(usage),
        system_fingerprint: None,
        prompt_logprobs,
    })
}

/// Convert one internal chat event stream into OpenAI chat-completion chunks.
#[try_stream(ok = ChatCompletionStreamResponse, error = ApiError)]
async fn chat_completion_chunk_stream(
    mut stream: impl ChatEventStreamTrait + Unpin,
    response_id: String,
    response_model: String,
    created: u64,
    include_usage: bool,
    requested_logprobs: bool,
) {
    let mut tool_call_indices = BTreeMap::<String, u32>::new();

    // If the client requested logprobs, we need to buffer chunks until we receive the separate
    // `LogprobsDelta` event, so that we can emit one combined chunk with both the semantic delta
    // and its logprobs.
    let mut pending_chunk = requested_logprobs.then(PendingChatChunk::default);

    while let Some(next) = stream.next().await {
        match next {
            Ok(ChatEvent::Start { .. }) => {
                yield start_chunk(&response_id, &response_model, created)
            }
            Ok(ChatEvent::BlockDelta { kind, delta, .. }) => {
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.push_block_delta(kind, delta);
                } else {
                    yield block_delta_chunk(&response_id, &response_model, created, kind, delta)
                }
            }
            Ok(ChatEvent::LogprobsDelta { logprobs }) => {
                let logprobs = decoded_logprobs_to_openai_chat(&logprobs)?;
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.logprobs = Some(logprobs);
                    if let Some(chunk) =
                        pending_chunk.take_chunk(&response_id, &response_model, created)
                    {
                        yield chunk;
                    }
                } else {
                    yield logprobs_only_chunk(&response_id, &response_model, created, logprobs);
                }
            }
            Ok(ChatEvent::BlockStart { kind, .. }) => {
                debug!(request_id = %response_id, ?kind, "starting new block");
            }
            Ok(ChatEvent::BlockEnd { .. }) => {
                debug!(request_id = %response_id, "ending current block");
            }
            Ok(ChatEvent::ToolCallStart { id, name, .. }) => {
                let tool_index = tool_call_indices.len() as u32;
                tool_call_indices.insert(id.clone(), tool_index);
                debug!(
                    request_id = %response_id,
                    tool_call_id = %id,
                    tool_call_name = %name,
                    "starting new tool call"
                );
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.push_tool_call_start(tool_index, id, name);
                } else {
                    yield tool_call_start_chunk(
                        &response_id,
                        &response_model,
                        created,
                        tool_index,
                        id,
                        name,
                    );
                }
            }
            Ok(ChatEvent::ToolCallArgumentsDelta { id, delta, .. }) => {
                let Some(&tool_index) = tool_call_indices.get(&id) else {
                    error!(request_id = %response_id, tool_call_id = %id, "missing tool call index");
                    bail_server_error!("tool call stream state is inconsistent");
                };
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.push_tool_call_arguments(tool_index, id, delta);
                } else {
                    yield tool_call_arguments_chunk(
                        &response_id,
                        &response_model,
                        created,
                        tool_index,
                        id,
                        delta,
                    );
                }
            }
            Ok(ChatEvent::ToolCallEnd { .. }) => {
                debug!(request_id = %response_id, "ending current tool call");
            }
            Ok(ChatEvent::Done {
                prompt_token_count,
                finish_reason: Some(finish_reason),
                stop_reason,
                token_ids,
                ..
            }) => {
                if let Some(pending_chunk) = pending_chunk.as_mut()
                    && let Some(chunk) =
                        pending_chunk.take_chunk(&response_id, &response_model, created)
                {
                    yield chunk;
                }

                match final_chunk(
                    &response_id,
                    &response_model,
                    created,
                    finish_reason,
                    stop_reason,
                    !tool_call_indices.is_empty(),
                ) {
                    Ok(chunk) => yield chunk,
                    Err(error) => {
                        error!(
                            request_id = %response_id,
                            error = %error.to_error_response().error.message,
                            "invalid terminal finish reason"
                        );
                        return Err(error);
                    }
                }

                if include_usage {
                    let completion_tokens = token_ids.len() as u32;
                    yield usage_chunk(
                        &response_id,
                        &response_model,
                        created,
                        Usage::from_counts(prompt_token_count as u32, completion_tokens),
                    );
                }

                return Ok(());
            }
            Ok(ChatEvent::Done { .. }) => {
                let error =
                    server_error!("chat stream terminated without a terminal finish reason",);
                error!(request_id = %response_id, "missing terminal finish reason");
                return Err(error);
            }
            Err(error) => {
                error!(
                    request_id = %response_id,
                    error = %error.as_report(),
                    "chat stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }
}

fn usage_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    usage: Usage,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: response_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: response_model.to_string(),
        system_fingerprint: None,
        choices: Vec::new(),
        usage: Some(usage),
    }
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
#[derive(Debug)]
struct PendingChatChunk {
    /// The currently buffered OpenAI delta payload assembled from one or more
    /// chat semantic events belonging to the same decoded update.
    delta: ChatMessageDelta,
    /// The token-aligned logprobs for that same decoded update, filled only
    /// once [`ChatEvent::LogprobsDelta`] arrives.
    logprobs: Option<ChatLogProbs>,
}

impl Default for PendingChatChunk {
    fn default() -> Self {
        Self {
            delta: ChatMessageDelta {
                role: None,
                content: None,
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
        }
    }
}

impl PendingChatChunk {
    /// Append one assistant text/reasoning block delta to the buffered OpenAI delta payload.
    fn push_block_delta(&mut self, kind: AssistantBlockKind, delta: String) {
        match kind {
            AssistantBlockKind::Text => append_delta_text(&mut self.delta.content, delta),
            AssistantBlockKind::Reasoning => {
                append_delta_text(&mut self.delta.reasoning_content, delta)
            }
            AssistantBlockKind::ToolCall => {
                unreachable!("tool calls must flow through dedicated tool-call chunks")
            }
        }
    }

    /// Append the OpenAI tool-call-start representation to the buffered delta.
    fn push_tool_call_start(&mut self, index: u32, id: String, name: String) {
        self.delta
            .tool_calls
            .get_or_insert_with(Vec::new)
            .push(ToolCallDelta {
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
    fn push_tool_call_arguments(&mut self, index: u32, id: String, delta: String) {
        self.delta
            .tool_calls
            .get_or_insert_with(Vec::new)
            .push(ToolCallDelta {
                index,
                id: Some(id),
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
        response_id: &str,
        response_model: &str,
        created: u64,
    ) -> Option<ChatCompletionStreamResponse> {
        let has_delta = self.delta.content.is_some()
            || self.delta.reasoning_content.is_some()
            || self.delta.tool_calls.is_some();
        let logprobs = self.logprobs.take();
        if !has_delta && logprobs.is_none() {
            return None;
        }

        Some(ChatCompletionStreamResponse {
            id: response_id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: response_model.to_string(),
            system_fingerprint: None,
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: self.take_delta(),
                logprobs,
                finish_reason: None,
                matched_stop: None,
            }],
            usage: None,
        })
    }

    /// Take the currently buffered OpenAI delta payload and leave this pending
    /// chunk empty for the next decoded update.
    fn take_delta(&mut self) -> ChatMessageDelta {
        ChatMessageDelta {
            role: self.delta.role.take(),
            content: self.delta.content.take(),
            tool_calls: self.delta.tool_calls.take(),
            reasoning_content: self.delta.reasoning_content.take(),
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
#[try_stream(ok = Event, error = Infallible)]
async fn chat_completion_sse_stream(
    stream: impl Stream<Item = Result<ChatCompletionStreamResponse, ApiError>>,
) {
    pin_mut!(stream);

    while let Some(next) = stream.next().await {
        match next {
            Ok(chunk) => yield to_sse_event(&chunk),
            Err(error) => {
                yield to_error_sse_event(&error);
                break;
            }
        }
    }

    yield done_sse_event();
}

/// Serialize one OpenAI chunk payload into one SSE `data:` event.
fn to_sse_event(chunk: &ChatCompletionStreamResponse) -> Event {
    let payload =
        serde_json::to_string(chunk).expect("ChatCompletionStreamResponse must serialize to JSON");
    Event::default().data(payload)
}

/// Serialize one OpenAI error payload into one SSE `data:` event.
fn to_error_sse_event(error: &ApiError) -> Event {
    let payload = serde_json::to_string(&error.to_error_response())
        .expect("ErrorResponse must serialize to JSON");
    Event::default().data(payload)
}

/// Build the terminal OpenAI SSE sentinel event.
fn done_sse_event() -> Event {
    Event::default().data("[DONE]")
}

/// Build the initial assistant-role SSE chunk required by the OpenAI streaming protocol.
fn start_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: response_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: response_model.to_string(),
        system_fingerprint: None,
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatMessageDelta {
                role: Some("assistant".to_string()),
                content: None,
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        }],
        usage: None,
    }
}

/// Build one content-delta SSE chunk from one internal assistant block delta.
fn block_delta_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    kind: AssistantBlockKind,
    delta: String,
) -> ChatCompletionStreamResponse {
    let delta = match kind {
        AssistantBlockKind::Text => ChatMessageDelta {
            role: None,
            content: Some(delta),
            tool_calls: None,
            reasoning_content: None,
        },
        AssistantBlockKind::Reasoning => ChatMessageDelta {
            role: None,
            content: None,
            tool_calls: None,
            reasoning_content: Some(delta),
        },
        AssistantBlockKind::ToolCall => {
            unreachable!("tool calls must flow through dedicated tool-call chunks")
        }
    };

    ChatCompletionStreamResponse {
        id: response_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: response_model.to_string(),
        system_fingerprint: None,
        choices: vec![ChatStreamChoice {
            index: 0,
            delta,
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        }],
        usage: None,
    }
}

fn tool_call_start_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    tool_index: u32,
    id: String,
    name: String,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: response_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: response_model.to_string(),
        system_fingerprint: None,
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatMessageDelta {
                role: None,
                content: None,
                tool_calls: Some(vec![ToolCallDelta {
                    index: tool_index,
                    id: Some(id),
                    tool_type: Some("function".to_string()),
                    function: Some(FunctionCallDelta {
                        name: Some(name),
                        arguments: None,
                    }),
                }]),
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        }],
        usage: None,
    }
}

fn tool_call_arguments_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    tool_index: u32,
    id: String,
    delta: String,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: response_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: response_model.to_string(),
        system_fingerprint: None,
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatMessageDelta {
                role: None,
                content: None,
                tool_calls: Some(vec![ToolCallDelta {
                    index: tool_index,
                    id: Some(id),
                    tool_type: None,
                    function: Some(FunctionCallDelta {
                        name: None,
                        arguments: Some(delta),
                    }),
                }]),
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        }],
        usage: None,
    }
}

fn logprobs_only_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    logprobs: ChatLogProbs,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: response_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: response_model.to_string(),
        system_fingerprint: None,
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatMessageDelta {
                role: None,
                content: None,
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: Some(logprobs),
            finish_reason: None,
            matched_stop: None,
        }],
        usage: None,
    }
}

/// Build the terminal SSE chunk carrying the OpenAI finish reason.
fn final_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    finish_reason: FinishReason,
    stop_reason: Option<StopReason>,
    saw_tool_calls: bool,
) -> Result<ChatCompletionStreamResponse, ApiError> {
    let finish_reason = chat_finish_reason_to_openai(finish_reason, saw_tool_calls)?;

    debug!(
        request_id = %response_id,
        finish_reason = %finish_reason,
        stop_reason = ?stop_reason,
        "chat stream finished"
    );

    let matched_stop = stop_reason.map(stop_reason_to_json);

    Ok(
        ChatCompletionStreamResponse::builder(response_id.to_string(), response_model.to_string())
            .created(created)
            .add_choice_finish_reason(0, finish_reason, matched_stop)
            .build(),
    )
}

fn chat_finish_reason_to_openai(
    finish_reason: FinishReason,
    saw_tool_calls: bool,
) -> Result<&'static str, ApiError> {
    match finish_reason {
        FinishReason::Stop if saw_tool_calls => Ok("tool_calls"),
        FinishReason::Stop => Ok("stop"),
        FinishReason::Length => Ok("length"),
        FinishReason::Repetition => Ok("stop"),
        FinishReason::Abort | FinishReason::Error => {
            bail_server_error!("stream terminated without a valid OpenAI finish reason");
        }
    }
}

/// Convert one internal stop reason into the OpenAI-compatible `matched_stop` JSON shape.
fn stop_reason_to_json(stop_reason: StopReason) -> Value {
    serde_json::to_value(stop_reason).expect("StopReason must serialize to JSON")
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt as _, stream};
    use openai_protocol::common::ChatLogProbs;
    use serde_json::json;
    use vllm_chat::{AssistantBlockKind, ChatEvent};
    use vllm_engine_core_client::protocol::{FinishReason, StopReason};
    use vllm_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob};

    use super::{block_delta_chunk, chat_completion_chunk_stream, final_chunk};

    #[test]
    fn text_chunk_uses_content_only_delta() {
        let chunk = block_delta_chunk(
            "chatcmpl-1",
            "model",
            1,
            AssistantBlockKind::Text,
            "hello".to_string(),
        );
        assert_eq!(chunk.choices[0].delta.role, None);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
        assert_eq!(chunk.choices[0].delta.reasoning_content, None);
    }

    #[test]
    fn reasoning_chunk_uses_reasoning_content_only_delta() {
        let chunk = block_delta_chunk(
            "chatcmpl-1",
            "model",
            1,
            AssistantBlockKind::Reasoning,
            "thinking".to_string(),
        );
        assert_eq!(chunk.choices[0].delta.role, None);
        assert_eq!(chunk.choices[0].delta.content, None);
        assert_eq!(
            chunk.choices[0].delta.reasoning_content.as_deref(),
            Some("thinking")
        );
    }

    #[test]
    fn final_chunk_maps_stop_finish_reason_and_matched_stop() {
        let chunk = final_chunk(
            "chatcmpl-1",
            "model",
            1,
            FinishReason::Stop,
            Some(StopReason::Text("stop".to_string())),
            false,
        )
        .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunk.choices[0].matched_stop, Some(json!("stop")));
    }

    #[test]
    fn final_chunk_maps_length_finish_reason() {
        let chunk = final_chunk(
            "chatcmpl-1",
            "model",
            1,
            FinishReason::Length,
            Some(StopReason::TokenId(42)),
            false,
        )
        .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
        assert_eq!(chunk.choices[0].matched_stop, Some(json!(42)));
    }

    #[test]
    fn final_chunk_rejects_abort_and_error_finish_reasons() {
        assert!(final_chunk("chatcmpl-1", "model", 1, FinishReason::Abort, None, false).is_err());
        assert!(final_chunk("chatcmpl-1", "model", 1, FinishReason::Error, None, false).is_err());
    }

    #[test]
    fn final_chunk_maps_stop_to_tool_calls_when_tool_calls_were_streamed() {
        let chunk = final_chunk("chatcmpl-1", "model", 1, FinishReason::Stop, None, true)
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
                prompt_token_count: 1,
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
                logprobs: DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "hi".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                },
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                prompt_token_count: 1,
                token_ids: vec![1],
                finish_reason: Some(FinishReason::Stop),
                stop_reason: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            false,
            true,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("hi"));
        match chunks[1].choices[0].logprobs.as_ref().expect("logprobs") {
            ChatLogProbs::Detailed {
                content: Some(content),
            } => {
                assert_eq!(content[0].token, "hi");
            }
            other => panic!("unexpected chat logprobs shape: {other:?}"),
        }
    }

    #[tokio::test]
    async fn chunk_stream_coalesces_reasoning_delta_with_logprobs() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_count: 1,
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
                logprobs: DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token: "think".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                },
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                prompt_token_count: 1,
                token_ids: vec![1],
                finish_reason: Some(FinishReason::Stop),
                stop_reason: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            false,
            true,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks[1].choices[0].delta.reasoning_content.as_deref(),
            Some("think")
        );
        assert!(chunks[1].choices[0].logprobs.is_some());
    }
}
