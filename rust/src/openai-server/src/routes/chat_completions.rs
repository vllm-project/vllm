use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use openai_protocol::chat::{
    ChatCompletionRequest, ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice,
};
use openai_protocol::validated::ValidatedJson;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::{error, info};
use vllm_chat::{ChatEvent, ChatEventStream};
use vllm_engine_core_client::protocol::{FinishReason, StopReason};

use crate::convert::prepare_chat_request;
use crate::error::ApiError;
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
    info!(request_id = %response_id, model = %response_model, "streaming chat completion");

    let chat_stream = match state.chat.chat(prepared.chat_request).await {
        Ok(stream) => stream,
        Err(error) => {
            return ApiError::server_error(format!(
                "failed to submit chat request: {}",
                error.to_report_string()
            ))
            .into_response();
        }
    };
    let chunk_stream =
        chat_completion_chunk_stream(chat_stream, response_id, response_model, created);
    let sse_stream = chat_completion_sse_stream(chunk_stream);

    Sse::new(sse_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Return the current Unix timestamp in seconds for OpenAI response objects.
fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

/// Convert one internal chat event stream into OpenAI chat-completion chunks.
#[try_stream(ok = ChatCompletionStreamResponse, error = ApiError)]
async fn chat_completion_chunk_stream(
    mut stream: ChatEventStream,
    response_id: String,
    response_model: String,
    created: u64,
) {
    while let Some(next) = stream.next().await {
        match next {
            Ok(ChatEvent::Start) => yield start_chunk(&response_id, &response_model, created),
            Ok(ChatEvent::TextDelta { delta, .. }) => {
                yield text_chunk(&response_id, &response_model, created, delta)
            }
            Ok(ChatEvent::Done {
                finish_reason: Some(finish_reason),
                stop_reason,
                ..
            }) => match final_chunk(
                &response_id,
                &response_model,
                created,
                finish_reason,
                stop_reason,
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
            },
            Ok(ChatEvent::Done { .. }) => {
                let error = ApiError::server_error(
                    "chat stream terminated without a terminal finish reason",
                );
                error!(request_id = %response_id, "missing terminal finish reason");
                return Err(error);
            }
            Err(error) => {
                error!(
                    request_id = %response_id,
                    error = %error.as_report(),
                    "chat stream failed"
                );
                return Err(ApiError::server_error(error.to_report_string()));
            }
        }
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
fn to_sse_event(chunk: &openai_protocol::chat::ChatCompletionStreamResponse) -> Event {
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

/// Build one content-delta SSE chunk from one internal text delta.
fn text_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
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
                content: Some(delta),
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

/// Build the terminal SSE chunk carrying the OpenAI finish reason.
fn final_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    finish_reason: FinishReason,
    stop_reason: Option<StopReason>,
) -> Result<ChatCompletionStreamResponse, ApiError> {
    let finish_reason = match finish_reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Repetition => "stop",
        FinishReason::Abort | FinishReason::Error => {
            return Err(ApiError::server_error(
                "stream terminated without a valid OpenAI finish reason",
            ));
        }
    };

    let matched_stop = stop_reason.map(stop_reason_to_json);

    Ok(
        ChatCompletionStreamResponse::builder(response_id.to_string(), response_model.to_string())
            .created(created)
            .add_choice_finish_reason(0, finish_reason, matched_stop)
            .build(),
    )
}

/// Convert one internal stop reason into the OpenAI-compatible `matched_stop` JSON shape.
fn stop_reason_to_json(stop_reason: StopReason) -> Value {
    serde_json::to_value(stop_reason).expect("StopReason must serialize to JSON")
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vllm_engine_core_client::protocol::{FinishReason, StopReason};

    use super::{final_chunk, text_chunk};

    #[test]
    fn text_chunk_uses_content_only_delta() {
        let chunk = text_chunk("chatcmpl-1", "model", 1, "hello".to_string());
        assert_eq!(chunk.choices[0].delta.role, None);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
    }

    #[test]
    fn final_chunk_maps_stop_finish_reason_and_matched_stop() {
        let chunk = final_chunk(
            "chatcmpl-1",
            "model",
            1,
            FinishReason::Stop,
            Some(StopReason::Text("stop".to_string())),
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
        )
        .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
        assert_eq!(chunk.choices[0].matched_stop, Some(json!(42)));
    }

    #[test]
    fn final_chunk_rejects_abort_and_error_finish_reasons() {
        assert!(final_chunk("chatcmpl-1", "model", 1, FinishReason::Abort, None).is_err());
        assert!(final_chunk("chatcmpl-1", "model", 1, FinishReason::Error, None).is_err());
    }
}
