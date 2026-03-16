use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt as _;
use openai_protocol::chat::ChatCompletionRequest;
use openai_protocol::validated::ValidatedJson;
use tracing::{error, info};
use vllm_chat::{ChatEvent, ChatEventStream};

use crate::convert::{final_chunk, prepare_chat_request, start_chunk, text_chunk};
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
            return ApiError::server_error(format!("failed to submit chat request: {error}"))
                .into_response();
        }
    };
    let sse_stream = event_stream(chat_stream, response_id, response_model, created);

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

/// Convert one internal chat event stream into OpenAI-style SSE events.
// TODO: refactor with `futures-async-stream`
fn event_stream(
    stream: ChatEventStream,
    response_id: String,
    response_model: String,
    created: u64,
) -> impl futures::Stream<Item = Result<Event, Infallible>> + Send {
    futures::stream::unfold(
        Some(SseState::Start {
            stream,
            response_id,
            response_model,
            created,
        }),
        |state| async move {
            let state = state?;
            next_sse_item(state).await
        },
    )
}

/// Small SSE state machine that injects the initial assistant-role chunk and trailing `[DONE]`.
enum SseState {
    Start {
        stream: ChatEventStream,
        response_id: String,
        response_model: String,
        created: u64,
    },
    Stream {
        stream: ChatEventStream,
        response_id: String,
        response_model: String,
        created: u64,
    },
    DoneMarker,
}

/// Advance the SSE state machine by one emitted event.
async fn next_sse_item(state: SseState) -> Option<(Result<Event, Infallible>, Option<SseState>)> {
    match state {
        SseState::Start {
            stream,
            response_id,
            response_model,
            created,
        } => {
            let chunk = start_chunk(&response_id, &response_model, created);
            Some((
                Ok(to_sse_event(&chunk)),
                Some(SseState::Stream {
                    stream,
                    response_id,
                    response_model,
                    created,
                }),
            ))
        }
        SseState::Stream {
            mut stream,
            response_id,
            response_model,
            created,
        } => loop {
            match stream.next().await {
                Some(Ok(ChatEvent::Start)) => continue,
                Some(Ok(ChatEvent::TextDelta { delta, .. })) => {
                    let chunk = text_chunk(&response_id, &response_model, created, delta);
                    return Some((
                        Ok(to_sse_event(&chunk)),
                        Some(SseState::Stream {
                            stream,
                            response_id,
                            response_model,
                            created,
                        }),
                    ));
                }
                Some(Ok(ChatEvent::Done {
                    finish_reason: Some(finish_reason),
                    stop_reason,
                    ..
                })) => match final_chunk(
                    &response_id,
                    &response_model,
                    created,
                    finish_reason,
                    stop_reason,
                ) {
                    Ok(chunk) => {
                        return Some((Ok(to_sse_event(&chunk)), Some(SseState::DoneMarker)));
                    }
                    Err(error) => {
                        error!(request_id = %response_id, ?error, "invalid terminal finish reason");
                        return None;
                    }
                },
                Some(Ok(ChatEvent::Done { .. })) => {
                    error!(request_id = %response_id, "missing terminal finish reason");
                    return None;
                }
                Some(Err(error)) => {
                    error!(request_id = %response_id, ?error, "chat stream failed");
                    return None;
                }
                None => return None,
            }
        },
        SseState::DoneMarker => Some((Ok(Event::default().data("[DONE]")), None)),
    }
}

/// Serialize one OpenAI chunk payload into one SSE `data:` event.
fn to_sse_event(chunk: &openai_protocol::chat::ChatCompletionStreamResponse) -> Event {
    let payload =
        serde_json::to_string(chunk).expect("ChatCompletionStreamResponse must serialize to JSON");
    Event::default().data(payload)
}
