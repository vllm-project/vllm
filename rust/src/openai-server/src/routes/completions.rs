mod convert;

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use openai_protocol::common::Usage;
use openai_protocol::completion::{
    CompletionChoice, CompletionResponse, CompletionStreamChoice, CompletionStreamResponse,
};
use openai_protocol::validated::ValidatedJson;
use serde::Serialize;
use thiserror_ext::AsReport as _;
use tracing::{debug, error, info};
use vllm_engine_core_client::protocol::FinishReason;
use vllm_text::{DecodedTextEvent, TextOutputStream, TextOutputStreamExt as _};

use super::utils::unix_timestamp;
use crate::error::{ApiError, bail_server_error, server_error};
use crate::routes::completions::convert::{CompletionRequest, prepare_completion_request};
use crate::state::AppState;

/// Validate one completions request and proxy it into the shared `vllm-text` stack.
pub(super) async fn completions(
    State(state): State<Arc<AppState>>,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Response {
    let prepared = match prepare_completion_request(&body, &state.model_id) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };

    let response_id = prepared.response_id.clone();
    let response_model = prepared.response_model.clone();
    let created = unix_timestamp();
    let echo = prepared.echo.clone();
    info!(
        request_id = %response_id,
        model = %response_model,
        stream = body.inner.stream,
        "completion"
    );

    let text_stream = match state.chat.text().generate(prepared.text_request).await {
        Ok(stream) => stream,
        Err(error) => {
            return server_error!(
                "failed to submit completion request: {}",
                error.to_report_string()
            )
            .into_response();
        }
    };

    if body.inner.stream {
        let chunk_stream = completion_chunk_stream(
            text_stream,
            response_id,
            response_model,
            created,
            prepared.include_usage,
            echo,
        );
        let sse_stream = completion_sse_stream(chunk_stream);

        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let response =
            match collect_completion(text_stream, response_id, response_model, created, echo).await
            {
                Ok(response) => response,
                Err(error) => return error.into_response(),
            };

        Json(response).into_response()
    }
}

async fn collect_completion(
    stream: impl TextOutputStream,
    response_id: String,
    response_model: String,
    created: u64,
    echo: Option<String>,
) -> Result<CompletionResponse, ApiError> {
    let collected = stream
        .collect_output()
        .await
        .map_err(|error| server_error!("completion stream failed: {}", error.to_report_string()))?;
    let finish_reason = collected.finish_reason.ok_or_else(|| {
        server_error!("completion stream terminated without a terminal finish reason")
    })?;

    let text = match echo {
        None => collected.text,
        Some(prompt) => format!("{prompt}{}", collected.text),
    };

    Ok(CompletionResponse {
        id: response_id,
        object: "text_completion".to_string(),
        created,
        model: response_model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: Some(completion_finish_reason_to_openai(finish_reason)?.into()),
            matched_stop: None,
        }],
        usage: Some(Usage::from_counts(
            collected.prompt_token_count,
            collected.token_ids.len() as u32,
        )),
        system_fingerprint: None,
    })
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum CompletionSseChunk {
    /// Ordinary OpenAI completions delta/final chunk.
    Chunk(CompletionStreamResponse),
    /// Final usage chunk emitted before `[DONE]` when `include_usage=true`.
    Usage(CompletionUsageChunk),
}

/// Minimal JSON shape for the extra streamed usage chunk used by `vllm-bench`.
///
/// `openai_protocol::completion::CompletionStreamResponse` does not currently include a `usage`
/// field, so the Rust server emits this thin local wrapper for the terminal accounting event.
#[derive(Debug, Clone, Serialize)]
struct CompletionUsageChunk {
    id: String,
    object: String,
    created: u64,
    choices: Vec<CompletionStreamChoice>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_fingerprint: Option<String>,
    usage: Usage,
}

/// Convert one internal decoded-text stream into OpenAI completions chunks.
#[try_stream(ok = CompletionSseChunk, error = ApiError)]
async fn completion_chunk_stream(
    stream: impl TextOutputStream,
    response_id: String,
    response_model: String,
    created: u64,
    include_usage: bool,
    echo: Option<String>,
) {
    pin_mut!(stream);

    while let Some(next) = stream.next().await {
        match next {
            Ok(DecodedTextEvent::Start) => {
                debug!(request_id = %response_id, "completion stream started");
                if let Some(prompt) = echo.as_ref() {
                    yield CompletionSseChunk::Chunk(delta_chunk(
                        &response_id,
                        &response_model,
                        created,
                        prompt.clone(),
                    ));
                }
            }
            Ok(DecodedTextEvent::TextDelta { delta, .. }) => {
                yield CompletionSseChunk::Chunk(delta_chunk(
                    &response_id,
                    &response_model,
                    created,
                    delta,
                ));
            }
            Ok(DecodedTextEvent::Done {
                prompt_token_count,
                finish_reason: Some(finish_reason),
                token_ids,
                ..
            }) => {
                yield CompletionSseChunk::Chunk(final_chunk(
                    &response_id,
                    &response_model,
                    created,
                    finish_reason,
                )?);

                if include_usage {
                    yield CompletionSseChunk::Usage(usage_chunk(
                        &response_id,
                        &response_model,
                        created,
                        Usage::from_counts(prompt_token_count, token_ids.len() as u32),
                    ));
                }
            }
            Ok(DecodedTextEvent::Done { .. }) => {
                let error =
                    server_error!("completion stream terminated without a terminal finish reason");
                error!(request_id = %response_id, "missing terminal finish reason");
                return Err(error);
            }
            Err(error) => {
                error!(
                    request_id = %response_id,
                    error = %error.as_report(),
                    "completion stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }
}

fn delta_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    text: String,
) -> CompletionStreamResponse {
    CompletionStreamResponse {
        id: response_id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: response_model.to_string(),
        system_fingerprint: None,
        choices: vec![CompletionStreamChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: None,
        }],
    }
}

fn final_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    finish_reason: FinishReason,
) -> Result<CompletionStreamResponse, ApiError> {
    // Match the chat route's finish-reason policy so engine-native abort/error termination still
    // becomes an OpenAI-style streamed error rather than an invalid terminal chunk.
    let finish_reason = completion_finish_reason_to_openai(finish_reason)?;

    Ok(CompletionStreamResponse {
        id: response_id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: response_model.to_string(),
        system_fingerprint: None,
        choices: vec![CompletionStreamChoice {
            text: String::new(),
            index: 0,
            logprobs: None,
            finish_reason: Some(finish_reason.to_string()),
        }],
    })
}

fn completion_finish_reason_to_openai(
    finish_reason: FinishReason,
) -> Result<&'static str, ApiError> {
    match finish_reason {
        FinishReason::Stop | FinishReason::Repetition => Ok("stop"),
        FinishReason::Length => Ok("length"),
        FinishReason::Abort | FinishReason::Error => {
            bail_server_error!("stream terminated without a valid OpenAI finish reason");
        }
    }
}

fn usage_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    usage: Usage,
) -> CompletionUsageChunk {
    CompletionUsageChunk {
        id: response_id.to_string(),
        object: "text_completion".to_string(),
        created,
        choices: Vec::new(),
        model: response_model.to_string(),
        system_fingerprint: None,
        usage,
    }
}

/// Convert one chunk stream into OpenAI-style SSE events.
///
/// OpenAI-style streaming errors are encoded as ordinary `data: {"error": ...}`
/// events followed by `data: [DONE]`, so the transport stream itself stays
/// infallible even when generation fails after the HTTP response has started.
#[try_stream(ok = Event, error = Infallible)]
async fn completion_sse_stream(stream: impl Stream<Item = Result<CompletionSseChunk, ApiError>>) {
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
fn to_sse_event(chunk: &CompletionSseChunk) -> Event {
    let payload = serde_json::to_string(chunk).expect("completion chunk must serialize to JSON");
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

#[cfg(test)]
mod tests {
    use vllm_engine_core_client::protocol::FinishReason;

    use super::final_chunk;

    #[test]
    fn final_chunk_maps_stop_finish_reason() {
        let chunk =
            final_chunk("cmpl-1", "model", 1, FinishReason::Stop).expect("finish reason valid");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunk.choices[0].text, "");
    }

    #[test]
    fn final_chunk_maps_length_finish_reason() {
        let chunk =
            final_chunk("cmpl-1", "model", 1, FinishReason::Length).expect("finish reason valid");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn final_chunk_rejects_abort_and_error_finish_reasons() {
        assert!(final_chunk("cmpl-1", "model", 1, FinishReason::Abort).is_err());
        assert!(final_chunk("cmpl-1", "model", 1, FinishReason::Error).is_err());
    }
}
