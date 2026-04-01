mod convert;
mod types;
mod validate;

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use openai_protocol::common::LogProbs;
use openai_protocol::validated::ValidatedJson;
use thiserror_ext::AsReport as _;
use tracing::{debug, error, info};
use vllm_text::{DecodedTextEvent, FinishReason, TextOutputStream, TextOutputStreamExt as _};

use super::utils::logprobs::{
    collected_logprobs_to_openai, decoded_logprobs_to_openai, decoded_prompt_logprobs_to_maps,
    text_len,
};
use super::utils::types::Usage;
use super::utils::unix_timestamp;
use crate::error::{ApiError, bail_server_error, server_error};
use crate::routes::completions::convert::prepare_completion_request;
use crate::routes::completions::types::{
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionSseChunk,
    CompletionStreamChoice, CompletionStreamResponse,
};
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
    let include_prompt_logprobs = prepared
        .text_request
        .sampling_params
        .prompt_logprobs
        .is_some();
    info!(
        request_id = %response_id,
        model = %response_model,
        stream = body.stream,
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

    if body.stream {
        let chunk_stream = completion_chunk_stream(
            text_stream,
            response_id,
            response_model,
            created,
            prepared.include_usage,
            echo,
            body.logprobs,
        );
        let sse_stream = completion_sse_stream(chunk_stream);

        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let response = match collect_completion(
            text_stream,
            response_id,
            response_model,
            created,
            echo,
            body.logprobs,
            include_prompt_logprobs,
        )
        .await
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
    requested_logprobs: Option<u32>,
    include_prompt_logprobs: bool,
) -> Result<CompletionResponse, ApiError> {
    let collected = stream
        .collect_output()
        .await
        .map_err(|error| server_error!("completion stream failed: {}", error.to_report_string()))?;
    let finish_reason = collected.finish_reason.clone();
    let stop_reason = finish_reason
        .as_stop_reason()
        .map(|sr| serde_json::to_value(sr).expect("StopReason must serialize to JSON"));

    let prompt_char_count = echo
        .as_ref()
        .map(|prompt| text_len(prompt))
        .unwrap_or_default();
    let prompt_logprobs = if include_prompt_logprobs {
        let prompt_logprobs = collected.prompt_logprobs.as_ref().ok_or_else(|| {
            server_error!(
                "completion response requested prompt_logprobs but generation returned none"
            )
        })?;
        Some(prompt_logprobs)
    } else {
        None
    };
    let logprobs = if requested_logprobs.is_some() {
        Some(collected_logprobs_to_openai(
            &collected,
            echo.is_some(),
            prompt_char_count,
        )?)
    } else {
        None
    };
    let prompt_logprobs = prompt_logprobs.map(decoded_prompt_logprobs_to_maps);
    let text = match &echo {
        None => collected.text,
        Some(prompt) => format!("{prompt}{}", collected.text),
    };

    Ok(CompletionResponse {
        id: response_id,
        object: "text_completion".to_string(),
        created,
        model: response_model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            logprobs,
            finish_reason: Some(completion_finish_reason_to_openai(finish_reason)?.into()),
            stop_reason,
            prompt_logprobs,
        }],
        usage: Some(Usage::from_counts(
            collected.prompt_token_count as u32,
            collected.token_ids.len() as u32,
        )),
        system_fingerprint: None,
    })
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
    requested_logprobs: Option<u32>,
) {
    pin_mut!(stream);
    let mut visible_text_len = 0_u32;

    while let Some(next) = stream.next().await {
        match next {
            Ok(DecodedTextEvent::Start { .. }) => {
                debug!(request_id = %response_id, "completion stream started");
                if let Some(prompt) = echo.as_ref() {
                    visible_text_len = text_len(prompt);
                    yield CompletionSseChunk::Chunk(delta_chunk(
                        &response_id,
                        &response_model,
                        created,
                        prompt.clone(),
                        None,
                    ));
                }
            }
            Ok(DecodedTextEvent::TextDelta {
                delta,
                logprobs,
                finished,
                ..
            }) => {
                let delta_text_len = text_len(&delta);
                let logprobs = if requested_logprobs.is_some() {
                    let decoded_logprobs = logprobs.as_ref().ok_or_else(|| {
                        server_error!(
                            "completion stream requested logprobs but generation returned none"
                        )
                    })?;
                    Some(decoded_logprobs_to_openai(
                        decoded_logprobs,
                        visible_text_len,
                    )?)
                } else {
                    None
                };
                yield CompletionSseChunk::Chunk(delta_chunk(
                    &response_id,
                    &response_model,
                    created,
                    delta,
                    logprobs,
                ));
                visible_text_len = visible_text_len.saturating_add(delta_text_len);

                if let Some(finished) = finished {
                    yield CompletionSseChunk::Chunk(final_chunk(
                        &response_id,
                        &response_model,
                        created,
                        finished.finish_reason,
                    )?);

                    if include_usage {
                        yield CompletionSseChunk::Usage(usage_chunk(
                            &response_id,
                            &response_model,
                            created,
                            Usage::from_counts(
                                finished.prompt_token_count as u32,
                                finished.output_token_count as u32,
                            ),
                        ));
                    }
                }
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
    logprobs: Option<LogProbs>,
) -> CompletionStreamResponse {
    let mut chunk = CompletionStreamResponse::new(response_id, response_model, created);
    chunk.choices.push(CompletionStreamChoice {
        text,
        logprobs,
        ..Default::default()
    });
    chunk
}

fn final_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    finish_reason: FinishReason,
) -> Result<CompletionStreamResponse, ApiError> {
    let finish_reason = completion_finish_reason_to_openai(finish_reason)?;

    let mut chunk = CompletionStreamResponse::new(response_id, response_model, created);
    chunk.choices.push(CompletionStreamChoice {
        finish_reason: Some(finish_reason.to_string()),
        ..Default::default()
    });
    Ok(chunk)
}

fn completion_finish_reason_to_openai(
    finish_reason: FinishReason,
) -> Result<&'static str, ApiError> {
    match finish_reason {
        FinishReason::Stop(_) | FinishReason::Repetition => Ok("stop"),
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
) -> CompletionStreamResponse {
    let mut chunk = CompletionStreamResponse::new(response_id, response_model, created);
    chunk.usage = Some(usage);
    chunk
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
    use futures::{StreamExt as _, stream};
    use vllm_text::{
        DecodedLogprobs, DecodedPositionLogprobs, DecodedTextEvent, DecodedTokenLogprob,
        FinishReason, Finished,
    };

    use super::{CompletionSseChunk, completion_chunk_stream, final_chunk};

    #[test]
    fn final_chunk_maps_stop_finish_reason() {
        let chunk = final_chunk("cmpl-1", "model", 1, FinishReason::stop_eos())
            .expect("finish reason valid");
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

    #[tokio::test]
    async fn completion_chunk_stream_maps_streaming_logprobs() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_count: 5,
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "h".to_string(),
                token_ids: vec![b'h' as u32],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![
                            DecodedTokenLogprob {
                                token: "h".to_string(),
                                logprob: -0.1,
                                rank: 1,
                            },
                            DecodedTokenLogprob {
                                token: "H".to_string(),
                                logprob: -0.2,
                                rank: 1,
                            },
                        ],
                    }],
                }),
                finished: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: String::new(),
                token_ids: vec![b'!' as u32],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![
                            DecodedTokenLogprob {
                                token: "!".to_string(),
                                logprob: -0.3,
                                rank: 1,
                            },
                            DecodedTokenLogprob {
                                token: "?".to_string(),
                                logprob: -0.4,
                                rank: 1,
                            },
                        ],
                    }],
                }),
                finished: Some(Finished {
                    prompt_token_count: 5,
                    output_token_count: 2,
                    finish_reason: FinishReason::stop_eos(),
                }),
            }),
        ]);

        let chunks = completion_chunk_stream(
            stream,
            "cmpl-1".to_string(),
            "model".to_string(),
            1,
            false,
            None,
            Some(1),
        )
        .collect::<Vec<_>>()
        .await;

        let chunks = chunks
            .into_iter()
            .try_collect::<Vec<_>>()
            .expect("stream should succeed");

        match &chunks[0] {
            CompletionSseChunk::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].text, "h");
                assert_eq!(
                    chunk.choices[0].logprobs.as_ref().expect("logprobs").tokens,
                    vec!["h".to_string()]
                );
                assert_eq!(
                    chunk.choices[0]
                        .logprobs
                        .as_ref()
                        .expect("logprobs")
                        .text_offset,
                    vec![0]
                );
            }
            CompletionSseChunk::Usage(_) => panic!("expected regular chunk"),
        }

        match &chunks[1] {
            CompletionSseChunk::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].text, "");
                assert_eq!(
                    chunk.choices[0].logprobs.as_ref().expect("logprobs").tokens,
                    vec!["!".to_string()]
                );
                assert_eq!(
                    chunk.choices[0]
                        .logprobs
                        .as_ref()
                        .expect("logprobs")
                        .text_offset,
                    vec![1]
                );
            }
            CompletionSseChunk::Usage(_) => panic!("expected regular chunk"),
        }
    }
}
