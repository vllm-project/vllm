mod convert;
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
use vllm_text::{DecodedTextEvent, FinishReason, TextOutputStream, TextOutputStreamExt as _};

use super::utils::logprobs::{
    collected_logprobs_to_openai, decoded_logprobs_to_openai, decoded_prompt_logprobs_to_maps,
    text_len,
};
use super::utils::types::Usage;
use crate::error::{ApiError, bail_server_error, server_error};
use crate::routes::openai::completions::convert::prepare_completion_request;
use crate::routes::openai::completions::types::{
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionSseChunk,
    CompletionStreamChoice, CompletionStreamResponse,
};
use crate::routes::openai::utils::types::LogProbs;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

type CompletionEventStream =
    Pin<Box<dyn Stream<Item = Result<CompletionStreamEvent, ApiError>> + Send>>;

#[derive(Debug, Clone, Copy)]
struct ChoiceUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug)]
struct CompletionChoiceOutput {
    choice: CompletionChoice,
    usage: ChoiceUsage,
    kv_transfer_params: Option<Value>,
}

#[derive(Debug)]
enum CompletionStreamEvent {
    Chunk(CompletionStreamResponse),
    Finished(ChoiceUsage),
}

/// Validate one completions request and proxy it into the shared `vllm-text`
/// stack.
pub async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Response {
    let stream = body.stream;
    let logprobs = body.logprobs;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(Some(&body.model)).await;

    let prepared = match prepare_completion_request(body, &lora_resolution, request_context) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    let request_span = tracing::info_span!(
        "completions",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let created = unix_timestamp();
    let include_prompt_logprobs = prepared.text_request.sampling_params.prompt_logprobs.is_some();
    let log_request = state.enable_log_requests;

    if stream {
        let mut event_streams: SelectAll<CompletionEventStream> = SelectAll::new();
        for choice_index in 0..prepared.n {
            let text_request = match child_text_request(
                &prepared.text_request,
                &prepared.request_id,
                choice_index,
                prepared.n,
            ) {
                Ok(request) => request,
                Err(error) => return error.into_response(),
            };
            let text_stream = match state
                .chat
                .text()
                .generate(text_request)
                .instrument(request_span.clone())
                .await
            {
                Ok(stream) => stream,
                Err(error) => {
                    return server_error!(
                        "failed to submit completion request: {}",
                        error.to_report_string()
                    )
                    .into_response();
                }
            };
            event_streams.push(Box::pin(completion_choice_event_stream(
                text_stream,
                prepared.request_id.clone(),
                prepared.response_model.clone(),
                created,
                choice_index,
                log_request,
                prepared.echo.clone(),
                logprobs,
                prepared.return_token_ids,
                prepared.return_tokens_as_token_ids,
            )));
        }

        let chunk_stream = completion_parallel_chunk_stream(
            event_streams,
            prepared.request_id,
            prepared.response_model,
            created,
            prepared.include_usage,
        );
        let sse_stream = completion_sse_stream(chunk_stream).instrument(request_span);

        Sse::new(sse_stream).into_response()
    } else {
        let mut choice_futures = Vec::with_capacity(prepared.n as usize);
        for choice_index in 0..prepared.n {
            let text_request = match child_text_request(
                &prepared.text_request,
                &prepared.request_id,
                choice_index,
                prepared.n,
            ) {
                Ok(request) => request,
                Err(error) => return error.into_response(),
            };
            let text_stream = match state
                .chat
                .text()
                .generate(text_request)
                .instrument(request_span.clone())
                .await
            {
                Ok(stream) => stream,
                Err(error) => {
                    return server_error!(
                        "failed to submit completion request: {}",
                        error.to_report_string()
                    )
                    .into_response();
                }
            };
            choice_futures.push(collect_completion_choice(
                text_stream,
                choice_index,
                prepared.echo.clone(),
                logprobs,
                include_prompt_logprobs,
                prepared.return_token_ids,
                prepared.return_tokens_as_token_ids,
            ));
        }

        let choices = match try_join_all(choice_futures).instrument(request_span.clone()).await {
            Ok(choices) => choices,
            Err(error) => return error.into_response(),
        };

        let response = completion_response_from_choices(
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
                "completion finished"
            );
        }

        Json(response).into_response()
    }
}

fn child_text_request(
    base: &vllm_text::TextRequest,
    parent_request_id: &str,
    choice_index: u32,
    choice_count: u32,
) -> Result<vllm_text::TextRequest, ApiError> {
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

async fn collect_completion_choice(
    stream: impl TextOutputStream,
    choice_index: u32,
    echo: Option<String>,
    requested_logprobs: Option<u32>,
    include_prompt_logprobs: bool,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
) -> Result<CompletionChoiceOutput, ApiError> {
    let collected = stream
        .collect_output()
        .await
        .map_err(|error| server_error!("completion stream failed: {}", error.to_report_string()))?;
    let finish_reason = collected.finish_reason.clone();
    let stop_reason = finish_reason
        .as_stop_reason()
        .map(|sr| serde_json::to_value(sr).expect("StopReason must serialize to JSON"));

    let prompt_char_count = echo.as_ref().map(|prompt| text_len(prompt)).unwrap_or_default();
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
            return_tokens_as_token_ids,
        )?)
    } else {
        None
    };
    let prompt_logprobs =
        prompt_logprobs.map(|lp| decoded_prompt_logprobs_to_maps(lp, return_tokens_as_token_ids));
    let text = match &echo {
        None => collected.text,
        Some(prompt) => format!("{prompt}{}", collected.text),
    };

    Ok(CompletionChoiceOutput {
        choice: CompletionChoice {
            index: choice_index,
            text,
            logprobs,
            finish_reason: Some(completion_finish_reason_to_openai(finish_reason)?.into()),
            stop_reason,
            prompt_logprobs,
            token_ids: return_token_ids.then(|| collected.token_ids.clone()),
            prompt_token_ids: return_token_ids.then(|| collected.prompt_token_ids.to_vec()),
        },
        usage: ChoiceUsage {
            prompt_tokens: collected.prompt_token_ids.len() as u32,
            completion_tokens: collected.token_ids.len() as u32,
        },
        kv_transfer_params: collected.kv_transfer_params,
    })
}

fn completion_response_from_choices(
    request_id: String,
    response_model: String,
    created: u64,
    outputs: Vec<CompletionChoiceOutput>,
) -> CompletionResponse {
    let prompt_tokens = outputs.first().map_or(0, |output| output.usage.prompt_tokens);
    let completion_tokens = outputs.iter().map(|output| output.usage.completion_tokens).sum();
    let kv_transfer_params = outputs.iter().find_map(|output| output.kv_transfer_params.clone());

    CompletionResponse {
        id: request_id,
        object: "text_completion".to_string(),
        created,
        model: response_model,
        choices: outputs.into_iter().map(|output| output.choice).collect(),
        usage: Some(Usage::from_counts(prompt_tokens, completion_tokens)),
        system_fingerprint: None,
        kv_transfer_params,
    }
}

/// Convert one internal decoded-text stream into OpenAI completions chunks.
#[try_stream]
async fn completion_chunk_stream(
    stream: impl TextOutputStream,
    request_id: String,
    response_model: String,
    created: u64,
    choice_index: u32,
    log_request: bool,
    include_usage: bool,
    echo: Option<String>,
    requested_logprobs: Option<u32>,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
    mut y: TryYielder<CompletionSseChunk, ApiError>,
) -> Result<(), ApiError> {
    let events = completion_choice_event_stream(
        stream,
        request_id.clone(),
        response_model.clone(),
        created,
        choice_index,
        log_request,
        echo,
        requested_logprobs,
        return_token_ids,
        return_tokens_as_token_ids,
    );
    pin_mut!(events);

    while let Some(next) = events.next().await {
        match next? {
            CompletionStreamEvent::Chunk(chunk) => {
                y.yield_ok(CompletionSseChunk::Chunk(chunk)).await;
            }
            CompletionStreamEvent::Finished(usage) => {
                if include_usage {
                    y.yield_ok(CompletionSseChunk::Usage(usage_chunk(
                        &request_id,
                        &response_model,
                        created,
                        Usage::from_counts(usage.prompt_tokens, usage.completion_tokens),
                    )))
                    .await;
                }
            }
        }
    }

    Ok(())
}

#[try_stream]
async fn completion_parallel_chunk_stream(
    stream: impl Stream<Item = Result<CompletionStreamEvent, ApiError>>,
    request_id: String,
    response_model: String,
    created: u64,
    include_usage: bool,
    mut y: TryYielder<CompletionSseChunk, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut prompt_tokens = None;
    let mut completion_tokens = 0_u32;

    while let Some(next) = stream.next().await {
        match next? {
            CompletionStreamEvent::Chunk(chunk) => {
                y.yield_ok(CompletionSseChunk::Chunk(chunk)).await;
            }
            CompletionStreamEvent::Finished(usage) => {
                prompt_tokens.get_or_insert(usage.prompt_tokens);
                completion_tokens = completion_tokens.saturating_add(usage.completion_tokens);
            }
        }
    }

    if include_usage {
        y.yield_ok(CompletionSseChunk::Usage(usage_chunk(
            &request_id,
            &response_model,
            created,
            Usage::from_counts(prompt_tokens.unwrap_or(0), completion_tokens),
        )))
        .await;
    }

    Ok(())
}

#[try_stream]
async fn completion_choice_event_stream(
    stream: impl TextOutputStream,
    request_id: String,
    response_model: String,
    created: u64,
    choice_index: u32,
    log_request: bool,
    echo: Option<String>,
    requested_logprobs: Option<u32>,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
    mut y: TryYielder<CompletionStreamEvent, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut visible_text_len = 0_u32;
    let mut first_chunk = true;

    while let Some(next) = stream.next().await {
        match next {
            Ok(DecodedTextEvent::Start {
                prompt_token_ids, ..
            }) => {
                debug!("completion stream started");
                if let Some(prompt) = echo.as_ref() {
                    visible_text_len = text_len(prompt);
                    let mut chunk = delta_chunk(
                        &request_id,
                        &response_model,
                        created,
                        choice_index,
                        prompt.clone(),
                        None,
                    );
                    if return_token_ids && first_chunk {
                        if let Some(choice) = chunk.choices.first_mut() {
                            choice.prompt_token_ids = Some(prompt_token_ids.to_vec());
                        }
                        first_chunk = false;
                    }
                    y.yield_ok(CompletionStreamEvent::Chunk(chunk)).await;
                } else if return_token_ids {
                    // Emit a chunk with prompt_token_ids in the first streaming response
                    let mut chunk = delta_chunk(
                        &request_id,
                        &response_model,
                        created,
                        choice_index,
                        String::new(),
                        None,
                    );
                    if let Some(choice) = chunk.choices.first_mut() {
                        choice.prompt_token_ids = Some(prompt_token_ids.to_vec());
                    }
                    first_chunk = false;
                    y.yield_ok(CompletionStreamEvent::Chunk(chunk)).await;
                }
            }
            Ok(DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
                finished,
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
                        return_tokens_as_token_ids,
                    )?)
                } else {
                    None
                };
                let mut chunk = delta_chunk(
                    &request_id,
                    &response_model,
                    created,
                    choice_index,
                    delta,
                    logprobs,
                );
                if return_token_ids && let Some(choice) = chunk.choices.first_mut() {
                    choice.token_ids = Some(token_ids);
                }
                y.yield_ok(CompletionStreamEvent::Chunk(chunk)).await;
                visible_text_len = visible_text_len.saturating_add(delta_text_len);

                if let Some(finished) = finished {
                    let finish_reason = finished.finish_reason.clone();
                    if log_request {
                        info!(
                            stream = true,
                            model = %response_model,
                            prompt_tokens = finished.prompt_token_count,
                            output_tokens = finished.output_token_count,
                            finish_reason = finish_reason.as_str(),
                            "completion finished"
                        );
                    }
                    y.yield_ok(CompletionStreamEvent::Chunk(final_chunk(
                        &request_id,
                        &response_model,
                        created,
                        choice_index,
                        finish_reason,
                    )?))
                    .await;
                    y.yield_ok(CompletionStreamEvent::Finished(ChoiceUsage {
                        prompt_tokens: finished.prompt_token_count as u32,
                        completion_tokens: finished.output_token_count as u32,
                    }))
                    .await;
                }
            }
            Err(error) => {
                error!(
                    error = %error.as_report(),
                    "completion stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }
    Ok(())
}

fn delta_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    choice_index: u32,
    text: String,
    logprobs: Option<LogProbs>,
) -> CompletionStreamResponse {
    let mut chunk = CompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(CompletionStreamChoice {
        index: choice_index,
        text,
        logprobs,
        ..Default::default()
    });
    chunk
}

fn final_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    choice_index: u32,
    finish_reason: FinishReason,
) -> Result<CompletionStreamResponse, ApiError> {
    let finish_reason = completion_finish_reason_to_openai(finish_reason)?;

    let mut chunk = CompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(CompletionStreamChoice {
        index: choice_index,
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
        FinishReason::Abort => Ok("abort"),
        FinishReason::Error => {
            bail_server_error!("Internal server error");
        }
    }
}

fn usage_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    usage: Usage,
) -> CompletionStreamResponse {
    let mut chunk = CompletionStreamResponse::new(request_id, response_model, created);
    chunk.usage = Some(usage);
    chunk
}

/// Convert one chunk stream into OpenAI-style SSE events.
///
/// OpenAI-style streaming errors are encoded as ordinary `data: {"error": ...}`
/// events followed by `data: [DONE]`, so the transport stream itself stays
/// infallible even when generation fails after the HTTP response has started.
#[try_stream]
async fn completion_sse_stream(
    stream: impl Stream<Item = Result<CompletionSseChunk, ApiError>>,
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
fn to_sse_event(chunk: &CompletionSseChunk) -> Event {
    let payload = serde_json::to_string(chunk).expect("completion chunk must serialize to JSON");
    trace!(payload, "completion emitting chunk");
    Event::default().data(payload)
}

/// Serialize one OpenAI error payload into one SSE `data:` event.
fn to_error_sse_event(error: &ApiError) -> Event {
    let payload = serde_json::to_string(&error.to_error_response())
        .expect("ErrorResponse must serialize to JSON");
    trace!(payload, "completion emitting error");
    Event::default().data(payload)
}

/// Build the terminal OpenAI SSE sentinel event.
fn done_sse_event() -> Event {
    trace!("completion emitting done");
    Event::default().data("[DONE]")
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt as _, stream};
    use itertools::Itertools as _;
    use vllm_text::{
        DecodedLogprobs, DecodedPositionLogprobs, DecodedTextEvent, DecodedTokenLogprob,
        FinishReason, Finished,
    };

    use super::{CompletionSseChunk, completion_chunk_stream, final_chunk};

    #[test]
    fn final_chunk_maps_stop_finish_reason() {
        let chunk = final_chunk("cmpl-1", "model", 1, 0, FinishReason::stop_eos())
            .expect("finish reason valid");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunk.choices[0].text, "");
    }

    #[test]
    fn final_chunk_maps_length_finish_reason() {
        let chunk = final_chunk("cmpl-1", "model", 1, 0, FinishReason::Length)
            .expect("finish reason valid");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn final_chunk_maps_abort_finish_reason() {
        let chunk =
            final_chunk("cmpl-1", "model", 1, 0, FinishReason::Abort).expect("finish reason valid");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("abort"));
    }

    #[test]
    fn final_chunk_rejects_error_finish_reason() {
        assert!(final_chunk("cmpl-1", "model", 1, 0, FinishReason::Error).is_err());
    }

    #[tokio::test]
    async fn completion_chunk_stream_maps_streaming_logprobs() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![1, 2, 3, 4, 5].into(),
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "h".to_string(),
                token_ids: vec![b'h' as u32],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![
                            DecodedTokenLogprob {
                                token_id: 0,
                                token: "h".to_string(),
                                logprob: -0.1,
                                rank: 1,
                            },
                            DecodedTokenLogprob {
                                token_id: 0,
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
                                token_id: 0,
                                token: "!".to_string(),
                                logprob: -0.3,
                                rank: 1,
                            },
                            DecodedTokenLogprob {
                                token_id: 0,
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
                    kv_transfer_params: None,
                }),
            }),
        ]);

        let chunks = completion_chunk_stream(
            stream,
            "cmpl-1".to_string(),
            "model".to_string(),
            1,
            0,
            false,
            false,
            None,
            Some(1),
            false,
            false,
        )
        .collect::<Vec<_>>()
        .await;

        let chunks: Vec<_> = chunks.into_iter().try_collect().expect("stream should succeed");

        match &chunks[0] {
            CompletionSseChunk::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].text, "h");
                assert_eq!(
                    chunk.choices[0].logprobs.as_ref().expect("logprobs").tokens,
                    vec!["h".to_string()]
                );
                assert_eq!(
                    chunk.choices[0].logprobs.as_ref().expect("logprobs").text_offset,
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
                    chunk.choices[0].logprobs.as_ref().expect("logprobs").text_offset,
                    vec![1]
                );
            }
            CompletionSseChunk::Usage(_) => panic!("expected regular chunk"),
        }
    }
}
