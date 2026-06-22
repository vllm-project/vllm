mod convert;
mod types;
mod validate;

use std::collections::HashMap;
use std::convert::Infallible;
use std::result::Result;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tracing::{debug, error, info, trace};
use tracing_futures::Instrument as _;
use vllm_text::{
    DecodedPromptLogprobs, DecodedTextEvent, FinishReason, TextOutputStream,
    TextOutputStreamExt as _,
};

use self::convert::{ResponseOptions, prepare_completion_request};
use super::utils::logprobs::{
    collected_logprobs_to_openai, decoded_logprobs_to_openai, decoded_prompt_logprobs_to_maps,
    decoded_prompt_logprobs_to_openai, text_len,
};
use super::utils::types::Usage;
use crate::config::ApiServerOptions;
use crate::error::{ApiError, bail_server_error, server_error, text_submit_error};
use crate::routes::openai::completions::types::{
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionSseChunk,
    CompletionStreamChoice, CompletionStreamResponse,
};
use crate::routes::openai::utils::types::LogProbs;
use crate::routes::openai::utils::usage::ContinuousUsage;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

/// Validate one completions request and proxy it into the shared `vllm-text`
/// stack.
pub async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Response {
    let stream = body.stream;
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
    let api_server_options = state.api_server_options;
    let text_stream = match state
        .chat
        .text()
        .generate(prepared.text_request)
        .instrument(request_span.clone())
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return text_submit_error("failed to submit completion request", error).into_response();
        }
    };

    if stream {
        let chunk_stream = completion_chunk_stream(
            text_stream,
            prepared.request_id,
            prepared.response_model,
            created,
            api_server_options,
            prepared.options,
        );
        let sse_stream = completion_sse_stream(chunk_stream).instrument(request_span);

        Sse::new(sse_stream).into_response()
    } else {
        let response = match collect_completion(
            text_stream,
            prepared.request_id,
            prepared.response_model,
            created,
            api_server_options,
            prepared.options,
        )
        .instrument(request_span.clone())
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
    request_id: String,
    response_model: String,
    created: u64,
    ApiServerOptions {
        enable_log_requests,
        enable_prompt_tokens_details,
        ..
    }: ApiServerOptions,
    ResponseOptions {
        // Ignored: non-streaming responses always include usage.
        include_usage: _,
        // Ignored: non-streaming responses are collected before usage is attached.
        include_continuous_usage: _,
        prompt_only,
        echo,
        requested_logprobs,
        include_prompt_logprobs,
        return_token_ids,
        return_tokens_as_token_ids,
    }: ResponseOptions,
) -> Result<CompletionResponse, ApiError> {
    let collected = stream
        .collect_output()
        .await
        .map_err(|error| server_error!("completion stream failed: {}", error.to_report_string()))?;
    let finish_reason = collected.finish_reason.clone();
    let stop_reason = finish_reason
        .as_stop_reason()
        .map(|sr| serde_json::to_value(sr).expect("StopReason must serialize to JSON"));

    let prompt_char_count = echo.as_ref().map(|prompt| text_len(prompt)).unwrap_or_default();
    let logprobs = if requested_logprobs.is_some() && prompt_only {
        let prompt = echo.as_deref().ok_or_else(|| {
            server_error!("prompt-only completion response missing echoed prompt")
        })?;
        Some(prompt_only_logprobs_to_openai(
            collected.prompt_logprobs.as_ref(),
            prompt,
            collected.prompt_token_ids.as_ref(),
            return_tokens_as_token_ids,
        )?)
    } else if requested_logprobs.is_some() {
        Some(collected_logprobs_to_openai(
            &collected,
            echo.is_some(),
            prompt_char_count,
            return_tokens_as_token_ids,
        )?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        Some(prompt_logprobs_to_maps(
            collected.prompt_logprobs.as_ref(),
            collected.prompt_token_ids.as_ref(),
            return_tokens_as_token_ids,
        )?)
    } else {
        None
    };
    let text = match &echo {
        None => collected.text,
        Some(prompt) if prompt_only => prompt.clone(),
        Some(prompt) => format!("{prompt}{}", collected.text),
    };
    let finish_reason = completion_finish_reason_to_openai(finish_reason)?.to_string();
    let usage = Usage::from_token_usage(collected.usage, enable_prompt_tokens_details);

    if enable_log_requests {
        info!(
            model = %response_model,
            prompt_tokens = usage.prompt_tokens,
            output_tokens = usage.completion_tokens.unwrap_or(0),
            %finish_reason,
            "completion finished"
        );
    }

    Ok(CompletionResponse {
        id: request_id,
        object: "text_completion".to_string(),
        created,
        model: response_model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            logprobs,
            finish_reason: Some(finish_reason),
            stop_reason,
            prompt_logprobs,
            token_ids: return_token_ids.then(|| collected.token_ids.clone()),
            prompt_token_ids: return_token_ids.then(|| collected.prompt_token_ids.to_vec()),
        }],
        usage: Some(usage),
        system_fingerprint: None,
        kv_transfer_params: collected.kv_transfer_params,
    })
}

/// Convert one internal decoded-text stream into OpenAI completions chunks.
#[try_stream]
async fn completion_chunk_stream(
    stream: impl TextOutputStream,
    request_id: String,
    response_model: String,
    created: u64,
    ApiServerOptions {
        enable_log_requests,
        enable_prompt_tokens_details,
        ..
    }: ApiServerOptions,
    ResponseOptions {
        include_usage,
        include_continuous_usage,
        prompt_only,
        echo,
        requested_logprobs,
        // Ignored: streaming prompt logprobs are rejected for Python parity.
        include_prompt_logprobs: _,
        return_token_ids,
        return_tokens_as_token_ids,
    }: ResponseOptions,
    mut y: TryYielder<CompletionSseChunk, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut visible_text_len = 0_u32;
    let mut first_chunk = true;
    let mut continuous_usage = ContinuousUsage::default();

    /// Yield a chunk with optional continuous usage attached.
    macro_rules! yield_chunk {
        ($chunk:expr) => {{
            let mut chunk = $chunk;
            if include_continuous_usage {
                chunk.usage = Some(continuous_usage.to_usage());
            }
            y.yield_ok(CompletionSseChunk::Chunk(chunk)).await;
        }};
    }

    while let Some(next) = stream.next().await {
        match next {
            Ok(DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            }) => {
                debug!("completion stream started");
                continuous_usage.set_prompt_tokens(prompt_token_ids.len());
                if let Some(prompt) = echo.as_ref() {
                    visible_text_len = text_len(prompt);
                    let logprobs = if prompt_only && requested_logprobs.is_some() {
                        Some(prompt_only_logprobs_to_openai(
                            prompt_logprobs.as_ref(),
                            prompt,
                            prompt_token_ids.as_ref(),
                            return_tokens_as_token_ids,
                        )?)
                    } else {
                        None
                    };
                    let mut chunk = delta_chunk(
                        &request_id,
                        &response_model,
                        created,
                        prompt.clone(),
                        logprobs,
                    );
                    if return_token_ids && first_chunk {
                        if let Some(choice) = chunk.choices.first_mut() {
                            choice.prompt_token_ids = Some(prompt_token_ids.to_vec());
                        }
                        first_chunk = false;
                    }
                    yield_chunk!(chunk);
                } else if return_token_ids {
                    // Emit a chunk with prompt_token_ids in the first streaming response
                    let mut chunk =
                        delta_chunk(&request_id, &response_model, created, String::new(), None);
                    if let Some(choice) = chunk.choices.first_mut() {
                        choice.prompt_token_ids = Some(prompt_token_ids.to_vec());
                    }
                    first_chunk = false;
                    yield_chunk!(chunk);
                }
            }
            Ok(DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
                finished,
            }) => {
                // Prompt-only streaming already emitted the echoed prompt in the Start chunk.
                // The one generated token is only used to drive the engine to a finished event,
                // so hide its delta and forward only the terminal finish/usage metadata.
                if prompt_only {
                    if let Some(finished) = finished {
                        if enable_log_requests {
                            info!(
                                stream = true,
                                model = %response_model,
                                prompt_tokens = finished.usage.prompt_token_count,
                                output_tokens = finished.usage.output_token_count,
                                finish_reason = finished.finish_reason.as_str(),
                                "completion finished"
                            );
                        }
                        continuous_usage.set_final_counts(
                            finished.usage.prompt_token_count,
                            finished.usage.output_token_count,
                        );
                        let final_chunk = final_chunk(
                            &request_id,
                            &response_model,
                            created,
                            finished.finish_reason,
                        )?;
                        yield_chunk!(final_chunk);

                        if include_usage {
                            y.yield_ok(CompletionSseChunk::Usage(usage_chunk(
                                &request_id,
                                &response_model,
                                created,
                                Usage::from_token_usage(
                                    finished.usage,
                                    enable_prompt_tokens_details,
                                ),
                            )))
                            .await;
                        }
                    }
                    continue;
                }
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
                let mut chunk = delta_chunk(&request_id, &response_model, created, delta, logprobs);
                let delta_token_count = token_ids.len();
                continuous_usage.add_output_tokens(delta_token_count);
                if return_token_ids && let Some(choice) = chunk.choices.first_mut() {
                    choice.token_ids = Some(token_ids);
                }
                yield_chunk!(chunk);
                visible_text_len = visible_text_len.saturating_add(delta_text_len);

                if let Some(finished) = finished {
                    if enable_log_requests {
                        info!(
                            stream = true,
                            model = %response_model,
                            prompt_tokens = finished.usage.prompt_token_count,
                            output_tokens = finished.usage.output_token_count,
                            finish_reason = finished.finish_reason.as_str(),
                            "completion finished"
                        );
                    }
                    continuous_usage.set_final_counts(
                        finished.usage.prompt_token_count,
                        finished.usage.output_token_count,
                    );
                    let final_chunk = final_chunk(
                        &request_id,
                        &response_model,
                        created,
                        finished.finish_reason,
                    )?;
                    yield_chunk!(final_chunk);

                    if include_usage {
                        y.yield_ok(CompletionSseChunk::Usage(usage_chunk(
                            &request_id,
                            &response_model,
                            created,
                            Usage::from_token_usage(finished.usage, enable_prompt_tokens_details),
                        )))
                        .await;
                    }
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
    text: String,
    logprobs: Option<LogProbs>,
) -> CompletionStreamResponse {
    let mut chunk = CompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(CompletionStreamChoice {
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
    finish_reason: FinishReason,
) -> Result<CompletionStreamResponse, ApiError> {
    let finish_reason = completion_finish_reason_to_openai(finish_reason)?;

    let mut chunk = CompletionStreamResponse::new(request_id, response_model, created);
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
        FinishReason::Abort => Ok("abort"),
        FinishReason::Error => {
            bail_server_error!("Internal server error");
        }
    }
}

fn prompt_only_logprobs_to_openai(
    prompt_logprobs: Option<&DecodedPromptLogprobs>,
    prompt: &str,
    prompt_token_ids: &[u32],
    return_tokens_as_token_ids: bool,
) -> Result<LogProbs, ApiError> {
    if let Some(prompt_logprobs) = prompt_logprobs {
        return decoded_prompt_logprobs_to_openai(prompt_logprobs, 0, return_tokens_as_token_ids);
    }

    if let [token_id] = prompt_token_ids {
        let token = if return_tokens_as_token_ids {
            format!("token_id:{token_id}")
        } else {
            prompt.to_string()
        };

        return Ok(LogProbs {
            tokens: vec![token],
            token_logprobs: vec![None],
            top_logprobs: vec![None],
            text_offset: vec![0],
        });
    }

    Err(server_error!(
        "prompt-only completion requested logprobs but generation returned none"
    ))
}

fn prompt_logprobs_to_maps(
    prompt_logprobs: Option<&DecodedPromptLogprobs>,
    prompt_token_ids: &[u32],
    return_tokens_as_token_ids: bool,
) -> Result<Vec<Option<HashMap<String, f32>>>, ApiError> {
    if let Some(prompt_logprobs) = prompt_logprobs {
        return Ok(decoded_prompt_logprobs_to_maps(
            prompt_logprobs,
            return_tokens_as_token_ids,
        ));
    }

    if let [_token_id] = prompt_token_ids {
        return Ok(vec![None]);
    }

    Err(server_error!(
        "completion response requested prompt_logprobs but generation returned none"
    ))
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
        DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTextEvent,
        DecodedTokenLogprob, FinishReason, Finished,
    };

    use super::{
        ApiServerOptions, CompletionSseChunk, ResponseOptions, completion_chunk_stream, final_chunk,
    };

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
    fn final_chunk_maps_abort_finish_reason() {
        let chunk =
            final_chunk("cmpl-1", "model", 1, FinishReason::Abort).expect("finish reason valid");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("abort"));
    }

    #[test]
    fn final_chunk_rejects_error_finish_reason() {
        assert!(final_chunk("cmpl-1", "model", 1, FinishReason::Error).is_err());
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
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 5,
                        output_token_count: 2,
                        cached_token_count: 3,
                    },
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
            ApiServerOptions {
                enable_prompt_tokens_details: true,
                ..Default::default()
            },
            ResponseOptions {
                include_usage: true,
                requested_logprobs: Some(1),
                ..Default::default()
            },
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

        match &chunks[3] {
            CompletionSseChunk::Usage(chunk) => {
                assert_eq!(
                    chunk
                        .usage
                        .as_ref()
                        .expect("usage")
                        .prompt_tokens_details
                        .as_ref()
                        .map(|details| details.cached_tokens),
                    Some(3)
                );
            }
            CompletionSseChunk::Chunk(_) => panic!("expected usage chunk"),
        }
    }

    #[tokio::test]
    async fn collect_completion_hides_internal_prompt_only_token() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![1, 2].into(),
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: " leaked".to_string(),
                token_ids: vec![3],
                logprobs: None,
                finished: Some(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 2,
                        output_token_count: 1,
                        cached_token_count: 0,
                    },
                    finish_reason: FinishReason::Length,
                    kv_transfer_params: None,
                }),
            }),
        ]);

        let response = super::collect_completion(
            stream,
            "cmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions::default(),
            ResponseOptions {
                prompt_only: true,
                echo: Some("hello".to_string()),
                return_token_ids: true,
                ..Default::default()
            },
        )
        .await
        .expect("collect completion");

        assert_eq!(response.choices[0].text, "hello");
        assert_eq!(response.choices[0].token_ids.as_deref(), Some(&[3][..]));
        assert_eq!(
            response.choices[0].prompt_token_ids.as_deref(),
            Some(&[1, 2][..])
        );
        let usage = response.usage.expect("usage");
        assert_eq!(usage.prompt_tokens, 2);
        assert_eq!(usage.completion_tokens, Some(1));
        assert_eq!(usage.total_tokens, 3);
    }

    #[tokio::test]
    async fn collect_completion_maps_prompt_logprobs_for_single_token_prompt() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![9707].into(),
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: " leaked".to_string(),
                token_ids: vec![3],
                logprobs: None,
                finished: Some(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 1,
                        output_token_count: 1,
                        cached_token_count: 0,
                    },
                    finish_reason: FinishReason::Length,
                    kv_transfer_params: None,
                }),
            }),
        ]);

        let response = super::collect_completion(
            stream,
            "cmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions::default(),
            ResponseOptions {
                prompt_only: true,
                echo: Some("Hello".to_string()),
                requested_logprobs: Some(1),
                include_prompt_logprobs: true,
                ..Default::default()
            },
        )
        .await
        .expect("collect completion");

        let choice = &response.choices[0];
        assert_eq!(choice.text, "Hello");
        assert_eq!(choice.prompt_logprobs, Some(vec![None]));
        let logprobs = choice.logprobs.as_ref().expect("logprobs");
        assert_eq!(logprobs.tokens, vec!["Hello".to_string()]);
        assert_eq!(logprobs.token_logprobs, vec![None]);
        assert_eq!(logprobs.top_logprobs, vec![None]);
        assert_eq!(logprobs.text_offset, vec![0]);
        let usage = response.usage.expect("usage");
        assert_eq!(usage.prompt_tokens, 1);
        assert_eq!(usage.completion_tokens, Some(1));
        assert_eq!(usage.total_tokens, 2);
    }

    #[tokio::test]
    async fn completion_chunk_stream_hides_internal_prompt_only_token() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![1, 2].into(),
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: " leaked".to_string(),
                token_ids: vec![3],
                logprobs: None,
                finished: Some(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 2,
                        output_token_count: 1,
                        cached_token_count: 0,
                    },
                    finish_reason: FinishReason::Length,
                    kv_transfer_params: None,
                }),
            }),
        ]);

        let chunks = completion_chunk_stream(
            stream,
            "cmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions::default(),
            ResponseOptions {
                include_usage: true,
                prompt_only: true,
                echo: Some("hello".to_string()),
                return_token_ids: true,
                ..Default::default()
            },
        )
        .collect::<Vec<_>>()
        .await;

        let chunks: Vec<_> = chunks.into_iter().try_collect().expect("stream should succeed");
        assert_eq!(chunks.len(), 3);

        match &chunks[0] {
            CompletionSseChunk::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].text, "hello");
                assert_eq!(
                    chunk.choices[0].prompt_token_ids.as_deref(),
                    Some(&[1, 2][..])
                );
            }
            CompletionSseChunk::Usage(_) => panic!("expected prompt chunk"),
        }
        match &chunks[1] {
            CompletionSseChunk::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].text, "");
                assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
            }
            CompletionSseChunk::Usage(_) => panic!("expected final chunk"),
        }
        match &chunks[2] {
            CompletionSseChunk::Usage(chunk) => {
                let usage = chunk.usage.as_ref().expect("usage");
                assert_eq!(usage.prompt_tokens, 2);
                assert_eq!(usage.completion_tokens, Some(1));
                assert_eq!(usage.total_tokens, 3);
            }
            CompletionSseChunk::Chunk(_) => panic!("expected usage chunk"),
        }
    }

    #[tokio::test]
    async fn completion_chunk_stream_maps_prompt_logprobs_for_single_token_prompt() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![9707].into(),
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: " leaked".to_string(),
                token_ids: vec![3],
                logprobs: None,
                finished: Some(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 1,
                        output_token_count: 1,
                        cached_token_count: 0,
                    },
                    finish_reason: FinishReason::Length,
                    kv_transfer_params: None,
                }),
            }),
        ]);

        let chunks = completion_chunk_stream(
            stream,
            "cmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions::default(),
            ResponseOptions {
                prompt_only: true,
                echo: Some("Hello".to_string()),
                requested_logprobs: Some(1),
                ..Default::default()
            },
        )
        .collect::<Vec<_>>()
        .await;

        let chunks: Vec<_> = chunks.into_iter().try_collect().expect("stream should succeed");
        assert_eq!(chunks.len(), 2);

        match &chunks[0] {
            CompletionSseChunk::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].text, "Hello");
                let logprobs = chunk.choices[0].logprobs.as_ref().expect("logprobs");
                assert_eq!(logprobs.tokens, vec!["Hello".to_string()]);
                assert_eq!(logprobs.token_logprobs, vec![None]);
                assert_eq!(logprobs.top_logprobs, vec![None]);
                assert_eq!(logprobs.text_offset, vec![0]);
            }
            CompletionSseChunk::Usage(_) => panic!("expected prompt chunk"),
        }
        match &chunks[1] {
            CompletionSseChunk::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].text, "");
                assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
            }
            CompletionSseChunk::Usage(_) => panic!("expected final chunk"),
        }
    }

    #[tokio::test]
    async fn completion_chunk_stream_maps_prompt_only_logprobs() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![1, 2].into(),
                prompt_logprobs: Some(DecodedPromptLogprobs {
                    first_token_id: 1,
                    first_token: "he".to_string(),
                    scored_positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 2,
                            token: "llo".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    }],
                }),
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: " leaked".to_string(),
                token_ids: vec![3],
                logprobs: None,
                finished: Some(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 2,
                        output_token_count: 1,
                        cached_token_count: 0,
                    },
                    finish_reason: FinishReason::Length,
                    kv_transfer_params: None,
                }),
            }),
        ]);

        let chunks = completion_chunk_stream(
            stream,
            "cmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions::default(),
            ResponseOptions {
                prompt_only: true,
                echo: Some("hello".to_string()),
                requested_logprobs: Some(1),
                ..Default::default()
            },
        )
        .collect::<Vec<_>>()
        .await;

        let chunks: Vec<_> = chunks.into_iter().try_collect().expect("stream should succeed");
        assert_eq!(chunks.len(), 2);

        match &chunks[0] {
            CompletionSseChunk::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].text, "hello");
                let logprobs = chunk.choices[0].logprobs.as_ref().expect("logprobs");
                assert_eq!(logprobs.tokens, vec!["he".to_string(), "llo".to_string()]);
                assert_eq!(logprobs.token_logprobs, vec![None, Some(-0.2)]);
                assert_eq!(logprobs.text_offset, vec![0, 2]);
            }
            CompletionSseChunk::Usage(_) => panic!("expected prompt chunk"),
        }
        match &chunks[1] {
            CompletionSseChunk::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].text, "");
                assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
            }
            CompletionSseChunk::Usage(_) => panic!("expected final chunk"),
        }
    }
}
