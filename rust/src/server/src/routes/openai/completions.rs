// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use futures::future::try_join_all;
use futures::stream::select_all;
use futures::{Stream, StreamExt as _, pin_mut};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::{debug, error, info, trace};
use tracing_futures::Instrument as _;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_text::tokenizer::Tokenizer;
use vllm_text::{
    DecodedPromptLogprobs, DecodedTextEvent, FinishReason, SampledDelta, TextOutputStream,
    TextOutputStreamExt as _, TextRequest,
};

use self::convert::{ResponseOptions, prepare_completion_request};
pub(crate) use self::types::CompletionRequest;
use super::utils::logprobs::{
    append_openai_logprobs, collected_logprobs_to_openai, decoded_logprobs_to_openai,
    decoded_prompt_logprobs_to_openai, prompt_logprobs_to_maps, text_len,
};
use super::utils::metrics::PerRequestMetrics;
use super::utils::types::{StreamResponseEnvelope, Usage};
use crate::config::ApiServerOptions;
use crate::error::{ApiError, bail_server_error, engine_error, server_error, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::routes::openai::completions::types::{
    CompletionChoice, CompletionResponse, CompletionSseChunk, CompletionStreamChoice,
    CompletionStreamResponse,
};
use crate::routes::openai::utils::types::LogProbs;
use crate::routes::openai::utils::usage::ContinuousUsage;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{ResolvedRequestContext, resolve_request_context, sse_response, unix_timestamp};

type CompletionChunkStream =
    Pin<Box<dyn Stream<Item = Result<CompletionSseChunk, ApiError>> + Send>>;

pub(crate) fn lower_completion_request(
    request: CompletionRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
    tokenizer: &dyn Tokenizer,
) -> Result<TextRequest, ApiError> {
    prepare_completion_request(request, lora_resolution, ctx, tokenizer)
        .map(|prepared| prepared.text_request)
}

/// Validate one completions request and proxy it into the shared `vllm-text`
/// stack.
pub async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(mut body): ValidatedJson<CompletionRequest>,
) -> Response {
    let stream = body.stream;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let public_request_id = format!("cmpl-{}", request_context.request_id);
    let requested_model = body.model.as_deref().filter(|model| !model.is_empty());
    let lora_resolution = state.resolve_model_with_loras(requested_model).await;
    if let Err(error) = validate::validate_request_compat(&body, &lora_resolution.model_names) {
        return error.into_response();
    }

    let tokenizer = state.chat.text().tokenizer();
    let is_batch = body.prompt.is_batch();
    let prompts = std::mem::replace(
        &mut body.prompt,
        types::CompletionPrompt::Text(String::new()),
    )
    .into_prompts();
    if prompts.is_empty() {
        return ApiError::invalid_request("You must pass at least one prompt", Some("prompt"))
            .into_response();
    }
    let mut prepared_requests = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let mut item = body.clone();
        item.prompt = prompt.into();
        match prepare_completion_request(
            item,
            &lora_resolution,
            request_context.clone(),
            tokenizer.as_ref(),
        ) {
            Ok(prepared) => {
                if let Err(error) = prepared.text_request.validate() {
                    return completion_error_response(
                        text_submit_error("invalid completion request", error),
                        stream && is_batch,
                        state.api_server_options,
                    );
                }
                prepared_requests.push(prepared);
            }
            Err(error) => return error.into_response(),
        }
    }
    let response_model = lora_resolution
        .lora_request
        .as_ref()
        .map(|request| request.lora_name.clone())
        .unwrap_or_else(|| lora_resolution.model_names.first().cloned().unwrap_or_default());
    let request_span = tracing::info_span!(
        "completions",
        request_id = %public_request_id,
        engine_request_id = tracing::field::Empty,
    );

    let created = unix_timestamp();
    let api_server_options = state.api_server_options;
    let mut submitted = Vec::with_capacity(prepared_requests.len());
    for prepared in prepared_requests {
        let text_stream = match state
            .chat
            .text()
            .generate(prepared.text_request)
            .instrument(request_span.clone())
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                return completion_error_response(
                    text_submit_error("failed to submit completion request", error),
                    stream && is_batch,
                    api_server_options,
                );
            }
        };
        submitted.push((text_stream, prepared.options));
    }

    if stream {
        if !is_batch {
            let (text_stream, options) = submitted.pop().expect("one submitted request");
            let chunk_stream = completion_chunk_stream(
                text_stream,
                public_request_id,
                response_model,
                created,
                api_server_options,
                options,
            );
            let sse_stream = completion_sse_stream(chunk_stream).instrument(request_span);
            return sse_response(sse_stream, api_server_options.sse_keep_alive_interval);
        }
        let chunk_streams = submitted
            .into_iter()
            .enumerate()
            .map(|(index, (text_stream, options))| {
                let stream = completion_chunk_stream(
                    text_stream,
                    public_request_id.clone(),
                    response_model.clone(),
                    created,
                    api_server_options,
                    options,
                )
                .map(move |result| result.map(|chunk| with_choice_index(chunk, index as u32)));
                Box::pin(stream) as CompletionChunkStream
            })
            .collect();
        let chunk_stream = aggregate_completion_chunk_streams(
            chunk_streams,
            public_request_id,
            response_model,
            created,
            body.stream_options
                .as_ref()
                .and_then(|options| options.include_usage)
                .unwrap_or(false),
        );
        let sse_stream = completion_sse_stream(chunk_stream).instrument(request_span);

        sse_response(sse_stream, api_server_options.sse_keep_alive_interval)
    } else {
        let responses = try_join_all(submitted.into_iter().map(|(text_stream, options)| {
            collect_completion(
                text_stream,
                public_request_id.clone(),
                response_model.clone(),
                created,
                api_server_options,
                options,
            )
        }))
        .instrument(request_span.clone())
        .await;
        let response = match responses {
            Ok(responses) => merge_completion_responses(responses),
            Err(error) => return error.into_response(),
        };

        Json(response).into_response()
    }
}

fn completion_error_response(
    error: ApiError,
    stream: bool,
    api_server_options: ApiServerOptions,
) -> Response {
    if stream {
        let chunks = futures::stream::once(async move { Err::<CompletionSseChunk, _>(error) });
        sse_response(
            completion_sse_stream(chunks),
            api_server_options.sse_keep_alive_interval,
        )
    } else {
        error.into_response()
    }
}

fn merge_completion_responses(mut responses: Vec<CompletionResponse>) -> CompletionResponse {
    let mut response = responses.remove(0);
    for next in responses {
        response.choices.extend(next.choices);
        response.usage = merge_usage(response.usage, next.usage);
        response.metrics = None;
    }
    for (index, choice) in response.choices.iter_mut().enumerate() {
        choice.index = index as u32;
    }
    response
}

fn merge_usage(left: Option<Usage>, right: Option<Usage>) -> Option<Usage> {
    match (left, right) {
        (Some(left), Some(right)) => Some(Usage::from_counts(
            left.prompt_tokens.saturating_add(right.prompt_tokens),
            left.completion_tokens
                .unwrap_or_default()
                .saturating_add(right.completion_tokens.unwrap_or_default()),
            match (left.prompt_tokens_details, right.prompt_tokens_details) {
                (Some(left), Some(right)) => {
                    Some(left.cached_tokens.saturating_add(right.cached_tokens))
                }
                (Some(details), None) | (None, Some(details)) => Some(details.cached_tokens),
                (None, None) => None,
            },
            left.completion_tokens_details
                .reasoning_tokens
                .saturating_add(right.completion_tokens_details.reasoning_tokens),
        )),
        (usage @ Some(_), None) | (None, usage @ Some(_)) => usage,
        (None, None) => None,
    }
}

fn with_choice_index(mut chunk: CompletionSseChunk, index: u32) -> CompletionSseChunk {
    if let CompletionSseChunk::Chunk(response) = &mut chunk {
        for choice in &mut response.choices {
            choice.index = index;
        }
    }
    chunk
}

#[try_stream]
async fn aggregate_completion_chunk_streams(
    streams: Vec<CompletionChunkStream>,
    request_id: String,
    response_model: String,
    created: u64,
    include_usage: bool,
    mut y: TryYielder<CompletionSseChunk, ApiError>,
) -> Result<(), ApiError> {
    let mut streams = select_all(streams);
    let mut usage = None;
    while let Some(next) = streams.next().await {
        let chunk = match next {
            Ok(chunk) => chunk,
            Err(error) => {
                drop(streams);
                y.yield_error(error).await;
                return Ok(());
            }
        };
        match chunk {
            CompletionSseChunk::Chunk(chunk) => {
                y.yield_ok(CompletionSseChunk::Chunk(chunk)).await;
            }
            CompletionSseChunk::Usage(chunk) => {
                usage = merge_usage(usage, chunk.usage);
            }
        }
    }
    if include_usage {
        let envelope = Arc::new(StreamResponseEnvelope::new(
            request_id,
            "text_completion",
            created,
            response_model,
        ));
        y.yield_ok(CompletionSseChunk::Usage(usage_chunk(
            &envelope,
            usage.unwrap_or_else(|| Usage::from_counts(0, 0, None, 0)),
            None,
        )))
        .await;
    }
    Ok(())
}

async fn collect_completion(
    stream: impl TextOutputStream,
    request_id: String,
    response_model: String,
    created: u64,
    ApiServerOptions {
        enable_log_requests,
        enable_prompt_tokens_details,
        default_return_tokens_as_token_ids,
        enable_per_request_metrics,
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
    let return_tokens_as_token_ids =
        return_tokens_as_token_ids.unwrap_or(default_return_tokens_as_token_ids);
    let collected = stream
        .collect_output()
        .await
        .map_err(|error| server_error!("completion stream failed: {}", error.to_report_string()))?;
    let finish_reason = collected.finish_reason.clone();
    let stop_reason = finish_reason.as_stop_reason().map(stop_reason_to_json);
    let finish_reason = completion_finish_reason_to_openai(&finish_reason)?.to_string();

    let prompt_char_count = echo.as_ref().map(|prompt| text_len(prompt)).unwrap_or_default();
    let logprobs = if requested_logprobs.is_some()
        && let Some(prompt) = echo.as_deref()
    {
        let prompt_logprobs = echo_prompt_logprobs_to_openai(
            collected.prompt_logprobs.as_ref(),
            prompt,
            collected.prompt_token_ids.as_ref(),
            return_tokens_as_token_ids,
        )?;
        Some(
            if !prompt_only && let Some(completion_logprobs) = collected.logprobs.as_ref() {
                let completion_offset = prompt_logprobs
                    .text_offset
                    .last()
                    .zip(prompt_logprobs.tokens.last())
                    .map(|(&offset, token)| offset.saturating_add(text_len(token)))
                    .unwrap_or_default();
                append_openai_logprobs(
                    prompt_logprobs,
                    decoded_logprobs_to_openai(
                        completion_logprobs,
                        completion_offset,
                        return_tokens_as_token_ids,
                    )?,
                )
            } else {
                prompt_logprobs
            },
        )
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
        )?)
    } else {
        None
    };
    let text = match &echo {
        None => collected.text,
        Some(prompt) if prompt_only => prompt.clone(),
        Some(prompt) => format!("{prompt}{}", collected.text),
    };
    let metrics = enable_per_request_metrics.then(|| {
        PerRequestMetrics::from_timestamps(collected.timestamps, collected.usage.output_token_count)
    });
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
        metrics,
        system_fingerprint: None,
        kv_transfer_params: collected.kv_transfer_params,
        ec_transfer_params: collected.ec_transfer_params,
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
        default_return_tokens_as_token_ids,
        enable_per_request_metrics,
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
    let return_tokens_as_token_ids =
        return_tokens_as_token_ids.unwrap_or(default_return_tokens_as_token_ids);
    pin_mut!(stream);
    let envelope = Arc::new(StreamResponseEnvelope::new(
        request_id,
        "text_completion",
        created,
        response_model,
    ));
    let mut visible_text_len = 0_u32;
    let mut first_chunk = true;
    let mut continuous_usage = ContinuousUsage::default();

    /// Yield a chunk with optional continuous usage attached.
    macro_rules! yield_chunk {
        ($chunk:expr) => {{
            let mut chunk = $chunk;
            if return_token_ids {
                for choice in &mut chunk.choices {
                    if choice.token_ids.is_none() {
                        choice.token_ids = Some(Vec::new());
                    }
                }
            }
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
                    let logprobs = if requested_logprobs.is_some() {
                        Some(echo_prompt_logprobs_to_openai(
                            prompt_logprobs.as_ref(),
                            prompt,
                            prompt_token_ids.as_ref(),
                            return_tokens_as_token_ids,
                        )?)
                    } else {
                        None
                    };
                    let mut chunk = delta_chunk(&envelope, prompt.clone(), logprobs);
                    if return_token_ids && first_chunk {
                        if let Some(choice) = chunk.choices.first_mut() {
                            choice.prompt_token_ids = Some(prompt_token_ids.to_vec());
                        }
                        first_chunk = false;
                    }
                    yield_chunk!(chunk);
                } else if return_token_ids {
                    // Emit a chunk with prompt_token_ids in the first streaming response
                    let mut chunk = delta_chunk(&envelope, String::new(), None);
                    if let Some(choice) = chunk.choices.first_mut() {
                        choice.prompt_token_ids = Some(prompt_token_ids.to_vec());
                    }
                    first_chunk = false;
                    yield_chunk!(chunk);
                }
            }
            Ok(DecodedTextEvent::TextDelta {
                decoded,
                sampled:
                    SampledDelta {
                        token_ids,
                        logprobs,
                    },
                finished,
            }) => {
                let delta = decoded.text;
                // Prompt-only streaming already emitted the echoed prompt in the Start chunk.
                // The one generated token is only used to drive the engine to a finished event,
                // so hide its delta and forward only the terminal finish/usage metadata.
                if prompt_only {
                    if let Some(finished) = finished {
                        if enable_log_requests {
                            info!(
                                stream = true,
                                model = %envelope.model(),
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
                        let final_chunk = final_chunk(&envelope, finished.finish_reason)?;
                        yield_chunk!(final_chunk);

                        if include_usage {
                            let metrics = enable_per_request_metrics.then(|| {
                                PerRequestMetrics::from_timestamps(
                                    finished.timestamps,
                                    finished.usage.output_token_count,
                                )
                            });
                            y.yield_ok(CompletionSseChunk::Usage(usage_chunk(
                                &envelope,
                                Usage::from_token_usage(
                                    finished.usage,
                                    enable_prompt_tokens_details,
                                ),
                                metrics,
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
                let mut chunk = delta_chunk(&envelope, delta, logprobs);
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
                            model = %envelope.model(),
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
                    let final_chunk = final_chunk(&envelope, finished.finish_reason)?;
                    yield_chunk!(final_chunk);

                    if include_usage {
                        let metrics = enable_per_request_metrics.then(|| {
                            PerRequestMetrics::from_timestamps(
                                finished.timestamps,
                                finished.usage.output_token_count,
                            )
                        });
                        y.yield_ok(CompletionSseChunk::Usage(usage_chunk(
                            &envelope,
                            Usage::from_token_usage(finished.usage, enable_prompt_tokens_details),
                            metrics,
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
    envelope: &Arc<StreamResponseEnvelope>,
    text: String,
    logprobs: Option<LogProbs>,
) -> CompletionStreamResponse {
    let mut chunk = CompletionStreamResponse::new(envelope);
    chunk.choices.push(CompletionStreamChoice {
        text,
        logprobs,
        ..Default::default()
    });
    chunk
}

fn final_chunk(
    envelope: &Arc<StreamResponseEnvelope>,
    finish_reason: FinishReason,
) -> Result<CompletionStreamResponse, ApiError> {
    let stop_reason = finish_reason.as_stop_reason().map(stop_reason_to_json);
    let finish_reason = completion_finish_reason_to_openai(&finish_reason)?;

    let mut chunk = CompletionStreamResponse::new(envelope);
    chunk.choices.push(CompletionStreamChoice {
        finish_reason: Some(finish_reason.to_string()),
        stop_reason,
        ..Default::default()
    });
    Ok(chunk)
}

fn completion_finish_reason_to_openai(
    finish_reason: &FinishReason,
) -> Result<&'static str, ApiError> {
    match finish_reason {
        FinishReason::Stop(_) => Ok("stop"),
        FinishReason::Repetition(_) => Ok("repetition"),
        FinishReason::Length => Ok("length"),
        FinishReason::Abort => Ok("abort"),
        FinishReason::Error(error) => Err(engine_error(error.as_deref())),
    }
}

fn stop_reason_to_json(stop_reason: &StopReason) -> Value {
    serde_json::to_value(stop_reason).expect("StopReason must serialize to JSON")
}

fn echo_prompt_logprobs_to_openai(
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
        "echoed completion requested logprobs but generation returned none"
    ))
}

fn usage_chunk(
    envelope: &Arc<StreamResponseEnvelope>,
    usage: Usage,
    metrics: Option<PerRequestMetrics>,
) -> CompletionStreamResponse {
    let mut chunk = CompletionStreamResponse::new(envelope);
    chunk.usage = Some(usage);
    chunk.metrics = metrics.map(Into::into);
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
    trace!(?chunk, "completion emitting chunk");
    Event::default()
        .json_data(chunk)
        .expect("completion chunk must serialize to JSON")
}

/// Serialize one OpenAI error payload into one SSE `data:` event.
fn to_error_sse_event(error: &ApiError) -> Event {
    let response = error.to_error_response();
    trace!(?response, "completion emitting error");
    Event::default()
        .json_data(response)
        .expect("ErrorResponse must serialize to JSON")
}

/// Build the terminal OpenAI SSE sentinel event.
fn done_sse_event() -> Event {
    trace!("completion emitting done");
    Event::default().data("[DONE]")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::{StreamExt as _, stream};
    use itertools::Itertools as _;
    use vllm_engine_core_client::protocol::output::StopReason;
    use vllm_text::{
        DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedText,
        DecodedTextEvent, DecodedTokenLogprob, FinishReason, Finished, SampledDelta,
    };

    use super::{
        ApiServerOptions, CompletionSseChunk, CompletionStreamResponse, ResponseOptions,
        StreamResponseEnvelope, collect_completion, completion_chunk_stream, final_chunk,
    };

    fn decoded_delta(
        text: impl Into<String>,
        token_ids: Vec<u32>,
        logprobs: Option<DecodedLogprobs>,
        finished: Option<Finished>,
    ) -> DecodedTextEvent {
        DecodedTextEvent::TextDelta {
            decoded: DecodedText::unattributed(text),
            sampled: SampledDelta {
                token_ids,
                logprobs,
            },
            finished: finished.map(Box::new),
        }
    }

    fn stream_envelope() -> Arc<StreamResponseEnvelope> {
        Arc::new(StreamResponseEnvelope::new(
            "cmpl-1".to_string(),
            "text_completion",
            1,
            "model".to_string(),
        ))
    }

    fn chunk_response(chunk: &CompletionSseChunk) -> &CompletionStreamResponse {
        match chunk {
            CompletionSseChunk::Chunk(response) | CompletionSseChunk::Usage(response) => response,
        }
    }

    #[test]
    fn final_chunk_maps_stop_finish_reason() {
        let chunk =
            final_chunk(&stream_envelope(), FinishReason::stop_eos()).expect("finish reason valid");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunk.choices[0].text, "");
    }

    #[test]
    fn final_chunk_maps_length_finish_reason() {
        let chunk =
            final_chunk(&stream_envelope(), FinishReason::Length).expect("finish reason valid");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn final_chunk_maps_abort_finish_reason() {
        let chunk =
            final_chunk(&stream_envelope(), FinishReason::Abort).expect("finish reason valid");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("abort"));
    }

    #[test]
    fn final_chunk_rejects_error_finish_reason() {
        assert!(final_chunk(&stream_envelope(), FinishReason::Error(None)).is_err());
    }

    #[tokio::test]
    async fn collected_completion_includes_metrics_only_when_enabled() {
        let make_stream = || {
            stream::iter(vec![
                Ok(DecodedTextEvent::Start {
                    prompt_token_ids: vec![1].into(),
                    prompt_logprobs: None,
                }),
                Ok(decoded_delta(
                    "hi",
                    vec![2],
                    None,
                    Some(Finished {
                        usage: vllm_llm::TokenUsage {
                            prompt_token_count: 1,
                            output_token_count: 1,
                            cached_token_count: 0,
                        },
                        finish_reason: FinishReason::stop_eos(),
                        kv_transfer_params: None,
                        ec_transfer_params: None,
                        sampling_mask: None,
                        timestamps: vllm_llm::RequestTimestamps {
                            queued_ts: 10.0,
                            scheduled_ts: 10.2,
                            first_token_ts: 10.5,
                            last_token_ts: 10.5,
                        },
                    }),
                )),
            ])
        };

        for enabled in [false, true] {
            let response = collect_completion(
                make_stream(),
                "cmpl-1".into(),
                "model".into(),
                1,
                ApiServerOptions {
                    enable_per_request_metrics: enabled,
                    ..Default::default()
                },
                ResponseOptions::default(),
            )
            .await
            .expect("completion response");
            let json = serde_json::to_value(response).expect("serializable response");
            if enabled {
                assert!(
                    (json["metrics"]["time_to_first_token_ms"].as_f64().unwrap() - 300.0).abs()
                        < 1e-9
                );
                assert_eq!(json["metrics"]["mean_itl_ms"], serde_json::Value::Null);
            } else {
                assert_eq!(json["metrics"], serde_json::Value::Null);
            }
        }
    }

    #[tokio::test]
    async fn completion_chunk_stream_maps_streaming_logprobs() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![1, 2, 3, 4, 5].into(),
                prompt_logprobs: None,
            }),
            Ok(decoded_delta(
                "h",
                vec![b'h' as u32],
                Some(DecodedLogprobs {
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
                None,
            )),
            Ok(decoded_delta(
                "",
                vec![b'!' as u32],
                Some(DecodedLogprobs {
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
                Some(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 5,
                        output_token_count: 2,
                        cached_token_count: 3,
                    },
                    finish_reason: FinishReason::Repetition(Some(StopReason::Text(
                        "repetition_detected".to_string(),
                    ))),
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                    sampling_mask: None,
                    timestamps: vllm_llm::RequestTimestamps {
                        queued_ts: 10.0,
                        scheduled_ts: 10.2,
                        first_token_ts: 10.5,
                        last_token_ts: 11.0,
                    },
                }),
            )),
        ]);

        let chunks = completion_chunk_stream(
            stream,
            "cmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions {
                enable_prompt_tokens_details: true,
                enable_per_request_metrics: true,
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

        assert!(chunks.windows(2).all(|pair| {
            Arc::ptr_eq(
                &chunk_response(&pair[0]).envelope,
                &chunk_response(&pair[1]).envelope,
            )
        }));

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

        match &chunks[2] {
            CompletionSseChunk::Chunk(chunk) => {
                assert_eq!(
                    chunk.choices[0].finish_reason.as_deref(),
                    Some("repetition")
                );
                assert_eq!(
                    chunk.choices[0].stop_reason,
                    Some(serde_json::json!("repetition_detected"))
                );
            }
            CompletionSseChunk::Usage(_) => panic!("expected regular chunk"),
        }

        match &chunks[3] {
            CompletionSseChunk::Usage(chunk) => {
                assert!(chunks[..3].iter().all(|chunk| chunk_response(chunk).metrics.is_none()));
                let metrics = chunk.metrics.as_ref().expect("per-request metrics");
                assert!((metrics.queue_time_ms.unwrap() - 200.0).abs() < 1e-9);
                assert!((metrics.tokens_per_second.unwrap() - 2.5).abs() < 1e-9);
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
    async fn completion_stream_serializes_requested_token_ids_on_every_choice() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![1, 2].into(),
                prompt_logprobs: None,
            }),
            Ok(decoded_delta(
                "ok",
                vec![7, 8],
                None,
                Some(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 2,
                        output_token_count: 2,
                        cached_token_count: 0,
                    },
                    finish_reason: FinishReason::Length,
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                    sampling_mask: None,
                    timestamps: Default::default(),
                }),
            )),
        ]);

        let chunks = completion_chunk_stream(
            stream,
            "cmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions::default(),
            ResponseOptions {
                return_token_ids: true,
                ..Default::default()
            },
        )
        .collect::<Vec<_>>()
        .await;

        let chunks: Vec<_> = chunks.into_iter().try_collect().expect("stream should succeed");
        let payloads: Vec<_> = chunks
            .iter()
            .map(|chunk| serde_json::to_value(chunk).expect("chunk should serialize"))
            .collect();

        assert_eq!(payloads.len(), 3);
        assert_eq!(
            payloads[0]["choices"][0]["prompt_token_ids"],
            serde_json::json!([1, 2])
        );
        assert_eq!(
            payloads[0]["choices"][0]["token_ids"],
            serde_json::json!([])
        );
        assert_eq!(
            payloads[1]["choices"][0]["token_ids"],
            serde_json::json!([7, 8])
        );
        assert_eq!(
            payloads[2]["choices"][0]["token_ids"],
            serde_json::json!([])
        );
    }

    #[tokio::test]
    async fn collect_completion_hides_internal_prompt_only_token() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: vec![1, 2].into(),
                prompt_logprobs: None,
            }),
            Ok(decoded_delta(
                " leaked",
                vec![3],
                None,
                Some(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 2,
                        output_token_count: 1,
                        cached_token_count: 0,
                    },
                    finish_reason: FinishReason::Length,
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                    sampling_mask: None,
                    timestamps: Default::default(),
                }),
            )),
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
            Ok(decoded_delta(
                " leaked",
                vec![3],
                None,
                Some(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 1,
                        output_token_count: 1,
                        cached_token_count: 0,
                    },
                    finish_reason: FinishReason::Length,
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                    sampling_mask: None,
                    timestamps: Default::default(),
                }),
            )),
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
            Ok(decoded_delta(
                " leaked",
                vec![3],
                None,
                Some(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 2,
                        output_token_count: 1,
                        cached_token_count: 0,
                    },
                    finish_reason: FinishReason::Length,
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                    sampling_mask: None,
                    timestamps: Default::default(),
                }),
            )),
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
            Ok(decoded_delta(
                " leaked",
                vec![3],
                None,
                Some(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 1,
                        output_token_count: 1,
                        cached_token_count: 0,
                    },
                    finish_reason: FinishReason::Length,
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                    sampling_mask: None,
                    timestamps: Default::default(),
                }),
            )),
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
            Ok(decoded_delta(
                " leaked",
                vec![3],
                None,
                Some(Finished {
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 2,
                        output_token_count: 1,
                        cached_token_count: 0,
                    },
                    finish_reason: FinishReason::Length,
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                    sampling_mask: None,
                    timestamps: Default::default(),
                }),
            )),
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
