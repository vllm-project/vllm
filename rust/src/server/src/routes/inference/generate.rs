// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
use tracing::{error, info, trace};
use tracing_futures::Instrument as _;
use vllm_engine_core_client::protocol::logprobs::Logprobs;
use vllm_llm::{
    CollectedGenerateOutput, FinishReason, GenerateOutput, GenerateOutputStreamExt as _, TokenUsage,
};
use vllm_text::{
    CollectedTextOutput, DecodedLogprobs, DecodedPromptLogprobs, DecodedTextEvent,
    TextOutputStreamExt as _,
};

use self::convert::{ResponseOptions, prepare_generate_request};
use self::types::{
    GenerateLogprob, GenerateResponse, GenerateResponseChoice, GenerateResponseStreamChoice,
    GenerateStreamResponse,
};
pub(crate) use self::types::{GenerateRequest, GenerateSamplingParams};
pub(crate) use self::validate::validate_request_compat;
use crate::config::ApiServerOptions;
use crate::error::{ApiError, bail_server_error, server_error, text_submit_error};
use crate::routes::openai::utils::logprobs::clamp_logprob;
use crate::routes::openai::utils::types::{ChatLogProbs, ChatLogProbsContent, TopLogProb, Usage};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::resolve_request_context;

/// Validate one token-in/token-out request and proxy it into the shared
/// `vllm-text` stack.
pub async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(mut body): ValidatedJson<GenerateRequest>,
) -> Response {
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(body.model.as_deref()).await;

    let mm_features = if let Some(parts) = body.content_parts.take() {
        match state.chat.prepare_media(parts, &mut body.token_ids).await {
            Ok(features) => features,
            Err(e) => {
                return ApiError::invalid_request(
                    format!("failed to resolve content_parts: {}", e.as_report()),
                    Some("content_parts"),
                )
                .into_response();
            }
        }
    } else {
        None
    };

    let prepared =
        match prepare_generate_request(body, &lora_resolution, request_context, mm_features) {
            Ok(prepared) => prepared,
            Err(error) => return error.into_response(),
        };
    let request_span = tracing::info_span!(
        "generate",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let api_server_options = state.api_server_options;
    let stream = prepared.stream;
    // Stop strings are matched on decoded text, so they are the only reason for
    // this token-in/token-out route to pay for detokenization. Without them the
    // raw token stream stays the fast path, mirroring Python, which builds no
    // detokenizer when it is not needed.
    if prepared.text_request.decode_options.stop_strings.is_none() {
        return raw_generate(state, prepared, request_span, api_server_options, stream).await;
    }

    let text_stream = match state
        .chat
        .text()
        .generate(prepared.text_request)
        .instrument(request_span.clone())
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return text_submit_error("failed to submit generate request", error).into_response();
        }
    };

    if stream {
        let chunk_stream = decoded_chunk_stream(
            text_stream,
            prepared.request_id,
            api_server_options,
            prepared.options,
        );
        let sse_stream = generate_sse_stream(chunk_stream).instrument(request_span);

        return Sse::new(sse_stream).into_response();
    }

    let collected = match text_stream.collect_output().instrument(request_span.clone()).await {
        Ok(collected) => collected,
        Err(error) => {
            return server_error!(
                "failed to collect generate response: {}",
                error.to_report_string()
            )
            .into_response();
        }
    };

    let response = match collect_decoded_generate(
        collected,
        prepared.request_id,
        api_server_options,
        prepared.options,
    ) {
        Ok(response) => response,
        Err(error) => return error.into_response(),
    };

    Json(response).into_response()
}

/// Serve one request over the raw token stream, without detokenization.
async fn raw_generate(
    state: Arc<AppState>,
    prepared: convert::PreparedRequest,
    request_span: tracing::Span,
    api_server_options: ApiServerOptions,
    stream: bool,
) -> Response {
    let raw_stream = match state
        .chat
        .text()
        .generate_raw(prepared.text_request)
        .instrument(request_span.clone())
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return text_submit_error("failed to submit raw generate request", error)
                .into_response();
        }
    };

    if stream {
        let chunk_stream = raw_chunk_stream(
            raw_stream,
            prepared.request_id,
            api_server_options,
            prepared.options,
        );
        let sse_stream = generate_sse_stream(chunk_stream).instrument(request_span);

        return Sse::new(sse_stream).into_response();
    }

    let collected = match raw_stream.collect_output().instrument(request_span.clone()).await {
        Ok(collected) => collected,
        Err(error) => {
            return server_error!(
                "failed to collect raw generate response: {}",
                error.to_report_string()
            )
            .into_response();
        }
    };

    let response = match collect_raw_generate(
        collected,
        prepared.request_id,
        api_server_options,
        prepared.options,
    ) {
        Ok(response) => response,
        Err(error) => return error.into_response(),
    };

    Json(response).into_response()
}

#[try_stream]
async fn decoded_chunk_stream(
    stream: impl Stream<Item = vllm_text::Result<DecodedTextEvent>>,
    request_id: String,
    ApiServerOptions {
        enable_log_requests,
        enable_prompt_tokens_details,
        ..
    }: ApiServerOptions,
    ResponseOptions {
        include_usage,
        include_continuous_usage,
        include_logprobs,
        // Ignored: generate streaming has no prompt-logprobs wire shape.
        include_prompt_logprobs: _,
    }: ResponseOptions,
    mut y: TryYielder<GenerateStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut usage = TokenUsage::default();

    while let Some(next) = stream.next().await {
        match next {
            Ok(DecodedTextEvent::Start {
                prompt_token_ids, ..
            }) => {
                usage.prompt_token_count = prompt_token_ids.len();
            }
            Ok(DecodedTextEvent::TextDelta {
                token_ids,
                logprobs,
                finished,
                ..
            }) => {
                usage.output_token_count = usage.output_token_count.saturating_add(token_ids.len());
                // Only the terminal event carries authoritative usage, so
                // continuous-usage chunks report no cached tokens until then,
                // as on the chat route (`ContinuousUsage::to_usage`).
                let finish_reason = finished.map(|finished| {
                    usage = finished.usage;
                    finished.finish_reason
                });

                if matches!(finish_reason.as_ref(), Some(FinishReason::Error)) {
                    bail_server_error!("Internal server error");
                }

                if let Some(finish_reason) = finish_reason.as_ref()
                    && enable_log_requests
                {
                    info!(
                        stream = true,
                        prompt_tokens = usage.prompt_token_count,
                        output_tokens = usage.output_token_count,
                        finish_reason = finish_reason.as_str(),
                        "generate finished"
                    );
                }

                if token_ids.is_empty() && finish_reason.is_none() {
                    continue;
                }

                let logprobs = if include_logprobs && !token_ids.is_empty() {
                    let logprobs = logprobs.as_ref().ok_or_else(|| {
                        server_error!(
                            "generate stream requested logprobs but generation returned none"
                        )
                    })?;
                    Some(decoded_logprobs_to_chat(logprobs)?)
                } else {
                    None
                };

                y.yield_ok(GenerateStreamResponse {
                    request_id: request_id.clone(),
                    choices: vec![GenerateResponseStreamChoice {
                        index: 0,
                        logprobs,
                        finish_reason: finish_reason.map(|reason| reason.as_str().to_string()),
                        token_ids,
                    }],
                    usage: include_continuous_usage
                        .then(|| Usage::from_token_usage(usage, enable_prompt_tokens_details)),
                })
                .await;
            }
            Err(error) => {
                error!(
                    error = %error.as_report(),
                    "generate stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }

    if include_usage {
        y.yield_ok(GenerateStreamResponse {
            request_id,
            choices: Vec::new(),
            usage: Some(Usage::from_token_usage(usage, enable_prompt_tokens_details)),
        })
        .await;
    }

    Ok(())
}

fn collect_decoded_generate(
    collected: CollectedTextOutput,
    request_id: String,
    ApiServerOptions {
        enable_log_requests,
        ..
    }: ApiServerOptions,
    ResponseOptions {
        // Ignored: non-streaming generate responses do not include usage.
        include_usage: _,
        // Ignored: continuous usage is a streaming-only option.
        include_continuous_usage: _,
        include_logprobs,
        include_prompt_logprobs,
    }: ResponseOptions,
) -> Result<GenerateResponse, ApiError> {
    let logprobs = if include_logprobs {
        let logprobs = collected.logprobs.as_ref().ok_or_else(|| {
            ApiError::server_error(
                "generate response requested logprobs but generation returned none".to_string(),
            )
        })?;
        Some(decoded_logprobs_to_chat(logprobs)?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        match collected.prompt_logprobs.as_ref() {
            Some(prompt_logprobs) => Some(decoded_prompt_logprobs_to_maps(prompt_logprobs)),
            // A single-token prompt has no scored positions; same mapping
            // as /v1/completions.
            None if collected.prompt_token_ids.len() == 1 => Some(vec![None]),
            None => {
                return Err(ApiError::server_error(
                    "generate response requested prompt_logprobs but generation returned none"
                        .to_string(),
                ));
            }
        }
    } else {
        None
    };
    let finish_reason = collected.finish_reason.as_str().to_string();

    if enable_log_requests {
        info!(
            prompt_tokens = collected.prompt_token_ids.len(),
            output_tokens = collected.token_ids.len(),
            %finish_reason,
            "generate finished"
        );
    }

    Ok(GenerateResponse {
        request_id,
        choices: vec![GenerateResponseChoice {
            index: 0,
            logprobs,
            finish_reason: Some(finish_reason),
            token_ids: collected.token_ids,
        }],
        prompt_logprobs,
        kv_transfer_params: collected.kv_transfer_params,
        ec_transfer_params: collected.ec_transfer_params,
    })
}

#[try_stream]
async fn raw_chunk_stream(
    stream: impl Stream<Item = vllm_llm::Result<GenerateOutput>>,
    request_id: String,
    ApiServerOptions {
        enable_log_requests,
        enable_prompt_tokens_details,
        ..
    }: ApiServerOptions,
    ResponseOptions {
        include_usage,
        include_continuous_usage,
        include_logprobs,
        // Ignored: raw generate streaming has no prompt-logprobs wire shape.
        include_prompt_logprobs: _,
    }: ResponseOptions,
    mut y: TryYielder<GenerateStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut prompt_tokens = None;
    let mut usage = TokenUsage::default();

    while let Some(next) = stream.next().await {
        match next {
            Ok(output) => {
                if prompt_tokens.is_none() {
                    prompt_tokens =
                        output.prompt_info.as_ref().map(|info| info.prompt_token_ids.len());
                }
                usage.prompt_token_count = prompt_tokens.unwrap_or_default();
                usage.cached_token_count = usage.cached_token_count.max(output.cached_token_count);

                let token_ids = output.token_ids;
                usage.output_token_count = usage.output_token_count.saturating_add(token_ids.len());
                let finish_reason = output.finish_reason;

                if matches!(finish_reason.as_ref(), Some(FinishReason::Error)) {
                    bail_server_error!("Internal server error");
                }

                if let Some(finish_reason) = finish_reason.as_ref()
                    && enable_log_requests
                {
                    info!(
                        stream = true,
                        prompt_tokens = usage.prompt_token_count,
                        output_tokens = usage.output_token_count,
                        finish_reason = finish_reason.as_str(),
                        "generate finished"
                    );
                }

                if token_ids.is_empty() && finish_reason.is_none() {
                    continue;
                }

                let logprobs = if include_logprobs && !token_ids.is_empty() {
                    let logprobs = output.logprobs.as_ref().ok_or_else(|| {
                        server_error!(
                            "raw generate stream requested logprobs but generation returned none"
                        )
                    })?;
                    Some(raw_logprobs_to_chat(logprobs)?)
                } else {
                    None
                };

                y.yield_ok(GenerateStreamResponse {
                    request_id: request_id.clone(),
                    choices: vec![GenerateResponseStreamChoice {
                        index: 0,
                        logprobs,
                        finish_reason: finish_reason.map(|reason| reason.as_str().to_string()),
                        token_ids,
                    }],
                    usage: include_continuous_usage
                        .then(|| Usage::from_token_usage(usage, enable_prompt_tokens_details)),
                })
                .await;
            }
            Err(error) => {
                error!(
                    error = %error.as_report(),
                    "raw generate stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }

    if include_usage {
        y.yield_ok(GenerateStreamResponse {
            request_id,
            choices: Vec::new(),
            usage: Some(Usage::from_token_usage(usage, enable_prompt_tokens_details)),
        })
        .await;
    }

    Ok(())
}

fn collect_raw_generate(
    collected: CollectedGenerateOutput,
    request_id: String,
    ApiServerOptions {
        enable_log_requests,
        ..
    }: ApiServerOptions,
    ResponseOptions {
        // Ignored: non-streaming raw generate responses do not include usage.
        include_usage: _,
        // Ignored: continuous usage is a streaming-only option.
        include_continuous_usage: _,
        include_logprobs,
        include_prompt_logprobs,
    }: ResponseOptions,
) -> Result<GenerateResponse, ApiError> {
    let logprobs = if include_logprobs {
        let logprobs = collected.logprobs.as_ref().ok_or_else(|| {
            ApiError::server_error(
                "raw generate response requested logprobs but generation returned none".to_string(),
            )
        })?;
        Some(raw_logprobs_to_chat(logprobs)?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        match collected.prompt_logprobs.as_ref() {
            Some(prompt_logprobs) => Some(raw_prompt_logprobs_to_maps(prompt_logprobs)),
            // A single-token prompt has no scored positions; same mapping
            // as /v1/completions.
            None if collected.prompt_token_ids.len() == 1 => Some(vec![None]),
            None => {
                return Err(ApiError::server_error(
                    "raw generate response requested prompt_logprobs but generation returned none"
                        .to_string(),
                ));
            }
        }
    } else {
        None
    };
    let finish_reason = collected.finish_reason.as_str().to_string();

    if enable_log_requests {
        info!(
            prompt_tokens = collected.prompt_token_ids.len(),
            output_tokens = collected.token_ids.len(),
            %finish_reason,
            "generate finished"
        );
    }

    Ok(GenerateResponse {
        request_id,
        choices: vec![GenerateResponseChoice {
            index: 0,
            logprobs,
            finish_reason: Some(finish_reason),
            token_ids: collected.token_ids,
        }],
        prompt_logprobs,
        kv_transfer_params: collected.kv_transfer_params,
        ec_transfer_params: collected.ec_transfer_params,
    })
}

/// One logprob candidate, as the raw and decoded streams each carry it.
///
/// The route renders token IDs rather than decoded strings, so both stream
/// shapes feed the same renderers through this accessor.
trait LogprobCandidate {
    fn token_id(&self) -> u32;
    fn logprob(&self) -> f32;
    fn rank(&self) -> u32;
}

impl LogprobCandidate for vllm_engine_core_client::protocol::logprobs::TokenLogprob {
    fn token_id(&self) -> u32 {
        self.token_id
    }

    fn logprob(&self) -> f32 {
        self.logprob
    }

    fn rank(&self) -> u32 {
        self.rank
    }
}

impl LogprobCandidate for vllm_text::DecodedTokenLogprob {
    fn token_id(&self) -> u32 {
        self.token_id
    }

    fn logprob(&self) -> f32 {
        self.logprob
    }

    fn rank(&self) -> u32 {
        self.rank
    }
}

/// Convert sample logprobs into the generate wire shape.
///
/// This keeps the route's own token-ID rendering and emits every candidate the
/// engine returned, rather than the shared chat converter's top-`k` policy.
fn raw_logprobs_to_chat(logprobs: &Logprobs) -> Result<ChatLogProbs, ApiError> {
    positions_to_chat_logprobs(logprobs.positions.iter().map(|position| &position.entries))
}

fn decoded_logprobs_to_chat(logprobs: &DecodedLogprobs) -> Result<ChatLogProbs, ApiError> {
    positions_to_chat_logprobs(logprobs.positions.iter().map(|position| &position.entries))
}

fn positions_to_chat_logprobs<'a, E: LogprobCandidate + 'a>(
    positions: impl Iterator<Item = &'a Vec<E>>,
) -> Result<ChatLogProbs, ApiError> {
    let content = positions
        .map(|entries| position_to_chat_logprobs_content(entries))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ChatLogProbs {
        content: Some(content),
    })
}

fn raw_prompt_logprobs_to_maps(
    prompt_logprobs: &Logprobs,
) -> Vec<Option<HashMap<u32, GenerateLogprob>>> {
    prompt_logprobs_to_maps(prompt_logprobs.positions.iter().map(|p| &p.entries))
}

fn decoded_prompt_logprobs_to_maps(
    prompt_logprobs: &DecodedPromptLogprobs,
) -> Vec<Option<HashMap<u32, GenerateLogprob>>> {
    prompt_logprobs_to_maps(prompt_logprobs.scored_positions.iter().map(|p| &p.entries))
}

/// The first prompt token has no left context to score, so it maps to `None`.
fn prompt_logprobs_to_maps<'a, E: LogprobCandidate + 'a>(
    positions: impl Iterator<Item = &'a Vec<E>>,
) -> Vec<Option<HashMap<u32, GenerateLogprob>>> {
    std::iter::once(None)
        .chain(positions.map(|entries| Some(position_to_logprob_map(entries))))
        .collect()
}

fn position_to_chat_logprobs_content<E: LogprobCandidate>(
    entries: &[E],
) -> Result<ChatLogProbsContent, ApiError> {
    let chosen = entries.first().ok_or_else(|| {
        ApiError::server_error(
            "generate logprobs position unexpectedly had no token candidates".to_string(),
        )
    })?;
    let token = format_token_id(chosen.token_id());

    Ok(ChatLogProbsContent {
        token: token.clone(),
        logprob: clamp_logprob(chosen.logprob()),
        bytes: Some(token.as_bytes().to_vec()),
        top_logprobs: entries
            .iter()
            .map(|entry| {
                let token = format_token_id(entry.token_id());
                TopLogProb {
                    token: token.clone(),
                    logprob: clamp_logprob(entry.logprob()),
                    bytes: Some(token.into_bytes()),
                }
            })
            .collect(),
    })
}

fn position_to_logprob_map<E: LogprobCandidate>(entries: &[E]) -> HashMap<u32, GenerateLogprob> {
    entries
        .iter()
        .map(|entry| {
            (
                entry.token_id(),
                GenerateLogprob {
                    logprob: clamp_logprob(entry.logprob()),
                    rank: Some(entry.rank()),
                    decoded_token: Some(format_token_id(entry.token_id())),
                },
            )
        })
        .collect()
}

fn format_token_id(token_id: u32) -> String {
    format!("token_id:{token_id}")
}

/// Convert one raw-generate chunk stream into SSE events.
#[try_stream]
async fn generate_sse_stream(
    stream: impl Stream<Item = Result<GenerateStreamResponse, ApiError>>,
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

fn to_sse_event(chunk: &GenerateStreamResponse) -> Event {
    let payload = serde_json::to_string(chunk).expect("generate chunk must serialize to JSON");
    trace!(payload, "generate emitting chunk");
    Event::default().data(payload)
}

fn to_error_sse_event(error: &ApiError) -> Event {
    let payload = serde_json::to_string(&error.to_error_response())
        .expect("ErrorResponse must serialize to JSON");
    trace!(payload, "generate emitting error");
    Event::default().data(payload)
}

fn done_sse_event() -> Event {
    trace!("generate emitting done");
    Event::default().data("[DONE]")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::{TryStreamExt as _, stream};
    use vllm_text::Finished;

    use super::*;

    #[tokio::test]
    async fn decoded_chunk_stream_reports_usage_from_the_terminal_event() {
        let stream = stream::iter(vec![
            Ok(DecodedTextEvent::Start {
                prompt_token_ids: Arc::from([11_u32, 22_u32]),
                prompt_logprobs: None,
            }),
            Ok(DecodedTextEvent::TextDelta {
                delta: "hi".to_string(),
                token_ids: vec![33],
                logprobs: None,
                finished: Some(Finished {
                    usage: TokenUsage {
                        prompt_token_count: 2,
                        output_token_count: 1,
                        cached_token_count: 2,
                    },
                    finish_reason: FinishReason::stop_eos(),
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                }),
            }),
        ]);

        let chunks: Vec<_> = decoded_chunk_stream(
            stream,
            "raw-stream".to_string(),
            ApiServerOptions {
                enable_prompt_tokens_details: true,
                ..Default::default()
            },
            ResponseOptions {
                include_usage: true,
                include_continuous_usage: true,
                ..Default::default()
            },
        )
        .try_collect()
        .await
        .expect("collect chunks");

        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0].usage.as_ref().expect("chunk usage").prompt_tokens,
            2
        );
        assert_eq!(
            chunks[0]
                .usage
                .as_ref()
                .expect("chunk usage")
                .prompt_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens),
            Some(2)
        );
        assert_eq!(
            chunks[1].usage.as_ref().expect("final usage").prompt_tokens,
            2
        );
        assert_eq!(
            chunks[1]
                .usage
                .as_ref()
                .expect("final usage")
                .prompt_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens),
            Some(2)
        );
    }

    #[test]
    fn collect_decoded_generate_maps_prompt_logprobs_for_single_token_prompt() {
        let output_without_payload = |prompt_token_ids: Vec<u32>| CollectedTextOutput {
            text: String::new(),
            prompt_logprobs: None,
            token_ids: vec![3],
            logprobs: None,
            finish_reason: FinishReason::stop_eos(),
            usage: vllm_llm::TokenUsage {
                prompt_token_count: prompt_token_ids.len(),
                output_token_count: 1,
                cached_token_count: 0,
            },
            kv_transfer_params: None,
            ec_transfer_params: None,
            prompt_token_ids: Arc::from(prompt_token_ids),
        };

        let response = collect_decoded_generate(
            output_without_payload(vec![9707]),
            "raw-1".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_prompt_logprobs: true,
                ..Default::default()
            },
        )
        .expect("single-token prompt without payload maps to [None]");
        let prompt_logprobs = response.prompt_logprobs.expect("prompt logprobs present");
        assert_eq!(prompt_logprobs.len(), 1);
        assert!(prompt_logprobs[0].is_none());

        collect_decoded_generate(
            output_without_payload(vec![9707, 11]),
            "raw-2".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_prompt_logprobs: true,
                ..Default::default()
            },
        )
        .expect_err("multi-token prompt without payload is an engine failure");
    }
}
