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
use vllm_engine_core_client::protocol::logprobs::{Logprobs, PositionLogprobs};
use vllm_llm::{
    CollectedGenerateOutput, FinishReason, GenerateOutput, GenerateOutputStreamExt as _,
    RequestTimingStats, TokenUsage,
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

    let prefill_metrics = body
        .kv_transfer_params
        .as_ref()
        .and_then(|params| params.get("prefill_metrics"))
        .cloned();
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
        let chunk_stream = generate_chunk_stream(
            raw_stream,
            prepared.request_id,
            api_server_options,
            prepared.options,
            prefill_metrics,
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

    let response = match collect_generate(
        collected,
        prepared.request_id,
        api_server_options,
        prepared.options,
        prefill_metrics,
    ) {
        Ok(response) => response,
        Err(error) => return error.into_response(),
    };

    Json(response).into_response()
}

#[try_stream]
async fn generate_chunk_stream(
    stream: impl Stream<Item = vllm_llm::Result<GenerateOutput>>,
    request_id: String,
    ApiServerOptions {
        enable_log_requests,
        enable_per_request_metrics,
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
    prefill_metrics: Option<serde_json::Value>,
    mut y: TryYielder<GenerateStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut prompt_tokens = None;
    let mut usage = TokenUsage::default();
    let mut metrics = serde_json::Map::new();

    while let Some(next) = stream.next().await {
        match next {
            Ok(output) => {
                if output.finish_reason.is_some() {
                    metrics = build_per_request_metrics(
                        enable_per_request_metrics,
                        output.request_timings,
                        output.remote_kv_wait_time,
                        output.kv_transfer_metrics.as_ref(),
                        prefill_metrics.as_ref(),
                    );
                    metrics.retain(|_, value| !value.is_null());
                }
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
                    Some(raw_logprobs_to_openai_chat(logprobs)?)
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
                    metrics: None,
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
            metrics: (!metrics.is_empty()).then_some(metrics),
        })
        .await;
    }

    Ok(())
}

fn collect_generate(
    collected: CollectedGenerateOutput,
    request_id: String,
    ApiServerOptions {
        enable_log_requests,
        enable_per_request_metrics,
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
    prefill_metrics: Option<serde_json::Value>,
) -> Result<GenerateResponse, ApiError> {
    let logprobs = if include_logprobs {
        let logprobs = collected.logprobs.as_ref().ok_or_else(|| {
            ApiError::server_error(
                "raw generate response requested logprobs but generation returned none".to_string(),
            )
        })?;
        Some(raw_logprobs_to_openai_chat(logprobs)?)
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

    let metrics = build_per_request_metrics(
        enable_per_request_metrics,
        collected.request_timings,
        collected.remote_kv_wait_time,
        collected.kv_transfer_metrics.as_ref(),
        prefill_metrics.as_ref(),
    );
    let mut kv_transfer_params = collected.kv_transfer_params;
    if enable_per_request_metrics {
        if let Some(serde_json::Value::Object(params)) = kv_transfer_params.as_mut() {
            let prefill = [
                "queue_time_ms",
                "time_to_first_token_ms",
                "kv_allocation_wait_time_ms",
                "kv_initial_queue_wait_time_ms",
            ]
            .into_iter()
            .map(|name| {
                (
                    name.to_string(),
                    metrics.get(name).cloned().unwrap_or(serde_json::Value::Null),
                )
            })
            .collect();
            params.insert("prefill_metrics".into(), serde_json::Value::Object(prefill));
        }
    }
    let metrics = (!metrics.is_empty()).then(|| {
        let mut metrics = metrics;
        metrics.insert("speculative_decoding".into(), serde_json::Value::Null);
        for name in [
            "time_to_first_token_ms",
            "generation_time_ms",
            "queue_time_ms",
            "mean_itl_ms",
            "tokens_per_second",
            "remote_kv_wait_time_ms",
            "kv_initial_queue_wait_time_ms",
            "kv_post_receive_queue_wait_time_ms",
            "kv_allocation_wait_time_ms",
            "kv_handshake_wait_worker_time_ms",
            "kv_transfer_worker_time_ms",
            "kv_transfer_post_worker_time_ms",
            "kv_transfer_bytes",
            "prefill_queue_time_ms",
            "prefill_time_to_first_token_ms",
            "prefill_kv_allocation_wait_time_ms",
            "prefill_kv_initial_queue_wait_time_ms",
        ] {
            metrics.entry(name).or_insert(serde_json::Value::Null);
        }
        metrics
    });
    Ok(GenerateResponse {
        request_id,
        choices: vec![GenerateResponseChoice {
            index: 0,
            logprobs,
            finish_reason: Some(finish_reason),
            token_ids: collected.token_ids,
        }],
        prompt_logprobs,
        kv_transfer_params,
        ec_transfer_params: collected.ec_transfer_params,
        metrics,
    })
}

fn build_per_request_metrics(
    enabled: bool,
    timings: Option<RequestTimingStats>,
    remote_kv_wait_time: Option<f64>,
    metrics: Option<&serde_json::Value>,
    prefill_metrics: Option<&serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    if !enabled {
        return serde_json::Map::new();
    }
    let mut metrics = metrics.and_then(serde_json::Value::as_object).cloned().unwrap_or_default();
    if let Some(timings) = timings {
        let duration = |start, end| (start > 0.0 && end > 0.0).then_some((end - start) * 1000.0);
        let generation_time = duration(timings.first_token_ts, timings.last_token_ts);
        for (name, value) in [
            (
                "time_to_first_token_ms",
                duration(timings.scheduled_ts, timings.first_token_ts),
            ),
            ("generation_time_ms", generation_time),
            (
                "queue_time_ms",
                duration(timings.queued_ts, timings.scheduled_ts),
            ),
            (
                "mean_itl_ms",
                generation_time
                    .filter(|_| timings.num_generation_tokens > 1)
                    .map(|elapsed| elapsed / (timings.num_generation_tokens - 1) as f64),
            ),
            (
                "tokens_per_second",
                duration(timings.scheduled_ts, timings.last_token_ts)
                    .filter(|elapsed| *elapsed > 0.0)
                    .map(|elapsed| timings.num_generation_tokens as f64 * 1000.0 / elapsed),
            ),
        ] {
            metrics.insert(name.into(), serde_json::json!(value));
        }
    }
    if let Some(wait) = remote_kv_wait_time {
        metrics.insert(
            "remote_kv_wait_time_ms".into(),
            serde_json::json!(wait * 1000.0),
        );
    }
    if let Some(prefill) = prefill_metrics {
        for name in [
            "queue_time_ms",
            "time_to_first_token_ms",
            "kv_allocation_wait_time_ms",
            "kv_initial_queue_wait_time_ms",
        ] {
            if let Some(value) = prefill.get(name) {
                metrics.insert(format!("prefill_{name}"), value.clone());
            }
        }
    }
    metrics
}

fn raw_logprobs_to_openai_chat(logprobs: &Logprobs) -> Result<ChatLogProbs, ApiError> {
    let content = logprobs
        .positions
        .iter()
        .map(position_to_chat_logprobs_content)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ChatLogProbs {
        content: Some(content),
    })
}

fn raw_prompt_logprobs_to_maps(
    prompt_logprobs: &Logprobs,
) -> Vec<Option<HashMap<u32, GenerateLogprob>>> {
    std::iter::once(None)
        .chain(
            prompt_logprobs
                .positions
                .iter()
                .map(|position| Some(position_to_logprob_map(position))),
        )
        .collect()
}

fn position_to_chat_logprobs_content(
    position: &PositionLogprobs,
) -> Result<ChatLogProbsContent, ApiError> {
    let chosen = position.entries.first().ok_or_else(|| {
        ApiError::server_error(
            "raw generate logprobs position unexpectedly had no token candidates".to_string(),
        )
    })?;
    let token = format_token_id(chosen.token_id);

    Ok(ChatLogProbsContent {
        token: token.clone(),
        logprob: clamp_logprob(chosen.logprob),
        bytes: Some(token.as_bytes().to_vec()),
        top_logprobs: position
            .entries
            .iter()
            .map(|entry| {
                let token = format_token_id(entry.token_id);
                TopLogProb {
                    token: token.clone(),
                    logprob: clamp_logprob(entry.logprob),
                    bytes: Some(token.into_bytes()),
                }
            })
            .collect(),
    })
}

fn position_to_logprob_map(position: &PositionLogprobs) -> HashMap<u32, GenerateLogprob> {
    position
        .entries
        .iter()
        .map(|entry| {
            (
                entry.token_id,
                GenerateLogprob {
                    logprob: clamp_logprob(entry.logprob),
                    rank: Some(entry.rank),
                    decoded_token: Some(format_token_id(entry.token_id)),
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
    trace!(?chunk, "generate emitting chunk");
    Event::default()
        .json_data(chunk)
        .expect("generate chunk must serialize to JSON")
}

fn to_error_sse_event(error: &ApiError) -> Event {
    let response = error.to_error_response();
    trace!(?response, "generate emitting error");
    Event::default()
        .json_data(response)
        .expect("ErrorResponse must serialize to JSON")
}

fn done_sse_event() -> Event {
    trace!("generate emitting done");
    Event::default().data("[DONE]")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::{TryStreamExt as _, stream};
    use vllm_llm::GeneratePromptInfo;

    use super::*;

    #[tokio::test]
    async fn pd_metrics_preserve_phase_owners_and_response_mode() {
        let output = GenerateOutput {
            request_id: "pd-test".to_string(),
            prompt_info: None,
            token_ids: vec![33, 44],
            logprobs: None,
            finish_reason: Some(FinishReason::Length),
            cached_token_count: 0,
            kv_transfer_params: Some(serde_json::json!({"do_remote_decode": true})),
            ec_transfer_params: None,
            remote_kv_wait_time: Some(0.125),
            kv_transfer_metrics: Some(
                serde_json::json!({"kv_transfer_bytes": 64, "kv_allocation_wait_time_ms": 7.0}),
            ),
            request_timings: Some(RequestTimingStats {
                queued_ts: 1.0,
                scheduled_ts: 3.0,
                first_token_ts: 4.0,
                last_token_ts: 5.0,
                num_generation_tokens: 2,
            }),
        };
        for enabled in [false, true] {
            let options = ApiServerOptions {
                enable_per_request_metrics: enabled,
                ..Default::default()
            };
            let incoming = Some(serde_json::json!({"queue_time_ms": 2.5}));
            let chunks: Vec<_> = generate_chunk_stream(
                stream::iter([Ok(output.clone())]),
                "pd-test".to_string(),
                options,
                ResponseOptions {
                    include_usage: true,
                    ..Default::default()
                },
                incoming.clone(),
            )
            .try_collect()
            .await
            .expect("stream chunks");
            assert_eq!(chunks.len(), 2);
            assert!(chunks[0].metrics.is_none());
            let collected = stream::iter([Ok(output.clone())])
                .collect_output()
                .await
                .expect("collect output");
            let response = collect_generate(
                collected,
                "pd-test".to_string(),
                options,
                ResponseOptions::default(),
                incoming,
            )
            .expect("response");
            if enabled {
                let metrics = response.metrics.expect("metrics enabled");
                assert_eq!(metrics.len(), 18);
                assert_eq!(metrics["queue_time_ms"], 2000.0);
                assert_eq!(metrics["prefill_queue_time_ms"], 2.5);
                assert_eq!(metrics["remote_kv_wait_time_ms"], 125.0);
                assert_eq!(metrics["kv_transfer_bytes"], 64);
                assert!(metrics["kv_handshake_wait_worker_time_ms"].is_null());
                let streamed = chunks[1].metrics.as_ref().expect("final metrics");
                assert!(streamed.values().all(|value| !value.is_null()));
                assert_eq!(streamed["prefill_queue_time_ms"], 2.5);
                assert_eq!(
                    response.kv_transfer_params.expect("P metadata")["prefill_metrics"]["queue_time_ms"],
                    2000.0
                );
            } else {
                assert!(response.metrics.is_none());
                assert!(chunks[1].metrics.is_none());
            }
        }
    }

    #[tokio::test]
    async fn generate_chunk_stream_captures_late_prompt_info() {
        let stream = stream::iter(vec![
            Ok(GenerateOutput {
                request_id: String::new(),
                prompt_info: None,
                token_ids: Vec::new(),
                logprobs: None,
                finish_reason: None,
                cached_token_count: 0,
                kv_transfer_params: None,
                ec_transfer_params: None,
                remote_kv_wait_time: None,
                kv_transfer_metrics: None,
                request_timings: None,
            }),
            Ok(GenerateOutput {
                request_id: String::new(),
                prompt_info: Some(GeneratePromptInfo {
                    prompt_token_ids: Arc::from([11_u32, 22_u32]),
                    prompt_logprobs: None,
                }),
                token_ids: vec![33],
                logprobs: None,
                finish_reason: Some(FinishReason::stop_eos()),
                cached_token_count: 2,
                kv_transfer_params: None,
                ec_transfer_params: None,
                remote_kv_wait_time: None,
                kv_transfer_metrics: None,
                request_timings: None,
            }),
        ]);

        let chunks: Vec<_> = generate_chunk_stream(
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
            None,
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
    fn collect_generate_maps_prompt_logprobs_for_single_token_prompt() {
        let output_without_payload = |prompt_token_ids: Vec<u32>| CollectedGenerateOutput {
            request_id: "raw-1".to_string(),
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
            remote_kv_wait_time: None,
            kv_transfer_metrics: None,
            request_timings: None,
            prompt_token_ids,
        };

        let response = collect_generate(
            output_without_payload(vec![9707]),
            "raw-1".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_prompt_logprobs: true,
                ..Default::default()
            },
            None,
        )
        .expect("single-token prompt without payload maps to [None]");
        let prompt_logprobs = response.prompt_logprobs.expect("prompt logprobs present");
        assert_eq!(prompt_logprobs.len(), 1);
        assert!(prompt_logprobs[0].is_none());

        collect_generate(
            output_without_payload(vec![9707, 11]),
            "raw-2".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_prompt_logprobs: true,
                ..Default::default()
            },
            None,
        )
        .expect_err("multi-token prompt without payload is an engine failure");
    }
}
