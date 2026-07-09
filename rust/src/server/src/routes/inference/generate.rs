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
    CollectedGenerateOutput, FinishReason, GenerateOutput, GenerateOutputStreamExt as _, TokenUsage,
};

use self::convert::{ResponseOptions, prepare_generate_request};
use self::types::{
    GenerateLogprob, GenerateRequest, GenerateResponse, GenerateResponseChoice,
    GenerateResponseStreamChoice, GenerateStreamResponse,
};
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
    ValidatedJson(body): ValidatedJson<GenerateRequest>,
) -> Response {
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(body.model.as_deref()).await;
    let prepared = match prepare_generate_request(body, &lora_resolution, request_context) {
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

fn collect_generate(
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
        Some(raw_logprobs_to_openai_chat(logprobs)?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        let prompt_logprobs = collected.prompt_logprobs.as_ref().ok_or_else(|| {
            ApiError::server_error(
                "raw generate response requested prompt_logprobs but generation returned none"
                    .to_string(),
            )
        })?;
        Some(raw_prompt_logprobs_to_maps(prompt_logprobs))
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
    })
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
    use vllm_llm::GeneratePromptInfo;

    use super::*;

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
}
