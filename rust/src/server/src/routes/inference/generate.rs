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
use vllm_llm::{CollectedGenerateOutput, GenerateOutput, GenerateOutputStreamExt as _};

use self::convert::prepare_generate_request;
use self::types::{
    GenerateLogprob, GenerateRequest, GenerateResponse, GenerateResponseChoice,
    GenerateResponseStreamChoice, GenerateStreamResponse,
};
use crate::error::{ApiError, bail_server_error, server_error};
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
    let prepared = match prepare_generate_request(body, state.served_model_names(), request_context)
    {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    let request_span = tracing::info_span!(
        "generate",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let log_request = state.enable_log_requests;
    let include_logprobs = prepared.include_logprobs;
    let include_prompt_logprobs = prepared.include_prompt_logprobs;
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
            return server_error!(
                "failed to submit raw generate request: {}",
                error.to_report_string()
            )
            .into_response();
        }
    };

    if stream {
        let chunk_stream = generate_chunk_stream(
            raw_stream,
            prepared.request_id,
            log_request,
            prepared.include_usage,
            prepared.include_continuous_usage,
            include_logprobs,
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

    if log_request {
        info!(
            parent: &request_span,
            prompt_tokens = collected.prompt_token_ids.len(),
            output_tokens = collected.token_ids.len(),
            finish_reason = collected.finish_reason.as_str(),
            "generate finished"
        );
    }

    let response = match collect_generate(
        collected,
        prepared.request_id,
        include_logprobs,
        include_prompt_logprobs,
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
    log_request: bool,
    include_usage: bool,
    include_continuous_usage: bool,
    include_logprobs: bool,
    mut y: TryYielder<GenerateStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut prompt_tokens = 0_u32;
    let mut output_tokens = 0_u32;
    let mut started = false;

    while let Some(next) = stream.next().await {
        match next {
            Ok(output) => {
                if !started {
                    prompt_tokens = output
                        .prompt_info
                        .as_ref()
                        .map(|info| info.prompt_token_ids.len() as u32)
                        .unwrap_or_default();
                    started = true;
                }

                let token_ids = output.token_ids;
                output_tokens = output_tokens.saturating_add(token_ids.len() as u32);
                let finish_reason = output.finish_reason;

                if let Some(finish_reason) = finish_reason.as_ref()
                    && log_request
                {
                    info!(
                        stream = true,
                        prompt_tokens,
                        output_tokens,
                        finish_reason = finish_reason.as_str(),
                        "generate finished"
                    );
                }

                if token_ids.is_empty() {
                    continue;
                }

                let logprobs = if include_logprobs {
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
                        .then(|| Usage::from_counts(prompt_tokens, output_tokens)),
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
            usage: Some(Usage::from_counts(prompt_tokens, output_tokens)),
        })
        .await;
    }

    Ok(())
}

fn collect_generate(
    collected: CollectedGenerateOutput,
    request_id: String,
    include_logprobs: bool,
    include_prompt_logprobs: bool,
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

    Ok(GenerateResponse {
        request_id,
        choices: vec![GenerateResponseChoice {
            index: 0,
            logprobs,
            finish_reason: Some(collected.finish_reason.as_str().to_string()),
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
