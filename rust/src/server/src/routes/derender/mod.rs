// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Postprocessing counterpart to the /render endpoints: turn
//! token-in/token-out `GenerateResponse`s back into OpenAI chat/completion
//! responses without a GPU.
//!
//! Mirrors the Python vLLM implementation in
//! `vllm/entrypoints/scale_out/derender/` (api_router.py, serving.py) and
//! `vllm/renderers/online_derenderer.py`.
//!
//! This is phase 1 of the derender port: shared detokenization/state plus the
//! plain non-streaming endpoints.
//! TODO: phase 2 adds reasoning/tool-call parsing of the detokenized output;
//! phase 3 adds the streaming endpoints (a `stream: true` body fails
//! deserialization with 400 until then).
//!
//! Deliberate deviations from the Python implementation:
//! - Non-streaming completion derender honours `completion_request.echo`
//!   (including the `max_tokens=0` prompt-only case); Python ignores `echo`.

mod detok;
mod logprobs;
mod types;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use thiserror_ext::AsReport as _;
use tracing::{debug, warn};
use vllm_chat::ChatRequestProcessor;
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::{Prompt, TextRequestProcessor};

use self::types::{
    DerenderChatCompletionResponse, DerenderChatRequest, DerenderChatRequestUnion,
    DerenderCompletionRequest, DerenderCompletionRequestUnion,
};
use crate::error::{ApiError, bail_invalid_request, server_error};
use crate::lora::LoraModelResolution;
use crate::render::RenderState;
use crate::routes::inference::generate::GenerateResponse;
use crate::routes::openai::chat_completions::{
    AssistantRole, ChatCompletionChoice, ChatCompletionMessage,
};
use crate::routes::openai::utils::logprobs::{append_openai_logprobs, text_len};
use crate::routes::openai::utils::types::Usage;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::openai::{CompletionChoice, CompletionResponse, completion_echo_text};
use crate::state::AppState;
use crate::utils::{ResolvedRequestContext, resolve_request_context, unix_timestamp};

/// TODO: plumb `VLLM_MAX_N_SEQUENCES` through the server configuration;
/// Python reads `envs.VLLM_MAX_N_SEQUENCES` (default 16384).
const MAX_N_SEQUENCES: usize = 16384;

/// Everything a derender handler needs, abstracted over the render-only and
/// engine-backed servers.
pub(crate) struct DerenderContext<'a> {
    /// Public model names accepted by the frontend (including LoRA names).
    lora_resolution: LoraModelResolution,
    /// Primary public model name, echoed when the request omits `model`.
    primary_model_name: &'a str,
    /// Backend model id used for parser resolution.
    // TODO: phase 2 (parsing) and phase 3 (streaming) read this again.
    #[allow(dead_code)]
    model_id: &'a str,
    tokenizer: DynTokenizer,
    max_model_len: u32,
    max_logprobs: i32,
    /// TODO: phase 2 (parser replay) reads this again.
    #[allow(dead_code)]
    text: &'a TextRequestProcessor,
    /// TODO: phase 2 (parsing) and phase 3 (streaming) read this again.
    #[allow(dead_code)]
    chat: &'a ChatRequestProcessor,
}

fn render_context(state: &RenderState) -> DerenderContext<'_> {
    DerenderContext {
        lora_resolution: LoraModelResolution {
            model_names: state.served_model_names.clone(),
            lora_request: None,
        },
        primary_model_name: &state.served_model_names[0],
        model_id: state.text.model_id(),
        tokenizer: state.text.tokenizer(),
        max_model_len: state.text.max_model_len(),
        max_logprobs: state.text.max_logprobs(),
        text: &state.text,
        chat: &state.chat,
    }
}

async fn app_context<'a>(state: &'a AppState, model: Option<&str>) -> DerenderContext<'a> {
    let text = state.chat.text().request_processor();
    DerenderContext {
        lora_resolution: state.resolve_model_with_loras(model).await,
        primary_model_name: state.primary_model_name(),
        model_id: state.chat.model_id(),
        tokenizer: text.tokenizer(),
        max_model_len: text.max_model_len(),
        max_logprobs: text.max_logprobs(),
        text,
        chat: state.chat.request_processor(),
    }
}

/// Python `BaseServing._check_model`: an omitted model resolves the served
/// name; an unknown one is a 404.
fn check_model(ctx: &DerenderContext<'_>, model: Option<&str>) -> Result<(), ApiError> {
    if let Some(model) = model
        && !ctx.lora_resolution.model_names.iter().any(|name| name == model)
    {
        return Err(ApiError::model_not_found(model.to_string()));
    }
    Ok(())
}

fn response_model_name(ctx: &DerenderContext<'_>, model: Option<String>) -> String {
    model.unwrap_or_else(|| ctx.primary_model_name.to_string())
}

/// Reject derender payloads that exceed resource bounds.
///
/// Runs before any tokenizer.decode() or parser invocation to prevent
/// CPU/memory exhaustion from oversized caller-supplied token structures.
fn validate_derender_bounds(
    generate_responses: &[GenerateResponse],
    max_model_len: u32,
    max_logprobs: i32,
) -> Result<(), ApiError> {
    let max_model_len = max_model_len as usize;

    if generate_responses.len() > MAX_N_SEQUENCES {
        bail_invalid_request!(
            "generate_responses count ({}) exceeds server maximum ({}). \
             Set VLLM_MAX_N_SEQUENCES to increase this limit.",
            generate_responses.len(),
            MAX_N_SEQUENCES
        );
    }

    for response in generate_responses {
        if response.choices.len() > MAX_N_SEQUENCES {
            bail_invalid_request!(
                "choices count ({}) in response '{}' exceeds server maximum ({}).",
                response.choices.len(),
                response.request_id,
                MAX_N_SEQUENCES
            );
        }

        for choice in &response.choices {
            if choice.token_ids.len() > max_model_len {
                bail_invalid_request!(
                    "token_ids length ({}) in choice {} exceeds max_model_len ({}).",
                    choice.token_ids.len(),
                    choice.index,
                    max_model_len
                );
            }
            if let Some(content) = choice.logprobs.as_ref().and_then(|l| l.content.as_ref()) {
                if content.len() > max_model_len {
                    bail_invalid_request!(
                        "logprobs.content length ({}) in choice {} exceeds max_model_len ({}).",
                        content.len(),
                        choice.index,
                        max_model_len
                    );
                }
                for entry in content {
                    if max_logprobs >= 0 && entry.top_logprobs.len() > max_logprobs as usize {
                        bail_invalid_request!(
                            "top_logprobs count ({}) in choice {} exceeds max_logprobs ({}).",
                            entry.top_logprobs.len(),
                            choice.index,
                            max_logprobs
                        );
                    }
                }
            }
        }

        if let Some(prompt_logprobs) = &response.prompt_logprobs
            && prompt_logprobs.len() > max_model_len
        {
            bail_invalid_request!(
                "prompt_logprobs length ({}) in response '{}' exceeds max_model_len ({}).",
                prompt_logprobs.len(),
                response.request_id,
                max_model_len
            );
        }
    }

    Ok(())
}

/// Postprocess a GenerateResponse into a chat completion response.
///
/// Non-streaming only: expects the complete GenerateResponse with all token
/// IDs present. Phase 1 always plain-detokenizes, honouring the request's
/// `skip_special_tokens` (default true when no request was given).
/// TODO: phase 2 replays the output through the configured reasoning/tool
/// parser (splitting reasoning, content and tool_calls) when `chat_request`
/// is supplied, mirroring Python's `parser.parse()` one-shot extraction.
async fn derender_chat(
    ctx: &DerenderContext<'_>,
    request: DerenderChatRequest,
    // Used by the phase-2 parser replay.
    _request_context: ResolvedRequestContext,
) -> Result<DerenderChatCompletionResponse, ApiError> {
    check_model(ctx, request.model.as_deref())?;
    validate_derender_bounds(
        std::slice::from_ref(&request.generate_response),
        ctx.max_model_len,
        ctx.max_logprobs,
    )?;

    let model_name = response_model_name(ctx, request.model.clone());
    let response = &request.generate_response;
    let mut choices = Vec::with_capacity(response.choices.len());

    for choice in &response.choices {
        if choice.token_ids.is_empty() {
            bail_invalid_request!("choice {} has empty or null token_ids", choice.index);
        }

        let resolved_logprobs = choice
            .logprobs
            .as_ref()
            .map(|logprobs| logprobs::resolve_logprobs(logprobs, &ctx.tokenizer))
            .transpose()?;

        let skip_special = request
            .chat_request
            .as_ref()
            .map(|request| request.skip_special_tokens)
            .unwrap_or(true);
        let decoded_text = ctx
            .tokenizer
            .decode(&choice.token_ids, skip_special)
            .map_err(|error| server_error!("derender decode failed: {}", error.as_report()))?;

        choices.push(ChatCompletionChoice {
            index: choice.index,
            message: ChatCompletionMessage {
                role: AssistantRole,
                content: Some(decoded_text),
                tool_calls: Vec::new(),
                reasoning: None,
            },
            logprobs: resolved_logprobs,
            finish_reason: choice.finish_reason.clone(),
            stop_reason: None,
            token_ids: None,
        });
    }

    let prompt_tokens = request.prompt_tokens.unwrap_or(0);
    let completion_tokens: usize =
        response.choices.iter().map(|choice| choice.token_ids.len()).sum();
    debug!(
        request_id = %response.request_id,
        model = %model_name,
        choices = choices.len(),
        completion_tokens,
        "derender_chat"
    );

    Ok(DerenderChatCompletionResponse {
        id: response.request_id.clone(),
        object: "chat.completion".to_string(),
        created: unix_timestamp(),
        model: model_name,
        choices,
        usage: Usage {
            prompt_tokens,
            total_tokens: checked_total_tokens(prompt_tokens, completion_tokens)?,
            completion_tokens: Some(completion_tokens),
            prompt_tokens_details: None,
        },
        prompt_logprobs: response.prompt_logprobs.clone(),
        kv_transfer_params: response.kv_transfer_params.clone(),
        // The Python derender endpoint does not pass ec_transfer_params through.
        ec_transfer_params: None,
    })
}

/// Postprocess a list of GenerateResponses into a completion response.
///
/// Non-streaming only. Mirrors the multi-prompt completions case: one
/// GenerateResponse per prompt, parallel to the list[GenerateRequest] from
/// /v1/completions/render.
fn derender_completion(
    ctx: &DerenderContext<'_>,
    request: DerenderCompletionRequest,
) -> Result<CompletionResponse, ApiError> {
    check_model(ctx, request.model.as_deref())?;

    if request.generate_responses.is_empty() {
        bail_invalid_request!("generate_responses must not be empty");
    }
    if let Some(prompt_tokens) = &request.prompt_tokens
        && prompt_tokens.len() != request.generate_responses.len()
    {
        bail_invalid_request!(
            "prompt_tokens length ({}) must equal generate_responses length ({})",
            prompt_tokens.len(),
            request.generate_responses.len()
        );
    }

    validate_derender_bounds(
        &request.generate_responses,
        ctx.max_model_len,
        ctx.max_logprobs,
    )?;

    let skip_special = request
        .completion_request
        .as_ref()
        .map(|request| request.skip_special_tokens)
        .unwrap_or(true);

    // Honour `completion_request.echo` like the normal completions path
    // (Python derender ignores `echo`): prepend the prompt to each choice's
    // text, and for prompt-only requests (echo + max_tokens=0) expose just
    // the prompt, hiding the one internally generated token.
    let completion_request = request.completion_request.as_ref();
    let echo = completion_request
        .map(|request| completion_echo_text(request, ctx.tokenizer.as_ref()))
        .transpose()?
        .flatten();
    let prompt_only =
        echo.is_some() && completion_request.is_some_and(|request| request.max_tokens == Some(0));
    let echo_prompt_token_ids = match completion_request.filter(|_| echo.is_some()) {
        Some(request) => Some(match &request.prompt {
            Prompt::Text(text) => {
                ctx.tokenizer.encode(text, request.add_special_tokens).map_err(|error| {
                    ApiError::invalid_request(
                        format!("Failed to tokenize prompt for echo: {}", error.as_report()),
                        Some("prompt"),
                    )
                })?
            }
            Prompt::TokenIds(token_ids) => token_ids.clone(),
        }),
        None => None,
    };

    let mut choices = Vec::new();
    let mut total_prompt_tokens = 0;
    let mut total_completion_tokens = 0;
    let mut index = 0;

    for (response_idx, response) in request.generate_responses.iter().enumerate() {
        let prompt_tokens =
            request.prompt_tokens.as_ref().map(|tokens| tokens[response_idx]).unwrap_or(0);
        for choice in &response.choices {
            if choice.token_ids.is_empty() {
                bail_invalid_request!(
                    "choice {} in response {} has empty or null token_ids",
                    choice.index,
                    response.request_id
                );
            }

            let decoded_text = ctx
                .tokenizer
                .decode(&choice.token_ids, skip_special)
                .map_err(|error| server_error!("derender decode failed: {}", error.as_report()))?;
            let text = match &echo {
                None => decoded_text,
                Some(prompt) if prompt_only => prompt.clone(),
                Some(prompt) => format!("{prompt}{decoded_text}"),
            };
            let mut logprobs = choice
                .logprobs
                .as_ref()
                .map(|logprobs| {
                    logprobs::resolve_logprobs(logprobs, &ctx.tokenizer)
                        .map(|resolved| logprobs::chat_logprobs_to_completion(&resolved))
                })
                .transpose()?;
            if let Some(prompt) = &echo {
                if prompt_only {
                    // Choice logprobs would describe the hidden internal
                    // token; echo back prompt logprobs like the normal path.
                    logprobs = match &response.prompt_logprobs {
                        Some(prompt_logprobs) => Some(logprobs::prompt_logprobs_to_completion(
                            prompt_logprobs,
                            echo_prompt_token_ids.as_deref().unwrap_or(&[]),
                            &ctx.tokenizer,
                        )?),
                        None => None,
                    };
                } else if let Some(completion_logprobs) = logprobs {
                    logprobs = Some(match &response.prompt_logprobs {
                        // The render step enables prompt logprobs for echo
                        // requests, so prepend them exactly like the normal
                        // echoed completions path.
                        Some(prompt_logprobs) => {
                            let prompt_logprobs = logprobs::prompt_logprobs_to_completion(
                                prompt_logprobs,
                                echo_prompt_token_ids.as_deref().unwrap_or(&[]),
                                &ctx.tokenizer,
                            )?;
                            let completion_start = prompt_logprobs
                                .text_offset
                                .last()
                                .zip(prompt_logprobs.tokens.last())
                                .map(|(&offset, token)| offset.saturating_add(text_len(token)))
                                .unwrap_or(0);
                            let mut completion_logprobs = completion_logprobs;
                            logprobs::shift_text_offsets(
                                &mut completion_logprobs,
                                completion_start,
                            );
                            append_openai_logprobs(prompt_logprobs, completion_logprobs)
                        }
                        None => {
                            let mut completion_logprobs = completion_logprobs;
                            logprobs::shift_text_offsets(
                                &mut completion_logprobs,
                                text_len(prompt),
                            );
                            completion_logprobs
                        }
                    });
                }
            }

            choices.push(CompletionChoice {
                index,
                text,
                logprobs,
                finish_reason: choice.finish_reason.clone(),
                stop_reason: None,
                prompt_logprobs: None,
                token_ids: None,
                prompt_token_ids: None,
            });
            total_completion_tokens += choice.token_ids.len();
            index += 1;
        }
        total_prompt_tokens = checked_total_tokens(total_prompt_tokens, prompt_tokens)?;
    }

    let first = &request.generate_responses[0];
    let kv_params = first.kv_transfer_params.clone();
    let kv_params = if request.generate_responses[1..]
        .iter()
        .any(|response| response.kv_transfer_params != kv_params)
    {
        warn!(
            "derender_completion: kv_transfer_params differ across responses; \
             setting to None on the aggregated response"
        );
        None
    } else {
        kv_params
    };

    let model_name = response_model_name(ctx, request.model.clone());
    debug!(
        request_id = %first.request_id,
        model = %model_name,
        choices = choices.len(),
        total_completion_tokens,
        "derender_completion"
    );

    Ok(CompletionResponse {
        id: first.request_id.clone(),
        object: "text_completion".to_string(),
        created: unix_timestamp(),
        model: model_name,
        choices,
        usage: Some(Usage {
            prompt_tokens: total_prompt_tokens,
            total_tokens: checked_total_tokens(total_prompt_tokens, total_completion_tokens)?,
            completion_tokens: Some(total_completion_tokens),
            prompt_tokens_details: None,
        }),
        system_fingerprint: None,
        kv_transfer_params: kv_params,
        ec_transfer_params: None,
    })
}

/// `prompt_tokens` in derender requests is caller supplied, so adding the
/// completion count with plain arithmetic could overflow.
fn checked_total_tokens(prompt_tokens: usize, completion_tokens: usize) -> Result<usize, ApiError> {
    prompt_tokens.checked_add(completion_tokens).ok_or_else(|| {
        ApiError::invalid_request(
            format!(
                "prompt_tokens ({prompt_tokens}) + completion_tokens \
                 ({completion_tokens}) overflows the token counter"
            ),
            None,
        )
    })
}

fn chat_request_model(request: &DerenderChatRequestUnion) -> Option<&str> {
    match request {
        DerenderChatRequestUnion::NonStreaming(request) => request.model.as_deref(),
    }
}

fn completion_request_model(request: &DerenderCompletionRequestUnion) -> Option<&str> {
    match request {
        DerenderCompletionRequestUnion::NonStreaming(request) => request.model.as_deref(),
    }
}

/// Derender a generate response into a chat completion response (render-only
/// server variant).
///
/// TODO: phase 3 accepts streaming (`stream=true`) request bodies on the same
/// path; until then they fail deserialization with a 400.
pub async fn derender_chat_render(
    State(state): State<Arc<RenderState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<DerenderChatRequestUnion>,
) -> Result<Response, ApiError> {
    let ctx = render_context(&state);
    let request_context = resolve_request_context(&headers, None);
    match body {
        DerenderChatRequestUnion::NonStreaming(request) => {
            derender_chat(&ctx, request, request_context)
                .await
                .map(|response| Json(response).into_response())
        }
    }
}

/// Derender a generate response into a completion response (render-only
/// server variant).
pub async fn derender_completion_render(
    State(state): State<Arc<RenderState>>,
    ValidatedJson(body): ValidatedJson<DerenderCompletionRequestUnion>,
) -> Result<Response, ApiError> {
    let ctx = render_context(&state);
    match body {
        DerenderCompletionRequestUnion::NonStreaming(request) => {
            derender_completion(&ctx, request).map(|response| Json(response).into_response())
        }
    }
}

/// Derender a generate response into a chat completion response
/// (engine-backed server variant).
pub async fn derender_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<DerenderChatRequestUnion>,
) -> Result<Response, ApiError> {
    let ctx = app_context(&state, chat_request_model(&body)).await;
    let request_context = resolve_request_context(&headers, None);
    match body {
        DerenderChatRequestUnion::NonStreaming(request) => {
            derender_chat(&ctx, request, request_context)
                .await
                .map(|response| Json(response).into_response())
        }
    }
}

/// Derender a generate response into a completion response (engine-backed
/// server variant).
pub async fn derender_completions(
    State(state): State<Arc<AppState>>,
    ValidatedJson(body): ValidatedJson<DerenderCompletionRequestUnion>,
) -> Result<Response, ApiError> {
    let ctx = app_context(&state, completion_request_model(&body)).await;
    match body {
        DerenderCompletionRequestUnion::NonStreaming(request) => {
            derender_completion(&ctx, request).map(|response| Json(response).into_response())
        }
    }
}
