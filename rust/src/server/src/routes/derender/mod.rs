// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Postprocessing counterpart to the /render endpoints: turn
//! token-in/token-out `GenerateResponse`s back into OpenAI chat/completion
//! responses without a GPU.
//!
//! Mirrors the Python vLLM implementation in
//! `vllm/entrypoints/scale_out/derender/` (api_router.py, serving.py) and
//! `vllm/renderers/online_derenderer.py`.

mod detok;
mod logprobs;
mod types;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use futures::StreamExt as _;
use thiserror_ext::AsReport as _;
use tracing::{debug, warn};
use vllm_chat::{AssistantContentBlock, AssistantMessageExt as _, ChatRequestProcessor};
use vllm_chat::{ChatEventStream, FinishReason};
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::{DecodedTextEvent, Finished};

use self::types::{
    DerenderChatCompletionResponse, DerenderChatRequest, DerenderChatRequestUnion,
    DerenderChatStreamRequest, DerenderChatStreamResponse, DerenderCompletionRequest,
    DerenderCompletionRequestUnion, DerenderCompletionStreamRequest,
    DerenderCompletionStreamResponse,
};
use crate::error::{ApiError, bail_invalid_request, chat_submit_error, server_error};
use crate::lora::LoraModelResolution;
use crate::render::RenderState;
use crate::routes::inference::generate::GenerateResponse;
use crate::routes::openai::chat_completions::{
    AssistantRole, ChatCompletionChoice, ChatCompletionMessage, ChatCompletionStreamChoice,
    ChatCompletionStreamResponse, ChatMessageDelta, lower_chat_request,
};
use crate::routes::openai::utils::types::{
    FunctionCallResponse, ToolCall, ToolChoice, ToolChoiceValue, Usage,
};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::openai::{
    CompletionChoice, CompletionResponse, CompletionStreamChoice, CompletionStreamResponse,
};
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
    model_id: &'a str,
    tokenizer: DynTokenizer,
    max_model_len: u32,
    max_logprobs: i32,
    chat: &'a ChatRequestProcessor,
}

impl DerenderContext<'_> {
    /// Whether a reasoning or tool parser is effectively configured for this
    /// model (Python: `self.parser is not None`).
    fn has_parser(&self) -> bool {
        self.chat.reasoning_parser_name(self.model_id).is_some()
            || self.chat.tool_call_parser_name(self.model_id).is_some()
    }
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

/// Map a wire `finish_reason` string to the internal enum for the replayed
/// parse pipeline. The response echoes the wire string verbatim; this mapping
/// only feeds the structured assembly.
fn wire_finish_reason(finish_reason: Option<&str>) -> FinishReason {
    match finish_reason {
        Some("length") => FinishReason::Length,
        Some("abort") => FinishReason::Abort,
        Some("repetition") => FinishReason::Repetition(None),
        _ => FinishReason::stop_eos(),
    }
}

/// Replay already-generated tokens through the chat output pipeline so the
/// configured reasoning/tool parser splits them into reasoning, content and
/// tool calls, mirroring Python's `parser.parse()` one-shot extraction.
async fn parse_chat_choice(
    ctx: &DerenderContext<'_>,
    chat_request: crate::routes::openai::chat_completions::ChatCompletionRequest,
    token_ids: &[u32],
    finish_reason: Option<&str>,
    ctx_request: ResolvedRequestContext,
) -> Result<ChatCompletionMessage, ApiError> {
    let include_reasoning = chat_request.include_reasoning;
    let is_named_tool_choice =
        matches!(&chat_request.tool_choice, Some(ToolChoice::Function { .. }));
    let is_required_tool_choice = matches!(
        &chat_request.tool_choice,
        Some(ToolChoice::Value(ToolChoiceValue::Required))
    );

    let lowered = lower_chat_request(chat_request, &ctx.lora_resolution, ctx_request)?;
    let (adjusted, processor) = ctx
        .chat
        .new_output_processor(lowered)
        .map_err(|error| chat_submit_error("failed to prepare derender chat request", error))?;

    // Parser path: decode with special tokens preserved so the parser can see
    // markers like </think>, <tool_call>, or Harmony channel tokens.
    let decoded_text = ctx
        .tokenizer
        .decode(token_ids, false)
        .map_err(|error| server_error!("derender decode failed: {}", error.as_report()))?;

    let events = vec![
        Ok(DecodedTextEvent::Start {
            prompt_token_ids: Arc::from([]),
            prompt_logprobs: None,
        }),
        Ok(DecodedTextEvent::TextDelta {
            delta: decoded_text,
            token_ids: token_ids.to_vec(),
            logprobs: None,
            finished: Some(Finished {
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 0,
                    output_token_count: token_ids.len(),
                    cached_token_count: 0,
                },
                finish_reason: wire_finish_reason(finish_reason),
                kv_transfer_params: None,
                ec_transfer_params: None,
            }),
        }),
    ];
    let decoded_stream = futures::stream::iter(events).boxed();
    let chat_stream = processor
        .process(decoded_stream)
        .map_err(|error| chat_submit_error("failed to derender chat response", error))?;
    let collected = ChatEventStream::from_stream(adjusted.request_id.clone(), chat_stream)
        .collect_message()
        .await
        .map_err(|error| chat_submit_error("failed to derender chat response", error))?;

    let has_content = collected
        .message
        .content
        .iter()
        .any(|block| matches!(block, AssistantContentBlock::Text { .. }));
    // A named or required tool choice forces a (possibly empty) content
    // string, matching Python's `content = content or ""`.
    let content = if has_content {
        Some(collected.message.text())
    } else if is_named_tool_choice || is_required_tool_choice {
        Some(String::new())
    } else {
        None
    };
    let tool_calls = collected
        .message
        .tool_calls()
        .map(|call| ToolCall {
            id: call.id.clone(),
            tool_type: "function".to_string(),
            function: FunctionCallResponse {
                name: call.name.clone(),
                arguments: Some(call.arguments.clone()),
            },
        })
        .collect();

    Ok(ChatCompletionMessage {
        role: AssistantRole,
        content,
        tool_calls,
        reasoning: if include_reasoning {
            collected.message.reasoning()
        } else {
            None
        },
    })
}

/// Postprocess a GenerateResponse into a chat completion response.
///
/// Non-streaming only: expects the complete GenerateResponse with all token
/// IDs present. When `request.chat_request` is provided and a parser is
/// configured, the parser splits the output into (reasoning, content,
/// tool_calls). Otherwise falls back to plain detokenization.
async fn derender_chat(
    ctx: &DerenderContext<'_>,
    request: DerenderChatRequest,
    request_context: ResolvedRequestContext,
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

        let message = if ctx.has_parser()
            && let Some(chat_request) = request.chat_request.clone()
        {
            parse_chat_choice(
                ctx,
                chat_request,
                &choice.token_ids,
                choice.finish_reason.as_deref(),
                request_context.clone(),
            )
            .await?
        } else {
            // No parser: plain detokenization honouring the request's
            // skip_special_tokens (default true when no request was given).
            let skip_special = request
                .chat_request
                .as_ref()
                .map(|request| request.skip_special_tokens)
                .unwrap_or(true);
            let decoded_text = ctx
                .tokenizer
                .decode(&choice.token_ids, skip_special)
                .map_err(|error| server_error!("derender decode failed: {}", error.as_report()))?;
            ChatCompletionMessage {
                role: AssistantRole,
                content: Some(decoded_text),
                tool_calls: Vec::new(),
                reasoning: None,
            }
        };

        choices.push(ChatCompletionChoice {
            index: choice.index,
            message,
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
            total_tokens: prompt_tokens + completion_tokens,
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
            let logprobs = choice
                .logprobs
                .as_ref()
                .map(|logprobs| {
                    logprobs::resolve_logprobs(logprobs, &ctx.tokenizer)
                        .map(|resolved| logprobs::chat_logprobs_to_completion(&resolved))
                })
                .transpose()?;

            choices.push(CompletionChoice {
                index,
                text: decoded_text,
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
        total_prompt_tokens += prompt_tokens;
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
            total_tokens: total_prompt_tokens + total_completion_tokens,
            completion_tokens: Some(total_completion_tokens),
            prompt_tokens_details: None,
        }),
        system_fingerprint: None,
        kv_transfer_params: kv_params,
        ec_transfer_params: None,
    })
}

/// Aggregate chunk usage, forwarding the caller-supplied prompt token count.
fn stream_usage(prompt_tokens: Option<usize>, usage: Option<&Usage>) -> Option<Usage> {
    let usage = usage?;
    let prompt_tokens = prompt_tokens.unwrap_or(usage.prompt_tokens);
    let completion_tokens = usage.completion_tokens.unwrap_or(0);
    Some(Usage {
        prompt_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        completion_tokens: Some(completion_tokens),
        prompt_tokens_details: None,
    })
}

/// Streaming counterpart to [`derender_chat`].
///
/// Processes one `GenerateStreamResponse` chunk and returns the derendered
/// chunk together with the updated client carried state.
///
/// TODO: parse path for reasoning and tool calls is implemented in a future
/// PR (Python raises NotImplementedError in the same situation).
fn derender_chat_stream(
    ctx: &DerenderContext<'_>,
    request: DerenderChatStreamRequest,
) -> Result<DerenderChatStreamResponse, ApiError> {
    check_model(ctx, request.model.as_deref())?;

    if ctx.has_parser() {
        // Fail closed on the parser alone. A parser configured model must
        // never fall through to plain detok on the streaming path, even when
        // `chat_request` is omitted or reasoning/tool markup would leak into
        // `delta.content`.
        bail_invalid_request!(
            "Streaming chat derender is not yet supported for models with a reasoning or \
             tool parser configured. Use the non-streaming derender endpoint (stream=false) \
             for parsed output."
        );
    }

    // A single DerenderStreamState is threaded through every choice in this
    // chunk. Correct only when there is at most one choice per SSE event
    // (n=1, one call per index), as the streaming derender protocol assumes.
    // Multiple choices sharing one chunk would corrupt each other's detok
    // window.
    if request.generate_chunk.choices.len() > 1 {
        bail_invalid_request!("derender_chat_stream expects at most one choice per chunk");
    }

    let mut state = request.stream_state.unwrap_or_default();
    state.validate()?;

    let skip_special = request
        .chat_request
        .as_ref()
        .map(|request| request.skip_special_tokens)
        .unwrap_or(true);
    let mut stream_choices = Vec::with_capacity(request.generate_chunk.choices.len());

    for choice in &request.generate_chunk.choices {
        let (new_text, updated_state) =
            detok::detokenize_delta(&ctx.tokenizer, &choice.token_ids, &state, skip_special)?;
        state = updated_state;

        // Unlike OpenAI's API, which always emits `role: "assistant"` on the
        // very first chunk, this emits it on the first chunk with a non empty
        // `choices` list. A leading usage only chunk therefore defers the role
        // to the following content chunk instead of sending an empty role only
        // delta.
        let include_role = !state.role_sent;
        if include_role {
            state.role_sent = true;
        }

        stream_choices.push(ChatCompletionStreamChoice {
            index: choice.index,
            delta: ChatMessageDelta {
                role: include_role.then_some(AssistantRole),
                content: (!new_text.is_empty()).then_some(new_text),
                tool_calls: None,
                reasoning: None,
            },
            logprobs: None,
            finish_reason: choice.finish_reason.clone(),
            stop_reason: None,
            token_ids: None,
        });
    }

    let model_name = response_model_name(ctx, request.model.clone());
    debug!(
        request_id = %request.generate_chunk.request_id,
        model = %model_name,
        "derender_chat_stream"
    );
    let mut chunk = ChatCompletionStreamResponse::new(
        &request.generate_chunk.request_id,
        &model_name,
        unix_timestamp(),
    );
    chunk.choices = stream_choices;
    chunk.usage = stream_usage(request.prompt_tokens, request.generate_chunk.usage.as_ref());

    Ok(DerenderChatStreamResponse {
        chunk,
        stream_state: state,
    })
}

/// Streaming counterpart to [`derender_completion`].
///
/// Processes one `GenerateStreamResponse` chunk (one output sequence's delta)
/// and returns the derendered chunk and updated state.
fn derender_completion_stream(
    ctx: &DerenderContext<'_>,
    request: DerenderCompletionStreamRequest,
) -> Result<DerenderCompletionStreamResponse, ApiError> {
    check_model(ctx, request.model.as_deref())?;

    // See the equivalent check in derender_chat_stream: a single
    // DerenderStreamState is threaded through every choice in this chunk, so
    // more than one choice per chunk would corrupt the detok window across
    // choices.
    if request.generate_chunk.choices.len() > 1 {
        bail_invalid_request!("derender_completion_stream expects at most one choice per chunk");
    }

    let mut state = request.stream_state.unwrap_or_default();
    state.validate()?;

    let skip_special = request
        .completion_request
        .as_ref()
        .map(|request| request.skip_special_tokens)
        .unwrap_or(true);
    let mut stream_choices = Vec::with_capacity(request.generate_chunk.choices.len());

    for choice in &request.generate_chunk.choices {
        let (new_text, updated_state) =
            detok::detokenize_delta(&ctx.tokenizer, &choice.token_ids, &state, skip_special)?;
        state = updated_state;

        stream_choices.push(CompletionStreamChoice {
            index: choice.index,
            text: new_text,
            logprobs: None,
            finish_reason: choice.finish_reason.clone(),
            stop_reason: None,
            token_ids: None,
            prompt_token_ids: None,
        });
    }

    let model_name = response_model_name(ctx, request.model.clone());
    debug!(
        request_id = %request.generate_chunk.request_id,
        model = %model_name,
        "derender_completion_stream"
    );
    let mut chunk = CompletionStreamResponse::new(
        &request.generate_chunk.request_id,
        &model_name,
        unix_timestamp(),
    );
    chunk.choices = stream_choices;
    chunk.usage = stream_usage(request.prompt_tokens, request.generate_chunk.usage.as_ref());

    Ok(DerenderCompletionStreamResponse {
        chunk,
        stream_state: state,
    })
}

fn chat_request_model(request: &DerenderChatRequestUnion) -> Option<&str> {
    match request {
        DerenderChatRequestUnion::Streaming(request) => request.model.as_deref(),
        DerenderChatRequestUnion::NonStreaming(request) => request.model.as_deref(),
    }
}

fn completion_request_model(request: &DerenderCompletionRequestUnion) -> Option<&str> {
    match request {
        DerenderCompletionRequestUnion::Streaming(request) => request.model.as_deref(),
        DerenderCompletionRequestUnion::NonStreaming(request) => request.model.as_deref(),
    }
}

/// Derender a generate response into a chat completion response (render-only
/// server variant).
///
/// Accepts both non-streaming (`stream=false`, default) and streaming
/// (`stream=true`) request bodies on the same path; the `stream`
/// discriminator selects the shape.
pub async fn derender_chat_render(
    State(state): State<Arc<RenderState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<DerenderChatRequestUnion>,
) -> Result<Response, ApiError> {
    let ctx = render_context(&state);
    let request_context = resolve_request_context(&headers, None);
    match body {
        DerenderChatRequestUnion::Streaming(request) => {
            derender_chat_stream(&ctx, request).map(|response| Json(response).into_response())
        }
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
        DerenderCompletionRequestUnion::Streaming(request) => {
            derender_completion_stream(&ctx, request).map(|response| Json(response).into_response())
        }
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
        DerenderChatRequestUnion::Streaming(request) => {
            derender_chat_stream(&ctx, request).map(|response| Json(response).into_response())
        }
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
        DerenderCompletionRequestUnion::Streaming(request) => {
            derender_completion_stream(&ctx, request).map(|response| Json(response).into_response())
        }
        DerenderCompletionRequestUnion::NonStreaming(request) => {
            derender_completion(&ctx, request).map(|response| Json(response).into_response())
        }
    }
}
