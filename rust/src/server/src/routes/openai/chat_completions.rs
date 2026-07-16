// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

pub(crate) mod convert;
mod types;
mod validate;

use std::collections::BTreeMap;
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
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::{debug, error, info, trace, warn};
use tracing_futures::Instrument as _;
use uuid::Uuid;
use vllm_chat::ReasoningParser;
use vllm_chat::ToolParser;
use vllm_chat::{
    AssistantBlockKind, AssistantMessageExt as _, ChatEvent, ChatEventStream, ChatEventStreamTrait,
    ChatTool, CollectedAssistantMessage, CombinedParser, FinishReason, ParserSelection,
    ReasoningParserFactory, ToolParserFactory, UnifiedParser, UnifiedParserEvent,
    UnifiedParserFactory, UnifiedParserOutput,
};
use vllm_engine_core_client::protocol::logprobs::TokenLogprob;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_text::BeamSearchOutput;
use vllm_text::tokenizer::{DynTokenizer, Tokenizer};

use self::convert::{ResponseOptions, prepare_chat_request};
use crate::config::ApiServerOptions;
use crate::error::{ApiError, bail_server_error, chat_submit_error, server_error};
use crate::routes::openai::chat_completions::types::{
    AssistantRole, ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStreamChoice, ChatCompletionStreamResponse,
    ChatMessageDelta,
};
use crate::routes::openai::utils::logprobs::{
    decoded_logprobs_to_openai_chat, decoded_prompt_logprobs_to_maps,
};
use crate::routes::openai::utils::types::{
    ChatLogProbs, ChatLogProbsContent, FunctionCallDelta, FunctionCallResponse, ToolCall,
    ToolCallDelta, TopLogProb, Usage,
};
use crate::routes::openai::utils::usage::ContinuousUsage;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

/// Validate one chat completion request and proxy it into the shared
/// `vllm-chat` stack.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    let stream = body.stream;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(Some(&body.model)).await;

    let mut prepared = match prepare_chat_request(body, &lora_resolution, request_context) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    let request_span = tracing::info_span!(
        "chat_completions",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let created = unix_timestamp();
    let api_server_options = state.api_server_options;

    if prepared.chat_request.sampling_params.use_beam_search {
        let model_id = state.chat.model_id().to_owned();
        let tool_call_parser = state.chat.tool_call_parser().clone();
        let reasoning_parser = state.chat.reasoning_parser().clone();
        let tools = prepared.chat_request.tools.clone();

        if !tools.is_empty() || prepared.options.include_reasoning {
            prepared.chat_request.decode_options.skip_special_tokens = false;
        }

        let beam_result = match state
            .chat
            .beam_search_chat(prepared.chat_request)
            .instrument(request_span.clone())
            .await
        {
            Ok(result) => result,
            Err(error) => {
                return chat_submit_error("failed to submit beam search chat request", error)
                    .into_response();
            }
        };

        let tokenizer = state.chat.text().tokenizer();
        let response = match collect_beam_search_chat_completion(
            beam_result,
            prepared.request_id,
            prepared.response_model,
            created,
            api_server_options,
            prepared.options,
            tokenizer.as_ref(),
            tokenizer.clone(),
            &model_id,
            &tool_call_parser,
            &reasoning_parser,
            &tools,
        ) {
            Ok(response) => response,
            Err(error) => return error.into_response(),
        };
        return Json(response).into_response();
    }

    let chat_stream =
        match state.chat.chat(prepared.chat_request).instrument(request_span.clone()).await {
            Ok(stream) => stream,
            Err(error) => {
                return chat_submit_error("failed to submit chat request", error).into_response();
            }
        };

    if stream {
        let chunk_stream = chat_completion_chunk_stream(
            chat_stream,
            prepared.request_id,
            prepared.response_model,
            created,
            api_server_options,
            prepared.options,
        );
        let sse_stream = chat_completion_sse_stream(chunk_stream).instrument(request_span);

        Sse::new(sse_stream).into_response()
    } else {
        let response = match collect_chat_completion(
            chat_stream,
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

async fn collect_chat_completion(
    stream: ChatEventStream,
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
        requested_logprobs,
        include_prompt_logprobs,
        include_reasoning,
        echo,
        return_token_ids,
        return_tokens_as_token_ids,
    }: ResponseOptions,
) -> Result<ChatCompletionResponse, ApiError> {
    let collected = stream.collect_message().await.map_err(|error| {
        server_error!(
            "failed to collect chat completion response: {}",
            error.to_report_string()
        )
    })?;
    let CollectedAssistantMessage {
        message,
        prompt_token_ids,
        prompt_logprobs,
        logprobs,
        token_ids,
        usage,
        finish_reason,
        kv_transfer_params,
        ec_transfer_params,
    } = collected;
    let stop_reason = finish_reason.as_stop_reason().map(stop_reason_to_json);
    let saw_tool_calls = message.tool_calls().next().is_some();
    let reasoning = message.reasoning();
    // Output logprobs and token IDs cover the complete generated token stream.
    // When reasoning is hidden, omit them rather than leaking hidden reasoning
    // tokens through per-token metadata.
    let include_output_metadata = include_reasoning || reasoning.is_none();
    let finish_reason = chat_finish_reason_to_openai(&finish_reason, saw_tool_calls)?.to_string();
    let tool_calls = message
        .tool_calls()
        .map(|call| ToolCall {
            id: call.id.clone(),
            tool_type: "function".to_string(),
            function: FunctionCallResponse {
                name: call.name.clone(),
                arguments: Some(call.arguments.clone()),
            },
        })
        .collect::<Vec<_>>();
    let logprobs = if requested_logprobs && include_output_metadata {
        Some(decoded_logprobs_to_openai_chat(
            logprobs.as_ref().ok_or_else(|| {
                server_error!("chat response requested logprobs but generation returned none")
            })?,
            return_tokens_as_token_ids,
        )?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        Some(decoded_prompt_logprobs_to_maps(
            prompt_logprobs.as_ref().ok_or_else(|| {
                server_error!(
                    "chat response requested prompt_logprobs but generation returned none"
                )
            })?,
            return_tokens_as_token_ids,
        ))
    } else {
        None
    };
    let usage = Usage::from_token_usage(usage, enable_prompt_tokens_details);

    if enable_log_requests {
        info!(
            model = %response_model,
            prompt_tokens = usage.prompt_tokens,
            output_tokens = usage.completion_tokens.unwrap_or(0),
            finish_reason = %finish_reason,
            "chat completion finished"
        );
    }

    Ok(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: response_model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: AssistantRole,
                content: match &echo {
                    Some(prefix) => Some(format!("{prefix}{}", message.text())),
                    None => Some(message.text()).filter(|t| !t.is_empty()),
                },
                tool_calls,
                reasoning: if include_reasoning { reasoning } else { None },
            },
            logprobs,
            finish_reason: Some(finish_reason),
            stop_reason,
            token_ids: (return_token_ids && include_output_metadata).then_some(token_ids),
        }],
        usage: Some(usage),
        system_fingerprint: None,
        prompt_logprobs,
        prompt_token_ids: return_token_ids.then(|| prompt_token_ids.to_vec()),
        kv_transfer_params,
        ec_transfer_params,
    })
}

fn build_beam_chat_logprobs(
    beam_logprobs: &[Vec<TokenLogprob>],
    generated_tokens: &[u32],
    tokenizer: &dyn Tokenizer,
) -> Option<ChatLogProbs> {
    if generated_tokens.is_empty() || beam_logprobs.len() < generated_tokens.len() {
        return None;
    }
    let n = generated_tokens.len();
    let mut content = Vec::with_capacity(n);
    for pos in 0..n {
        let entries = &beam_logprobs[pos];
        let token_id = generated_tokens[pos];
        let selected_logprob = entries.iter().find(|e| e.token_id == token_id).map(|e| e.logprob);
        let token = tokenizer.decode(&[token_id], false).unwrap_or_default();
        let token_bytes = token.as_bytes().to_vec();
        let top_logprobs: Vec<TopLogProb> = entries
            .iter()
            .map(|e| {
                let decoded = tokenizer.decode(&[e.token_id], false).unwrap_or_default();
                TopLogProb {
                    token: if decoded.is_empty() {
                        format!("token_id:{}", e.token_id)
                    } else {
                        decoded.clone()
                    },
                    logprob: e.logprob,
                    bytes: Some(decoded.as_bytes().to_vec()),
                }
            })
            .collect();
        content.push(ChatLogProbsContent {
            token: if token.is_empty() {
                format!("token_id:{}", token_id)
            } else {
                token
            },
            logprob: selected_logprob.unwrap_or(0.0),
            bytes: Some(token_bytes),
            top_logprobs,
        });
    }
    Some(ChatLogProbs {
        content: Some(content),
    })
}

fn resolve_tool_parser(
    tools: &[ChatTool],
    model_id: &str,
    selection: &ParserSelection,
) -> Result<Box<dyn ToolParser>, ApiError> {
    let factory = ToolParserFactory::global();
    let parser_name = match selection {
        ParserSelection::Auto => factory
            .resolve_name_for_model(model_id)
            .ok_or_else(|| server_error!("no tool parser available for model `{}`", model_id))?,
        ParserSelection::None => {
            return Err(server_error!("tool parsing disabled"));
        }
        ParserSelection::Explicit(name) => name.as_str(),
    };
    let parser = factory.create(parser_name, tools).map_err(|e| {
        server_error!(
            "failed to create tool parser `{}`: {}",
            parser_name,
            e.to_report_string()
        )
    })?;
    Ok(parser)
}

fn resolve_optional_reasoning_parser(
    model_id: &str,
    tokenizer: DynTokenizer,
    selection: &ParserSelection,
) -> Option<Box<dyn ReasoningParser>> {
    let factory = ReasoningParserFactory::global();
    let parser_name = match selection {
        ParserSelection::Auto => factory.resolve_name_for_model(model_id)?,
        ParserSelection::None => return None,
        ParserSelection::Explicit(name) => name,
    };
    match factory.create(parser_name, tokenizer) {
        Ok(parser) => Some(parser),
        Err(e) => {
            warn!(
                parser_name,
                error = %e.to_report_string(),
                "reasoning parser creation failed for model `{}`, reasoning parsing disabled",
                model_id,
            );
            None
        }
    }
}

fn resolve_optional_unified_parser(
    tools: &[ChatTool],
    model_id: &str,
    tokenizer: DynTokenizer,
    selection: &ParserSelection,
) -> Result<Option<Box<dyn UnifiedParser>>, ApiError> {
    let factory = UnifiedParserFactory::global();
    let parser_name = match selection {
        ParserSelection::Auto => factory.resolve_name_for_model(model_id),
        ParserSelection::None => None,
        ParserSelection::Explicit(name) if factory.contains(name) => Some(name.as_str()),
        ParserSelection::Explicit(_) => None,
    };
    let Some(parser_name) = parser_name else {
        return Ok(None);
    };
    let parser = factory.create(parser_name, tools, tokenizer).map_err(|e| {
        server_error!(
            "failed to create unified parser `{}`: {}",
            parser_name,
            e.to_report_string()
        )
    })?;
    Ok(Some(parser))
}

/// Parse decoded text through reasoning and tool parsers, returning extracted
/// content, reasoning, and tool calls.
fn parse_beam_chat_output(
    decoded_text: &str,
    model_id: &str,
    tools: &[ChatTool],
    tokenizer: DynTokenizer,
    tool_call_parser: &ParserSelection,
    reasoning_parser: &ParserSelection,
) -> Result<(Option<String>, Option<String>, Vec<ToolCall>), ApiError> {
    let has_tools = !tools.is_empty() && !matches!(tool_call_parser, ParserSelection::None);

    let mut combined: Box<dyn UnifiedParser> = if tool_call_parser == reasoning_parser
        && let Some(unified) =
            resolve_optional_unified_parser(tools, model_id, tokenizer.clone(), tool_call_parser)?
    {
        unified
    } else {
        let tool_parser = if has_tools {
            Some(resolve_tool_parser(tools, model_id, tool_call_parser)?)
        } else {
            None
        };
        let reasoning_parser_inst =
            resolve_optional_reasoning_parser(model_id, tokenizer.clone(), reasoning_parser);
        Box::new(CombinedParser::new(reasoning_parser_inst, tool_parser))
    };

    let mut output = UnifiedParserOutput::default();
    combined.parse_into(decoded_text, &mut output).map_err(|e| {
        server_error!(
            "beam search output parsing failed: {}",
            e.to_report_string()
        )
    })?;
    output.append(combined.finish().map_err(|e| {
        server_error!(
            "beam search parsing finish failed: {}",
            e.to_report_string()
        )
    })?);

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut acc: BTreeMap<usize, (String, String)> = BTreeMap::new();

    for event in output.events {
        match event {
            UnifiedParserEvent::Text(text) => content.push_str(&text),
            UnifiedParserEvent::Reasoning(text) => reasoning.push_str(&text),
            UnifiedParserEvent::ToolCall(delta) => {
                let entry = acc.entry(delta.tool_index).or_default();
                if let Some(name) = delta.name {
                    entry.0 = name;
                }
                entry.1.push_str(&delta.arguments);
            }
        }
    }

    let tool_calls: Vec<ToolCall> = acc
        .into_iter()
        .map(|(_idx, (name, arguments))| ToolCall {
            id: format!("call_{}", &Uuid::new_v4().simple().to_string()[..24]),
            tool_type: "function".to_string(),
            function: FunctionCallResponse {
                name,
                arguments: Some(arguments),
            },
        })
        .collect();

    Ok((
        Some(content).filter(|t| !t.is_empty()),
        Some(reasoning).filter(|t| !t.is_empty()),
        tool_calls,
    ))
}

fn collect_beam_search_chat_completion(
    beam_result: BeamSearchOutput,
    request_id: String,
    response_model: String,
    created: u64,
    ApiServerOptions {
        enable_log_requests,
        enable_prompt_tokens_details,
        ..
    }: ApiServerOptions,
    ResponseOptions {
        requested_logprobs,
        return_token_ids,
        include_reasoning,
        echo,
        ..
    }: ResponseOptions,
    tokenizer: &dyn Tokenizer,
    tokenizer_arc: DynTokenizer,
    model_id: &str,
    tool_call_parser: &ParserSelection,
    reasoning_parser: &ParserSelection,
    tools: &[ChatTool],
) -> Result<ChatCompletionResponse, ApiError> {
    let prompt_len = beam_result.prompt_token_ids.len();
    let output_tokens: usize = beam_result
        .sequences
        .iter()
        .map(|s| s.tokens.len().saturating_sub(prompt_len))
        .sum();
    let usage = Usage::from_counts(
        prompt_len,
        output_tokens,
        enable_prompt_tokens_details.then_some(0),
    );

    let mut choices = Vec::with_capacity(beam_result.sequences.len());
    for (i, seq) in beam_result.sequences.iter().enumerate() {
        let generated_tokens = seq.tokens[beam_result.prompt_token_ids.len()..].to_vec();
        let stop_reason = seq
            .stop_reason
            .map(|token_id| serde_json::Value::Number(serde_json::Number::from(token_id)));
        let decoded = tokenizer.decode(&generated_tokens, false).unwrap_or_default();
        let logprobs = requested_logprobs
            .then(|| build_beam_chat_logprobs(&seq.logprobs, &generated_tokens, tokenizer))
            .flatten();

        let (raw_content, reasoning, tool_calls) = if decoded.is_empty() {
            (None, None, vec![])
        } else {
            let (parsed_content, parsed_reasoning, parsed_tool_calls) = parse_beam_chat_output(
                &decoded,
                model_id,
                tools,
                tokenizer_arc.clone(),
                tool_call_parser,
                reasoning_parser,
            )?;
            let reasoning = if include_reasoning {
                parsed_reasoning
            } else {
                None
            };
            (parsed_content, reasoning, parsed_tool_calls)
        };

        let saw_tool_calls = !tool_calls.is_empty();
        let openai_finish_reason = seq
            .finish_reason
            .as_ref()
            .map(|fr| chat_finish_reason_to_openai(fr, saw_tool_calls).unwrap_or("error"))
            .unwrap_or("length");

        let content = raw_content.map(|c| match &echo {
            Some(prefix) => format!("{prefix}{c}"),
            None => c,
        });

        choices.push(ChatCompletionChoice {
            index: i as u32,
            message: ChatCompletionMessage {
                role: AssistantRole,
                content,
                tool_calls,
                reasoning,
            },
            logprobs,
            finish_reason: Some(openai_finish_reason.to_string()),
            stop_reason,
            token_ids: return_token_ids.then_some(generated_tokens),
        });
    }

    if enable_log_requests {
        info!(
            model = %response_model,
            prompt_tokens = usage.prompt_tokens,
            output_tokens = usage.completion_tokens.unwrap_or(0),
            num_beams = beam_result.sequences.len(),
            "beam search chat completion finished"
        );
    }

    Ok(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: response_model,
        choices,
        usage: Some(usage),
        system_fingerprint: None,
        prompt_logprobs: None,
        prompt_token_ids: return_token_ids.then_some(beam_result.prompt_token_ids),
        kv_transfer_params: beam_result.kv_transfer_params,
        ec_transfer_params: beam_result.ec_transfer_params,
    })
}

/// Convert one internal chat event stream into OpenAI chat-completion chunks.
#[try_stream]
async fn chat_completion_chunk_stream(
    mut stream: impl ChatEventStreamTrait + Unpin,
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
        requested_logprobs,
        // Ignored: chat streaming prompt logprobs are rejected for Python parity.
        include_prompt_logprobs: _,
        include_reasoning,
        echo,
        return_token_ids,
        return_tokens_as_token_ids,
    }: ResponseOptions,
    mut y: TryYielder<ChatCompletionStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    let mut saw_tool_calls = false;
    // `LogprobsDelta` is emitted after all chat events for one decoded update.
    // If that update contains hidden reasoning, including delimiter-only block
    // starts or ends, omit its token metadata as well as its visible delta.
    let mut inside_hidden_reasoning = false;
    let mut suppress_current_update_metadata = false;
    let mut continuous_usage = ContinuousUsage::default();

    /// Yield a chunk with optional continuous usage attached.
    macro_rules! yield_chunk {
        ($chunk:expr) => {{
            let mut chunk = $chunk;
            if include_continuous_usage {
                chunk.usage = Some(continuous_usage.to_usage());
            }
            y.yield_ok(chunk).await;
        }};
    }

    // If the client requested logprobs or token_ids, we need to buffer chunks until
    // we receive the separate `LogprobsDelta` event, so that we can emit one
    // combined chunk with both the semantic delta and its per-update metadata.
    // Continuous usage also buffers so the token count from `LogprobsDelta` can
    // be attached to the matching semantic chunk.
    let mut pending_chunk = (requested_logprobs || return_token_ids || include_continuous_usage)
        .then(PendingChatChunk::default);

    while let Some(next) = stream.next().await {
        match next {
            Ok(ChatEvent::Start {
                prompt_token_ids, ..
            }) => {
                continuous_usage.set_prompt_tokens(prompt_token_ids.len());
                let mut chunk = start_chunk(&request_id, &response_model, created);
                if return_token_ids {
                    chunk.prompt_token_ids = Some(prompt_token_ids.to_vec());
                }
                yield_chunk!(chunk);
                // When echo=true, emit the last assistant message content as a delta chunk.
                if let Some(echo_text) = &echo {
                    yield_chunk!(block_delta_chunk(
                        &request_id,
                        &response_model,
                        created,
                        AssistantBlockKind::Text,
                        echo_text.clone(),
                    ));
                }
            }
            Ok(ChatEvent::BlockDelta { kind, delta, .. }) => {
                let include_delta =
                    include_reasoning || !matches!(kind, AssistantBlockKind::Reasoning);
                if include_delta {
                    if let Some(pending_chunk) = pending_chunk.as_mut() {
                        pending_chunk.push_block_delta(kind, delta);
                    } else {
                        yield_chunk!(block_delta_chunk(
                            &request_id,
                            &response_model,
                            created,
                            kind,
                            delta,
                        ));
                    }
                } else {
                    suppress_current_update_metadata = true;
                }
            }
            Ok(ChatEvent::LogprobsDelta {
                logprobs,
                token_ids,
            }) => {
                let delta_token_count = token_ids.len();
                continuous_usage.add_output_tokens(delta_token_count);
                let include_metadata =
                    !suppress_current_update_metadata && !inside_hidden_reasoning;
                suppress_current_update_metadata = false;
                let openai_logprobs = if include_metadata {
                    logprobs
                        .as_ref()
                        .map(|lp| decoded_logprobs_to_openai_chat(lp, return_tokens_as_token_ids))
                        .transpose()?
                } else {
                    None
                };
                let openai_token_ids = include_metadata
                    .then_some(token_ids)
                    .and_then(|token_ids| return_token_ids.then_some(token_ids))
                    .filter(|t| !t.is_empty());
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.logprobs = openai_logprobs;
                    pending_chunk.token_ids = openai_token_ids;
                    if let Some(chunk) =
                        pending_chunk.take_chunk(&request_id, &response_model, created)
                    {
                        yield_chunk!(chunk);
                    }
                } else if let Some(logprobs) = openai_logprobs {
                    yield_chunk!(logprobs_only_chunk(
                        &request_id,
                        &response_model,
                        created,
                        logprobs,
                    ));
                }
            }
            Ok(ChatEvent::BlockStart { kind, .. }) => {
                debug!(?kind, "starting new block");
                if !include_reasoning && matches!(kind, AssistantBlockKind::Reasoning) {
                    inside_hidden_reasoning = true;
                    suppress_current_update_metadata = true;
                }
            }
            Ok(ChatEvent::BlockEnd { .. }) => {
                debug!("ending current block");
                if inside_hidden_reasoning {
                    inside_hidden_reasoning = false;
                    suppress_current_update_metadata = true;
                }
            }
            Ok(ChatEvent::ToolCallStart { index, id, name }) => {
                let tool_index = index as u32;
                saw_tool_calls = true;
                debug!(
                    tool_call_id = %id,
                    tool_call_name = %name,
                    "starting new tool call"
                );
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.push_tool_call_start(tool_index, id, name);
                } else {
                    yield_chunk!(tool_call_start_chunk(
                        &request_id,
                        &response_model,
                        created,
                        tool_index,
                        id,
                        name,
                    ));
                }
            }
            Ok(ChatEvent::ToolCallArgumentsDelta { index, delta }) => {
                let tool_index = index as u32;
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.push_tool_call_arguments(tool_index, delta);
                } else {
                    yield_chunk!(tool_call_arguments_chunk(
                        &request_id,
                        &response_model,
                        created,
                        tool_index,
                        delta,
                    ));
                }
            }
            Ok(ChatEvent::ToolCallEnd { .. }) => {
                debug!("ending current tool call");
            }
            Ok(ChatEvent::Done {
                usage: final_usage,
                finish_reason,
                ..
            }) => {
                if enable_log_requests {
                    info!(
                        stream = true,
                        model = %response_model,
                        prompt_tokens = final_usage.prompt_token_count,
                        output_tokens = final_usage.output_token_count,
                        finish_reason = finish_reason.as_str(),
                        "chat completion finished"
                    );
                }

                continuous_usage.set_final_counts(
                    final_usage.prompt_token_count,
                    final_usage.output_token_count,
                );

                if let Some(pending_chunk) = pending_chunk.as_mut()
                    && let Some(chunk) =
                        pending_chunk.take_chunk(&request_id, &response_model, created)
                {
                    yield_chunk!(chunk);
                }

                match final_chunk(
                    &request_id,
                    &response_model,
                    created,
                    finish_reason,
                    saw_tool_calls,
                ) {
                    Ok(chunk) => yield_chunk!(chunk),
                    Err(error) => {
                        error!(
                            error = %error.to_error_response().error.message,
                            "invalid terminal finish reason"
                        );
                        return Err(error);
                    }
                }

                if include_usage {
                    y.yield_ok(usage_chunk(
                        &request_id,
                        &response_model,
                        created,
                        Usage::from_token_usage(final_usage, enable_prompt_tokens_details),
                    ))
                    .await;
                }

                return Ok(());
            }
            Err(error) => {
                error!(
                    error = %error.as_report(),
                    "chat stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }
    Ok(())
}

fn usage_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    usage: Usage,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.usage = Some(usage);
    chunk
}

/// One in-flight chat-completions SSE chunk being assembled at the route layer.
///
/// `vllm-chat` emits semantic chat events first and `LogprobsDelta` separately,
/// because one decoded update may be rewritten into multiple chat events.
/// The OpenAI chat API, though, wants one streamed chunk to optionally carry
/// both the delta and its logprobs.
///
/// This small buffer accumulates the semantic delta first, then attaches the
/// following `LogprobsDelta` and flushes one combined chunk. It relies on the
/// current `vllm-chat` invariant that all semantic events from one decoded
/// update are emitted before that update's `LogprobsDelta`.
#[derive(Debug, Default)]
struct PendingChatChunk {
    /// The currently buffered OpenAI delta payload assembled from one or more
    /// chat semantic events belonging to the same decoded update.
    delta: ChatMessageDelta,
    /// The token-aligned logprobs for that same decoded update.
    logprobs: Option<ChatLogProbs>,
    /// Per-update output token IDs for the same decoded update.
    token_ids: Option<Vec<u32>>,
}

impl PendingChatChunk {
    /// Append one assistant text/reasoning block delta to the buffered OpenAI
    /// delta payload.
    fn push_block_delta(&mut self, kind: AssistantBlockKind, delta: String) {
        match kind {
            AssistantBlockKind::Text => append_delta_text(&mut self.delta.content, delta),
            AssistantBlockKind::Reasoning => append_delta_text(&mut self.delta.reasoning, delta),
            AssistantBlockKind::ToolCall => {
                unreachable!("tool calls must flow through dedicated tool-call chunks")
            }
        }
    }

    /// Append the OpenAI tool-call-start representation to the buffered delta.
    fn push_tool_call_start(&mut self, index: u32, id: String, name: String) {
        self.delta.tool_calls.get_or_insert_with(Vec::new).push(ToolCallDelta {
            index,
            id: Some(id),
            tool_type: Some("function".to_string()),
            function: Some(FunctionCallDelta {
                name: Some(name),
                arguments: None,
            }),
        });
    }

    /// Append one incremental tool-call arguments update to the buffered delta.
    fn push_tool_call_arguments(&mut self, index: u32, delta: String) {
        self.delta.tool_calls.get_or_insert_with(Vec::new).push(ToolCallDelta {
            index,
            id: None,
            tool_type: None,
            function: Some(FunctionCallDelta {
                name: None,
                arguments: Some(delta),
            }),
        });
    }

    /// Finalize the currently buffered SSE chunk, if it contains either a
    /// semantic delta or a logprobs payload.
    ///
    /// This may produce:
    /// - a combined delta + logprobs chunk
    /// - a delta-only chunk
    /// - a logprobs-only chunk
    ///
    /// The logprobs-only case is intentional: token-level metadata in one
    /// decoded update is correlated with the same update boundary, not
    /// necessarily with a visible/chat-semantic delta.
    fn take_chunk(
        &mut self,
        request_id: &str,
        response_model: &str,
        created: u64,
    ) -> Option<ChatCompletionStreamResponse> {
        let has_delta = self.delta.content.is_some()
            || self.delta.reasoning.is_some()
            || self.delta.tool_calls.is_some();
        let logprobs = self.logprobs.take();
        let token_ids = self.token_ids.take();
        if !has_delta && logprobs.is_none() && token_ids.is_none() {
            return None;
        }

        let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
        chunk.choices.push(ChatCompletionStreamChoice {
            delta: self.take_delta(),
            logprobs,
            token_ids,
            ..Default::default()
        });
        Some(chunk)
    }

    /// Take the currently buffered OpenAI delta payload and leave this pending
    /// chunk empty for the next decoded update.
    fn take_delta(&mut self) -> ChatMessageDelta {
        ChatMessageDelta {
            role: self.delta.role.take(),
            content: self.delta.content.take(),
            tool_calls: self.delta.tool_calls.take(),
            reasoning: self.delta.reasoning.take(),
        }
    }
}

/// Append one text fragment to an optional OpenAI delta string field.
fn append_delta_text(slot: &mut Option<String>, delta: String) {
    match slot {
        Some(existing) => existing.push_str(&delta),
        None => *slot = Some(delta),
    }
}

/// Convert one chunk stream into OpenAI-style SSE events.
///
/// OpenAI-style streaming errors are encoded as ordinary `data: {"error": ...}`
/// events followed by `data: [DONE]`, so the transport stream itself stays
/// infallible even when generation fails after the HTTP response has started.
#[try_stream]
async fn chat_completion_sse_stream(
    stream: impl Stream<Item = Result<ChatCompletionStreamResponse, ApiError>>,
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
fn to_sse_event(chunk: &ChatCompletionStreamResponse) -> Event {
    let payload =
        serde_json::to_string(chunk).expect("ChatCompletionStreamResponse must serialize to JSON");
    trace!(payload, "chat completion emitting chunk");
    Event::default().data(payload)
}

/// Serialize one OpenAI error payload into one SSE `data:` event.
fn to_error_sse_event(error: &ApiError) -> Event {
    let payload = serde_json::to_string(&error.to_error_response())
        .expect("ErrorResponse must serialize to JSON");
    trace!(payload, "chat completion emitting error");
    Event::default().data(payload)
}

/// Build the terminal OpenAI SSE sentinel event.
fn done_sse_event() -> Event {
    trace!("chat completion emitting done");
    Event::default().data("[DONE]")
}

/// Build the initial assistant-role SSE chunk required by the OpenAI streaming
/// protocol.
fn start_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta: ChatMessageDelta {
            role: Some(AssistantRole),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

/// Build one content-delta SSE chunk from one internal assistant block delta.
fn block_delta_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    kind: AssistantBlockKind,
    delta: String,
) -> ChatCompletionStreamResponse {
    let delta = match kind {
        AssistantBlockKind::Text => ChatMessageDelta {
            content: Some(delta),
            ..Default::default()
        },
        AssistantBlockKind::Reasoning => ChatMessageDelta {
            reasoning: Some(delta),
            ..Default::default()
        },
        AssistantBlockKind::ToolCall => {
            unreachable!("tool calls must flow through dedicated tool-call chunks")
        }
    };

    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta,
        ..Default::default()
    });
    chunk
}

fn tool_call_start_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    tool_index: u32,
    id: String,
    name: String,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta: ChatMessageDelta {
            tool_calls: Some(vec![ToolCallDelta {
                index: tool_index,
                id: Some(id),
                tool_type: Some("function".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some(name),
                    arguments: None,
                }),
            }]),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

fn tool_call_arguments_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    tool_index: u32,
    delta: String,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta: ChatMessageDelta {
            tool_calls: Some(vec![ToolCallDelta {
                index: tool_index,
                id: None,
                tool_type: None,
                function: Some(FunctionCallDelta {
                    name: None,
                    arguments: Some(delta),
                }),
            }]),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

fn logprobs_only_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    logprobs: ChatLogProbs,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        logprobs: Some(logprobs),
        ..Default::default()
    });
    chunk
}

/// Build the terminal SSE chunk carrying the OpenAI finish reason.
fn final_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    finish_reason: FinishReason,
    saw_tool_calls: bool,
) -> Result<ChatCompletionStreamResponse, ApiError> {
    let stop_reason = finish_reason.as_stop_reason().map(stop_reason_to_json);
    let finish_reason = chat_finish_reason_to_openai(&finish_reason, saw_tool_calls)?;

    debug!(
        finish_reason = %finish_reason,
        stop_reason = ?stop_reason,
        "chat stream finished"
    );

    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        finish_reason: Some(finish_reason.to_string()),
        stop_reason,
        ..Default::default()
    });
    Ok(chunk)
}

fn chat_finish_reason_to_openai(
    finish_reason: &FinishReason,
    saw_tool_calls: bool,
) -> Result<&'static str, ApiError> {
    match finish_reason {
        FinishReason::Stop(_) if saw_tool_calls => Ok("tool_calls"),
        FinishReason::Stop(_) => Ok("stop"),
        FinishReason::Length => Ok("length"),
        FinishReason::Abort => Ok("abort"),
        FinishReason::Repetition(_) => Ok("repetition"),
        FinishReason::Error => {
            bail_server_error!("Internal server error");
        }
    }
}

/// Convert one internal stop reason into the OpenAI-compatible `stop_reason`
/// JSON shape.
fn stop_reason_to_json(stop_reason: &StopReason) -> Value {
    serde_json::to_value(stop_reason).expect("StopReason must serialize to JSON")
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt as _, stream};
    use serde_json::json;
    use vllm_chat::{
        AssistantBlockKind, AssistantContentBlock, AssistantToolCall, ChatEvent, FinishReason,
    };
    use vllm_engine_core_client::protocol::output::StopReason;
    use vllm_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob};

    use super::{
        ApiServerOptions, ResponseOptions, block_delta_chunk, chat_completion_chunk_stream,
        final_chunk,
    };

    #[test]
    fn text_chunk_uses_content_only_delta() {
        let chunk = block_delta_chunk(
            "chatcmpl-1",
            "model",
            1,
            AssistantBlockKind::Text,
            "hello".to_string(),
        );
        assert_eq!(chunk.choices[0].delta.role, None);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
        assert_eq!(chunk.choices[0].delta.reasoning, None);
    }

    #[test]
    fn reasoning_chunk_uses_reasoning_only_delta() {
        let chunk = block_delta_chunk(
            "chatcmpl-1",
            "model",
            1,
            AssistantBlockKind::Reasoning,
            "thinking".to_string(),
        );
        assert_eq!(chunk.choices[0].delta.role, None);
        assert_eq!(chunk.choices[0].delta.content, None);
        assert_eq!(
            chunk.choices[0].delta.reasoning.as_deref(),
            Some("thinking")
        );
    }

    #[test]
    fn final_chunk_maps_stop_finish_reason_and_stop_reason() {
        let chunk = final_chunk(
            "chatcmpl-1",
            "model",
            1,
            FinishReason::Stop(Some(StopReason::Text("stop".to_string()))),
            false,
        )
        .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunk.choices[0].stop_reason, Some(json!("stop")));
    }

    #[test]
    fn final_chunk_maps_length_finish_reason() {
        let chunk = final_chunk("chatcmpl-1", "model", 1, FinishReason::Length, false)
            .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
        assert_eq!(chunk.choices[0].stop_reason, None);
    }

    #[test]
    fn final_chunk_maps_abort_finish_reason() {
        let chunk = final_chunk("chatcmpl-1", "model", 1, FinishReason::Abort, false)
            .expect("abort is a valid finish reason");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("abort"));
        assert_eq!(chunk.choices[0].stop_reason, None);
    }

    #[test]
    fn final_chunk_rejects_error_finish_reason() {
        assert!(final_chunk("chatcmpl-1", "model", 1, FinishReason::Error, false).is_err());
    }

    #[test]
    fn final_chunk_maps_stop_to_tool_calls_when_tool_calls_were_streamed() {
        let chunk = final_chunk("chatcmpl-1", "model", 1, FinishReason::stop_eos(), true)
            .expect("finish reason is valid");

        assert_eq!(
            chunk.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[tokio::test]
    async fn chunk_stream_coalesces_text_delta_with_logprobs() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::BlockStart {
                index: 0,
                kind: AssistantBlockKind::Text,
            }),
            Ok(ChatEvent::BlockDelta {
                index: 0,
                kind: AssistantBlockKind::Text,
                delta: "hi".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "hi".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![],
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 1,
                    output_token_count: 1,
                    cached_token_count: 1,
                },
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
                ec_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions {
                enable_prompt_tokens_details: true,
                ..Default::default()
            },
            ResponseOptions {
                include_usage: true,
                requested_logprobs: true,
                include_reasoning: true,
                ..Default::default()
            },
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("hi"));
        let logprobs = chunks[1].choices[0].logprobs.as_ref().expect("logprobs");
        let content = logprobs.content.as_ref().expect("logprobs content");
        assert_eq!(content[0].token, "hi");
        assert_eq!(
            chunks[3]
                .usage
                .as_ref()
                .expect("usage")
                .prompt_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens),
            Some(1)
        );
    }

    #[tokio::test]
    async fn chunk_stream_coalesces_reasoning_delta_with_logprobs() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::BlockStart {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
            }),
            Ok(ChatEvent::BlockDelta {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
                delta: "think".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "think".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![],
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 1,
                    output_token_count: 1,
                    cached_token_count: 0,
                },
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
                ec_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions::default(),
            ResponseOptions {
                requested_logprobs: true,
                include_reasoning: true,
                ..Default::default()
            },
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks[1].choices[0].delta.reasoning.as_deref(),
            Some("think")
        );
        assert!(chunks[1].choices[0].logprobs.is_some());
    }

    #[tokio::test]
    async fn chunk_stream_omits_reasoning_delta_when_disabled() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::BlockDelta {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
                delta: "think".to_string(),
            }),
            Ok(ChatEvent::BlockDelta {
                index: 1,
                kind: AssistantBlockKind::Text,
                delta: "answer".to_string(),
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 1,
                    output_token_count: 2,
                    cached_token_count: 0,
                },
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
                ec_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions::default(),
            ResponseOptions::default(),
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks[1].choices[0].delta.content.as_deref(),
            Some("answer")
        );
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.choices.iter().all(|choice| choice.delta.reasoning.is_none()))
        );
    }

    #[tokio::test]
    async fn chunk_stream_omits_logprobs_for_suppressed_reasoning() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::BlockDelta {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
                delta: "think".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 11,
                            token: "think".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![11],
            }),
            Ok(ChatEvent::BlockDelta {
                index: 1,
                kind: AssistantBlockKind::Text,
                delta: "answer".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 22,
                            token: "answer".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![22],
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 1,
                    output_token_count: 2,
                    cached_token_count: 0,
                },
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
                ec_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions::default(),
            ResponseOptions {
                requested_logprobs: true,
                return_token_ids: true,
                ..Default::default()
            },
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        let choice = &chunks[1].choices[0];
        assert_eq!(choice.delta.content.as_deref(), Some("answer"));
        assert_eq!(choice.token_ids.as_deref(), Some(&[22][..]));
        let logprobs = choice.logprobs.as_ref().expect("answer logprobs");
        let content = logprobs.content.as_ref().expect("logprobs content");
        assert_eq!(content[0].token, "answer");
        assert!(chunks.iter().all(|chunk| {
            chunk.choices.iter().all(|choice| {
                choice.delta.reasoning.is_none()
                    && choice.token_ids.as_deref() != Some(&[11][..])
                    && choice
                        .logprobs
                        .as_ref()
                        .and_then(|logprobs| logprobs.content.as_ref())
                        .is_none_or(|content| content.iter().all(|entry| entry.token != "think"))
            })
        }));
    }

    #[tokio::test]
    async fn chunk_stream_omits_logprobs_for_hidden_reasoning_delimiters() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::BlockStart {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 11,
                            token: "<think>".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![11],
            }),
            Ok(ChatEvent::BlockDelta {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
                delta: "think".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 12,
                            token: "think".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![12],
            }),
            Ok(ChatEvent::BlockEnd {
                index: 0,
                block: AssistantContentBlock::Reasoning {
                    text: "think".to_string(),
                },
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 13,
                            token: "</think>".to_string(),
                            logprob: -0.3,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![13],
            }),
            Ok(ChatEvent::BlockStart {
                index: 1,
                kind: AssistantBlockKind::Text,
            }),
            Ok(ChatEvent::BlockDelta {
                index: 1,
                kind: AssistantBlockKind::Text,
                delta: "answer".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 22,
                            token: "answer".to_string(),
                            logprob: -0.4,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![22],
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 1,
                    output_token_count: 4,
                    cached_token_count: 0,
                },
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
                ec_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions::default(),
            ResponseOptions {
                requested_logprobs: true,
                return_token_ids: true,
                ..Default::default()
            },
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        let choice = &chunks[1].choices[0];
        assert_eq!(choice.delta.content.as_deref(), Some("answer"));
        assert_eq!(choice.token_ids.as_deref(), Some(&[22][..]));
        let logprobs = choice.logprobs.as_ref().expect("answer logprobs");
        let content = logprobs.content.as_ref().expect("logprobs content");
        assert_eq!(content[0].token, "answer");
        assert!(chunks.iter().all(|chunk| {
            chunk.choices.iter().all(|choice| {
                choice.delta.reasoning.is_none()
                    && !choice
                        .token_ids
                        .as_ref()
                        .is_some_and(|ids| matches!(ids.as_slice(), [11] | [12] | [13]))
                    && choice
                        .logprobs
                        .as_ref()
                        .and_then(|logprobs| logprobs.content.as_ref())
                        .is_none_or(|content| {
                            content.iter().all(|entry| {
                                !matches!(entry.token.as_str(), "<think>" | "think" | "</think>")
                            })
                        })
            })
        }));
    }

    #[tokio::test]
    async fn chunk_stream_preserves_tool_call_index_and_omits_id_from_arguments_delta() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::ToolCallStart {
                index: 3,
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
            }),
            Ok(ChatEvent::ToolCallArgumentsDelta {
                index: 3,
                delta: r#"{"city":"Paris"}"#.to_string(),
            }),
            Ok(ChatEvent::ToolCallEnd {
                index: 3,
                call: AssistantToolCall {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"Paris"}"#.to_string(),
                },
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 1,
                    output_token_count: 1,
                    cached_token_count: 0,
                },
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
                ec_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            ApiServerOptions::default(),
            ResponseOptions {
                include_reasoning: true,
                ..Default::default()
            },
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(
            chunks[1].choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
            3
        );
        assert_eq!(
            chunks[1].choices[0].delta.tool_calls.as_ref().unwrap()[0].id,
            Some("call_1".to_string())
        );
        assert_eq!(
            chunks[2].choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
            3
        );
        assert_eq!(
            chunks[2].choices[0].delta.tool_calls.as_ref().unwrap()[0].id,
            None
        );
    }
}
