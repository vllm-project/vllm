// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC `Render` service: engine-free chat tokenize and detokenize, the gRPC
//! counterpart of the HTTP render API.

use std::pin::Pin;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _, TryStreamExt as _, stream};
use itertools::Itertools as _;
use thiserror_ext::AsReport as _;
use tokio::sync::oneshot;
use tonic::{Request, Response, Status, Streaming};
use tracing::{info_span, warn};
use tracing_futures::Instrument as _;
use uuid::Uuid;
use vllm_chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessageExt as _, AssistantToolCall,
    ChatContent, ChatContentPart, ChatEvent, ChatEventStream, ChatMessage, ChatOptions,
    ChatRequest, ChatRequestProcessor, ChatTool, ChatToolChoice, DetokenizeChunk,
    ResolvedToolContext, SamplingParams,
};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputsParams;
use vllm_text::{
    DecodedLogprobs, FinishReason, PromptTruncation, PromptTruncationLimit, TextDecodeOptions,
    TextRequestProcessor, TruncationSide,
};

use super::convert::{self, json_to_proto_struct};
use super::inference::text_error_to_status;
use super::media;
use super::pb;
use super::pb::render_server::{Render, RenderServer};
use crate::error::ApiError;
use crate::render::RenderState;
use crate::routes::openai::chat_completions::convert::normalize_generation_prompt_mode;
use crate::state::AppState;

pub(crate) type RenderGrpcService<S> = RenderServer<RenderServiceImpl<S>>;

/// Chat and text processors shared by the engine-backed and render-only servers.
pub(crate) trait RenderContext: Send + Sync + 'static {
    fn chat_processor(&self) -> &ChatRequestProcessor;
    fn text_processor(&self) -> &TextRequestProcessor;
    /// Whether `model` names a served model or LoRA adapter.
    fn is_served_model(&self, model: &str) -> impl Future<Output = bool> + Send;
}

impl RenderContext for AppState {
    fn chat_processor(&self) -> &ChatRequestProcessor {
        self.chat.request_processor()
    }

    fn text_processor(&self) -> &TextRequestProcessor {
        self.chat.text().request_processor()
    }

    async fn is_served_model(&self, model: &str) -> bool {
        let resolution = self.resolve_model_with_loras(Some(model)).await;
        resolution.model_names.iter().any(|name| name == model)
    }
}

impl RenderContext for RenderState {
    fn chat_processor(&self) -> &ChatRequestProcessor {
        &self.chat
    }

    fn text_processor(&self) -> &TextRequestProcessor {
        &self.text
    }

    async fn is_served_model(&self, model: &str) -> bool {
        self.model_names().iter().any(|name| name == model)
    }
}

/// gRPC render service backed by either server's chat and text processors.
pub(crate) struct RenderServiceImpl<S> {
    state: Arc<S>,
}

impl<S> RenderServiceImpl<S> {
    pub(crate) fn new(state: Arc<S>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl<S: RenderContext> Render for RenderServiceImpl<S> {
    async fn tokenize(
        &self,
        request: Request<pb::TokenizeRequest>,
    ) -> Result<Response<pb::TokenizeResponse>, Status> {
        let pb::TokenizeRequest {
            chat,
            omit_media_kwargs,
        } = request.into_inner();
        let span = info_span!("grpc_render", rpc = "Tokenize");
        let result = tokenize(
            self.state.as_ref(),
            chat.unwrap_or_default(),
            omit_media_kwargs,
        )
        .instrument(span.clone())
        .await;
        log_failure(&span, &result);
        result.map(Response::new)
    }

    async fn detokenize(
        &self,
        request: Request<pb::DetokenizeRequest>,
    ) -> Result<Response<pb::DetokenizeResponse>, Status> {
        let span = info_span!("grpc_render", rpc = "Detokenize");
        let result = detokenize(self.state.as_ref(), request.into_inner())
            .instrument(span.clone())
            .await;
        log_failure(&span, &result);
        result.map(Response::new)
    }

    type DetokenizeStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::DetokenizeStreamResponse, Status>> + Send>>;

    async fn detokenize_stream(
        &self,
        request: Request<Streaming<pb::DetokenizeStreamRequest>>,
    ) -> Result<Response<Self::DetokenizeStreamStream>, Status> {
        let span = info_span!("grpc_render", rpc = "DetokenizeStream");
        let result = detokenize_stream(self.state.as_ref(), request.into_inner())
            .instrument(span.clone())
            .await;
        log_failure(&span, &result);
        let responses = result?.inspect_err(move |status| {
            log_failure::<()>(&span, &Err(status.clone()));
        });
        Ok(Response::new(Box::pin(responses)))
    }
}

fn log_failure<T>(span: &tracing::Span, result: &Result<T, Status>) {
    if let Err(status) = result {
        warn!(
            parent: span,
            grpc_code = ?status.code(),
            error = %status.message(),
            "gRPC render request failed"
        );
    }
}

async fn tokenize(
    state: &impl RenderContext,
    chat: pb::ChatRequest,
    omit_media_kwargs: bool,
) -> Result<pb::TokenizeResponse, Status> {
    check_model(state, &chat.model).await?;
    let chat_request = to_chat_request(chat)?;
    let structured_outputs = chat_request.sampling_params.structured_outputs.clone();

    let (text_request, _) = state
        .chat_processor()
        .prepare(chat_request, state.text_processor())
        .await
        .map_err(chat_error_to_status)?;
    let prepared = state.text_processor().prepare(text_request).map_err(text_error_to_status)?;
    let text_request = prepared.text_request;
    let media = (prepared.generate_request.mm_features.into_iter().flatten())
        .map(|feature| media::feature_to_proto(feature, !omit_media_kwargs))
        .try_collect()?;

    // A parser-built output grammar replaces the request's structured output.
    let structural_tag = text_request
        .sampling_params
        .structured_outputs
        .filter(|params| Some(params) != structured_outputs.as_ref())
        .map(|params| {
            params.constraint.into_structural_tag().map_err(|constraint| {
                Status::internal(format!(
                    "parser output grammar is not a structural tag: {constraint:?}"
                ))
            })
        })
        .transpose()?;
    let ReasoningParserKwargs {
        chat_template_kwargs,
    } = text_request.reasoning_parser_kwargs;
    let engine_reasoning_gate = (text_request.reasoning_ended.is_some()
        || !chat_template_kwargs.is_empty())
    .then(|| pb::EngineReasoningGate {
        reasoning_ended: text_request.reasoning_ended,
        chat_template_kwargs: json_to_proto_struct(&serde_json::Value::Object(
            chat_template_kwargs.into_iter().collect(),
        )),
    });

    Ok(pb::TokenizeResponse {
        prompt_token_ids: prepared.generate_request.prompt_token_ids,
        media,
        structural_tag,
        skip_special_tokens: text_request.decode_options.skip_special_tokens,
        engine_reasoning_gate,
    })
}

async fn detokenize(
    state: &impl RenderContext,
    mut request: pb::DetokenizeRequest,
) -> Result<pb::DetokenizeResponse, Status> {
    let output = request.output.take().unwrap_or_default();
    let reported_finish_info = output.finish_info.clone();
    // A stop string reported by Generate cuts the text as well.
    let reported_stop_string = reported_stop_string(&output).cloned();
    let (chat_request, prompt_token_ids) =
        prepare_detokenize(state, request, reported_stop_string).await?;
    let mut chunk = chunk_from_proto(output)?;
    // Unlike in a stream, an omitted `finish_info` ends the output.
    chunk.finish_reason.get_or_insert(FinishReason::Length);
    let collected = state
        .chat_processor()
        .detokenize(
            chat_request,
            state.text_processor(),
            prompt_token_ids,
            chunk,
        )
        .await
        .map_err(chat_error_to_status)?;

    let finish_info = detokenized_finish_info(
        reported_finish_info,
        &collected.finish_reason,
        collected.usage.output_token_count as u32,
        &collected.token_ids,
    )?;
    let message = collected.message;
    Ok(pb::DetokenizeResponse {
        content: message.text(),
        reasoning: message.reasoning().unwrap_or_default(),
        tool_calls: message
            .tool_calls()
            .map(|call| pb::ChatToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            })
            .collect(),
        num_reasoning_tokens: collected.usage.reasoning_tokens as u32,
        logprobs: logprobs_to_proto(collected.logprobs),
        finish_info: Some(finish_info),
    })
}

/// Start a DetokenizeStream from its first message. The returned response
/// stream owns `inbound`.
async fn detokenize_stream(
    state: &impl RenderContext,
    mut inbound: Streaming<pb::DetokenizeStreamRequest>,
) -> Result<impl Stream<Item = Result<pb::DetokenizeStreamResponse, Status>> + Send + 'static, Status>
{
    use pb::detokenize_stream_request::Request as StreamRequest;

    let Some(pb::DetokenizeStreamRequest {
        request: Some(StreamRequest::Start(mut start)),
    }) = inbound.message().await?
    else {
        return Err(Status::invalid_argument(
            "the first DetokenizeStream message must be `start`",
        ));
    };
    let (end_tx, end_rx) = oneshot::channel();
    let inbound = DetokenizeInbound {
        messages: inbound,
        first_output: start.output.take(),
        stop_strings: start.stop_strings.clone(),
        end_tx: Some(end_tx),
    };
    let (chat_request, prompt_token_ids) = prepare_detokenize(state, *start, None).await?;

    let chunks = stream::unfold(Some(inbound), |inbound| async move {
        let mut inbound = inbound?;
        match inbound.next_chunk().await {
            Ok(chunk) => {
                let inbound = chunk.finish_reason.is_none().then_some(inbound);
                Some((chunk, inbound))
            }
            Err(status) => {
                inbound.end(Err(status));
                None
            }
        }
    })
    .boxed();

    let events = state
        .chat_processor()
        .detokenize_stream(
            chat_request,
            state.text_processor(),
            prompt_token_ids,
            chunks,
        )
        .map_err(chat_error_to_status)?;
    Ok(detokenize_stream_responses(events, end_rx))
}

/// The request stream of a DetokenizeStream after its start message.
struct DetokenizeInbound {
    messages: Streaming<pb::DetokenizeStreamRequest>,
    /// `start.output`, read before `messages`.
    first_output: Option<pb::SequenceOutput>,
    stop_strings: Vec<String>,
    /// Reports how the chunks ended to the response stream: the finish info of
    /// Generate, or the input error that ended them early in place of the
    /// resulting unfinished-output error.
    end_tx: Option<oneshot::Sender<Result<pb::FinishInfo, Status>>>,
}

impl DetokenizeInbound {
    /// Read the next output chunk, applying only the stop strings of the start
    /// message: text before a stop string reported in the final chunk may have
    /// been sent already. A finished chunk reports its finish info through
    /// `end_tx`.
    async fn next_chunk(&mut self) -> Result<DetokenizeChunk, Status> {
        use pb::detokenize_stream_request::Request as StreamRequest;

        let output = match self.first_output.take() {
            Some(output) => output,
            None => match self.messages.message().await? {
                Some(pb::DetokenizeStreamRequest {
                    request: Some(StreamRequest::Output(output)),
                }) => *output,
                Some(_) => {
                    return Err(Status::invalid_argument(
                        "only the first DetokenizeStream message may be `start`",
                    ));
                }
                // Closing early ends the output like an omitted `finish_info`
                // in Detokenize.
                None => {
                    return Ok(DetokenizeChunk {
                        finish_reason: Some(FinishReason::Length),
                        ..Default::default()
                    });
                }
            },
        };
        if let Some(stop) = reported_stop_string(&output)
            && !self.stop_strings.contains(stop)
        {
            return Err(Status::invalid_argument(format!(
                "output stopped on stop string {stop:?}, which is not in `start.stop_strings`"
            )));
        }
        let finish_info = output.finish_info.clone();
        let chunk = chunk_from_proto(output)?;
        if chunk.finish_reason.is_some()
            && let Some(finish_info) = finish_info
        {
            self.end(Ok(finish_info));
        }
        Ok(chunk)
    }

    fn end(&mut self, end: Result<pb::FinishInfo, Status>) {
        if let Some(tx) = self.end_tx.take() {
            let _ = tx.send(end);
        }
    }
}

/// Coalesce the chat events available at once into one response, ending with
/// the response that carries `finish_info`.
#[try_stream]
async fn detokenize_stream_responses(
    events: ChatEventStream,
    mut end_rx: oneshot::Receiver<Result<pb::FinishInfo, Status>>,
    mut y: TryYielder<pb::DetokenizeStreamResponse, Status>,
) -> Result<(), Status> {
    // The cap only bounds the size of one response.
    let mut events = events.ready_chunks(64);
    let mut last_token_id = None;
    while let Some(batch) = events.next().await {
        let mut response = pb::DetokenizeStreamResponse::default();
        for event in batch {
            let event = event.map_err(|error| match end_rx.try_recv() {
                Ok(Err(status)) => status,
                _ => chat_error_to_status(error),
            })?;
            match event {
                ChatEvent::BlockDelta {
                    kind: AssistantBlockKind::Text,
                    delta,
                    ..
                } => response.content.push_str(&delta),
                ChatEvent::BlockDelta {
                    kind: AssistantBlockKind::Reasoning,
                    delta,
                    ..
                } => response.reasoning.push_str(&delta),
                ChatEvent::ToolCallStart { index, id, name } => {
                    response.tool_calls.push(pb::ChatToolCallDelta {
                        index: index as u32,
                        id,
                        name,
                        arguments: String::new(),
                    });
                }
                ChatEvent::ToolCallArgumentsDelta { index, delta } => {
                    match response.tool_calls.last_mut() {
                        Some(call) if call.index == index as u32 => call.arguments.push_str(&delta),
                        _ => response.tool_calls.push(pb::ChatToolCallDelta {
                            index: index as u32,
                            arguments: delta,
                            ..Default::default()
                        }),
                    }
                }
                ChatEvent::LogprobsDelta {
                    logprobs,
                    token_ids,
                } => {
                    last_token_id = token_ids.last().copied().or(last_token_id);
                    response.logprobs.extend(logprobs_to_proto(logprobs));
                }
                ChatEvent::Done {
                    usage,
                    finish_reason,
                    ..
                } => {
                    // Unset when the output ended before a finished chunk, such
                    // as at a stop string.
                    let reported_finish_info = end_rx.try_recv().ok().and_then(Result::ok);
                    response.finish_info = Some(detokenized_finish_info(
                        reported_finish_info,
                        &finish_reason,
                        usage.output_token_count as u32,
                        last_token_id.as_slice(),
                    )?);
                    response.num_reasoning_tokens = usage.reasoning_tokens as u32;
                    y.yield_ok(response).await;
                    return Ok(());
                }
                ChatEvent::Start { .. }
                | ChatEvent::BlockStart { .. }
                | ChatEvent::BlockDelta {
                    kind: AssistantBlockKind::ToolCall,
                    ..
                }
                | ChatEvent::BlockEnd { .. }
                | ChatEvent::ToolCallEnd { .. } => {}
            }
        }
        if response != pb::DetokenizeStreamResponse::default() {
            y.yield_ok(response).await;
        }
    }
    Err(match end_rx.try_recv() {
        Ok(Err(status)) => status,
        _ => Status::internal("detokenize stream ended without a final output"),
    })
}

/// Generate's finish info with the decoded output's finish reason, stop reason
/// and token count.
fn detokenized_finish_info(
    reported: Option<pb::FinishInfo>,
    finish_reason: &FinishReason,
    num_output_tokens: u32,
    token_ids: &[u32],
) -> Result<pb::FinishInfo, Status> {
    use pb::finish_info::{FinishReason as PbFinishReason, StopReason as PbStopReason};

    let reported = reported.unwrap_or_default();
    let (finish_reason, stop_reason) = match (finish_reason, &reported.stop_reason) {
        (FinishReason::Stop(None), Some(stop @ PbStopReason::EosTokenId(_))) => {
            (PbFinishReason::Stop as i32, Some(stop.clone()))
        }
        (FinishReason::Length, Some(PbStopReason::StopString(stop))) => {
            return Err(Status::invalid_argument(format!(
                "output stopped on stop string {stop:?}, which the decoded output does not \
                 contain past `min_new_tokens`"
            )));
        }
        (finish_reason, _) => convert::finish_reason_to_proto(finish_reason, token_ids)?,
    };
    Ok(pb::FinishInfo {
        num_output_tokens,
        finish_reason,
        stop_reason,
        ..reported
    })
}

/// Check the model and build the chat request of either Detokenize RPC.
/// `extra_stop_string` applies on top of the requested stop strings.
async fn prepare_detokenize(
    state: &impl RenderContext,
    request: pb::DetokenizeRequest,
    extra_stop_string: Option<String>,
) -> Result<(ChatRequest, Vec<u32>), Status> {
    let pb::DetokenizeRequest {
        chat,
        prompt_token_ids,
        output: _,
        mut stop_strings,
        include_stop_strings,
        min_new_tokens,
    } = request;
    if let Some(stop) = extra_stop_string
        && !stop_strings.contains(&stop)
    {
        stop_strings.push(stop);
    }
    let chat = chat.unwrap_or_default();
    check_model(state, &chat.model).await?;
    let mut chat_request = to_chat_request(chat)?;
    let decode_options = &mut chat_request.decode_options;
    decode_options.stop_strings = (!stop_strings.is_empty()).then_some(stop_strings);
    decode_options.include_stop_str_in_output = include_stop_strings;
    decode_options.min_tokens = min_new_tokens;
    Ok((chat_request, prompt_token_ids))
}

fn chunk_from_proto(output: pb::SequenceOutput) -> Result<DetokenizeChunk, Status> {
    Ok(DetokenizeChunk {
        logprobs: convert::output_logprobs_from_proto(&output)?,
        finish_reason: output.finish_info.as_ref().and_then(finish_reason_from_proto),
        token_ids: output.token_ids,
    })
}

fn reported_stop_string(output: &pb::SequenceOutput) -> Option<&String> {
    match &output.finish_info {
        Some(pb::FinishInfo {
            stop_reason: Some(pb::finish_info::StopReason::StopString(stop)),
            ..
        }) => Some(stop),
        _ => None,
    }
}

fn logprobs_to_proto(logprobs: Option<DecodedLogprobs>) -> Vec<pb::TokenLogprobs> {
    logprobs
        .map(|logprobs| logprobs.positions)
        .unwrap_or_default()
        .into_iter()
        .map(|position| pb::TokenLogprobs {
            tokens: position
                .entries
                .into_iter()
                .map(|entry| pb::DecodedTokenLogprob {
                    id: entry.token_id,
                    token: entry.token,
                    logprob: entry.logprob,
                    rank: entry.rank,
                })
                .collect(),
        })
        .collect()
}

/// Map Generate finish info to the finish reason that drives decoding, or
/// `None` while the output is not finished.
///
/// Only token stops end the output with a token that is hidden from the text.
/// A stop string is not a token stop: the decoder cuts the text at it instead.
fn finish_reason_from_proto(info: &pb::FinishInfo) -> Option<FinishReason> {
    use pb::finish_info::{FinishReason as PbFinishReason, StopReason as PbStopReason};

    Some(match info.finish_reason() {
        PbFinishReason::Stop => match &info.stop_reason {
            Some(PbStopReason::StopTokenId(id)) => {
                FinishReason::Stop(Some(StopReason::TokenId(*id)))
            }
            Some(PbStopReason::EosTokenId(_)) | None => FinishReason::stop_eos(),
            Some(PbStopReason::StopString(_)) => FinishReason::Length,
        },
        PbFinishReason::Aborted => FinishReason::Abort,
        PbFinishReason::Length => FinishReason::Length,
        PbFinishReason::NotFinished => return None,
    })
}

// ========================================================================================
// Request conversion
// ========================================================================================

/// Reject a model that is neither served nor a LoRA adapter. An empty name
/// selects the default model.
async fn check_model(state: &impl RenderContext, model: &str) -> Result<(), Status> {
    if model.is_empty() || state.is_served_model(model).await {
        Ok(())
    } else {
        Err(Status::not_found(format!("model `{model}` not found")))
    }
}

fn to_chat_request(chat: pb::ChatRequest) -> Result<ChatRequest, Status> {
    let messages: Vec<_> = chat.messages.into_iter().map(message_from_proto).try_collect()?;
    let generation_prompt_mode = normalize_generation_prompt_mode(
        chat.add_generation_prompt,
        chat.continue_final_message,
        &messages,
    )
    .map_err(api_error_to_status)?;
    let tool_choice = chat.tool_choice.and_then(|choice| choice.choice).and_then(|choice| {
        use pb::chat_tool_choice::{Choice, Mode};
        match choice {
            Choice::Mode(mode) => match Mode::try_from(mode).unwrap_or_default() {
                Mode::Unspecified => None,
                Mode::Auto => Some(ChatToolChoice::Auto),
                Mode::None => Some(ChatToolChoice::None),
                Mode::Required => Some(ChatToolChoice::Required),
            },
            Choice::FunctionName(name) => Some(ChatToolChoice::Function { name }),
        }
    });
    let tool_context = ResolvedToolContext::new(
        &messages,
        tools_from_proto(chat.tools)?,
        tool_choice,
        chat.parallel_tool_calls.unwrap_or(true),
    )
    .map_err(chat_error_to_status)?;

    Ok(ChatRequest {
        request_id: Uuid::new_v4().to_string(),
        messages,
        sampling_params: SamplingParams {
            structured_outputs: structured_output_from_proto(chat.structured_output)?,
            ..Default::default()
        },
        chat_options: ChatOptions {
            generation_prompt_mode,
            chat_template: chat.chat_template,
            reasoning_effort: chat.reasoning_effort.as_deref().map(Into::into),
            response_format: None,
            template_kwargs: chat
                .chat_template_kwargs
                .map(|kwargs| match convert::proto_struct_to_json(&kwargs) {
                    serde_json::Value::Object(map) => map.into_iter().collect(),
                    _ => unreachable!("a proto struct converts to a JSON object"),
                })
                .unwrap_or_default(),
        },
        tool_context,
        decode_options: TextDecodeOptions {
            skip_special_tokens: chat.skip_special_tokens.unwrap_or(true),
            ..Default::default()
        },
        intermediate: false,
        // Proto3 uses zero as unset, as in `GenerateRequest`.
        prompt_truncation: (chat.truncate_prompt_tokens != 0).then_some(PromptTruncation {
            limit: PromptTruncationLimit::Fixed(u64::from(chat.truncate_prompt_tokens)),
            side: TruncationSide::Left,
        }),
        priority: 0,
        documents: None,
        cache_salt: None,
        add_special_tokens: chat.add_special_tokens,
        data_parallel_rank: None,
        session_id: None,
        lora_request: None,
    })
}

fn message_from_proto(message: pb::ChatMessage) -> Result<ChatMessage, Status> {
    let pb::ChatMessage {
        role,
        content,
        reasoning,
        tool_calls,
        tool_call_id,
        tools,
    } = message;
    let content = content_from_proto(content)?;
    Ok(match role.as_str() {
        "system" => ChatMessage::system(content),
        "user" => ChatMessage::user(content),
        "developer" => {
            let tools = tools_from_proto(tools)?;
            ChatMessage::developer(content, (!tools.is_empty()).then_some(tools))
        }
        "tool" => ChatMessage::tool_response(content, tool_call_id),
        "assistant" => {
            let mut blocks = Vec::new();
            if !reasoning.is_empty() {
                blocks.push(AssistantContentBlock::Reasoning { text: reasoning });
            }
            match content {
                ChatContent::Text(text) => blocks.push(AssistantContentBlock::Text { text }),
                ChatContent::Parts(parts) => {
                    for part in parts {
                        let ChatContentPart::Text { text } = part else {
                            return Err(Status::invalid_argument(
                                "assistant messages support text content parts only",
                            ));
                        };
                        blocks.push(AssistantContentBlock::Text { text });
                    }
                }
            }
            blocks.extend(tool_calls.into_iter().map(|call| {
                AssistantContentBlock::ToolCall(AssistantToolCall {
                    id: call.id,
                    name: call.name,
                    arguments: call.arguments,
                })
            }));
            ChatMessage::assistant_blocks(blocks)
        }
        "" => return Err(Status::invalid_argument("chat message role is required")),
        _ => ChatMessage::custom(role, content),
    })
}

fn content_from_proto(content: Option<pb::chat_message::Content>) -> Result<ChatContent, Status> {
    use pb::chat_content_part::Part;
    use pb::chat_message::Content;

    Ok(match content {
        None => ChatContent::Text(String::new()),
        Some(Content::Text(text)) => ChatContent::Text(text),
        Some(Content::Parts(parts)) => ChatContent::Parts(
            parts
                .parts
                .into_iter()
                .filter_map(|part| part.part)
                .map(|part| {
                    Ok(match part {
                        Part::Text(text) => ChatContentPart::text(text),
                        Part::ImageUrl(image) => ChatContentPart::ImageUrl {
                            image_url: image.url,
                            detail: image
                                .detail
                                .map(|detail| {
                                    serde_json::from_value(serde_json::Value::String(detail))
                                        .map_err(|error| {
                                            Status::invalid_argument(format!(
                                                "invalid image detail: {error}"
                                            ))
                                        })
                                })
                                .transpose()?,
                            uuid: image.uuid,
                        },
                        Part::VideoUrl(video) => ChatContentPart::VideoUrl {
                            video_url: video.url,
                            uuid: video.uuid,
                        },
                        Part::AudioUrl(audio) => ChatContentPart::AudioUrl {
                            audio_url: audio.url,
                            uuid: audio.uuid,
                        },
                        Part::InputAudio(audio) => ChatContentPart::InputAudio {
                            data: audio.data,
                            format: audio.format,
                            uuid: audio.uuid,
                        },
                    })
                })
                .collect::<Result<_, Status>>()?,
        ),
    })
}

fn tools_from_proto(tools: Vec<pb::ChatTool>) -> Result<Vec<ChatTool>, Status> {
    tools
        .into_iter()
        .map(|tool| {
            let parameters = if tool.parameters_json.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::from_str(&tool.parameters_json).map_err(|error| {
                    Status::invalid_argument(format!(
                        "invalid parameters_json for tool `{}`: {error}",
                        tool.name
                    ))
                })?
            };
            Ok(ChatTool {
                name: tool.name,
                description: tool.description,
                parameters,
                strict: tool.strict,
                defer_loading: tool.defer_loading,
            })
        })
        .collect()
}

fn structured_output_from_proto(
    structured_output: Option<pb::chat_request::StructuredOutput>,
) -> Result<Option<StructuredOutputsParams>, Status> {
    use pb::chat_request::StructuredOutput as Chat;
    use pb::decoding_parameters::StructuredOutput as Decoding;

    let structured_output = structured_output.map(|output| match output {
        Chat::Json(schema) => Decoding::Json(schema),
        Chat::Regex(regex) => Decoding::Regex(regex),
        Chat::Choice(choices) => Decoding::Choice(choices),
        Chat::Grammar(grammar) => Decoding::Grammar(grammar),
        Chat::JsonObject(json_object) => Decoding::JsonObject(json_object),
        Chat::StructuralTag(tag) => Decoding::StructuralTag(tag),
    });
    convert::convert_structured_output(&pb::DecodingParameters {
        structured_output,
        ..Default::default()
    })
}

// ========================================================================================
// Errors
// ========================================================================================

fn chat_error_to_status(error: vllm_chat::Error) -> Status {
    let message = error.to_report_string();
    if error.is_request_validation_error() {
        Status::invalid_argument(message)
    } else {
        Status::internal(message)
    }
}

fn api_error_to_status(error: ApiError) -> Status {
    let message = error.to_error_response().error.message;
    match error {
        ApiError::InvalidRequest { .. } | ApiError::JsonParseError { .. } => {
            Status::invalid_argument(message)
        }
        ApiError::ModelNotFound { .. } | ApiError::NotFound { .. } => Status::not_found(message),
        ApiError::ServiceUnavailable { .. } => Status::unavailable(message),
        ApiError::Conflict { .. } => Status::failed_precondition(message),
        ApiError::ServerError { .. } => Status::internal(message),
    }
}
