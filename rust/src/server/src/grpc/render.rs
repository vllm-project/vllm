// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC `Render` service: engine-free chat render and derender, the gRPC
//! counterpart of the HTTP render API.

use std::num::NonZeroU32;
use std::pin::Pin;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _, TryStreamExt as _, stream};
use itertools::Itertools as _;
use llm_multimodal::ImageDetail;
use thiserror_ext::AsReport as _;
use tokio::sync::oneshot;
use tonic::{Request, Response, Status, Streaming};
use tracing::{info_span, warn};
use tracing_futures::Instrument as _;
use uuid::Uuid;
use vllm_chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart,
    ChatEvent, ChatMessage, ChatOptions, ChatRequest, ChatRequestProcessor, ChatTool,
    ChatToolChoice, DetokenizeChunk, GenerationPromptMode, ResolvedToolContext, SamplingParams,
};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
use vllm_text::{
    DecodedLogprobs, FinishReason, PromptTruncation, PromptTruncationLimit, TextDecodeOptions,
    TextRequestProcessor, TruncationSide,
};

use super::convert::{self, json_to_proto_struct};
use super::inference::text_error_to_status;
use super::media;
use super::pb;
use super::pb::render_server::{Render, RenderServer};
use crate::render::RenderState;
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
        resolution.model_names.accepts(Some(model))
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
        self.model_names().accepts(Some(model))
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
    async fn render_chat(
        &self,
        request: Request<pb::RenderChatRequest>,
    ) -> Result<Response<pb::RenderChatResponse>, Status> {
        let pb::RenderChatRequest {
            chat,
            omit_media_kwargs,
        } = request.into_inner();
        let span = info_span!("grpc_render", rpc = "RenderChat");
        let result = render_chat(
            self.state.as_ref(),
            chat.unwrap_or_default(),
            omit_media_kwargs,
        )
        .instrument(span.clone())
        .await;
        log_failure(&span, &result);
        result.map(Response::new)
    }

    type DerenderChatStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::DerenderChatStreamResponse, Status>> + Send>>;

    async fn derender_chat_stream(
        &self,
        request: Request<Streaming<pb::DerenderChatStreamRequest>>,
    ) -> Result<Response<Self::DerenderChatStreamStream>, Status> {
        let span = info_span!("grpc_render", rpc = "DerenderChatStream");
        let result = derender_chat_stream(self.state.as_ref(), request.into_inner())
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

async fn render_chat(
    state: &impl RenderContext,
    chat: pb::ChatRequest,
    omit_media_kwargs: bool,
) -> Result<pb::RenderChatResponse, Status> {
    check_model(state, &chat.model).await?;
    let requested_structured_output = chat.structured_output.clone();
    let chat_request = to_chat_request(chat)?;
    let structured_outputs = chat_request.sampling_params.structured_outputs.clone();

    let text_request = state
        .chat_processor()
        .prepare(chat_request, NonZeroU32::MIN, state.text_processor())
        .await
        .map_err(chat_error_to_status)?
        .text_request;
    let prepared = state.text_processor().prepare(text_request).map_err(text_error_to_status)?;
    let text_request = prepared.text_request;
    let media: Vec<pb::MediaItem> = (prepared.generate_request.mm_features.into_iter().flatten())
        .map(|feature| media::feature_to_proto(feature, !omit_media_kwargs))
        .try_collect()?;
    // Fail here rather than in Inference.Generate, which rejects the same media.
    media::check_inline_kwargs_size(
        media
            .iter()
            .filter_map(|item| match &item.source {
                Some(pb::media_item::Source::Features(feature)) => feature.kwargs.as_ref(),
                _ => None,
            })
            .map(Vec::len)
            .sum(),
    )?;

    // A parser-built output grammar replaces the request's constraint. Echo an
    // unchanged constraint as sent, so a JSON schema keeps its exact text.
    let structured_output = match text_request.sampling_params.structured_outputs {
        Some(params) if Some(&params) != structured_outputs.as_ref() => {
            Some(convert::structured_output_to_proto(&params))
        }
        Some(_) => requested_structured_output,
        None => None,
    };
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

    Ok(pb::RenderChatResponse {
        prompt_token_ids: prepared.generate_request.prompt_token_ids,
        structured_output,
        engine_reasoning_gate,
        media,
    })
}

// ========================================================================================
// Derender
// ========================================================================================

/// Start a DerenderChatStream from its first message. The returned response
/// stream owns `inbound`.
async fn derender_chat_stream(
    state: &impl RenderContext,
    mut inbound: Streaming<pb::DerenderChatStreamRequest>,
) -> Result<
    impl Stream<Item = Result<pb::DerenderChatStreamResponse, Status>> + Send + 'static,
    Status,
> {
    use pb::derender_chat_stream_request::Request as StreamRequest;

    let Some(pb::DerenderChatStreamRequest {
        request: Some(StreamRequest::Start(mut start)),
    }) = inbound.message().await?
    else {
        return Err(Status::invalid_argument(
            "the first DerenderChatStream message must be `start`",
        ));
    };
    let (end_tx, end_rx) = oneshot::channel();
    let inbound = DerenderInbound {
        messages: inbound,
        first_output: start.output.take(),
        stop_strings: start.stop_strings.clone(),
        end_tx: Some(end_tx),
    };
    let (chat_request, prompt_token_ids) = prepare_derender(state, *start).await?;

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
    Ok(derender_chat_stream_responses(events, end_rx))
}

/// The request stream of a DerenderChatStream after its start message.
struct DerenderInbound {
    messages: Streaming<pb::DerenderChatStreamRequest>,
    /// `start.output`, read before `messages`.
    first_output: Option<pb::SequenceOutput>,
    stop_strings: Vec<String>,
    /// Reports how the chunks ended to the response stream: the finish info of
    /// Generate, or the input error that ended them early in place of the
    /// resulting unfinished-output error.
    end_tx: Option<oneshot::Sender<Result<pb::FinishInfo, Status>>>,
}

impl DerenderInbound {
    /// Read the next output chunk, applying only the stop strings of the start
    /// message: text before a stop string reported in the final chunk may have
    /// been sent already. A finished chunk reports its finish info through
    /// `end_tx`.
    async fn next_chunk(&mut self) -> Result<DetokenizeChunk, Status> {
        use pb::derender_chat_stream_request::Request as StreamRequest;

        let output = match self.first_output.take() {
            Some(output) => output,
            None => match self.messages.message().await? {
                Some(pb::DerenderChatStreamRequest {
                    request: Some(StreamRequest::Output(output)),
                }) => *output,
                Some(_) => {
                    return Err(Status::invalid_argument(
                        "only the first DerenderChatStream message may be `start`",
                    ));
                }
                None => {
                    return Err(Status::invalid_argument(
                        "the DerenderChatStream request stream closed before a finished output",
                    ));
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
async fn derender_chat_stream_responses(
    events: impl Stream<Item = vllm_chat::Result<ChatEvent>> + Send + Unpin,
    mut end_rx: oneshot::Receiver<Result<pb::FinishInfo, Status>>,
    mut y: TryYielder<pb::DerenderChatStreamResponse, Status>,
) -> Result<(), Status> {
    // The cap only bounds the size of one response.
    let mut events = events.ready_chunks(64);
    let mut last_token_id = None;
    while let Some(batch) = events.next().await {
        let mut response = pb::DerenderChatStreamResponse::default();
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
        if response != pb::DerenderChatStreamResponse::default() {
            y.yield_ok(response).await;
        }
    }
    Err(match end_rx.try_recv() {
        Ok(Err(status)) => status,
        _ => Status::internal("derender stream ended without a final output"),
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

/// Check the model and build the chat request of a DerenderChatStream.
async fn prepare_derender(
    state: &impl RenderContext,
    start: pb::DerenderChatStart,
) -> Result<(ChatRequest, Vec<u32>), Status> {
    let pb::DerenderChatStart {
        chat,
        prompt_token_ids,
        output: _,
        stop_strings,
        include_stop_strings,
        min_new_tokens,
        skip_special_tokens,
    } = start;
    let chat = chat.unwrap_or_default();
    check_model(state, &chat.model).await?;
    let mut chat_request = to_chat_request(chat)?;
    let decode_options = &mut chat_request.decode_options;
    decode_options.skip_special_tokens = skip_special_tokens.unwrap_or(true);
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
    // `ChatRequest::validate` checks a continued final message is an assistant's.
    let generation_prompt_mode =
        match enum_from_proto(chat.generation_prompt_mode, "generation_prompt_mode")? {
            pb::GenerationPromptMode::Unspecified | pb::GenerationPromptMode::StartNewAssistant => {
                GenerationPromptMode::StartNewAssistant
            }
            pb::GenerationPromptMode::ContinueFinalAssistant => {
                GenerationPromptMode::ContinueFinalAssistant
            }
            pb::GenerationPromptMode::NoGenerationPrompt => {
                GenerationPromptMode::NoGenerationPrompt
            }
        };
    let messages: Vec<_> = chat.messages.into_iter().map(message_from_proto).try_collect()?;
    let tool_choice = {
        use pb::chat_tool_choice::{Choice, Mode};
        match chat.tool_choice.and_then(|choice| choice.choice) {
            None => None,
            Some(Choice::Mode(mode)) => match enum_from_proto(mode, "tool_choice.mode")? {
                Mode::Unspecified => None,
                Mode::Auto => Some(ChatToolChoice::Auto),
                Mode::None => Some(ChatToolChoice::None),
                Mode::Required => Some(ChatToolChoice::Required),
            },
            Some(Choice::FunctionName(name)) => Some(ChatToolChoice::Function { name }),
        }
    };
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
            structured_outputs: convert::structured_output_from_proto(
                chat.structured_output.as_ref(),
            )?,
            ..Default::default()
        },
        chat_options: ChatOptions {
            generation_prompt_mode,
            chat_template: chat.chat_template,
            reasoning_effort: chat.reasoning_effort.as_deref().map(Into::into),
            response_format: None,
            template_kwargs: chat
                .chat_template_kwargs
                .map(|kwargs| {
                    (kwargs.fields.iter())
                        .map(|(key, value)| (key.clone(), convert::proto_value_to_json(value)))
                        .collect()
                })
                .unwrap_or_default(),
        },
        tool_context,
        // Derender sets the decode options; render does not decode.
        decode_options: TextDecodeOptions::default(),
        intermediate: false,
        return_prompt_text: false,
        prompt_truncation: chat.truncation.map(truncation_from_proto).transpose()?,
        priority: 0,
        documents: None,
        cache_salt: None,
        add_special_tokens: chat.add_special_tokens,
        data_parallel_rank: None,
        session_id: None,
        lora_request: None,
    })
}

fn truncation_from_proto(truncation: pb::PromptTruncation) -> Result<PromptTruncation, Status> {
    use pb::prompt_truncation::{Limit, Side};

    let side = match enum_from_proto(truncation.side, "truncation.side")? {
        Side::Unspecified | Side::Left => TruncationSide::Left,
        Side::Right => TruncationSide::Right,
    };
    let limit = match truncation.limit {
        Some(Limit::MaxTokens(max_tokens)) => PromptTruncationLimit::Fixed(max_tokens),
        Some(Limit::InputBudget(())) => PromptTruncationLimit::InputBudget,
        None => return Err(Status::invalid_argument("truncation.limit is required")),
    };
    Ok(PromptTruncation { limit, side })
}

fn message_from_proto(message: pb::ChatMessage) -> Result<ChatMessage, Status> {
    use pb::chat_message::Message;

    Ok(match message.message {
        Some(Message::System(system)) => ChatMessage::system(content_from_proto(system.content)?),
        Some(Message::Developer(developer)) => {
            let tools = tools_from_proto(developer.tools)?;
            ChatMessage::developer(
                content_from_proto(developer.content)?,
                (!tools.is_empty()).then_some(tools),
            )
        }
        Some(Message::User(user)) => ChatMessage::user(content_from_proto(user.content)?),
        Some(Message::Assistant(assistant)) => ChatMessage::assistant_blocks(
            assistant.blocks.into_iter().map(assistant_block_from_proto).try_collect()?,
        ),
        Some(Message::Tool(tool)) => {
            ChatMessage::tool_response(content_from_proto(tool.content)?, tool.tool_call_id)
        }
        Some(Message::Custom(custom)) if custom.role.is_empty() => {
            return Err(Status::invalid_argument("custom message role is required"));
        }
        Some(Message::Custom(custom)) => {
            ChatMessage::custom(custom.role, content_from_proto(custom.content)?)
        }
        None => return Err(Status::invalid_argument("chat message is required")),
    })
}

fn assistant_block_from_proto(
    block: pb::AssistantContentBlock,
) -> Result<AssistantContentBlock, Status> {
    use pb::assistant_content_block::Block;

    Ok(match block.block {
        Some(Block::Text(text)) => AssistantContentBlock::Text { text },
        Some(Block::Reasoning(text)) => AssistantContentBlock::Reasoning { text },
        Some(Block::ToolCall(call)) => AssistantContentBlock::ToolCall(AssistantToolCall {
            id: call.id,
            name: call.name,
            // Like the HTTP chat route and Python, a call without arguments
            // renders as an empty object.
            arguments: if call.arguments.is_empty() {
                "{}".to_string()
            } else {
                call.arguments
            },
        }),
        None => {
            return Err(Status::invalid_argument(
                "assistant content block is required",
            ));
        }
    })
}

/// Omitted content renders as empty text.
fn content_from_proto(content: Option<pb::ChatContent>) -> Result<ChatContent, Status> {
    use pb::chat_content::Content;

    Ok(match content.and_then(|content| content.content) {
        None => ChatContent::Text(String::new()),
        Some(Content::Text(text)) => ChatContent::Text(text),
        Some(Content::Parts(parts)) => {
            ChatContent::Parts(parts.parts.into_iter().map(part_from_proto).try_collect()?)
        }
    })
}

fn part_from_proto(part: pb::ChatContentPart) -> Result<ChatContentPart, Status> {
    use pb::chat_content_part::Part;

    Ok(match part.part {
        Some(Part::Text(text)) => ChatContentPart::text(text),
        Some(Part::ImageUrl(image)) => ChatContentPart::ImageUrl {
            detail: match enum_from_proto(image.detail, "image_url.detail")? {
                pb::ImageDetail::Unspecified => None,
                pb::ImageDetail::Auto => Some(ImageDetail::Auto),
                pb::ImageDetail::Low => Some(ImageDetail::Low),
                pb::ImageDetail::High => Some(ImageDetail::High),
            },
            image_url: image.url,
            uuid: image.uuid,
        },
        Some(Part::VideoUrl(video)) => ChatContentPart::VideoUrl {
            video_url: video.url,
            uuid: video.uuid,
        },
        Some(Part::AudioUrl(audio)) => ChatContentPart::AudioUrl {
            audio_url: audio.url,
            uuid: audio.uuid,
        },
        Some(Part::InputAudio(audio)) => ChatContentPart::InputAudio {
            data: audio.data,
            format: audio.format,
            uuid: audio.uuid,
        },
        None => return Err(Status::invalid_argument("chat content part is required")),
    })
}

/// Read a proto enum field, rejecting values this server does not know.
fn enum_from_proto<E: TryFrom<i32>>(value: i32, field: &str) -> Result<E, Status> {
    E::try_from(value)
        .map_err(|_| Status::invalid_argument(format!("unknown {field} value {value}")))
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

// ========================================================================================
// Errors
// ========================================================================================

/// Media that cannot be fetched or preprocessed is the request's fault, as in
/// the HTTP render route.
fn chat_error_to_status(error: vllm_chat::Error) -> Status {
    let message = error.to_report_string();
    if error.is_request_validation_error()
        || matches!(
            error,
            vllm_chat::Error::Multimodal(_) | vllm_chat::Error::Media(_)
        )
    {
        Status::invalid_argument(message)
    } else if error.is_unavailable() {
        Status::unavailable(message)
    } else {
        Status::internal(message)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use llm_multimodal::MediaConnectorError;

    use super::*;

    /// A media fetch that failed on the host's side is still the request's
    /// error here, as in HTTP render; only chat routes tell it apart.
    #[test]
    fn media_fetch_timeout_is_invalid_argument() {
        let error = vllm_chat::Error::from(MediaConnectorError::Timeout {
            url: "https://example.com/a.png".to_string(),
            budget: Duration::from_secs(5),
        });
        assert_eq!(
            chat_error_to_status(error).code(),
            tonic::Code::InvalidArgument
        );
    }
}
