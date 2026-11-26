// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC `Render` service: engine-free chat rendering, the gRPC
//! counterpart of the HTTP render API.

use std::num::NonZeroU32;
use std::pin::Pin;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::Json;
use axum::http::HeaderMap;
use futures::{Stream, StreamExt as _, stream};
use itertools::Itertools as _;
use llm_multimodal::ImageDetail;
use thiserror_ext::AsReport as _;
use tokio::sync::oneshot;
use tonic::{Request, Response, Status, Streaming};
use tracing::{info_span, warn};
use tracing_futures::Instrument as _;
use uuid::Uuid;
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatEvent, ChatMessage,
    ChatOptions, ChatRequest, ChatTool, ChatToolChoice, DetokenizeChunk, GenerationPromptMode,
    ResolvedToolContext, SamplingParams,
};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
use vllm_text::{
    FinishReason, PromptTruncation, PromptTruncationLimit, TextDecodeOptions, TruncationSide,
};

use super::convert::{self, json_to_proto_struct};
use super::inference::text_error_to_status;
use super::media;
use super::pb;
use super::pb::render_server::{Render, RenderServer};
use crate::error::ApiError;
use crate::render::RenderContext;
use crate::routes::openai::ChatCompletionRequest;
use crate::routes::openai::chat_completions::convert::PreparedRequest;
use crate::routes::openai::chat_completions::{
    ChatOutputContext, chat_completion_chunk_json, collect_chat_completion_json,
};
use crate::routes::render::{LoweredChatRender, lower_chat_render_request};
use crate::utils::{SSE_DONE_FRAME, sse_data_frame, unix_timestamp};

pub(crate) type RenderGrpcService<S> = RenderServer<RenderServiceImpl<S>>;

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

    async fn render(
        &self,
        request: Request<pb::RenderRequest>,
    ) -> Result<Response<pb::RenderResponse>, Status> {
        let headers = request.metadata().clone().into_headers();
        let request = request.into_inner();
        let span = info_span!("grpc_render", rpc = "Render");
        let result = render(self.state.as_ref(), request, &headers).instrument(span.clone()).await;
        log_failure(&span, &result);
        result.map(Response::new)
    }

    type DerenderStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::DerenderStreamResponse, Status>> + Send>>;

    async fn derender_stream(
        &self,
        request: Request<Streaming<pb::DerenderStreamRequest>>,
    ) -> Result<Response<Self::DerenderStreamStream>, Status> {
        let span = info_span!("grpc_render", rpc = "DerenderStream");
        let headers = request.metadata().clone().into_headers();
        let result = derender_stream(self.state.as_ref(), request.into_inner(), &headers)
            .instrument(span.clone())
            .await;
        log_failure(&span, &result);
        let responses = result?.instrument(span.clone()).inspect(move |response| {
            log_failure(&span, response);
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

async fn render(
    state: &impl RenderContext,
    request: pb::RenderRequest,
    headers: &HeaderMap,
) -> Result<pb::RenderResponse, Status> {
    check_chat_api(request.api)?;

    let omit_media_kwargs =
        match enum_from_proto::<pb::MmPayload>(request.mm_payload, "mm_payload")? {
            pb::MmPayload::Unspecified | pb::MmPayload::Tensors => false,
            pb::MmPayload::None => true,
        };

    let RawChat { prepared, .. } = prepare_raw_chat(state, &request.body, headers).await?;

    let RenderedChat {
        prompt_token_ids,
        structured_output,
        engine_reasoning_gate,
        media,
    } = prepare_and_render_chat(state, prepared.chat_request, None, omit_media_kwargs).await?;

    Ok(pb::RenderResponse {
        prompt_token_ids,
        structured_output,
        engine_reasoning_gate,
        media,
    })
}

/// Only Chat Completions bodies are supported.
fn check_chat_api(api: i32) -> Result<(), Status> {
    match enum_from_proto::<pb::Api>(api, "api")? {
        pb::Api::Chat => Ok(()),
        pb::Api::Unspecified => Err(Status::invalid_argument("api is required")),
        other => Err(Status::unimplemented(format!(
            "api `{}` is not supported",
            other.as_str_name()
        ))),
    }
}

/// A raw chat completion body lowered the way the HTTP route lowers it.
struct RawChat {
    prepared: PreparedRequest,
    /// The body's `stream`.
    stream: bool,
}

/// Parse, validate and lower a raw Chat Completions body, shared by `Render`
/// and `DerenderStream` so both lower the same body the same way. `headers` is the
/// gRPC metadata, read as the HTTP route reads its headers.
async fn prepare_raw_chat(
    state: &impl RenderContext,
    body: &[u8],
    headers: &HeaderMap,
) -> Result<RawChat, Status> {
    // Parse exactly as the HTTP extractor does, so a body gets the same result
    // and error on both protocols.
    let Json(body) = Json::<ChatCompletionRequest>::from_bytes(body)
        .map_err(|rejection| api_error_to_status(rejection.into()))?;
    let stream = body.stream;

    let LoweredChatRender { prepared, .. } = lower_chat_render_request(state, body, headers)
        .await
        .map_err(api_error_to_status)?;

    Ok(RawChat { prepared, stream })
}

async fn render_chat(
    state: &impl RenderContext,
    chat: pb::ChatRequest,
    omit_media_kwargs: bool,
) -> Result<pb::RenderChatResponse, Status> {
    check_model(state, &chat.model).await?;
    let requested_structured_output = chat.structured_output.clone();
    let chat_request = to_chat_request(chat)?;

    let RenderedChat {
        prompt_token_ids,
        structured_output,
        engine_reasoning_gate,
        media,
    } = prepare_and_render_chat(
        state,
        chat_request,
        requested_structured_output,
        omit_media_kwargs,
    )
    .await?;

    Ok(pb::RenderChatResponse {
        prompt_token_ids,
        structured_output,
        engine_reasoning_gate,
        media,
    })
}

struct RenderedChat {
    prompt_token_ids: Vec<u32>,
    structured_output: Option<pb::StructuredOutput>,
    engine_reasoning_gate: Option<pb::EngineReasoningGate>,
    media: Vec<pb::MediaItem>,
}

async fn prepare_and_render_chat(
    state: &impl RenderContext,
    chat_request: ChatRequest,
    requested_structured_output: Option<pb::StructuredOutput>,
    omit_media_kwargs: bool,
) -> Result<RenderedChat, Status> {
    let structured_outputs = chat_request.sampling_params.structured_outputs.clone();

    let text_request = state
        .chat_processor()
        .prepare(chat_request, NonZeroU32::MIN, state.text_processor())
        .await
        .map_err(chat_error_to_status)?
        .text_request;
    let prepared = state.text_processor().prepare(text_request).map_err(text_error_to_status)?;
    if prepared.generate_request.prompt_token_ids.is_empty() {
        return Err(Status::invalid_argument(
            "token_ids must contain at least one token ID.",
        ));
    }
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
        Some(params) => requested_structured_output
            .or_else(|| Some(convert::structured_output_to_proto(&params))),
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

    Ok(RenderedChat {
        prompt_token_ids: prepared.generate_request.prompt_token_ids,
        structured_output,
        engine_reasoning_gate,
        media,
    })
}

// ========================================================================================
// Derender
// ========================================================================================

/// Start a DerenderStream from its first message. The returned response stream
/// owns `inbound`.
async fn derender_stream(
    state: &impl RenderContext,
    mut inbound: Streaming<pb::DerenderStreamRequest>,
    headers: &HeaderMap,
) -> Result<
    Pin<Box<dyn Stream<Item = Result<pb::DerenderStreamResponse, Status>> + Send + 'static>>,
    Status,
> {
    use pb::derender_stream_request::Request as StreamRequest;

    let Some(pb::DerenderStreamRequest {
        request: Some(StreamRequest::Start(start)),
    }) = inbound.message().await?
    else {
        return Err(Status::invalid_argument(
            "the first DerenderStream message must be `start`",
        ));
    };
    let pb::DerenderStart {
        context,
        prompt_token_ids,
    } = *start;
    let Some(pb::derender_start::Context::Request(request)) = context else {
        return Err(Status::invalid_argument("`start.request` is required"));
    };
    check_chat_api(request.api)?;
    let RawChat { prepared, stream } = prepare_raw_chat(state, &request.body, headers).await?;
    // `UNIMPLEMENTED` lets a gateway fall back to HTTP for these.
    let unsupported = [
        (prepared.n.get() > 1, "`n` greater than 1"),
        (prepared.options.include_prompt_logprobs, "prompt logprobs"),
        (
            prepared.chat_request.return_prompt_text,
            "`return_prompt_text`",
        ),
        (
            prepared.options.return_weight_versions,
            "`return_weight_versions`",
        ),
    ];
    if let Some((_, feature)) = unsupported.iter().find(|(requested, _)| *requested) {
        return Err(Status::unimplemented(format!(
            "DerenderStream does not support {feature}"
        )));
    }

    let context = ChatOutputContext {
        request_id: prepared.request_id.clone(),
        response_model: prepared.response_model.clone(),
        options: prepared.options.clone(),
        created: unix_timestamp(),
        system_fingerprint: state.system_fingerprint(),
        api_server_options: state.api_server_options(),
        count_reasoning_tokens: state.counts_reasoning_tokens(),
    };
    let (end_tx, end_rx) = oneshot::channel();
    let inbound = DerenderInbound {
        messages: inbound,
        stop_strings: prepared.chat_request.decode_options.stop_strings.clone().unwrap_or_default(),
        end_tx: Some(end_tx),
    };
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

    let (done_tx, done_rx) = oneshot::channel();
    let mut done_tx = Some(done_tx);
    let mut last_token_id = None;
    let events = state
        .chat_processor()
        .detokenize_stream(
            prepared.chat_request.clone(),
            state.text_processor(),
            prompt_token_ids,
            chunks,
        )
        .map_err(chat_error_to_status)?
        .inspect(move |event| match event {
            Ok(ChatEvent::LogprobsDelta { token_ids, .. }) => {
                last_token_id = token_ids.last().copied().or(last_token_id);
            }
            Ok(ChatEvent::Done {
                usage,
                finish_reason,
                ..
            }) => {
                if let Some(tx) = done_tx.take() {
                    let _ = tx.send(DecodedFinish {
                        finish_reason: finish_reason.clone(),
                        num_output_tokens: usage.output_token_count as u32,
                        last_token_id,
                    });
                }
            }
            _ => {}
        });
    let mut end = DerenderEnd::new(end_rx, done_rx);

    if stream {
        let chunks = chat_completion_chunk_json(events, context);
        Ok(derender_sse_responses(Box::pin(chunks), end).boxed())
    } else {
        Ok(stream::once(async move {
            let body = collect_chat_completion_json(events, context)
                .await
                .map_err(|error| end.input_error().unwrap_or_else(|| api_error_to_status(error)))?;
            Ok(pb::DerenderStreamResponse {
                output: Some(pb::derender_stream_response::Output::Body(body)),
                finished: end.finished()?,
            })
        })
        .boxed())
    }
}

/// How the decoded output finished.
struct DecodedFinish {
    finish_reason: FinishReason,
    num_output_tokens: u32,
    last_token_id: Option<u32>,
}

/// The request stream of a DerenderStream after its start message.
struct DerenderInbound {
    messages: Streaming<pb::DerenderStreamRequest>,
    /// The `stop` of the rendered request.
    stop_strings: Vec<String>,
    /// Reports how the chunks ended to the response stream: the finish info of
    /// GenerateStream, or the input error that ended them early in place of the
    /// resulting unfinished-output error.
    end_tx: Option<oneshot::Sender<Result<pb::FinishInfo, Status>>>,
}

impl DerenderInbound {
    /// Read the next output chunk. A stop string reported in the final chunk
    /// must be one of the request's, as text before it may have been sent
    /// already. A finished chunk reports its finish info through `end_tx`.
    async fn next_chunk(&mut self) -> Result<DetokenizeChunk, Status> {
        use pb::derender_stream_request::Request as StreamRequest;

        // The first GenerateStream response carries only `prompt_info`.
        let output = loop {
            match self.messages.message().await? {
                Some(pb::DerenderStreamRequest {
                    request: Some(StreamRequest::Output(response)),
                }) => {
                    if let Some(output) = response.outputs {
                        break output;
                    }
                }
                Some(_) => {
                    return Err(Status::invalid_argument(
                        "only the first DerenderStream message may be `start`",
                    ));
                }
                None => {
                    return Err(Status::internal(
                        "the DerenderStream request stream closed before a finished output",
                    ));
                }
            }
        };
        if output.index != 0 {
            return Err(Status::internal(format!(
                "output index {} is out of range for a single sequence",
                output.index
            )));
        }
        if let Some(stop) = reported_stop_string(&output)
            && !self.stop_strings.contains(stop)
        {
            return Err(Status::internal(format!(
                "output stopped on stop string {stop:?}, which is not in the request's `stop`"
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

/// What the response side learns about how the output ended.
struct DerenderEnd {
    end_rx: oneshot::Receiver<Result<pb::FinishInfo, Status>>,
    /// The value received from `end_rx`, kept until it is used.
    end: Option<Result<pb::FinishInfo, Status>>,
    done_rx: oneshot::Receiver<DecodedFinish>,
}

impl DerenderEnd {
    fn new(
        end_rx: oneshot::Receiver<Result<pb::FinishInfo, Status>>,
        done_rx: oneshot::Receiver<DecodedFinish>,
    ) -> Self {
        Self {
            end_rx,
            end: None,
            done_rx,
        }
    }

    fn poll_end(&mut self) {
        if self.end.is_none() {
            self.end = self.end_rx.try_recv().ok();
        }
    }

    /// The input error that ended the chunks, if any. Leaves a reported finish
    /// info for [`Self::finished`].
    fn input_error(&mut self) -> Option<Status> {
        self.poll_end();
        match self.end.take()? {
            Err(status) => Some(status),
            reported => {
                self.end = Some(reported);
                None
            }
        }
    }

    /// The finished sequence: GenerateStream's finish info with the decoded
    /// output's finish reason, stop reason and token count. Empty when the
    /// output did not finish.
    fn finished(&mut self) -> Result<Vec<pb::SequenceFinish>, Status> {
        let Ok(decoded) = self.done_rx.try_recv() else {
            return Ok(Vec::new());
        };
        // Unset when the output ended before a finished chunk, such as at a
        // stop string.
        self.poll_end();
        let reported = self.end.take().and_then(Result::ok);
        Ok(vec![pb::SequenceFinish {
            index: 0,
            finish_info: Some(detokenized_finish_info(reported, decoded)?),
        }])
    }
}

/// Coalesce the chunks available at once into one response of SSE events. The
/// last response ends the stream with `data: [DONE]` and the finished sequence.
/// A failure after events went out is sent in-band, as the HTTP route does;
/// errors of the request stream end the RPC with a status.
#[try_stream]
async fn derender_sse_responses(
    chunks: Pin<Box<dyn Stream<Item = Result<Vec<u8>, ApiError>> + Send>>,
    mut end: DerenderEnd,
    mut y: TryYielder<pb::DerenderStreamResponse, Status>,
) -> Result<(), Status> {
    // The cap only bounds the size of one response.
    let mut chunks = chunks.ready_chunks(64);
    while let Some(batch) = chunks.next().await {
        let mut events = Vec::new();
        for chunk in batch {
            match chunk {
                Ok(json) => events.extend(sse_data_frame(&json)),
                Err(error) => {
                    if let Some(status) = end.input_error() {
                        return Err(status);
                    }
                    let response = serde_json::to_vec(&error.to_error_response())
                        .map_err(|error| Status::internal(error.to_report_string()))?;
                    events.extend(sse_data_frame(&response));
                    events.extend(SSE_DONE_FRAME);
                    y.yield_ok(events_response(events, Vec::new())).await;
                    return Ok(());
                }
            }
        }
        if !events.is_empty() {
            y.yield_ok(events_response(events, Vec::new())).await;
        }
    }
    if let Some(status) = end.input_error() {
        return Err(status);
    }
    y.yield_ok(events_response(SSE_DONE_FRAME.to_vec(), end.finished()?)).await;
    Ok(())
}

fn events_response(
    events: Vec<u8>,
    finished: Vec<pb::SequenceFinish>,
) -> pb::DerenderStreamResponse {
    pb::DerenderStreamResponse {
        output: Some(pb::derender_stream_response::Output::Events(events)),
        finished,
    }
}

/// GenerateStream's finish info with the decoded output's finish reason, stop
/// reason and token count.
fn detokenized_finish_info(
    reported: Option<pb::FinishInfo>,
    decoded: DecodedFinish,
) -> Result<pb::FinishInfo, Status> {
    use pb::finish_info::{FinishReason as PbFinishReason, StopReason as PbStopReason};

    let reported = reported.unwrap_or_default();
    let (finish_reason, stop_reason) = match (&decoded.finish_reason, &reported.stop_reason) {
        (FinishReason::Stop(None), Some(stop @ PbStopReason::EosTokenId(_))) => {
            (PbFinishReason::Stop as i32, Some(stop.clone()))
        }
        (FinishReason::Length, Some(PbStopReason::StopString(stop))) => {
            return Err(Status::internal(format!(
                "output stopped on stop string {stop:?}, which the decoded output does not \
                 contain past `min_tokens`"
            )));
        }
        (
            FinishReason::Stop(Some(StopReason::Text(matched))),
            Some(PbStopReason::StopString(stop)),
        ) if matched != stop => {
            return Err(Status::internal(format!(
                "output stopped on stop string {stop:?}, but the decoded output stops on \
                 {matched:?} first"
            )));
        }
        (finish_reason, _) => {
            convert::finish_reason_to_proto(finish_reason, decoded.last_token_id.as_slice())?
        }
    };
    Ok(pb::FinishInfo {
        num_output_tokens: decoded.num_output_tokens,
        finish_reason,
        stop_reason,
        ..reported
    })
}

fn chunk_from_proto(output: pb::SequenceOutput) -> Result<DetokenizeChunk, Status> {
    let finish_info = output.finish_info.as_ref();
    Ok(DetokenizeChunk {
        logprobs: convert::output_logprobs_from_proto(&output)?,
        finish_reason: finish_info.and_then(finish_reason_from_proto),
        num_cached_tokens: finish_info.and_then(|info| info.num_cached_tokens).unwrap_or_default()
            as usize,
        kv_transfer_params: finish_info
            .and_then(|info| info.kv_transfer_params.as_ref())
            .map(|params| Box::new(convert::proto_struct_to_json(params))),
        ec_transfer_params: finish_info
            .and_then(|info| info.ec_transfer_params.as_ref())
            .map(|params| Box::new(convert::proto_struct_to_json(params))),
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

/// Map GenerateStream finish info to the finish reason that drives decoding, or
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
    let resolution = state.resolve_model(Some(model)).await;
    if resolution.model_names.accepts(Some(model)) {
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
        // Rendering does not decode.
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
            // gRPC render has always matched the name case-insensitively; only
            // the HTTP routes follow OpenAI's lowercase schema.
            format: audio
                .format
                .map(|format| {
                    format.parse().map_err(|_| {
                        Status::invalid_argument(format!(
                            "unsupported input_audio format `{format}`; expected `wav` or `mp3`"
                        ))
                    })
                })
                .transpose()?,
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
            // A JSON schema object or `null`, as the HTTP tool DTO accepts.
            let parameters = if tool.parameters_json.is_empty() {
                None
            } else {
                serde_json::from_str::<Option<serde_json::Map<_, _>>>(&tool.parameters_json)
                    .map_err(|error| {
                        Status::invalid_argument(format!(
                            "invalid parameters_json for tool `{}`: {error}",
                            tool.name
                        ))
                    })?
            };
            Ok(ChatTool {
                name: tool.name,
                description: tool.description,
                parameters: parameters.map_or(serde_json::Value::Null, serde_json::Value::Object),
                strict: tool.strict,
                defer_loading: tool.defer_loading,
            })
        })
        .collect()
}

// ========================================================================================
// Errors
// ========================================================================================

fn chat_error_to_status(error: vllm_chat::Error) -> Status {
    let message = error.to_report_string();
    if error.kind().is_invalid_request() {
        Status::invalid_argument(message)
    } else if error.is_unavailable() {
        Status::unavailable(message)
    } else {
        Status::internal(message)
    }
}

fn api_error_to_status(error: ApiError) -> Status {
    let response = error.to_error_response();
    let message = response.error.message;
    match error {
        ApiError::InvalidRequest { .. }
        | ApiError::JsonParseError { .. }
        | ApiError::InvalidQuery { .. } => Status::invalid_argument(message),
        ApiError::ModelNotFound { .. } | ApiError::NotFound { .. } => Status::not_found(message),
        ApiError::Conflict { .. } => Status::failed_precondition(message),
        ApiError::ServiceUnavailable { .. } => Status::unavailable(message),
        ApiError::ServerError { .. } => Status::internal(message),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use llm_multimodal::MediaConnectorError;
    use vllm_chat::InputAudioFormat;

    use super::*;

    /// Render answers a chat error like the chat routes: media the request
    /// got wrong is `INVALID_ARGUMENT`, a media host that timed out is this
    /// side's trouble.
    #[test]
    fn chat_errors_map_to_status_like_the_chat_routes() {
        let errors = [
            vllm_chat::Error::from(MediaConnectorError::DataUrl("missing comma".to_string())),
            vllm_chat::Error::from(MediaConnectorError::Timeout {
                url: "https://example.com/a.png".to_string(),
                budget: Duration::from_secs(5),
            }),
        ];
        let codes = errors
            .into_iter()
            .map(|error| chat_error_to_status(error).code())
            .collect::<Vec<_>>();
        assert_eq!(codes, [tonic::Code::InvalidArgument, tonic::Code::Internal]);
    }

    #[test]
    fn input_audio_format_is_optional_but_must_name_wav_or_mp3() {
        let part = |format: Option<&str>| pb::ChatContentPart {
            part: Some(pb::chat_content_part::Part::InputAudio(
                pb::ChatInputAudio {
                    data: "dGVzdA==".to_string(),
                    format: format.map(str::to_string),
                    uuid: None,
                },
            )),
        };
        for (format, expected) in [
            (None, None),
            (Some("wav"), Some(InputAudioFormat::Wav)),
            (Some("MP3"), Some(InputAudioFormat::Mp3)),
        ] {
            let ChatContentPart::InputAudio { format, .. } =
                part_from_proto(part(format)).expect("valid input_audio part")
            else {
                panic!("expected an input_audio part");
            };
            assert_eq!(format, expected);
        }
        for format in ["flac", ""] {
            assert_eq!(
                part_from_proto(part(Some(format))).unwrap_err().code(),
                tonic::Code::InvalidArgument
            );
        }
    }
}
