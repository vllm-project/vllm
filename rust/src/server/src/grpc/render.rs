// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC `Render` service: engine-free chat tokenize and detokenize, the gRPC
//! counterpart of the HTTP render API.

use std::sync::Arc;

use itertools::Itertools as _;
use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};
use tracing::{info_span, warn};
use tracing_futures::Instrument as _;
use uuid::Uuid;
use vllm_chat::{
    AssistantContentBlock, AssistantMessageExt as _, AssistantToolCall, ChatContent,
    ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRequestProcessor, ChatTool,
    ChatToolChoice, ResolvedToolContext, SamplingParams,
};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputsParams;
use vllm_text::{
    FinishReason, PromptTruncation, PromptTruncationLimit, TextDecodeOptions, TextRequestProcessor,
    TruncationSide,
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
    request: pb::DetokenizeRequest,
) -> Result<pb::DetokenizeResponse, Status> {
    let pb::DetokenizeRequest {
        chat,
        prompt_token_ids,
        output,
    } = request;
    let output = output.unwrap_or_default();
    let logprobs = convert::output_logprobs_from_proto(&output)?;
    let chat = chat.unwrap_or_default();
    check_model(state, &chat.model).await?;
    let chat_request = to_chat_request(chat)?;
    let collected = state
        .chat_processor()
        .detokenize(
            chat_request,
            state.text_processor(),
            prompt_token_ids,
            output.token_ids,
            logprobs,
            finish_reason_from_proto(output.finish_info.as_ref()),
        )
        .await
        .map_err(chat_error_to_status)?;

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
        logprobs: collected
            .logprobs
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
            .collect(),
    })
}

/// Map the final Generate finish info to the finish reason that drives
/// decoding.
///
/// Only token stops end the output with a token that is hidden from the text.
fn finish_reason_from_proto(finish_info: Option<&pb::FinishInfo>) -> FinishReason {
    use pb::finish_info::{FinishReason as PbFinishReason, StopReason as PbStopReason};

    let Some(info) = finish_info else {
        return FinishReason::Length;
    };
    match info.finish_reason() {
        PbFinishReason::Stop => match &info.stop_reason {
            Some(PbStopReason::StopTokenId(id)) => {
                FinishReason::Stop(Some(StopReason::TokenId(*id)))
            }
            Some(PbStopReason::EosTokenId(_)) | None => FinishReason::stop_eos(),
            Some(PbStopReason::StopString(_)) => FinishReason::Length,
        },
        PbFinishReason::Aborted => FinishReason::Abort,
        PbFinishReason::Length | PbFinishReason::NotFinished => FinishReason::Length,
    }
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
