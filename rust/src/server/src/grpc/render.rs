// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC `Render` service: engine-free chat rendering, the gRPC
//! counterpart of the HTTP render API.

use std::num::NonZeroU32;
use std::sync::Arc;

use axum::Json;
use axum::http::HeaderMap;
use itertools::Itertools as _;
use llm_multimodal::ImageDetail;
use tonic::{Request, Response, Status};
use tracing::{info_span, warn};
use tracing_futures::Instrument as _;
use uuid::Uuid;
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatMessage,
    ChatOptions, ChatRequest, ChatTool, ChatToolChoice, GenerationPromptMode, ResolvedToolContext,
    SamplingParams,
};
use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
use vllm_text::{PromptTruncation, PromptTruncationLimit, TextDecodeOptions, TruncationSide};

use super::convert::{self, json_to_proto_struct};
use super::inference::{chat_error_to_status, text_error_to_status};
use super::media;
use super::pb;
use super::pb::render_server::{Render, RenderServer};
use crate::error::ApiError;
use crate::render::RenderContext;
use crate::routes::openai::ChatCompletionRequest;
use crate::routes::render::{LoweredChatRender, lower_chat_render_request};

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
        let request = request.into_inner();
        let span = info_span!("grpc_render", rpc = "Render");
        let result = render(self.state.as_ref(), request).instrument(span.clone()).await;
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

async fn render(
    state: &impl RenderContext,
    request: pb::RenderRequest,
) -> Result<pb::RenderResponse, Status> {
    let api = enum_from_proto::<pb::Api>(request.api, "api")?;
    match api {
        pb::Api::Chat => {}
        pb::Api::Unspecified => return Err(Status::invalid_argument("api is required")),
        other => {
            return Err(Status::unimplemented(format!(
                "api `{}` is not supported",
                other.as_str_name()
            )));
        }
    }

    let omit_media_kwargs =
        match enum_from_proto::<pb::MmPayload>(request.mm_payload, "mm_payload")? {
            pb::MmPayload::Unspecified | pb::MmPayload::Tensors => false,
            pb::MmPayload::None => true,
        };

    // Parse exactly as the HTTP extractor does, so a body gets the same result
    // and error on both protocols.
    let Json(body) = Json::<ChatCompletionRequest>::from_bytes(&request.body)
        .map_err(|rejection| api_error_to_status(rejection.into()))?;

    let LoweredChatRender { chat_request, .. } =
        lower_chat_render_request(state, body, &HeaderMap::new())
            .await
            .map_err(api_error_to_status)?;

    let RenderedChat {
        prompt_token_ids,
        structured_output,
        engine_reasoning_gate,
        media,
        render_fingerprint,
    } = prepare_and_render_chat(state, chat_request, None, omit_media_kwargs).await?;

    Ok(pb::RenderResponse {
        prompt_token_ids,
        structured_output,
        engine_reasoning_gate,
        media,
        render_fingerprint,
    })
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
        render_fingerprint,
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
        render_fingerprint,
    })
}

struct RenderedChat {
    prompt_token_ids: Vec<u32>,
    structured_output: Option<pb::StructuredOutput>,
    engine_reasoning_gate: Option<pb::EngineReasoningGate>,
    media: Vec<pb::MediaItem>,
    render_fingerprint: String,
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
        render_fingerprint: state.render_fingerprint().unwrap_or_default().to_string(),
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
