// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! gRPC `Render` service: engine-free chat rendering, the gRPC
//! counterpart of the HTTP render API.

use std::num::NonZeroU32;
use std::sync::Arc;

use itertools::Itertools as _;
use llm_multimodal::ImageDetail;
use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};
use tracing::{info_span, warn};
use tracing_futures::Instrument as _;
use uuid::Uuid;
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatMessage,
    ChatOptions, ChatRequest, ChatRequestProcessor, ChatTool, ChatToolChoice, GenerationPromptMode,
    ResolvedToolContext, SamplingParams,
};
use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
use vllm_text::{
    PromptTruncation, PromptTruncationLimit, TextDecodeOptions, TextRequestProcessor,
    TruncationSide,
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
    let media = (prepared.generate_request.mm_features.into_iter().flatten())
        .map(|feature| media::feature_to_proto(feature, !omit_media_kwargs))
        .try_collect()?;

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
    let generation_prompt_mode = match chat.generation_prompt_mode() {
        pb::GenerationPromptMode::Unspecified | pb::GenerationPromptMode::StartNewAssistant => {
            GenerationPromptMode::StartNewAssistant
        }
        pb::GenerationPromptMode::ContinueFinalAssistant => {
            GenerationPromptMode::ContinueFinalAssistant
        }
        pb::GenerationPromptMode::NoGenerationPrompt => GenerationPromptMode::NoGenerationPrompt,
    };
    let messages: Vec<_> = chat.messages.into_iter().map(message_from_proto).try_collect()?;
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
                .map(|kwargs| match convert::proto_struct_to_json(&kwargs) {
                    serde_json::Value::Object(map) => map.into_iter().collect(),
                    _ => unreachable!("a proto struct converts to a JSON object"),
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

    let side = match truncation.side() {
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
            arguments: call.arguments,
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
    use pb::chat_content_part::Part;

    Ok(match content.and_then(|content| content.content) {
        None => ChatContent::Text(String::new()),
        Some(Content::Text(text)) => ChatContent::Text(text),
        Some(Content::Parts(parts)) => ChatContent::Parts(
            parts
                .parts
                .into_iter()
                .filter_map(|part| part.part)
                .map(|part| match part {
                    Part::Text(text) => ChatContentPart::text(text),
                    Part::ImageUrl(image) => ChatContentPart::ImageUrl {
                        detail: match image.detail() {
                            pb::ImageDetail::Unspecified => None,
                            pb::ImageDetail::Auto => Some(ImageDetail::Auto),
                            pb::ImageDetail::Low => Some(ImageDetail::Low),
                            pb::ImageDetail::High => Some(ImageDetail::High),
                        },
                        image_url: image.url,
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
                .collect(),
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
