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
use vllm_chat::{
    AssistantContentBlock, AssistantMessageExt as _, AssistantToolCall, ChatContent,
    ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRequestProcessor, ChatTool,
    ChatToolChoice, ResolvedToolContext,
};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
use vllm_engine_core_client::protocol::structured_outputs::{
    StructuredOutputConstraint, StructuredOutputsParams,
};
use vllm_text::{FinishReason, TextRequestProcessor};

use super::convert::{self, json_to_proto_struct};
use super::inference::text_error_to_status;
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
    fn served_model_names(&self) -> &[String];
}

impl RenderContext for AppState {
    fn chat_processor(&self) -> &ChatRequestProcessor {
        self.chat.request_processor()
    }

    fn text_processor(&self) -> &TextRequestProcessor {
        self.chat.text().request_processor()
    }

    fn served_model_names(&self) -> &[String] {
        AppState::served_model_names(self)
    }
}

impl RenderContext for RenderState {
    fn chat_processor(&self) -> &ChatRequestProcessor {
        &self.chat
    }

    fn text_processor(&self) -> &TextRequestProcessor {
        &self.text
    }

    fn served_model_names(&self) -> &[String] {
        &self.served_model_names
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
        let pb::TokenizeRequest { chat, generate } = request.into_inner();
        let generate = generate.unwrap_or_default();
        let span = info_span!("grpc_render", rpc = "Tokenize", request_id = %generate.request_id);
        let result = tokenize(self.state.as_ref(), chat.unwrap_or_default(), generate)
            .instrument(span.clone())
            .await;
        log_failure(&span, &result);
        result.map(|generate_request| {
            Response::new(pb::TokenizeResponse {
                generate_request: Some(generate_request),
            })
        })
    }

    async fn detokenize(
        &self,
        request: Request<pb::DetokenizeRequest>,
    ) -> Result<Response<pb::DetokenizeResponse>, Status> {
        let request = request.into_inner();
        let span = info_span!(
            "grpc_render",
            rpc = "Detokenize",
            request_id = request.generate_request.as_ref().map_or("", |r| &r.request_id),
        );
        let result = detokenize(self.state.as_ref(), request).instrument(span.clone()).await;
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
    mut generate: pb::GenerateRequest,
) -> Result<pb::GenerateRequest, Status> {
    if generate.prompt.is_some() {
        return Err(Status::invalid_argument(
            "generate.prompt must be unset; Tokenize renders it from the chat request",
        ));
    }
    // TODO: support multimodal inputs.
    if !generate.media.is_empty() {
        return Err(Status::invalid_argument(
            "generate.media is not supported by Tokenize",
        ));
    }
    // TODO: support stop strings, which need text from Generate.
    if generate
        .stopping
        .as_ref()
        .is_some_and(|stopping| !stopping.stop_strings.is_empty())
    {
        return Err(Status::invalid_argument(
            "generate.stopping.stop_strings is not supported by Tokenize",
        ));
    }
    // Generate returns token IDs only; Detokenize produces the text.
    let response = generate.response.get_or_insert_default();
    response.output_token_ids = true;
    response.output_text = Some(false);

    let chat_request = to_chat_request(state, chat, &generate)?;
    let structured_outputs = chat_request.sampling_params.structured_outputs.clone();
    let skip_special_tokens = chat_request.decode_options.skip_special_tokens;

    let (text_request, _) = state
        .chat_processor()
        .prepare(chat_request, state.text_processor())
        .await
        .map_err(chat_error_to_status)?;
    let prepared = state.text_processor().prepare(text_request).map_err(text_error_to_status)?;
    let text_request = prepared.text_request;

    // Patch what rendering decided on top of the caller's generation parameters.
    generate.request_id = text_request.request_id;
    generate.prompt = Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
        ids: prepared.generate_request.prompt_token_ids,
    }));
    // The returned prompt is already truncated.
    generate.truncate_prompt_tokens = 0;
    if text_request.sampling_params.structured_outputs != structured_outputs {
        generate.decoding.get_or_insert_default().structured_output = text_request
            .sampling_params
            .structured_outputs
            .as_ref()
            .map(structured_output_to_proto);
    }
    if text_request.decode_options.skip_special_tokens != skip_special_tokens {
        generate.response.get_or_insert_default().skip_special_tokens =
            Some(text_request.decode_options.skip_special_tokens);
    }
    let ReasoningParserKwargs {
        chat_template_kwargs,
    } = text_request.reasoning_parser_kwargs;
    generate.engine_reasoning_gate = (text_request.reasoning_ended.is_some()
        || !chat_template_kwargs.is_empty())
    .then(|| pb::EngineReasoningGate {
        reasoning_ended: text_request.reasoning_ended,
        chat_template_kwargs: json_to_proto_struct(&serde_json::Value::Object(
            chat_template_kwargs.into_iter().collect(),
        )),
    });
    Ok(generate)
}

async fn detokenize(
    state: &impl RenderContext,
    request: pb::DetokenizeRequest,
) -> Result<pb::DetokenizeResponse, Status> {
    let pb::DetokenizeRequest {
        chat,
        generate_request,
        token_ids,
        finish_info,
    } = request;
    let generate_request =
        generate_request.ok_or_else(|| Status::invalid_argument("generate_request is required"))?;
    let Some(pb::generate_request::Prompt::TokenIds(prompt)) = &generate_request.prompt else {
        return Err(Status::invalid_argument(
            "generate_request.prompt must be the token_ids returned by Tokenize",
        ));
    };
    let prompt_token_ids = prompt.ids.clone();
    let finish_reason = finish_reason_from_proto(finish_info.as_ref());

    let mut chat_request = to_chat_request(state, chat.unwrap_or_default(), &generate_request)?;
    // Tokenize turns off text output for Generate, not for Detokenize.
    chat_request.decode_options.detokenize = true;
    let collected = state
        .chat_processor()
        .detokenize(
            chat_request,
            state.text_processor(),
            prompt_token_ids,
            token_ids,
            finish_reason,
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
    })
}

/// Map the final Generate finish info to the finish reason that drives
/// decoding.
///
/// Only token stops end the output with a token that is hidden from the text.
/// Stop strings are matched again while decoding, from the stop strings of the
/// generate request.
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

/// Build the chat request from its proto form plus the generation parameters
/// of `generate`, whose prompt is ignored.
fn to_chat_request(
    state: &impl RenderContext,
    chat: pb::ChatRequest,
    generate: &pb::GenerateRequest,
) -> Result<ChatRequest, Status> {
    // Reuse the Generate conversion for the generation parameters only.
    let text_request = convert::to_text_request(
        pb::GenerateRequest {
            prompt: Some(pb::generate_request::Prompt::TokenIds(Default::default())),
            ..generate.clone()
        },
        /* stream */ false,
        state.served_model_names(),
    )?;

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
        request_id: text_request.request_id,
        messages,
        sampling_params: text_request.sampling_params,
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
        decode_options: text_request.decode_options,
        intermediate: false,
        prompt_truncation: text_request.prompt_truncation,
        priority: text_request.priority,
        documents: None,
        cache_salt: text_request.cache_salt,
        add_special_tokens: chat.add_special_tokens,
        data_parallel_rank: None,
        session_id: text_request.session_id,
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
    let content = content_from_proto(content);
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
                    blocks.extend(parts.into_iter().map(|part| match part {
                        ChatContentPart::Text { text } => AssistantContentBlock::Text { text },
                        _ => unreachable!("proto content parts are text only"),
                    }))
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

fn content_from_proto(content: Option<pb::chat_message::Content>) -> ChatContent {
    use pb::chat_content_part::Part;
    use pb::chat_message::Content;

    match content {
        None => ChatContent::Text(String::new()),
        Some(Content::Text(text)) => ChatContent::Text(text),
        Some(Content::Parts(parts)) => ChatContent::Parts(
            parts
                .parts
                .into_iter()
                .filter_map(|part| match part.part? {
                    Part::Text(text) => Some(ChatContentPart::text(text)),
                })
                .collect(),
        ),
    }
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

fn structured_output_to_proto(
    params: &StructuredOutputsParams,
) -> pb::decoding_parameters::StructuredOutput {
    use pb::decoding_parameters::{StringChoices, StructuredOutput};

    match &params.constraint {
        StructuredOutputConstraint::Json(serde_json::Value::String(schema)) => {
            StructuredOutput::Json(schema.clone())
        }
        StructuredOutputConstraint::Json(schema) => StructuredOutput::Json(schema.to_string()),
        StructuredOutputConstraint::Regex(regex) => StructuredOutput::Regex(regex.clone()),
        StructuredOutputConstraint::Choice(choices) => StructuredOutput::Choice(StringChoices {
            choices: choices.clone(),
        }),
        StructuredOutputConstraint::Grammar(grammar) => StructuredOutput::Grammar(grammar.clone()),
        StructuredOutputConstraint::JsonObject => StructuredOutput::JsonObject(true),
        StructuredOutputConstraint::StructuralTag(tag) => {
            StructuredOutput::StructuralTag(tag.clone())
        }
    }
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
