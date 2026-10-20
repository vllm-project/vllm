// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport as _;
use validator::Validate;
use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
use vllm_text::{TextRequest, TextRequestProcessor};

use crate::DEFAULT_REQUEST_BODY_LIMIT_BYTES;
use crate::error::{ApiError, server_error, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::render::RenderState;
use crate::routes::inference::generate::{
    GenerateRequest, GenerateSamplingParams, validate_request_compat as validate_generate_request,
};
use crate::routes::openai::utils::types::{
    ListModelsResponse, ModelObject, Normalizable, StreamOptions,
};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::openai::{
    ChatCompletionRequest, CompletionRequest, lower_chat_request, lower_completion_request,
};
use crate::utils::{resolve_request_context, unix_timestamp};

#[derive(Deserialize, Validate)]
struct RenderRequest<T: Validate> {
    #[serde(flatten)]
    #[validate(nested)]
    request: T,
    return_token_offsets: Option<bool>,
}

impl<T: Validate + Normalizable> Normalizable for RenderRequest<T> {
    fn normalize(&mut self) {
        self.request.normalize();
    }
}

#[derive(Serialize)]
struct RenderResponse {
    #[serde(flatten)]
    request: GenerateRequest,
    token_offsets: Option<Vec<(usize, usize)>>,
}

async fn render_processor(
    state: &RenderState,
    return_token_offsets: Option<bool>,
) -> Result<TextRequestProcessor, ApiError> {
    let enabled = return_token_offsets.unwrap_or(false);
    if enabled {
        let tokenizer = state.text.tokenizer();
        tokio::task::spawn_blocking(move || tokenizer.warm_offsets())
            .await
            .map_err(|error| {
                server_error!(
                    "offset tokenizer initialization failed: {}",
                    error.as_report()
                )
            })?;
    }
    Ok(state.text.clone().with_token_offsets(enabled))
}

pub(crate) fn build_router(state: Arc<RenderState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ping", get(health).post(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions/render", post(render_chat))
        .route("/v1/completions/render", post(render_completion))
        .with_state(state)
        .layer(DefaultBodyLimit::max(DEFAULT_REQUEST_BODY_LIMIT_BYTES))
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn list_models(State(state): State<Arc<RenderState>>) -> Json<ListModelsResponse> {
    let created = unix_timestamp() as i64;
    Json(ListModelsResponse {
        object: "list".to_string(),
        data: state
            .served_model_names
            .iter()
            .map(|model| ModelObject {
                id: model.clone(),
                object: "model".to_string(),
                created,
                owned_by: "vllm".to_string(),
                root: Some(state.model.clone()),
                parent: None,
                max_model_len: state.max_model_len,
            })
            .collect(),
    })
}

fn model_resolution(state: &RenderState) -> LoraModelResolution {
    LoraModelResolution {
        model_names: state.served_model_names.clone(),
        lora_request: None,
    }
}

fn response_model(state: &RenderState, requested_model: Option<&str>) -> String {
    requested_model
        .filter(|model| !model.is_empty())
        .or_else(|| state.served_model_names.first().map(String::as_str))
        .unwrap_or_default()
        .to_string()
}

fn lower_render_request(
    state: &RenderState,
    processor: &TextRequestProcessor,
    text_request: TextRequest,
    model: String,
    stream: bool,
    stream_options: Option<StreamOptions>,
) -> Result<RenderResponse, ApiError> {
    let prepared = processor
        .prepare(text_request)
        .map_err(|error| text_submit_error("failed to prepare render request", error))?;
    let token_ids = prepared.generate_request.prompt_token_ids;
    let text_request = prepared.text_request;

    let request = GenerateRequest {
        request_id: Some(text_request.request_id),
        model: Some(model),
        token_ids,
        sampling_params: GenerateSamplingParams {
            n: None,
            inner: text_request.sampling_params,
        },
        stream,
        stream_options,
        cache_salt: text_request.cache_salt,
        priority: text_request.priority,
        kv_transfer_params: None,
        ec_transfer_params: None,
        content_parts: None,
        return_token_ids: None,
        // Carry what the chat route would pass to the engine, so the engine
        // gates structured outputs the same way on `/inference/v1/generate`.
        reasoning_ended: text_request.reasoning_ended,
        reasoning_parser_kwargs: (text_request.reasoning_parser_kwargs
            != ReasoningParserKwargs::default())
        .then_some(text_request.reasoning_parser_kwargs),
        other: Default::default(),
    };
    validate_generate_request(&request, &state.served_model_names)?;
    Ok(RenderResponse {
        request,
        token_offsets: text_request.prompt_token_offsets,
    })
}

async fn render_chat(
    State(state): State<Arc<RenderState>>,
    headers: HeaderMap,
    ValidatedJson(render): ValidatedJson<RenderRequest<ChatCompletionRequest>>,
) -> Result<Json<RenderResponse>, ApiError> {
    let processor = render_processor(&state, render.return_token_offsets).await?;
    let body = render.request;
    let model = response_model(&state, body.model.as_deref());
    let stream = body.stream;
    let stream_options = body.stream_options.clone();
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let chat_request = lower_chat_request(body, &model_resolution(&state), request_context)?;
    let (text_request, _) = state
        .chat
        .prepare(chat_request, &processor)
        .await
        .map_err(|error| ApiError::invalid_request(error.to_report_string(), None))?;
    Ok(Json(lower_render_request(
        &state,
        &processor,
        text_request,
        model,
        stream,
        stream_options,
    )?))
}

async fn render_completion(
    State(state): State<Arc<RenderState>>,
    headers: HeaderMap,
    ValidatedJson(render): ValidatedJson<RenderRequest<CompletionRequest>>,
) -> Result<Json<Vec<RenderResponse>>, ApiError> {
    let processor = render_processor(&state, render.return_token_offsets).await?;
    let body = render.request;
    let model = response_model(&state, body.model.as_deref());
    let stream = body.stream;
    let stream_options = body.stream_options.clone();
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let tokenizer = state.text.tokenizer();
    let text_request = lower_completion_request(
        body,
        &model_resolution(&state),
        request_context,
        tokenizer.as_ref(),
    )?;
    Ok(Json(vec![lower_render_request(
        &state,
        &processor,
        text_request,
        model,
        stream,
        stream_options,
    )?]))
}
