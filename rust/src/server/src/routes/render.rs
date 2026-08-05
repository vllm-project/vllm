// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use thiserror_ext::AsReport as _;
use vllm_chat::NewChatOutputProcessorOptions;
use vllm_llm::GenerateRequest;
use vllm_text::TextRequest;

use crate::error::{ApiError, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::render::RenderState;
use crate::routes::DEFAULT_JSON_BODY_LIMIT_BYTES;
use crate::routes::openai::utils::types::{ListModelsResponse, ModelObject};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::openai::{
    ChatCompletionRequest, CompletionRequest, lower_chat_request, lower_completion_request,
};
use crate::utils::{resolve_request_context, unix_timestamp};

pub(crate) fn build_router(state: Arc<RenderState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ping", get(health).post(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions/render", post(render_chat))
        .route("/v1/completions/render", post(render_completion))
        .with_state(state)
        .layer(DefaultBodyLimit::max(DEFAULT_JSON_BODY_LIMIT_BYTES))
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
                max_model_len: Some(state.text.max_model_len()),
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

fn lower_render_request(
    state: &RenderState,
    text_request: TextRequest,
) -> Result<GenerateRequest, ApiError> {
    state
        .text
        .prepare(text_request)
        .map(|prepared| prepared.generate_request)
        .map_err(|error| text_submit_error("failed to prepare render request", error))
}

async fn render_chat(
    State(state): State<Arc<RenderState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Result<Json<GenerateRequest>, ApiError> {
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let chat_request = lower_chat_request(body, &model_resolution(&state), request_context)?;
    let (text_request, _) = state
        .chat
        .prepare(
            chat_request,
            NewChatOutputProcessorOptions {
                tool_call_parser: &state.tool_call_parser,
                reasoning_parser: &state.reasoning_parser,
            },
        )
        .await
        .map_err(|error| ApiError::invalid_request(error.to_report_string(), None))?;
    Ok(Json(lower_render_request(&state, text_request)?))
}

async fn render_completion(
    State(state): State<Arc<RenderState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Result<Json<Vec<GenerateRequest>>, ApiError> {
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let tokenizer = state.text.tokenizer();
    let text_request = lower_completion_request(
        body,
        &model_resolution(&state),
        request_context,
        tokenizer.as_ref(),
    )?;
    Ok(Json(vec![lower_render_request(&state, text_request)?]))
}
