// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
use vllm_text::TextRequest;

use crate::DEFAULT_REQUEST_BODY_LIMIT_BYTES;
use crate::error::{ApiError, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::render::RenderState;
use crate::routes::inference::generate::{
    GenerateRequest, GenerateSamplingParams, validate_request_compat as validate_generate_request,
};
use crate::routes::openai::utils::types::{ListModelsResponse, ModelObject, StreamOptions};
use crate::routes::openai::utils::validate_max_output_tokens;
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
    text_request: TextRequest,
    model: String,
    stream: bool,
    stream_options: Option<StreamOptions>,
    return_token_ids: Option<bool>,
) -> Result<GenerateRequest, ApiError> {
    let prepared = state
        .text
        .prepare(text_request)
        .map_err(|error| text_submit_error("failed to prepare render request", error))?;
    let token_ids = prepared.generate_request.prompt_token_ids;
    let mut text_request = prepared.text_request;
    let xargs = &mut text_request.sampling_params.vllm_xargs;
    let kv_transfer_params = hoist_transfer_params(xargs, "kv_transfer_params");
    let ec_transfer_params = hoist_transfer_params(xargs, "ec_transfer_params");

    let request = GenerateRequest {
        request_id: Some(text_request.request_id),
        model: Some(model),
        token_ids,
        sampling_params: GenerateSamplingParams {
            n: None,
            stop: None,
            include_stop_str_in_output: None,
            trace_decode_token_ids: None,
            inner: text_request.sampling_params,
        },
        stream,
        stream_options,
        cache_salt: text_request.cache_salt,
        priority: text_request.priority,
        kv_transfer_params,
        ec_transfer_params,
        content_parts: None,
        return_token_ids,
        output_mode: None,
        // Carry what the chat route would pass to the engine, so the engine
        // gates structured outputs the same way on `/inference/v1/generate`.
        reasoning_ended: text_request.reasoning_ended,
        reasoning_parser_kwargs: (text_request.reasoning_parser_kwargs
            != ReasoningParserKwargs::default())
        .then_some(text_request.reasoning_parser_kwargs),
        other: Default::default(),
    };
    validate_generate_request(&request, &state.served_model_names)?;
    Ok(request)
}

/// Move KV/EC transfer params out of `vllm_xargs`.
///
/// Render emits them only as top-level `GenerateRequest` fields, like Python.
/// Generate merges them back into `vllm_xargs` for the engine.
fn hoist_transfer_params(
    xargs: &mut Option<HashMap<String, Value>>,
    key: &str,
) -> Option<HashMap<String, Value>> {
    let map = xargs.as_mut()?;
    let params = match map.remove(key)? {
        Value::Object(params) => params.into_iter().collect(),
        // Only an object fits the top-level field; keep anything else where
        // the request put it.
        other => {
            map.insert(key.to_owned(), other);
            return None;
        }
    };
    if map.is_empty() {
        *xargs = None;
    }
    Some(params)
}

async fn render_chat(
    State(state): State<Arc<RenderState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Result<Json<GenerateRequest>, ApiError> {
    let model = response_model(&state, body.model.as_deref());
    let stream = body.stream;
    let stream_options = body.stream_options.clone();
    let return_token_ids = body.return_token_ids;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let chat_request = lower_chat_request(body, &model_resolution(&state), request_context)?;
    validate_max_output_tokens(
        "max_completion_tokens",
        chat_request.sampling_params.max_tokens,
        state.text.max_model_len(),
    )?;
    let text_request = state
        .chat
        .prepare(chat_request, &state.text)
        .await
        .map_err(|error| ApiError::invalid_request(error.to_report_string(), None))?
        .text_request;
    Ok(Json(lower_render_request(
        &state,
        text_request,
        model,
        stream,
        stream_options,
        return_token_ids,
    )?))
}

async fn render_completion(
    State(state): State<Arc<RenderState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Result<Json<Vec<GenerateRequest>>, ApiError> {
    let model = response_model(&state, body.model.as_deref());
    let stream = body.stream;
    let stream_options = body.stream_options.clone();
    let return_token_ids = body.return_token_ids;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let tokenizer = state.text.tokenizer();
    let text_request = lower_completion_request(
        body,
        &model_resolution(&state),
        request_context,
        tokenizer.as_ref(),
    )?;
    validate_max_output_tokens(
        "max_tokens",
        text_request.sampling_params.max_tokens,
        state.text.max_model_len(),
    )?;
    Ok(Json(vec![lower_render_request(
        &state,
        text_request,
        model,
        stream,
        stream_options,
        return_token_ids,
    )?]))
}
