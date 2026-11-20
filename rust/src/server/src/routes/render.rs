// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::num::NonZeroU32;
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
use crate::model_names::ServedModelNames;
use crate::render::{RenderContext, RenderState};
use crate::routes::inference::generate::{
    GenerateRequest, GenerateSamplingParams, MmPayload, MultiModalFeatures,
    validate_request_compat as validate_generate_request,
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
        .route(
            "/v1/chat/completions/render",
            post(render_chat::<RenderState>),
        )
        .route(
            "/v1/completions/render",
            post(render_completion::<RenderState>),
        )
        .with_state(state)
        .layer(DefaultBodyLimit::max(DEFAULT_REQUEST_BODY_LIMIT_BYTES))
}

async fn health() -> StatusCode {
    StatusCode::OK
}

/// Base cards carry `max_model_len` and `root` = model path; LoRA cards carry
/// `root` = adapter path and `parent` = base model.
async fn list_models(State(state): State<Arc<RenderState>>) -> Json<ListModelsResponse> {
    let created = unix_timestamp() as i64;
    let base_cards = state.served_model_names.base_names().iter().map(|model| ModelObject {
        id: model.clone(),
        object: "model".to_string(),
        created,
        owned_by: "vllm".to_string(),
        root: Some(state.model.clone()),
        parent: None,
        max_model_len: state.max_model_len,
    });
    let primary = state.served_model_names.primary().to_string();
    let lora_cards = state.lora_modules.iter().map(|module| ModelObject {
        id: module.name.clone(),
        object: "model".to_string(),
        created,
        owned_by: "vllm".to_string(),
        root: Some(module.path.clone()),
        parent: Some(module.base_model_name.clone().unwrap_or_else(|| primary.clone())),
        max_model_len: None,
    });
    Json(ListModelsResponse {
        object: "list".to_string(),
        data: base_cards.chain(lora_cards).collect(),
    })
}

fn lower_render_request(
    state: &impl RenderContext,
    model_names: &ServedModelNames,
    text_request: TextRequest,
    model: String,
    n: Option<u32>,
    stream: bool,
    stream_options: Option<StreamOptions>,
    return_token_ids: Option<bool>,
    mm_payload: MmPayload,
) -> Result<GenerateRequest, ApiError> {
    let prepared = state
        .text_processor()
        .prepare(text_request)
        .map_err(|error| text_submit_error("failed to prepare render request", error))?;
    let token_ids = prepared.generate_request.prompt_token_ids;
    let features = (prepared.generate_request.mm_features)
        .filter(|features| !features.is_empty())
        .map(|features| MultiModalFeatures::from_prepared(features, mm_payload))
        .transpose()?;
    let mut text_request = prepared.text_request;
    let xargs = &mut text_request.sampling_params.vllm_xargs;
    let kv_transfer_params = hoist_transfer_params(xargs, "kv_transfer_params");
    let ec_transfer_params = hoist_transfer_params(xargs, "ec_transfer_params");

    let request = GenerateRequest {
        request_id: Some(text_request.request_id),
        model: Some(model),
        token_ids,
        sampling_params: GenerateSamplingParams {
            n,
            stop: None,
            include_stop_str_in_output: None,
            detokenize: None,
            skip_special_tokens: None,
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
        features,
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
    validate_generate_request(&request, model_names)?;
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

/// `/v1/chat/completions/render` on both the render-only server and, with
/// `--enable-scale-out`, the engine-backed server.
pub(crate) async fn render_chat<S: RenderContext>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Result<Json<GenerateRequest>, ApiError> {
    let requested_model = body.model.as_deref().filter(|model| !model.is_empty());
    let lora_resolution = state.resolve_model(requested_model).await;
    let model = requested_model.unwrap_or(lora_resolution.model_names.primary()).to_string();
    let n = body.n;
    let stream = body.stream;
    let stream_options = body.stream_options.clone();
    let return_token_ids = body.return_token_ids;
    let mm_payload = body.mm_payload.unwrap_or(match body.return_mm_kwargs {
        Some(false) => MmPayload::None,
        _ => MmPayload::Tensors,
    });
    let request_context = resolve_request_context(&headers, body.request_id.as_deref())?;
    let chat_request = lower_chat_request(body, &lora_resolution, request_context)?;
    validate_max_output_tokens(
        "max_completion_tokens",
        chat_request.sampling_params.max_tokens,
        state.text_processor().max_model_len(),
    )?;
    let text_request = state
        .chat_processor()
        // The parallel samples of `n` are the generate request's to fan out;
        // its output grammar is the same for every sample.
        .prepare(chat_request, NonZeroU32::MIN, state.text_processor())
        .await
        .map_err(|error| ApiError::invalid_request(error.to_report_string(), None))?
        .text_request;
    Ok(Json(lower_render_request(
        state.as_ref(),
        &lora_resolution.model_names,
        text_request,
        model,
        n,
        stream,
        stream_options,
        return_token_ids,
        mm_payload,
    )?))
}

/// `/v1/completions/render` on both the render-only server and, with
/// `--enable-scale-out`, the engine-backed server.
pub(crate) async fn render_completion<S: RenderContext>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Result<Json<Vec<GenerateRequest>>, ApiError> {
    let requested_model = body.model.as_deref().filter(|model| !model.is_empty());
    let lora_resolution = state.resolve_model(requested_model).await;
    let model = requested_model.unwrap_or(lora_resolution.model_names.primary()).to_string();
    let n = body.n;
    let stream = body.stream;
    let stream_options = body.stream_options.clone();
    let return_token_ids = body.return_token_ids;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref())?;
    let text_requests = lower_completion_request(
        body,
        &lora_resolution,
        request_context,
        state.text_processor(),
    )?;
    for text_request in &text_requests {
        validate_max_output_tokens(
            "max_tokens",
            text_request.sampling_params.max_tokens,
            state.text_processor().max_model_len(),
        )?;
    }
    // Like Python, one generate request per prompt.
    let generate_requests = text_requests
        .into_iter()
        .map(|text_request| {
            lower_render_request(
                state.as_ref(),
                &lora_resolution.model_names,
                text_request,
                model.clone(),
                n,
                stream,
                stream_options.clone(),
                return_token_ids,
                MmPayload::Tensors,
            )
        })
        .collect::<Result<_, _>>()?;
    Ok(Json(generate_requests))
}
