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
use vllm_chat::ChatRequestProcessor;
use vllm_engine_core_client::protocol::request::ReasoningParserKwargs;
use vllm_text::TextRequest;
use vllm_text::TextRequestProcessor;

use crate::DEFAULT_REQUEST_BODY_LIMIT_BYTES;
use crate::error::{ApiError, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::model_names::ServedModelNames;
use crate::render::RenderState;
use crate::routes::inference::generate::{
    GenerateRequest, GenerateSamplingParams, MmPayload, MultiModalFeatures,
    validate_request_compat as validate_generate_request,
};
use crate::routes::openai::utils::types::{ListModelsResponse, ModelObject, StreamOptions};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::openai::utils::{validate_max_output_tokens, validate_n_limit};
use crate::routes::openai::{
    ChatCompletionRequest, CompletionRequest, lower_chat_request, lower_completion_request,
};
use crate::state::AppState;
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

/// Base cards carry `max_model_len` and `root` = model path; LoRA cards carry
/// `root` = adapter path and `parent` = base model.
async fn list_models(State(state): State<Arc<RenderState>>) -> Json<ListModelsResponse> {
    let created = unix_timestamp() as i64;
    let base_cards = state.served_model_names.names().iter().map(|model| ModelObject {
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

/// Accept LoRA adapter names as the model, like Python. Rendering does not use
/// the adapter, so no LoRA request is resolved; the adapter name is passed
/// through as the generate request's model.
fn model_resolution(state: &RenderState) -> LoraModelResolution {
    LoraModelResolution {
        model_names: state.model_names(),
        lora_request: None,
    }
}

/// Rendering pipeline shared by the standalone render server (`RenderState`)
/// and the main API server's scale-out render routes (`AppState`).
///
/// Both apply the same lowerings; only the state that provides the request
/// processors differs, and validated model names come from each caller's
/// LoRA model resolution.
struct RenderCore<'a> {
    text: &'a TextRequestProcessor,
    chat: &'a ChatRequestProcessor,
}

impl<'a> RenderCore<'a> {
    /// Bind the core to the standalone render server state.
    fn from_render_state(state: &'a RenderState) -> Self {
        Self {
            text: &state.text,
            chat: &state.chat,
        }
    }

    /// Bind the core to the processors the main API server already holds for
    /// chat and completions, so scale-out rendering needs no second model
    /// load. Validated model names come from the request's LoRA resolution,
    /// as on `/inference/v1/generate`.
    fn from_app_state(state: &'a AppState) -> Self {
        Self {
            text: state.chat.text().request_processor(),
            chat: state.chat.request_processor(),
        }
    }

    /// The `model` reported in the rendered request: the requested name when
    /// given, otherwise the primary served name.
    fn response_model(names: &ServedModelNames, requested_model: Option<&str>) -> String {
        requested_model
            .filter(|model| !model.is_empty())
            .unwrap_or(names.primary())
            .to_string()
    }

    /// Prepare a lowered text request and emit the `GenerateRequest` payload
    /// the scale-out render routes return for `/inference/v1/generate`.
    fn lower_render_request(
        &self,
        names: &ServedModelNames,
        text_request: TextRequest,
        model: String,
        n: Option<u32>,
        stream: bool,
        stream_options: Option<StreamOptions>,
        return_token_ids: Option<bool>,
        mm_payload: MmPayload,
    ) -> Result<GenerateRequest, ApiError> {
        let prepared = self
            .text
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
        validate_generate_request(&request, names)?;
        Ok(request)
    }
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

/// Lower one chat request for `/v1/chat/completions/render`, shared by the
/// standalone render server and the main API server's scale-out route.
///
/// `max_model_len` and `max_n_sequences` come from the caller: the standalone
/// server applies its configured renderer limit, while the main API server uses
/// the engine-reported context length and its `max_n_sequences` option, like
/// its chat route.
async fn lower_chat_render(
    core: &RenderCore<'_>,
    lora_resolution: &LoraModelResolution,
    max_model_len: u32,
    max_n_sequences: i64,
    headers: HeaderMap,
    body: ChatCompletionRequest,
) -> Result<Json<GenerateRequest>, ApiError> {
    let model = RenderCore::response_model(&lora_resolution.model_names, body.model.as_deref());
    let n = body.n;
    let stream = body.stream;
    let stream_options = body.stream_options.clone();
    let return_token_ids = body.return_token_ids;
    let mm_payload = body.mm_payload.unwrap_or(match body.return_mm_kwargs {
        Some(false) => MmPayload::None,
        _ => MmPayload::Tensors,
    });
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let chat_request = lower_chat_request(body, lora_resolution, request_context)?;
    // Lowering has checked that `n` is at least 1.
    validate_n_limit(n.unwrap_or(1), max_n_sequences)?;
    validate_max_output_tokens(
        "max_completion_tokens",
        chat_request.sampling_params.max_tokens,
        max_model_len,
    )?;
    let text_request = core
        .chat
        // The parallel samples of `n` are the generate request's to fan out;
        // its output grammar is the same for every sample.
        .prepare(chat_request, NonZeroU32::MIN, core.text)
        .await
        .map_err(|error| ApiError::invalid_request(error.to_report_string(), None))?
        .text_request;
    Ok(Json(core.lower_render_request(
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

/// Lower one completion request for `/v1/completions/render`, shared by the
/// standalone render server and the main API server's scale-out route.
/// Python answers with a list of requests, so a single prompt still yields a
/// one-element list.
fn lower_completion_render(
    core: &RenderCore<'_>,
    lora_resolution: &LoraModelResolution,
    max_model_len: u32,
    max_n_sequences: i64,
    headers: HeaderMap,
    body: CompletionRequest,
) -> Result<Json<Vec<GenerateRequest>>, ApiError> {
    let model = RenderCore::response_model(&lora_resolution.model_names, body.model.as_deref());
    let n = body.n;
    let stream = body.stream;
    let stream_options = body.stream_options.clone();
    let return_token_ids = body.return_token_ids;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let text_request = lower_completion_request(body, lora_resolution, request_context, core.text)?;
    // Lowering has checked that `n` is at least 1.
    validate_n_limit(n.unwrap_or(1), max_n_sequences)?;
    validate_max_output_tokens(
        "max_tokens",
        text_request.sampling_params.max_tokens,
        max_model_len,
    )?;
    Ok(Json(vec![core.lower_render_request(
        &lora_resolution.model_names,
        text_request,
        model,
        n,
        stream,
        stream_options,
        return_token_ids,
        MmPayload::Tensors,
    )?]))
}

async fn render_chat(
    State(state): State<Arc<RenderState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Result<Json<GenerateRequest>, ApiError> {
    lower_chat_render(
        &RenderCore::from_render_state(&state),
        &model_resolution(&state),
        state.text.max_model_len(),
        state.max_n_sequences,
        headers,
        body,
    )
    .await
}

async fn render_completion(
    State(state): State<Arc<RenderState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Result<Json<Vec<GenerateRequest>>, ApiError> {
    lower_completion_render(
        &RenderCore::from_render_state(&state),
        &model_resolution(&state),
        state.text.max_model_len(),
        state.max_n_sequences,
        headers,
        body,
    )
}

/// Scale-out render handled by the main API server, enabled together with
/// `/inference/v1/generate` by `--enable-scale-out`. Python serves the render
/// endpoints next to generation with one model load, so the Rust frontend does
/// the same over its shared processors.
pub(crate) async fn server_render_chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Result<Json<GenerateRequest>, ApiError> {
    let requested_model = body.model.as_deref().filter(|model| !model.is_empty());
    let lora_resolution = state.resolve_model_with_loras(requested_model).await;
    let core = RenderCore::from_app_state(&state);
    let max_model_len = state.llm().info().max_model_len;
    lower_chat_render(
        &core,
        &lora_resolution,
        max_model_len,
        state.api_server_options.max_n_sequences,
        headers,
        body,
    )
    .await
}

/// See [`server_render_chat`]. Python answers with a list of requests.
pub(crate) async fn server_render_completion(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Result<Json<Vec<GenerateRequest>>, ApiError> {
    let requested_model = body.model.as_deref().filter(|model| !model.is_empty());
    let lora_resolution = state.resolve_model_with_loras(requested_model).await;
    let core = RenderCore::from_app_state(&state);
    let max_model_len = state.llm().info().max_model_len;
    lower_completion_render(
        &core,
        &lora_resolution,
        max_model_len,
        state.api_server_options.max_n_sequences,
        headers,
        body,
    )
}
