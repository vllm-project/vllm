// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use vllm_metrics::METRICS;

use crate::error::{ApiError, invalid_request};
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Deserialize)]
pub(crate) struct InitWeightTransferRequest {
    init_info: Option<JsonValue>,
}

#[derive(Deserialize)]
pub(crate) struct UpdateWeightsRequest {
    update_info: Option<JsonValue>,
}

#[derive(Default, Deserialize)]
pub(crate) struct FinishWeightUpdateRequest {
    weight_version: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct UpdateWeightVersionRequest {
    new_version: String,
}

/// Require a top-level JSON object before typed deserialization.
///
/// `Json<T>` accepts a positional array for a struct (`[{}]`, `["v"]`), which
/// would let a body that is not an object through to the recorder. The Python
/// frontend rejects those bodies, so both ends use the same payload shape.
fn body_object(body: Json<JsonValue>) -> Result<JsonValue, ApiError> {
    let value = body.0;
    if !value.is_object() {
        return Err(ApiError::InvalidRequest {
            message: "Request body must be a JSON object".to_string(),
            param: None,
        });
    }
    Ok(value)
}

/// Parse a validated object into its typed request struct.
fn parse_body<T: serde::de::DeserializeOwned>(
    value: JsonValue,
    param: Option<&'static str>,
    message: &'static str,
) -> Result<T, ApiError> {
    serde_json::from_value(value).map_err(|_| ApiError::InvalidRequest {
        message: message.to_string(),
        param,
    })
}

#[derive(Serialize)]
pub(crate) struct MessageResponse {
    message: &'static str,
}

#[derive(Serialize)]
pub(crate) struct UpdateWeightVersionResponse {
    success: bool,
    new_version: String,
}

#[derive(Serialize)]
pub(crate) struct WeightInfoResponse {
    weight_version: String,
}

/// Initialize weight transfer for RL training.
pub async fn init_weight_transfer_engine(
    State(state): State<Arc<AppState>>,
    body: Result<Json<JsonValue>, JsonRejection>,
) -> Result<Json<MessageResponse>, ApiError> {
    let value = body_object(body?)?;
    let body: InitWeightTransferRequest =
        parse_body(value, Some("init_info"), "Invalid 'init' request body")?;
    let init_info = body.init_info.filter(JsonValue::is_object).ok_or_else(|| {
        invalid_request!(
            param = Some("init_info"),
            "'init_info' must be a JSON object"
        )
    })?;

    let recorder = METRICS.api_server.record_weight_operation("init");
    state
        .engine_core_client()
        .init_weight_transfer_engine(init_info)
        .await
        .map_err(|error| utility_call_error("init_weight_transfer_engine", error))?;
    recorder.success();

    Ok(Json(MessageResponse {
        message: "Weight transfer initialized",
    }))
}

/// Start a weight update transaction.
pub async fn start_weight_update(
    State(state): State<Arc<AppState>>,
) -> Result<Json<MessageResponse>, ApiError> {
    let recorder = METRICS.api_server.record_weight_operation("start");
    state
        .engine_core_client()
        .start_weight_update()
        .await
        .map_err(|error| utility_call_error("start_weight_update", error))?;
    recorder.success();

    Ok(Json(MessageResponse {
        message: "Weight update started",
    }))
}

/// Start a draft-model weight update transaction.
pub async fn start_draft_weight_update(
    State(state): State<Arc<AppState>>,
) -> Result<Json<MessageResponse>, ApiError> {
    let recorder = METRICS.api_server.record_weight_operation("start_draft");
    state
        .engine_core_client()
        .start_draft_weight_update()
        .await
        .map_err(|error| utility_call_error("start_draft_weight_update", error))?;
    recorder.success();

    Ok(Json(MessageResponse {
        message: "Draft weight update started",
    }))
}

/// Update model weights with backend-specific transfer metadata.
pub async fn update_weights(
    State(state): State<Arc<AppState>>,
    body: Result<Json<JsonValue>, JsonRejection>,
) -> Result<Json<MessageResponse>, ApiError> {
    let value = body_object(body?)?;
    let body: UpdateWeightsRequest = parse_body(
        value,
        Some("update_info"),
        "Invalid 'update_weights' request body",
    )?;
    let update_info = body
        .update_info
        .filter(|info| {
            info.is_object()
                || info.as_array().is_some_and(|items| items.iter().all(JsonValue::is_object))
        })
        .ok_or_else(|| {
            invalid_request!(
                param = Some("update_info"),
                "'update_info' must be a JSON object or a list of per-worker JSON objects",
            )
        })?;

    let recorder = METRICS.api_server.record_weight_operation("update");
    state
        .engine_core_client()
        .update_weights(update_info)
        .await
        .map_err(|error| utility_call_error("update_weights", error))?;
    recorder.success();

    Ok(Json(MessageResponse {
        message: "Weights updated",
    }))
}

/// Finish a weight update transaction and optionally set the weight version.
pub async fn finish_weight_update(
    State(state): State<Arc<AppState>>,
    body: Result<Option<Json<JsonValue>>, JsonRejection>,
) -> Result<Json<MessageResponse>, ApiError> {
    // HTTPVLLMWeightSyncClient omits the body when no version is supplied.
    let request: FinishWeightUpdateRequest = match body? {
        Some(body) => parse_body(
            body_object(body)?,
            Some("weight_version"),
            "Invalid 'finish_weight_update' request body",
        )?,
        None => FinishWeightUpdateRequest::default(),
    };

    let client = state.engine_core_client();
    let recorder = METRICS.api_server.record_weight_operation("finish");
    client
        .finish_weight_update()
        .await
        .map_err(|error| utility_call_error("finish_weight_update", error))?;
    recorder.success();
    // Version bookkeeping is a separate operation: a failure here must not be
    // reported as a failed finish, and the Python frontend records it the same way.
    if let Some(version) = request.weight_version {
        let recorder = METRICS.api_server.record_weight_operation("set_version");
        client
            .set_weight_version(&version)
            .await
            .map_err(|error| utility_call_error("set_weight_version", error))?;
        recorder.success();
    }

    Ok(Json(MessageResponse {
        message: "Weight update finished",
    }))
}

/// Update the weight version identifier.
pub async fn update_weight_version(
    State(state): State<Arc<AppState>>,
    body: Result<Json<JsonValue>, JsonRejection>,
) -> Result<Json<UpdateWeightVersionResponse>, ApiError> {
    let value = body_object(body?)?;
    let body: UpdateWeightVersionRequest = parse_body(
        value,
        Some("new_version"),
        "Invalid 'update_weight_version' request body",
    )?;

    let recorder = METRICS.api_server.record_weight_operation("set_version");
    state
        .engine_core_client()
        .set_weight_version(&body.new_version)
        .await
        .map_err(|error| utility_call_error("set_weight_version", error))?;
    recorder.success();

    Ok(Json(UpdateWeightVersionResponse {
        success: true,
        new_version: body.new_version,
    }))
}

/// Get current weight metadata.
pub async fn weight_info(
    State(state): State<Arc<AppState>>,
) -> Result<Json<WeightInfoResponse>, ApiError> {
    let weight_version = state
        .engine_core_client()
        .get_weight_version()
        .await
        .map_err(|error| utility_call_error("get_weight_version", error))?;

    Ok(Json(WeightInfoResponse { weight_version }))
}
