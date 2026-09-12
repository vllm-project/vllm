// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use thiserror_ext::AsReport as _;

use crate::error::{ApiError, invalid_request, json_parse_error};
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
struct FinishWeightUpdateRequest {
    weight_version: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct UpdateWeightVersionRequest {
    new_version: String,
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
    body: Result<Json<InitWeightTransferRequest>, JsonRejection>,
) -> Result<Json<MessageResponse>, ApiError> {
    let Json(body) = body.map_err(|error| ApiError::json_parse_error(error.body_text()))?;
    let init_info = body.init_info.filter(JsonValue::is_object).ok_or_else(|| {
        invalid_request!(
            param = Some("init_info"),
            "'init_info' must be a JSON object"
        )
    })?;

    state
        .engine_core_client()
        .init_weight_transfer_engine(init_info)
        .await
        .map_err(|error| utility_call_error("init_weight_transfer_engine", error))?;

    Ok(Json(MessageResponse {
        message: "Weight transfer initialized",
    }))
}

/// Start a weight update transaction.
pub async fn start_weight_update(
    State(state): State<Arc<AppState>>,
) -> Result<Json<MessageResponse>, ApiError> {
    state
        .engine_core_client()
        .start_weight_update()
        .await
        .map_err(|error| utility_call_error("start_weight_update", error))?;

    Ok(Json(MessageResponse {
        message: "Weight update started",
    }))
}

/// Start a draft-model weight update transaction.
pub async fn start_draft_weight_update(
    State(state): State<Arc<AppState>>,
) -> Result<Json<MessageResponse>, ApiError> {
    state
        .engine_core_client()
        .start_draft_weight_update()
        .await
        .map_err(|error| utility_call_error("start_draft_weight_update", error))?;

    Ok(Json(MessageResponse {
        message: "Draft weight update started",
    }))
}

/// Update model weights with backend-specific transfer metadata.
pub async fn update_weights(
    State(state): State<Arc<AppState>>,
    body: Result<Json<UpdateWeightsRequest>, JsonRejection>,
) -> Result<Json<MessageResponse>, ApiError> {
    let Json(body) = body.map_err(|error| ApiError::json_parse_error(error.body_text()))?;
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

    state
        .engine_core_client()
        .update_weights(update_info)
        .await
        .map_err(|error| utility_call_error("update_weights", error))?;

    Ok(Json(MessageResponse {
        message: "Weights updated",
    }))
}

/// Finish a weight update transaction and optionally set the weight version.
pub async fn finish_weight_update(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Json<MessageResponse>, ApiError> {
    // HTTPVLLMWeightSyncClient omits the body when no version is supplied.
    let request = if body.is_empty() {
        FinishWeightUpdateRequest::default()
    } else {
        serde_json::from_slice::<Option<FinishWeightUpdateRequest>>(&body)
            .map_err(|error| json_parse_error!("{}", error.as_report()))?
            .unwrap_or_default()
    };

    let client = state.engine_core_client();
    client
        .finish_weight_update()
        .await
        .map_err(|error| utility_call_error("finish_weight_update", error))?;
    if let Some(version) = request.weight_version {
        client
            .set_weight_version(&version)
            .await
            .map_err(|error| utility_call_error("set_weight_version", error))?;
    }

    Ok(Json(MessageResponse {
        message: "Weight update finished",
    }))
}

/// Update the weight version identifier.
pub async fn update_weight_version(
    State(state): State<Arc<AppState>>,
    body: Result<Json<UpdateWeightVersionRequest>, JsonRejection>,
) -> Result<Json<UpdateWeightVersionResponse>, ApiError> {
    let Json(body) = body.map_err(|error| ApiError::json_parse_error(error.body_text()))?;
    state
        .engine_core_client()
        .set_weight_version(&body.new_version)
        .await
        .map_err(|error| utility_call_error("set_weight_version", error))?;

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
