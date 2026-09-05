use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Debug, Deserialize)]
pub(crate) struct InitWeightTransferRequest {
    init_info: Option<JsonValue>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct UpdateWeightsRequest {
    update_info: Option<JsonValue>,
}

#[derive(Debug, Serialize)]
pub(crate) struct WeightTransferResponse {
    message: &'static str,
}

fn required_field(value: Option<JsonValue>, field: &'static str) -> Result<JsonValue, ApiError> {
    value.ok_or_else(|| {
        ApiError::invalid_request(format!("Missing '{field}' in request body"), Some(field))
    })
}

async fn call_weight_transfer(
    state: &AppState,
    method: &'static str,
    kwargs: BTreeMap<&'static str, JsonValue>,
) -> Result<(), ApiError> {
    state
        .engine_core_client()
        .collective_rpc(method, None, Vec::<JsonValue>::new(), kwargs)
        .await
        .map_err(|error| utility_call_error(method, error))?;
    Ok(())
}

/// Initialize the configured weight transfer backend.
pub async fn init_weight_transfer_engine(
    State(state): State<Arc<AppState>>,
    body: Result<Json<InitWeightTransferRequest>, JsonRejection>,
) -> Result<Json<WeightTransferResponse>, ApiError> {
    let Json(body) = body.map_err(|error| ApiError::json_parse_error(error.body_text()))?;
    let init_info = required_field(body.init_info, "init_info")?;

    call_weight_transfer(
        &state,
        "init_weight_transfer_engine",
        BTreeMap::from([("init_info", init_info)]),
    )
    .await?;

    Ok(Json(WeightTransferResponse {
        message: "Weight transfer initialized",
    }))
}

/// Prepare the engine for a new weight update.
pub async fn start_weight_update(
    State(state): State<Arc<AppState>>,
) -> Result<Json<WeightTransferResponse>, ApiError> {
    call_weight_transfer(&state, "start_weight_update", BTreeMap::new()).await?;

    Ok(Json(WeightTransferResponse {
        message: "Weight update started",
    }))
}

/// Transfer one backend-specific weight update payload.
pub async fn update_weights(
    State(state): State<Arc<AppState>>,
    body: Result<Json<UpdateWeightsRequest>, JsonRejection>,
) -> Result<Json<WeightTransferResponse>, ApiError> {
    let Json(body) = body.map_err(|error| ApiError::json_parse_error(error.body_text()))?;
    let update_info = required_field(body.update_info, "update_info")?;

    call_weight_transfer(
        &state,
        "update_weights",
        BTreeMap::from([("update_info", update_info)]),
    )
    .await?;

    Ok(Json(WeightTransferResponse {
        message: "Weights updated",
    }))
}

/// Finalize the current weight update.
pub async fn finish_weight_update(
    State(state): State<Arc<AppState>>,
) -> Result<Json<WeightTransferResponse>, ApiError> {
    call_weight_transfer(&state, "finish_weight_update", BTreeMap::new()).await?;

    Ok(Json(WeightTransferResponse {
        message: "Weight update finished",
    }))
}
