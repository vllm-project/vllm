// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use vllm_metrics::METRICS;

use crate::error::{ApiError, invalid_request};
use crate::lora::{LoadedLora, LoraWeightUpdateSpec};
use crate::routes::lora::load_lora_api_error;
use crate::routes::weight_checker::{mismatches, weight_checksums};
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Deserialize)]
pub(crate) struct InitWeightTransferRequest {
    init_info: Option<JsonValue>,
}

#[derive(Deserialize)]
pub(crate) struct LoraAdapterUpdateRequest {
    lora_name: String,
    peft_config: JsonValue,
    #[serde(default)]
    lora_int_id: Option<u64>,
    #[serde(default)]
    base_model_name: Option<String>,
    #[serde(default)]
    is_3d_lora_weight: bool,
}

#[derive(Default, Deserialize)]
pub(crate) struct StartWeightUpdateRequest {
    #[serde(default)]
    lora_adapter: Option<LoraAdapterUpdateRequest>,
}

#[derive(Deserialize)]
pub(crate) struct UpdateWeightsRequest {
    update_info: Option<JsonValue>,
}

#[derive(Default, Deserialize)]
pub(crate) struct FinishWeightUpdateRequest {
    weight_version: Option<String>,
    baseline: Option<BTreeMap<String, String>>,
    #[serde(default)]
    reset_prefix_cache: Option<bool>,
}

#[derive(Deserialize)]
pub(crate) struct UpdateWeightVersionRequest {
    new_version: String,
}

#[derive(Serialize)]
pub(crate) struct LoraAdapterSummary {
    lora_name: String,
    lora_int_id: u64,
    lora_path: String,
}

impl From<&LoadedLora> for LoraAdapterSummary {
    fn from(adapter: &LoadedLora) -> Self {
        Self {
            lora_name: adapter.lora_name.clone(),
            lora_int_id: adapter.lora_int_id,
            lora_path: adapter.lora_path.clone(),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct MessageResponse {
    message: &'static str,
}

#[derive(Serialize)]
pub(crate) struct StartWeightUpdateResponse {
    message: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    lora_adapter: Option<LoraAdapterSummary>,
}

#[derive(Serialize)]
pub(crate) struct FinishWeightUpdateResponse {
    message: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    lora_adapter: Option<LoraAdapterSummary>,
    #[serde(rename = "match", skip_serializing_if = "Option::is_none")]
    matched: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mismatches: Option<Vec<String>>,
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
    let Json(body) = body?;
    let init_info = body.init_info.filter(JsonValue::is_object).ok_or_else(|| {
        invalid_request!(
            param = Some("init_info"),
            "'init_info' must be a JSON object"
        )
    })?;

    let _recorder = METRICS.api_server.record_weight_operation("init");
    state
        .engine_core_client()
        .init_weight_transfer_engine(init_info)
        .await
        .map_err(|error| utility_call_error("init_weight_transfer_engine", error))?;

    Ok(Json(MessageResponse {
        message: "Weight transfer initialized",
    }))
}

/// Release the trainer connection without shutting down inference.
pub async fn shutdown_weight_transfer_engine(
    State(state): State<Arc<AppState>>,
) -> Result<Json<MessageResponse>, ApiError> {
    let _recorder = METRICS.api_server.record_weight_operation("shutdown");
    state
        .engine_core_client()
        .shutdown_weight_transfer_engine()
        .await
        .map_err(|error| utility_call_error("shutdown_weight_transfer_engine", error))?;

    Ok(Json(MessageResponse {
        message: "Weight transfer shut down",
    }))
}

/// Start a weight update transaction (base model when body is empty, or one
/// LoRA adapter when `lora_adapter` is provided).
pub async fn start_weight_update(
    State(state): State<Arc<AppState>>,
    body: Result<Option<Json<StartWeightUpdateRequest>>, JsonRejection>,
) -> Result<Json<StartWeightUpdateResponse>, ApiError> {
    let request = body?.map(|Json(request)| request).unwrap_or_default();
    let Some(adapter) = request.lora_adapter else {
        let _recorder = METRICS.api_server.record_weight_operation("start");
        state
            .engine_core_client()
            .start_weight_update()
            .await
            .map_err(|error| utility_call_error("start_weight_update", error))?;
        state.set_pending_lora_weight_update(None).await;

        return Ok(Json(StartWeightUpdateResponse {
            message: "Weight update started",
            lora_adapter: None,
        }));
    };

    let spec = LoraWeightUpdateSpec {
        lora_name: adapter.lora_name,
        lora_int_id: adapter.lora_int_id,
        peft_config: adapter.peft_config,
        base_model_name: adapter.base_model_name,
        is_3d_lora_weight: adapter.is_3d_lora_weight,
    };
    let prepared = state.prepare_lora_weight_update(&spec).await.map_err(load_lora_api_error)?;

    let _recorder = METRICS.api_server.record_weight_operation("start");
    state
        .engine_core_client()
        .start_lora_weight_update(prepared.worker_spec)
        .await
        .map_err(|error| utility_call_error("start_weight_update", error))?;

    let summary = LoraAdapterSummary::from(&prepared.loaded);
    state.set_pending_lora_weight_update(Some(prepared.loaded)).await;

    Ok(Json(StartWeightUpdateResponse {
        message: "Weight update started",
        lora_adapter: Some(summary),
    }))
}

/// Start a draft-model weight update transaction.
pub async fn start_draft_weight_update(
    State(state): State<Arc<AppState>>,
) -> Result<Json<MessageResponse>, ApiError> {
    let _recorder = METRICS.api_server.record_weight_operation("start_draft");
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
    let Json(body) = body?;
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

    let _recorder = METRICS.api_server.record_weight_operation("update");
    if let Err(error) = state.engine_core_client().update_weights(update_info).await {
        state.set_pending_lora_weight_update(None).await;
        return Err(utility_call_error("update_weights", error));
    }

    Ok(Json(MessageResponse {
        message: "Weights updated",
    }))
}

/// Finish a weight update; with a baseline, also compare the weights against it.
pub async fn finish_weight_update(
    State(state): State<Arc<AppState>>,
    body: Result<Option<Json<FinishWeightUpdateRequest>>, JsonRejection>,
) -> Result<Json<FinishWeightUpdateResponse>, ApiError> {
    // HTTPVLLMWeightSyncClient omits the body when no field is supplied.
    let request = body?.map(|Json(request)| request).unwrap_or_default();

    let client = state.engine_core_client();
    let pending_lora = state.take_pending_lora_weight_update().await;

    let _recorder = METRICS.api_server.record_weight_operation("finish");
    client
        .finish_weight_update()
        .await
        .map_err(|error| utility_call_error("finish_weight_update", error))?;

    let lora_summary = if let Some(adapter) = pending_lora {
        let summary = LoraAdapterSummary::from(&adapter);
        state.commit_lora_weight_update(adapter).await;
        Some(summary)
    } else {
        None
    };

    if request.reset_prefix_cache == Some(true) {
        client
            .reset_prefix_cache(false, false)
            .await
            .map_err(|error| utility_call_error("reset_prefix_cache", error))?;
    }

    if let Some(version) = request.weight_version {
        client
            .set_weight_version(&version)
            .await
            .map_err(|error| utility_call_error("set_weight_version", error))?;
    }
    let mismatches = match request.baseline {
        Some(baseline) => Some(mismatches(&weight_checksums(client).await?, &baseline)),
        None => None,
    };

    Ok(Json(FinishWeightUpdateResponse {
        message: "Weight update finished",
        lora_adapter: lora_summary,
        matched: mismatches.as_ref().map(Vec::is_empty),
        mismatches,
    }))
}

/// Update the weight version identifier.
pub async fn update_weight_version(
    State(state): State<Arc<AppState>>,
    body: Result<Json<UpdateWeightVersionRequest>, JsonRejection>,
) -> Result<Json<UpdateWeightVersionResponse>, ApiError> {
    let Json(body) = body?;
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
