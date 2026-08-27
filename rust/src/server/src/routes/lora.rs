// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::extract::State;
use serde::Deserialize;
use thiserror_ext::AsReport;
use validator::Validate;

use crate::error::ApiError;
use crate::lora::{LoadLoraError, LoraPathAccessError, UnloadLoraError};
use crate::routes::openai::utils::types::Normalizable;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;

#[derive(Debug, Deserialize, Validate)]
pub(crate) struct LoadLoraAdapterRequest {
    lora_name: String,
    lora_path: String,
    #[serde(default)]
    load_inplace: bool,
    #[serde(default)]
    is_3d_lora_weight: bool,
}

impl Normalizable for LoadLoraAdapterRequest {}

#[derive(Debug, Deserialize, Validate)]
pub(crate) struct UnloadLoraAdapterRequest {
    lora_name: String,
    #[serde(default)]
    lora_int_id: Option<u64>,
}

impl Normalizable for UnloadLoraAdapterRequest {}

fn load_lora_api_error(error: LoadLoraError) -> ApiError {
    let message = error.to_report_string();
    match error {
        LoadLoraError::Disabled(_) => ApiError::invalid_request(message, None),
        LoadLoraError::InvalidRequest(_) => ApiError::invalid_request(message, None),
        LoadLoraError::PathAccess(LoraPathAccessError::InvalidPath { .. }) => {
            ApiError::invalid_request(message, Some("lora_path"))
        }
        LoadLoraError::PathAccess(LoraPathAccessError::InvalidConfiguration { .. }) => {
            ApiError::server_error(message)
        }
        LoadLoraError::AlreadyLoaded { .. } => ApiError::invalid_request(
            format!(
                "{message}. If you want to load the adapter in place, set 'load_inplace' to true."
            ),
            Some("lora_name"),
        ),
        LoadLoraError::BaseModelName { .. } => {
            ApiError::invalid_request(message, Some("lora_name"))
        }
        LoadLoraError::Engine { .. } | LoadLoraError::NotLoaded { .. } => {
            ApiError::server_error(message)
        }
    }
}

fn unload_lora_api_error(error: UnloadLoraError) -> ApiError {
    let message = error.to_report_string();
    match error {
        UnloadLoraError::Disabled(_) => ApiError::invalid_request(message, None),
        UnloadLoraError::NotFound { lora_name } => ApiError::model_not_found(lora_name),
        UnloadLoraError::IntIdMismatch { .. } => {
            ApiError::invalid_request(message, Some("lora_int_id"))
        }
        UnloadLoraError::Engine { .. } | UnloadLoraError::NotRemoved { .. } => {
            ApiError::server_error(message)
        }
    }
}

/// Dynamically load one LoRA adapter and expose it as an OpenAI model id.
pub async fn load_lora_adapter(
    State(state): State<Arc<AppState>>,
    ValidatedJson(request): ValidatedJson<LoadLoraAdapterRequest>,
) -> Result<String, ApiError> {
    let lora_name = request.lora_name;
    state
        .load_lora(
            lora_name.clone(),
            request.lora_path,
            request.load_inplace,
            request.is_3d_lora_weight,
        )
        .await
        .map_err(load_lora_api_error)?;

    Ok(format!(
        "Success: LoRA adapter '{lora_name}' added successfully."
    ))
}

/// Remove one LoRA adapter from the engine and frontend registry.
pub async fn unload_lora_adapter(
    State(state): State<Arc<AppState>>,
    ValidatedJson(request): ValidatedJson<UnloadLoraAdapterRequest>,
) -> Result<String, ApiError> {
    if request.lora_name.is_empty() {
        return Err(ApiError::invalid_request(
            "'lora_name' needs to be provided to unload a LoRA adapter.".to_string(),
            Some("lora_name"),
        ));
    }

    let lora_request = state
        .unload_lora(&request.lora_name, request.lora_int_id)
        .await
        .map_err(unload_lora_api_error)?;

    Ok(format!(
        "Success: LoRA adapter '{}' removed successfully.",
        lora_request.lora_name
    ))
}
