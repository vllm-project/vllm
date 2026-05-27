use std::sync::Arc;

use axum::extract::State;
use serde::Deserialize;
use thiserror_ext::AsReport;
use validator::Validate;

use crate::error::ApiError;
use crate::routes::openai::utils::types::Normalizable;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::{AppState, LoadLoRAError, UnloadLoRAError};

#[derive(Debug, Deserialize, Validate)]
pub(crate) struct LoadLoRAAdapterRequest {
    lora_name: String,
    lora_path: String,
    #[serde(default)]
    load_inplace: bool,
    #[serde(default)]
    is_3d_lora_weight: bool,
}

impl Normalizable for LoadLoRAAdapterRequest {}

#[derive(Debug, Deserialize, Validate)]
pub(crate) struct UnloadLoRAAdapterRequest {
    lora_name: String,
    #[serde(default)]
    lora_int_id: Option<u64>,
}

impl Normalizable for UnloadLoRAAdapterRequest {}

/// Dynamically load one LoRA adapter and expose it as an OpenAI model id.
pub async fn load_lora_adapter(
    State(state): State<Arc<AppState>>,
    ValidatedJson(request): ValidatedJson<LoadLoRAAdapterRequest>,
) -> Result<String, ApiError> {
    if request.lora_name.is_empty() || request.lora_path.is_empty() {
        return Err(ApiError::invalid_request(
            "Both 'lora_name' and 'lora_path' must be provided.".to_string(),
            None,
        ));
    }
    let lora_name = request.lora_name;
    state
        .load_lora(
            lora_name.clone(),
            request.lora_path,
            request.load_inplace,
            request.is_3d_lora_weight,
        )
        .await
        .map_err(|error| match error {
            LoadLoRAError::AlreadyLoaded { lora_name } => ApiError::invalid_request(
                format!(
                    "The lora adapter '{lora_name}' has already been loaded. If you want to load the adapter in place, set 'load_inplace' to true."
                ),
                Some("lora_name"),
            ),
            LoadLoRAError::Engine(error) => ApiError::server_error(format!(
                "failed to load LoRA adapter '{lora_name}': {}",
                error.to_report_string()
            )),
        })?;

    Ok(format!(
        "Success: LoRA adapter '{lora_name}' added successfully."
    ))
}

/// Remove one LoRA adapter from the engine and frontend registry.
pub async fn unload_lora_adapter(
    State(state): State<Arc<AppState>>,
    ValidatedJson(request): ValidatedJson<UnloadLoRAAdapterRequest>,
) -> Result<String, ApiError> {
    let _ = request.lora_int_id;
    if request.lora_name.is_empty() {
        return Err(ApiError::invalid_request(
            "'lora_name' needs to be provided to unload a LoRA adapter.".to_string(),
            Some("lora_name"),
        ));
    }

    let lora_request =
        state.unload_lora(&request.lora_name).await.map_err(|error| match error {
            UnloadLoRAError::NotFound { lora_name } => ApiError::model_not_found(lora_name),
            UnloadLoRAError::Engine(error) => ApiError::server_error(format!(
                "failed to unload LoRA adapter '{}': {}",
                request.lora_name,
                error.to_report_string()
            )),
            UnloadLoRAError::NotRemoved {
                lora_name,
                lora_int_id,
            } => ApiError::server_error(format!(
                "failed to unload LoRA adapter '{lora_name}' with id {lora_int_id}"
            )),
        })?;

    Ok(format!(
        "Success: LoRA adapter '{}' removed successfully.",
        lora_request.lora_name
    ))
}
