use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::State;
use serde::Deserialize;
use thiserror_ext::AsReport;
use validator::Validate;

use crate::error::ApiError;
use crate::routes::openai::utils::types::Normalizable;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::{AppState, LoadLoRAError, UnloadLoRAError};

const RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV: &str = "VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES";

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

fn runtime_lora_allowed_path_prefixes() -> Option<Vec<PathBuf>> {
    let prefixes = std::env::var_os(RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV)?;
    let prefixes: Vec<_> = std::env::split_paths(&prefixes)
        .filter(|path| !path.as_os_str().is_empty())
        .collect();
    (!prefixes.is_empty()).then_some(prefixes)
}

fn looks_like_local_lora_path(lora_path: &str) -> bool {
    let path = Path::new(lora_path);
    path.is_absolute() || lora_path.starts_with('~') || lora_path.starts_with('.')
}

fn validate_lora_path_access(
    lora_path: &str,
    allowed_prefixes: Option<&[PathBuf]>,
) -> Result<(), ApiError> {
    if !looks_like_local_lora_path(lora_path) {
        return Ok(());
    }

    let path = Path::new(lora_path);
    let Some(allowed_prefixes) = allowed_prefixes else {
        return Err(ApiError::invalid_request(
            format!(
                "Local LoRA adapter paths require {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV} to be configured."
            ),
            Some("lora_path"),
        ));
    };

    if !path.is_absolute() {
        return Err(ApiError::invalid_request(
            format!(
                "Local LoRA adapter paths must be absolute and under one of the prefixes configured by {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV}."
            ),
            Some("lora_path"),
        ));
    }

    if !allowed_prefixes.iter().any(|prefix| path.starts_with(prefix)) {
        return Err(ApiError::invalid_request(
            "Local LoRA adapter path is outside the configured allowed prefixes.".to_string(),
            Some("lora_path"),
        ));
    }

    Ok(())
}

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
    let allowed_prefixes = runtime_lora_allowed_path_prefixes();
    validate_lora_path_access(&request.lora_path, allowed_prefixes.as_deref())?;

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
            LoadLoRAError::BaseModelName { lora_name } => ApiError::invalid_request(
                format!("The lora adapter name '{lora_name}' conflicts with a served base model."),
                Some("lora_name"),
            ),
            LoadLoRAError::Engine(error) => ApiError::server_error(format!(
                "failed to load LoRA adapter '{lora_name}': {}",
                error.to_report_string()
            )),
            LoadLoRAError::NotLoaded { lora_name } => ApiError::server_error(format!(
                "failed to load LoRA adapter '{lora_name}': engine rejected the adapter"
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::validate_lora_path_access;

    #[test]
    fn lora_path_allows_hf_repo_ids_without_prefixes() {
        validate_lora_path_access("org/adapter-a", None).expect("hf repo id should be allowed");
    }

    #[test]
    fn lora_path_rejects_local_paths_without_prefixes() {
        assert!(validate_lora_path_access("/tmp/adapter-a", None).is_err());
        assert!(validate_lora_path_access("./adapter-a", None).is_err());
        assert!(validate_lora_path_access("~/adapter-a", None).is_err());
    }

    #[test]
    fn lora_path_allows_absolute_paths_under_configured_prefixes() {
        let prefixes = [PathBuf::from("/srv/vllm-loras")];
        validate_lora_path_access("/srv/vllm-loras/adapter-a", Some(&prefixes))
            .expect("path under configured prefix should be allowed");
        assert!(validate_lora_path_access("/tmp/adapter-a", Some(&prefixes)).is_err());
    }
}
