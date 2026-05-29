use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::State;
use serde::Deserialize;
use thiserror_ext::AsReport;
use validator::Validate;

use crate::error::ApiError;
use crate::lora::{LoadLoraError, UnloadLoraError};
use crate::routes::openai::utils::types::Normalizable;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;

const RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV: &str = "VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES";

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

    let canonical_path = path.canonicalize().map_err(|_| {
        ApiError::invalid_request(
            "Local LoRA adapter path must exist and be accessible.".to_string(),
            Some("lora_path"),
        )
    })?;
    let canonical_prefixes = allowed_prefixes
        .iter()
        .map(|prefix| {
            prefix.canonicalize().map_err(|_| {
                ApiError::server_error(format!(
                    "configured {RUNTIME_LORA_ALLOWED_PATH_PREFIXES_ENV} path prefix must exist and be accessible"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if !canonical_prefixes.iter().any(|prefix| canonical_path.starts_with(prefix)) {
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
    ValidatedJson(request): ValidatedJson<LoadLoraAdapterRequest>,
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
            LoadLoraError::AlreadyLoaded { lora_name } => ApiError::invalid_request(
                format!(
                    "The lora adapter '{lora_name}' has already been loaded. If you want to load the adapter in place, set 'load_inplace' to true."
                ),
                Some("lora_name"),
            ),
            LoadLoraError::BaseModelName { lora_name } => ApiError::invalid_request(
                format!("The lora adapter name '{lora_name}' conflicts with a served base model."),
                Some("lora_name"),
            ),
            LoadLoraError::Engine(error) => ApiError::server_error(format!(
                "failed to load LoRA adapter '{lora_name}': {}",
                error.to_report_string()
            )),
            LoadLoraError::NotLoaded { lora_name } => ApiError::server_error(format!(
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
        .map_err(|error| match error {
            UnloadLoraError::NotFound { lora_name } => ApiError::model_not_found(lora_name),
            UnloadLoraError::IntIdMismatch {
                lora_name,
                expected,
                actual,
            } => ApiError::invalid_request(
                format!(
                    "The requested lora_int_id {actual} does not match loaded adapter '{lora_name}' with id {expected}."
                ),
                Some("lora_int_id"),
            ),
            UnloadLoraError::Engine(error) => ApiError::server_error(format!(
                "failed to unload LoRA adapter '{}': {}",
                request.lora_name,
                error.to_report_string()
            )),
            UnloadLoraError::NotRemoved {
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
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::validate_lora_path_access;

    fn temp_lora_dir(test_name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "vllm-lora-{test_name}-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temp lora dir");
        path
    }

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
        let root = temp_lora_dir("allowed-prefix");
        let allowed = root.join("allowed");
        let adapter = allowed.join("adapter-a");
        fs::create_dir_all(&adapter).expect("create adapter dir");

        let prefixes = [allowed];
        validate_lora_path_access(adapter.to_str().expect("utf-8 temp path"), Some(&prefixes))
            .expect("path under configured prefix should be allowed");

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn lora_path_rejects_parent_escape_from_configured_prefixes() {
        let root = temp_lora_dir("parent-escape");
        let allowed = root.join("allowed");
        let private_adapter = root.join("private").join("adapter-a");
        fs::create_dir_all(&allowed).expect("create allowed dir");
        fs::create_dir_all(&private_adapter).expect("create private adapter dir");

        let escaped = allowed.join("../private/adapter-a");
        let prefixes = [allowed];
        assert!(
            validate_lora_path_access(escaped.to_str().expect("utf-8 temp path"), Some(&prefixes))
                .is_err()
        );

        fs::remove_dir_all(root).ok();
    }
}
