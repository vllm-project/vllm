use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::error::ApiError;
use crate::state::AppState;

#[derive(Serialize)]
pub(crate) struct VersionResponse {
    version: String,
}

pub async fn version(
    State(state): State<Arc<AppState>>,
) -> Result<Json<VersionResponse>, ApiError> {
    let version = state
        .engine_core_client()
        .vllm_version()
        .ok_or_else(|| {
            ApiError::server_error("engine ready response did not include vLLM version".to_string())
        })?
        .to_string();

    Ok(Json(VersionResponse { version }))
}
