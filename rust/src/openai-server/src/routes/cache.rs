use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use serde::Deserialize;

use crate::error::ApiError;
use crate::routes::utils::utility_call_error;
use crate::state::AppState;

#[derive(Debug, Default, Deserialize)]
pub(crate) struct ResetPrefixCacheParams {
    #[serde(default)]
    reset_running_requests: bool,
    #[serde(default)]
    reset_external: bool,
}

/// Reset the local prefix cache and optionally the connector-managed external cache.
pub async fn reset_prefix_cache(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ResetPrefixCacheParams>,
) -> Result<StatusCode, ApiError> {
    state
        .engine_core_client()
        .reset_prefix_cache(params.reset_running_requests, params.reset_external)
        .await
        .map_err(|error| utility_call_error("reset_prefix_cache", error))?;

    Ok(StatusCode::OK)
}

/// Reset the multi-modal cache.
pub async fn reset_mm_cache(State(state): State<Arc<AppState>>) -> Result<StatusCode, ApiError> {
    state
        .engine_core_client()
        .reset_mm_cache()
        .await
        .map_err(|error| utility_call_error("reset_mm_cache", error))?;

    Ok(StatusCode::OK)
}

/// Reset the encoder cache.
pub async fn reset_encoder_cache(
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, ApiError> {
    state
        .engine_core_client()
        .reset_encoder_cache()
        .await
        .map_err(|error| utility_call_error("reset_encoder_cache", error))?;

    Ok(StatusCode::OK)
}
