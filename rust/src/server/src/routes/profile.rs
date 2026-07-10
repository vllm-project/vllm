use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use tracing::info;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

/// Start profiling the engine.
pub async fn start_profile(State(state): State<Arc<AppState>>) -> Result<StatusCode, ApiError> {
    info!("starting profiler");
    state
        .engine_core_client()
        .start_profile(None)
        .await
        .map_err(|error| utility_call_error("start_profile", error))?;
    info!("profiler started");
    Ok(StatusCode::OK)
}

/// Stop profiling the engine.
pub async fn stop_profile(State(state): State<Arc<AppState>>) -> Result<StatusCode, ApiError> {
    info!("stopping profiler");
    state
        .engine_core_client()
        .stop_profile(None)
        .await
        .map_err(|error| utility_call_error("stop_profile", error))?;
    info!("profiler stopped");
    Ok(StatusCode::OK)
}
