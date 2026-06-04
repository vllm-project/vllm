use std::sync::Arc;

use axum::Json;
use axum::extract::{Query, State};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Debug, Deserialize)]
pub(crate) struct PauseParams {
    #[serde(default = "default_pause_mode")]
    mode: String,
    #[serde(default = "default_clear_cache")]
    clear_cache: bool,
}

#[derive(Serialize)]
pub(crate) struct StatusResponse {
    status: &'static str,
}

#[derive(Serialize)]
pub(crate) struct IsPausedResponse {
    is_paused: bool,
}

/// Pause modes accepted by the engine (`PauseMode` in the Python frontend).
const VALID_PAUSE_MODES: [&str; 3] = ["abort", "wait", "keep"];

fn default_pause_mode() -> String {
    "abort".to_string()
}

const fn default_clear_cache() -> bool {
    true
}

// TODO: the Python frontend also accepts the deprecated
// `wait_for_inflight_requests` flag (equivalent to `mode="wait"`); it is
// intentionally omitted here in favor of the `mode` parameter.

/// Pause the scheduler so generation can be halted (e.g. for weight updates).
pub async fn pause(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PauseParams>,
) -> Result<Json<StatusResponse>, ApiError> {
    if !VALID_PAUSE_MODES.contains(&params.mode.as_str()) {
        return Err(ApiError::invalid_request(
            format!(
                "Invalid pause mode '{}'; expected one of: abort, wait, keep",
                params.mode
            ),
            Some("mode"),
        ));
    }

    state
        .engine_core_client()
        .pause_scheduler(&params.mode, params.clear_cache)
        .await
        .map_err(|error| utility_call_error("pause", error))?;

    Ok(Json(StatusResponse { status: "paused" }))
}

/// Resume the scheduler after a pause.
pub async fn resume(State(state): State<Arc<AppState>>) -> Result<Json<StatusResponse>, ApiError> {
    state
        .engine_core_client()
        .resume_scheduler()
        .await
        .map_err(|error| utility_call_error("resume", error))?;

    Ok(Json(StatusResponse { status: "resumed" }))
}

/// Return whether the scheduler is currently paused.
pub async fn is_paused(
    State(state): State<Arc<AppState>>,
) -> Result<Json<IsPausedResponse>, ApiError> {
    let is_paused = state
        .engine_core_client()
        .is_scheduler_paused()
        .await
        .map_err(|error| utility_call_error("is_paused", error))?;

    Ok(Json(IsPausedResponse { is_paused }))
}
