use std::sync::Arc;

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Serialize)]
pub(crate) struct IsSleepingResponse {
    is_sleeping: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SleepParams {
    #[serde(default = "default_sleep_level")]
    level: u32,
    #[serde(default = "default_sleep_mode")]
    mode: String,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct WakeUpParams {
    #[serde(default)]
    tags: Option<Vec<String>>,
}

const fn default_sleep_level() -> u32 {
    1
}

fn default_sleep_mode() -> String {
    "abort".to_string()
}

/// Put the engine to sleep.
pub async fn sleep(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SleepParams>,
) -> Result<StatusCode, ApiError> {
    state
        .engine_core_client()
        .sleep(params.level, &params.mode)
        .await
        .map_err(|error| utility_call_error("sleep", error))?;

    Ok(StatusCode::OK)
}

/// Wake the engine from sleep mode.
pub async fn wake_up(
    State(state): State<Arc<AppState>>,
    Query(params): Query<WakeUpParams>,
) -> Result<StatusCode, ApiError> {
    state
        .engine_core_client()
        .wake_up(params.tags)
        .await
        .map_err(|error| utility_call_error("wake_up", error))?;

    Ok(StatusCode::OK)
}

/// Return whether the engine is currently sleeping at any level.
pub async fn is_sleeping(
    State(state): State<Arc<AppState>>,
) -> Result<Json<IsSleepingResponse>, ApiError> {
    let is_sleeping = state
        .engine_core_client()
        .is_sleeping()
        .await
        .map_err(|error| utility_call_error("is_sleeping", error))?;

    Ok(Json(IsSleepingResponse { is_sleeping }))
}
