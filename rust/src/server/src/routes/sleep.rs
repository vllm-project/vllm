// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Query, RawQuery, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use vllm_engine_core_client::protocol::utility::PauseMode;

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
    #[serde(default)]
    mode: PauseMode,
}

const fn default_sleep_level() -> u32 {
    1
}

fn invalid_query(error: QueryRejection) -> ApiError {
    ApiError::invalid_request(error.body_text(), Some("mode"))
}

/// Put the engine to sleep.
pub async fn sleep(
    State(state): State<Arc<AppState>>,
    params: Result<Query<SleepParams>, QueryRejection>,
) -> Result<StatusCode, ApiError> {
    let Query(params) = params.map_err(invalid_query)?;

    state
        .engine_core_client()
        .sleep(params.level, params.mode)
        .await
        .map_err(|error| utility_call_error("sleep", error))?;

    Ok(StatusCode::OK)
}

/// Release KV cache memory while keeping model weights resident.
pub async fn release_kv_cache_memory(
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, ApiError> {
    state
        .engine_core_client()
        .release_kv_cache_memory()
        .await
        .map_err(|error| utility_call_error("release_kv_cache_memory", error))?;

    Ok(StatusCode::OK)
}

/// Wake the engine from sleep mode.
pub async fn wake_up(
    State(state): State<Arc<AppState>>,
    RawQuery(query): RawQuery,
) -> Result<StatusCode, ApiError> {
    let tags = query.and_then(|query| {
        let tags = url::form_urlencoded::parse(query.as_bytes())
            .filter(|(key, _)| key == "tags")
            .map(|(_, value)| value.into_owned())
            .collect::<Vec<_>>();
        (!tags.is_empty()).then_some(tags)
    });

    state
        .engine_core_client()
        .wake_up(tags)
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
