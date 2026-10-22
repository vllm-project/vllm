// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Query, RawQuery, State};
use serde::{Deserialize, Serialize};
use vllm_engine_core_client::protocol::utility::PauseMode;
use vllm_metrics::METRICS;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Serialize)]
pub(crate) struct IsSleepingResponse {
    is_sleeping: bool,
}

#[derive(Serialize)]
pub struct SleepResponse {
    status: &'static str,
    level: u32,
}

#[derive(Serialize)]
pub struct ReleaseKvCacheMemoryResponse {
    status: &'static str,
}

#[derive(Serialize)]
pub struct WakeUpResponse {
    status: &'static str,
    /// Tags requested by the caller; null means wake all resources.
    tags: Option<Vec<String>>,
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
    ApiError::invalid_request(error.body_text(), None)
}

/// Put the engine to sleep.
pub async fn sleep(
    State(state): State<Arc<AppState>>,
    params: Result<Query<SleepParams>, QueryRejection>,
) -> Result<Json<SleepResponse>, ApiError> {
    let Query(SleepParams { level, mode }) = params.map_err(invalid_query)?;
    if level > 2 {
        return Err(ApiError::invalid_request(
            "level must be between 0 and 2",
            Some("level"),
        ));
    }
    let _recorder = METRICS.api_server.record_sleep_mode_operation("sleep");

    state
        .engine_core_client()
        .sleep(level, mode)
        .await
        .map_err(|error| utility_call_error("sleep", error))?;

    Ok(Json(SleepResponse {
        status: "sleeping",
        level,
    }))
}

/// Release KV cache memory while keeping model weights resident.
pub async fn release_kv_cache_memory(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ReleaseKvCacheMemoryResponse>, ApiError> {
    let _recorder = METRICS.api_server.record_sleep_mode_operation("release_kv_cache_memory");
    state
        .engine_core_client()
        .release_kv_cache_memory()
        .await
        .map_err(|error| utility_call_error("release_kv_cache_memory", error))?;

    Ok(Json(ReleaseKvCacheMemoryResponse {
        status: "kv_cache_released",
    }))
}

/// Wake the engine from sleep mode.
pub async fn wake_up(
    State(state): State<Arc<AppState>>,
    RawQuery(query): RawQuery,
) -> Result<Json<WakeUpResponse>, ApiError> {
    let tags = query.and_then(|query| {
        let tags = url::form_urlencoded::parse(query.as_bytes())
            .filter(|(key, _)| key == "tags")
            .map(|(_, value)| value.into_owned())
            .collect::<Vec<_>>();
        (!tags.is_empty()).then_some(tags)
    });

    let _recorder = METRICS.api_server.record_sleep_mode_operation("wake");
    let fully_awake = state
        .engine_core_client()
        .wake_up(tags.clone())
        .await
        .map_err(|error| utility_call_error("wake_up", error))?;

    Ok(Json(WakeUpResponse {
        status: if fully_awake { "awake" } else { "sleeping" },
        tags,
    }))
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
