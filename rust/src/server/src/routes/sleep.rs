// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::{RawQuery, State};
use serde::Serialize;
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
    tags_woken: Option<Vec<String>>,
}

fn sleep_params(query: Option<String>) -> Result<(u32, PauseMode), ApiError> {
    let mut level = None;
    let mut mode = None;
    if let Some(query) = query {
        for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
            match key.as_ref() {
                "level" => level = Some(value.into_owned()),
                "mode" => mode = Some(value.into_owned()),
                _ => {}
            }
        }
    }
    let level = level.unwrap_or_else(|| "1".to_owned());
    let level = level
        .parse::<u32>()
        .map_err(|_| ApiError::invalid_request("level must be an integer", Some("query.level")))?;
    if level > 2 {
        return Err(ApiError::invalid_request(
            "level must be between 0 and 2",
            Some("query.level"),
        ));
    }
    let mode = mode.unwrap_or_else(|| "abort".to_owned());
    let mode = mode.parse::<PauseMode>().map_err(|_| {
        ApiError::invalid_request("mode must be abort, wait, or keep", Some("query.mode"))
    })?;
    Ok((level, mode))
}

/// Put the engine to sleep.
pub async fn sleep(
    State(state): State<Arc<AppState>>,
    RawQuery(query): RawQuery,
) -> Result<Json<SleepResponse>, ApiError> {
    let (level, mode) = sleep_params(query)?;
    let recorder = METRICS.api_server.record_sleep_mode_operation("sleep");

    state
        .engine_core_client()
        .sleep(level, mode)
        .await
        .map_err(|error| utility_call_error("sleep", error))?;

    recorder.success();
    Ok(Json(SleepResponse { status: "sleeping", level }))
}

/// Release KV cache memory while keeping model weights resident.
pub async fn release_kv_cache_memory(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ReleaseKvCacheMemoryResponse>, ApiError> {
    let recorder = METRICS
        .api_server
        .record_sleep_mode_operation("release_kv_cache_memory");
    state
        .engine_core_client()
        .release_kv_cache_memory()
        .await
        .map_err(|error| utility_call_error("release_kv_cache_memory", error))?;

    recorder.success();
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

    let recorder = METRICS.api_server.record_sleep_mode_operation("wake");
    let fully_awake = state
        .engine_core_client()
        .wake_up(tags.clone())
        .await
        .map_err(|error| utility_call_error("wake_up", error))?;

    recorder.success();
    Ok(Json(WakeUpResponse {
        status: if fully_awake { "awake" } else { "sleeping" },
        tags_woken: tags,
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

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::sleep_params;

    #[test]
    fn sleep_query_validates_level_and_mode_before_dispatch() {
        for level in 0..=2 {
            let (parsed, _) = sleep_params(Some(format!("level={level}&mode=wait")))
                .expect("valid sleep query");
            assert_eq!(parsed, level);
        }
        for (query, parameter) in [
            ("level=invalid", "query.level"),
            ("level=-1", "query.level"),
            ("level=3", "query.level"),
            ("mode=invalid", "query.mode"),
        ] {
            let error = sleep_params(Some(query.to_owned())).expect_err("invalid sleep query");
            assert_eq!(error.status_code(), StatusCode::BAD_REQUEST);
            assert_eq!(error.to_error_response().error.param.as_deref(), Some(parameter));
        }
    }
}
