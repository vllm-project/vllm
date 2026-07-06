use std::sync::Arc;

use axum::Json;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Query, RawQuery, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use vllm_engine_core_client::protocol::utility::PauseMode;
use vllm_metrics::{METRICS, SleepStateLabels};

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

/// Parse repeated `tags` query parameters (`?tags=weights&tags=kv_cache`),
/// mirroring FastAPI's `tags: list[str] | None = Query(None)`. Plain
/// `axum::extract::Query` cannot deserialize repeated keys into a `Vec`, so
/// the raw query string is split by hand; unknown parameters are ignored like
/// in the Python frontend.
fn parse_wake_up_tags(query: Option<&str>) -> Option<Vec<String>> {
    let query = query?;
    let tags: Vec<String> = form_urlencoded::parse(query.as_bytes())
        .filter(|(key, _)| key == "tags")
        .map(|(_, value)| value.into_owned())
        .collect();
    (!tags.is_empty()).then_some(tags)
}

const fn default_sleep_level() -> u32 {
    1
}

fn invalid_query(error: QueryRejection) -> ApiError {
    ApiError::invalid_request(error.body_text(), Some("mode"))
}

/// Update the `vllm:engine_sleep_state` gauges, mirroring the Python
/// `PrometheusStatLogger.record_sleep_state` semantics: any successful sleep
/// records the level flags, and any successful wake-up (even a partial one
/// with tags) records the engine as awake.
pub(crate) fn record_sleep_state(state: &AppState, sleep_level: Option<u32>) {
    let (awake, weights_offloaded, discard_all) = match sleep_level {
        Some(level) => (0, u64::from(level == 1), u64::from(level == 2)),
        None => (1, 0, 0),
    };

    let model_name = state.primary_model_name();
    for engine in state.engine_core_client().known_engine_indices() {
        for (sleep_state, value) in [
            ("awake", awake),
            ("weights_offloaded", weights_offloaded),
            ("discard_all", discard_all),
        ] {
            METRICS
                .scheduler
                .engine_sleep_state
                .get_or_create(&SleepStateLabels {
                    model_name: model_name.to_string(),
                    engine,
                    sleep_state,
                })
                .set(value);
        }
    }
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

    record_sleep_state(&state, Some(params.level));

    Ok(StatusCode::OK)
}

/// Wake the engine from sleep mode.
pub async fn wake_up(
    State(state): State<Arc<AppState>>,
    RawQuery(query): RawQuery,
) -> Result<StatusCode, ApiError> {
    let tags = parse_wake_up_tags(query.as_deref());

    state
        .engine_core_client()
        .wake_up(tags)
        .await
        .map_err(|error| utility_call_error("wake_up", error))?;

    record_sleep_state(&state, None);

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

#[cfg(test)]
mod tests {
    use super::parse_wake_up_tags;

    #[test]
    fn parse_wake_up_tags_handles_absent_single_and_repeated() {
        assert_eq!(parse_wake_up_tags(None), None);
        assert_eq!(parse_wake_up_tags(Some("")), None);
        assert_eq!(
            parse_wake_up_tags(Some("tags=weights")),
            Some(vec!["weights".to_string()])
        );
        assert_eq!(
            parse_wake_up_tags(Some("tags=weights&tags=kv_cache")),
            Some(vec!["weights".to_string(), "kv_cache".to_string()])
        );
        // Unknown parameters are ignored, like in the Python frontend.
        assert_eq!(parse_wake_up_tags(Some("other=1")), None);
    }
}
