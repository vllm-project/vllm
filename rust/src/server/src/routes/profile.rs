// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Deserialize;
use tracing::info;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StartProfileRequest {
    profile_prefix: Option<String>,
    delay_iterations: Option<u64>,
    max_iterations: Option<u64>,
}

fn valid_profile_prefix(prefix: &str) -> bool {
    let bytes = prefix.as_bytes();
    (1..=128).contains(&bytes.len())
        && bytes[0].is_ascii_alphanumeric()
        && bytes[1..]
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

/// Start profiling the engine.
pub async fn start_profile(
    State(state): State<Arc<AppState>>,
    body: Option<Json<StartProfileRequest>>,
) -> Result<StatusCode, ApiError> {
    let body = body.map(|Json(body)| body).unwrap_or_default();
    if body
        .profile_prefix
        .as_deref()
        .is_some_and(|prefix| !valid_profile_prefix(prefix))
    {
        return Err(ApiError::invalid_request(
            "profile_prefix must be 1-128 characters, start with a letter or number, and \
             contain only letters, numbers, '.', '_', or '-'"
                .to_string(),
            Some("profile_prefix"),
        ));
    }

    info!("starting profiler");
    state
        .engine_core_client()
        .start_profile(
            body.profile_prefix.as_deref(),
            body.delay_iterations,
            body.max_iterations,
        )
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

#[cfg(test)]
mod tests {
    use super::valid_profile_prefix;

    #[test]
    fn validates_profile_prefixes() {
        for prefix in ["benchmark", "sharegpt_run-1", "profile.2"] {
            assert!(valid_profile_prefix(prefix), "{prefix}");
        }
        for prefix in ["", "../trace", "/tmp/trace", "trace name", "_trace"] {
            assert!(!valid_profile_prefix(prefix), "{prefix}");
        }
        assert!(!valid_profile_prefix(&"a".repeat(129)));
    }
}
