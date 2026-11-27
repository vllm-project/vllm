// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use vllm_llm::NotReady;

use crate::error::{ApiError, bail_invalid_query};
use crate::state::AppState;

pub async fn health(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.llm().is_healthy() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct HealthGenerateParams {
    /// How long to wait for the engines before reporting `probe_timeout`.
    #[serde(default = "default_timeout_s")]
    timeout_s: f64,
}

const fn default_timeout_s() -> f64 {
    20.0
}

impl HealthGenerateParams {
    /// Validate `timeout_s`: it must be positive, as Python's `Query(gt=0)`.
    pub(crate) fn timeout(&self) -> Result<Duration, ApiError> {
        let timeout_s = self.timeout_s;
        if timeout_s.is_nan() || timeout_s <= 0.0 {
            bail_invalid_query!(
                param = Some("query.timeout_s".to_string()),
                "timeout_s must be greater than 0"
            );
        }
        // A timeout too long to represent waits forever, as Python's does.
        Ok(Duration::try_from_secs_f64(timeout_s).unwrap_or(Duration::MAX))
    }
}

/// The 503 body of [`health_generate`], as Python's: `reason` names the
/// [`NotReady`] variant and the other fields describe it.
#[serde_with::skip_serializing_none]
#[derive(Serialize)]
struct NotReadyResponse<'a> {
    status: &'static str,
    reason: &'static str,
    message: Option<String>,
    engine_ranks: Option<&'a [u32]>,
    timeout_s: Option<f64>,
    error: Option<&'a str>,
}

impl<'a> NotReadyResponse<'a> {
    fn new(not_ready: &'a NotReady) -> Self {
        let (reason, engine_ranks, timeout_s, error) = match not_ready {
            NotReady::EngineDead => ("engine_dead", None, None, None),
            NotReady::Sleeping { engine_ranks } => ("sleeping", Some(engine_ranks), None, None),
            NotReady::ProbeTimeout {
                engine_ranks,
                timeout,
            } => (
                "probe_timeout",
                Some(engine_ranks),
                Some(timeout.as_secs_f64()),
                None,
            ),
            NotReady::ProbeFailed {
                engine_ranks,
                error,
            } => ("probe_failed", Some(engine_ranks), None, Some(error)),
        };
        Self {
            status: "not_ready",
            reason,
            // Like Python, a dead engine is reported by `reason` alone.
            message: (!matches!(not_ready, NotReady::EngineDead)).then(|| not_ready.to_string()),
            engine_ranks: engine_ranks.map(Vec::as_slice),
            timeout_s,
            error: error.map(String::as_str),
        }
    }
}

/// Generation readiness check: 503 when an engine cannot run the model.
pub async fn health_generate(
    State(state): State<Arc<AppState>>,
    params: Result<Query<HealthGenerateParams>, QueryRejection>,
) -> Result<Response, ApiError> {
    let timeout = params?.0.timeout()?;
    Ok(match state.llm().check_health_generate(timeout).await {
        Ok(()) => StatusCode::OK.into_response(),
        Err(not_ready) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(NotReadyResponse::new(&not_ready)),
        )
            .into_response(),
    })
}
