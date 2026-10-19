// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use vllm_engine_core_client::protocol::utility::{CheckpointState, PauseMode};

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Debug, Deserialize)]
pub(crate) struct PrepareParams {
    #[serde(default = "default_prepare_mode")]
    mode: PauseMode,
}

const fn default_prepare_mode() -> PauseMode {
    PauseMode::Wait
}

#[derive(Serialize)]
pub(crate) struct CheckpointStateResponse {
    state: CheckpointState,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

type CheckpointResult = Result<(StatusCode, Json<CheckpointStateResponse>), ApiError>;

fn state_response(
    status: StatusCode,
    state: CheckpointState,
    error: Option<String>,
) -> CheckpointResult {
    Ok((status, Json(CheckpointStateResponse { state, error })))
}

async fn current_state(state: &AppState) -> Result<(CheckpointState, Option<String>), ApiError> {
    state
        .engine_core_client()
        .checkpoint_state()
        .await
        .map_err(|error| utility_call_error("checkpoint_state", error))
}

/// Drain with `mode` and release what a process checkpoint must not contain.
/// Returns once started: poll `/checkpoint/status` until it is 200. Repeating
/// is a no-op.
pub async fn prepare(
    State(state): State<Arc<AppState>>,
    params: Result<Query<PrepareParams>, QueryRejection>,
) -> CheckpointResult {
    let Query(params) =
        params.map_err(|error| ApiError::invalid_request(error.body_text(), Some("mode")))?;
    let (mut checkpoint_state, mut error) = current_state(&state).await?;
    match checkpoint_state {
        CheckpointState::Running => {
            if let Some(refusal) = state
                .engine_core_client()
                .checkpoint_prepare(params.mode)
                .await
                .map_err(|error| utility_call_error("checkpoint_prepare", error))?
            {
                return state_response(StatusCode::BAD_REQUEST, checkpoint_state, Some(refusal));
            }
            (checkpoint_state, error) = current_state(&state).await?;
        }
        CheckpointState::Restoring | CheckpointState::Failed => {
            return state_response(StatusCode::BAD_REQUEST, checkpoint_state, error);
        }
        CheckpointState::Preparing | CheckpointState::Prepared => {}
    }
    state_response(StatusCode::OK, checkpoint_state, error)
}

/// Bring back what preparing released and serve, in the restored process or
/// in place. Orchestrators may deliver this more than once, so a restore in
/// progress or done returns 200.
pub async fn restore(State(state): State<Arc<AppState>>) -> CheckpointResult {
    let (mut checkpoint_state, mut error) = current_state(&state).await?;
    match checkpoint_state {
        CheckpointState::Preparing | CheckpointState::Failed => {
            return state_response(StatusCode::BAD_REQUEST, checkpoint_state, error);
        }
        CheckpointState::Prepared => {
            state
                .engine_core_client()
                .checkpoint_restore()
                .await
                .map_err(|error| utility_call_error("checkpoint_restore", error))?;
            (checkpoint_state, error) = current_state(&state).await?;
        }
        CheckpointState::Running | CheckpointState::Restoring => {}
    }
    state_response(StatusCode::OK, checkpoint_state, error)
}

/// Return 200 once every engine is prepared, 503 until then and 500 if that
/// cannot happen, so orchestrators can poll it to learn when the process is
/// safe to capture.
pub async fn status(State(state): State<Arc<AppState>>) -> CheckpointResult {
    let (checkpoint_state, error) = current_state(&state).await?;
    let status = match checkpoint_state {
        CheckpointState::Prepared => StatusCode::OK,
        CheckpointState::Failed => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::SERVICE_UNAVAILABLE,
    };
    state_response(status, checkpoint_state, error)
}
