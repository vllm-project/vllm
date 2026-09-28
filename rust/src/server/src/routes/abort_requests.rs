// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use serde::Deserialize;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Debug, Deserialize)]
pub(crate) struct AbortRequestsRequest {
    request_ids: Option<Vec<String>>,
}

pub async fn abort_requests(
    State(state): State<Arc<AppState>>,
    body: Result<Json<AbortRequestsRequest>, JsonRejection>,
) -> Result<StatusCode, ApiError> {
    let Json(body) = body.map_err(|error| ApiError::json_parse_error(error.body_text()))?;
    // Empty/missing `request_ids` aborts all in-flight requests.
    let request_ids = body.request_ids.unwrap_or_default();

    state
        .chat
        .abort(&request_ids)
        .await
        .map_err(|error| utility_call_error("abort_requests", error))?;

    Ok(StatusCode::OK)
}
