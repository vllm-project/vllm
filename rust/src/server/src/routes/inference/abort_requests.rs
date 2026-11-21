// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Deserialize;

use crate::error::ApiError;
use crate::routes::openai::utils::types::StringOrArray;
use crate::state::AppState;
use crate::utils::utility_call_error;

/// Request body of `POST /inference/v1/abort_requests`.
#[derive(Debug, Deserialize)]
pub(crate) struct AbortRequestsRequest {
    /// External request IDs, as `/inference/v1/generate` returns them.
    request_ids: StringOrArray,
}

/// Abort in-flight requests by their external request IDs. Unknown IDs are
/// ignored.
///
/// Unlike the dev-mode `/abort_requests`, `request_ids` is required and an
/// empty list aborts nothing. Like Python, the body is parsed as JSON whatever
/// its `Content-Type`.
pub async fn abort_requests(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<StatusCode, ApiError> {
    let Json(body) = Json::<AbortRequestsRequest>::from_bytes(&body)?;
    let request_ids = body.request_ids.as_slice();
    // `abort` takes an empty slice as every in-flight request.
    if !request_ids.is_empty() {
        state
            .chat
            .abort(request_ids)
            .await
            .map_err(|error| utility_call_error("abort_requests", error))?;
    }
    Ok(StatusCode::OK)
}
