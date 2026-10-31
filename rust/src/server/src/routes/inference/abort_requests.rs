// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Deserialize;
use thiserror_ext::AsReport as _;
use tracing::error;
use tracing::instrument::WithSubscriber as _;

use crate::error::ApiError;
use crate::routes::openai::utils::types::StringOrArray;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
struct AbortRequestsRequest {
    request_ids: Option<StringOrArray>,
}

/// Accept cancellation of the supplied external request IDs.
pub async fn abort_requests(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<StatusCode, ApiError> {
    // Like Python's request.json(), accept JSON regardless of Content-Type.
    let Json(value) = Json::<serde_json::Value>::from_bytes(&body)?;
    if !value.is_object() {
        return Err(ApiError::json_parse_error(
            "Request body must be a JSON object",
        ));
    }
    // Unlike Python's untyped body, reject invalid ID types before scheduling.
    let body: AbortRequestsRequest = serde_json::from_value(value)
        .map_err(|error| ApiError::json_parse_error(error.to_report_string()))?;
    let request_ids = body
        .request_ids
        .ok_or_else(|| ApiError::json_parse_error("Missing 'request_ids' in request body"))?
        .into_vec();
    // The Llm facade treats an empty list as abort-all; scale-out treats it as a no-op.
    if request_ids.is_empty() {
        return Ok(StatusCode::OK);
    }

    // Match Python's accepted/background semantics and keep state alive during cancellation.
    tokio::spawn(
        async move {
            if let Err(error) = state.chat.abort(&request_ids).await {
                error!(error = %error.as_report(), "failed to abort scale-out requests");
            }
        }
        .with_current_subscriber(),
    );
    Ok(StatusCode::OK)
}
