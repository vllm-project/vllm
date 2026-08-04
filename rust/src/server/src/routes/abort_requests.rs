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
    let request_ids = body.request_ids.ok_or_else(|| {
        ApiError::invalid_request(
            "Missing 'request_ids' in request body".to_string(),
            Some("request_ids"),
        )
    })?;

    state
        .chat
        .abort(&request_ids)
        .await
        .map_err(|error| utility_call_error("abort_requests", error))?;

    Ok(StatusCode::OK)
}
