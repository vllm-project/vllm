// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};

use crate::anthropic::{MessagesLlm, MessagesRequest};
use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::resolve_request_context;

#[derive(Debug, serde::Serialize)]
struct ErrorResponse {
    #[serde(rename = "type")]
    kind: &'static str,
    error: ErrorDetail,
}

#[derive(Debug, serde::Serialize)]
struct ErrorDetail {
    #[serde(rename = "type")]
    kind: &'static str,
    message: String,
}

fn error_body(error: ApiError) -> ErrorResponse {
    let kind = match error {
        ApiError::InvalidRequest { .. } | ApiError::JsonParseError { .. } => {
            "invalid_request_error"
        }
        ApiError::ModelNotFound { .. } | ApiError::NotFound { .. } => "not_found_error",
        ApiError::Conflict { .. } => "conflict_error",
        ApiError::ServiceUnavailable { .. } => "overloaded_error",
        ApiError::ServerError { .. } => "api_error",
    };
    ErrorResponse {
        kind: "error",
        error: ErrorDetail {
            kind,
            message: error.message(),
        },
    }
}

fn error_response(error: ApiError) -> Response {
    (error.status_code(), Json(error_body(error))).into_response()
}

pub async fn count_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(body) => body.0,
        Err(error) => return error_response(error.into()),
    };
    let model = state.resolve_model_with_loras(Some(&body.model)).await;
    match MessagesLlm::new(&state.chat, state.api_server_options)
        .count_tokens(body, &model, resolve_request_context(&headers, None))
        .await
    {
        Ok(response) => Json(response).into_response(),
        Err(error) => error_response(error),
    }
}

pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(body) => body.0,
        Err(error) => return error_response(error.into()),
    };
    let model = state.resolve_model_with_loras(Some(&body.model)).await;
    match MessagesLlm::new(&state.chat, state.api_server_options)
        .create(body, &model, resolve_request_context(&headers, None))
        .await
    {
        Ok(response) => Json(response).into_response(),
        Err(error) => error_response(error),
    }
}
