// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Error response log for `--log-error-stack`.

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use tracing::error;

use crate::routes::openai::utils::types::ErrorResponse;

/// Log each `ApiError` response at ERROR.
///
/// There is no stack trace to log in Rust, so this logs the error message.
/// Errors sent mid-stream are skipped, same as Python.
pub async fn log_error_response(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let response = next.run(request).await;
    if let Some(ErrorResponse { error: detail }) = response.extensions().get::<ErrorResponse>() {
        error!(
            %method,
            path,
            status = detail.code,
            error_type = detail.error_type,
            error = detail.message,
            "Error response"
        );
    }
    response
}
