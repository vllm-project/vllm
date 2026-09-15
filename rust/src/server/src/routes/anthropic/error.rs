// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Anthropic error envelope over the shared [`ApiError`] family.

use axum::Json;
use axum::extract::{FromRequest, Request};
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;
use validator::Validate;

use super::types::{AnthropicError, AnthropicErrorResponse};
use crate::error::ApiError;
use crate::routes::openai::utils::types::Normalizable;
use crate::routes::openai::utils::validated_json::ValidatedJson;

// ============================================================================
// Oracle-mirrored error rendering
// ============================================================================

/// [`ApiError`] rendered in Anthropic's error envelope
/// (`{"type": "error", "error": {"type", "message"}}`) instead of the OpenAI
/// shape.
///
/// Mirrors the Python vLLM `translate_error_response` (api_router.py):
/// message and HTTP status carry over from the internal error, while the
/// error type maps to the Anthropic wire names (see [`anthropic_error_type`])
/// rather than being forwarded verbatim.
pub(crate) struct AnthropicApiError(pub ApiError);

impl IntoResponse for AnthropicApiError {
    fn into_response(self) -> Response {
        let status = self.0.status_code();
        let message = self.0.to_error_response().error.message;
        let body = AnthropicErrorResponse {
            response_type: "error",
            error: AnthropicError {
                error_type: anthropic_error_type(&self.0).to_string(),
                message,
            },
        };
        (status, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
// Pieces with no named Python counterpart: a deliberate divergence from the
// oracle's error-type forwarding, and extractor plumbing FastAPI provides
// implicitly on the Python side.

/// Anthropic wire error type for each [`ApiError`] class, following the real
/// API's status↔type pairing: 400 `invalid_request_error`,
/// 404 `not_found_error`, 500 `api_error`.
///
/// Deliberate divergence from the `type=response.error.type` forwarding in
/// the Python vLLM `translate_error_response`, which leaks OpenAI-side
/// exception class names onto the Anthropic wire.
fn anthropic_error_type(error: &ApiError) -> &'static str {
    match error {
        ApiError::InvalidRequest { .. } | ApiError::JsonParseError { .. } => {
            "invalid_request_error"
        }
        ApiError::ModelNotFound { .. } => "not_found_error",
        ApiError::ServerError { .. } => "api_error",
    }
}

/// [`ValidatedJson`] with rejections rendered in the Anthropic error envelope,
/// so parse and validation failures on the Anthropic routes never leak the
/// OpenAI error shape.
///
/// Covers the FastAPI plumbing the Python router leans on
/// (`validate_json_request` dependency + pydantic rejection handling); the
/// Rust stack needs the explicit extractor.
pub(crate) struct AnthropicJson<T>(pub T);

impl<S, T> FromRequest<S> for AnthropicJson<T>
where
    T: DeserializeOwned + Validate + Normalizable + Send,
    S: Send + Sync,
{
    type Rejection = AnthropicApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match ValidatedJson::<T>::from_request(req, state).await {
            Ok(ValidatedJson(data)) => Ok(Self(data)),
            Err(error) => Err(AnthropicApiError(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::*;

    #[test]
    fn error_types_map_to_anthropic_wire_names() {
        let cases = [
            (
                ApiError::InvalidRequest {
                    message: "m".to_string(),
                    param: None,
                },
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                ApiError::JsonParseError {
                    message: "m".to_string(),
                },
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                ApiError::ModelNotFound {
                    model: "m".to_string(),
                },
                StatusCode::NOT_FOUND,
                "not_found_error",
            ),
            (
                ApiError::ServerError {
                    message: "m".to_string(),
                },
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
            ),
        ];
        for (error, status, wire_type) in cases {
            assert_eq!(error.status_code(), status);
            assert_eq!(anthropic_error_type(&error), wire_type);
        }
    }
}
