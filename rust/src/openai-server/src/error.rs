use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use openai_protocol::common::{ErrorDetail, ErrorResponse};
use thiserror_ext::{Construct, Macro};

/// Small OpenAI-style error family used by the minimal HTTP layer.
#[derive(Debug, Construct, Macro)]
pub enum ApiError {
    /// The request is syntactically valid OpenAI JSON but asks for unsupported behavior.
    InvalidRequest {
        message: String,
        param: Option<&'static str>,
    },
    /// The requested model name does not match the single configured model.
    ModelNotFound { model: String },
    /// An unexpected internal failure happened before streaming started.
    ServerError { message: String },
}

impl ApiError {
    /// Return the HTTP status code associated with this API error.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidRequest { .. } => StatusCode::BAD_REQUEST,
            Self::ModelNotFound { .. } => StatusCode::NOT_FOUND,
            Self::ServerError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Convert this error into the standard OpenAI-compatible JSON error payload.
    pub fn to_error_response(&self) -> ErrorResponse {
        let (error_type, message, param, code) = match self {
            Self::InvalidRequest { message, param } => (
                "invalid_request_error",
                message.clone(),
                param.map(|p| p.to_string()),
                Some("invalid_request_error".to_string()),
            ),
            Self::ModelNotFound { model } => (
                "invalid_request_error",
                format!("The model `{model}` does not exist."),
                Some("model".to_string()),
                Some("model_not_found".to_string()),
            ),
            Self::ServerError { message } => (
                "server_error",
                message.clone(),
                None,
                Some("server_error".to_string()),
            ),
        };

        ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: error_type.to_string(),
                param,
                code,
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status_code(), Json(self.to_error_response())).into_response()
    }
}
