use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror_ext::{Construct, Macro};

use crate::routes::openai::utils::types::{ErrorDetail, ErrorResponse};

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
    /// The request body could not be parsed as valid JSON.
    JsonParseError { message: String },
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
            Self::JsonParseError { .. } => StatusCode::BAD_REQUEST,
        }
    }

    /// Convert this error into the standard OpenAI-compatible JSON error payload.
    pub fn to_error_response(&self) -> ErrorResponse {
        let error = match self {
            Self::InvalidRequest { message, param } => ErrorDetail {
                message: message.clone(),
                error_type: "invalid_request_error".to_string(),
                param: param.map(|p| p.to_string()),
                code: Some("invalid_request_error".to_string()),
            },
            Self::ModelNotFound { model } => ErrorDetail {
                message: format!("The model `{model}` does not exist."),
                error_type: "invalid_request_error".to_string(),
                param: Some("model".to_string()),
                code: Some("model_not_found".to_string()),
            },
            Self::ServerError { message } => ErrorDetail {
                message: message.clone(),
                error_type: "server_error".to_string(),
                param: None,
                code: Some("server_error".to_string()),
            },
            Self::JsonParseError { message } => ErrorDetail {
                message: message.clone(),
                error_type: "invalid_request_error".to_string(),
                param: None,
                code: Some("json_parse_error".to_string()),
            },
        };

        ErrorResponse { error }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status_code(), Json(self.to_error_response())).into_response()
    }
}
