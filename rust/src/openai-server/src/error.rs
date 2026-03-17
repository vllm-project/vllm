use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use openai_protocol::common::{ErrorDetail, ErrorResponse};

/// Small OpenAI-style error family used by the minimal HTTP layer.
#[derive(Debug, Clone)]
pub enum ApiError {
    /// The request is syntactically valid OpenAI JSON but asks for unsupported behavior.
    InvalidRequest {
        message: String,
        param: Option<String>,
    },
    /// The requested model name does not match the single configured model.
    ModelNotFound { model: String },
    /// An unexpected internal failure happened before streaming started.
    ServerError { message: String },
}

// TODO: use `thiserror-ext`.
impl ApiError {
    /// Build a generic invalid-request error without a parameter name.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            param: None,
        }
    }

    /// Build an invalid-request error tied to one request parameter.
    pub fn invalid_request_param(message: impl Into<String>, param: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            param: Some(param.into()),
        }
    }

    /// Build the standard model-not-found error used by OpenAI-compatible APIs.
    pub fn model_not_found(model: impl Into<String>) -> Self {
        Self::ModelNotFound {
            model: model.into(),
        }
    }

    /// Build an internal server error.
    pub fn server_error(message: impl Into<String>) -> Self {
        Self::ServerError {
            message: message.into(),
        }
    }

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
                param.clone(),
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
