use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use openai_protocol::common::{ErrorDetail, ErrorResponse};

#[derive(Debug)]
pub enum ApiError {
    InvalidRequest {
        message: String,
        param: Option<String>,
    },
    ModelNotFound {
        model: String,
    },
    ServerError {
        message: String,
    },
}

impl ApiError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            param: None,
        }
    }

    pub fn invalid_request_param(message: impl Into<String>, param: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            param: Some(param.into()),
        }
    }

    pub fn model_not_found(model: impl Into<String>) -> Self {
        Self::ModelNotFound {
            model: model.into(),
        }
    }

    pub fn server_error(message: impl Into<String>) -> Self {
        Self::ServerError {
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type, message, param, code) = match self {
            Self::InvalidRequest { message, param } => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
                param,
                Some("invalid_request_error".to_string()),
            ),
            Self::ModelNotFound { model } => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                format!("The model `{model}` does not exist."),
                Some("model".to_string()),
                Some("model_not_found".to_string()),
            ),
            Self::ServerError { message } => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                message,
                None,
                Some("server_error".to_string()),
            ),
        };

        (
            status,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message,
                    error_type: error_type.to_string(),
                    param,
                    code,
                },
            }),
        )
            .into_response()
    }
}
