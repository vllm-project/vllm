// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror_ext::{AsReport as _, Construct, Macro};
use vllm_engine_core_client::protocol::output::RequestError;

use crate::routes::openai::utils::types::{ErrorDetail, ErrorResponse};

/// Small OpenAI-style error family used by the minimal HTTP layer.
#[derive(Debug, Construct, Macro)]
pub enum ApiError {
    /// The request is syntactically valid OpenAI JSON but asks for unsupported
    /// behavior.
    InvalidRequest {
        message: String,
        param: Option<&'static str>,
    },
    /// The requested model name does not match the single configured model.
    ModelNotFound { model: String },
    /// A resource other than the model, such as a LoRA adapter, does not exist.
    NotFound { message: String },
    /// The request body could not be parsed as valid JSON.
    JsonParseError { message: String },
    /// The requested operation conflicts with current server state.
    Conflict { message: String },
    /// An unexpected internal failure happened before streaming started.
    ServerError { message: String },
    /// The engine rejected the request with a retryable error.
    ServiceUnavailable { message: String },
}

impl ApiError {
    /// Return the HTTP status code associated with this API error.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidRequest { .. } => StatusCode::BAD_REQUEST,
            Self::ModelNotFound { .. } | Self::NotFound { .. } => StatusCode::NOT_FOUND,
            Self::Conflict { .. } => StatusCode::CONFLICT,
            Self::ServerError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::JsonParseError { .. } => StatusCode::BAD_REQUEST,
            Self::ServiceUnavailable { .. } => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    /// Convert this error into the vLLM JSON error payload: `type` names the
    /// error class as the Python frontend does, and `code` is the HTTP status.
    pub fn to_error_response(&self) -> ErrorResponse {
        let (error_type, message, param) = match self {
            Self::InvalidRequest { message, param } => (
                "BadRequestError",
                message.clone(),
                param.map(str::to_string),
            ),
            Self::ModelNotFound { model } => (
                "NotFoundError",
                format!("The model `{model}` does not exist."),
                Some("model".to_string()),
            ),
            Self::NotFound { message } => ("NotFoundError", message.clone(), None),
            // Python reports a body that fails to parse into the request
            // model through FastAPI's validation handler, which names the
            // HTTP status instead of an error class.
            Self::JsonParseError { message } => ("Bad Request", message.clone(), None),
            // Python has no error class for 409; like its `HTTPException`
            // handler, name the status.
            Self::Conflict { message } => ("Conflict", message.clone(), None),
            Self::ServerError { message } => ("InternalServerError", message.clone(), None),
            Self::ServiceUnavailable { message } => {
                ("ServiceUnavailableError", message.clone(), None)
            }
        };

        ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: error_type.to_string(),
                param,
                code: self.status_code().as_u16(),
            },
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(error: JsonRejection) -> Self {
        Self::json_parse_error(error.body_text())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status_code(), Json(self.to_error_response())).into_response()
    }
}

/// Classify a text-pipeline submit failure: request validation failures are
/// the client's fault and map to HTTP 400, mirroring the Python frontend.
/// Everything else stays an internal 500.
pub fn text_submit_error(context: &'static str, error: vllm_text::Error) -> ApiError {
    if error.is_request_validation_error() {
        return invalid_request!(param = error.param(), "{error}");
    }
    server_error!("{}: {}", context, error.to_report_string())
}

/// Like [`text_submit_error`], for the chat pipeline, which wraps the text
/// errors.
pub fn chat_submit_error(context: &'static str, error: vllm_chat::Error) -> ApiError {
    if error.is_request_validation_error() {
        return invalid_request!(param = error.param(), "{error}");
    }
    server_error!("{}: {}", context, error.to_report_string())
}

/// Classify an engine-reported request failure: a validation error (e.g. a
/// constraint the structured-output backend cannot compile) maps to HTTP 400,
/// a retryable error (e.g. a paused engine's rejection) to 503, anything else
/// to a 500.
pub fn engine_error(error: Option<&RequestError>) -> ApiError {
    match error {
        Some(error) if error.is_validation_error() => invalid_request!("{}", error.message),
        Some(error) if error.retryable => ApiError::ServiceUnavailable {
            message: error.message.clone(),
        },
        _ => server_error!("Internal server error"),
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use itertools::Itertools as _;

    use super::*;

    /// Each error class serializes like Python's `ErrorInfo`: the class name
    /// in `type` and the HTTP status as an integer `code`.
    #[test]
    fn error_response_names_error_class_and_status_code() {
        let errors = [
            ApiError::invalid_request("top_p must be in (0, 1].", Some("top_p")),
            ApiError::model_not_found("missing"),
            ApiError::not_found("No adapter found for /not/an/adapter"),
            ApiError::json_parse_error("expected value at line 1 column 1"),
            ApiError::conflict("A profiling session is already active."),
            ApiError::server_error("Internal server error"),
            ApiError::service_unavailable("Generation is paused."),
        ];
        let rendered = errors
            .iter()
            .map(|error| serde_json::to_string(&error.to_error_response()).unwrap())
            .join("\n");
        expect![[r#"
            {"error":{"message":"top_p must be in (0, 1].","type":"BadRequestError","param":"top_p","code":400}}
            {"error":{"message":"The model `missing` does not exist.","type":"NotFoundError","param":"model","code":404}}
            {"error":{"message":"No adapter found for /not/an/adapter","type":"NotFoundError","param":null,"code":404}}
            {"error":{"message":"expected value at line 1 column 1","type":"Bad Request","param":null,"code":400}}
            {"error":{"message":"A profiling session is already active.","type":"Conflict","param":null,"code":409}}
            {"error":{"message":"Internal server error","type":"InternalServerError","param":null,"code":500}}
            {"error":{"message":"Generation is paused.","type":"ServiceUnavailableError","param":null,"code":503}}"#]].assert_eq(&rendered);
    }

    /// Reported like Python's renderer length check, whose message clients
    /// match to detect context-window overflows.
    #[test]
    fn prompt_too_long_maps_to_invalid_request() {
        let errors = [
            vllm_text::Error::PromptTooLong {
                max_model_len: 8192,
                prompt_len: 9000,
            },
            vllm_text::Error::PromptFillsContext {
                max_model_len: 8192,
            },
        ];
        let response = errors
            .into_iter()
            .map(|error| {
                let api_error = text_submit_error("failed to submit completion request", error);
                assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
                serde_json::to_string(&api_error.to_error_response()).unwrap()
            })
            .join("\n");
        expect![[r#"
            {"error":{"message":"This model's maximum context length is 8192 tokens. However, your prompt contains 9000 input tokens, which exceeds this limit even without any output tokens. Please reduce the length of the input prompt.","type":"BadRequestError","param":"input_tokens","code":400}}
            {"error":{"message":"This model's maximum context length is 8192 tokens. However, your prompt fills it with 8192 input tokens, leaving no room for output tokens. Please reduce the length of the input prompt.","type":"BadRequestError","param":"input_tokens","code":400}}"#]].assert_eq(&response);
    }

    #[test]
    fn invalid_thinking_token_budget_maps_to_invalid_request() {
        let api_error = text_submit_error(
            "failed to submit completion request",
            vllm_text::Error::InvalidThinkingTokenBudget,
        );
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
        let response = api_error.to_error_response();
        assert_eq!(response.error.error_type, "BadRequestError");
        assert!(response.error.message.contains("thinking_token_budget"));
    }

    #[test]
    fn min_tokens_above_max_tokens_maps_to_invalid_request() {
        let api_error = text_submit_error(
            "failed to submit completion request",
            vllm_text::Error::MinTokensExceedsMaxTokens {
                min_tokens: 5,
                max_tokens: 4,
            },
        );
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
        let response = api_error.to_error_response();
        assert_eq!(response.error.error_type, "BadRequestError");
        assert!(response.error.message.contains("min_tokens=5"));
        assert!(response.error.message.contains("max_tokens=4"));
    }

    #[test]
    fn sampling_params_validation_maps_to_invalid_request() {
        let api_error = text_submit_error(
            "failed to submit completion request",
            vllm_text::Error::SamplingParams(vllm_text::SamplingParamsError::OutOfRange {
                parameter: "top_p",
                value: 0.0,
                expected: "(0, 1]",
            }),
        );
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
        let response = api_error.to_error_response();
        assert_eq!(response.error.error_type, "BadRequestError");
        assert!(response.error.message.contains("top_p"));
    }

    #[test]
    fn chat_wrapped_prompt_too_long_maps_to_invalid_request() {
        let error = vllm_chat::Error::Text(vllm_text::Error::PromptTooLong {
            max_model_len: 8192,
            prompt_len: 9000,
        });
        let api_error = chat_submit_error("failed to submit chat request", error);
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(
            api_error.to_error_response().error.param.as_deref(),
            Some("input_tokens")
        );
    }

    #[test]
    fn invalid_reasoning_parameters_map_to_invalid_request() {
        for error in [
            vllm_chat::Error::InvalidReasoningEffort(
                "DeepSeek V4.1 reasoning_effort must be within [1, 100]".to_string(),
            ),
            vllm_chat::Error::InvalidReasoningControl {
                message: "template kwarg `thinking` must be a boolean".to_string(),
            },
        ] {
            let api_error = chat_submit_error("failed to submit chat request", error);
            assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
            assert_eq!(
                api_error.to_error_response().error.error_type,
                "BadRequestError"
            );
        }
    }

    #[test]
    fn chat_template_throw_maps_to_invalid_request() {
        let error = vllm_chat::Error::ChatTemplateThrown {
            message: "Unexpected reasoning effort high.".to_string(),
        };
        let api_error = chat_submit_error("failed to submit chat request", error);
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
        let response = api_error.to_error_response();
        assert_eq!(response.error.error_type, "BadRequestError");
        assert_eq!(response.error.message, "Unexpected reasoning effort high.");
    }

    #[test]
    fn untrusted_request_chat_template_maps_to_invalid_request() {
        let api_error = chat_submit_error(
            "tokenize failed",
            vllm_chat::Error::UntrustedRequestChatTemplate,
        );
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(
            api_error.to_error_response().error.message,
            "Chat template is passed with request, but --trust-request-chat-template is not set. \
             Refused request with untrusted chat template."
        );
    }

    #[test]
    fn chat_template_render_failure_stays_internal() {
        let error = vllm_chat::Error::ChatTemplate(
            "failed to render jinja template: unknown function".to_string(),
        );
        let api_error = chat_submit_error("failed to submit chat request", error);
        assert_eq!(api_error.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn llm_wrapped_empty_prompt_maps_to_invalid_request() {
        let error = vllm_text::Error::Llm(vllm_llm::Error::EmptyPromptTokenIds {
            request_id: "req-1".to_string(),
        });
        let api_error = text_submit_error("failed to submit completion request", error);
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn logprobs_validation_maps_to_invalid_request() {
        let error = vllm_text::Error::Logprobs(vllm_text::LogprobsError::TooManyCount {
            parameter: "logprobs",
            requested: 1000,
            max_allowed: 20,
        });
        let api_error = text_submit_error("failed to submit completion request", error);
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
        let response = api_error.to_error_response();
        assert_eq!(response.error.error_type, "BadRequestError");
        assert!(response.error.message.contains("logprobs"));
    }

    #[test]
    fn chat_wrapped_logprobs_validation_maps_to_invalid_request() {
        let error = vllm_chat::Error::Text(vllm_text::Error::Logprobs(
            vllm_text::LogprobsError::TooManyCount {
                parameter: "prompt_logprobs",
                requested: 1000,
                max_allowed: 20,
            },
        ));
        let api_error = chat_submit_error("failed to submit chat request", error);
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn out_of_vocab_validation_maps_to_invalid_request() {
        let error = vllm_text::Error::TokenIds(vllm_text::TokenIdsError::OutOfVocab {
            parameter: "logprob_token_ids",
            token_ids: vec![1000],
            vocab_size: 1000,
        });
        let api_error = text_submit_error("failed to submit completion request", error);
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn empty_allowed_token_ids_maps_to_invalid_request() {
        let error = vllm_text::Error::TokenIds(vllm_text::TokenIdsError::EmptyAllowedTokenIds);
        let api_error = text_submit_error("failed to submit completion request", error);
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
        let response = api_error.to_error_response();
        assert_eq!(response.error.error_type, "BadRequestError");
        assert!(response.error.message.contains("allowed_token_ids"));
    }

    #[test]
    fn truncate_exceeds_max_model_len_maps_to_invalid_request() {
        let error = vllm_text::Error::TruncatePromptTokensExceedsMaxModelLen {
            value: 8000,
            max_model_len: 4000,
        };
        let api_error = text_submit_error("failed to submit completion request", error);
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
        let response = api_error.to_error_response();
        assert_eq!(response.error.error_type, "BadRequestError");
        assert!(response.error.message.contains("truncate_prompt_tokens=8000"));
        assert!(response.error.message.contains("max_model_len = 4000"));
    }

    #[test]
    fn invalid_truncate_prompt_tokens_maps_to_invalid_request() {
        let error = vllm_text::Error::InvalidTruncatePromptTokens { value: -2 };
        let api_error = text_submit_error("failed to submit completion request", error);
        assert_eq!(api_error.status_code(), StatusCode::BAD_REQUEST);
        let response = api_error.to_error_response();
        assert_eq!(response.error.error_type, "BadRequestError");
        assert!(response.error.message.contains("-2"));
        assert!(response.error.message.contains("must be >= -1"));
    }

    #[test]
    fn other_submit_errors_stay_internal() {
        let error = vllm_text::Error::Tokenizer("backend exploded".to_string());
        let api_error = text_submit_error("failed to submit completion request", error);
        assert_eq!(api_error.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        let response = api_error.to_error_response();
        assert!(response.error.message.starts_with("failed to submit completion request:"));
    }
}
