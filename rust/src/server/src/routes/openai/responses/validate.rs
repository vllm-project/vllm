// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::types::{ResponseToolChoice, ResponsesRequest};
use crate::error::{ApiError, bail_invalid_request};

/// Enforce the compatibility contract of the Rust Responses API frontend.
///
/// Store-dependent features are rejected because the Rust frontend has no
/// response store (the Python frontend gates them behind
/// `VLLM_ENABLE_RESPONSES_API_STORE=1`):
/// - `store=true` requests still execute, matching the Python frontend's
///   implicit downgrade when the store is disabled; only the combination
///   with `background=true` is rejected.
/// - `previous_response_id` cannot be resolved without a store, and is
///   rejected with a 404 like an unknown response ID would be.
pub(super) fn validate_request_compat(request: &ResponsesRequest) -> Result<(), ApiError> {
    if request.background.unwrap_or(false) {
        bail_invalid_request!(
            param = "background",
            "background=true requires the Responses API store, which is not \
             supported by this frontend."
        );
    }

    if request.previous_response_id.is_some() {
        return Err(ApiError::response_not_found(
            request.previous_response_id.clone().unwrap_or_default(),
        ));
    }

    if request.prompt.is_some() {
        bail_invalid_request!(param = "prompt", "prompt template is not supported");
    }

    if let Some(ResponseToolChoice::Mode(mode)) = &request.tool_choice
        && !matches!(mode.as_str(), "none" | "auto" | "required")
    {
        bail_invalid_request!(
            param = "tool_choice",
            "tool_choice string form must be one of 'none', 'auto', or 'required'; \
             got '{mode}'."
        );
    }

    if let Some(truncation) = &request.truncation {
        // TODO: implement context-length truncation for `truncation="auto"`.
        if !matches!(truncation.as_str(), "auto" | "disabled") {
            bail_invalid_request!(
                param = "truncation",
                "truncation must be one of 'auto' or 'disabled'; got '{truncation}'."
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn base_request() -> ResponsesRequest {
        serde_json::from_value(json!({
            "model": "test-model",
            "input": "hello",
        }))
        .unwrap()
    }

    #[test]
    fn background_true_is_rejected() {
        let mut request = base_request();
        request.background = Some(true);
        let error = validate_request_compat(&request).unwrap_err();
        assert!(matches!(error, ApiError::InvalidRequest { .. }));
    }

    #[test]
    fn previous_response_id_is_rejected_with_not_found() {
        let mut request = base_request();
        request.previous_response_id = Some("resp_missing".to_string());
        let error = validate_request_compat(&request).unwrap_err();
        assert!(matches!(error, ApiError::ResponseNotFound { .. }));
        assert_eq!(error.status_code(), axum::http::StatusCode::NOT_FOUND);
    }

    #[test]
    fn store_true_without_background_is_accepted() {
        let mut request = base_request();
        request.store = Some(true);
        assert!(validate_request_compat(&request).is_ok());
    }

    #[test]
    fn prompt_template_is_rejected() {
        let mut request = base_request();
        request.prompt = Some(json!({"id": "ptmpl"}));
        let error = validate_request_compat(&request).unwrap_err();
        assert!(matches!(error, ApiError::InvalidRequest { .. }));
    }

    #[test]
    fn unknown_tool_choice_mode_is_rejected() {
        let mut request = base_request();
        request.tool_choice = Some(ResponseToolChoice::Mode("sometimes".to_string()));
        let error = validate_request_compat(&request).unwrap_err();
        assert!(matches!(error, ApiError::InvalidRequest { .. }));
    }

    #[test]
    fn unknown_truncation_value_is_rejected() {
        let mut request = base_request();
        request.truncation = Some("maybe".to_string());
        let error = validate_request_compat(&request).unwrap_err();
        assert!(matches!(error, ApiError::InvalidRequest { .. }));
    }
}
