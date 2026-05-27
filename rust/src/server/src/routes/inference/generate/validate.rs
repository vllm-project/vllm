use super::types::GenerateRequest;
use crate::error::{ApiError, bail_invalid_request};

/// Enforce the minimal compatibility contract for the Rust token generate
/// route.
pub(super) fn validate_request_compat(
    request: &GenerateRequest,
    served_model_names: &[String],
) -> Result<(), ApiError> {
    if let Some(model) = request.model.as_ref()
        && !served_model_names.iter().any(|n| n == model)
    {
        return Err(ApiError::model_not_found(model.clone()));
    }

    if request.stream {
        bail_invalid_request!(param = "stream", "stream=true is not supported.");
    }

    if request.token_ids.is_empty() {
        bail_invalid_request!(
            param = "token_ids",
            "token_ids must contain at least one token ID."
        );
    }

    if request.sampling_params.max_tokens == Some(0) {
        bail_invalid_request!(
            param = "sampling_params",
            "max_tokens must be greater than 0."
        );
    }

    if let Some(prompt_logprobs) = request.sampling_params.prompt_logprobs
        && prompt_logprobs < 0
        && prompt_logprobs != -1
    {
        bail_invalid_request!(
            param = "sampling_params",
            "`prompt_logprobs` must be a non-negative value or -1."
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::validate_request_compat;
    use crate::routes::inference::generate::types::GenerateRequest;

    fn base_request() -> GenerateRequest {
        serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {}
        }))
        .expect("parse request")
    }

    fn served(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn validate_request_compat_rejects_streaming() {
        let request = GenerateRequest {
            stream: true,
            ..base_request()
        };
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());
    }

    #[test]
    fn validate_request_compat_rejects_empty_token_ids() {
        let request = GenerateRequest {
            token_ids: Vec::new(),
            ..base_request()
        };
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());
    }
}
