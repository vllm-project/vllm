use super::types::GenerateRequest;
use crate::error::{ApiError, bail_invalid_request};
use crate::routes::openai::utils::token_ids::{
    validate_allowed_token_ids, validate_logit_bias_token_ids, validate_token_ids,
};

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

    if request.stream_options.is_some() && !request.stream {
        bail_invalid_request!(
            param = "stream_options",
            "stream_options are only supported when stream=true."
        );
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

/// Reject out-of-vocab token ids, mirroring the OpenAI routes' `validate_token_id_ranges`.
pub(super) fn validate_token_id_ranges(
    request: &GenerateRequest,
    tokenizer_vocab_size: usize,
    model_vocab_size: Option<usize>,
) -> Result<(), ApiError> {
    let prompt_bound = tokenizer_vocab_size.max(model_vocab_size.unwrap_or(0));
    validate_token_ids(&request.token_ids, prompt_bound)?;
    validate_allowed_token_ids(
        request.sampling_params.allowed_token_ids.as_deref(),
        tokenizer_vocab_size,
    )?;
    validate_logit_bias_token_ids(
        request.sampling_params.logit_bias.as_ref(),
        model_vocab_size.unwrap_or(usize::MAX),
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use std::collections::HashMap;

    use super::{validate_request_compat, validate_token_id_ranges};
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
    fn validate_request_compat_accepts_streaming() {
        let request = GenerateRequest {
            stream: true,
            ..base_request()
        };
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok());
    }

    #[test]
    fn validate_request_compat_rejects_stream_options_without_streaming() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": false,
            "stream_options": {"include_usage": true},
            "sampling_params": {}
        }))
        .expect("parse request");
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

    #[test]
    fn validate_token_id_ranges_rejects_oob_token_ids_and_params() {
        let mut request = base_request();
        request.token_ids = vec![5, 150];
        assert!(validate_token_id_ranges(&request, 100, Some(200)).is_ok());
        request.token_ids = vec![5, 200];
        assert!(validate_token_id_ranges(&request, 100, Some(200)).is_err());
        // prompt bound is max(tokenizer, model): an id in [model, tokenizer) is accepted
        request.token_ids = vec![150];
        assert!(validate_token_id_ranges(&request, 200, Some(100)).is_ok());

        // allowed_token_ids is bounded by the tokenizer vocab, logit_bias by the model vocab
        let mut request = base_request();
        request.sampling_params.allowed_token_ids = Some(vec![150]);
        assert!(validate_token_id_ranges(&request, 100, Some(200)).is_err());
        request.sampling_params.allowed_token_ids = None;
        request.sampling_params.logit_bias = Some(HashMap::from([(150, 1.0)]));
        assert!(validate_token_id_ranges(&request, 200, Some(100)).is_err());

        // unknown vocab sizes skip the check
        request = base_request();
        request.token_ids = vec![1_000_000];
        assert!(validate_token_id_ranges(&request, usize::MAX, None).is_ok());
    }
}
