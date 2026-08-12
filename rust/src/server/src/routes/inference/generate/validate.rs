// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::types::GenerateRequest;
use crate::error::{ApiError, bail_invalid_request};

/// Enforce the minimal compatibility contract for the Rust token generate
/// route.
pub(crate) fn validate_request_compat(
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

    if request.sampling_params.sampling.max_tokens == Some(0) {
        bail_invalid_request!(
            param = "sampling_params",
            "max_tokens must be greater than 0."
        );
    }

    // Both rules mirror `SamplingParams._verify_args`, in Python's order: the
    // empty-string check runs first, so an empty stop reports that rather than
    // the detokenize error.
    if let Some(stop) = request.sampling_params.stop.as_ref()
        && stop.as_slice().iter().any(String::is_empty)
    {
        bail_invalid_request!(
            param = "sampling_params",
            "stop cannot contain an empty string."
        );
    }

    // The guard is on the normalized list being non-empty, so `stop: null` and
    // `stop: []` stay legal with `detokenize: false`.
    if !request.sampling_params.detokenize
        && request
            .sampling_params
            .stop
            .as_ref()
            .is_some_and(|stop| !stop.as_slice().is_empty())
    {
        bail_invalid_request!(
            param = "sampling_params",
            "stop strings are only supported when detokenize is True. \
             Set detokenize=True to use stop."
        );
    }

    if let Some(prompt_logprobs) = request.sampling_params.sampling.prompt_logprobs {
        if prompt_logprobs < 0 && prompt_logprobs != -1 {
            bail_invalid_request!(
                param = "sampling_params",
                "`prompt_logprobs` must be a non-negative value or -1."
            );
        }

        if request.stream {
            bail_invalid_request!(
                param = "sampling_params",
                "`prompt_logprobs` are not available when `stream=true`."
            );
        }
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
    fn validate_request_compat_accepts_stop_with_default_detokenize() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {"stop": ["END"]}
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok());
    }

    #[test]
    fn validate_request_compat_rejects_stop_without_detokenize() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {"detokenize": false, "stop": ["END"]}
        }))
        .expect("parse request");
        let error = validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"]))
            .expect_err("stop without detokenize should be rejected");
        assert_eq!(
            error.to_error_response().error.message,
            "stop strings are only supported when detokenize is True. \
             Set detokenize=True to use stop."
        );
    }

    #[test]
    fn validate_request_compat_accepts_empty_stop_without_detokenize() {
        // Python only raises when the normalized stop list is non-empty.
        for stop in [json!(null), json!([])] {
            let request: GenerateRequest = serde_json::from_value(json!({
                "model": "Qwen/Qwen1.5-0.5B-Chat",
                "token_ids": [11, 22],
                "sampling_params": {"detokenize": false, "stop": stop}
            }))
            .expect("parse request");
            assert!(
                validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok()
            );
        }
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
    fn validate_request_compat_rejects_streaming_prompt_logprobs() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": true,
            "sampling_params": {
                "prompt_logprobs": 0
            }
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());

        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": true,
            "sampling_params": {
                "prompt_logprobs": 1
            }
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());

        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": true,
            "sampling_params": {
                "prompt_logprobs": -1
            }
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());
    }

    #[test]
    fn validate_request_compat_accepts_non_stream_prompt_logprobs() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": false,
            "sampling_params": {
                "prompt_logprobs": 1
            }
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok());
    }
}
