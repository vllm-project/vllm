// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::types::{GenerateOutputMode, GenerateRequest};
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

    if request.sampling_params.n.unwrap_or(1) != 1 {
        bail_invalid_request!(param = "n", "Only n=1 is supported.");
    }

    // TODO: support `output_mode: "text"`.
    if request.output_mode == Some(GenerateOutputMode::Text) {
        bail_invalid_request!(
            param = "output_mode",
            "Only output_mode=\"tokens\" is supported."
        );
    }

    // TODO: support stop strings, which need incremental detokenization.
    if request
        .sampling_params
        .stop
        .as_ref()
        .is_some_and(|stop| !stop.as_slice().is_empty())
    {
        bail_invalid_request!(param = "stop", "stop is not supported.");
    }
    if request.sampling_params.include_stop_str_in_output == Some(true) {
        bail_invalid_request!(
            param = "include_stop_str_in_output",
            "include_stop_str_in_output is not supported."
        );
    }

    if request.sampling_params.trace_decode_token_ids.is_some() {
        bail_invalid_request!(
            param = "trace_decode_token_ids",
            "trace_decode_token_ids is not supported."
        );
    }

    if request.token_ids.is_empty() {
        bail_invalid_request!(
            param = "token_ids",
            "token_ids must contain at least one token ID."
        );
    }

    if request.sampling_params.inner.max_tokens == Some(0) {
        bail_invalid_request!(
            param = "sampling_params",
            "max_tokens must be greater than 0."
        );
    }

    if let Some(prompt_logprobs) = request.sampling_params.inner.prompt_logprobs {
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

    if request.stream && request.sampling_params.inner.prompt_logprob_token_ids.is_some() {
        bail_invalid_request!(
            param = "sampling_params",
            "`prompt_logprob_token_ids` are not available when `stream=true`."
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
    fn validate_request_compat_accepts_streaming() {
        let request = GenerateRequest {
            stream: true,
            ..base_request()
        };
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok());
    }

    /// Python's generate route has no `stream_options` check and ignores the
    /// options when not streaming.
    #[test]
    fn validate_request_compat_accepts_stream_options_without_streaming() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": false,
            "stream_options": {"include_usage": true},
            "sampling_params": {}
        }))
        .expect("parse request");
        validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"]))
            .expect("stream_options without streaming should be ignored");
    }

    #[test]
    fn validate_request_compat_rejects_parallel_sampling() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {"n": 4}
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());
    }

    #[test]
    fn validate_request_compat_rejects_text_output_mode() {
        for (output_mode, accepted) in [
            (json!(null), true),
            (json!("tokens"), true),
            (json!("text"), false),
        ] {
            let request: GenerateRequest = serde_json::from_value(json!({
                "token_ids": [11, 22],
                "sampling_params": {},
                "output_mode": output_mode,
            }))
            .expect("parse request");
            assert_eq!(
                validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok(),
                accepted,
                "{output_mode}"
            );
        }
    }

    #[test]
    fn validate_request_compat_rejects_stop_strings() {
        for (sampling_params, accepted) in [
            (
                json!({"stop": null, "include_stop_str_in_output": false}),
                true,
            ),
            (json!({"stop": []}), true),
            (json!({"stop": "\n"}), false),
            (json!({"stop": ["\n"]}), false),
            (json!({"include_stop_str_in_output": true}), false),
        ] {
            let request: GenerateRequest = serde_json::from_value(json!({
                "token_ids": [11, 22],
                "sampling_params": sampling_params,
            }))
            .expect("parse request");
            assert_eq!(
                validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok(),
                accepted,
                "{sampling_params}"
            );
        }
    }

    #[test]
    fn validate_request_compat_rejects_trace_decode_token_ids() {
        for (trace_decode_token_ids, accepted) in [(json!(null), true), (json!([7, 8]), false)] {
            let request: GenerateRequest = serde_json::from_value(json!({
                "token_ids": [11, 22],
                "sampling_params": {"trace_decode_token_ids": trace_decode_token_ids},
            }))
            .expect("parse request");
            assert_eq!(
                validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok(),
                accepted,
                "{trace_decode_token_ids}"
            );
        }
    }

    #[test]
    fn validate_request_compat_accepts_explicit_n_one() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {"n": 1}
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok());
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
    fn validate_request_compat_rejects_streaming_prompt_logprob_token_ids() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "token_ids": [11, 22],
            "stream": true,
            "sampling_params": {"prompt_logprob_token_ids": [[1, 2]]}
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["test-model"])).is_err());
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
