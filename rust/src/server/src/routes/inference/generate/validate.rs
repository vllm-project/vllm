// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::types::{GenerateOutputMode, GenerateRequest};
use crate::error::{ApiError, bail_invalid_request};

/// Validate raw token-generate protocol invariants and reject unimplemented
/// request fields.
///
/// Cross-route domain rules (sampling bounds, `max_tokens`, logprobs, token
/// IDs, `cache_salt`, prompt length/truncation) belong in `vllm-text`, not
/// here.
pub(crate) fn validate_request(request: &GenerateRequest) -> Result<(), ApiError> {
    if request.output_mode == Some(GenerateOutputMode::Text)
        && request.sampling_params.detokenize == Some(false)
    {
        bail_invalid_request!(
            param = "output_mode",
            "output_mode='text' requires sampling_params.detokenize to be true"
        );
    }

    if request.token_ids.is_empty() {
        bail_invalid_request!(
            param = "token_ids",
            "token_ids must contain at least one token ID."
        );
    }

    if request.stream && request.sampling_params.inner.prompt_logprobs.is_some() {
        bail_invalid_request!(
            param = "sampling_params",
            "`prompt_logprobs` are not available when `stream=true`."
        );
    }

    if request.stream && request.sampling_params.inner.prompt_logprob_token_ids.is_some() {
        bail_invalid_request!(
            param = "sampling_params",
            "`prompt_logprob_token_ids` are not available when `stream=true`."
        );
    }

    reject_unsupported_fields(request)?;

    Ok(())
}

/// Reject request fields that are accepted for deserialization but not yet
/// implemented in the Rust frontend.
fn reject_unsupported_fields(request: &GenerateRequest) -> Result<(), ApiError> {
    if request.sampling_params.trace_decode_token_ids.is_some() {
        bail_invalid_request!(
            param = "trace_decode_token_ids",
            "trace_decode_token_ids is not supported."
        );
    }

    if request.sampling_params.reasoning_end_str.is_some() {
        bail_invalid_request!(
            param = "reasoning_end_str",
            "reasoning_end_str is not supported."
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::validate_request;
    use crate::routes::inference::generate::types::GenerateRequest;

    fn base_request() -> GenerateRequest {
        serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {}
        }))
        .expect("parse request")
    }

    #[test]
    fn validate_request_rejects_reasoning_end_str() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "token_ids": [11, 22],
            "sampling_params": {"reasoning_end_str": "Time to answer.</think>"},
        }))
        .expect("parse request");
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn validate_request_accepts_streaming() {
        let request = GenerateRequest {
            stream: true,
            ..base_request()
        };
        assert!(validate_request(&request).is_ok());
    }

    /// Python's generate route has no `stream_options` check and ignores the
    /// options when not streaming.
    #[test]
    fn validate_request_accepts_stream_options_without_streaming() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": false,
            "stream_options": {"include_usage": true},
            "sampling_params": {}
        }))
        .expect("parse request");
        validate_request(&request).expect("stream_options without streaming should be ignored");
    }

    /// Like Python, the `text` level needs the output to be detokenized.
    #[test]
    fn validate_request_rejects_text_output_mode_without_detokenize() {
        for (output_mode, detokenize, accepted) in [
            (json!(null), json!(false), true),
            (json!("tokens"), json!(false), true),
            (json!("text"), json!(null), true),
            (json!("text"), json!(true), true),
            (json!("text"), json!(false), false),
        ] {
            let request: GenerateRequest = serde_json::from_value(json!({
                "token_ids": [11, 22],
                "sampling_params": {"detokenize": detokenize},
                "output_mode": output_mode,
            }))
            .expect("parse request");
            assert_eq!(
                validate_request(&request).is_ok(),
                accepted,
                "{output_mode} {detokenize}"
            );
        }
    }

    #[test]
    fn validate_request_rejects_trace_decode_token_ids() {
        for (trace_decode_token_ids, accepted) in [(json!(null), true), (json!([7, 8]), false)] {
            let request: GenerateRequest = serde_json::from_value(json!({
                "token_ids": [11, 22],
                "sampling_params": {"trace_decode_token_ids": trace_decode_token_ids},
            }))
            .expect("parse request");
            assert_eq!(
                validate_request(&request).is_ok(),
                accepted,
                "{trace_decode_token_ids}"
            );
        }
    }

    #[test]
    fn validate_request_rejects_empty_token_ids() {
        let request = GenerateRequest {
            token_ids: Vec::new(),
            ..base_request()
        };
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn validate_request_rejects_streaming_prompt_logprobs() {
        for prompt_logprobs in [0, 1, -1] {
            let request: GenerateRequest = serde_json::from_value(json!({
                "model": "Qwen/Qwen1.5-0.5B-Chat",
                "token_ids": [11, 22],
                "stream": true,
                "sampling_params": {
                    "prompt_logprobs": prompt_logprobs
                }
            }))
            .expect("parse request");
            assert!(validate_request(&request).is_err(), "{prompt_logprobs}");
        }
    }

    #[test]
    fn validate_request_rejects_streaming_prompt_logprob_token_ids() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "token_ids": [11, 22],
            "stream": true,
            "sampling_params": {"prompt_logprob_token_ids": [[1, 2]]}
        }))
        .expect("parse request");
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn validate_request_accepts_non_stream_prompt_logprobs() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": false,
            "sampling_params": {
                "prompt_logprobs": 1
            }
        }))
        .expect("parse request");
        assert!(validate_request(&request).is_ok());
    }
}
