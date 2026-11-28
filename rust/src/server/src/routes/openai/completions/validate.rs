// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::Value;

use super::types::{CompletionPrompt, CompletionRequest};
use crate::error::{ApiError, bail_invalid_request};

/// Validate OpenAI completions protocol invariants and reject unimplemented
/// request fields.
///
/// Cross-route domain rules (sampling bounds, `max_tokens`, logprobs, token
/// IDs, `cache_salt`, prompt length/truncation) belong in `vllm-text`, not
/// here.
pub(super) fn validate_request(request: &CompletionRequest) -> Result<(), ApiError> {
    if request.stream_options.as_ref().is_some_and(|options| !options.is_empty()) && !request.stream
    {
        bail_invalid_request!(
            param = "stream_options",
            "stream_options are only supported when stream=true."
        );
    }

    if request.max_tokens == Some(0) && !request.echo {
        bail_invalid_request!(
            param = "max_tokens",
            "max_tokens=0 is only supported when echo=true."
        );
    }

    if request.stream && request.prompt_logprobs.is_some_and(|n| n > 0 || n == -1) {
        bail_invalid_request!(
            param = "prompt_logprobs",
            "`prompt_logprobs` are not available when `stream=true`."
        );
    }

    if request.logprob_token_ids.as_ref().is_some_and(|ids| !ids.is_empty()) {
        if request.logprobs.is_none() {
            bail_invalid_request!(
                param = "logprob_token_ids",
                "when using `logprob_token_ids`, `logprobs` must be set."
            );
        }
        if request.echo && request.max_tokens == Some(0) {
            bail_invalid_request!(
                param = "logprob_token_ids",
                "`logprob_token_ids` is not supported when `echo=True` and \
                 `max_tokens=0` because no output tokens are generated."
            );
        }
    }

    let batch_size = match &request.prompt {
        Some(CompletionPrompt::Texts(texts)) => Some(texts.len()),
        Some(CompletionPrompt::TokenIds(token_ids)) => Some(token_ids.len()),
        Some(CompletionPrompt::Single(_)) | None => None,
    };
    let max_prompts = max_completion_prompts();
    if let Some(prompt_count) = batch_size
        && prompt_count > max_prompts
    {
        bail_invalid_request!(
            param = "prompt",
            "prompt list length {prompt_count} exceeds the maximum allowed count of \
             {max_prompts}. To increase this limit, set the \
             VLLM_MAX_COMPLETION_PROMPTS environment variable."
        );
    }

    reject_unsupported_fields(request)?;

    Ok(())
}

/// Reject request fields that are accepted for deserialization but not yet
/// implemented in the Rust frontend.
fn reject_unsupported_fields(request: &CompletionRequest) -> Result<(), ApiError> {
    // Python's prompt-presence validator treats an empty `prompt_embeds` list
    // like an absent one; lowering ignores it.
    if (request.prompt_embeds.as_ref())
        .is_some_and(|embeds| !matches!(embeds, Value::Array(embeds) if embeds.is_empty()))
    {
        bail_invalid_request!(param = "prompt_embeds", "prompt_embeds is not supported.");
    }

    if request.use_beam_search {
        bail_invalid_request!(
            param = "use_beam_search",
            "use_beam_search is not supported."
        );
    }

    // Python's default is 1.0, which SDKs may send explicitly.
    if request.length_penalty.is_some_and(|penalty| penalty != 1.0) {
        bail_invalid_request!(param = "length_penalty", "length_penalty is not supported.");
    }

    Ok(())
}

fn max_completion_prompts() -> usize {
    parse_max_completion_prompts(std::env::var("VLLM_MAX_COMPLETION_PROMPTS").ok().as_deref())
}

/// Python reads the value with `int()`, which ignores surrounding whitespace.
fn parse_max_completion_prompts(value: Option<&str>) -> usize {
    value.and_then(|value| value.trim().parse().ok()).unwrap_or(1024)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{parse_max_completion_prompts, validate_request};
    use crate::routes::openai::completions::types::CompletionRequest;

    fn base_request() -> CompletionRequest {
        serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": true,
        }))
        .expect("parse request")
    }

    fn non_streaming_request(stream_options: serde_json::Value) -> CompletionRequest {
        serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": false,
            "stream_options": stream_options,
        }))
        .expect("parse request")
    }

    #[test]
    fn validate_request_accepts_empty_stream_options_without_streaming() {
        let request = non_streaming_request(json!({}));
        validate_request(&request).expect("empty stream_options sets nothing");
    }

    #[test]
    fn validate_request_rejects_stream_options_without_streaming() {
        let cases = [
            json!({"include_usage": false}),
            json!({"continuous_usage_stats": true}),
        ];
        for stream_options in cases {
            let request = non_streaming_request(stream_options.clone());
            assert!(validate_request(&request).is_err(), "{stream_options}");
        }
    }

    #[test]
    fn validate_request_accepts_spaces_between_special_tokens_false() {
        let request = CompletionRequest {
            spaces_between_special_tokens: false,
            ..base_request()
        };
        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn validate_request_accepts_logprobs() {
        let request = CompletionRequest {
            logprobs: Some(1),
            ..base_request()
        };
        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn validate_request_accepts_full_vocab_logprobs() {
        let request = CompletionRequest {
            logprobs: Some(-1),
            ..base_request()
        };
        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn validate_request_rejects_only_non_default_length_penalty() {
        let request = CompletionRequest {
            length_penalty: Some(1.0),
            ..base_request()
        };
        assert!(validate_request(&request).is_ok());

        let request = CompletionRequest {
            length_penalty: Some(2.0),
            ..base_request()
        };
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn validate_request_rejects_prompt_embeds() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "prompt": "hello",
            "prompt_embeds": "AAAA",
        }))
        .expect("parse request");
        assert!(validate_request(&request).is_err());

        let request: CompletionRequest = serde_json::from_value(json!({
            "prompt": "hello",
            "prompt_embeds": null,
        }))
        .expect("parse request");
        assert!(validate_request(&request).is_ok());

        let request: CompletionRequest = serde_json::from_value(json!({
            "prompt": "hello",
            "prompt_embeds": [],
        }))
        .expect("parse request");
        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn validate_request_rejects_streaming_prompt_logprobs() {
        let request = CompletionRequest {
            prompt_logprobs: Some(1),
            ..base_request()
        };
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn validate_request_accepts_non_stream_prompt_logprobs() {
        let request = CompletionRequest {
            stream: false,
            prompt_logprobs: Some(-1),
            ..base_request()
        };
        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn validate_request_accepts_prompt_only_echo() {
        let request = CompletionRequest {
            stream: false,
            echo: true,
            max_tokens: Some(0),
            ..base_request()
        };
        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn validate_request_accepts_token_id_prompt_echo() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": [104, 101, 108, 108, 111],
            "stream": true,
            "echo": true,
        }))
        .expect("parse request");

        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn validate_request_rejects_prompt_only_without_echo() {
        let request = CompletionRequest {
            stream: false,
            echo: false,
            max_tokens: Some(0),
            ..base_request()
        };
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn max_completion_prompts_parses_like_python_int() {
        for (value, expected) in [
            (None, 1024),
            (Some("2"), 2),
            (Some("2 "), 2),
            (Some(" 2\n"), 2),
            (Some("not a number"), 1024),
        ] {
            assert_eq!(parse_max_completion_prompts(value), expected, "{value:?}");
        }
    }
}
