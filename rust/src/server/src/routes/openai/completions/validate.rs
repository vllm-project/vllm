// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::types::CompletionRequest;
use crate::error::{ApiError, bail_invalid_request};
use crate::model_names::ServedModelNames;
use crate::routes::openai::utils::validate_generation_prompt_truncation;

/// Enforce the minimal compatibility contract for the Rust OpenAI server.
pub(super) fn validate_request_compat(
    request: &CompletionRequest,
    served_model_names: &ServedModelNames,
) -> Result<(), ApiError> {
    // This path is intentionally scoped to the minimum surface needed by
    // `vllm-bench` random workload compatibility, so unsupported legacy
    // completions features fail early here.
    served_model_names.check(request.model.as_deref())?;

    if request.stream_options.as_ref().is_some_and(|options| !options.is_empty()) && !request.stream
    {
        bail_invalid_request!(
            param = "stream_options",
            "stream_options are only supported when stream=true."
        );
    }

    if request.n.unwrap_or(1) > 1 {
        bail_invalid_request!(param = "n", "Only n=1 is supported.");
    }

    if request.max_tokens == Some(0) && !request.echo {
        bail_invalid_request!(
            param = "max_tokens",
            "max_tokens=0 is only supported when echo=true."
        );
    }

    if request.suffix.is_some() {
        bail_invalid_request!(param = "suffix", "suffix is not supported.");
    }

    if request.prompt_embeds.is_some() {
        bail_invalid_request!(param = "prompt_embeds", "prompt_embeds is not supported.");
    }

    if let Some(logprobs) = request.logprobs
        && logprobs < 0
        && logprobs != -1
    {
        bail_invalid_request!(
            param = "logprobs",
            "`logprobs` must be a non-negative value or -1."
        );
    }

    if let Some(prompt_logprobs) = request.prompt_logprobs {
        if request.stream && (prompt_logprobs > 0 || prompt_logprobs == -1) {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "`prompt_logprobs` are not available when `stream=true`."
            );
        }

        if prompt_logprobs < 0 && prompt_logprobs != -1 {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "`prompt_logprobs` must be a non-negative value or -1."
            );
        }
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

    if request.use_beam_search {
        bail_invalid_request!(
            param = "use_beam_search",
            "use_beam_search is not supported."
        );
    }

    // ---- Reject parameters that are accepted for deserialization but not yet
    // implemented ----

    // Python's default is 1.0, which SDKs may send explicitly.
    if request.length_penalty.is_some_and(|penalty| penalty != 1.0) {
        bail_invalid_request!(param = "length_penalty", "length_penalty is not supported.");
    }
    if !request.spaces_between_special_tokens {
        bail_invalid_request!(
            param = "spaces_between_special_tokens",
            "spaces_between_special_tokens is not supported."
        );
    }
    validate_generation_prompt_truncation(request.truncate_prompt_tokens, request.echo)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::validate_request_compat;
    use crate::model_names::ServedModelNames;
    use crate::routes::openai::completions::types::CompletionRequest;

    fn base_request() -> CompletionRequest {
        serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": true,
        }))
        .expect("parse request")
    }

    fn served_names(names: &[&str]) -> ServedModelNames {
        ServedModelNames::new(names.iter().map(|s| s.to_string()).collect())
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
    fn validate_request_compat_accepts_empty_stream_options_without_streaming() {
        let request = non_streaming_request(json!({}));
        validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"]))
            .expect("empty stream_options sets nothing");
    }

    #[test]
    fn validate_request_compat_rejects_stream_options_without_streaming() {
        let cases = [
            json!({"include_usage": false}),
            json!({"continuous_usage_stats": true}),
        ];
        for stream_options in cases {
            let request = non_streaming_request(stream_options.clone());
            assert!(
                validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"]))
                    .is_err(),
                "{stream_options}"
            );
        }
    }

    #[test]
    fn validate_request_compat_accepts_logprobs() {
        let request = CompletionRequest {
            logprobs: Some(1),
            ..base_request()
        };
        assert!(
            validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok()
        );
    }

    #[test]
    fn validate_request_compat_accepts_full_vocab_logprobs() {
        let request = CompletionRequest {
            logprobs: Some(-1),
            ..base_request()
        };
        assert!(
            validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok()
        );
    }

    #[test]
    fn validate_request_compat_rejects_other_negative_logprobs() {
        let request = CompletionRequest {
            logprobs: Some(-2),
            ..base_request()
        };
        assert!(
            validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err()
        );
    }

    #[test]
    fn validate_request_compat_rejects_only_non_default_length_penalty() {
        let request = CompletionRequest {
            length_penalty: Some(1.0),
            ..base_request()
        };
        assert!(
            validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok()
        );

        let request = CompletionRequest {
            length_penalty: Some(2.0),
            ..base_request()
        };
        assert!(
            validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err()
        );
    }

    #[test]
    fn validate_request_compat_rejects_prompt_embeds() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "prompt": "hello",
            "prompt_embeds": "AAAA",
        }))
        .expect("parse request");
        assert!(
            validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err()
        );

        let request: CompletionRequest = serde_json::from_value(json!({
            "prompt": "hello",
            "prompt_embeds": null,
        }))
        .expect("parse request");
        assert!(
            validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok()
        );
    }

    #[test]
    fn validate_request_compat_accepts_any_served_name() {
        let request = base_request();
        assert!(
            validate_request_compat(
                &request,
                &served_names(&["other-alias", "Qwen/Qwen1.5-0.5B-Chat"])
            )
            .is_ok()
        );
    }

    #[test]
    fn validate_request_compat_accepts_default_model_inputs() {
        for model in [None, Some(String::new())] {
            let request = CompletionRequest {
                model,
                ..base_request()
            };
            assert!(validate_request_compat(&request, &served_names(&["served-model"])).is_ok());
        }
    }

    #[test]
    fn validate_request_compat_rejects_unknown_model() {
        let request = base_request();
        assert!(validate_request_compat(&request, &served_names(&["other-model"])).is_err());
    }

    #[test]
    fn validate_request_compat_rejects_streaming_prompt_logprobs() {
        let request = CompletionRequest {
            prompt_logprobs: Some(1),
            ..base_request()
        };
        assert!(
            validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err()
        );
    }

    #[test]
    fn validate_request_compat_accepts_non_stream_prompt_logprobs() {
        let request = CompletionRequest {
            stream: false,
            prompt_logprobs: Some(-1),
            ..base_request()
        };
        assert!(
            validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok()
        );
    }

    #[test]
    fn validate_request_compat_accepts_prompt_only_echo() {
        let request = CompletionRequest {
            stream: false,
            echo: true,
            max_tokens: Some(0),
            ..base_request()
        };
        assert!(
            validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok()
        );
    }

    #[test]
    fn validate_request_compat_accepts_token_id_prompt_echo() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": [104, 101, 108, 108, 111],
            "stream": true,
            "echo": true,
        }))
        .expect("parse request");

        assert!(
            validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok()
        );
    }

    #[test]
    fn validate_request_compat_rejects_prompt_only_without_echo() {
        let request = CompletionRequest {
            stream: false,
            echo: false,
            max_tokens: Some(0),
            ..base_request()
        };
        assert!(
            validate_request_compat(&request, &served_names(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err()
        );
    }
}
