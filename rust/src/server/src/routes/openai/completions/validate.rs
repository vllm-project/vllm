use vllm_text::Prompt;

use super::types::CompletionRequest;
use crate::error::{ApiError, bail_invalid_request};

/// Enforce the minimal compatibility contract for the Rust OpenAI server.
pub(super) fn validate_request_compat(
    request: &CompletionRequest,
    configured_model: &str,
) -> Result<(), ApiError> {
    // This path is intentionally scoped to the minimum surface needed by `vllm-bench` random
    // workload compatibility, so unsupported legacy completions features fail early here.
    if request.model != configured_model {
        return Err(ApiError::model_not_found(request.model.clone()));
    }

    if request.stream_options.is_some() && !request.stream {
        bail_invalid_request!(
            param = "stream_options",
            "stream_options are only supported when stream=true."
        );
    }

    if request.n.unwrap_or(1) > 1 {
        bail_invalid_request!(param = "n", "Only n=1 is supported.");
    }

    if request.max_tokens == Some(0) {
        bail_invalid_request!(param = "max_tokens", "max_tokens must be greater than 0.");
    }

    if request.echo && matches!(request.prompt, Prompt::TokenIds(_)) {
        bail_invalid_request!(
            param = "echo",
            "echo is not supported with token-ID prompts."
        );
    }

    if request.suffix.is_some() {
        bail_invalid_request!(param = "suffix", "suffix is not supported.");
    }

    if let Some(logprobs) = request.logprobs
        && logprobs > i32::MAX as u32
    {
        bail_invalid_request!(
            param = "logprobs",
            "`logprobs` must fit within a signed 32-bit integer."
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

    if request.use_beam_search {
        bail_invalid_request!(
            param = "use_beam_search",
            "use_beam_search is not supported."
        );
    }

    if request.response_format.is_some() {
        bail_invalid_request!(
            param = "response_format",
            "response_format is not supported."
        );
    }

    if request.structured_outputs.is_some() {
        bail_invalid_request!(
            param = "structured_outputs",
            "structured_outputs is not supported."
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::validate_request_compat;
    use crate::routes::openai::completions::types::CompletionRequest;

    fn base_request() -> CompletionRequest {
        serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": true,
        }))
        .expect("parse request")
    }

    #[test]
    fn validate_request_compat_accepts_logprobs() {
        let request = CompletionRequest {
            logprobs: Some(1),
            ..base_request()
        };
        assert!(validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat").is_ok());
    }

    #[test]
    fn validate_request_compat_rejects_streaming_prompt_logprobs() {
        let request = CompletionRequest {
            prompt_logprobs: Some(1),
            ..base_request()
        };
        assert!(validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }

    #[test]
    fn validate_request_compat_accepts_non_stream_prompt_logprobs() {
        let request = CompletionRequest {
            stream: false,
            prompt_logprobs: Some(-1),
            ..base_request()
        };
        assert!(validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat").is_ok());
    }
}
