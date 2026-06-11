use vllm_text::Prompt;

use super::types::CompletionRequest;
use crate::error::{ApiError, bail_invalid_request};
use crate::routes::openai::utils::token_ids::{
    validate_allowed_token_ids, validate_logit_bias, validate_prompt_token_ids,
};

/// Enforce the minimal compatibility contract for the Rust OpenAI server.
pub(super) fn validate_request_compat(
    request: &CompletionRequest,
    served_model_names: &[String],
) -> Result<(), ApiError> {
    // This path is intentionally scoped to the minimum surface needed by
    // `vllm-bench` random workload compatibility, so unsupported legacy
    // completions features fail early here.
    if !served_model_names.iter().any(|n| n == &request.model) {
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

    // ---- Reject parameters that are accepted for deserialization but not yet
    // implemented ----

    if request.length_penalty.is_some() {
        bail_invalid_request!(param = "length_penalty", "length_penalty is not supported.");
    }
    if !request.spaces_between_special_tokens {
        bail_invalid_request!(
            param = "spaces_between_special_tokens",
            "spaces_between_special_tokens is not supported."
        );
    }
    if request.truncate_prompt_tokens.is_some() {
        bail_invalid_request!(
            param = "truncate_prompt_tokens",
            "truncate_prompt_tokens is not supported."
        );
    }

    Ok(())
}

/// Reject out-of-vocab token ids, mirroring the Python input processor. A token-id
/// prompt may reference ids the engine embeds beyond either vocab alone (Qwen3
/// extra LM tokens, multimodal placeholders), so it is bounded by the union of the
/// tokenizer and model vocabularies; `allowed_token_ids` by the tokenizer vocab;
/// `logit_bias` keys by the model vocab (skipped when the model size is unknown).
pub(super) fn validate_token_id_ranges(
    request: &CompletionRequest,
    tokenizer_vocab_size: usize,
    model_vocab_size: Option<usize>,
) -> Result<(), ApiError> {
    let prompt_bound = tokenizer_vocab_size.max(model_vocab_size.unwrap_or(0));
    validate_prompt_token_ids(&request.prompt, prompt_bound)?;
    validate_allowed_token_ids(request.allowed_token_ids.as_deref(), tokenizer_vocab_size)?;
    validate_logit_bias(
        request.logit_bias.as_ref(),
        model_vocab_size.unwrap_or(usize::MAX),
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vllm_text::Prompt;

    use super::{validate_request_compat, validate_token_id_ranges};
    use crate::routes::openai::completions::types::CompletionRequest;

    #[test]
    fn validate_token_id_ranges_rejects_oob_prompt_and_params() {
        // a token-id prompt below both vocabs is accepted (the engine can embed it)
        let mut request = base_request();
        request.prompt = Prompt::TokenIds(vec![5, 150]);
        assert!(validate_token_id_ranges(&request, 100, Some(200)).is_ok());
        // an id at or above the union of the two vocabs is rejected
        let mut request = base_request();
        request.prompt = Prompt::TokenIds(vec![5, 200]);
        assert!(validate_token_id_ranges(&request, 100, Some(200)).is_err());
        // an id beyond the model vocab but within the (larger) tokenizer vocab is
        // accepted: the engine embeds added/placeholder ids above the model vocab,
        // matching the Python input processor's max(tokenizer, model) bound
        let mut request = base_request();
        request.prompt = Prompt::TokenIds(vec![150]);
        assert!(validate_token_id_ranges(&request, 200, Some(100)).is_ok());
        // falls back to the tokenizer vocab when the model size is unknown
        let mut request = base_request();
        request.prompt = Prompt::TokenIds(vec![150]);
        assert!(validate_token_id_ranges(&request, 100, None).is_err());
        // allowed_token_ids are bounded by the tokenizer vocab -> reject
        let mut request = base_request();
        request.allowed_token_ids = Some(vec![150]);
        assert!(validate_token_id_ranges(&request, 100, Some(200)).is_err());
        // unknown sizes -> skip
        let mut request = base_request();
        request.prompt = Prompt::TokenIds(vec![1_000_000]);
        assert!(validate_token_id_ranges(&request, usize::MAX, None).is_ok());
    }

    fn base_request() -> CompletionRequest {
        serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": true,
        }))
        .expect("parse request")
    }

    fn served_names(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
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
}
