use openai_protocol::common::StringOrArray;
use uuid::Uuid;
use vllm_text::{Prompt, SamplingParams, TextDecodeOptions, TextRequest};

use super::types::CompletionRequest;
use crate::error::ApiError;
use crate::routes::openai::completions::validate;
use crate::utils::{convert_logit_bias, merge_kv_transfer_params};

/// Lowered completion request plus the public response metadata carried by every SSE chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRequest {
    /// Stable OpenAI-style response ID, also reused as the internal text request ID.
    pub response_id: String,
    /// Public model ID echoed back to the client.
    pub response_model: String,
    /// Whether the caller asked for the final streamed usage chunk.
    pub include_usage: bool,
    /// Lowered text request for the shared `vllm-text` facade.
    pub text_request: TextRequest,
    /// Original text prompt that should be echoed back northbound when `echo=true`.
    pub echo: Option<String>,
}

/// Validate and lower one OpenAI completions request into the internal text-generation format.
pub fn prepare_completion_request(
    request: &CompletionRequest,
    configured_model: &str,
) -> Result<PreparedRequest, ApiError> {
    validate::validate_request_compat(request, configured_model)?;

    let logprobs = match request.logprobs {
        Some(logprobs) => Some(i32::try_from(logprobs).map_err(|_| {
            ApiError::invalid_request(
                "`logprobs` must fit within a signed 32-bit integer.".to_string(),
                Some("logprobs"),
            )
        })?),
        None => None,
    };
    let prompt_logprobs = request
        .prompt_logprobs
        .or(if request.echo && !request.stream {
            logprobs
        } else {
            None
        });

    let response_id = format!("cmpl-{}", Uuid::new_v4());
    let include_usage = (request.stream_options.as_ref())
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let echo = request.echo.then(|| match &request.prompt {
        Prompt::Text(text) => text.clone(),
        Prompt::TokenIds(_) => unreachable!("validated above"),
    });

    let text_request = TextRequest {
        request_id: response_id.clone(),
        prompt: request.prompt.clone(),
        sampling_params: SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            seed: request.seed,
            max_tokens: request.max_tokens,
            min_tokens: request.min_tokens,
            logprobs,
            prompt_logprobs,
            min_p: request.min_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            repetition_penalty: request.repetition_penalty,
            stop_token_ids: request.stop_token_ids.clone(),
            ignore_eos: request.ignore_eos,
            logit_bias: convert_logit_bias(request.logit_bias.clone())?,
            allowed_token_ids: request.allowed_token_ids.clone(),
            bad_words: None,
            vllm_xargs: merge_kv_transfer_params(
                request.vllm_xargs.clone(),
                request.kv_transfer_params.as_ref(),
            ),
        },
        decode_options: TextDecodeOptions {
            skip_special_tokens: request.skip_special_tokens,
            include_stop_str_in_output: request.include_stop_str_in_output,
            stop_strings: request.stop.as_ref().map(|stop| match stop {
                StringOrArray::String(string) => vec![string.clone()],
                StringOrArray::Array(arr) => arr.clone(),
            }),
            min_tokens: request.min_tokens.unwrap_or(0),
        },
        intermediate: request.stream,
        priority: request.priority.unwrap_or(0),
    };

    Ok(PreparedRequest {
        response_id,
        response_model: configured_model.to_string(),
        include_usage,
        text_request,
        echo,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vllm_text::Prompt;

    use super::prepare_completion_request;
    use crate::routes::openai::completions::types::CompletionRequest;

    fn base_request_json() -> serde_json::Value {
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": true
        })
    }

    #[test]
    fn completion_http_request_deserializes_text_prompt() {
        let request: CompletionRequest =
            serde_json::from_value(base_request_json()).expect("parse request");

        assert_eq!(request.prompt, Prompt::Text("hello".to_string()));
        assert_eq!(request.model, "Qwen/Qwen1.5-0.5B-Chat");
    }

    #[test]
    fn completion_http_request_deserializes_token_id_prompt() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": [11, 22, 33],
            "stream": true,
            "ignore_eos": true,
            "max_tokens": 7
        }))
        .expect("parse request");

        assert_eq!(request.prompt, Prompt::TokenIds(vec![11, 22, 33]));
        assert_eq!(request.max_tokens, Some(7));
        assert!(request.ignore_eos);
    }

    #[test]
    fn prepare_completion_request_maps_sampling_fields() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": [11, 22, 33],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 7,
            "logprobs": 2,
            "top_p": 0.9,
            "top_k": 42,
            "min_p": 0.1,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.3,
            "repetition_penalty": 1.1,
            "ignore_eos": true,
            "skip_special_tokens": false
        }))
        .expect("parse request");

        let prepared =
            prepare_completion_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("prepare");

        assert!(prepared.include_usage);
        assert_eq!(
            prepared.text_request.prompt,
            Prompt::TokenIds(vec![11, 22, 33])
        );
        assert_eq!(prepared.text_request.sampling_params.max_tokens, Some(7));
        assert_eq!(prepared.text_request.sampling_params.logprobs, Some(2));
        assert_eq!(prepared.text_request.sampling_params.top_p, Some(0.9));
        assert_eq!(prepared.text_request.sampling_params.top_k, Some(42));
        assert_eq!(prepared.text_request.sampling_params.min_p, Some(0.1));
        assert_eq!(
            prepared.text_request.sampling_params.frequency_penalty,
            Some(0.2)
        );
        assert_eq!(
            prepared.text_request.sampling_params.presence_penalty,
            Some(0.3)
        );
        assert_eq!(
            prepared.text_request.sampling_params.repetition_penalty,
            Some(1.1)
        );
        assert!(prepared.text_request.sampling_params.ignore_eos);
        assert!(!prepared.text_request.decode_options.skip_special_tokens);
    }

    #[test]
    fn prepare_completion_request_accepts_text_echo() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": true,
            "echo": true,
            "max_tokens": 7
        }))
        .expect("parse request");

        let prepared =
            prepare_completion_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("prepare");

        assert_eq!(prepared.echo, Some("hello".to_string()));
        assert_eq!(prepared.text_request.sampling_params.max_tokens, Some(7));
    }

    #[test]
    fn prepare_completion_request_enables_prompt_logprobs_for_non_stream_echo() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "echo": true,
            "stream": false,
            "logprobs": 3
        }))
        .expect("parse request");

        let prepared =
            prepare_completion_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("prepare");

        assert_eq!(prepared.text_request.sampling_params.logprobs, Some(3));
        assert_eq!(
            prepared.text_request.sampling_params.prompt_logprobs,
            Some(3)
        );
    }

    #[test]
    fn prepare_completion_request_rejects_token_id_prompt_echo() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": [11, 22, 33],
            "stream": true,
            "echo": true
        }))
        .expect("parse request");

        assert!(prepare_completion_request(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }

    #[test]
    fn prepare_completion_request_accepts_logprobs_fields() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": false,
            "logprobs": 1,
            "prompt_logprobs": 2
        }))
        .expect("parse request");

        let prepared =
            prepare_completion_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("prepare");
        assert_eq!(prepared.text_request.sampling_params.logprobs, Some(1));
        assert_eq!(
            prepared.text_request.sampling_params.prompt_logprobs,
            Some(2)
        );
    }
}
