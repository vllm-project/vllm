use thiserror_ext::AsReport as _;
use vllm_text::tokenizer::Tokenizer;
use vllm_text::{Prompt, SamplingParams, TextDecodeOptions, TextRequest};

use super::types::CompletionRequest;
use crate::error::ApiError;
use crate::lora::LoraModelResolution;
use crate::routes::openai::completions::validate;
use crate::routes::openai::utils::structured_outputs::convert_from_response_format_value;
use crate::utils::{ResolvedRequestContext, convert_logit_bias, merge_kv_transfer_params};

/// Lowered completion request plus the public response metadata carried by
/// every SSE chunk.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PreparedRequest {
    /// Stable OpenAI-style request ID, reused as the external text request ID.
    pub request_id: String,
    /// Public model ID echoed back to the client.
    pub response_model: String,
    /// Public response rendering options for route-layer helpers.
    pub options: ResponseOptions,
    /// Lowered text request for the shared `vllm-text` facade.
    pub text_request: TextRequest,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct ResponseOptions {
    /// Whether the caller asked for the final streamed usage chunk.
    pub include_usage: bool,
    /// Whether every streamed chunk should carry cumulative usage.
    pub include_continuous_usage: bool,
    /// Whether the caller requested prompt-only echo via `max_tokens=0`.
    pub prompt_only: bool,
    /// Prompt text that should be echoed back northbound when `echo=true`.
    pub echo: Option<String>,
    /// Whether the caller requested output logprobs on completion choices.
    pub requested_logprobs: Option<u32>,
    /// Whether the caller requested choice-level prompt logprobs.
    pub include_prompt_logprobs: bool,
    /// Whether to include token IDs alongside generated text.
    pub return_token_ids: bool,
    /// Whether to format logprob tokens as `token_id:{id}`.
    pub return_tokens_as_token_ids: bool,
}

/// Validate and lower one OpenAI completions request into the internal
/// text-generation format.
///
/// `lora_resolution.model_names` must be non-empty; the first entry is used as
/// the base `model` field in responses when no LoRA adapter is selected.
pub(super) fn prepare_completion_request(
    request: CompletionRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
    tokenizer: &dyn Tokenizer,
) -> Result<PreparedRequest, ApiError> {
    validate::validate_request_compat(&request, &lora_resolution.model_names)?;

    let request_id = format!("cmpl-{}", ctx.request_id);
    let response_model = lora_resolution
        .lora_request
        .as_ref()
        .map(|request| request.lora_name.clone())
        .unwrap_or_else(|| lora_resolution.model_names.first().cloned().unwrap_or_default());

    let logprobs = match request.logprobs {
        Some(logprobs) => Some(i32::try_from(logprobs).map_err(|_| {
            ApiError::invalid_request(
                "`logprobs` must fit within a signed 32-bit integer.".to_string(),
                Some("logprobs"),
            )
        })?),
        None => None,
    };
    let prompt_only = request.echo && request.max_tokens == Some(0);
    let prompt_logprobs =
        request.prompt_logprobs.or(if request.echo && (!request.stream || prompt_only) {
            logprobs
        } else {
            None
        });
    let include_usage = (request.stream_options.as_ref())
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let include_continuous_usage = include_usage
        && request
            .stream_options
            .as_ref()
            .and_then(|options| options.continuous_usage_stats)
            .unwrap_or(false);
    let include_prompt_logprobs = prompt_logprobs.is_some();
    let max_tokens = if prompt_only {
        Some(1)
    } else {
        request.max_tokens
    };
    let echo = completion_echo_text(&request, tokenizer)?;

    let structured_outputs =
        convert_from_response_format_value(&request.response_format, &request.structured_outputs)?;

    let text_request = TextRequest {
        request_id: request_id.clone(),
        prompt: request.prompt,
        mm_features: None,
        sampling_params: SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            seed: request.seed,
            max_tokens,
            min_tokens: request.min_tokens,
            thinking_token_budget: request.thinking_token_budget,
            logprobs,
            prompt_logprobs,
            min_p: request.min_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            repetition_penalty: request.repetition_penalty,
            stop_token_ids: request.stop_token_ids,
            ignore_eos: request.ignore_eos,
            logit_bias: convert_logit_bias(request.logit_bias)?,
            allowed_token_ids: request.allowed_token_ids,
            bad_words: None,
            logprob_token_ids: None,
            structured_outputs,
            skip_reading_prefix_cache: None,
            vllm_xargs: merge_kv_transfer_params(
                request.vllm_xargs,
                request.kv_transfer_params.as_ref(),
            ),
        },
        decode_options: TextDecodeOptions {
            skip_special_tokens: request.skip_special_tokens,
            include_stop_str_in_output: request.include_stop_str_in_output,
            stop_strings: request.stop.map(|stop| stop.into_vec()),
            min_tokens: request.min_tokens.unwrap_or(0),
        },
        intermediate: request.stream,
        priority: request.priority.unwrap_or(0),
        cache_salt: request.cache_salt,
        add_special_tokens: request.add_special_tokens,
        data_parallel_rank: ctx.data_parallel_rank,
        reasoning_parser_kwargs: None,
        lora_request: lora_resolution.lora_request.clone(),
    };

    Ok(PreparedRequest {
        request_id,
        response_model,
        options: ResponseOptions {
            include_usage,
            include_continuous_usage,
            prompt_only,
            echo,
            requested_logprobs: request.logprobs,
            include_prompt_logprobs,
            return_token_ids: request.return_token_ids.unwrap_or(false),
            return_tokens_as_token_ids: request.return_tokens_as_token_ids.unwrap_or(false),
        },
        text_request,
    })
}

fn completion_echo_text(
    request: &CompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> Result<Option<String>, ApiError> {
    if !request.echo {
        return Ok(None);
    }

    match &request.prompt {
        Prompt::Text(prompt) => Ok(Some(prompt.clone())),
        Prompt::TokenIds(token_ids) if request.return_token_ids.unwrap_or(false) => {
            Ok(Some(String::new()))
        }
        Prompt::TokenIds(token_ids) => {
            tokenizer.decode(token_ids, false).map(Some).map_err(|error| {
                ApiError::invalid_request(
                    format!(
                        "Failed to decode token-ID prompt for echo: {}",
                        error.to_report_string()
                    ),
                    Some("prompt"),
                )
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderMap;
    use serde_json::json;
    use vllm_text::Prompt;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::prepare_completion_request;
    use crate::lora::LoraModelResolution;
    use crate::routes::openai::completions::types::CompletionRequest;
    use crate::utils::{ResolvedRequestContext, resolve_request_context};

    fn request_context(headers: &HeaderMap, request_id: Option<&str>) -> ResolvedRequestContext {
        resolve_request_context(headers, request_id)
    }

    fn served(names: &[&str]) -> LoraModelResolution {
        LoraModelResolution {
            model_names: names.iter().map(|s| s.to_string()).collect(),
            lora_request: None,
        }
    }

    fn test_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
    }

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

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            &test_tokenizer(),
        )
        .expect("prepare");

        assert!(prepared.options.include_usage);
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
    fn prepare_completion_request_passes_through_thinking_token_budget() {
        let prepare = |budget: serde_json::Value| {
            let request: CompletionRequest = serde_json::from_value(json!({
                "model": "Qwen/Qwen1.5-0.5B-Chat",
                "prompt": "hello",
                "thinking_token_budget": budget,
            }))
            .expect("parse request");
            prepare_completion_request(
                request,
                &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
                ResolvedRequestContext::default(),
                &test_tokenizer(),
            )
            .expect("prepare")
            .text_request
            .sampling_params
            .thinking_token_budget
        };

        // The convert layer forwards the raw value verbatim (including the `-1`
        // "unlimited" sentinel); normalization/validation happens during
        // lowering (see `vllm_text::lower`).
        assert_eq!(prepare(json!(64)), Some(64));
        assert_eq!(prepare(json!(-1)), Some(-1));
        assert_eq!(prepare(json!(null)), None);
    }

    #[test]
    fn prepare_completion_request_maps_stream_usage_and_token_format_options() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": true,
            "stream_options": {
                "include_usage": true,
                "continuous_usage_stats": true
            },
            "return_tokens_as_token_ids": true
        }))
        .expect("parse request");

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            &test_tokenizer(),
        )
        .expect("prepare");

        assert!(prepared.options.include_usage);
        assert!(prepared.options.include_continuous_usage);
        assert!(prepared.options.return_tokens_as_token_ids);
    }

    #[test]
    fn prepare_completion_request_gates_continuous_usage_on_include_usage() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": true,
            "stream_options": {
                "continuous_usage_stats": true
            }
        }))
        .expect("parse request");

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            &test_tokenizer(),
        )
        .expect("prepare");

        assert!(!prepared.options.include_usage);
        assert!(!prepared.options.include_continuous_usage);
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

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            &test_tokenizer(),
        )
        .expect("prepare");

        assert_eq!(prepared.options.echo, Some("hello".to_string()));
        assert_eq!(prepared.text_request.sampling_params.max_tokens, Some(7));
        assert!(!prepared.options.prompt_only);
    }

    #[test]
    fn prepare_completion_request_lowers_prompt_only_echo_as_one_internal_token() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": false,
            "echo": true,
            "max_tokens": 0
        }))
        .expect("parse request");

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            &test_tokenizer(),
        )
        .expect("prepare");

        assert!(prepared.options.prompt_only);
        assert_eq!(prepared.options.echo, Some("hello".to_string()));
        assert_eq!(prepared.text_request.sampling_params.max_tokens, Some(1));
    }

    #[test]
    fn prepare_completion_request_enables_prompt_logprobs_for_stream_prompt_only_echo() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "echo": true,
            "stream": true,
            "max_tokens": 0,
            "logprobs": 3
        }))
        .expect("parse request");

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            &test_tokenizer(),
        )
        .expect("prepare");

        assert!(prepared.options.prompt_only);
        assert_eq!(prepared.text_request.sampling_params.logprobs, Some(3));
        assert_eq!(
            prepared.text_request.sampling_params.prompt_logprobs,
            Some(3)
        );
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

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            &test_tokenizer(),
        )
        .expect("prepare");

        assert_eq!(prepared.text_request.sampling_params.logprobs, Some(3));
        assert_eq!(
            prepared.text_request.sampling_params.prompt_logprobs,
            Some(3)
        );
    }

    #[test]
    fn prepare_completion_request_decodes_token_id_prompt_echo() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": [104, 101, 108, 108, 111],
            "stream": true,
            "echo": true
        }))
        .expect("parse request");

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            &test_tokenizer(),
        )
        .expect("prepare");

        assert_eq!(prepared.options.echo, Some("hello".to_string()));
        assert_eq!(
            prepared.text_request.prompt,
            Prompt::TokenIds(vec![104, 101, 108, 108, 111])
        );
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

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            &test_tokenizer(),
        )
        .expect("prepare");
        assert_eq!(prepared.text_request.sampling_params.logprobs, Some(1));
        assert_eq!(
            prepared.text_request.sampling_params.prompt_logprobs,
            Some(2)
        );
    }

    #[test]
    fn prepare_completion_request_threads_data_parallel_rank() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": false,
        }))
        .expect("parse request");

        let mut headers = HeaderMap::new();
        headers.insert("X-data-parallel-rank", "3".parse().unwrap());
        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            request_context(&headers, None),
            &test_tokenizer(),
        )
        .expect("prepare");
        assert_eq!(prepared.text_request.data_parallel_rank, Some(3));
    }

    #[test]
    fn prepare_completion_request_leaves_data_parallel_rank_none_when_absent() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": false,
        }))
        .expect("parse request");

        let prepared = prepare_completion_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            &test_tokenizer(),
        )
        .expect("prepare");
        assert_eq!(prepared.text_request.data_parallel_rank, None);
    }
}
