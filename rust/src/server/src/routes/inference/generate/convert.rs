// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_engine_core_client::protocol::multimodal::MmFeatures;
use vllm_text::{Prompt, TextDecodeOptions, TextRequest};

use super::types::{GenerateRequest, GenerateSamplingParams};
use super::validate;
use crate::error::ApiError;
use crate::lora::LoraModelResolution;
use crate::routes::openai::utils::types::StringOrArray;
use crate::utils::{ResolvedRequestContext, merge_ec_transfer_params, merge_kv_transfer_params};

/// Lowered generate request plus the response request ID.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PreparedRequest {
    pub request_id: String,
    pub text_request: TextRequest,
    pub stream: bool,
    /// Public response rendering options for route-layer helpers.
    pub options: ResponseOptions,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct ResponseOptions {
    /// Whether the caller asked for the final streamed usage chunk.
    pub include_usage: bool,
    /// Whether the caller asked for usage on every streamed chunk.
    pub include_continuous_usage: bool,
    /// Whether the caller requested output logprobs on generate choices.
    pub include_logprobs: bool,
    /// Whether the caller requested top-level prompt logprobs.
    pub include_prompt_logprobs: bool,
}

/// Validate and lower one raw generate request into the internal
/// text-generation format.
pub(super) fn prepare_generate_request(
    request: GenerateRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
    mm_features: Option<MmFeatures>,
) -> Result<PreparedRequest, ApiError> {
    validate::validate_request_compat(&request, &lora_resolution.model_names)?;

    let stream = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let include_continuous_usage = include_usage
        && request
            .stream_options
            .as_ref()
            .and_then(|options| options.continuous_usage_stats)
            .unwrap_or(false);
    let include_logprobs = request.sampling_params.sampling.logprobs.is_some();
    let include_prompt_logprobs = request.sampling_params.sampling.prompt_logprobs.is_some();

    let GenerateSamplingParams {
        sampling: mut sampling_params,
        stop,
        include_stop_str_in_output,
        skip_special_tokens,
        // Only gates `stop` (checked while validating); the Rust frontend
        // always runs the shared detokenizer.
        detokenize: _,
    } = request.sampling_params;

    sampling_params.vllm_xargs = merge_kv_transfer_params(
        sampling_params.vllm_xargs,
        request.kv_transfer_params.as_ref(),
    );
    sampling_params.vllm_xargs = merge_ec_transfer_params(
        sampling_params.vllm_xargs,
        request.ec_transfer_params.as_ref(),
    );
    let min_tokens = sampling_params.min_tokens.unwrap_or(0);

    let text_request = TextRequest {
        request_id: ctx.request_id.clone(),
        prompt: Prompt::TokenIds(request.token_ids),
        mm_features,
        sampling_params,
        decode_options: TextDecodeOptions {
            skip_special_tokens,
            include_stop_str_in_output,
            stop_strings: stop.map(StringOrArray::into_vec),
            min_tokens,
        },
        intermediate: stream,
        priority: request.priority,
        cache_salt: request.cache_salt,
        add_special_tokens: false,
        data_parallel_rank: ctx.data_parallel_rank,
        session_id: ctx.session_id,
        reasoning_parser_kwargs: None,
        lora_request: lora_resolution.lora_request.clone(),
        arrival_time: None,
    };

    Ok(PreparedRequest {
        request_id: ctx.request_id,
        text_request,
        stream,
        options: ResponseOptions {
            include_usage,
            include_continuous_usage,
            include_logprobs,
            include_prompt_logprobs,
        },
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vllm_text::Prompt;

    use super::prepare_generate_request;
    use crate::lora::LoraModelResolution;
    use crate::routes::inference::generate::types::GenerateRequest;
    use crate::utils::ResolvedRequestContext;

    fn served(names: &[&str]) -> LoraModelResolution {
        LoraModelResolution {
            model_names: names.iter().map(|s| s.to_string()).collect(),
            lora_request: None,
        }
    }

    #[test]
    fn prepare_generate_request_maps_token_prompt_and_sampling_params() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22, 33],
            "priority": -3,
            "cache_salt": "salt",
            "sampling_params": {
                "max_tokens": 7,
                "logprobs": 2,
                "prompt_logprobs": 1,
                "ignore_eos": true
            },
            "kv_transfer_params": {
                "connector": "x"
            }
        }))
        .expect("parse request");

        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            None,
        )
        .expect("prepare");

        assert_eq!(
            prepared.text_request.prompt,
            Prompt::TokenIds(vec![11, 22, 33])
        );
        assert_eq!(prepared.text_request.sampling_params.max_tokens, Some(7));
        assert_eq!(prepared.text_request.sampling_params.logprobs, Some(2));
        assert_eq!(
            prepared.text_request.sampling_params.prompt_logprobs,
            Some(1)
        );
        assert!(prepared.text_request.sampling_params.ignore_eos);
        assert_eq!(prepared.text_request.priority, -3);
        assert_eq!(prepared.text_request.cache_salt.as_deref(), Some("salt"));
        assert_eq!(
            prepared
                .text_request
                .sampling_params
                .vllm_xargs
                .and_then(|mut xargs| xargs.remove("kv_transfer_params")),
            Some(json!({"connector": "x"}))
        );
    }

    #[test]
    fn prepare_generate_request_lowers_decode_options() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {
                "stop": ["END", "STOP"],
                "include_stop_str_in_output": true,
                "skip_special_tokens": false,
                "min_tokens": 4
            }
        }))
        .expect("parse request");

        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            None,
        )
        .expect("prepare");

        let decode_options = prepared.text_request.decode_options;
        assert_eq!(
            decode_options.stop_strings,
            Some(vec!["END".to_string(), "STOP".to_string()])
        );
        assert!(decode_options.include_stop_str_in_output);
        assert!(!decode_options.skip_special_tokens);
        assert_eq!(decode_options.min_tokens, 4);
    }

    #[test]
    fn prepare_generate_request_normalizes_scalar_stop() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {"stop": "END"}
        }))
        .expect("parse request");

        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            None,
        )
        .expect("prepare");

        assert_eq!(
            prepared.text_request.decode_options.stop_strings,
            Some(vec!["END".to_string()])
        );
    }

    #[test]
    fn prepare_generate_request_defaults_decode_options_like_python() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {}
        }))
        .expect("parse request");

        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            None,
        )
        .expect("prepare");

        let decode_options = prepared.text_request.decode_options;
        assert_eq!(decode_options.stop_strings, None);
        assert!(!decode_options.include_stop_str_in_output);
        assert!(decode_options.skip_special_tokens);
        assert_eq!(decode_options.min_tokens, 0);
    }

    #[test]
    fn prepare_generate_request_keeps_integer_logit_bias_keys() {
        // `logit_bias` has integer keys, which serde's `flatten` buffering
        // cannot deserialize; the sampling params are split by hand instead.
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {"logit_bias": {"123": 1.5}, "stop": ["END"]}
        }))
        .expect("parse request");

        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            None,
        )
        .expect("prepare");

        assert_eq!(
            prepared
                .text_request
                .sampling_params
                .logit_bias
                .and_then(|bias| bias.get(&123).copied()),
            Some(1.5)
        );
    }

    #[test]
    fn prepare_generate_request_streams_incrementally() {
        // Streaming responses need intermediate decoded events; a non-stream
        // request only observes the terminal one.
        for (stream, intermediate) in [(true, true), (false, false)] {
            let request: GenerateRequest = serde_json::from_value(json!({
                "model": "Qwen/Qwen1.5-0.5B-Chat",
                "token_ids": [11, 22],
                "stream": stream,
                "sampling_params": {}
            }))
            .expect("parse request");

            let prepared = prepare_generate_request(
                request,
                &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
                ResolvedRequestContext::default(),
                None,
            )
            .expect("prepare");

            assert_eq!(prepared.text_request.intermediate, intermediate);
        }
    }

    #[test]
    fn prepare_generate_request_forwards_thinking_token_budget() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22, 33],
            "sampling_params": {
                "thinking_token_budget": 64
            }
        }))
        .expect("parse request");

        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            None,
        )
        .expect("prepare");

        // The raw inference route shares `vllm_text::SamplingParams`, so the
        // field is carried through to lowering exactly like the OpenAI routes
        // (normalization/validation then happens in `lower_sampling_params`).
        assert_eq!(
            prepared.text_request.sampling_params.thinking_token_budget,
            Some(64)
        );
    }

    #[test]
    fn prepare_generate_request_gates_continuous_usage_on_include_usage() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": true,
            "stream_options": {
                "continuous_usage_stats": true
            },
            "sampling_params": {}
        }))
        .expect("parse request");

        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            None,
        )
        .expect("prepare");

        assert!(!prepared.options.include_usage);
        assert!(!prepared.options.include_continuous_usage);
    }
}
