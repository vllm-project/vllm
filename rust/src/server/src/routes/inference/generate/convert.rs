use vllm_text::{Prompt, TextDecodeOptions, TextRequest};

use super::types::GenerateRequest;
use super::validate;
use crate::error::ApiError;
use crate::utils::{ResolvedRequestContext, merge_kv_transfer_params};

/// Lowered generate request plus the response request ID.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRequest {
    pub request_id: String,
    pub text_request: TextRequest,
    pub include_logprobs: bool,
    pub include_prompt_logprobs: bool,
}

/// Validate and lower one raw generate request into the internal text-generation format.
pub fn prepare_generate_request(
    request: GenerateRequest,
    configured_model: &str,
    ctx: ResolvedRequestContext,
) -> Result<PreparedRequest, ApiError> {
    validate::validate_request_compat(&request, configured_model)?;

    let include_logprobs = request.sampling_params.logprobs.is_some();
    let include_prompt_logprobs = request.sampling_params.prompt_logprobs.is_some();
    let mut sampling_params = request.sampling_params;
    sampling_params.vllm_xargs = merge_kv_transfer_params(
        sampling_params.vllm_xargs,
        request.kv_transfer_params.as_ref(),
    );

    let text_request = TextRequest {
        request_id: ctx.request_id.clone(),
        prompt: Prompt::TokenIds(request.token_ids),
        sampling_params,
        decode_options: TextDecodeOptions::default(),
        intermediate: false,
        priority: request.priority,
        cache_salt: request.cache_salt,
        add_special_tokens: false,
        data_parallel_rank: ctx.data_parallel_rank,
    };

    Ok(PreparedRequest {
        request_id: ctx.request_id,
        text_request,
        include_logprobs,
        include_prompt_logprobs,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vllm_text::Prompt;

    use super::prepare_generate_request;
    use crate::routes::inference::generate::types::GenerateRequest;
    use crate::utils::ResolvedRequestContext;

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
            "Qwen/Qwen1.5-0.5B-Chat",
            ResolvedRequestContext::default(),
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
}
