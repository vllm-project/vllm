use vllm_engine_core_client::protocol::RequestOutputKind;
use vllm_llm::GenerateRequest;

use crate::backend::SamplingHints;
use crate::error::Result;
use crate::request::ChatRequest;

#[derive(Debug)]
pub(crate) struct PreparedChatRequest {
    pub request_id: String,
    pub generate_request: GenerateRequest,
}

pub(crate) fn lower_chat_request(
    request: ChatRequest,
    prompt_token_ids: Vec<u32>,
    sampling_hints: SamplingHints,
) -> Result<PreparedChatRequest> {
    let ChatRequest {
        request_id,
        messages: _,
        mut sampling_params,
        chat_options: _,
    } = request;

    // TODO: we should not expose `RequestOutputKind` at the chat layer
    sampling_params.output_kind = RequestOutputKind::Delta;
    sampling_params.apply_model_eos_token_ids(
        sampling_hints.primary_eos_token_id,
        &sampling_hints.extra_eos_token_ids,
    );

    let generate_request = GenerateRequest {
        request_id: request_id.clone(),
        prompt_token_ids,
        sampling_params,
        arrival_time: None,
        cache_salt: None,
        trace_headers: None,
        priority: 0,
        data_parallel_rank: None,
        reasoning_ended: None,
        lora_request: None,
    };

    Ok(PreparedChatRequest {
        request_id,
        generate_request,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use vllm_engine_core_client::protocol::SamplingParams;

    use super::*;
    use crate::backend::SamplingHints;
    use crate::request::{ChatOptions, ChatRequest};

    fn sample_request() -> ChatRequest {
        ChatRequest {
            request_id: "chat-1".to_string(),
            messages: vec![],
            sampling_params: SamplingParams::default(),
            chat_options: ChatOptions::default(),
        }
    }

    #[test]
    fn lower_chat_request_applies_python_style_eos_hints() {
        let prepared = lower_chat_request(
            sample_request(),
            vec![1, 2, 3],
            SamplingHints {
                primary_eos_token_id: Some(99),
                extra_eos_token_ids: BTreeSet::from([77]),
            },
        )
        .unwrap();

        let params = prepared.generate_request.sampling_params;
        assert_eq!(params._eos_token_id, Some(99));
        assert_eq!(params.stop_token_ids, vec![77]);
        assert_eq!(params._all_stop_token_ids, BTreeSet::from([77, 99]));
        assert_eq!(params.output_kind, RequestOutputKind::Delta);
    }

    #[test]
    fn lower_chat_request_respects_ignore_eos_for_stop_token_ids() {
        let mut request = sample_request();
        request.sampling_params.ignore_eos = true;

        let prepared = lower_chat_request(
            request,
            vec![1, 2, 3],
            SamplingHints {
                primary_eos_token_id: Some(99),
                extra_eos_token_ids: BTreeSet::from([77]),
            },
        )
        .unwrap();

        let params = prepared.generate_request.sampling_params;
        assert_eq!(params._eos_token_id, None);
        assert_eq!(params.stop_token_ids, Vec::<u32>::new());
        assert_eq!(params._all_stop_token_ids, BTreeSet::from([77, 99]));
        assert_eq!(params.output_kind, RequestOutputKind::Delta);
    }
}
