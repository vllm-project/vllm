use std::collections::BTreeSet;

use vllm_engine_core_client::protocol::{EngineCoreSamplingParams, RequestOutputKind};
use vllm_llm::GenerateRequest;

use crate::backend::SamplingHints;
use crate::error::Result;
use crate::request::{ChatRequest, UserSamplingParams};

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
        sampling_params,
        chat_options: _,
    } = request;

    let generate_request = GenerateRequest {
        request_id: request_id.clone(),
        prompt_token_ids,
        sampling_params: lower_sampling_params(sampling_params, sampling_hints),
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

/// Convert [`UserSamplingParams`] to the lower-level [`EngineCoreSamplingParams`] with the given
/// context from the chat backend.
fn lower_sampling_params(
    sampling_params: UserSamplingParams,
    SamplingHints {
        primary_eos_token_id,
        extra_eos_token_ids,
    }: SamplingHints,
) -> EngineCoreSamplingParams {
    let UserSamplingParams {
        temperature,
        top_p,
        top_k,
        max_tokens,
        min_tokens,
        mut stop_token_ids,
        ignore_eos,
    } = sampling_params;

    let mut all_stop_token_ids = BTreeSet::from_iter(stop_token_ids.iter().copied());
    if let Some(primary_eos_token_id) = primary_eos_token_id {
        all_stop_token_ids.insert(primary_eos_token_id);
    }
    all_stop_token_ids.extend(extra_eos_token_ids.iter().copied());

    if !ignore_eos {
        merge_unique_token_ids(&mut stop_token_ids, extra_eos_token_ids.iter().copied());
    }

    EngineCoreSamplingParams {
        temperature,
        top_p,
        top_k,
        max_tokens,
        min_tokens,
        stop_token_ids,
        eos_token_id: (!ignore_eos).then_some(primary_eos_token_id).flatten(),
        all_stop_token_ids,
        output_kind: RequestOutputKind::Delta,
    }
}

fn merge_unique_token_ids(
    stop_token_ids: &mut Vec<u32>,
    extra_token_ids: impl Iterator<Item = u32>,
) {
    for token_id in extra_token_ids {
        if !stop_token_ids.contains(&token_id) {
            stop_token_ids.push(token_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::backend::SamplingHints;
    use crate::request::{ChatOptions, ChatRequest, UserSamplingParams};

    fn sample_request() -> ChatRequest {
        ChatRequest {
            request_id: "chat-1".to_string(),
            messages: vec![],
            sampling_params: UserSamplingParams::default(),
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
        assert_eq!(params.eos_token_id, Some(99));
        assert_eq!(params.stop_token_ids, vec![77]);
        assert_eq!(params.all_stop_token_ids, BTreeSet::from([77, 99]));
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
        assert_eq!(params.eos_token_id, None);
        assert_eq!(params.stop_token_ids, Vec::<u32>::new());
        assert_eq!(params.all_stop_token_ids, BTreeSet::from([77, 99]));
        assert_eq!(params.output_kind, RequestOutputKind::Delta);
    }

    #[test]
    fn lower_chat_request_preserves_explicit_stop_token_ids_in_all_stop_set() {
        let mut request = sample_request();
        request.sampling_params.stop_token_ids = vec![11, 77];

        let prepared = lower_chat_request(
            request,
            vec![1, 2, 3],
            SamplingHints {
                primary_eos_token_id: Some(99),
                extra_eos_token_ids: BTreeSet::from([77, 88]),
            },
        )
        .unwrap();

        let params = prepared.generate_request.sampling_params;
        assert_eq!(params.stop_token_ids, vec![11, 77, 88]);
        assert_eq!(params.all_stop_token_ids, BTreeSet::from([11, 77, 88, 99]));
    }
}
