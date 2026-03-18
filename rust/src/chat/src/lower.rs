use std::collections::BTreeSet;

use vllm_engine_core_client::protocol::{EngineCoreSamplingParams, RequestOutputKind};
use vllm_llm::GenerateRequest;

use crate::backend::SamplingHints;
use crate::error::Result;
use crate::request::{ChatRequest, UserSamplingParams};

#[derive(Debug)]
pub(crate) struct PreparedChatRequest {
    pub chat_request: ChatRequest,
    pub generate_request: GenerateRequest,
}

/// Convert a high-level [`ChatRequest`] into a lower-level [`GenerateRequest`] ready to be sent to
/// the `llm` crate, enriching the user sampling parameters with tokenizer/model-derived hints from
/// the chat backend as needed.
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
        tools: _,
        tool_choice: _,
    } = &request;

    let generate_request = GenerateRequest {
        request_id: request_id.clone(),
        prompt_token_ids,
        sampling_params: lower_sampling_params(sampling_params.clone(), sampling_hints),
        arrival_time: None,
        cache_salt: None,
        trace_headers: None,
        priority: 0,
        data_parallel_rank: None,
        reasoning_ended: None,
        lora_request: None,
    };

    Ok(PreparedChatRequest {
        chat_request: request.clone(),
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
        default_temperature,
        default_top_p,
        default_top_k,
        default_min_p,
        default_repetition_penalty,
        default_max_tokens,
    }: SamplingHints,
) -> EngineCoreSamplingParams {
    let UserSamplingParams {
        temperature,
        top_p,
        top_k,
        seed,
        max_tokens,
        min_tokens,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        include_stop_str_in_output: _,
        stop_token_ids,
        ignore_eos,
        skip_special_tokens: _,
    } = sampling_params;

    // Mirrors the model-generation-config inheritance used by vLLM's OpenAI chat path:
    // https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/entrypoints/openai/chat_completion/protocol.py#L424-L450
    let temperature = temperature.or(default_temperature).unwrap_or(1.0);
    let top_p = top_p.or(default_top_p).unwrap_or(1.0);
    let top_k = top_k.or(default_top_k).unwrap_or(0);
    // TODO: resolve `max_tokens` against prompt length / max model length later, like vLLM's
    // OpenAI frontend does today.
    let max_tokens = max_tokens.or(default_max_tokens).unwrap_or(65536);
    let min_p = min_p.or(default_min_p).unwrap_or(0.0);
    let repetition_penalty = repetition_penalty
        .or(default_repetition_penalty)
        .unwrap_or(1.0);

    let min_tokens = min_tokens.unwrap_or(0);
    let frequency_penalty = frequency_penalty.unwrap_or(0.0);
    let presence_penalty = presence_penalty.unwrap_or(0.0);

    let mut stop_token_ids = stop_token_ids.unwrap_or_default();

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
        seed,
        max_tokens,
        min_tokens,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
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
    use crate::backend::{ChatBackend, SamplingHints};
    use crate::backends::hf::HfChatBackend;
    use crate::request::{ChatOptions, ChatRequest, ChatToolChoice, UserSamplingParams};

    fn sample_request() -> ChatRequest {
        ChatRequest {
            request_id: "chat-1".to_string(),
            messages: vec![],
            sampling_params: UserSamplingParams::default(),
            chat_options: ChatOptions::default(),
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
        }
    }

    fn sample_sampling_hints() -> SamplingHints {
        SamplingHints {
            primary_eos_token_id: Some(99),
            extra_eos_token_ids: BTreeSet::from([77]),
            default_temperature: None,
            default_top_p: None,
            default_top_k: None,
            default_min_p: None,
            default_repetition_penalty: None,
            default_max_tokens: None,
        }
    }

    #[test]
    fn lower_chat_request_applies_python_style_eos_hints() {
        let prepared =
            lower_chat_request(sample_request(), vec![1, 2, 3], sample_sampling_hints()).unwrap();

        let params = prepared.generate_request.sampling_params;
        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
                max_tokens: 65536,
                min_tokens: 0,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                stop_token_ids: [
                    77,
                ],
                eos_token_id: Some(
                    99,
                ),
                all_stop_token_ids: {
                    77,
                    99,
                },
                output_kind: Delta,
            }
        "#]]
        .assert_debug_eq(&params);
    }

    #[test]
    fn lower_chat_request_respects_ignore_eos_for_stop_token_ids() {
        let mut request = sample_request();
        request.sampling_params.ignore_eos = true;

        let prepared = lower_chat_request(request, vec![1, 2, 3], sample_sampling_hints()).unwrap();

        let params = prepared.generate_request.sampling_params;
        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
                max_tokens: 65536,
                min_tokens: 0,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                stop_token_ids: [],
                eos_token_id: None,
                all_stop_token_ids: {
                    77,
                    99,
                },
                output_kind: Delta,
            }
        "#]]
        .assert_debug_eq(&params);
    }

    #[test]
    fn lower_chat_request_preserves_explicit_stop_token_ids_in_all_stop_set() {
        let mut request = sample_request();
        request.sampling_params.stop_token_ids = Some(vec![11, 77]);

        let prepared = lower_chat_request(
            request,
            vec![1, 2, 3],
            SamplingHints {
                primary_eos_token_id: Some(99),
                extra_eos_token_ids: BTreeSet::from([77, 88]),
                default_temperature: None,
                default_top_p: None,
                default_top_k: None,
                default_min_p: None,
                default_repetition_penalty: None,
                default_max_tokens: None,
            },
        )
        .unwrap();

        let params = prepared.generate_request.sampling_params;
        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
                max_tokens: 65536,
                min_tokens: 0,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                stop_token_ids: [
                    11,
                    77,
                    88,
                ],
                eos_token_id: Some(
                    99,
                ),
                all_stop_token_ids: {
                    11,
                    77,
                    88,
                    99,
                },
                output_kind: Delta,
            }
        "#]]
        .assert_debug_eq(&params);
    }

    #[test]
    fn lower_chat_request_prefers_user_values_over_generation_defaults() {
        let mut request = sample_request();
        request.sampling_params.temperature = Some(0.2);
        request.sampling_params.top_p = Some(0.3);
        request.sampling_params.top_k = Some(4);
        request.sampling_params.max_tokens = Some(32);
        request.sampling_params.min_tokens = Some(2);

        let prepared = lower_chat_request(
            request,
            vec![1, 2, 3],
            SamplingHints {
                primary_eos_token_id: None,
                extra_eos_token_ids: BTreeSet::new(),
                default_temperature: Some(0.8),
                default_top_p: Some(0.9),
                default_top_k: Some(12),
                default_min_p: Some(0.1),
                default_repetition_penalty: Some(1.2),
                default_max_tokens: Some(128),
            },
        )
        .unwrap();

        let params = prepared.generate_request.sampling_params;
        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 0.2,
                top_p: 0.3,
                top_k: 4,
                seed: None,
                max_tokens: 32,
                min_tokens: 2,
                min_p: 0.1,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.2,
                stop_token_ids: [],
                eos_token_id: None,
                all_stop_token_ids: {},
                output_kind: Delta,
            }
        "#]]
        .assert_debug_eq(&params);
    }

    #[test]
    fn lower_chat_request_uses_generation_defaults_when_user_omits_values() {
        let prepared = lower_chat_request(
            sample_request(),
            vec![1, 2, 3],
            SamplingHints {
                primary_eos_token_id: None,
                extra_eos_token_ids: BTreeSet::new(),
                default_temperature: Some(0.8),
                default_top_p: Some(0.9),
                default_top_k: Some(12),
                default_min_p: Some(0.1),
                default_repetition_penalty: Some(1.2),
                default_max_tokens: Some(128),
            },
        )
        .unwrap();

        let params = prepared.generate_request.sampling_params;
        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 0.8,
                top_p: 0.9,
                top_k: 12,
                seed: None,
                max_tokens: 128,
                min_tokens: 0,
                min_p: 0.1,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.2,
                stop_token_ids: [],
                eos_token_id: None,
                all_stop_token_ids: {},
                output_kind: Delta,
            }
        "#]]
        .assert_debug_eq(&params);
    }

    #[tokio::test]
    #[ignore = "requires network access to Hugging Face"]
    async fn lower_chat_request_uses_real_qwen_generation_defaults() {
        let backend = HfChatBackend::from_model("Qwen/Qwen3-0.6B")
            .await
            .expect("load qwen tokenizer and generation config");
        let hints = backend.sampling_hints().expect("collect sampling hints");

        expect_test::expect![[r#"
            SamplingHints {
                primary_eos_token_id: Some(
                    151645,
                ),
                extra_eos_token_ids: {
                    151643,
                },
                default_temperature: Some(
                    0.6,
                ),
                default_top_p: Some(
                    0.95,
                ),
                default_top_k: Some(
                    20,
                ),
                default_min_p: Some(
                    0.1,
                ),
                default_repetition_penalty: Some(
                    1.2,
                ),
                default_max_tokens: None,
            }
        "#]]
        .assert_debug_eq(&hints);

        let prepared =
            lower_chat_request(sample_request(), vec![1, 2, 3], hints).expect("lower request");
        let params = prepared.generate_request.sampling_params;

        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 0.6,
                top_p: 0.95,
                top_k: 20,
                seed: None,
                max_tokens: 65536,
                min_tokens: 0,
                min_p: 0.1,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.2,
                stop_token_ids: [
                    151643,
                ],
                eos_token_id: Some(
                    151645,
                ),
                all_stop_token_ids: {
                    151643,
                    151645,
                },
                output_kind: Delta,
            }
        "#]]
        .assert_debug_eq(&params);
    }
}
