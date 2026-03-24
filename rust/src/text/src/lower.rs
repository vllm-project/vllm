use std::collections::BTreeSet;

use vllm_engine_core_client::protocol::{EngineCoreSamplingParams, RequestOutputKind};
use vllm_llm::GenerateRequest;

use crate::backend::SamplingHints;
use crate::error::{Error, Result};
use crate::request::{SamplingParams, TextRequest};

/// One text request after it has been lowered into the raw generate boundary.
#[derive(Debug)]
pub struct PreparedTextRequest {
    /// The original high-level request, preserved for response-side metadata and decoding options.
    pub text_request: TextRequest,
    /// The southbound request ready to be sent to `vllm-llm`.
    pub generate_request: GenerateRequest,
}

/// Convert a high-level [`TextRequest`] into one lower-level [`GenerateRequest`] ready for the
/// `llm` crate.
pub fn lower_text_request(
    request: TextRequest,
    prompt_token_ids: Vec<u32>,
    sampling_hints: SamplingHints,
) -> Result<PreparedTextRequest> {
    let prompt_len = prompt_token_ids.len() as u32;
    let intermediate = request.intermediate;
    let generate_request = GenerateRequest {
        request_id: request.request_id.clone(),
        prompt_token_ids,
        sampling_params: lower_sampling_params(
            request.sampling_params.clone(),
            sampling_hints,
            prompt_len,
            intermediate,
        )?,
        // Fields below are currently placeholders.
        arrival_time: None,
        cache_salt: None,
        trace_headers: None,
        priority: 0,
        data_parallel_rank: None,
        reasoning_ended: None,
        lora_request: None,
    };

    Ok(PreparedTextRequest {
        text_request: request,
        generate_request,
    })
}

/// Convert [`SamplingParams`] into [`EngineCoreSamplingParams`], enriching omitted user values with
/// tokenizer/model-derived hints when available.
pub fn lower_sampling_params(
    sampling_params: SamplingParams,
    SamplingHints {
        primary_eos_token_id,
        extra_eos_token_ids,
        default_temperature,
        default_top_p,
        default_top_k,
        default_min_p,
        default_repetition_penalty,
        default_max_tokens,
        max_model_len,
    }: SamplingHints,
    prompt_len: u32,
    intermediate: bool,
) -> Result<EngineCoreSamplingParams> {
    let SamplingParams {
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
        ignore_eos,
    } = sampling_params;

    // Mirrors the model-generation-config inheritance used by vLLM's OpenAI chat path:
    // https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/entrypoints/openai/chat_completion/protocol.py#L424-L450
    let temperature = temperature.or(default_temperature).unwrap_or(1.0);
    let top_p = top_p.or(default_top_p).unwrap_or(1.0);
    let top_k = top_k.or(default_top_k).unwrap_or(0);
    let min_p = min_p.or(default_min_p).unwrap_or(0.0);
    let repetition_penalty = repetition_penalty
        .or(default_repetition_penalty)
        .unwrap_or(1.0);
    let max_tokens = resolve_max_tokens(max_tokens, default_max_tokens, max_model_len, prompt_len)?;
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

    Ok(EngineCoreSamplingParams {
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
        // Only request intermediate deltas if the caller explicitly opts in.
        output_kind: if intermediate {
            RequestOutputKind::Delta
        } else {
            RequestOutputKind::FinalOnly
        },
    })
}

/// Resolve the effective `max_tokens` for generation, mirroring vLLM Python's `get_max_tokens()`
/// in `vllm/entrypoints/utils.py`.
///
/// Takes the minimum of all available limits (user-specified, generation-config default, and
/// `max_model_len - prompt_len`). When nothing is known, falls back to `u32::MAX` so the
/// engine-core can apply its own context-window limit.
pub fn resolve_max_tokens(
    user_max_tokens: Option<u32>,
    default_max_tokens: Option<u32>,
    max_model_len: Option<u32>,
    prompt_len: u32,
) -> Result<u32> {
    let model_max_tokens = match max_model_len {
        Some(max_model_len) if prompt_len >= max_model_len => {
            return Err(Error::PromptTooLong {
                max_model_len,
                prompt_len,
            });
        }
        Some(max_model_len) => Some(max_model_len - prompt_len),
        None => None,
    };

    let fallback_max_tokens = user_max_tokens.or(default_max_tokens);
    Ok([fallback_max_tokens, model_max_tokens]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(u32::MAX /* TODO: a reasonable fallback? */))
}

fn merge_unique_token_ids(
    stop_token_ids: &mut Vec<u32>,
    extra_token_ids: impl Iterator<Item = u32>,
) {
    // Keep user-provided ordering stable while still folding in backend-derived EOS aliases.
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
    use crate::backend::{SamplingHints, TextBackend as _};
    use crate::backends::hf::HfTextBackend;
    use crate::request::{Prompt, TextRequest};

    fn sample_request() -> TextRequest {
        TextRequest {
            request_id: "text-1".to_string(),
            prompt: Prompt::TokenIds(vec![1, 2, 3]),
            sampling_params: SamplingParams::default(),
            decode_options: Default::default(),
            intermediate: true,
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
            max_model_len: None,
        }
    }

    #[test]
    fn lower_text_request_applies_python_style_eos_hints() {
        let prepared =
            lower_text_request(sample_request(), vec![1, 2, 3], sample_sampling_hints()).unwrap();

        let params = prepared.generate_request.sampling_params;
        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
                max_tokens: 4294967295,
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
    fn lower_text_request_respects_ignore_eos_for_stop_token_ids() {
        let mut request = sample_request();
        request.sampling_params.ignore_eos = true;

        let prepared = lower_text_request(request, vec![1, 2, 3], sample_sampling_hints()).unwrap();

        let params = prepared.generate_request.sampling_params;
        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
                max_tokens: 4294967295,
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

    #[tokio::test]
    #[ignore = "requires network access to Hugging Face"]
    async fn lower_text_request_uses_real_qwen_generation_defaults() {
        let backend = HfTextBackend::from_model("Qwen/Qwen3-0.6B")
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
                max_model_len: Some(
                    40960,
                ),
            }
        "#]]
        .assert_debug_eq(&hints);

        let prepared =
            lower_text_request(sample_request(), vec![1, 2, 3], hints).expect("lower request");
        let params = prepared.generate_request.sampling_params;

        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 0.6,
                top_p: 0.95,
                top_k: 20,
                seed: None,
                max_tokens: 40957,
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

    #[test]
    fn lower_sampling_params_preserves_explicit_stop_token_ids_in_all_stop_set() {
        let mut sampling_params = SamplingParams::default();
        sampling_params.stop_token_ids = Some(vec![11, 77]);

        let params = lower_sampling_params(
            sampling_params,
            SamplingHints {
                primary_eos_token_id: Some(99),
                extra_eos_token_ids: BTreeSet::from([77, 88]),
                default_temperature: None,
                default_top_p: None,
                default_top_k: None,
                default_min_p: None,
                default_repetition_penalty: None,
                default_max_tokens: None,
                max_model_len: None,
            },
            3,
            true,
        )
        .unwrap();

        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
                max_tokens: 4294967295,
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
    fn lower_sampling_params_prefers_user_values_over_generation_defaults() {
        let sampling_params = SamplingParams {
            temperature: Some(0.2),
            top_p: Some(0.3),
            top_k: Some(4),
            max_tokens: Some(32),
            min_tokens: Some(2),
            ..Default::default()
        };

        let params = lower_sampling_params(
            sampling_params,
            SamplingHints {
                primary_eos_token_id: None,
                extra_eos_token_ids: BTreeSet::new(),
                default_temperature: Some(0.8),
                default_top_p: Some(0.9),
                default_top_k: Some(12),
                default_min_p: Some(0.1),
                default_repetition_penalty: Some(1.2),
                default_max_tokens: Some(128),
                max_model_len: None,
            },
            3,
            true,
        )
        .unwrap();

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
    fn lower_sampling_params_uses_generation_defaults_when_user_omits_values() {
        let params = lower_sampling_params(
            SamplingParams::default(),
            SamplingHints {
                primary_eos_token_id: None,
                extra_eos_token_ids: BTreeSet::new(),
                default_temperature: Some(0.8),
                default_top_p: Some(0.9),
                default_top_k: Some(12),
                default_min_p: Some(0.1),
                default_repetition_penalty: Some(1.2),
                default_max_tokens: Some(128),
                max_model_len: None,
            },
            3,
            true,
        )
        .unwrap();

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

    #[test]
    fn resolve_max_tokens_caps_by_model_len() {
        let result = resolve_max_tokens(Some(150), None, Some(200), 100);
        assert_eq!(result.unwrap(), 100);
    }

    #[test]
    fn lower_sampling_params_uses_final_only_when_intermediate_is_false() {
        let params = lower_sampling_params(
            SamplingParams::default(),
            SamplingHints {
                primary_eos_token_id: None,
                extra_eos_token_ids: BTreeSet::new(),
                default_temperature: None,
                default_top_p: None,
                default_top_k: None,
                default_min_p: None,
                default_repetition_penalty: None,
                default_max_tokens: None,
                max_model_len: None,
            },
            3,
            false,
        )
        .unwrap();

        assert_eq!(params.output_kind, RequestOutputKind::FinalOnly);
    }

    #[test]
    fn resolve_max_tokens_user_smaller_than_model_limit() {
        let result = resolve_max_tokens(Some(50), None, Some(200), 100);
        assert_eq!(result.unwrap(), 50);
    }

    #[test]
    fn resolve_max_tokens_uses_default_when_user_omits() {
        let result = resolve_max_tokens(None, Some(64), Some(200), 100);
        assert_eq!(result.unwrap(), 64);
    }

    #[test]
    fn resolve_max_tokens_default_capped_by_model_len() {
        let result = resolve_max_tokens(None, Some(256), Some(200), 100);
        assert_eq!(result.unwrap(), 100);
    }

    #[test]
    fn resolve_max_tokens_no_model_len_falls_back() {
        let result = resolve_max_tokens(Some(9999), None, None, 100);
        assert_eq!(result.unwrap(), 9999);
    }

    #[test]
    fn resolve_max_tokens_no_limits_known_falls_back_to_u32_max() {
        let result = resolve_max_tokens(None, None, None, 100);
        assert_eq!(result.unwrap(), u32::MAX);
    }

    #[test]
    fn resolve_max_tokens_prompt_too_long() {
        let result = resolve_max_tokens(Some(10), None, Some(100), 100);
        assert!(matches!(
            result,
            Err(Error::PromptTooLong {
                max_model_len: 100,
                prompt_len: 100,
            })
        ));
    }

    #[test]
    fn resolve_max_tokens_prompt_exceeds_model_len() {
        let result = resolve_max_tokens(Some(10), None, Some(100), 200);
        assert!(matches!(
            result,
            Err(Error::PromptTooLong {
                max_model_len: 100,
                prompt_len: 200,
            })
        ));
    }
}
