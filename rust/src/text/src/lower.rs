use std::collections::BTreeSet;

pub(crate) mod logprobs;
pub(crate) mod token_ids;

use vllm_engine_core_client::protocol::EngineCoreSamplingParams;
use vllm_llm::GenerateRequest;
use vllm_tokenizer::Tokenizer;

use crate::backend::{SamplingHints, SamplingLimits};
use crate::error::{Error, Result};
use crate::request::{SamplingParams, TextRequest};
use logprobs::validate_logprobs;
use token_ids::{validate_prompt_token_ids, validate_vocab_range};

/// One text request after it has been lowered into the raw generate boundary.
#[derive(Debug)]
pub struct PreparedTextRequest {
    /// The original high-level request, preserved for response-side metadata
    /// and decoding options.
    pub text_request: TextRequest,
    /// The southbound request ready to be sent to `vllm-llm`.
    pub generate_request: GenerateRequest,
}

/// Convert a high-level [`TextRequest`] into one lower-level
/// [`GenerateRequest`] ready for the `llm` crate.
pub fn lower_text_request(
    request: TextRequest,
    prompt_token_ids: Vec<u32>,
    sampling_hints: SamplingHints,
    sampling_limits: SamplingLimits,
    tokenizer: &dyn Tokenizer,
) -> Result<PreparedTextRequest> {
    let prompt_len = prompt_token_ids.len() as u32;
    validate_prompt_token_ids(&prompt_token_ids, &sampling_limits)?;

    let generate_request = GenerateRequest {
        request_id: request.request_id.clone(),
        prompt_token_ids,
        mm_features: request.mm_features.clone(),
        sampling_params: lower_sampling_params(
            request.sampling_params.clone(),
            sampling_hints,
            sampling_limits,
            prompt_len,
            tokenizer,
        )?,
        cache_salt: request.cache_salt.clone(),
        priority: request.priority,
        data_parallel_rank: request.data_parallel_rank,
        reasoning_parser_kwargs: request.reasoning_parser_kwargs.clone(),
        lora_request: request.lora_request.clone(),
        arrival_time: None,
        trace_headers: None,
    };

    Ok(PreparedTextRequest {
        text_request: request,
        generate_request,
    })
}

/// Convert [`SamplingParams`] into [`EngineCoreSamplingParams`], enriching
/// omitted user values with tokenizer/model-derived hints when available.
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
    }: SamplingHints,
    sampling_limits: SamplingLimits,
    prompt_len: u32,
    tokenizer: &dyn Tokenizer,
) -> Result<EngineCoreSamplingParams> {
    let SamplingParams {
        temperature,
        top_p,
        top_k,
        seed,
        max_tokens,
        min_tokens,
        thinking_token_budget,
        logprobs,
        prompt_logprobs,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        repetition_detection,
        stop_token_ids,
        ignore_eos,
        logit_bias,
        allowed_token_ids,
        bad_words,
        logprob_token_ids,
        structured_outputs,
        skip_reading_prefix_cache,
        vllm_xargs,
    } = sampling_params;

    validate_logprobs(
        logprobs,
        prompt_logprobs,
        logprob_token_ids.as_deref(),
        sampling_limits,
    )?;

    // Mirrors the model-generation-config inheritance used by vLLM's OpenAI chat
    // path: https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/entrypoints/openai/chat_completion/protocol.py#L424-L450
    // If neither the caller nor the model provides a value, fall back to 1.0 — the
    // default used by the Python vLLM OpenAI-compatible API (via
    // `_DEFAULT_SAMPLING_PARAMS`).
    let temperature = temperature.or(default_temperature).unwrap_or(1.0);
    let top_p = top_p.or(default_top_p).unwrap_or(1.0);
    let top_k = top_k.or(default_top_k).unwrap_or(0);
    let min_p = min_p.or(default_min_p).unwrap_or(0.0);
    let repetition_penalty = repetition_penalty.or(default_repetition_penalty).unwrap_or(1.0);
    let max_tokens = resolve_max_tokens(
        max_tokens,
        default_max_tokens,
        sampling_limits.max_model_len,
        prompt_len,
    )?;
    let min_tokens = min_tokens.unwrap_or(0);
    if min_tokens > max_tokens {
        return Err(Error::MinTokensExceedsMaxTokens {
            min_tokens,
            max_tokens,
        });
    }
    let thinking_token_budget = normalize_thinking_token_budget(thinking_token_budget)?;
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

    let params = EngineCoreSamplingParams {
        temperature,
        top_p,
        top_k,
        seed,
        max_tokens,
        min_tokens,
        thinking_token_budget,
        logprobs,
        prompt_logprobs,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        repetition_detection: if repetition_detection.as_ref().is_some_and(|p| p.is_disabled()) {
            None
        } else {
            repetition_detection
        },
        stop_token_ids,
        eos_token_id: (!ignore_eos).then_some(primary_eos_token_id).flatten(),
        all_stop_token_ids,
        logit_bias,
        allowed_token_ids,
        bad_words_token_ids: tokenize_bad_words(bad_words.as_deref(), tokenizer)?,
        // TODO: Validate structured-output schemas and regexes before submitting requests to engine-core.
        structured_outputs,
        logprob_token_ids,
        skip_reading_prefix_cache,
        extra_args: vllm_xargs,
    };
    validate_vocab_range(&params, &sampling_limits)?;
    Ok(params)
}

/// Normalize the user-facing `thinking_token_budget` into the engine value.
///
/// Mirrors Python's `validate_thinking_token_budget`
/// (<https://github.com/vllm-project/vllm/blob/ecf9d83520eb217401b47d8a5451a27c5231b8c2/vllm/sampling_params.py#L35-L55>):
/// `None` and the `-1` "unlimited" sentinel both map to `None`; any other
/// negative value is rejected; non-negative values pass through unchanged. Like
/// Python's `int`, no upper bound is imposed.
fn normalize_thinking_token_budget(value: Option<i64>) -> Result<Option<u64>> {
    match value {
        None | Some(-1) => Ok(None),
        Some(budget) if budget >= 0 => Ok(Some(budget as u64)),
        Some(_) => Err(Error::InvalidThinkingTokenBudget),
    }
}

/// Convert bad-word strings into token-ID sequences, following the Python vLLM
/// logic in `SamplingParams.update_from_tokenizer()`.
///
/// Each word is encoded both with and without a leading space so that the ban
/// applies regardless of whether the word appears at the beginning or in the
/// middle of generated text (this accounts for tokenizers that use an
/// `add_prefix_space` convention).
///
/// Reference: <https://github.com/vllm-project/vllm/blob/f22d6e026/vllm/sampling_params.py#L555-L594>
fn tokenize_bad_words(
    bad_words: Option<&[String]>,
    tokenizer: &dyn Tokenizer,
) -> Result<Option<Vec<Vec<u32>>>> {
    let bad_words = bad_words.filter(|w| !w.is_empty());
    let mut all_token_ids = Vec::new();

    for bad_word in bad_words.into_iter().flatten() {
        // Without a leading space we always keep the encoding.
        // With a leading space we only keep it when the prefix-space variant produces a
        // distinct first token but the same sequence length — this mirrors the Python
        // dedup condition that avoids redundant entries.
        let without_space = tokenizer.encode(bad_word, false)?;
        let with_space = tokenizer.encode(&format!(" {}", bad_word.trim_start()), false)?;

        if !without_space.is_empty() {
            all_token_ids.push(without_space);
        }
        if !with_space.is_empty()
            && all_token_ids.last().is_some_and(|prev: &Vec<u32>| {
                with_space[0] != prev[0] && with_space.len() == prev.len()
            })
        {
            all_token_ids.push(with_space);
        }
    }

    Ok((!all_token_ids.is_empty()).then_some(all_token_ids))
}

/// Resolve the effective `max_tokens` for generation, mirroring vLLM Python's
/// `get_max_tokens()` in `vllm/entrypoints/utils.py`.
///
/// Takes the minimum of all available limits: user-specified, generation-config
/// default, and `max_model_len - prompt_len`.
pub fn resolve_max_tokens(
    user_max_tokens: Option<u32>,
    default_max_tokens: Option<u32>,
    max_model_len: u32,
    prompt_len: u32,
) -> Result<u32> {
    let model_max_tokens = if prompt_len >= max_model_len {
        return Err(Error::PromptTooLong {
            max_model_len,
            prompt_len,
        });
    } else {
        max_model_len - prompt_len
    };

    let request_max_tokens = user_max_tokens.or(default_max_tokens);
    Ok(request_max_tokens.map_or(model_max_tokens, |n| n.min(model_max_tokens)))
}

fn merge_unique_token_ids(
    stop_token_ids: &mut Vec<u32>,
    extra_token_ids: impl Iterator<Item = u32>,
) {
    // Keep user-provided ordering stable while still folding in backend-derived EOS
    // aliases.
    for token_id in extra_token_ids {
        if !stop_token_ids.contains(&token_id) {
            stop_token_ids.push(token_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use serial_test::file_serial;

    use super::*;
    use crate::backend::hf::HfTextBackend;
    use crate::backend::{SamplingHints, TextBackend as _};
    use crate::error::{LogprobsError, TokenIdsError};
    use crate::request::{Prompt, TextRequest};

    /// Stub tokenizer that returns empty token IDs — sufficient for tests that
    /// don't exercise bad-words tokenization.
    struct StubTokenizer;

    impl Tokenizer for StubTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_special_tokens: bool,
        ) -> vllm_tokenizer::Result<Vec<u32>> {
            Ok(vec![])
        }

        fn decode(
            &self,
            _token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> vllm_tokenizer::Result<String> {
            Ok(String::new())
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }
    }

    fn stub_tokenizer() -> StubTokenizer {
        StubTokenizer
    }

    struct FixedTokenizer {
        token_ids: Vec<u32>,
    }

    impl Tokenizer for FixedTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_special_tokens: bool,
        ) -> vllm_tokenizer::Result<Vec<u32>> {
            Ok(self.token_ids.clone())
        }

        fn decode(
            &self,
            _token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> vllm_tokenizer::Result<String> {
            Ok(String::new())
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }
    }

    fn sample_request() -> TextRequest {
        TextRequest {
            prompt: Prompt::TokenIds(vec![1, 2, 3]),
            request_id: "text-1".to_string(),
            ..TextRequest::for_test()
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

    fn sample_sampling_limits() -> SamplingLimits {
        SamplingLimits {
            max_model_len: 1_000_000,
            max_logprobs: SamplingLimits::DEFAULT_MAX_LOGPROBS,
            model_vocab_size: 1000,
            tokenizer_vocab_size: 2000,
        }
    }

    fn lower_sampling_params_with_limits(
        sampling_params: SamplingParams,
        sampling_limits: SamplingLimits,
    ) -> Result<EngineCoreSamplingParams> {
        lower_sampling_params(
            sampling_params,
            SamplingHints {
                primary_eos_token_id: None,
                extra_eos_token_ids: BTreeSet::new(),
                default_temperature: None,
                default_top_p: None,
                default_top_k: None,
                default_min_p: None,
                default_repetition_penalty: None,
                default_max_tokens: None,
            },
            sampling_limits,
            3,
            &stub_tokenizer(),
        )
    }

    #[test]
    fn lower_sampling_params_normalizes_thinking_token_budget() {
        let lower = |budget: Option<i64>| {
            lower_sampling_params_with_limits(
                SamplingParams {
                    thinking_token_budget: budget,
                    ..SamplingParams::default()
                },
                sample_sampling_limits(),
            )
        };

        // Non-negative budgets (including 0) pass through unchanged.
        assert_eq!(lower(Some(256)).unwrap().thinking_token_budget, Some(256));
        assert_eq!(lower(Some(0)).unwrap().thinking_token_budget, Some(0));
        // `None` and the `-1` "unlimited" sentinel both disable the budget.
        assert_eq!(lower(None).unwrap().thinking_token_budget, None);
        assert_eq!(lower(Some(-1)).unwrap().thinking_token_budget, None);
        // No upper bound is imposed, matching Python's `int`.
        assert_eq!(
            lower(Some(i64::from(u32::MAX) + 1)).unwrap().thinking_token_budget,
            Some(u64::from(u32::MAX) + 1)
        );
        // Other negatives are rejected.
        assert!(matches!(
            lower(Some(-2)),
            Err(Error::InvalidThinkingTokenBudget)
        ));
    }

    #[test]
    fn lower_sampling_params_rejects_min_tokens_above_resolved_max_tokens() {
        let error = lower_sampling_params_with_limits(
            SamplingParams {
                max_tokens: Some(4),
                min_tokens: Some(5),
                ..SamplingParams::default()
            },
            sample_sampling_limits(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::MinTokensExceedsMaxTokens {
                min_tokens: 5,
                max_tokens: 4,
            }
        ));
    }

    #[test]
    fn lower_text_request_applies_python_style_eos_hints() {
        let prepared = lower_text_request(
            sample_request(),
            vec![1, 2, 3],
            sample_sampling_hints(),
            sample_sampling_limits(),
            &stub_tokenizer(),
        )
        .unwrap();

        let params = prepared.generate_request.sampling_params;
        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
                max_tokens: 999997,
                min_tokens: 0,
                thinking_token_budget: None,
                logprobs: None,
                prompt_logprobs: None,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                repetition_detection: None,
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
                logit_bias: None,
                allowed_token_ids: None,
                bad_words_token_ids: None,
                structured_outputs: None,
                logprob_token_ids: None,
                skip_reading_prefix_cache: None,
                extra_args: None,
            }
        "#]]
        .assert_debug_eq(&params);
    }

    #[test]
    fn lower_text_request_respects_ignore_eos_for_stop_token_ids() {
        let mut request = sample_request();
        request.sampling_params.ignore_eos = true;

        let prepared = lower_text_request(
            request,
            vec![1, 2, 3],
            sample_sampling_hints(),
            sample_sampling_limits(),
            &stub_tokenizer(),
        )
        .unwrap();

        let params = prepared.generate_request.sampling_params;
        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
                max_tokens: 999997,
                min_tokens: 0,
                thinking_token_budget: None,
                logprobs: None,
                prompt_logprobs: None,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                repetition_detection: None,
                stop_token_ids: [],
                eos_token_id: None,
                all_stop_token_ids: {
                    77,
                    99,
                },
                logit_bias: None,
                allowed_token_ids: None,
                bad_words_token_ids: None,
                structured_outputs: None,
                logprob_token_ids: None,
                skip_reading_prefix_cache: None,
                extra_args: None,
            }
        "#]]
        .assert_debug_eq(&params);
    }

    #[test]
    fn lower_text_request_uses_union_vocab_for_prompt_token_ids() {
        lower_text_request(
            sample_request(),
            vec![1500],
            sample_sampling_hints(),
            SamplingLimits {
                model_vocab_size: 2000,
                tokenizer_vocab_size: 1000,
                ..sample_sampling_limits()
            },
            &stub_tokenizer(),
        )
        .expect("model vocab extends prompt token range");

        lower_text_request(
            sample_request(),
            vec![1500],
            sample_sampling_hints(),
            SamplingLimits {
                model_vocab_size: 1000,
                tokenizer_vocab_size: 2000,
                ..sample_sampling_limits()
            },
            &stub_tokenizer(),
        )
        .expect("tokenizer vocab extends prompt token range");

        let error = lower_text_request(
            sample_request(),
            vec![2000],
            sample_sampling_hints(),
            SamplingLimits {
                model_vocab_size: 1000,
                tokenizer_vocab_size: 2000,
                ..sample_sampling_limits()
            },
            &stub_tokenizer(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::TokenIds(TokenIdsError::OutOfVocab {
                parameter: "prompt",
                token_ids,
                vocab_size: 2000,
            }) if token_ids == vec![2000]
        ));
    }

    #[tokio::test]
    #[file_serial(hf_qwen3)]
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
                default_min_p: None,
                default_repetition_penalty: None,
                default_max_tokens: None,
            }
        "#]]
        .assert_debug_eq(&hints);

        let prepared = lower_text_request(
            sample_request(),
            vec![1, 2, 3],
            hints,
            SamplingLimits {
                max_model_len: 40960,
                max_logprobs: SamplingLimits::DEFAULT_MAX_LOGPROBS,
                model_vocab_size: backend.model_vocab_size(),
                tokenizer_vocab_size: backend.tokenizer_vocab_size(),
            },
            &stub_tokenizer(),
        )
        .expect("lower request");
        let params = prepared.generate_request.sampling_params;

        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 0.6,
                top_p: 0.95,
                top_k: 20,
                seed: None,
                max_tokens: 40957,
                min_tokens: 0,
                thinking_token_budget: None,
                logprobs: None,
                prompt_logprobs: None,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                repetition_detection: None,
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
                logit_bias: None,
                allowed_token_ids: None,
                bad_words_token_ids: None,
                structured_outputs: None,
                logprob_token_ids: None,
                skip_reading_prefix_cache: None,
                extra_args: None,
            }
        "#]]
        .assert_debug_eq(&params);
    }

    #[test]
    fn lower_sampling_params_preserves_explicit_stop_token_ids_in_all_stop_set() {
        let sampling_params = SamplingParams {
            stop_token_ids: Some(vec![11, 77]),
            ..SamplingParams::default()
        };

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
            },
            sample_sampling_limits(),
            3,
            &stub_tokenizer(),
        )
        .unwrap();

        expect_test::expect![[r#"
            EngineCoreSamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
                max_tokens: 999997,
                min_tokens: 0,
                thinking_token_budget: None,
                logprobs: None,
                prompt_logprobs: None,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                repetition_detection: None,
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
                logit_bias: None,
                allowed_token_ids: None,
                bad_words_token_ids: None,
                structured_outputs: None,
                logprob_token_ids: None,
                skip_reading_prefix_cache: None,
                extra_args: None,
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
            },
            sample_sampling_limits(),
            3,
            &stub_tokenizer(),
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
                thinking_token_budget: None,
                logprobs: None,
                prompt_logprobs: None,
                min_p: 0.1,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.2,
                repetition_detection: None,
                stop_token_ids: [],
                eos_token_id: None,
                all_stop_token_ids: {},
                logit_bias: None,
                allowed_token_ids: None,
                bad_words_token_ids: None,
                structured_outputs: None,
                logprob_token_ids: None,
                skip_reading_prefix_cache: None,
                extra_args: None,
            }
        "#]]
        .assert_debug_eq(&params);
    }

    #[test]
    fn lower_sampling_params_passes_logprobs_fields_through() {
        let sampling_params = SamplingParams {
            logprobs: Some(3),
            prompt_logprobs: Some(-1),
            ..Default::default()
        };

        let params = lower_sampling_params(
            sampling_params,
            SamplingHints {
                primary_eos_token_id: None,
                extra_eos_token_ids: BTreeSet::new(),
                default_temperature: None,
                default_top_p: None,
                default_top_k: None,
                default_min_p: None,
                default_repetition_penalty: None,
                default_max_tokens: None,
            },
            SamplingLimits {
                max_logprobs: -1,
                ..sample_sampling_limits()
            },
            3,
            &stub_tokenizer(),
        )
        .unwrap();

        assert_eq!(params.logprobs, Some(3));
        assert_eq!(params.prompt_logprobs, Some(-1));
    }

    #[test]
    fn lower_sampling_params_rejects_full_vocab_logprobs_over_default_cap() {
        let error = lower_sampling_params_with_limits(
            SamplingParams {
                logprobs: Some(-1),
                ..Default::default()
            },
            sample_sampling_limits(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::Logprobs(LogprobsError::TooManyCount {
                parameter: "logprobs",
                requested: 1000,
                max_allowed: 20,
            })
        ));
    }

    #[test]
    fn lower_sampling_params_expands_full_vocab_logprobs_from_model_vocab() {
        let params = lower_sampling_params_with_limits(
            SamplingParams {
                logprobs: Some(-1),
                ..Default::default()
            },
            SamplingLimits {
                max_logprobs: 1500,
                ..sample_sampling_limits()
            },
        )
        .unwrap();

        assert_eq!(params.logprobs, Some(-1));
    }

    #[test]
    fn lower_sampling_params_rejects_invalid_logprob_token_ids() {
        let error = lower_sampling_params_with_limits(
            SamplingParams {
                logprobs: Some(1),
                logprob_token_ids: Some(vec![1000]),
                ..Default::default()
            },
            sample_sampling_limits(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::TokenIds(TokenIdsError::OutOfVocab {
                parameter: "logprob_token_ids",
                token_ids,
                vocab_size: 1000,
            }) if token_ids == vec![1000]
        ));
    }

    #[test]
    fn lower_sampling_params_rejects_out_of_vocab_stop_token_ids() {
        let error = lower_sampling_params_with_limits(
            SamplingParams {
                stop_token_ids: Some(vec![999, 1000]),
                ..Default::default()
            },
            sample_sampling_limits(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::TokenIds(TokenIdsError::OutOfVocab {
                parameter: "stop_token_ids",
                token_ids,
                vocab_size: 1000,
            }) if token_ids == vec![1000]
        ));
    }

    #[test]
    fn lower_sampling_params_rejects_out_of_vocab_allowed_token_ids() {
        let error = lower_sampling_params_with_limits(
            SamplingParams {
                allowed_token_ids: Some(vec![1999, 2000]),
                ..Default::default()
            },
            sample_sampling_limits(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::TokenIds(TokenIdsError::OutOfVocab {
                parameter: "allowed_token_ids",
                token_ids,
                vocab_size: 2000,
            }) if token_ids == vec![2000]
        ));
    }

    #[test]
    fn lower_sampling_params_rejects_empty_allowed_token_ids() {
        let error = lower_sampling_params_with_limits(
            SamplingParams {
                allowed_token_ids: Some(vec![]),
                ..Default::default()
            },
            sample_sampling_limits(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::TokenIds(TokenIdsError::EmptyAllowedTokenIds)
        ));
    }

    #[test]
    fn lower_sampling_params_rejects_out_of_vocab_bad_words() {
        let tokenizer = FixedTokenizer {
            token_ids: vec![1999, 2000],
        };
        let error = lower_sampling_params(
            SamplingParams {
                bad_words: Some(vec!["blocked".to_string()]),
                ..Default::default()
            },
            SamplingHints::default(),
            sample_sampling_limits(),
            3,
            &tokenizer,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::TokenIds(TokenIdsError::OutOfVocab {
                parameter: "bad_words",
                token_ids,
                vocab_size: 2000,
            }) if token_ids == vec![2000]
        ));
    }

    #[test]
    fn lower_sampling_params_rejects_out_of_vocab_logit_bias() {
        let error = lower_sampling_params_with_limits(
            SamplingParams {
                logit_bias: Some(HashMap::from([(1000, 1.0)])),
                ..Default::default()
            },
            sample_sampling_limits(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::TokenIds(TokenIdsError::OutOfVocab {
                parameter: "logit_bias",
                token_ids,
                vocab_size: 1000,
            }) if token_ids == vec![1000]
        ));
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
            },
            sample_sampling_limits(),
            3,
            &stub_tokenizer(),
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
                thinking_token_budget: None,
                logprobs: None,
                prompt_logprobs: None,
                min_p: 0.1,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.2,
                repetition_detection: None,
                stop_token_ids: [],
                eos_token_id: None,
                all_stop_token_ids: {},
                logit_bias: None,
                allowed_token_ids: None,
                bad_words_token_ids: None,
                structured_outputs: None,
                logprob_token_ids: None,
                skip_reading_prefix_cache: None,
                extra_args: None,
            }
        "#]]
        .assert_debug_eq(&params);
    }

    #[test]
    fn resolve_max_tokens_caps_by_model_len() {
        let result = resolve_max_tokens(Some(150), None, 200, 100);
        assert_eq!(result.unwrap(), 100);
    }

    #[test]
    fn lower_text_request_preserves_non_streaming_request_metadata() {
        let mut request = sample_request();
        request.intermediate = false;

        let prepared = lower_text_request(
            request,
            vec![1, 2, 3],
            sample_sampling_hints(),
            sample_sampling_limits(),
            &stub_tokenizer(),
        )
        .unwrap();

        assert!(!prepared.text_request.intermediate);
        assert_eq!(prepared.generate_request.request_id, "text-1");
    }

    #[test]
    fn resolve_max_tokens_user_smaller_than_model_limit() {
        let result = resolve_max_tokens(Some(50), None, 200, 100);
        assert_eq!(result.unwrap(), 50);
    }

    #[test]
    fn resolve_max_tokens_uses_default_when_user_omits() {
        let result = resolve_max_tokens(None, Some(64), 200, 100);
        assert_eq!(result.unwrap(), 64);
    }

    #[test]
    fn resolve_max_tokens_default_capped_by_model_len() {
        let result = resolve_max_tokens(None, Some(256), 200, 100);
        assert_eq!(result.unwrap(), 100);
    }

    #[test]
    fn resolve_max_tokens_uses_model_limit_when_user_omits() {
        let result = resolve_max_tokens(None, None, 200, 100);
        assert_eq!(result.unwrap(), 100);
    }

    #[test]
    fn resolve_max_tokens_prompt_too_long() {
        let result = resolve_max_tokens(Some(10), None, 100, 100);
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
        let result = resolve_max_tokens(Some(10), None, 100, 200);
        assert!(matches!(
            result,
            Err(Error::PromptTooLong {
                max_model_len: 100,
                prompt_len: 200,
            })
        ));
    }
}
