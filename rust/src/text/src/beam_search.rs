use std::collections::BTreeSet;

use vllm_engine_core_client::protocol::logprobs::TokenLogprob;
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;
use vllm_llm::{FinishReason, GenerateOutputStreamExt, GenerateRequest};

use crate::error::Result;

/// One beam in the beam search.
#[derive(Debug, Clone)]
pub struct BeamSearchSequence {
    /// Full token sequence (prompt + generated tokens).
    pub tokens: Vec<u32>,
    /// Cumulative log probability of the beam.
    pub cum_logprob: f32,
    /// Per-step logprob details.
    pub logprobs: Vec<Vec<TokenLogprob>>,
    /// Terminal finish reason.
    pub finish_reason: Option<FinishReason>,
    /// EOS token ID that stopped this beam, if any.
    pub stop_reason: Option<u32>,
    /// LoRA request for this beam.
    pub lora_request: Option<LoraRequest>,
    // TODO: store the original multimodal features (MmFeatures) so that
    // per-step GenerateRequests can pass them to the engine, matching
    // Python's BeamSearchSequence.get_prompt() which rebuilds mm_input()
    // with mm_kwargs / mm_hashes / mm_placeholders preserved.
}

/// Collected result of one beam search invocation.
#[derive(Debug, Clone)]
pub struct BeamSearchOutput {
    /// Original prompt token IDs (without generated tokens).
    pub prompt_token_ids: Vec<u32>,
    /// Best beams after beam search, sorted by score.
    pub sequences: Vec<BeamSearchSequence>,
}

/// Configuration for one beam search invocation.
pub(crate) struct BeamSearchParams {
    pub request_id: String,
    pub beam_width: u32,
    pub max_tokens: u32,
    pub temperature: f32,
    pub length_penalty: f32,
    pub ignore_eos: bool,
    pub include_stop_str_in_output: bool,
    pub eos_token_id: Option<u32>,
    pub extra_eos_token_ids: BTreeSet<u32>,
    pub lora_request: Option<LoraRequest>,
    pub cache_salt: Option<String>,
    pub priority: i32,
    pub data_parallel_rank: Option<u32>,
}

/// One candidate extension during beam search step processing.
struct BeamCandidate {
    cum_logprob: f32,
    token_id: u32,
    parent_seq: usize,
    logprobs: Vec<TokenLogprob>,
}

/// Calculate the beam search score with length penalty.
///
/// Adapted from HuggingFace transformers.
///
/// When `eos_token_id` is `None` the EOS length adjustment is skipped,
/// matching Python's graceful handling of `tokenizer.eos_token_id == None`.
fn get_beam_search_score(
    tokens: &[u32],
    cumulative_logprob: f32,
    eos_token_id: Option<u32>,
    length_penalty: f32,
) -> f32 {
    let mut seq_len = tokens.len();
    if let Some(eos) = eos_token_id {
        if tokens.last() == Some(&eos) {
            seq_len -= 1;
        }
    }
    let denom = (seq_len as f32).powf(length_penalty);
    if denom == 0.0 {
        cumulative_logprob
    } else {
        cumulative_logprob / denom
    }
}

fn all_stop_ids(params: &BeamSearchParams) -> BTreeSet<u32> {
    let mut ids: BTreeSet<u32> = params.extra_eos_token_ids.iter().copied().collect();
    if let Some(eos) = params.eos_token_id {
        ids.insert(eos);
    }
    ids
}

/// Given a scored token produced by the engine, determine whether it should
/// terminate the current beam (move it to `completed`) rather than become a
/// candidate for the next step.
///
/// Matching Python's online and offline beam search, only the primary EOS
/// token terminates a beam.  Custom stop tokens (`stop_token_ids`) and
/// extra EOS tokens (`extra_eos_token_ids`) remain active candidates.
fn should_terminate_beam(eos_token_id: Option<u32>, token_id: u32, ignore_eos: bool) -> bool {
    eos_token_id.is_some_and(|eos| token_id == eos) && !ignore_eos
}

pub(crate) async fn run_beam_search(
    llm: &vllm_llm::Llm,
    prompt_token_ids: Vec<u32>,
    params: BeamSearchParams,
) -> Result<BeamSearchOutput> {
    let beam_width = params.beam_width.max(1) as usize;
    let logprobs_num = 2 * beam_width;

    let mut active: Vec<BeamSearchSequence> = vec![BeamSearchSequence {
        tokens: prompt_token_ids.clone(),
        cum_logprob: 0.0,
        logprobs: vec![],
        finish_reason: None,
        stop_reason: None,
        lora_request: params.lora_request.clone(),
    }];
    let mut completed: Vec<BeamSearchSequence> = vec![];

    let all_stop_token_ids = all_stop_ids(&params);

    for step in 0..params.max_tokens {
        let mut futures = Vec::with_capacity(active.len());

        for (i, seq) in active.iter().enumerate() {
            let sampling_params = EngineCoreSamplingParams {
                temperature: params.temperature,
                max_tokens: 1,
                min_tokens: 0,
                logprobs: Some(logprobs_num as i32),
                stop_token_ids: vec![],
                eos_token_id: if params.ignore_eos {
                    None
                } else {
                    params.eos_token_id
                },
                all_stop_token_ids: all_stop_token_ids.clone(),
                ..Default::default()
            };

            let gen_req = GenerateRequest {
                request_id: format!("{}-beam-{i}-step-{step}", params.request_id),
                prompt_token_ids: seq.tokens.clone(),
                sampling_params,
                // TODO: pass through seq.mm_features once BeamSearchSequence
                // stores them (see TODO on BeamSearchSequence struct).
                mm_features: None,
                arrival_time: None,
                cache_salt: params.cache_salt.clone(),
                trace_headers: None,
                priority: params.priority,
                data_parallel_rank: params.data_parallel_rank,
                reasoning_parser_kwargs: None,
                lora_request: seq.lora_request.clone(),
            };

            futures.push(async { llm.generate(gen_req).await?.collect_output().await });
        }

        let results = futures::future::join_all(futures).await;

        let mut candidates: Vec<BeamCandidate> = vec![];

        for (i, result) in results.into_iter().enumerate() {
            let output = result?;
            let current_seq = &active[i];

            if matches!(output.finish_reason, FinishReason::Error) {
                return Err(crate::error::Error::BeamSearchEngineError);
            }

            if let Some(logprobs) = &output.logprobs {
                if let Some(position) = logprobs.positions.first() {
                    let mut seen_token_ids: BTreeSet<u32> = BTreeSet::new();
                    for entry in &position.entries {
                        if !seen_token_ids.insert(entry.token_id) {
                            continue;
                        }
                        let candidate_logprob = current_seq.cum_logprob + entry.logprob;

                        if should_terminate_beam(
                            params.eos_token_id,
                            entry.token_id,
                            params.ignore_eos,
                        ) {
                            let mut tokens = current_seq.tokens.clone();
                            if params.include_stop_str_in_output {
                                tokens.push(entry.token_id);
                            }
                            let mut logprobs = current_seq.logprobs.clone();
                            logprobs.push(position.entries.clone());
                            completed.push(BeamSearchSequence {
                                tokens,
                                cum_logprob: candidate_logprob,
                                logprobs,
                                finish_reason: Some(FinishReason::stop_eos()),
                                stop_reason: Some(entry.token_id),
                                lora_request: current_seq.lora_request.clone(),
                            });
                        } else {
                            candidates.push(BeamCandidate {
                                cum_logprob: candidate_logprob,
                                token_id: entry.token_id,
                                parent_seq: i,
                                logprobs: position.entries.clone(),
                            });
                        }
                    }
                }
            }
        }

        if candidates.is_empty() {
            break;
        }

        candidates.sort_by(|a, b| b.cum_logprob.total_cmp(&a.cum_logprob));
        let top_candidates: Vec<_> = candidates.into_iter().take(beam_width).collect();

        let mut next: Vec<BeamSearchSequence> = Vec::with_capacity(top_candidates.len());
        for candidate in top_candidates {
            let parent = &active[candidate.parent_seq];
            let mut tokens = parent.tokens.clone();
            tokens.push(candidate.token_id);
            let mut logprobs = parent.logprobs.clone();
            logprobs.push(candidate.logprobs);
            next.push(BeamSearchSequence {
                tokens,
                cum_logprob: candidate.cum_logprob,
                logprobs,
                finish_reason: None,
                stop_reason: None,
                lora_request: parent.lora_request.clone(),
            });
        }

        active = next;
        if active.is_empty() {
            break;
        }
    }

    for mut seq in active {
        if seq.finish_reason.is_none() {
            seq.finish_reason = Some(FinishReason::Length);
        }
        completed.push(seq);
    }

    let eos = params.eos_token_id;
    completed.sort_by(|a, b| {
        let score_a =
            get_beam_search_score(&a.tokens, a.cum_logprob, eos, params.length_penalty);
        let score_b =
            get_beam_search_score(&b.tokens, b.cum_logprob, eos, params.length_penalty);
        score_b.total_cmp(&score_a)
    });

    let best_sequences: Vec<BeamSearchSequence> =
        completed.into_iter().take(beam_width).collect();

    Ok(BeamSearchOutput {
        prompt_token_ids,
        sequences: best_sequences,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use vllm_engine_core_client::protocol::logprobs::TokenLogprob;
    use super::*;

    fn make_token_logprob(token_id: u32, logprob: f32) -> TokenLogprob {
        TokenLogprob {
            token_id,
            logprob,
            rank: 1,
        }
    }

    fn make_sequence(
        tokens: Vec<u32>,
        cum_logprob: f32,
        logprobs: Vec<Vec<TokenLogprob>>,
    ) -> BeamSearchSequence {
        BeamSearchSequence {
            tokens,
            cum_logprob,
            logprobs,
            finish_reason: None,
            stop_reason: None,
            lora_request: None,
        }
    }

    // -----------------------------------------------------------------------
    // get_beam_search_score
    // -----------------------------------------------------------------------

    #[test]
    fn beam_score_identity_length_penalty() {
        let tokens = vec![1, 2, 3, 4, 5];
        let score = get_beam_search_score(&tokens, -10.0, Some(99), 1.0);
        assert!((score - (-10.0 / 5.0)).abs() < 1e-6);
    }

    #[test]
    fn beam_score_longer_higher_with_same_cum_logprob_and_lp_gt_1() {
        let short = get_beam_search_score(&[1, 2], -3.0, Some(99), 2.0);
        let long = get_beam_search_score(&[1, 2, 3, 4, 5], -3.0, Some(99), 2.0);
        assert!(long > short);
    }

    #[test]
    fn beam_score_longer_favored_with_lp_gt_1() {
        let short = get_beam_search_score(&[1, 2], -5.0, Some(99), 2.0);
        let long = get_beam_search_score(&[1, 2, 3, 4, 5], -5.0, Some(99), 2.0);
        assert!(long > short);
    }

    #[test]
    fn beam_score_longer_still_favored_with_lp_lt_1() {
        let short = get_beam_search_score(&[1, 2], -5.0, Some(99), 0.5);
        let long = get_beam_search_score(&[1, 2, 3, 4, 5], -5.0, Some(99), 0.5);
        assert!(long > short);
    }

    #[test]
    fn beam_score_lp_affects_relative_ordering() {
        let short_strong = get_beam_search_score(&[1, 2], -5.0, Some(99), 1.0);
        let long_weak = get_beam_search_score(&[1, 2, 3, 4, 5], -5.5, Some(99), 1.0);
        assert!(long_weak > short_strong);
    }

    #[test]
    fn beam_score_eos_subtracts_from_length() {
        let with_eos = get_beam_search_score(&[1, 2, 99], -6.0, Some(99), 0.0);
        let no_eos = get_beam_search_score(&[1, 2], -6.0, Some(99), 0.0);
        assert!((with_eos - no_eos).abs() < 1e-6);
    }

    #[test]
    fn beam_score_eos_none_skips_length_adjustment() {
        let with_eos_none = get_beam_search_score(&[1, 2, 99], -6.0, None, 0.0);
        let with_eos_some = get_beam_search_score(&[1, 2, 99], -6.0, Some(99), 0.0);
        assert!((with_eos_none - with_eos_some).abs() < 1e-6);
    }

    #[test]
    fn beam_score_zero_denom_returns_cum_logprob() {
        let score = get_beam_search_score(&[], -7.0, Some(99), 1.0);
        assert!((score - (-7.0)).abs() < 1e-6);
    }

    #[test]
    fn beam_score_lp_zero_no_penalty() {
        let tokens = vec![1, 2, 3, 4, 5];
        let score = get_beam_search_score(&tokens, -10.0, Some(99), 0.0);
        assert!((score - (-10.0)).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // all_stop_ids
    // -----------------------------------------------------------------------

    #[test]
    fn all_stop_ids_collects_eos_and_extra_eos() {
        let params = BeamSearchParams {
            request_id: "test".into(),
            beam_width: 1,
            max_tokens: 10,
            temperature: 1.0,
            length_penalty: 1.0,
            ignore_eos: false,
            include_stop_str_in_output: false,
            eos_token_id: Some(99),
            extra_eos_token_ids: BTreeSet::from([151643]),
            lora_request: None,
            cache_salt: None,
            priority: 0,
            data_parallel_rank: None,
        };
        let all = all_stop_ids(&params);
        assert!(all.contains(&151643));
        assert!(all.contains(&99));
    }

    #[test]
    fn all_stop_ids_always_includes_extra_eos_regardless_of_ignore_eos() {
        let params = BeamSearchParams {
            request_id: "test".into(),
            beam_width: 1,
            max_tokens: 10,
            temperature: 1.0,
            length_penalty: 1.0,
            ignore_eos: true,
            include_stop_str_in_output: false,
            eos_token_id: Some(99),
            extra_eos_token_ids: BTreeSet::from([151643]),
            lora_request: None,
            cache_salt: None,
            priority: 0,
            data_parallel_rank: None,
        };
        let all = all_stop_ids(&params);
        assert!(all.contains(&151643));
        assert!(all.contains(&99));
    }

    // -----------------------------------------------------------------------
    // BeamSearchSequence / BeamSearchOutput
    // -----------------------------------------------------------------------

    #[test]
    fn output_preserves_prompt_token_ids() {
        let output = BeamSearchOutput {
            prompt_token_ids: vec![1, 2, 3],
            sequences: vec![make_sequence(vec![1, 2, 3, 4], -1.0, vec![])],
        };
        assert_eq!(output.prompt_token_ids, vec![1, 2, 3]);
        assert_eq!(output.sequences.len(), 1);
        assert_eq!(output.sequences[0].tokens, vec![1, 2, 3, 4]);
    }

    #[test]
    fn beam_candidate_sort_stable_by_cum_logprob() {
        let mut candidates = vec![
            BeamCandidate {
                cum_logprob: -2.0,
                token_id: 10,
                parent_seq: 0,
                logprobs: vec![],
            },
            BeamCandidate {
                cum_logprob: -1.0,
                token_id: 20,
                parent_seq: 0,
                logprobs: vec![],
            },
            BeamCandidate {
                cum_logprob: -3.0,
                token_id: 30,
                parent_seq: 0,
                logprobs: vec![],
            },
        ];
        candidates.sort_by(|a, b| b.cum_logprob.total_cmp(&a.cum_logprob));
        assert_eq!(candidates[0].token_id, 20);
        assert_eq!(candidates[1].token_id, 10);
        assert_eq!(candidates[2].token_id, 30);
    }

    // -----------------------------------------------------------------------
    // BeamSearchParams default behavior
    // -----------------------------------------------------------------------

    #[test]
    fn params_beam_width_clamped_to_one() {
        let params = BeamSearchParams {
            request_id: "test".into(),
            beam_width: 0,
            max_tokens: 10,
            temperature: 1.0,
            length_penalty: 1.0,
            ignore_eos: false,
            include_stop_str_in_output: false,
            eos_token_id: None,
            extra_eos_token_ids: BTreeSet::new(),
            lora_request: None,
            cache_salt: None,
            priority: 0,
            data_parallel_rank: None,
        };
        assert_eq!(params.beam_width.max(1) as usize, 1);
    }

    #[test]
    fn logprobs_num_is_double_beam_width() {
        let beam_width: usize = 3;
        let logprobs_num = 2 * beam_width;
        assert_eq!(logprobs_num, 6);
    }


    // -----------------------------------------------------------------------
    // EOS / stop completion logic — behavior via scoring
    // -----------------------------------------------------------------------

    #[test]
    fn beams_scored_by_length_penalty() {
        let eos = Some(99);
        let short_seq = make_sequence(vec![1, 2, 3], -5.0, vec![vec![make_token_logprob(3, -5.0)]]);
        let long_seq = make_sequence(
            vec![1, 2, 3, 4, 5],
            -5.5,
            vec![vec![make_token_logprob(5, -5.5)]],
        );

        let score_short =
            get_beam_search_score(&short_seq.tokens, short_seq.cum_logprob, eos, 1.0);
        let score_long = get_beam_search_score(&long_seq.tokens, long_seq.cum_logprob, eos, 1.0);

        // lp=1.0: short=-5/3=-1.667, long=-5.5/5=-1.1 → long wins (length reward)
        assert!(score_long > score_short);

        let score_short_lp3 =
            get_beam_search_score(&short_seq.tokens, short_seq.cum_logprob, eos, 3.0);
        let score_long_lp3 =
            get_beam_search_score(&long_seq.tokens, long_seq.cum_logprob, eos, 3.0);
        // lp=3.0: short=-5/27=-0.185, long=-5.5/125=-0.044 → long still wins
        assert!(score_long_lp3 > score_short_lp3);
    }

    #[test]
    fn beams_sorted_by_raw_logprob_when_no_eos() {
        let seq_a = make_sequence(vec![1, 2, 3], -3.0, vec![]);
        let seq_b = make_sequence(vec![1, 2, 3, 4, 5], -5.0, vec![]);

        let mut sequences = vec![seq_a.clone(), seq_b.clone()];
        sequences.sort_by(|a, b| b.cum_logprob.total_cmp(&a.cum_logprob));

        assert_eq!(sequences[0].cum_logprob, -3.0);
        assert_eq!(sequences[1].cum_logprob, -5.0);
    }

    // -----------------------------------------------------------------------
    // should_terminate_beam — EOS-only termination (Python compatibility)
    // -----------------------------------------------------------------------

    #[test]
    fn eos_terminates_when_not_ignored() {
        assert!(should_terminate_beam(Some(99), 99, false));
    }

    #[test]
    fn eos_does_not_terminate_when_ignored() {
        assert!(!should_terminate_beam(Some(99), 99, true));
    }

    #[test]
    fn non_eos_token_never_terminates() {
        assert!(!should_terminate_beam(Some(99), 42, false));
        assert!(!should_terminate_beam(Some(99), 42, true));
    }

    #[test]
    fn nothing_terminates_when_eos_token_id_is_none() {
        assert!(!should_terminate_beam(None, 99, false));
        assert!(!should_terminate_beam(None, 42, false));
    }

    // -----------------------------------------------------------------------
    // candidate dedup — duplicate token_ids in logprobs
    // -----------------------------------------------------------------------

    #[test]
    fn seen_token_ids_dedup_skips_duplicate_entries() {
        // Simulate the dedup loop: entries with duplicate token_id
        // should be skipped, so only the first occurrence is processed.
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        let entries = [1, 2, 1, 3, 2, 4];
        let mut accepted: Vec<u32> = vec![];

        for &id in &entries {
            if seen.insert(id) {
                accepted.push(id);
            }
        }

        assert_eq!(accepted, vec![1, 2, 3, 4]);
    }

    // -----------------------------------------------------------------------
    // Beam candidate split — eos vs active per step
    // -----------------------------------------------------------------------

    #[test]
    fn per_step_eos_goes_to_completed_non_eos_to_candidates() {
        // Simulate one step: given a parent beam and the top logprob
        // entries from the engine, split them into "completed" (EOS hits)
        // and "candidates" (all other tokens).
        let eos_token_id = Some(99u32);
        let ignore_eos = false;

        let logprob_entries: Vec<(u32, f32)> = vec![
            (99, -0.5),   // EOS — should complete
            (42, -1.0),   // normal token
            (77, -1.5),   // normal token
            (151643, -2.0), // extra EOS — should NOT complete (matching Python)
        ];

        let mut completed_count = 0;
        let mut candidate_count = 0;

        for (token_id, _) in &logprob_entries {
            if should_terminate_beam(eos_token_id, *token_id, ignore_eos) {
                completed_count += 1;
            } else {
                candidate_count += 1;
            }
        }

        assert_eq!(completed_count, 1, "only primary EOS (99) terminates");
        assert_eq!(candidate_count, 3, "all non-EOS including extra_eos are candidates");
    }

    // -----------------------------------------------------------------------
    // include_stop_str_in_output — EOS token affects length-penalty scoring
    // -----------------------------------------------------------------------

    #[test]
    fn include_stop_str_in_output_affects_score_via_length_penalty() {
        let eos = 99u32;
        // Sequence without EOS appended: length 3, score = -3.0 / 3^2 = -0.333
        let tokens_without_eos = vec![1, 2, 3];
        // Sequence with EOS appended: length 4, but EOS subtracts 1 → 3
        // score = -3.0 / 3^2 = -0.333
        let tokens_with_eos = vec![1, 2, 3, eos];

        let score_without = get_beam_search_score(&tokens_without_eos, -3.0, Some(eos), 2.0);
        let score_with = get_beam_search_score(&tokens_with_eos, -3.0, Some(eos), 2.0);

        assert!((score_without - score_with).abs() < 1e-6,
            "including EOS should not change the score since EOS subtracts 1 from length");
    }
}
