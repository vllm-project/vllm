use std::collections::BTreeSet;

use tracing::warn;
use vllm_engine_core_client::protocol::logprobs::TokenLogprob;
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;
use vllm_llm::{FinishReason, GenerateOutputStreamExt, GenerateRequest};

use crate::error::Result;

/// Maximum beam width to prevent resource exhaustion from
/// a single API call spawning excessive concurrent engine requests.
const MAX_BEAM_WIDTH: u32 = 16;

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

/// Process one engine output for a single active beam at one search step.
///
/// Returns beam candidates (non-EOS tokens to explore next) and any newly
/// completed EOS sequences.
fn process_beam_output(
    output: &vllm_llm::CollectedGenerateOutput,
    current_seq: &BeamSearchSequence,
    parent_seq: usize,
    params: &BeamSearchParams,
) -> (Vec<BeamCandidate>, Vec<BeamSearchSequence>) {
    let mut candidates = Vec::new();
    let mut completed = Vec::new();

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
                    let mut logprobs_stack = current_seq.logprobs.clone();
                    logprobs_stack.push(position.entries.clone());
                    completed.push(BeamSearchSequence {
                        tokens,
                        cum_logprob: candidate_logprob,
                        logprobs: logprobs_stack,
                        finish_reason: Some(FinishReason::stop_eos()),
                        stop_reason: Some(entry.token_id),
                        lora_request: current_seq.lora_request.clone(),
                    });
                } else {
                    candidates.push(BeamCandidate {
                        cum_logprob: candidate_logprob,
                        token_id: entry.token_id,
                        parent_seq,
                        logprobs: position.entries.clone(),
                    });
                }
            }
        }
    }

    (candidates, completed)
}

pub(crate) async fn run_beam_search(
    llm: &vllm_llm::Llm,
    prompt_token_ids: Vec<u32>,
    params: BeamSearchParams,
) -> Result<BeamSearchOutput> {
    if params.beam_width > MAX_BEAM_WIDTH {
        warn!(
            "beam_width {} exceeds MAX_BEAM_WIDTH, clamping to {MAX_BEAM_WIDTH}",
            params.beam_width
        );
    }
    let beam_width = params.beam_width.max(1).min(MAX_BEAM_WIDTH) as usize;
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

            if matches!(output.finish_reason, FinishReason::Error) {
                return Err(crate::error::Error::BeamSearchEngineError);
            }

            let (step_candidates, step_completed) =
                process_beam_output(&output, &active[i], i, &params);
            candidates.extend(step_candidates);
            completed.extend(step_completed);
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

    use vllm_engine_core_client::protocol::logprobs::{
        Logprobs, PositionLogprobs, TokenLogprob,
    };
    use vllm_llm::{CollectedGenerateOutput, FinishReason, TokenUsage};
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
    fn params_beam_width_clamped_to_min() {
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
        assert_eq!(params.beam_width.max(1).min(MAX_BEAM_WIDTH) as usize, 1);
    }

    #[test]
    fn params_beam_width_clamped_to_max() {
        let params = BeamSearchParams {
            request_id: "test".into(),
            beam_width: 100_000,
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
        assert_eq!(
            params.beam_width.max(1).min(MAX_BEAM_WIDTH) as usize,
            MAX_BEAM_WIDTH as usize
        );
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

    // -----------------------------------------------------------------------
    // process_beam_output — per-step engine result processing
    // -----------------------------------------------------------------------

    fn make_logprobs(entries: Vec<(u32, f32)>) -> Logprobs {
        Logprobs {
            positions: vec![PositionLogprobs {
                entries: entries
                    .into_iter()
                    .map(|(token_id, logprob)| TokenLogprob {
                        token_id,
                        logprob,
                        rank: 1,
                    })
                    .collect(),
            }],
        }
    }

    fn make_collected_output(
        logprobs: Option<Logprobs>,
        finish_reason: FinishReason,
    ) -> CollectedGenerateOutput {
        CollectedGenerateOutput {
            request_id: String::new(),
            prompt_token_ids: vec![],
            prompt_logprobs: None,
            token_ids: vec![],
            logprobs,
            finish_reason,
            usage: TokenUsage::default(),
            kv_transfer_params: None,
        }
    }

    fn make_params(
        eos_token_id: Option<u32>,
        ignore_eos: bool,
        include_stop_str_in_output: bool,
    ) -> BeamSearchParams {
        BeamSearchParams {
            request_id: "test".into(),
            beam_width: 2,
            max_tokens: 10,
            temperature: 1.0,
            length_penalty: 1.0,
            ignore_eos,
            include_stop_str_in_output,
            eos_token_id,
            extra_eos_token_ids: BTreeSet::from([151643]),
            lora_request: None,
            cache_salt: None,
            priority: 0,
            data_parallel_rank: None,
        }
    }

    fn empty_seq(tokens: Vec<u32>, cum_logprob: f32) -> BeamSearchSequence {
        BeamSearchSequence {
            tokens,
            cum_logprob,
            logprobs: vec![],
            finish_reason: None,
            stop_reason: None,
            lora_request: None,
        }
    }

    #[test]
    fn process_beam_output_normal_tokens_become_candidates() {
        let seq = empty_seq(vec![1, 2], -0.5);
        let output = make_collected_output(
            Some(make_logprobs(vec![(10, -0.3), (20, -0.8)])),
            FinishReason::Length,
        );
        let params = make_params(Some(99), false, false);

        let (candidates, completed) = process_beam_output(&output, &seq, 0, &params);

        assert!(completed.is_empty());
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].token_id, 10);
        assert_eq!(candidates[0].parent_seq, 0);
        assert!((candidates[0].cum_logprob - (-0.8)).abs() < 1e-6);
        assert_eq!(candidates[1].token_id, 20);
        assert!((candidates[1].cum_logprob - (-1.3)).abs() < 1e-6);
    }

    #[test]
    fn process_beam_output_eos_becomes_completed() {
        let seq = empty_seq(vec![1, 2], -0.5);
        let output = make_collected_output(
            Some(make_logprobs(vec![(99, -1.5)])),
            FinishReason::Length,
        );
        let params = make_params(Some(99), false, false);

        let (candidates, completed) = process_beam_output(&output, &seq, 0, &params);

        assert!(candidates.is_empty());
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].finish_reason, Some(FinishReason::stop_eos()));
        assert_eq!(completed[0].stop_reason, Some(99));
        assert!((completed[0].cum_logprob - (-2.0)).abs() < 1e-6);
        assert_eq!(completed[0].tokens, vec![1, 2]);
    }

    #[test]
    fn process_beam_output_ignore_eos_keeps_eos_as_candidate() {
        let seq = empty_seq(vec![1, 2], -0.5);
        let output = make_collected_output(
            Some(make_logprobs(vec![(99, -1.5)])),
            FinishReason::Length,
        );
        let params = make_params(Some(99), true, false);

        let (candidates, completed) = process_beam_output(&output, &seq, 0, &params);

        assert!(completed.is_empty());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].token_id, 99);
    }

    #[test]
    fn process_beam_output_include_stop_str_in_output_appends_eos() {
        let seq = empty_seq(vec![1, 2], -0.5);
        let output = make_collected_output(
            Some(make_logprobs(vec![(99, -1.0)])),
            FinishReason::Length,
        );
        let params = make_params(Some(99), false, true);

        let (_, completed) = process_beam_output(&output, &seq, 0, &params);

        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].tokens, vec![1, 2, 99]);
    }

    #[test]
    fn process_beam_output_dedup_skips_duplicate_entries() {
        let seq = empty_seq(vec![1], -0.1);
        let output = make_collected_output(
            Some(make_logprobs(vec![(10, -0.3), (10, -0.7), (20, -1.0)])),
            FinishReason::Length,
        );
        let params = make_params(Some(99), false, false);

        let (candidates, completed) = process_beam_output(&output, &seq, 0, &params);

        assert!(completed.is_empty());
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].token_id, 10);
        assert_eq!(candidates[1].token_id, 20);
    }

    #[test]
    fn process_beam_output_none_logprobs_returns_empty() {
        let seq = empty_seq(vec![1], -0.1);
        let output = make_collected_output(None, FinishReason::Length);
        let params = make_params(Some(99), false, false);

        let (candidates, completed) = process_beam_output(&output, &seq, 0, &params);

        assert!(candidates.is_empty());
        assert!(completed.is_empty());
    }

    #[test]
    fn process_beam_output_empty_entries_returns_empty() {
        let seq = empty_seq(vec![1], -0.1);
        let output = make_collected_output(Some(make_logprobs(vec![])), FinishReason::Length);
        let params = make_params(Some(99), false, false);

        let (candidates, completed) = process_beam_output(&output, &seq, 0, &params);

        assert!(candidates.is_empty());
        assert!(completed.is_empty());
    }

    #[test]
    fn process_beam_output_extra_eos_is_candidate_not_completed() {
        let seq = empty_seq(vec![1, 2], -0.5);
        let output = make_collected_output(
            Some(make_logprobs(vec![(151643, -1.0)])),
            FinishReason::Length,
        );
        let params = make_params(Some(99), false, false);

        let (candidates, completed) = process_beam_output(&output, &seq, 0, &params);

        assert!(completed.is_empty());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].token_id, 151643);
    }

    // -----------------------------------------------------------------------
    // Full multi-step beam search simulation (integration test)
    // -----------------------------------------------------------------------

    #[test]
    fn beam_search_full_algorithm_two_steps_with_eos() {
        // Simulates a complete beam_width=2, max_tokens=2 run.
        // Prompt: [1], cum_logprob = 0.0
        //
        // Step 1 engine output for the single beam:
        //   10: -0.3, 20: -0.8, 99(eos): -2.0
        //   → EOS → completed: tokens=[1], cum=-2.0
        //   → Candidates: 10(-0.3), 20(-0.8) → both top-2 → active
        //
        // Step 2:
        //   Beam [1,10], cum=-0.3:
        //     30: -1.0, 99(eos): -2.0
        //     → EOS → completed: tokens=[1,10], cum=-2.3
        //     → Candidate: 30 cum=-1.3
        //   Beam [1,20], cum=-0.8:
        //     40: -0.3, 50: -0.8
        //     → Candidates: 40 cum=-1.1, 50 cum=-1.6
        //
        // All step 2 candidates sorted: 40(-1.1) > 30(-1.3) > 50(-1.6)
        // Top 2: 40(parent 1 → [1,20,40]) and 30(parent 0 → [1,10,30])
        // Remaining active → completed (Length):
        //   [1,20,40] cum=-1.1, score=-1.1/3=-0.367
        //   [1,10,30] cum=-1.3, score=-1.3/3=-0.433
        //
        // Final ranking by score (lp=1.0, higher is better):
        // 1. [1,20,40]: -0.367
        // 2. [1,10,30]: -0.433
        // 3. [1,10] EOS: -1.15
        // 4. [1] EOS:    -2.0

        let eos = Some(99u32);
        let length_penalty = 1.0;

        // Initial prompt beam
        let prompt = empty_seq(vec![1], 0.0);
        let active = vec![prompt];
        let mut completed: Vec<BeamSearchSequence> = vec![];
        let beam_width: usize = 2;

        // --- Step 1 ---
        let output1 = make_collected_output(
            Some(make_logprobs(vec![(10, -0.3), (20, -0.8), (99, -2.0)])),
            FinishReason::Length,
        );
        let params = make_params(eos, false, false);
        let (step1_candidates, step1_completed) =
            process_beam_output(&output1, &active[0], 0, &params);
        completed.extend(step1_completed);

        let mut sorted: Vec<_> = step1_candidates;
        sorted.sort_by(|a, b| b.cum_logprob.total_cmp(&a.cum_logprob));
        let mut next_active: Vec<BeamSearchSequence> = vec![];
        for candidate in sorted.into_iter().take(beam_width) {
            let mut tokens = vec![1];
            tokens.push(candidate.token_id);
            next_active.push(BeamSearchSequence {
                tokens,
                cum_logprob: candidate.cum_logprob,
                logprobs: vec![candidate.logprobs],
                finish_reason: None,
                stop_reason: None,
                lora_request: None,
            });
        }

        assert_eq!(next_active.len(), 2);
        assert_eq!(next_active[0].tokens, vec![1, 10]);
        assert!((next_active[0].cum_logprob - (-0.3)).abs() < 1e-6);
        assert_eq!(next_active[1].tokens, vec![1, 20]);
        assert!((next_active[1].cum_logprob - (-0.8)).abs() < 1e-6);

        // --- Step 2: beam [1,10] ---
        let output2a = make_collected_output(
            Some(make_logprobs(vec![(30, -1.0), (99, -2.0)])),
            FinishReason::Length,
        );
        let (step2_a_cands, step2_a_completed) =
            process_beam_output(&output2a, &next_active[0], 0, &params);
        completed.extend(step2_a_completed);

        // --- Step 2: beam [1,20] ---
        let output2b = make_collected_output(
            Some(make_logprobs(vec![(40, -0.3), (50, -0.8)])),
            FinishReason::Length,
        );
        let (step2_b_cands, step2_b_completed) =
            process_beam_output(&output2b, &next_active[1], 1, &params);
        completed.extend(step2_b_completed);

        // Combine and select top beam_width
        let mut all_cands = step2_a_cands;
        all_cands.extend(step2_b_cands);
        all_cands.sort_by(|a, b| b.cum_logprob.total_cmp(&a.cum_logprob));

        let mut step2_active: Vec<BeamSearchSequence> = vec![];
        for candidate in all_cands.into_iter().take(beam_width) {
            let parent = &next_active[candidate.parent_seq];
            let mut tokens = parent.tokens.clone();
            tokens.push(candidate.token_id);
            step2_active.push(BeamSearchSequence {
                tokens,
                cum_logprob: candidate.cum_logprob,
                logprobs: vec![],
                finish_reason: None,
                stop_reason: None,
                lora_request: None,
            });
        }

        // Step 2 active → completed (Length)
        for mut seq in step2_active {
            seq.finish_reason = Some(FinishReason::Length);
            completed.push(seq);
        }

        // --- Final ranking ---
        completed.sort_by(|a, b| {
            let score_a =
                get_beam_search_score(&a.tokens, a.cum_logprob, eos, length_penalty);
            let score_b =
                get_beam_search_score(&b.tokens, b.cum_logprob, eos, length_penalty);
            score_b.total_cmp(&score_a)
        });

        let best: Vec<_> = completed.into_iter().take(beam_width).collect();

        assert_eq!(best.len(), 2);
        assert_eq!(best[0].tokens, vec![1, 20, 40]);
        assert!((best[0].cum_logprob - (-1.1)).abs() < 1e-6);
        assert_eq!(best[1].tokens, vec![1, 10, 30]);
        assert!((best[1].cum_logprob - (-1.3)).abs() < 1e-6);
    }
}
