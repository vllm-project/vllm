use std::collections::BTreeSet;

use vllm_engine_core_client::protocol::logprobs::TokenLogprob;
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;
use vllm_llm::{FinishReason, GenerateOutputStreamExt, GenerateRequest};

use crate::error::Result;

/// One beam in the beam search.
#[derive(Debug, Clone)]
pub struct BeamSearchBeam {
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
}

/// Collected result of one beam search invocation.
#[derive(Debug, Clone)]
pub struct BeamSearchOutput {
    /// Original prompt token IDs (without generated tokens).
    pub prompt_token_ids: Vec<u32>,
    /// Best beams after beam search, sorted by score.
    pub beams: Vec<BeamSearchBeam>,
}

/// Configuration for one beam search invocation.
pub(crate) struct BeamSearchConfig {
    pub beam_width: u32,
    pub max_tokens: u32,
    pub temperature: f32,
    pub length_penalty: f32,
    pub ignore_eos: bool,
    pub eos_token_id: Option<u32>,
    pub lora_request: Option<LoraRequest>,
    pub cache_salt: Option<String>,
    pub priority: i32,
    pub data_parallel_rank: Option<u32>,
    pub stop_token_ids: Vec<u32>,
}

/// One candidate extension during beam search step processing.
struct BeamCandidate {
    cum_logprob: f32,
    token_id: u32,
    parent_beam: usize,
    logprobs: Vec<TokenLogprob>,
}

/// Calculate the beam search score with length penalty.
///
/// Adapted from HuggingFace transformers.
fn get_beam_search_score(
    tokens: &[u32],
    cumulative_logprob: f32,
    eos_token_id: u32,
    length_penalty: f32,
) -> f32 {
    let mut seq_len = tokens.len();
    // Currently EOS is not pushed into beam tokens (matching Python's
    // include_stop_str_in_output=False default), so this check is inactive.
    // It exists so the scoring stays correct if that behavior is ever
    // made configurable.
    if tokens.last() == Some(&eos_token_id) {
        seq_len -= 1;
    }
    let denom = (seq_len as f32).powf(length_penalty);
    if denom == 0.0 {
        cumulative_logprob
    } else {
        cumulative_logprob / denom
    }
}

/// Run beam search for one prompt.
pub(crate) async fn run_beam_search(
    llm: &vllm_llm::Llm,
    prompt_token_ids: Vec<u32>,
    config: BeamSearchConfig,
) -> Result<BeamSearchOutput> {
    let beam_width = config.beam_width.max(1) as usize;
    let logprobs_num = 2 * beam_width;

    let mut all_beams: Vec<BeamSearchBeam> = vec![BeamSearchBeam {
        tokens: prompt_token_ids.clone(),
        cum_logprob: 0.0,
        logprobs: vec![],
        finish_reason: None,
        stop_reason: None,
        lora_request: config.lora_request.clone(),
    }];
    let mut completed: Vec<BeamSearchBeam> = vec![];

    let stop_token_ids = config.stop_token_ids.clone();
    let all_stop_token_ids: BTreeSet<u32> = {
        let mut ids: BTreeSet<u32> = config.stop_token_ids.iter().copied().collect();
        if let Some(eos) = config.eos_token_id {
            ids.insert(eos);
        }
        ids
    };

    for step in 0..config.max_tokens {
        let mut futures = Vec::with_capacity(all_beams.len());

        for (i, beam) in all_beams.iter().enumerate() {
            let sampling_params = EngineCoreSamplingParams {
                temperature: config.temperature,
                max_tokens: 1,
                min_tokens: 0,
                logprobs: Some(logprobs_num as i32),
                stop_token_ids: stop_token_ids.clone(),
                eos_token_id: config.eos_token_id,
                all_stop_token_ids: all_stop_token_ids.clone(),
                ..Default::default()
            };

            let gen_req = GenerateRequest {
                request_id: format!("beam-{i}-step-{step}"),
                prompt_token_ids: beam.tokens.clone(),
                sampling_params,
                mm_features: None,
                arrival_time: None,
                cache_salt: config.cache_salt.clone(),
                trace_headers: None,
                priority: config.priority,
                data_parallel_rank: config.data_parallel_rank,
                reasoning_parser_kwargs: None,
                lora_request: beam.lora_request.clone(),
            };

            futures.push(async { llm.generate(gen_req).await?.collect_output().await });
        }

        let results = futures::future::join_all(futures).await;

        let mut candidates: Vec<BeamCandidate> = vec![];

        for (i, result) in results.into_iter().enumerate() {
            let output = result?;
            let current_beam = &all_beams[i];

            if let Some(logprobs) = &output.logprobs {
                if let Some(position) = logprobs.positions.first() {
                    let mut seen_token_ids: BTreeSet<u32> = BTreeSet::new();
                    for entry in &position.entries {
                        if !seen_token_ids.insert(entry.token_id) {
                            continue;
                        }
                        let candidate_logprob = current_beam.cum_logprob + entry.logprob;
                        let is_eos = config.eos_token_id.is_some_and(|eos| entry.token_id == eos);
                        let is_stop = config.stop_token_ids.contains(&entry.token_id);

                        if (is_eos || is_stop) && !config.ignore_eos {
                            let mut logprobs = current_beam.logprobs.clone();
                            logprobs.push(position.entries.clone());
                            completed.push(BeamSearchBeam {
                                tokens: current_beam.tokens.clone(),
                                cum_logprob: candidate_logprob,
                                logprobs,
                                finish_reason: Some(FinishReason::stop_eos()),
                                stop_reason: Some(entry.token_id),
                                lora_request: current_beam.lora_request.clone(),
                            });
                        } else {
                            candidates.push(BeamCandidate {
                                cum_logprob: candidate_logprob,
                                token_id: entry.token_id,
                                parent_beam: i,
                                logprobs: position.entries.clone(),
                            });
                        }
                    }
                }
            }

            if matches!(output.finish_reason, FinishReason::Error) {
                return Err(crate::error::Error::BeamSearchEngineError);
            }
        }

        if candidates.is_empty() {
            break;
        }

        candidates.sort_by(|a, b| b.cum_logprob.total_cmp(&a.cum_logprob));
        let top_candidates: Vec<_> = candidates.into_iter().take(beam_width).collect();

        let mut new_beams: Vec<BeamSearchBeam> = Vec::with_capacity(top_candidates.len());
        for candidate in top_candidates {
            let parent = &all_beams[candidate.parent_beam];
            let mut tokens = parent.tokens.clone();
            tokens.push(candidate.token_id);
            let mut logprobs = parent.logprobs.clone();
            logprobs.push(candidate.logprobs);
            new_beams.push(BeamSearchBeam {
                tokens,
                cum_logprob: candidate.cum_logprob,
                logprobs,
                finish_reason: None,
                stop_reason: None,
                lora_request: parent.lora_request.clone(),
            });
        }

        all_beams = new_beams;
        if all_beams.is_empty() {
            break;
        }
    }

    for mut beam in all_beams {
        if beam.finish_reason.is_none() {
            beam.finish_reason = Some(FinishReason::Length);
        }
        completed.push(beam);
    }

    if let Some(eos) = config.eos_token_id {
        completed.sort_by(|a, b| {
            let score_a =
                get_beam_search_score(&a.tokens, a.cum_logprob, eos, config.length_penalty);
            let score_b =
                get_beam_search_score(&b.tokens, b.cum_logprob, eos, config.length_penalty);
            score_b.total_cmp(&score_a)
        });
    } else {
        completed.sort_by(|a, b| {
            b.cum_logprob.total_cmp(&a.cum_logprob)
        });
    }

    let best_beams: Vec<BeamSearchBeam> = completed.into_iter().take(beam_width).collect();

    Ok(BeamSearchOutput {
        prompt_token_ids,
        beams: best_beams,
    })
}
