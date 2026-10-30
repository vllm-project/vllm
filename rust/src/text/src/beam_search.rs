// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Beam search, ported from Python's `BeamSearchOnlineMixin.beam_search`:
//! <https://github.com/vllm-project/vllm/blob/c9af9acc6a/vllm/entrypoints/generate/beam_search/online.py>

use std::collections::BTreeSet;
use std::num::NonZeroU32;
use std::sync::Arc;

use futures::future::try_join_all;
use futures::stream;
use itertools::Itertools as _;
use uuid::Uuid;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_llm::{
    CollectedGenerateOutput, FinishReason, GenerateOutput, GenerateOutputStreamExt as _,
    GeneratePromptInfo, GenerateRequest, Logprobs, PositionLogprobs, TokenLogprob,
};

use crate::error::{Error, Result};
use crate::lower::{PreparedTextRequest, resolve_max_tokens};
use crate::output::{TextDecodeOptions, decoded_text_event_stream};
use crate::request::{SamplingParams, TextRequest};
use crate::{TextLlm, TextOutputStream};

/// Python's default `VLLM_MAX_N_SEQUENCES`, which bounds `beam_width`.
const MAX_BEAM_WIDTH: u32 = 16384;

/// Parameters that only beam search reads; the rest come from the request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamSearchParams {
    /// Number of beams kept at each step, and of sequences returned.
    pub beam_width: NonZeroU32,
    /// Exponent on the number of generated tokens (including a final EOS) that
    /// normalizes a beam's cumulative log-probability when ranking beams, as in
    /// Transformers. Values above 0 favor longer beams, below 0 shorter ones.
    pub length_penalty: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct BeamSequence {
    /// Generated token IDs, without the prompt. A terminating EOS token is kept
    /// only with `include_stop_str_in_output`.
    tokens: Vec<u32>,
    /// Per generated position, including a terminating EOS: the chosen token,
    /// then the engine's top-k.
    logprobs: Vec<PositionLogprobs>,
    cum_logprob: f64,
    /// `None` while the beam is live; reported as `length` once returned.
    finish_reason: Option<FinishReason>,
}

impl BeamSequence {
    fn root() -> Self {
        Self {
            tokens: Vec::new(),
            logprobs: Vec::new(),
            cum_logprob: 0.0,
            finish_reason: None,
        }
    }

    fn extend(
        &self,
        chosen: &TokenLogprob,
        position: &PositionLogprobs,
        cum_logprob: f64,
        keep_token: bool,
        finish_reason: Option<FinishReason>,
    ) -> Self {
        let mut tokens = self.tokens.clone();
        if keep_token {
            tokens.push(chosen.token_id);
        }
        let mut logprobs = self.logprobs.clone();
        // Replace the child's sampled token with the chosen one.
        let entries = std::iter::once(chosen).chain(position.entries.iter().skip(1));
        logprobs.push(PositionLogprobs {
            entries: entries.cloned().collect(),
        });
        Self {
            tokens,
            logprobs,
            cum_logprob,
            finish_reason,
        }
    }
}

/// Python's `get_beam_search_score`. An aborted beam may have no tokens.
fn beam_score(num_generated_tokens: usize, cum_logprob: f64, length_penalty: f64) -> f64 {
    cum_logprob / (num_generated_tokens.max(1) as f64).powf(length_penalty)
}

/// Sort key that ranks NaN last, as NumPy's `argsort` does.
fn rank_key(value: f64) -> f64 {
    if value.is_nan() {
        f64::NEG_INFINITY
    } else {
        value
    }
}

#[derive(Debug)]
struct BeamSelector {
    beam_width: usize,
    length_penalty: f64,
    eos_token_ids: BTreeSet<u32>,
    ignore_eos: bool,
    include_stop_str_in_output: bool,
}

impl BeamSelector {
    /// Run one step: EOS candidates finish into `completed`, and the best
    /// `beam_width` of the rest become the new live beams.
    fn expand(
        &self,
        beams: &[BeamSequence],
        step_logprobs: &[Option<PositionLogprobs>],
        completed: &mut Vec<BeamSequence>,
    ) -> Vec<BeamSequence> {
        let mut candidates = Vec::new();
        for (beam_index, (beam, position)) in beams.iter().zip(step_logprobs).enumerate() {
            let Some(position) = position else {
                continue;
            };
            // The sampled token may repeat among the top-k.
            for entry in position.entries.iter().unique_by(|entry| entry.token_id) {
                let cum_logprob = beam.cum_logprob + f64::from(entry.logprob);
                if !self.ignore_eos && self.eos_token_ids.contains(&entry.token_id) {
                    completed.push(beam.extend(
                        entry,
                        position,
                        cum_logprob,
                        self.include_stop_str_in_output,
                        Some(FinishReason::Stop(Some(StopReason::TokenId(
                            entry.token_id,
                        )))),
                    ));
                } else {
                    candidates.push((cum_logprob, beam_index, entry, position));
                }
            }
        }

        // Stable, so ties are deterministic, unlike Python's `np.argpartition`.
        candidates.sort_by(|a, b| rank_key(b.0).total_cmp(&rank_key(a.0)));
        candidates
            .into_iter()
            .take(self.beam_width)
            .map(|(cum_logprob, beam_index, entry, position)| {
                beams[beam_index].extend(entry, position, cum_logprob, true, None)
            })
            .collect()
    }

    fn finalize(
        &self,
        mut completed: Vec<BeamSequence>,
        live: Vec<BeamSequence>,
    ) -> Vec<BeamSequence> {
        completed.extend(live);
        let score = |beam: &BeamSequence| {
            // `logprobs`, unlike `tokens`, always counts a terminating EOS.
            rank_key(beam_score(
                beam.logprobs.len(),
                beam.cum_logprob,
                self.length_penalty,
            ))
        };
        completed.sort_by(|a, b| score(b).total_cmp(&score(a)));
        completed.truncate(self.beam_width);
        completed
    }
}

/// Reject non-default request fields that Python's beam search silently ignores.
fn reject_unsupported_params(request: &TextRequest) -> Result<()> {
    let SamplingParams {
        // Read by beam search.
        temperature: _,
        max_tokens: _,
        logprobs: _,
        ignore_eos: _,
        // Like Python, the per-step requests never apply the watermark.
        watermarking: _,
        // Only affects streaming, which beam search does not do.
        stream_interval: _,
        // Forwarded to every per-step request.
        skip_reading_prefix_cache: _,
        top_p,
        top_k,
        seed,
        min_tokens,
        thinking_token_budget,
        prompt_logprobs,
        prompt_logprob_token_ids,
        prompt_logprob_start,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        repetition_detection,
        stop_token_ids,
        logit_bias,
        allowed_token_ids,
        bad_words,
        bad_words_token_ids,
        logprob_token_ids,
        structured_outputs,
        routed_experts_prompt_start,
        vllm_xargs,
    } = &request.sampling_params;

    let unsupported = [
        ("top_p", top_p.is_some_and(|value| value != 1.0)),
        ("top_k", top_k.is_some_and(|value| value != 0)),
        ("seed", seed.is_some()),
        ("min_tokens", min_tokens.is_some_and(|value| value != 0)),
        (
            "thinking_token_budget",
            thinking_token_budget.is_some_and(|value| value != -1),
        ),
        ("prompt_logprobs", prompt_logprobs.is_some()),
        (
            "prompt_logprob_token_ids",
            prompt_logprob_token_ids.is_some(),
        ),
        ("prompt_logprob_start", prompt_logprob_start.is_some()),
        ("min_p", min_p.is_some_and(|value| value != 0.0)),
        (
            "frequency_penalty",
            frequency_penalty.is_some_and(|value| value != 0.0),
        ),
        (
            "presence_penalty",
            presence_penalty.is_some_and(|value| value != 0.0),
        ),
        (
            "repetition_penalty",
            repetition_penalty.is_some_and(|value| value != 1.0),
        ),
        (
            "repetition_detection",
            repetition_detection.as_ref().is_some_and(|params| !params.is_disabled()),
        ),
        (
            "stop_token_ids",
            stop_token_ids.as_ref().is_some_and(|ids| !ids.is_empty()),
        ),
        (
            "logit_bias",
            logit_bias.as_ref().is_some_and(|bias| !bias.is_empty()),
        ),
        // TODO: Python scores allowlisted beams with exact-ID logprobs, which
        // need the V2 model runner.
        ("allowed_token_ids", allowed_token_ids.is_some()),
        (
            "bad_words",
            bad_words.as_ref().is_some_and(|words| !words.is_empty()),
        ),
        (
            "bad_words_token_ids",
            bad_words_token_ids.as_ref().is_some_and(|ids| !ids.is_empty()),
        ),
        ("logprob_token_ids", logprob_token_ids.is_some()),
        ("structured_outputs", structured_outputs.is_some()),
        (
            "routed_experts_prompt_start",
            routed_experts_prompt_start.is_some(),
        ),
        (
            "vllm_xargs",
            vllm_xargs.as_ref().is_some_and(|args| !args.is_empty()),
        ),
        (
            "stop",
            (request.decode_options.stop_strings.as_ref()).is_some_and(|stops| !stops.is_empty()),
        ),
    ];
    match unsupported.into_iter().find(|(_, set)| *set) {
        Some((param, _)) => Err(Error::UnsupportedWithBeamSearch { param }),
        None => Ok(()),
    }
}

/// Sampling parameters of every per-step request. The neutral values are
/// explicit so generation config defaults do not apply, as in Python.
fn step_sampling_params(request: &SamplingParams, beam_width: NonZeroU32) -> SamplingParams {
    SamplingParams {
        temperature: request.temperature,
        watermarking: Some(false),
        top_p: Some(1.0),
        top_k: Some(0),
        min_p: Some(0.0),
        frequency_penalty: Some(0.0),
        presence_penalty: Some(0.0),
        repetition_penalty: Some(1.0),
        max_tokens: Some(1),
        logprobs: Some(beam_width.get().saturating_mul(2).try_into().unwrap_or(i32::MAX)),
        skip_reading_prefix_cache: request.skip_reading_prefix_cache,
        ..SamplingParams::default()
    }
}

impl TextLlm {
    /// Return one finished stream per beam, best first, or one error stream if a
    /// step fails. Dropping the future aborts the in-flight step requests.
    pub async fn beam_search(
        &self,
        mut request: TextRequest,
        params: BeamSearchParams,
    ) -> Result<Vec<impl TextOutputStream + use<>>> {
        let BeamSearchParams {
            beam_width,
            length_penalty,
        } = params;
        if beam_width.get() > MAX_BEAM_WIDTH {
            return Err(Error::BeamWidthTooLarge {
                beam_width: beam_width.get(),
                max: MAX_BEAM_WIDTH,
            });
        }
        reject_unsupported_params(&request)?;

        let sampling_hints = self.processor.backend.sampling_hints()?;
        let eos_token_ids = (sampling_hints.primary_eos_token_id.into_iter())
            .chain(sampling_hints.extra_eos_token_ids.iter().copied())
            .collect();
        let user_max_tokens = request.sampling_params.max_tokens;
        let requested_logprobs = request.sampling_params.logprobs;
        let ignore_eos = request.sampling_params.ignore_eos;
        request.sampling_params = step_sampling_params(&request.sampling_params, beam_width);

        let PreparedTextRequest {
            text_request,
            generate_request: step_template,
        } = self.processor.prepare(request)?;
        let prompt_token_ids: Arc<[u32]> = step_template.prompt_token_ids.as_slice().into();
        let max_tokens = resolve_max_tokens(
            user_max_tokens,
            sampling_hints.default_max_tokens,
            self.processor.max_model_len(),
            prompt_token_ids.len() as u32,
        )?;
        let selector = BeamSelector {
            beam_width: beam_width.get() as usize,
            length_penalty,
            eos_token_ids,
            ignore_eos,
            include_stop_str_in_output: text_request.decode_options.include_stop_str_in_output,
        };

        let mut beams = vec![BeamSequence::root()];
        let mut completed = Vec::new();
        for _ in 0..max_tokens {
            let step_outputs = self.run_step(&step_template, &beams).await?;

            if let Some(error) = (step_outputs.iter())
                .find(|output| output.finish_reason.is_error())
                .map(|output| output.finish_reason.clone())
            {
                let error_beam = BeamSequence {
                    finish_reason: Some(error),
                    ..BeamSequence::root()
                };
                return Ok(vec![self.beam_output_stream(
                    &text_request,
                    error_beam,
                    &prompt_token_ids,
                    requested_logprobs,
                )]);
            }
            if step_outputs.iter().any(|output| output.finish_reason.is_abort()) {
                for beam in &mut beams {
                    beam.finish_reason = Some(FinishReason::Abort);
                }
                break;
            }

            let step_logprobs = step_outputs
                .into_iter()
                .map(|output| {
                    output.logprobs.and_then(|logprobs| logprobs.positions.into_iter().next())
                })
                .collect_vec();
            beams = selector.expand(&beams, &step_logprobs, &mut completed);
            if beams.is_empty() {
                break;
            }
        }

        Ok(selector
            .finalize(completed, beams)
            .into_iter()
            .map(|beam| {
                self.beam_output_stream(&text_request, beam, &prompt_token_ids, requested_logprobs)
            })
            .collect())
    }

    async fn run_step(
        &self,
        template: &GenerateRequest,
        beams: &[BeamSequence],
    ) -> Result<Vec<CollectedGenerateOutput>> {
        let step_id = format!("{}-{}", template.request_id, Uuid::new_v4().simple());
        try_join_all(beams.iter().enumerate().map(|(index, beam)| {
            let mut prompt_token_ids = template.prompt_token_ids.clone();
            prompt_token_ids.extend_from_slice(&beam.tokens);
            let request = GenerateRequest {
                request_id: format!("{step_id}-beam-{index}"),
                prompt_token_ids,
                // Stamped at submission, like Python.
                arrival_time: None,
                ..template.clone()
            };
            async move { Ok(self.llm.generate(request).await?.collect_output().await?) }
        }))
        .await
    }

    fn beam_output_stream(
        &self,
        text_request: &TextRequest,
        beam: BeamSequence,
        prompt_token_ids: &Arc<[u32]>,
        requested_logprobs: Option<i32>,
    ) -> impl TextOutputStream + use<> {
        let BeamSequence {
            tokens,
            mut logprobs,
            cum_logprob: _,
            finish_reason,
        } = beam;
        // Like Python: one position per returned token, `requested` top-k each.
        logprobs.truncate(tokens.len());
        let logprobs = requested_logprobs.map(|requested| Logprobs {
            positions: logprobs
                .into_iter()
                .map(|mut position| {
                    if let Ok(requested) = usize::try_from(requested) {
                        position.entries.truncate(requested + 1);
                    }
                    position
                })
                .collect(),
        });
        let output = GenerateOutput {
            request_id: text_request.request_id.clone(),
            prompt_info: Some(Box::new(GeneratePromptInfo {
                prompt_token_ids: prompt_token_ids.clone(),
                prompt_logprobs: None,
                prompt_token_id_logprobs: None,
            })),
            token_ids: tokens,
            logprobs,
            finish_reason: Some(finish_reason.unwrap_or(FinishReason::Length)),
            prompt_cache_usage: Default::default(),
            kv_transfer_params: None,
            ec_transfer_params: None,
            sampling_mask: None,
            spec_decode_metrics: None,
            weight_versions: None,
            routed_experts: None,
            timestamps: Default::default(),
        };
        let decode_options = TextDecodeOptions {
            // Python's text never includes a terminating EOS. The decoder drops
            // a stopped output's last token only when this is off, so invert it.
            include_stop_str_in_output: !text_request.decode_options.include_stop_str_in_output,
            ..text_request.decode_options.clone()
        };
        decoded_text_event_stream(
            text_request.request_id.clone(),
            self.processor.tokenizer(),
            stream::iter([Ok(output)]),
            decode_options,
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::*;

    fn position(entries: &[(u32, f32)]) -> PositionLogprobs {
        PositionLogprobs {
            entries: entries
                .iter()
                .enumerate()
                .map(|(rank, &(token_id, logprob))| TokenLogprob {
                    token_id,
                    logprob,
                    rank: rank.max(1) as u32,
                })
                .collect(),
        }
    }

    fn selector(beam_width: usize) -> BeamSelector {
        BeamSelector {
            beam_width,
            length_penalty: 1.0,
            eos_token_ids: BTreeSet::from([0]),
            ignore_eos: false,
            include_stop_str_in_output: false,
        }
    }

    /// Token IDs, rounded cumulative logprob, and finish reason of each beam.
    fn summary(beams: &[BeamSequence]) -> Vec<(Vec<u32>, String, Option<FinishReason>)> {
        beams
            .iter()
            .map(|beam| {
                (
                    beam.tokens.clone(),
                    format!("{:.2}", beam.cum_logprob),
                    beam.finish_reason.clone(),
                )
            })
            .collect()
    }

    #[test]
    fn beam_score_normalizes_by_generated_length() {
        let scores = [
            beam_score(4, -2.0, 0.0),
            beam_score(4, -2.0, 1.0),
            beam_score(4, -2.0, 2.0),
            beam_score(0, -2.0, 2.0),
        ];
        expect![[r#"
            [
                -2.0,
                -0.5,
                -0.125,
                -2.0,
            ]
        "#]]
        .assert_debug_eq(&scores);
    }

    /// A beam that is worse after the first step wins after the second, which
    /// greedy decoding would miss.
    #[test]
    fn search_keeps_a_worse_first_step_that_wins_later() {
        let selector = selector(2);
        let mut completed = Vec::new();

        // Step 1: token 1 (-0.4) beats token 2 (-0.6).
        let beams = selector.expand(
            &[BeamSequence::root()],
            &[Some(position(&[
                (1, -0.4),
                (1, -0.4),
                (2, -0.6),
                (3, -3.0),
            ]))],
            &mut completed,
        );
        // Step 2: after token 1 every continuation is unlikely, after token 2
        // one is almost certain.
        let beams = selector.expand(
            &beams,
            &[
                Some(position(&[(4, -1.5), (4, -1.5), (5, -1.6)])),
                Some(position(&[(6, -0.1), (6, -0.1), (7, -2.5)])),
            ],
            &mut completed,
        );

        expect![[r#"
            [
                (
                    [
                        2,
                        6,
                    ],
                    "-0.70",
                    None,
                ),
                (
                    [
                        1,
                        4,
                    ],
                    "-1.90",
                    None,
                ),
            ]
        "#]]
        .assert_debug_eq(&summary(&beams));
        assert!(completed.is_empty());
    }

    #[test]
    fn eos_candidates_finish_their_beam_instead_of_expanding() {
        let selector = selector(2);
        let mut completed = Vec::new();
        let beams = selector.expand(
            &[BeamSequence::root()],
            &[Some(position(&[
                (0, -0.1),
                (0, -0.1),
                (1, -1.0),
                (2, -2.0),
            ]))],
            &mut completed,
        );

        // The EOS token is not in `tokens`, but its logprob counts toward the
        // length that normalizes the score.
        expect![[r#"
            (
                [
                    (
                        [
                            1,
                        ],
                        "-1.00",
                        None,
                    ),
                    (
                        [
                            2,
                        ],
                        "-2.00",
                        None,
                    ),
                ],
                [
                    (
                        [],
                        "-0.10",
                        Some(
                            Stop(
                                Some(
                                    TokenId(
                                        0,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ],
                1,
            )
        "#]]
        .assert_debug_eq(&(
            summary(&beams),
            summary(&completed),
            completed[0].logprobs.len(),
        ));
    }

    #[test]
    fn include_stop_str_in_output_keeps_the_eos_token() {
        let selector = BeamSelector {
            include_stop_str_in_output: true,
            ..selector(1)
        };
        let mut completed = Vec::new();
        selector.expand(
            &[BeamSequence::root()],
            &[Some(position(&[(0, -0.1), (0, -0.1), (1, -1.0)]))],
            &mut completed,
        );
        assert_eq!(completed[0].tokens, [0]);
    }

    #[test]
    fn ignore_eos_expands_eos_like_any_token() {
        let selector = BeamSelector {
            ignore_eos: true,
            ..selector(1)
        };
        let mut completed = Vec::new();
        let beams = selector.expand(
            &[BeamSequence::root()],
            &[Some(position(&[(0, -0.1), (0, -0.1), (1, -1.0)]))],
            &mut completed,
        );
        assert_eq!(summary(&beams), [(vec![0], "-0.10".to_string(), None)]);
        assert!(completed.is_empty());
    }

    #[test]
    fn chosen_token_leads_its_logprobs_position() {
        let selector = selector(2);
        let mut completed = Vec::new();
        // The child sampled token 3, which beam search ignores.
        let beams = selector.expand(
            &[BeamSequence::root()],
            &[Some(position(&[(3, -2.0), (1, -0.5), (2, -1.0)]))],
            &mut completed,
        );
        let ids = |beam: &BeamSequence| {
            beam.logprobs[0].entries.iter().map(|entry| entry.token_id).collect_vec()
        };
        assert_eq!(ids(&beams[0]), [1, 1, 2]);
        assert_eq!(ids(&beams[1]), [2, 1, 2]);
    }

    #[test]
    fn ties_keep_beam_then_candidate_order() {
        let selector = selector(3);
        let mut completed = Vec::new();
        let beams = selector.expand(
            &[
                BeamSequence {
                    tokens: vec![10],
                    ..BeamSequence::root()
                },
                BeamSequence {
                    tokens: vec![20],
                    ..BeamSequence::root()
                },
            ],
            &[
                Some(position(&[(1, -1.0), (1, -1.0), (2, -1.0)])),
                Some(position(&[(3, -1.0), (3, -1.0), (4, f32::NAN)])),
            ],
            &mut completed,
        );
        let tokens = beams.iter().map(|beam| beam.tokens.clone()).collect_vec();
        assert_eq!(tokens, [vec![10, 1], vec![10, 2], vec![20, 3]]);
    }

    #[test]
    fn finalize_ranks_by_length_penalized_score() {
        let beam = |len: usize, cum_logprob: f64| BeamSequence {
            tokens: vec![1; len],
            logprobs: vec![position(&[(1, 0.0)]); len],
            cum_logprob,
            finish_reason: None,
        };
        let completed = vec![beam(1, -1.0), beam(4, -2.0)];
        let live = vec![beam(2, -1.6)];

        // Scores with length_penalty = 1: -1.0, -0.5, -0.8.
        let ranked = selector(2).finalize(completed.clone(), live.clone());
        assert_eq!(
            summary(&ranked).iter().map(|(_, cum, _)| cum.as_str()).collect_vec(),
            ["-2.00", "-1.60"]
        );

        // With length_penalty = 0 the raw cumulative logprob decides.
        let ranked = BeamSelector {
            length_penalty: 0.0,
            ..selector(2)
        }
        .finalize(completed, live);
        assert_eq!(
            summary(&ranked).iter().map(|(_, cum, _)| cum.as_str()).collect_vec(),
            ["-1.00", "-1.60"]
        );
    }

    #[test]
    fn rejects_non_default_ignored_params() {
        let mut request = TextRequest::for_test();
        request.sampling_params = SamplingParams {
            top_p: Some(1.0),
            repetition_penalty: Some(1.0),
            stop_token_ids: Some(vec![]),
            ..SamplingParams::default()
        };
        reject_unsupported_params(&request).expect("neutral values are accepted");

        request.sampling_params.top_p = Some(0.9);
        let error = reject_unsupported_params(&request).unwrap_err();
        expect!["`top_p` is not supported with beam search."].assert_eq(&error.to_string());

        let mut request = TextRequest::for_test();
        request.decode_options.stop_strings = Some(vec!["x".to_string()]);
        assert!(matches!(
            reject_unsupported_params(&request),
            Err(Error::UnsupportedWithBeamSearch { param: "stop" })
        ));
    }
}
