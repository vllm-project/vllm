// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tracing::{Level, debug, trace};
use vllm_engine_core_client::AbortCause;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_llm::{FinishReason, GenerateOutput, TokenUsage};
use vllm_tokenizer::{DecodedText, DynTokenizer, IncrementalDecoder};

use super::logprobs::{
    DecodedLogprobs, DecodedPromptLogprobs, decode_logprobs, decode_prompt_logprobs,
};
use crate::error::Error;

/// Request-neutral options for incremental text decoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextDecodeOptions {
    pub skip_special_tokens: bool,
    pub include_stop_str_in_output: bool,
    pub stop_strings: Option<Vec<String>>,
    /// Minimum number of tokens to generate before stop-string checking kicks
    /// in. Stop strings found within the first `min_tokens` tokens are
    /// ignored.
    pub min_tokens: u32,
}

impl Default for TextDecodeOptions {
    fn default() -> Self {
        Self {
            skip_special_tokens: true,
            include_stop_str_in_output: false,
            stop_strings: None,
            min_tokens: 0,
        }
    }
}

/// Terminal metadata carried on the final [`DecodedTextEvent`].
#[derive(Debug, Clone, PartialEq)]
pub struct Finished {
    pub usage: TokenUsage,
    pub finish_reason: FinishReason,
    /// Connector-specific KV transfer parameters for disaggregated serving.
    pub kv_transfer_params: Option<serde_json::Value>,
    /// Connector-specific encoder cache transfer parameters for disaggregated
    /// serving.
    pub ec_transfer_params: Option<serde_json::Value>,
}

/// Sample metadata emitted by one engine output update.
///
/// Token IDs and logprobs share the engine-update clock and align by generated
/// token position. They may arrive before the corresponding decoded text is
/// visible or alongside decoded text attributed to earlier updates.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SampledDelta {
    /// Token IDs produced by this engine update, after stop-string truncation.
    pub token_ids: Vec<u32>,
    /// Per-position logprobs aligned with `token_ids`, when requested.
    pub logprobs: Option<DecodedLogprobs>,
}

/// Internal decoded-text event emitted before higher-level assistant
/// adaptation.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedTextEvent {
    /// The request has reached the point where prompt-scoped decoding metadata
    /// is ready.
    Start {
        /// The actual prompt token IDs for this request.
        prompt_token_ids: Arc<[u32]>,
        /// Once-only prompt logprobs metadata, when requested.
        ///
        /// The first prompt token is carried separately because it has no left
        /// context to score against; `scored_positions` covers the
        /// remaining prompt positions.
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    /// Decoded text and sampled metadata observed during one output update.
    ///
    /// `decoded` follows the decoder clock. It may be empty while tokens remain
    /// undecodable, and may later contain token attributions from earlier
    /// engine updates. Tokens suppressed before decoding are absent.
    ///
    /// `sampled` follows the engine-update clock. Its token IDs and logprobs
    /// align with each other. `decoded.attributions` has an independent decoder
    /// clock and positional sequence.
    ///
    /// Upper-level may further parse `decoded.text` as reasoning or tool calls.
    ///
    /// When `finished` is `Some`, this is the terminal event for the request.
    TextDelta {
        decoded: DecodedText,
        sampled: SampledDelta,
        /// Heap-backed because terminal metadata is large and appears once per
        /// request, while every streaming event carries this enum variant.
        finished: Option<Box<Finished>>,
    },
}

/// Convert the output token stream from the `vllm_llm` layer into incrementally
/// decoded text.
#[try_stream]
pub async fn decoded_text_event_stream(
    request_id: String,
    tokenizer: DynTokenizer,
    mut raw_stream: impl Stream<Item = vllm_llm::Result<GenerateOutput>> + Unpin,
    mut decode_options: TextDecodeOptions,
    intermediate: bool,
    mut y: TryYielder<DecodedTextEvent, Error>,
) -> crate::Result<()> {
    let mut decoder: Option<Box<dyn IncrementalDecoder>> = None;
    let mut prompt_token_count = 0_usize;
    let mut cached_token_count = 0_usize;
    let mut sampled_token_ids = Vec::new();
    let mut output_token_count: usize = 0;
    let mut sampled_logprobs: Option<DecodedLogprobs> = None;

    while let Some(next) = raw_stream.next().await {
        let output = next?;
        cached_token_count = cached_token_count.max(output.cached_token_count);

        // If it's the first output, init states and yield `Start` event.
        if decoder.is_none() {
            let prompt_token_ids =
                output.prompt_token_ids().expect("first llm output must carry prompt token ids");
            prompt_token_count = prompt_token_ids.len();

            let dec = tokenizer.create_decode_stream(
                prompt_token_ids,
                decode_options.skip_special_tokens,
                // If we are excluding stop strings from output, we need to buffer
                // the output so that we don't return the beginning of a stop string
                // when streaming the outputs.
                match decode_options.include_stop_str_in_output {
                    true => 0,
                    false => decode_options
                        .stop_strings
                        .as_ref()
                        .and_then(|stops| stops.iter().map(|ss| ss.len()).max())
                        .unwrap_or(1)
                        .saturating_sub(1),
                },
            );
            decoder = Some(dec);

            y.yield_ok(DecodedTextEvent::Start {
                prompt_token_ids: prompt_token_ids.clone(),
                prompt_logprobs: output
                    .prompt_logprobs()
                    .map(|logprobs| {
                        decode_prompt_logprobs(
                            tokenizer.as_ref(),
                            prompt_token_ids,
                            logprobs,
                            decode_options.skip_special_tokens,
                        )
                    })
                    .transpose()?,
            })
            .await;
        };
        let decoder = decoder.as_mut().unwrap();

        let kv_transfer_params = output.kv_transfer_params;
        let ec_transfer_params = output.ec_transfer_params;
        let mut finish_reason = output.finish_reason;
        let mut stop_str_matched = false;
        let suppress_terminal_stop_token = finish_reason.as_ref().is_some_and(|r| r.is_stop())
            && !decode_options.include_stop_str_in_output;
        let decodable_token_ids = if suppress_terminal_stop_token {
            // Match Python V1 token-stop detokenization by keeping the stop token
            // in metadata while excluding it from user-visible text.
            output.token_ids.split_last().map(|(_, rest)| rest).unwrap_or(&[])
        } else {
            &output.token_ids
        };

        let mut decoded = DecodedText::default();
        let mut truncate_output_to = None;
        let mut truncate_tokens_to = None;
        for (tok_idx, &token_id) in decodable_token_ids.iter().enumerate() {
            let new_bytes = decoder.push_token(token_id)?;
            if output_token_count + tok_idx + 1 > decode_options.min_tokens as usize
                && let Some(stops) = decode_options.stop_strings.as_mut()
                && let Some((idx, off)) = matches_stop_string(stops, decoder.output(), new_bytes)
            {
                let stop_str = stops.swap_remove(idx);
                truncate_output_to = match decode_options.include_stop_str_in_output {
                    true => Some(off + stop_str.len()),
                    false => Some(off),
                };
                finish_reason = Some(FinishReason::Stop(Some(StopReason::Text(stop_str))));
                truncate_tokens_to = Some(tok_idx + 1);
                stop_str_matched = true;

                break;
            }

            // TODO: avoid generating a chunk every time a token is pushed
            if intermediate && let Some(chunk) = decoder.next_chunk() {
                decoded.append(chunk);
            }
        }

        let mut new_token_ids = output.token_ids;
        let mut new_logprobs = output.logprobs;

        // Trim tokens and logprobs if we matched stop string.
        if let Some(num_tokens) = truncate_tokens_to {
            new_token_ids.truncate(num_tokens);
            if let Some(logprobs) = &mut new_logprobs {
                logprobs.positions.truncate(num_tokens);
            }
        }

        output_token_count += new_token_ids.len();

        let decoded_logprobs = new_logprobs
            .as_ref()
            .map(|logprobs| {
                decode_logprobs(
                    tokenizer.as_ref(),
                    logprobs,
                    decode_options.skip_special_tokens,
                )
            })
            .transpose()?;

        if !intermediate {
            sampled_token_ids.extend(&new_token_ids);
            if let Some(dlp) = decoded_logprobs.as_ref() {
                sampled_logprobs
                    .get_or_insert_with(|| DecodedLogprobs { positions: vec![] })
                    .positions
                    .extend_from_slice(&dlp.positions);
            }
        }

        if let Some(reason) = finish_reason {
            // Flush any remaining buffered text.
            let (last_chunk, full_decoded) = decoder.flush(truncate_output_to)?;
            let text_len = full_decoded.text.len();
            let full_text = tracing::enabled!(Level::TRACE).then(|| full_decoded.text.clone());
            let decoded = if intermediate {
                if let Some(chunk) = last_chunk {
                    decoded.append(chunk);
                }
                sampled_token_ids = new_token_ids;
                sampled_logprobs = decoded_logprobs;
                decoded
            } else {
                full_decoded
            };

            debug!(
                finish_reason = ?reason,
                text_length_bytes = text_len,
                output_token_count = output_token_count,
                "request finished with terminal output"
            );
            if let Some(full_text) = full_text {
                trace!(full_text, "request finished with terminal decoded text");
            }

            // Intentionally drop the stream with explicit cause, so that the engine core
            // can distinguish between such normal completion vs an unexpected
            // early drop.
            if stop_str_matched {
                AbortCause::StopStringMatched.drop_as(raw_stream);
            }

            y.yield_ok(DecodedTextEvent::TextDelta {
                decoded,
                sampled: SampledDelta {
                    token_ids: sampled_token_ids,
                    logprobs: sampled_logprobs,
                },
                finished: Some(Box::new(Finished {
                    usage: TokenUsage {
                        prompt_token_count,
                        output_token_count,
                        cached_token_count,
                    },
                    finish_reason: reason,
                    kv_transfer_params,
                    ec_transfer_params,
                })),
            })
            .await;
            return Ok(());
        }

        if intermediate {
            y.yield_ok(DecodedTextEvent::TextDelta {
                decoded,
                sampled: SampledDelta {
                    token_ids: new_token_ids,
                    logprobs: decoded_logprobs,
                },
                finished: None,
            })
            .await;
        }
    }

    Err(Error::StreamClosedBeforeTerminalOutput { request_id })
}

/// If stop string matches, returns tuple
/// (index into stop string vec, byte index of first byte of stop string in
/// output)
///
/// When several stop strings match within the newly generated text (for example
/// when a step appends multiple tokens, or one token decodes to several
/// characters), the stop string that completes earliest in the text is selected,
/// so the result matches appending one character at a time. Ties are broken by
/// stop-list order.
fn matches_stop_string(stops: &[String], output: &str, new_bytes: usize) -> Option<(usize, usize)> {
    // We compare byte subslices to avoid utf8 boundary problem
    let output = output.as_bytes();
    let next_off = (output.len() + 1) - new_bytes;
    stops
        .iter()
        .map(|ss| (ss.as_bytes(), ss.len(), next_off.saturating_sub(ss.len())))
        .enumerate()
        .filter_map(|(ss_idx, (ss, len, start_off))| {
            if len == 0 {
                return None;
            }
            output[start_off..]
                .windows(len)
                .position(|w| w == ss)
                .map(|pos| (ss_idx, start_off + pos, start_off + pos + len))
        })
        // `min_by_key` keeps the first minimum, so ties fall back to stop-list order.
        .min_by_key(|&(_, _, end)| end)
        .map(|(ss_idx, start, _)| (ss_idx, start))
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll};

    use futures::{Stream, StreamExt as _, stream};
    use vllm_engine_core_client::AbortCause;
    use vllm_llm::GenerateOutput;
    use vllm_tokenizer::{TokenAnchor, TokenAttribution, test_utils::TestTokenizer};

    use super::*;
    use crate::output::TextOutputStreamExt as _;

    /// Helper: run `decoded_text_event_stream` to completion and return the
    /// collected output.
    async fn run_to_completion(
        token_ids: Vec<u32>,
        decode_options: TextDecodeOptions,
    ) -> crate::output::CollectedTextOutput {
        let prompt: Arc<[u32]> = Arc::from([]);
        let raw_stream = stream::iter(vec![Ok(GenerateOutput::for_test(
            Some(prompt),
            token_ids,
            Some(FinishReason::Length),
        ))]);
        let tokenizer: DynTokenizer = Arc::new(TestTokenizer::new());
        decoded_text_event_stream("test".into(), tokenizer, raw_stream, decode_options, false)
            .collect_output()
            .await
            .unwrap()
    }

    /// Convert ASCII string to token IDs (one byte per token).
    fn ascii_tokens(s: &str) -> Vec<u32> {
        s.bytes().map(u32::from).collect()
    }

    fn opts(stop: &[&str], min_tokens: u32) -> TextDecodeOptions {
        TextDecodeOptions {
            stop_strings: Some(stop.iter().map(|s| s.to_string()).collect()),
            min_tokens,
            ..Default::default()
        }
    }

    fn visible(token_id: u32, byte_offset: u32) -> TokenAttribution {
        TokenAttribution {
            token_id,
            anchor: TokenAnchor::Visible { byte_offset },
        }
    }

    fn text_delta(
        text: impl Into<String>,
        attributions: impl IntoIterator<Item = TokenAttribution>,
        sampled_token_ids: Vec<u32>,
        finished: Option<Finished>,
    ) -> DecodedTextEvent {
        DecodedTextEvent::TextDelta {
            decoded: DecodedText {
                text: text.into(),
                attributions: attributions.into_iter().collect(),
            },
            sampled: SampledDelta {
                token_ids: sampled_token_ids,
                logprobs: None,
            },
            finished: finished.map(Box::new),
        }
    }

    async fn run_intermediate(
        outputs: Vec<GenerateOutput>,
        decode_options: TextDecodeOptions,
    ) -> Vec<DecodedTextEvent> {
        let tokenizer: DynTokenizer = Arc::new(TestTokenizer::new());
        decoded_text_event_stream(
            "test".into(),
            tokenizer,
            stream::iter(outputs.into_iter().map(Ok)),
            decode_options,
            true,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<crate::Result<Vec<_>>>()
        .unwrap()
    }

    #[tokio::test]
    async fn decoded_and_sampled_clocks_are_independent() {
        struct Case {
            name: &'static str,
            outputs: Vec<GenerateOutput>,
            expected: Vec<DecodedTextEvent>,
        }

        let prompt: Arc<[u32]> = Arc::from([]);
        let cases = [
            Case {
                name: "utf8 byte fallback resolves across engine updates",
                outputs: vec![
                    GenerateOutput::for_test(Some(Arc::clone(&prompt)), vec![0xe4], None),
                    GenerateOutput::for_test(None, vec![0xbd, 0xa0], Some(FinishReason::Length)),
                ],
                expected: vec![
                    DecodedTextEvent::Start {
                        prompt_token_ids: Arc::clone(&prompt),
                        prompt_logprobs: None,
                    },
                    text_delta("", [], vec![0xe4], None),
                    text_delta(
                        "你",
                        [visible(0xe4, 0), visible(0xbd, 0), visible(0xa0, 0)],
                        vec![0xbd, 0xa0],
                        Some(Finished {
                            usage: TokenUsage {
                                prompt_token_count: 0,
                                output_token_count: 3,
                                cached_token_count: 0,
                            },
                            finish_reason: FinishReason::Length,
                            kv_transfer_params: None,
                            ec_transfer_params: None,
                        }),
                    ),
                ],
            },
            Case {
                name: "terminal stop token remains sampled and bypasses decoder",
                outputs: vec![GenerateOutput::for_test(
                    Some(Arc::clone(&prompt)),
                    ascii_tokens("a!"),
                    Some(FinishReason::Stop(Some(StopReason::TokenId(b'!' as u32)))),
                )],
                expected: vec![
                    DecodedTextEvent::Start {
                        prompt_token_ids: Arc::clone(&prompt),
                        prompt_logprobs: None,
                    },
                    text_delta(
                        "a",
                        [visible(b'a' as u32, 0)],
                        ascii_tokens("a!"),
                        Some(Finished {
                            usage: TokenUsage {
                                prompt_token_count: 0,
                                output_token_count: 2,
                                cached_token_count: 0,
                            },
                            finish_reason: FinishReason::Stop(Some(StopReason::TokenId(
                                b'!' as u32,
                            ))),
                            kv_transfer_params: None,
                            ec_transfer_params: None,
                        }),
                    ),
                ],
            },
        ];

        for case in cases {
            let actual = run_intermediate(case.outputs, TextDecodeOptions::default()).await;
            assert_eq!(actual, case.expected, "{}", case.name);
        }
    }

    struct DropRecordingStream {
        next: Option<vllm_llm::Result<GenerateOutput>>,
        dropped_cause: Arc<Mutex<Option<AbortCause>>>,
    }

    impl Stream for DropRecordingStream {
        type Item = vllm_llm::Result<GenerateOutput>;

        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.next.take())
        }
    }

    impl Drop for DropRecordingStream {
        fn drop(&mut self) {
            *self.dropped_cause.lock().unwrap() = Some(AbortCause::current());
        }
    }

    // --- stop string stream tests ---

    #[tokio::test]
    async fn stream_stop_string_sets_task_local_abort_cause_on_raw_stream_drop() {
        let prompt: Arc<[u32]> = Arc::from([]);
        let dropped_cause = Arc::new(Mutex::new(None));
        let raw_stream = DropRecordingStream {
            next: Some(Ok(GenerateOutput::for_test(
                Some(prompt),
                ascii_tokens("hello"),
                Some(FinishReason::Length),
            ))),
            dropped_cause: Arc::clone(&dropped_cause),
        };
        let tokenizer: DynTokenizer = Arc::new(TestTokenizer::new());

        let output = decoded_text_event_stream(
            "test".into(),
            tokenizer,
            raw_stream,
            opts(&["ll"], 0),
            false,
        )
        .collect_output()
        .await
        .unwrap();

        assert_eq!(output.text, "he");
        assert!(output.finish_reason.is_stop());
        assert_eq!(
            *dropped_cause.lock().unwrap(),
            Some(AbortCause::StopStringMatched)
        );
    }

    #[tokio::test]
    async fn stream_stop_string_truncates_at_match() {
        let output = run_to_completion(ascii_tokens("hello"), opts(&["e"], 0)).await;
        assert_eq!(output.text, "h");
        assert!(output.finish_reason.is_stop());
    }

    #[tokio::test]
    async fn stream_stop_string_at_end() {
        let output = run_to_completion(ascii_tokens("abcxyz"), opts(&["xyz"], 0)).await;
        assert_eq!(output.text, "abc");
        assert!(output.finish_reason.is_stop());
    }

    #[tokio::test]
    async fn stream_stop_string_first_token() {
        let output = run_to_completion(ascii_tokens("xhello"), opts(&["x"], 0)).await;
        assert_eq!(output.text, "");
        assert!(output.finish_reason.is_stop());
    }

    #[tokio::test]
    async fn stream_stop_string_no_match_runs_to_completion() {
        let output = run_to_completion(ascii_tokens("hello"), opts(&["z"], 0)).await;
        assert_eq!(output.text, "hello");
        assert_eq!(output.finish_reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn stream_empty_stop_string_does_not_underflow_buffer() {
        let output = run_to_completion(ascii_tokens("hello"), opts(&[""], 0)).await;
        assert_eq!(output.text, "hello");
        assert_eq!(output.finish_reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn stream_stop_string_multi_char() {
        let output = run_to_completion(ascii_tokens("say hello world"), opts(&["lo"], 0)).await;
        assert_eq!(output.text, "say hel");
        assert!(output.finish_reason.is_stop());
    }

    #[tokio::test]
    async fn stream_stop_string_first_of_multiple_wins() {
        // Both "ll" and "lo" are present; "ll" appears first in the output.
        let output = run_to_completion(ascii_tokens("hello"), opts(&["ll", "lo"], 0)).await;
        assert_eq!(output.text, "he");
        assert!(output.finish_reason.is_stop());
    }

    #[tokio::test]
    async fn stream_stop_string_include_in_output() {
        let output = run_to_completion(
            ascii_tokens("hello"),
            TextDecodeOptions {
                stop_strings: Some(vec!["ll".to_string()]),
                include_stop_str_in_output: true,
                ..Default::default()
            },
        )
        .await;
        assert_eq!(output.text, "hell");
        assert!(output.finish_reason.is_stop());
    }

    // --- min_tokens + stop string interaction ---

    #[tokio::test]
    async fn min_tokens_suppresses_early_stop_string() {
        // stop="e", min_tokens=3: the 'e' at token 2 is within the first 3 tokens,
        // so it should be skipped. No later 'e' exists, so output runs to completion.
        let output = run_to_completion(ascii_tokens("hello"), opts(&["e"], 3)).await;
        assert_eq!(output.text, "hello");
        assert_eq!(output.finish_reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn min_tokens_allows_stop_string_after_threshold() {
        // stop="e", min_tokens=2: the first 'e' at token 3 is past the threshold.
        let output = run_to_completion(ascii_tokens("greet"), opts(&["e"], 2)).await;
        assert_eq!(output.text, "gr");
        assert!(output.finish_reason.is_stop());
    }

    #[tokio::test]
    async fn min_tokens_zero_behaves_like_absent() {
        let output = run_to_completion(ascii_tokens("hello"), opts(&["e"], 0)).await;
        assert_eq!(output.text, "h");
        assert!(output.finish_reason.is_stop());
    }

    #[test]
    fn stop_string_matches_at_end() {
        let stops = vec!["wor".to_string()];
        // Output: "say wor", last byte 'r' was just added (new_bytes=1)
        let result = matches_stop_string(&stops, "say wor", 1);
        assert_eq!(result, Some((0, 4)));
    }

    #[test]
    fn stop_string_no_match() {
        let stops = vec!["xyz".to_string()];
        let result = matches_stop_string(&stops, "say wor", 1);
        assert_eq!(result, None);
    }

    #[test]
    fn stop_string_matches_first_of_multiple() {
        let stops = vec!["wor".to_string(), "say".to_string()];
        // Only "wor" can complete in the 1-new-byte window; the "say" at index 0
        // is behind the window start and is not a candidate.
        let result = matches_stop_string(&stops, "say wor", 1);
        assert_eq!(result, Some((0, 4)));
    }

    /// Several stop strings can land in the same window when a step appends
    /// multiple tokens (speculative decoding) or when one token decodes to
    /// several characters. The earliest-completing one must win over stop-list
    /// order, so that a batched step agrees with character-at-a-time appending.
    #[test]
    fn stop_string_earliest_completing_wins_regardless_of_list_order() {
        // " The user is a": " is a" (5 bytes) was appended in one step. Both
        // "is" (index 10, completes at 12) and " a" (index 12, completes at 14)
        // land in the window. "is" completes earlier, so it wins either order.
        for stops in [vec!["a", "is"], vec!["is", "a"]] {
            let owned: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
            let is_idx = stops.iter().position(|s| *s == "is").unwrap();
            let result = matches_stop_string(&owned, " The user is a", " is a".len());
            assert_eq!(result, Some((is_idx, 10)), "stop list order {stops:?}");
        }
    }

    /// Both stops complete at the same offset, so stop-list order decides. This
    /// pins `min_by_key`'s first-minimum behavior, which is what implements the
    /// tie-break.
    #[test]
    fn stop_string_ties_broken_by_list_order() {
        for (stops, expected) in [(vec!["ab", "b"], (0, 0)), (vec!["b", "ab"], (0, 1))] {
            let owned: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
            let result = matches_stop_string(&owned, "ab", "ab".len());
            assert_eq!(result, Some(expected), "stop list order {stops:?}");
        }
    }

    #[test]
    fn stop_string_completion_position_not_start_position() {
        // "b" starts later than "abc" but completes earlier, so it must win.
        let stops = vec!["abc".to_string(), "b".to_string()];
        let result = matches_stop_string(&stops, "abc", 3);
        assert_eq!(result, Some((1, 1)));
    }

    #[test]
    fn stop_string_matches_second_of_multiple() {
        let stops = vec!["xyz".to_string(), "wor".to_string()];
        let result = matches_stop_string(&stops, "say wor", 1);
        assert_eq!(result, Some((1, 4)));
    }

    #[test]
    fn stop_string_matches_with_multiple_new_bytes() {
        let stops = vec!["wor".to_string()];
        // "say wor" where last 3 bytes "wor" were added at once
        let result = matches_stop_string(&stops, "say wor", 3);
        assert_eq!(result, Some((0, 4)));
    }

    #[test]
    fn stop_string_matches_leftmost_with_multiple_new_bytes() {
        let stops = vec!["\n".to_string()];
        let result = matches_stop_string(&stops, "Answer\n\n", 2);
        assert_eq!(result, Some((0, 6)));
    }

    #[test]
    fn stop_string_matches_at_beginning() {
        let stops = vec!["say".to_string()];
        let result = matches_stop_string(&stops, "say wor", 7);
        assert_eq!(result, Some((0, 0)));
    }

    #[test]
    fn stop_string_exact_output() {
        let stops = vec!["abc".to_string()];
        let result = matches_stop_string(&stops, "abc", 3);
        assert_eq!(result, Some((0, 0)));
    }

    #[test]
    fn stop_string_single_char() {
        let stops = vec!["!".to_string()];
        let result = matches_stop_string(&stops, "hello!", 1);
        assert_eq!(result, Some((0, 5)));
    }

    #[test]
    fn stop_string_not_in_new_bytes_region() {
        let stops = vec!["say".to_string()];
        // "say" is in the output but before the new byte region.
        // new_bytes=1 means only 'r' was added; "say" ended at byte 3,
        // but the search window starts at next_off - stop_len = 7+1-1 - 3 = 4.
        let result = matches_stop_string(&stops, "say wor", 1);
        assert_eq!(result, None);
    }

    #[test]
    fn stop_string_empty_list() {
        let stops: Vec<String> = vec![];
        let result = matches_stop_string(&stops, "hello", 1);
        assert_eq!(result, None);
    }

    #[test]
    fn empty_stop_string_is_ignored_by_matcher() {
        let stops = vec!["".to_string(), "world".to_string()];
        let result = matches_stop_string(&stops, "hello", 1);
        assert_eq!(result, None);
    }

    #[test]
    fn stop_string_multibyte_utf8() {
        let stops = vec!["世界".to_string()];
        // "你好世界" is 12 bytes: 你(3) + 好(3) + 世(3) + 界(3)
        // "世界" starts at byte 6
        let result = matches_stop_string(&stops, "你好世界", 3);
        assert_eq!(result, Some((0, 6)));
    }
}
