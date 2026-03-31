use std::sync::Arc;

use futures_async_stream::try_stream;
use serde::{Deserialize, Serialize};
use tracing::info;
use vllm_engine_core_client::protocol::StopReason;
use vllm_llm::{FinishReason, GenerateOutputStream};

use super::logprobs::{
    DecodedLogprobs, DecodedPromptLogprobs, decode_logprobs, decode_prompt_logprobs,
};
use crate::backend::TextBackend;
use crate::error::Error;
use crate::incremental::IncrementalDecoder;
use crate::take;

/// Request-neutral options for incremental text decoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextDecodeOptions {
    pub skip_special_tokens: bool,
    pub include_stop_str_in_output: bool,
    pub stop_strings: Option<Vec<String>>,
}

impl Default for TextDecodeOptions {
    fn default() -> Self {
        Self {
            skip_special_tokens: true,
            include_stop_str_in_output: false,
            stop_strings: None,
        }
    }
}

/// Terminal metadata carried on the final [`DecodedTextEvent`].
#[derive(Debug, Clone, PartialEq)]
pub struct Finished {
    pub prompt_token_count: usize,
    pub output_token_count: usize,
    pub finish_reason: FinishReason,
}

/// Internal decoded-text event emitted before higher-level assistant adaptation.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedTextEvent {
    /// The request has reached the point where prompt-scoped decoding metadata is ready.
    Start {
        /// Number of prompt tokens for this request.
        prompt_token_count: usize,
        /// Once-only prompt logprobs metadata, when requested.
        ///
        /// The first prompt token is carried separately because it has no left context to score
        /// against; `scored_positions` covers the remaining prompt positions.
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    /// A delta of text has been decoded, optionally alongside token-position logprobs.
    ///
    /// `delta` is the newly visible decoded text fragment for this update.
    ///
    /// `logprobs` covers the newly generated token positions from the same update, but is not
    /// guaranteed to align with `delta` by character span. One update may carry token logprobs
    /// but no newly visible text yet, and one visible text fragment may reflect multiple token
    /// positions becoming decodable together.
    ///
    /// Upper-level may further parse `delta` as reasoning or tool calls.
    ///
    /// When `finished` is `Some`, this is the terminal event for the request.
    TextDelta {
        delta: String,
        token_ids: Vec<u32>,
        logprobs: Option<DecodedLogprobs>,
        finished: Option<Finished>,
    },
}

/// Convert the output token stream from the `vllm_llm` layer into incrementally decoded text.
#[try_stream(ok = DecodedTextEvent, error = Error)]
pub async fn decoded_text_event_stream<B: TextBackend + ?Sized>(
    request_id: String,
    backend: Arc<B>,
    raw_stream: GenerateOutputStream,
    mut decode_options: TextDecodeOptions,
    intermediate: bool,
) {
    let mut decoder: Option<Box<dyn IncrementalDecoder>> = None;
    let mut started = false;
    let mut prompt_token_count: Option<usize> = None;
    let mut token_ids = Vec::new();
    let mut output_token_count: usize = 0;
    let mut logprobs: Option<DecodedLogprobs> = None;

    #[for_await]
    for next in raw_stream {
        let mut output = next?;

        let decoder = decoder.get_or_insert_with(|| {
            let prompt_token_ids = output
                .prompt_token_ids()
                .expect("first llm output must carry prompt token ids");
            prompt_token_count = Some(prompt_token_ids.len());
            backend.create_decode_stream(
                prompt_token_ids,
                decode_options.skip_special_tokens,
                // If we are excluding stop strings from output, we need to buffer
                // the output so that we don't return the beginning of a stop string
                // when streaming the outputs.
                match decode_options.include_stop_str_in_output {
                    true => 0,
                    false => {
                        decode_options
                            .stop_strings
                            .as_ref()
                            .and_then(|stops| stops.iter().map(|ss| ss.len()).max())
                            .unwrap_or(1)
                            - 1
                    }
                },
            )
        });
        let prompt_token_count =
            prompt_token_count.expect("first llm output must carry prompt token ids");

        if !started {
            yield DecodedTextEvent::Start {
                prompt_token_count,
                prompt_logprobs: output
                    .prompt_logprobs()
                    .map(|logprobs| {
                        decode_prompt_logprobs(
                            backend.as_ref(),
                            output
                                .prompt_token_ids()
                                .expect("first llm output must carry prompt token ids"),
                            logprobs,
                            decode_options.skip_special_tokens,
                        )
                    })
                    .transpose()?,
            };
            started = true;
        }

        let mut finish_reason = output.finish_reason();
        let suppress_terminal_stop_token = finish_reason.as_ref().is_some_and(|r| r.is_stop())
            && !decode_options.include_stop_str_in_output;
        let decodable_token_ids = if suppress_terminal_stop_token {
            // Match Python V1 token-stop detokenization by keeping the stop token
            // in metadata while excluding it from user-visible text.
            output
                .token_ids
                .split_last()
                .map(|(_, rest)| rest)
                .unwrap_or(&[])
        } else {
            &output.token_ids
        };

        let mut delta: Option<String> = None;
        let mut truncate_output_to = None;
        let mut tuncate_tokens_to = None;
        for (tok_idx, &token_id) in decodable_token_ids.iter().enumerate() {
            let new_bytes = decoder.push_token(token_id)?;
            if let Some(stops) = decode_options.stop_strings.as_mut()
                && let Some((idx, off)) = matches_stop_string(stops, decoder.output(), new_bytes)
            {
                let stop_str = stops.swap_remove(idx);
                truncate_output_to = match decode_options.include_stop_str_in_output {
                    true => Some(off + stop_str.len()),
                    false => Some(off),
                };
                finish_reason = Some(FinishReason::Stop(Some(StopReason::Text(stop_str))));
                tuncate_tokens_to = Some(tok_idx + 1);
                break;
            }

            if intermediate && let Some(chunk) = decoder.next_chunk() {
                if let Some(delta_str) = delta.as_mut() {
                    delta_str.push_str(&chunk);
                } else {
                    delta = Some(chunk);
                }
            }
        }

        let mut new_token_ids = take(&mut output.token_ids);

        // Trim tokens and logprobs if we matched stop string.
        if let Some(num_tokens) = tuncate_tokens_to {
            new_token_ids.truncate(num_tokens);
            if let Some(logprobs) = &mut output.logprobs {
                logprobs.positions.truncate(num_tokens);
            }
        }

        output_token_count += new_token_ids.len();

        let decoded_logprobs = output
            .logprobs
            .as_ref()
            .map(|logprobs| {
                decode_logprobs(
                    backend.as_ref(),
                    logprobs,
                    decode_options.skip_special_tokens,
                )
            })
            .transpose()?;

        if !intermediate {
            token_ids.extend(&new_token_ids);
            if let Some(dlp) = decoded_logprobs.as_ref() {
                logprobs
                    .get_or_insert_with(|| DecodedLogprobs { positions: vec![] })
                    .positions
                    .extend_from_slice(&dlp.positions);
            }
        }

        if let Some(reason) = finish_reason {
            // Flush any remaining buffered text.
            let (last_chunk, mut text) = decoder.flush(truncate_output_to)?;

            if intermediate {
                if let Some(chunk) = last_chunk {
                    if let Some(delta_str) = delta.as_mut() {
                        delta_str.push_str(&chunk);
                    } else {
                        delta = Some(chunk);
                    }
                }
                token_ids = new_token_ids;
                logprobs = decoded_logprobs;
                text = delta.unwrap_or_default();
            }

            let text_len = text.len();

            info!(
                request_id = %request_id,
                finish_reason = ?reason,
                text_length_bytes = text_len,
                output_token_count = output_token_count,
                "request finished with terminal output"
            );

            yield DecodedTextEvent::TextDelta {
                delta: text,
                token_ids,
                logprobs,
                finished: Some(Finished {
                    prompt_token_count,
                    output_token_count,
                    finish_reason: reason,
                }),
            };
            return Ok(());
        }

        if intermediate {
            yield DecodedTextEvent::TextDelta {
                delta: delta.unwrap_or_default(),
                token_ids: new_token_ids,
                logprobs: decoded_logprobs,
                finished: None,
            };
        }
    }

    Err(Error::StreamClosedBeforeTerminalOutput { request_id })?;
}

/// If stop string matches, returns tuple
/// (index into stop string vec, byte index of first byte of stop string in output)
fn matches_stop_string(stops: &[String], output: &str, new_bytes: usize) -> Option<(usize, usize)> {
    // We compare byte subslices to avoid utf8 boundary problem
    let output = output.as_bytes();
    let next_off = (output.len() + 1) - new_bytes;
    stops
        .iter()
        .map(|ss| (ss.as_bytes(), ss.len(), next_off.saturating_sub(ss.len())))
        .enumerate()
        .find_map(|(ss_idx, (ss, len, start_off))| {
            output[start_off..]
                .windows(len)
                .rposition(|w| w == ss)
                .map(|pos| (ss_idx, start_off + pos))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // "say" appears earlier but "wor" is checked first (index 0)
        let result = matches_stop_string(&stops, "say wor", 1);
        assert_eq!(result, Some((0, 4)));
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
    fn stop_string_multibyte_utf8() {
        let stops = vec!["世界".to_string()];
        // "你好世界" is 12 bytes: 你(3) + 好(3) + 世(3) + 界(3)
        // "世界" starts at byte 6
        let result = matches_stop_string(&stops, "你好世界", 3);
        assert_eq!(result, Some((0, 6)));
    }
}
