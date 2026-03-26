use std::sync::Arc;

use futures_async_stream::try_stream;
use serde::{Deserialize, Serialize};
use tracing::info;
use vllm_llm::{FinishReason, GenerateOutputStream};

use super::logprobs::{
    DecodedLogprobs, DecodedPromptLogprobs, decode_logprobs, decode_prompt_logprobs,
};
use crate::backend::{IncrementalDecoder, TextBackend};
use crate::error::Error;

/// Request-neutral options for incremental text decoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextDecodeOptions {
    pub skip_special_tokens: bool,
    pub include_stop_str_in_output: bool,
}

impl Default for TextDecodeOptions {
    fn default() -> Self {
        Self {
            skip_special_tokens: true,
            include_stop_str_in_output: false,
        }
    }
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
    TextDelta {
        delta: String,
        logprobs: Option<DecodedLogprobs>,
    },
    /// Terminal event carrying the full decoded text and final metadata.
    Done {
        text: String,
        prompt_token_count: usize,
        /// Raw cumulative output token IDs, including a terminal stop token when
        /// the engine emitted one.
        token_ids: Vec<u32>,
        finish_reason: FinishReason,
    },
}

/// Convert the output token stream from the `vllm_llm` layer into incrementally decoded text.
#[try_stream(ok = DecodedTextEvent, error = Error)]
pub async fn decoded_text_event_stream<B: TextBackend + ?Sized>(
    request_id: String,
    backend: Arc<B>,
    raw_stream: GenerateOutputStream,
    decode_options: TextDecodeOptions,
) {
    let mut decoder: Option<Box<dyn IncrementalDecoder>> = None;
    let mut started = false;
    let mut prompt_token_count: Option<usize> = None;
    let mut text = String::new();
    let mut token_ids = Vec::new();

    #[for_await]
    for next in raw_stream {
        let output = next?;
        token_ids.extend_from_slice(&output.token_ids);

        let decoder = decoder.get_or_insert_with(|| {
            let prompt_token_ids = output
                .prompt_token_ids()
                .expect("first llm output must carry prompt token ids");
            prompt_token_count = Some(prompt_token_ids.len());
            backend.create_decode_stream(prompt_token_ids, decode_options.skip_special_tokens)
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

        let suppress_terminal_stop_token = output.finish_reason().is_some_and(|r| r.is_stop())
            && !decode_options.include_stop_str_in_output;
        let decodable_token_ids = if suppress_terminal_stop_token {
            // Match Python V1 token-stop detokenization by keeping the stop token
            // in metadata while excluding it from user-visible text.
            // TODO: when northbound stop strings are supported, mirror Python's
            // richer stop-string trimming behavior here as well.
            output
                .token_ids
                .split_last()
                .map(|(_, rest)| rest)
                .unwrap_or(&[])
        } else {
            &output.token_ids
        };

        let mut delta = String::new();
        for &token_id in decodable_token_ids {
            if let Some(chunk) = decoder.step(token_id)? {
                delta.push_str(&chunk);
            }
        }
        if output.finished()
            && let Some(chunk) = decoder.flush()?
        {
            // Flush any remaining buffered text after the final token.
            delta.push_str(&chunk);
        }
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

        if !delta.is_empty() || decoded_logprobs.is_some() {
            text.push_str(&delta);
            yield DecodedTextEvent::TextDelta {
                delta,
                logprobs: decoded_logprobs,
            };
        }

        if output.finished() {
            let finish_reason = output
                .finish_reason()
                .expect("terminal output must have a finish reason");
            info!(
                request_id = %request_id,
                finish_reason = ?finish_reason,
                text_length = text.chars().count(),
                token_count = token_ids.len(),
                "request finished with terminal output"
            );

            yield DecodedTextEvent::Done {
                text,
                prompt_token_count,
                token_ids,
                finish_reason,
            };
            return Ok(());
        }
    }

    Err(Error::StreamClosedBeforeTerminalOutput { request_id })?;
}
