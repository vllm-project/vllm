use futures::Stream;
use futures_async_stream::try_stream;
use tracing::info;
use vllm_engine_core_client::protocol::{FinishReason, StopReason};
use vllm_llm::GenerateOutputStream;

use crate::ChatRequest;
use crate::backend::DynChatBackend;
use crate::error::{Error, Result};
use crate::incremental::IncrementalTextDecoder;

/// Internal decoded-text event emitted before higher-level reasoning adaptation.
///
/// For user-facing structured events, see [`crate::ChatEvent`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DecodedTextEvent {
    /// The request was accepted and streaming has started.
    Start,
    /// A delta of text has been decoded.
    ///
    /// Upper-level may further parse this as reasoning or tool calls.
    TextDelta { delta: String, text: String },
    /// Terminal event carrying the full decoded text and final metadata.
    Done {
        text: String,
        /// Raw cumulative output token IDs, including a terminal stop token when
        /// the engine emitted one.
        token_ids: Vec<u32>,
        finish_reason: Option<FinishReason>,
        stop_reason: Option<StopReason>,
    },
}

pub(crate) trait DecodedTextEventStream = Stream<Item = Result<DecodedTextEvent>> + Send + 'static;

/// Convert the output token stream from the `vllm_llm` layer into incrementally decoded text.
// TODO: apply small-string-optimization
#[try_stream(ok = DecodedTextEvent, error = Error)]
pub(crate) async fn decoded_text_event_stream(
    request: ChatRequest,
    backend: DynChatBackend,
    raw_stream: GenerateOutputStream,
) {
    yield DecodedTextEvent::Start;

    let mut decoder: Option<IncrementalTextDecoder> = None;
    let mut text = String::new();
    let mut token_ids = Vec::new();

    #[for_await]
    for next in raw_stream {
        let output: vllm_llm::GenerateOutput = next?;
        token_ids.extend_from_slice(&output.token_ids);
        let decoder = decoder.get_or_insert_with(|| {
            IncrementalTextDecoder::new(
                backend.clone(),
                &output.prompt_token_ids,
                request.sampling_params.skip_special_tokens,
            )
        });

        let suppress_terminal_stop_token = output.finished()
            && output.raw.finish_reason == Some(FinishReason::Stop)
            && !request.sampling_params.include_stop_str_in_output;
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
            if let Some(chunk) = decoder.push_token(token_id)? {
                delta.push_str(&chunk);
            }
        }
        if output.finished()
            && let Some(chunk) = decoder.flush()?
        {
            // Flush any remaining buffered text after the final token.
            delta.push_str(&chunk);
        }

        if !delta.is_empty() {
            text.push_str(&delta);
            yield DecodedTextEvent::TextDelta {
                delta,
                text: text.clone(),
            };
        }

        if output.finished() {
            info!(
                request_id = %request.request_id,
                finish_reason = ?output.raw.finish_reason,
                stop_reason = ?output.raw.stop_reason,
                text_length = text.chars().count(),
                token_count = token_ids.len(),
                "request finished with terminal output"
            );

            yield DecodedTextEvent::Done {
                text,
                token_ids,
                finish_reason: output.raw.finish_reason,
                stop_reason: output.raw.stop_reason,
            };
            return Ok(());
        }
    }

    Err(Error::StreamClosedBeforeTerminalOutput {
        request_id: request.request_id,
    })?;
}
