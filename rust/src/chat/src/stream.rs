use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use futures_async_stream::try_stream;
use vllm_engine_core_client::protocol::FinishReason;
use vllm_llm::GenerateOutputStream;

use crate::ChatRequest;
use crate::backend::DynChatBackend;
use crate::error::{Error, Result};
use crate::event::ChatEvent;
use crate::incremental::IncrementalTextDecoder;

/// Per-request stream of chat events.
pub struct ChatEventStream {
    request_id: String,
    inner: Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>,
}

impl ChatEventStream {
    pub(crate) fn new(
        request: ChatRequest,
        backend: DynChatBackend,
        raw_stream: GenerateOutputStream,
    ) -> Self {
        Self {
            request_id: request.request_id.clone(),
            inner: chat_event_stream(request.clone(), backend, raw_stream),
        }
    }

    /// Return the request ID associated with this stream.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Collect the stream to completion and return the final cumulative text.
    pub async fn collect_text(mut self) -> Result<String> {
        use futures::StreamExt as _;

        let mut text = String::new();
        while let Some(event) = self.next().await.transpose()? {
            match event {
                ChatEvent::TextDelta { text: next, .. } | ChatEvent::Done { text: next, .. } => {
                    text = next;
                }
                ChatEvent::Start => {}
            }
        }
        Ok(text)
    }
}

impl Stream for ChatEventStream {
    type Item = Result<ChatEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// Convert the output token stream from the `vllm_llm` layer into a stream of higher-level chat
/// events by incrementally decoding token deltas into text.
// TODO: apply small-string-optimization
#[try_stream(boxed, ok = ChatEvent, error = Error)]
async fn chat_event_stream(
    request: ChatRequest,
    backend: DynChatBackend,
    raw_stream: GenerateOutputStream,
) {
    yield ChatEvent::Start;

    let mut decoder: Option<IncrementalTextDecoder> = None;
    let mut text = String::new();

    #[for_await]
    for next in raw_stream {
        let output: vllm_llm::GenerateOutput = next?;
        let decoder = decoder.get_or_insert_with(|| {
            IncrementalTextDecoder::new(backend.clone(), &output.prompt_token_ids)
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
            yield ChatEvent::TextDelta {
                delta,
                text: text.clone(),
            };
        }

        if output.finished() {
            yield ChatEvent::Done {
                text,
                finish_reason: output.raw.finish_reason,
                stop_reason: output.raw.stop_reason,
            };
            return Ok(());
        }
    }

    return Err(Error::StreamClosedBeforeTerminalOutput {
        request_id: request.request_id,
    });
}
