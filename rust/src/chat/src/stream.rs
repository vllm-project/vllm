use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use futures_async_stream::try_stream;
use vllm_llm::GenerateOutputStream;

use crate::backend::DynChatBackend;
use crate::error::{Error, Result};
use crate::event::ChatEvent;

/// Per-request stream of chat events.
pub struct ChatEventStream {
    request_id: String,
    inner: Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>,
}

impl ChatEventStream {
    pub(crate) fn new(
        request_id: String,
        backend: DynChatBackend,
        raw_stream: GenerateOutputStream,
    ) -> Self {
        Self {
            inner: chat_event_stream(request_id.clone(), backend, raw_stream),
            request_id,
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
/// events, by decoding tokens and diffing against previously emitted text.
#[try_stream(boxed, ok = ChatEvent, error = Error)]
async fn chat_event_stream(
    request_id: String,
    backend: DynChatBackend,
    raw_stream: GenerateOutputStream,
) {
    yield ChatEvent::Start;

    let mut emitted_text = String::new();

    #[for_await]
    for next in raw_stream {
        let output: vllm_llm::GenerateOutput = next?;
        let decoded = backend.decode(&output.token_ids, false)?;

        let delta = suffix_after_lcp(&emitted_text, &decoded).to_string();
        emitted_text = decoded.clone();

        if !delta.is_empty() {
            yield ChatEvent::TextDelta {
                delta,
                text: decoded.clone(),
            };
        }

        if output.finished() {
            yield ChatEvent::Done {
                text: decoded,
                finish_reason: output.raw.finish_reason,
                stop_reason: output.raw.stop_reason,
            };
            return Ok(());
        }
    }

    return Err(Error::StreamClosedBeforeTerminalOutput { request_id });
}

fn suffix_after_lcp<'a>(before: &str, after: &'a str) -> &'a str {
    let mut prefix_end = 0;
    for ((before_idx, before_ch), (after_idx, after_ch)) in
        before.char_indices().zip(after.char_indices())
    {
        if before_ch != after_ch {
            prefix_end = after_idx;
            return &after[prefix_end..];
        }
        prefix_end = after_idx + before_ch.len_utf8();
        let _ = before_idx;
    }

    if after.len() >= prefix_end {
        &after[prefix_end..]
    } else {
        ""
    }
}

#[cfg(test)]
mod tests {
    use super::suffix_after_lcp;

    #[test]
    fn suffix_after_lcp_returns_new_tail() {
        assert_eq!(suffix_after_lcp("hel", "hello"), "lo");
        assert_eq!(suffix_after_lcp("", "hello"), "hello");
        assert_eq!(suffix_after_lcp("hello", "hello"), "");
        assert_eq!(suffix_after_lcp("hello", "help"), "p");
    }
}
