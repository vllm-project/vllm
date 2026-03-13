use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll, ready};

use futures::Stream;
use futures::stream::FusedStream;
use vllm_llm::GenerateOutputStream;

use crate::event::ChatEvent;
use crate::tokenizer::DynTokenizer;

pub struct ChatEventStream {
    request_id: String,
    tokenizer: DynTokenizer,
    raw_stream: GenerateOutputStream,
    pending: VecDeque<ChatEvent>,
    emitted_text: String,
    done_emitted: bool,
    terminated: bool,
}

impl ChatEventStream {
    pub(crate) fn new(
        request_id: String,
        tokenizer: DynTokenizer,
        raw_stream: GenerateOutputStream,
    ) -> Self {
        let mut pending = VecDeque::new();
        pending.push_back(ChatEvent::Start {
            request_id: request_id.clone(),
        });
        Self {
            request_id,
            tokenizer,
            raw_stream,
            pending,
            emitted_text: String::new(),
            done_emitted: false,
            terminated: false,
        }
    }

    pub async fn collect_text(mut self) -> String {
        use futures::StreamExt as _;

        let mut text = String::new();
        while let Some(event) = self.next().await {
            match event {
                ChatEvent::TextDelta { text: next, .. } | ChatEvent::Done { text: next, .. } => {
                    text = next;
                }
                ChatEvent::Start { .. } | ChatEvent::Error { .. } => {}
            }
        }
        text
    }

    fn push_error(&mut self, message: impl Into<String>) {
        self.pending.push_back(ChatEvent::Error {
            request_id: self.request_id.clone(),
            message: message.into(),
        });
        self.terminated = true;
    }
}

impl Stream for ChatEventStream {
    type Item = ChatEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Poll::Ready(Some(event));
            }

            if self.terminated {
                return Poll::Ready(None);
            }

            let next = match ready!(Pin::new(&mut self.raw_stream).poll_next(cx)) {
                Some(next) => next,
                None if self.done_emitted => {
                    self.terminated = true;
                    return Poll::Ready(None);
                }
                None => {
                    self.push_error("request stream closed before terminal output");
                    continue;
                }
            };

            let output = match next {
                Ok(output) => output,
                Err(error) => {
                    self.push_error(error.to_string());
                    continue;
                }
            };

            let decoded = match self.tokenizer.decode(&output.token_ids, false) {
                Ok(decoded) => decoded,
                Err(error) => {
                    self.push_error(error.to_string());
                    continue;
                }
            };

            let delta = suffix_after_lcp(&self.emitted_text, &decoded).to_string();
            self.emitted_text = decoded.clone();

            if !delta.is_empty() {
                let request_id = self.request_id.clone();
                self.pending.push_back(ChatEvent::TextDelta {
                    request_id,
                    delta,
                    text: decoded.clone(),
                });
            }

            if output.raw.finished() {
                self.done_emitted = true;
                let request_id = self.request_id.clone();
                self.pending.push_back(ChatEvent::Done {
                    request_id,
                    text: decoded,
                    finish_reason: output.raw.finish_reason,
                    stop_reason: output.raw.stop_reason.clone(),
                });
            }
        }
    }
}

impl FusedStream for ChatEventStream {
    fn is_terminated(&self) -> bool {
        self.terminated && self.pending.is_empty()
    }
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
