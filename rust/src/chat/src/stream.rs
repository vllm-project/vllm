use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;

use crate::error::Result;
use crate::event::{AssistantMessage, ChatEvent};

/// Per-request stream of chat events.
pub struct ChatEventStream {
    request_id: String,
    inner: Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>,
}

impl ChatEventStream {
    pub(crate) fn new(
        request_id: String,
        inner: Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>,
    ) -> Self {
        Self { request_id, inner }
    }

    /// Return the request ID associated with this stream.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Collect the stream to completion and return the final assembled assistant message.
    pub async fn collect_message(mut self) -> Result<AssistantMessage> {
        use futures::StreamExt as _;

        let mut message = AssistantMessage::default();
        while let Some(event) = self.next().await.transpose()? {
            match event {
                ChatEvent::BlockEnd { block, .. } => message.push_block(block),
                ChatEvent::Done { message: done, .. } => return Ok(done),
                ChatEvent::Start | ChatEvent::BlockStart { .. } | ChatEvent::BlockDelta { .. } => {}
            }
        }
        Ok(message)
    }
}

impl Stream for ChatEventStream {
    type Item = Result<ChatEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}
