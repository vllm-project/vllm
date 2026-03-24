use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use serde::{Deserialize, Serialize};
use vllm_engine_core_client::protocol::{FinishReason, StopReason};

use crate::error::{Error, Result};
use crate::event::{AssistantContentBlock, AssistantMessage, ChatEvent};

/// Final structured assistant message plus terminal stream metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectedAssistantMessage {
    pub message: AssistantMessage,
    pub prompt_token_count: u32,
    pub token_ids: Vec<u32>,
    pub finish_reason: Option<FinishReason>,
    pub stop_reason: Option<StopReason>,
}

/// Per-request stream of chat events.
pub struct ChatEventStream {
    request_id: String,
    inner: Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>,
}

impl ChatEventStream {
    pub(crate) fn new(request_id: String, inner: impl crate::output::ChatEventStream) -> Self {
        Self {
            request_id,
            inner: Box::pin(inner),
        }
    }

    /// Return the request ID associated with this stream.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Collect the stream to completion and return the final assembled assistant message.
    pub async fn collect_message(mut self) -> Result<CollectedAssistantMessage> {
        use futures::StreamExt as _;

        let mut message = AssistantMessage::default();
        while let Some(event) = self.next().await.transpose()? {
            match event {
                ChatEvent::BlockEnd { block, .. } => message.push_block(block),
                ChatEvent::Done {
                    message: done,
                    prompt_token_count,
                    token_ids,
                    finish_reason,
                    stop_reason,
                } => {
                    return Ok(CollectedAssistantMessage {
                        message: done,
                        prompt_token_count,
                        token_ids,
                        finish_reason,
                        stop_reason,
                    });
                }
                ChatEvent::ToolCallEnd { call, .. } => {
                    message.push_block(AssistantContentBlock::ToolCall(call));
                }
                ChatEvent::Start
                | ChatEvent::BlockStart { .. }
                | ChatEvent::BlockDelta { .. }
                | ChatEvent::ToolCallStart { .. }
                | ChatEvent::ToolCallArgumentsDelta { .. } => {}
            }
        }

        // Note: this is actually unreachable, as the underlying stream always emit an error on
        // unexpected close.
        Err(Error::StreamClosedBeforeTerminalOutput {
            request_id: self.request_id,
        })
    }
}

impl Stream for ChatEventStream {
    type Item = Result<ChatEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use futures::stream;

    use super::ChatEventStream;
    use crate::error::Error;
    use crate::event::ChatEvent;

    #[tokio::test]
    async fn collect_message_requires_terminal_done_event() {
        let stream = ChatEventStream::new(
            "chat-missing-done".to_string(),
            stream::iter([Ok(ChatEvent::Start)]),
        );

        let error = stream.collect_message().await.expect_err("missing done");
        assert!(matches!(
            error,
            Error::StreamClosedBeforeTerminalOutput { request_id }
            if request_id == "chat-missing-done"
        ));
    }
}
