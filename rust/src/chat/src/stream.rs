use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use trait_set::trait_set;
use vllm_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs};

use crate::FinishReason;
use crate::error::{Error, Result};
use crate::event::{AssistantContentBlock, AssistantMessage, ChatEvent};

/// Final structured assistant message plus terminal stream metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectedAssistantMessage {
    pub message: AssistantMessage,
    pub prompt_token_count: usize,
    pub prompt_token_ids: Arc<[u32]>,
    pub prompt_logprobs: Option<DecodedPromptLogprobs>,
    pub logprobs: Option<DecodedLogprobs>,
    pub token_ids: Vec<u32>,
    pub output_token_count: usize,
    pub finish_reason: FinishReason,
    /// Connector-specific KV transfer parameters for disaggregated serving.
    pub kv_transfer_params: Option<serde_json::Value>,
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

    /// Collect the stream to completion and return the final assembled
    /// assistant message.
    pub async fn collect_message(mut self) -> Result<CollectedAssistantMessage> {
        use futures::StreamExt as _;

        let mut message = AssistantMessage::default();
        let mut prompt_logprobs = None;
        let mut prompt_token_ids: Arc<[u32]> = Arc::from([]);
        let mut logprob_positions: Vec<DecodedPositionLogprobs> = Vec::new();
        let mut token_ids: Vec<u32> = Vec::new();
        while let Some(event) = self.next().await.transpose()? {
            match event {
                ChatEvent::Start {
                    prompt_logprobs: start_prompt_logprobs,
                    prompt_token_ids: start_prompt_token_ids,
                } => {
                    prompt_logprobs = start_prompt_logprobs;
                    prompt_token_ids = start_prompt_token_ids;
                }
                ChatEvent::BlockEnd { block, .. } => message.push_block(block),
                ChatEvent::LogprobsDelta {
                    logprobs,
                    token_ids: delta_ids,
                } => {
                    if let Some(logprobs) = logprobs {
                        logprob_positions.extend(logprobs.positions);
                    }
                    token_ids.extend(delta_ids);
                }
                ChatEvent::Done {
                    message: done,
                    prompt_token_count,
                    output_token_count,
                    finish_reason,
                    kv_transfer_params,
                } => {
                    return Ok(CollectedAssistantMessage {
                        message: done,
                        prompt_token_count,
                        prompt_token_ids,
                        prompt_logprobs,
                        logprobs: (!logprob_positions.is_empty()).then_some(DecodedLogprobs {
                            positions: logprob_positions,
                        }),
                        token_ids,
                        output_token_count,
                        finish_reason,
                        kv_transfer_params,
                    });
                }
                ChatEvent::ToolCallEnd { call, .. } => {
                    message.push_block(AssistantContentBlock::ToolCall(call));
                }
                ChatEvent::BlockStart { .. }
                | ChatEvent::BlockDelta { .. }
                | ChatEvent::ToolCallStart { .. }
                | ChatEvent::ToolCallArgumentsDelta { .. } => {}
            }
        }

        // Note: this is actually unreachable, as the underlying stream always emit an
        // error on unexpected close.
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

trait_set! {
    pub trait ChatEventStreamTrait = Stream<Item = Result<ChatEvent>> + Send + 'static;
}

#[cfg(test)]
mod tests {

    use futures::stream;
    use vllm_llm::FinishReason;
    use vllm_text::{
        DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
    };

    use super::{ChatEventStream, CollectedAssistantMessage};
    use crate::error::Error;
    use crate::event::ChatEvent;

    #[tokio::test]
    async fn collect_message_requires_terminal_done_event() {
        let stream = ChatEventStream::new(
            "chat-missing-done".to_string(),
            stream::iter([Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            })]),
        );

        let error = stream.collect_message().await.expect_err("missing done");
        assert!(matches!(
            error,
            Error::StreamClosedBeforeTerminalOutput { request_id }
            if request_id == "chat-missing-done"
        ));
    }

    #[tokio::test]
    async fn collect_message_retains_prompt_and_sample_logprobs() {
        let stream = ChatEventStream::new(
            "chat-logprobs".to_string(),
            stream::iter(vec![
                Ok(ChatEvent::Start {
                    prompt_token_ids: vec![10, 11].into(),
                    prompt_logprobs: Some(DecodedPromptLogprobs {
                        first_token_id: 0,
                        first_token: "o".to_string(),
                        scored_positions: vec![DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token_id: 0,
                                token: "p".to_string(),
                                logprob: -0.1,
                                rank: 1,
                            }],
                        }],
                    }),
                }),
                Ok(ChatEvent::LogprobsDelta {
                    logprobs: Some(DecodedLogprobs {
                        positions: vec![DecodedPositionLogprobs {
                            entries: vec![DecodedTokenLogprob {
                                token_id: 0,
                                token: "a".to_string(),
                                logprob: -0.2,
                                rank: 1,
                            }],
                        }],
                    }),
                    token_ids: vec![],
                }),
                Ok(ChatEvent::Done {
                    message: Default::default(),
                    prompt_token_count: 2,
                    output_token_count: 1,
                    finish_reason: FinishReason::stop_eos(),
                    kv_transfer_params: None,
                }),
            ]),
        );

        let collected = stream.collect_message().await.unwrap();
        assert_eq!(
            collected,
            CollectedAssistantMessage {
                message: Default::default(),
                prompt_token_count: 2,
                prompt_token_ids: vec![10, 11].into(),
                prompt_logprobs: Some(DecodedPromptLogprobs {
                    first_token_id: 0,
                    first_token: "o".to_string(),
                    scored_positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "p".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "a".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![],
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }
        );
    }
}
