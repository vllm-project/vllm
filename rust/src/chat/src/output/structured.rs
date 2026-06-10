//! Adapts parsed assistant updates into structured chat events.
//!
//! This module remains the final assembly stage in `vllm-chat`. Token-to-text
//! decoding still lives in `decoded.rs`, while reasoning separation and tool
//! parsing are handled earlier by their own adapters. This stage consumes those
//! parsed deltas and assembles higher-level assistant content blocks.

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{StreamExt as _, pin_mut};
use vllm_text::DecodedLogprobs;

use super::{AssistantEvent, AssistantEventStream};
use crate::error::Error;
use crate::event::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall, ChatEvent,
};
use crate::{FinishReason, Result};

/// One currently open assistant text-like block being assembled from streamed
/// deltas.
struct OpenTextBlock {
    /// Stable position of this block in the final assistant message.
    index: usize,
    /// Semantic kind of the block being assembled.
    kind: AssistantBlockKind,
    /// Accumulated text payload for the block.
    text: String,
}

/// One currently open assistant tool call being assembled from streamed deltas.
struct OpenToolCall {
    /// Stable ordinal of this tool call in the assistant tool-call list.
    index: usize,
    /// Stable tool-call ID exposed northbound.
    id: String,
    /// Function name.
    name: String,
    /// Incremental JSON arguments accumulated so far.
    arguments: String,
}

/// Per-stream block assembly state.
///
/// The adapter maintains at most one open text block and one open tool call,
/// and appends deltas to them until the semantic kind changes or the stream
/// terminates.
struct StructuredEventState {
    /// Final assistant message assembled so far.
    message: AssistantMessage,
    /// Currently open text or reasoning block, if any.
    open_text_block: Option<OpenTextBlock>,
    /// Currently open tool call, if any.
    open_tool_call: Option<OpenToolCall>,
    /// Next OpenAI-compatible tool-call ordinal.
    next_tool_call_index: usize,
}

impl StructuredEventState {
    /// Create one fresh assembly state for a new streamed response.
    fn new() -> Self {
        Self {
            message: AssistantMessage::default(),
            open_text_block: None,
            open_tool_call: None,
            next_tool_call_index: 0,
        }
    }

    /// Convert one parsed text delta into zero or more structured chat events.
    fn process_text_delta(
        &mut self,
        kind: AssistantBlockKind,
        delta: String,
    ) -> Result<Vec<ChatEvent>> {
        let mut events = Vec::new();
        self.close_open_tool_call(&mut events);
        self.push_text_delta(kind, delta, &mut events);
        Ok(events)
    }

    /// Forward per-update sample metadata without attaching it to text blocks.
    fn process_logprobs_delta(
        &mut self,
        logprobs: Option<DecodedLogprobs>,
        token_ids: Vec<u32>,
    ) -> Result<Vec<ChatEvent>> {
        Ok(vec![ChatEvent::LogprobsDelta {
            logprobs,
            token_ids,
        }])
    }

    /// Start one new tool call, closing any incompatible open block first.
    fn start_tool_call(&mut self, id: String, name: String) -> Result<Vec<ChatEvent>> {
        let mut events = Vec::new();
        self.close_open_text_block(&mut events);
        self.close_open_tool_call(&mut events);

        let index = self.next_tool_call_index;
        self.next_tool_call_index += 1;
        self.open_tool_call = Some(OpenToolCall {
            index,
            id: id.clone(),
            name: name.clone(),
            arguments: String::new(),
        });
        events.push(ChatEvent::ToolCallStart { index, id, name });
        Ok(events)
    }

    /// Append one incremental tool-call arguments delta.
    fn push_tool_call_arguments(&mut self, delta: String) -> Result<Vec<ChatEvent>> {
        let mut events = Vec::new();
        let Some(open_tool_call) = self.open_tool_call.as_mut() else {
            return Err(Error::ToolCallStreamInvariant {
                message: "received tool-call arguments delta without an open tool call".to_string(),
            });
        };
        open_tool_call.arguments.push_str(&delta);
        events.push(ChatEvent::ToolCallArgumentsDelta {
            index: open_tool_call.index,
            delta,
        });
        Ok(events)
    }

    /// Close any open block and emit the terminal `Done` event.
    fn finish(
        &mut self,
        prompt_token_count: usize,
        output_token_count: usize,
        finish_reason: FinishReason,
        kv_transfer_params: Option<serde_json::Value>,
    ) -> Result<Vec<ChatEvent>> {
        let mut events = Vec::new();
        self.close_open_text_block(&mut events);
        self.close_open_tool_call(&mut events);
        events.push(ChatEvent::Done {
            message: self.message.clone(),
            prompt_token_count,
            output_token_count,
            finish_reason,
            kv_transfer_params,
        });
        Ok(events)
    }

    /// Append one semantic text delta to the current block, or open a new block
    /// when the semantic kind changes.
    fn push_text_delta(
        &mut self,
        kind: AssistantBlockKind,
        delta: String,
        events: &mut Vec<ChatEvent>,
    ) {
        if delta.is_empty() {
            return;
        }

        match self.open_text_block.as_mut() {
            // If there's a currently open block of the same kind, append to it.
            Some(open_block) if open_block.kind == kind => {
                open_block.text.push_str(&delta);
                events.push(ChatEvent::BlockDelta {
                    index: open_block.index,
                    kind,
                    delta,
                });
            }
            // Otherwise, close the currently open block (if any) and start a
            // new one.
            _ => {
                self.close_open_text_block(events);
                let index = self.message.content.len();
                self.open_text_block = Some(OpenTextBlock {
                    index,
                    kind,
                    text: delta.clone(),
                });
                events.push(ChatEvent::BlockStart { index, kind });
                events.push(ChatEvent::BlockDelta { index, kind, delta });
            }
        }
    }

    /// Finalize the currently open text block, if present.
    fn close_open_text_block(&mut self, events: &mut Vec<ChatEvent>) {
        let Some(open_block) = self.open_text_block.take() else {
            return;
        };

        let block = match open_block.kind {
            AssistantBlockKind::Text => AssistantContentBlock::Text {
                text: open_block.text,
            },
            AssistantBlockKind::Reasoning => AssistantContentBlock::Reasoning {
                text: open_block.text,
            },
            AssistantBlockKind::ToolCall => {
                unreachable!("tool calls must not be assembled as text blocks")
            }
        };
        self.message.push_block(block.clone());
        events.push(ChatEvent::BlockEnd {
            index: open_block.index,
            block,
        });
    }

    /// Finalize the currently open tool call, if present.
    fn close_open_tool_call(&mut self, events: &mut Vec<ChatEvent>) {
        let Some(open_tool_call) = self.open_tool_call.take() else {
            return;
        };

        let call = AssistantToolCall {
            id: open_tool_call.id,
            name: open_tool_call.name,
            arguments: open_tool_call.arguments,
        };
        self.message.push_block(AssistantContentBlock::ToolCall(call.clone()));
        events.push(ChatEvent::ToolCallEnd {
            index: open_tool_call.index,
            call,
        });
    }
}

/// Wrap one parsed assistant stream into the public structured chat event
/// stream.
#[try_stream]
pub(crate) async fn structured_chat_event_stream(
    stream: impl AssistantEventStream,
    mut y: TryYielder<ChatEvent, Error>,
) -> Result<()> {
    pin_mut!(stream);

    let mut state = StructuredEventState::new();

    while let Some(event) = stream.next().await.transpose()? {
        match event {
            AssistantEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => {
                y.yield_ok(ChatEvent::Start {
                    prompt_token_ids,
                    prompt_logprobs,
                })
                .await;
            }
            AssistantEvent::TextDelta { kind, delta } => {
                for next in state.process_text_delta(kind, delta)? {
                    y.yield_ok(next).await;
                }
            }
            AssistantEvent::LogprobsDelta {
                logprobs,
                token_ids,
            } => {
                for next in state.process_logprobs_delta(logprobs, token_ids)? {
                    y.yield_ok(next).await;
                }
            }
            AssistantEvent::ToolCallStart { id, name } => {
                for next in state.start_tool_call(id, name)? {
                    y.yield_ok(next).await;
                }
            }
            AssistantEvent::ToolCallArgumentsDelta { delta } => {
                for next in state.push_tool_call_arguments(delta)? {
                    y.yield_ok(next).await;
                }
            }
            AssistantEvent::Done {
                prompt_token_count,
                output_token_count,
                finish_reason,
                kv_transfer_params,
            } => {
                for next in state.finish(
                    prompt_token_count,
                    output_token_count,
                    finish_reason,
                    kv_transfer_params,
                )? {
                    y.yield_ok(next).await;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt as _, stream};

    use super::structured_chat_event_stream;
    use crate::FinishReason;
    use crate::error::Error;
    use crate::event::{AssistantBlockKind, AssistantMessageExt as _, ChatEvent};
    use crate::output::AssistantEvent;

    #[tokio::test]
    async fn structured_stream_closes_tool_call_on_done() {
        let events = stream::iter(vec![
            Ok(AssistantEvent::ToolCallStart {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
            }),
            Ok(AssistantEvent::ToolCallArgumentsDelta {
                delta: r#"{"city":"Paris"}"#.to_string(),
            }),
            Ok(AssistantEvent::Done {
                prompt_token_count: 1,
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let events = structured_chat_event_stream(events)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .unwrap();

        assert!(matches!(events[0], ChatEvent::ToolCallStart { .. }));
        assert!(matches!(
            events[1],
            ChatEvent::ToolCallArgumentsDelta { .. }
        ));
        let ChatEvent::ToolCallEnd { call, .. } = &events[2] else {
            panic!("expected tool call end");
        };
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, r#"{"city":"Paris"}"#);
        let ChatEvent::Done { message, .. } = &events[3] else {
            panic!("expected done");
        };
        let tool_calls = message.tool_calls().collect::<Vec<_>>();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].arguments, r#"{"city":"Paris"}"#);
    }

    #[tokio::test]
    async fn structured_stream_closes_previous_tool_call_on_next_start() {
        let events = stream::iter(vec![
            Ok(AssistantEvent::ToolCallStart {
                id: "call_1".to_string(),
                name: "first".to_string(),
            }),
            Ok(AssistantEvent::ToolCallArgumentsDelta {
                delta: r#"{"a":1}"#.to_string(),
            }),
            Ok(AssistantEvent::ToolCallStart {
                id: "call_2".to_string(),
                name: "second".to_string(),
            }),
            Ok(AssistantEvent::ToolCallArgumentsDelta {
                delta: r#"{"b":2}"#.to_string(),
            }),
            Ok(AssistantEvent::Done {
                prompt_token_count: 1,
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let events = structured_chat_event_stream(events)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .unwrap();

        assert!(matches!(events[0], ChatEvent::ToolCallStart { .. }));
        assert!(matches!(
            events[1],
            ChatEvent::ToolCallArgumentsDelta { .. }
        ));
        let ChatEvent::ToolCallEnd { call, .. } = &events[2] else {
            panic!("expected first tool call end");
        };
        assert_eq!(call.name, "first");
        assert!(matches!(events[3], ChatEvent::ToolCallStart { .. }));
        let ChatEvent::Done { message, .. } = &events[6] else {
            panic!("expected done");
        };
        let tool_calls = message.tool_calls().collect::<Vec<_>>();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].name, "first");
        assert_eq!(tool_calls[1].name, "second");
    }

    #[tokio::test]
    async fn structured_stream_numbers_tool_calls_independent_of_text_blocks() {
        let events = stream::iter(vec![
            Ok(AssistantEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: "before".to_string(),
            }),
            Ok(AssistantEvent::ToolCallStart {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
            }),
            Ok(AssistantEvent::ToolCallArgumentsDelta {
                delta: r#"{"city":"Paris"}"#.to_string(),
            }),
            Ok(AssistantEvent::Done {
                prompt_token_count: 1,
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let events = structured_chat_event_stream(events)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .unwrap();

        assert!(matches!(
            events[0],
            ChatEvent::BlockStart {
                index: 0,
                kind: AssistantBlockKind::Text,
            }
        ));
        assert!(matches!(events[2], ChatEvent::BlockEnd { index: 0, .. }));
        assert!(matches!(
            events[3],
            ChatEvent::ToolCallStart { index: 0, .. }
        ));
        assert!(matches!(
            events[4],
            ChatEvent::ToolCallArgumentsDelta { index: 0, .. }
        ));
        assert!(matches!(events[5], ChatEvent::ToolCallEnd { index: 0, .. }));
    }

    #[tokio::test]
    async fn structured_stream_closes_tool_call_before_text() {
        let events = stream::iter(vec![
            Ok(AssistantEvent::ToolCallStart {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
            }),
            Ok(AssistantEvent::ToolCallArgumentsDelta {
                delta: r#"{"city":"Paris"}"#.to_string(),
            }),
            Ok(AssistantEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: "done".to_string(),
            }),
            Ok(AssistantEvent::Done {
                prompt_token_count: 1,
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let events = structured_chat_event_stream(events)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .unwrap();

        assert!(matches!(events[2], ChatEvent::ToolCallEnd { .. }));
        assert!(matches!(
            events[3],
            ChatEvent::BlockStart {
                kind: AssistantBlockKind::Text,
                ..
            }
        ));
        let ChatEvent::Done { message, .. } = &events[6] else {
            panic!("expected done");
        };
        assert_eq!(message.text(), "done");
        assert_eq!(message.tool_calls().count(), 1);
    }

    #[tokio::test]
    async fn structured_stream_rejects_arguments_without_open_tool_call() {
        let events = stream::iter(vec![Ok(AssistantEvent::ToolCallArgumentsDelta {
            delta: "{}".to_string(),
        })]);

        let err = structured_chat_event_stream(events)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .next()
            .expect("expected one event")
            .expect_err("expected invariant error");

        assert!(matches!(err, Error::ToolCallStreamInvariant { .. }));
    }
}
