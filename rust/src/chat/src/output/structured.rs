//! Adapts parsed assistant updates into structured chat events.
//!
//! This module remains the final assembly stage in `vllm-chat`. Token-to-text
//! decoding still lives in `decoded.rs`, while reasoning separation and tool
//! parsing are handled earlier by their own adapters. This stage consumes those
//! parsed deltas and assembles higher-level assistant content blocks.

use futures::{StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use vllm_engine_core_client::protocol::{FinishReason, StopReason};

use super::{AssistantEvent, AssistantEventStream};
use crate::error::Error;
use crate::event::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall, ChatEvent,
};

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
    /// Stable position of this tool call in the final assistant message.
    index: usize,
    /// Accumulated finalized tool-call payload.
    call: AssistantToolCall,
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
}

impl StructuredEventState {
    /// Create one fresh assembly state for a new streamed response.
    fn new() -> Self {
        Self {
            message: AssistantMessage::default(),
            open_text_block: None,
            open_tool_call: None,
        }
    }

    /// Convert one parsed text delta into zero or more structured chat events.
    fn process_text_delta(&mut self, kind: AssistantBlockKind, delta: String) -> Vec<ChatEvent> {
        let mut events = Vec::new();
        self.close_open_tool_call(&mut events);
        self.push_text_delta(kind, delta, &mut events);
        events
    }

    /// Start one new tool call, closing any incompatible open block first.
    fn start_tool_call(&mut self, id: String, name: String) -> Vec<ChatEvent> {
        let mut events = Vec::new();
        self.close_open_text_block(&mut events);
        self.close_open_tool_call(&mut events);

        let index = self.message.content.len();
        self.open_tool_call = Some(OpenToolCall {
            index,
            call: AssistantToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: String::new(),
            },
        });
        events.push(ChatEvent::ToolCallStart { index, id, name });
        events
    }

    /// Append one incremental tool-call arguments delta.
    fn push_tool_call_arguments(&mut self, id: String, delta: String) -> Vec<ChatEvent> {
        let mut events = Vec::new();
        let Some(open_tool_call) = self.open_tool_call.as_mut() else {
            return events;
        };
        open_tool_call.call.arguments.push_str(&delta);
        events.push(ChatEvent::ToolCallArgumentsDelta {
            index: open_tool_call.index,
            id,
            delta,
        });
        events
    }

    /// Finalize one tool call and append it to the assembled assistant message.
    fn end_tool_call(&mut self, call: AssistantToolCall) -> Vec<ChatEvent> {
        let mut events = Vec::new();
        let index = self
            .open_tool_call
            .as_ref()
            .map(|open_tool_call| open_tool_call.index)
            .unwrap_or(self.message.content.len());
        self.open_tool_call = None;
        self.message
            .push_block(AssistantContentBlock::ToolCall(call.clone()));
        events.push(ChatEvent::ToolCallEnd { index, call });
        events
    }

    /// Close any open block and emit the terminal `Done` event.
    fn finish(
        &mut self,
        token_ids: Vec<u32>,
        finish_reason: Option<FinishReason>,
        stop_reason: Option<StopReason>,
    ) -> Vec<ChatEvent> {
        let mut events = Vec::new();
        self.close_open_text_block(&mut events);
        self.close_open_tool_call(&mut events);
        events.push(ChatEvent::Done {
            message: self.message.clone(),
            token_ids,
            finish_reason,
            stop_reason,
        });
        events
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

        self.message
            .push_block(AssistantContentBlock::ToolCall(open_tool_call.call.clone()));
        events.push(ChatEvent::ToolCallEnd {
            index: open_tool_call.index,
            call: open_tool_call.call,
        });
    }
}

/// Wrap one parsed assistant stream into the public structured chat event
/// stream.
#[try_stream(ok = ChatEvent, error = Error)]
pub(super) async fn structured_chat_event_stream(stream: impl AssistantEventStream) {
    pin_mut!(stream);

    let mut state = StructuredEventState::new();

    while let Some(event) = stream.next().await.transpose()? {
        match event {
            AssistantEvent::Start => yield ChatEvent::Start,
            AssistantEvent::TextDelta { kind, delta } => {
                for next in state.process_text_delta(kind, delta) {
                    yield next;
                }
            }
            AssistantEvent::ToolCallStart { id, name } => {
                for next in state.start_tool_call(id, name) {
                    yield next;
                }
            }
            AssistantEvent::ToolCallArgumentsDelta { id, delta } => {
                for next in state.push_tool_call_arguments(id, delta) {
                    yield next;
                }
            }
            AssistantEvent::ToolCallEnd { call } => {
                for next in state.end_tool_call(call) {
                    yield next;
                }
            }
            AssistantEvent::Done {
                token_ids,
                finish_reason,
                stop_reason,
            } => {
                for next in state.finish(token_ids, finish_reason, stop_reason) {
                    yield next;
                }
            }
        }
    }
}
