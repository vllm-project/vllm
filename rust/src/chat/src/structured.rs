//! Adapts low-level decoded text updates into structured chat events.
//!
//! This module is the only place in `vllm-chat` that understands reasoning
//! separation. The token-to-text decoding path remains isolated in
//! `decoded.rs`, while this adapter consumes decoded text deltas and assembles
//! higher-level assistant content blocks.

use futures::{StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use reasoning_parser::ReasoningParser;
use thiserror_ext::AsReport;
use tracing::warn;
use vllm_engine_core_client::protocol::{FinishReason, StopReason};

use crate::decoded::{DecodedTextEvent, DecodedTextEventStream};
use crate::error::Error;
use crate::event::{AssistantBlockKind, AssistantContentBlock, AssistantMessage, ChatEvent};

/// One currently open assistant block being assembled from streamed deltas.
struct OpenBlock {
    /// Stable position of this block in the final assistant message.
    index: usize,
    /// Semantic kind of the block being assembled.
    kind: AssistantBlockKind,
    /// Accumulated text payload for the block.
    text: String,
}

/// Per-stream block assembly state.
///
/// The adapter maintains one currently open block and appends deltas to it until the semantic kind
/// changes or the stream terminates.
struct StructuredEventState {
    /// Final assistant message assembled so far.
    message: AssistantMessage,
    /// Currently open block, if any.
    open_block: Option<OpenBlock>,
    /// Optional reasoning parser for streams whose model supports reasoning separation.
    reasoning_parser: Option<Box<dyn ReasoningParser>>,
    /// Whether reasoning parsing has already failed for this stream.
    reasoning_parser_failed: bool,
}

impl StructuredEventState {
    /// Create one fresh assembly state for a new streamed response.
    fn new(reasoning_parser: Option<Box<dyn ReasoningParser>>) -> Self {
        Self {
            message: AssistantMessage::default(),
            open_block: None,
            reasoning_parser,
            reasoning_parser_failed: false,
        }
    }

    /// Convert one decoded text delta into zero or more structured chat events.
    fn process_text_delta(&mut self, delta: &str) -> Vec<ChatEvent> {
        let mut events = Vec::new();

        // If we have a reasoning parser, try to parse the delta into reasoning vs. normal text.
        if let Some(parser) = self.reasoning_parser.as_mut() {
            match parser.parse_reasoning_streaming_incremental(delta) {
                Ok(result) => {
                    self.push_delta(
                        AssistantBlockKind::Reasoning,
                        result.reasoning_text,
                        &mut events,
                    );
                    self.push_delta(AssistantBlockKind::Text, result.normal_text, &mut events);
                    return events;
                }
                Err(error) => {
                    if !self.reasoning_parser_failed {
                        warn!(
                            parser = parser.model_type(),
                            error = %error.as_report(),
                            "reasoning parser failed; falling back to plain text blocks"
                        );
                        self.reasoning_parser_failed = true;
                    }
                    self.reasoning_parser = None;
                    self.push_delta(AssistantBlockKind::Text, delta.to_string(), &mut events);
                    return events;
                }
            }
        }

        self.push_delta(AssistantBlockKind::Text, delta.to_string(), &mut events);
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
        self.close_open_block(&mut events);
        events.push(ChatEvent::Done {
            message: self.message.clone(),
            token_ids,
            finish_reason,
            stop_reason,
        });
        events
    }

    /// Append one semantic delta to the current block, or open a new block when
    /// the semantic kind changes.
    fn push_delta(&mut self, kind: AssistantBlockKind, delta: String, events: &mut Vec<ChatEvent>) {
        if delta.is_empty() {
            return;
        }

        match self.open_block.as_mut() {
            // If there's a currently open block of the same kind, append to it.
            Some(open) if open.kind == kind => {
                open.text.push_str(&delta);
                events.push(ChatEvent::BlockDelta {
                    index: open.index,
                    kind,
                    delta,
                });
            }
            // Otherwise, close the currently open block (if any) and start a new one.
            _ => {
                self.close_open_block(events);
                let index = self.message.content.len();
                self.open_block = Some(OpenBlock {
                    index,
                    kind,
                    text: delta.clone(),
                });
                events.push(ChatEvent::BlockStart { index, kind });
                events.push(ChatEvent::BlockDelta { index, kind, delta });
            }
        }
    }

    /// Finalize the currently open block, if present.
    fn close_open_block(&mut self, events: &mut Vec<ChatEvent>) {
        let Some(open) = self.open_block.take() else {
            return;
        };

        let block = AssistantContentBlock::from_text_delta(open.kind, open.text);
        self.message.push_block(block.clone());
        events.push(ChatEvent::BlockEnd {
            index: open.index,
            block,
        });
    }
}

/// Wrap one decoded-text stream into the public structured chat event stream.
#[try_stream(ok = ChatEvent, error = Error)]
pub(crate) async fn structured_chat_event_stream(
    decoded_stream: impl DecodedTextEventStream,
    reasoning_parser: Option<Box<dyn ReasoningParser>>,
) {
    pin_mut!(decoded_stream);

    let mut state = StructuredEventState::new(reasoning_parser);

    while let Some(event) = decoded_stream.next().await.transpose()? {
        match event {
            DecodedTextEvent::Start => yield ChatEvent::Start,
            DecodedTextEvent::TextDelta { delta, .. } => {
                for structured_event in state.process_text_delta(&delta) {
                    yield structured_event;
                }
            }
            DecodedTextEvent::Done {
                token_ids,
                finish_reason,
                stop_reason,
                ..
            } => {
                for structured_event in state.finish(token_ids, finish_reason, stop_reason) {
                    yield structured_event;
                }
            }
        }
    }
}
