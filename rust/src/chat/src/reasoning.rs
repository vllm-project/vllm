//! Adapts decoded text updates into reasoning-aware assistant deltas.
//!
//! This stage sits between low-level token decoding and final block assembly.
//! It is the only place in the new pipeline that understands reasoning
//! separation: `decoded.rs` still only produces plain text deltas, while later
//! stages consume the semantic `Text` / `Reasoning` split emitted here.

use futures::{StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use reasoning_parser::ReasoningParser;
use thiserror_ext::AsReport;
use tracing::warn;

use crate::decoded::{DecodedTextEvent, DecodedTextEventStream};
use crate::error::Error;
use crate::event::AssistantBlockKind;
use crate::pipeline::AssistantStreamEvent;

/// Per-stream reasoning parsing state.
struct ReasoningState {
    /// Optional reasoning parser for streams whose model supports reasoning
    /// separation.
    reasoning_parser: Option<Box<dyn ReasoningParser>>,
    /// Whether reasoning parsing has already failed for this stream.
    reasoning_parser_failed: bool,
}

impl ReasoningState {
    /// Create one fresh reasoning-adaptation state for a new streamed response.
    fn new(reasoning_parser: Option<Box<dyn ReasoningParser>>) -> Self {
        Self {
            reasoning_parser,
            reasoning_parser_failed: false,
        }
    }

    /// Convert one decoded text delta into zero or more semantic assistant
    /// deltas.
    fn process_delta(&mut self, delta: String) -> Vec<AssistantStreamEvent> {
        let mut events = Vec::new();

        // If we have a reasoning parser, try to split the delta into reasoning
        // vs. normal assistant text.
        if let Some(parser) = self.reasoning_parser.as_mut() {
            match parser.parse_reasoning_streaming_incremental(&delta) {
                Ok(result) => {
                    push_text_delta(
                        &mut events,
                        AssistantBlockKind::Reasoning,
                        result.reasoning_text,
                    );
                    push_text_delta(&mut events, AssistantBlockKind::Text, result.normal_text);
                    return events;
                }
                Err(error) => {
                    if !self.reasoning_parser_failed {
                        warn!(
                            parser = parser.model_type(),
                            error = %error.as_report(),
                            "reasoning parser failed; falling back to plain text deltas"
                        );
                        self.reasoning_parser_failed = true;
                    }
                    self.reasoning_parser = None;
                }
            }
        }

        push_text_delta(&mut events, AssistantBlockKind::Text, delta);
        events
    }
}

/// Push one semantic text delta if it is non-empty.
fn push_text_delta(
    events: &mut Vec<AssistantStreamEvent>,
    kind: AssistantBlockKind,
    delta: String,
) {
    if delta.is_empty() {
        return;
    }
    events.push(AssistantStreamEvent::TextDelta { kind, delta });
}

/// Wrap one decoded-text stream into the internal reasoning-aware assistant
/// stream.
#[try_stream(ok = AssistantStreamEvent, error = Error)]
pub(crate) async fn reasoning_event_stream(
    decoded_stream: impl DecodedTextEventStream,
    reasoning_parser: Option<Box<dyn ReasoningParser>>,
) {
    pin_mut!(decoded_stream);

    let mut state = ReasoningState::new(reasoning_parser);

    while let Some(event) = decoded_stream.next().await.transpose()? {
        match event {
            DecodedTextEvent::Start => yield AssistantStreamEvent::Start,
            DecodedTextEvent::TextDelta { delta, .. } => {
                for next in state.process_delta(delta) {
                    yield next;
                }
            }
            DecodedTextEvent::Done {
                token_ids,
                finish_reason,
                stop_reason,
                ..
            } => {
                yield AssistantStreamEvent::Done {
                    token_ids,
                    finish_reason,
                    stop_reason,
                };
            }
        }
    }
}
