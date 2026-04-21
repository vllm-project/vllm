//! Adapts plain assistant text deltas into tool-call-aware assistant updates.
//!
//! This stage runs after reasoning separation and before final block assembly.
//! It only inspects normal assistant text, leaves reasoning deltas untouched,
//! and translates incremental `tool-parser` output into internal tool-call
//! events while preserving plain-text fallback behavior.

use std::collections::{BTreeMap, btree_map};

use futures::{StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use thiserror_ext::AsReport;
use tracing::warn;
use uuid::Uuid;

use super::{AssistantEvent, ContentEvent, ContentEventStream};
use crate::error::Error;
use crate::event::{AssistantBlockKind, AssistantToolCall};
use crate::parser::tool::{ToolCallDelta, ToolParseResult, ToolParser};

/// One currently open tool call being assembled from streaming parser output.
struct OpenToolCallState {
    /// Stable tool-call ID exposed northbound.
    id: String,
    /// Function name.
    name: String,
    /// Incremental JSON arguments accumulated so far.
    arguments: String,
}

/// Per-stream tool parsing state.
struct ToolState {
    /// Parser for the current model family.
    parser: Box<dyn ToolParser>,
    /// Whether tool parsing has already failed for this stream.
    parser_failed: bool,
    /// Open tool calls keyed by the parser's tool index.
    open_calls: BTreeMap<usize, OpenToolCallState>,
}

impl ToolState {
    /// Create one fresh tool-parsing state for a new streamed response.
    fn new(parser: Box<dyn ToolParser>) -> Self {
        Self {
            parser,
            parser_failed: false,
            open_calls: BTreeMap::new(),
        }
    }

    /// Convert one semantic assistant text delta into zero or more tool-aware
    /// internal events.
    fn process_text_delta(
        &mut self,
        kind: AssistantBlockKind,
        delta: String,
    ) -> Vec<AssistantEvent> {
        let mut events = Vec::new();

        // Only normal assistant text is eligible for tool parsing. Reasoning
        // blocks and plain-text fallback should pass through unchanged.
        if kind != AssistantBlockKind::Text || self.parser_failed {
            self.close_all_open_calls(&mut events);
            events.push(AssistantEvent::TextDelta { kind, delta });
            return events;
        }

        let parse_result = self.parser.push(&delta);

        match parse_result {
            Ok(result) => self.process_parse_result(kind, result, &mut events),
            Err(error) => {
                if !self.parser_failed {
                    warn!(
                        error = %error.as_report(),
                        "tool parser failed; falling back to plain text deltas"
                    );
                    self.parser_failed = true;
                }
                self.close_all_open_calls(&mut events);
                events.push(AssistantEvent::TextDelta { kind, delta });
            }
        }

        events
    }

    /// Apply one parsed tool result to the current stream state.
    fn process_parse_result(
        &mut self,
        kind: AssistantBlockKind,
        result: ToolParseResult,
        events: &mut Vec<AssistantEvent>,
    ) {
        // When we are not currently streaming a tool call, preserve plain
        // text first and then surface any new tool call items.
        if self.open_calls.is_empty() {
            push_text_delta(events, kind, result.normal_text);
            self.process_tool_items(result.calls, events);
        } else {
            // Once a tool call is open, prioritize tool deltas first. If the
            // parser emits normal text again, close the tool call and resume
            // plain text output.
            self.process_tool_items(result.calls, events);
            if !result.normal_text.is_empty() {
                self.close_all_open_calls(events);
                push_text_delta(events, kind, result.normal_text);
            }
        }
    }

    /// Apply one batch of parsed tool-call deltas emitted by the parser.
    fn process_tool_items(&mut self, items: Vec<ToolCallDelta>, events: &mut Vec<AssistantEvent>) {
        for item in items {
            if let Some(name) = item.name {
                // The parser is now advancing a specific tool index, so any
                // previously open sibling calls must be finalized first.
                self.close_calls_not_matching(item.tool_index, events);

                if let btree_map::Entry::Vacant(e) = self.open_calls.entry(item.tool_index) {
                    let id = generate_tool_call_id();
                    e.insert(OpenToolCallState {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: String::new(),
                    });
                    events.push(AssistantEvent::ToolCallStart { id, name });
                }
            }

            if item.arguments.is_empty() {
                // No arguments delta to apply.
                continue;
            }
            let Some(open_call) = self.open_calls.get_mut(&item.tool_index) else {
                continue;
            };
            open_call.arguments.push_str(&item.arguments);

            events.push(AssistantEvent::ToolCallArgumentsDelta {
                id: open_call.id.clone(),
                delta: item.arguments,
            });
        }
    }

    /// Close every open tool call except the one still associated with the
    /// parser's current tool index.
    fn close_calls_not_matching(
        &mut self,
        keep_tool_index: usize,
        events: &mut Vec<AssistantEvent>,
    ) {
        let old = std::mem::take(&mut self.open_calls);
        for (idx, open_call) in old {
            if idx == keep_tool_index {
                self.open_calls.insert(idx, open_call);
            } else {
                push_tool_call_end(events, open_call);
            }
        }
    }

    /// Close every currently open tool call.
    fn close_all_open_calls(&mut self, events: &mut Vec<AssistantEvent>) {
        for (_, open_call) in std::mem::take(&mut self.open_calls) {
            push_tool_call_end(events, open_call);
        }
    }

    /// Flush parser state at end-of-stream and close any remaining open calls.
    fn finish(&mut self) -> Vec<AssistantEvent> {
        let mut events = Vec::new();

        if self.parser_failed {
            self.close_all_open_calls(&mut events);
            return events;
        }

        match self.parser.finish() {
            Ok(result) => self.process_parse_result(AssistantBlockKind::Text, result, &mut events),
            Err(error) => {
                warn!(
                    error = %error.as_report(),
                    "tool parser finish failed; closing open tool calls with buffered state"
                );
                self.parser_failed = true;
            }
        }

        self.close_all_open_calls(&mut events);
        events
    }
}

/// Emit one `ToolCallEnd` event from a completed open call.
fn push_tool_call_end(events: &mut Vec<AssistantEvent>, open_call: OpenToolCallState) {
    events.push(AssistantEvent::ToolCallEnd {
        call: AssistantToolCall {
            id: open_call.id,
            name: open_call.name,
            arguments: open_call.arguments,
        },
    });
}

/// Push one plain-text delta if it is non-empty.
fn push_text_delta(events: &mut Vec<AssistantEvent>, kind: AssistantBlockKind, delta: String) {
    if delta.is_empty() {
        return;
    }
    events.push(AssistantEvent::TextDelta { kind, delta });
}

/// Generate the northbound tool-call ID using the OpenAI-style `call_<id>` format.
///
/// TODO: support other ID scheme like Kimi-K2's `functions.{name}:{global_index}`.
fn generate_tool_call_id() -> String {
    format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
}

/// Tool parsing when `intermediate=false` (`FinalOnly` mode).
///
/// We keep this separate because some adaptor-backed parsers may not correctly handle the full text
/// passed to incremental `push` interface, but override `parse_complete()` with a dedicated
/// one-shot implementation to ensure correctness.
#[try_stream(ok = AssistantEvent, error = Error)]
async fn final_only_tool_event_stream(
    stream: impl ContentEventStream,
    mut parser: Box<dyn ToolParser>,
) {
    pin_mut!(stream);

    let mut final_text = String::new();

    while let Some(event) = stream.next().await.transpose()? {
        match event {
            ContentEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => {
                yield AssistantEvent::Start {
                    prompt_token_ids,
                    prompt_logprobs,
                }
            }
            ContentEvent::TextDelta { kind, delta } => {
                if kind == AssistantBlockKind::Text {
                    final_text.push_str(&delta);
                } else {
                    yield AssistantEvent::TextDelta { kind, delta };
                }
            }
            ContentEvent::LogprobsDelta {
                logprobs,
                token_ids,
            } => {
                yield AssistantEvent::LogprobsDelta {
                    logprobs,
                    token_ids,
                };
            }
            ContentEvent::Done {
                prompt_token_count,
                output_token_count,
                finish_reason,
                kv_transfer_params,
            } => {
                match parser.parse_complete(&final_text) {
                    Ok(ToolParseResult { normal_text, calls }) => {
                        if !normal_text.is_empty() {
                            yield AssistantEvent::TextDelta {
                                kind: AssistantBlockKind::Text,
                                delta: normal_text,
                            };
                        }
                        // `parse_complete` currently returns one complete delta
                        // per tool call, so we can finalize each call directly
                        // without reusing the streaming state machine here.
                        for tool_call in calls {
                            let Some(name) = tool_call.name else {
                                continue;
                            };
                            // It's okay to only emit `ToolCallEnd` without a preceding
                            // `ToolCallStart` or `ToolCallArgumentsDelta`.
                            yield AssistantEvent::ToolCallEnd {
                                call: AssistantToolCall {
                                    id: generate_tool_call_id(),
                                    name,
                                    arguments: tool_call.arguments,
                                },
                            };
                        }
                    }
                    Err(error) => {
                        warn!(
                            error = %error.as_report(),
                            "tool parser full-output parse failed; falling back to plain text"
                        );
                        yield AssistantEvent::TextDelta {
                            kind: AssistantBlockKind::Text,
                            delta: final_text,
                        };
                    }
                }

                yield AssistantEvent::Done {
                    prompt_token_count,
                    output_token_count,
                    finish_reason,
                    kv_transfer_params,
                };
                return Ok(());
            }
        }
    }
}

/// Wrap one semantic assistant stream into the internal tool-aware assistant
/// stream.
#[try_stream(ok = AssistantEvent, error = Error)]
pub(crate) async fn tool_event_stream(
    stream: impl ContentEventStream,
    intermediate: bool,
    parser: Option<Box<dyn ToolParser>>,
) {
    // Without a parser, pass through the input stream unchanged.
    let Some(parser) = parser else {
        pin_mut!(stream);
        while let Some(event) = stream.next().await.transpose()? {
            yield event.into();
        }
        return Ok(());
    };

    // `FinalOnly` needs one-shot parsing over the final text.
    if !intermediate {
        let final_stream = final_only_tool_event_stream(stream, parser);
        pin_mut!(final_stream);
        while let Some(event) = final_stream.next().await.transpose()? {
            yield event;
        }
        return Ok(());
    }

    pin_mut!(stream);
    let mut state = ToolState::new(parser);

    while let Some(event) = stream.next().await.transpose()? {
        match event {
            ContentEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => {
                yield AssistantEvent::Start {
                    prompt_token_ids,
                    prompt_logprobs,
                }
            }
            ContentEvent::TextDelta { kind, delta } => {
                for next in state.process_text_delta(kind, delta) {
                    yield next;
                }
            }
            ContentEvent::LogprobsDelta {
                logprobs,
                token_ids,
            } => {
                yield AssistantEvent::LogprobsDelta {
                    logprobs,
                    token_ids,
                };
            }
            ContentEvent::Done {
                prompt_token_count,
                output_token_count,
                finish_reason,
                kv_transfer_params,
            } => {
                for next in state.finish() {
                    yield next;
                }

                yield AssistantEvent::Done {
                    prompt_token_count,
                    output_token_count,
                    finish_reason,
                    kv_transfer_params,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use futures::{StreamExt as _, stream};
    use vllm_llm::FinishReason;
    use vllm_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob};

    use super::super::structured::structured_chat_event_stream;
    use super::super::{AssistantEvent, ContentEvent};
    use super::tool_event_stream;
    use crate::event::{AssistantBlockKind, AssistantMessageExt as _};
    use crate::parser::tool::{Result, ToolParseResult, ToolParser};
    use crate::request::ChatTool;
    use crate::stream::ChatEventStream;

    struct FailingParser {
        fail_next: bool,
    }

    impl ToolParser for FailingParser {
        fn create(_tools: &[ChatTool]) -> crate::parser::tool::Result<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self { fail_next: false }))
        }

        fn push(&mut self, _chunk: &str) -> Result<ToolParseResult> {
            if self.fail_next {
                self.fail_next = false;
                return Err(
                    tool_parser::errors::ParserError::ParsingFailed("boom".to_string()).into(),
                );
            }

            Ok(ToolParseResult::default())
        }
    }

    #[tokio::test]
    async fn tool_parser_failure_falls_back_to_plain_text() {
        let events = stream::iter(vec![
            Ok(ContentEvent::Start {
                prompt_token_ids: vec![1, 2, 3].into(),
                prompt_logprobs: None,
            }),
            Ok(ContentEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: "abc".to_string(),
            }),
            Ok(ContentEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: "def".to_string(),
            }),
            Ok(ContentEvent::Done {
                prompt_token_count: 3,
                output_token_count: 0,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let collected = tool_event_stream(
            events,
            true,
            Some(Box::new(FailingParser { fail_next: true })),
        )
        .collect::<Vec<_>>()
        .await;

        let events = collected
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .expect("tool stream should not fail");

        assert_eq!(
            events,
            vec![
                AssistantEvent::Start {
                    prompt_token_ids: vec![1, 2, 3].into(),
                    prompt_logprobs: None,
                },
                AssistantEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "abc".to_string(),
                },
                AssistantEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "def".to_string(),
                },
                AssistantEvent::Done {
                    prompt_token_count: 3,
                    output_token_count: 0,
                    finish_reason: FinishReason::stop_eos(),
                    kv_transfer_params: None,
                },
            ]
        );

        let message = ChatEventStream::new(
            "req_fallback".to_string(),
            Box::pin(structured_chat_event_stream(stream::iter(
                events.into_iter().map(Ok),
            ))),
        )
        .collect_message()
        .await
        .expect("collect_message should succeed");
        assert_eq!(message.message.text(), "abcdef");
        assert!(message.message.tool_calls().next().is_none());
    }

    #[tokio::test]
    async fn tool_stream_preserves_logprobs_delta_in_final_only_mode() {
        let events = stream::iter(vec![
            Ok(ContentEvent::Start {
                prompt_token_ids: vec![1].into(),
                prompt_logprobs: None,
            }),
            Ok(ContentEvent::LogprobsDelta {
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
            Ok(ContentEvent::Done {
                prompt_token_count: 1,
                output_token_count: 0,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);
        let events = tool_event_stream(
            events,
            false,
            Some(Box::new(FailingParser { fail_next: false })),
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<crate::Result<Vec<_>>>()
        .unwrap();

        assert_eq!(
            events,
            vec![
                AssistantEvent::Start {
                    prompt_token_ids: vec![1].into(),
                    prompt_logprobs: None,
                },
                AssistantEvent::LogprobsDelta {
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
                },
                AssistantEvent::Done {
                    prompt_token_count: 1,
                    output_token_count: 0,
                    finish_reason: FinishReason::stop_eos(),
                    kv_transfer_params: None,
                },
            ]
        );
    }
}
