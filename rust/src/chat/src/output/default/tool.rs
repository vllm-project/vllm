//! Adapts plain assistant text deltas into tool-call-aware assistant updates.
//!
//! This stage runs after reasoning separation and before final block assembly.
//! It only inspects normal assistant text, leaves reasoning deltas untouched,
//! and translates incremental `tool-parser` output into internal tool-call
//! events while preserving plain-text fallback behavior.

use futures::{StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use thiserror_ext::AsReport;
use tracing::warn;

use super::{AssistantEvent, ContentEvent, ContentEventStream};
use crate::Result;
use crate::error::Error;
use crate::event::AssistantBlockKind;
use crate::output::generate_tool_call_id;
use crate::parser::tool::{ToolCallDelta, ToolParseResult, ToolParser};

/// Per-stream tool parsing state.
struct ToolState {
    /// Parser for the current model family.
    parser: Box<dyn ToolParser>,
    /// Whether tool parsing has already failed for this stream.
    parser_failed: bool,
    /// The parser-local index of the currently open tool call, if any.
    // NOTE: We only allow single open tool call at a time right now, since that's what all
    // supported parsers currently emit. Change this to a `BTreeMap` if we need to support multiple
    // interleaved calls in the future.
    open_call_index: Option<usize>,
}

impl ToolState {
    /// Create one fresh tool-parsing state for a new streamed response.
    fn new(parser: Box<dyn ToolParser>) -> Self {
        Self {
            parser,
            parser_failed: false,
            open_call_index: None,
        }
    }

    /// Convert one semantic assistant text delta into zero or more tool-aware
    /// internal events.
    fn process_text_delta(
        &mut self,
        kind: AssistantBlockKind,
        delta: String,
    ) -> Result<Vec<AssistantEvent>> {
        let mut events = Vec::new();

        // Only normal assistant text is eligible for tool parsing. Reasoning
        // blocks and plain-text fallback should pass through unchanged.
        if kind != AssistantBlockKind::Text || self.parser_failed {
            self.open_call_index = None;
            events.push(AssistantEvent::TextDelta { kind, delta });
            return Ok(events);
        }

        let parse_result = self.parser.push(&delta);

        match parse_result {
            Ok(result) => self.process_parse_result(kind, result, &mut events)?,
            Err(error) => {
                if !self.parser_failed {
                    warn!(
                        error = %error.as_report(),
                        "tool parser failed; falling back to plain text deltas"
                    );
                    self.parser_failed = true;
                }
                self.open_call_index = None;
                events.push(AssistantEvent::TextDelta { kind, delta });
            }
        }

        Ok(events)
    }

    /// Apply one parsed tool result to the current stream state.
    fn process_parse_result(
        &mut self,
        kind: AssistantBlockKind,
        result: ToolParseResult,
        events: &mut Vec<AssistantEvent>,
    ) -> Result<()> {
        // When we are not currently streaming a tool call, preserve plain
        // text first and then surface any new tool call items.
        if self.open_call_index.is_none() {
            push_text_delta(events, kind, result.normal_text);
            self.process_tool_items(result.calls, events)?;
        } else {
            // Once a tool call is open, prioritize tool deltas first. If the
            // parser emits normal text again, close the tool call and resume
            // plain text output.
            self.process_tool_items(result.calls, events)?;
            if !result.normal_text.is_empty() {
                self.open_call_index = None;
                push_text_delta(events, kind, result.normal_text);
            }
        }
        Ok(())
    }

    /// Apply one batch of parsed tool-call deltas emitted by the parser.
    fn process_tool_items(
        &mut self,
        items: Vec<ToolCallDelta>,
        events: &mut Vec<AssistantEvent>,
    ) -> Result<()> {
        for item in items {
            if let Some(name) = item.name {
                let is_new_tool = match self.open_call_index {
                    Some(open_call_index) => open_call_index != item.tool_index,
                    None => true,
                };
                if is_new_tool {
                    let id = generate_tool_call_id();
                    self.open_call_index = Some(item.tool_index);
                    events.push(AssistantEvent::ToolCallStart { id, name });
                }
            }

            if item.arguments.is_empty() {
                // No arguments delta to apply.
                continue;
            }
            let Some(open_call_index) = self.open_call_index else {
                return Err(Error::ToolCallStreamInvariant {
                    message: format!(
                        "received arguments for tool index {} before any tool-call start",
                        item.tool_index
                    ),
                });
            };
            if open_call_index != item.tool_index {
                return Err(Error::ToolCallStreamInvariant {
                    message: format!(
                        "received arguments for tool index {} while tool index {} is open",
                        item.tool_index, open_call_index
                    ),
                });
            }

            events.push(AssistantEvent::ToolCallArgumentsDelta {
                delta: item.arguments,
            });
        }
        Ok(())
    }

    /// Flush parser state at end-of-stream and close any remaining open calls.
    fn finish(&mut self) -> Result<Vec<AssistantEvent>> {
        let mut events = Vec::new();

        if self.parser_failed {
            return Ok(events);
        }

        match self.parser.finish() {
            Ok(result) => {
                self.process_parse_result(AssistantBlockKind::Text, result, &mut events)?
            }
            Err(error) => {
                warn!(
                    error = %error.as_report(),
                    "tool parser finish failed; closing open tool calls with buffered state"
                );
                self.parser_failed = true;
            }
        }

        Ok(events)
    }
}

/// Push one plain-text delta if it is non-empty.
fn push_text_delta(events: &mut Vec<AssistantEvent>, kind: AssistantBlockKind, delta: String) {
    if delta.is_empty() {
        return;
    }
    events.push(AssistantEvent::TextDelta { kind, delta });
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
                        // per tool call, so we can surface each one as a start
                        // plus its complete arguments payload.
                        for tool_call in calls {
                            let Some(name) = tool_call.name else {
                                return Err(Error::ToolCallStreamInvariant {
                                    message: format!(
                                        "final-only tool parse produced tool index {} without a function name",
                                        tool_call.tool_index
                                    ),
                                });
                            };
                            yield AssistantEvent::ToolCallStart {
                                id: generate_tool_call_id(),
                                name,
                            };
                            if !tool_call.arguments.is_empty() {
                                yield AssistantEvent::ToolCallArgumentsDelta {
                                    delta: tool_call.arguments,
                                };
                            }
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
                for next in state.process_text_delta(kind, delta)? {
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
                for next in state.finish()? {
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

    use super::super::{AssistantEvent, ContentEvent};
    use super::tool_event_stream;
    use crate::error::Error;
    use crate::event::{AssistantBlockKind, AssistantMessageExt as _};
    use crate::output::structured::structured_chat_event_stream;
    use crate::parser::tool::{Result, ToolParseResult, ToolParser};
    use crate::request::ChatTool;
    use crate::stream::ChatEventStream;

    struct FailingParser {
        fail_next: bool,
    }

    struct ScriptedParser {
        push_results: Vec<ToolParseResult>,
        finish_result: ToolParseResult,
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

    impl ToolParser for ScriptedParser {
        fn create(_tools: &[ChatTool]) -> crate::parser::tool::Result<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self {
                push_results: Vec::new(),
                finish_result: ToolParseResult::default(),
            }))
        }

        fn push(&mut self, _chunk: &str) -> Result<ToolParseResult> {
            Ok(self.push_results.pop().unwrap_or_default())
        }

        fn finish(&mut self) -> Result<ToolParseResult> {
            Ok(std::mem::take(&mut self.finish_result))
        }

        fn parse_complete(&mut self, _output: &str) -> Result<ToolParseResult> {
            Ok(self.finish_result.clone())
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

    #[tokio::test]
    async fn tool_stream_rejects_interleaved_tool_indices() {
        let events = stream::iter(vec![
            Ok(ContentEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: "ignored".to_string(),
            }),
            Ok(ContentEvent::Done {
                prompt_token_count: 1,
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let parser = ScriptedParser {
            push_results: vec![ToolParseResult {
                normal_text: String::new(),
                calls: vec![
                    crate::parser::tool::ToolCallDelta {
                        tool_index: 0,
                        name: Some("first".to_string()),
                        arguments: String::new(),
                    },
                    crate::parser::tool::ToolCallDelta {
                        tool_index: 1,
                        name: None,
                        arguments: "{}".to_string(),
                    },
                ],
            }],
            finish_result: ToolParseResult::default(),
        };

        let err = tool_event_stream(events, true, Some(Box::new(parser)))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .find_map(|result| result.err())
            .expect("expected invariant error");

        assert!(matches!(err, Error::ToolCallStreamInvariant { .. }));
    }

    #[tokio::test]
    async fn tool_stream_resets_open_tool_when_normal_text_interrupts_it() {
        let events = stream::iter(vec![
            Ok(ContentEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: "start".to_string(),
            }),
            Ok(ContentEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: "text".to_string(),
            }),
            Ok(ContentEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: "args".to_string(),
            }),
        ]);

        let parser = ScriptedParser {
            push_results: vec![
                ToolParseResult {
                    normal_text: String::new(),
                    calls: vec![crate::parser::tool::ToolCallDelta {
                        tool_index: 0,
                        name: None,
                        arguments: "}".to_string(),
                    }],
                },
                ToolParseResult {
                    normal_text: "plain text".to_string(),
                    calls: Vec::new(),
                },
                ToolParseResult {
                    normal_text: String::new(),
                    calls: vec![crate::parser::tool::ToolCallDelta {
                        tool_index: 0,
                        name: Some("first".to_string()),
                        arguments: "{".to_string(),
                    }],
                },
            ],
            finish_result: ToolParseResult::default(),
        };

        let err = tool_event_stream(events, true, Some(Box::new(parser)))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .find_map(|result| result.err())
            .expect("expected invariant error");

        assert!(matches!(
            err,
            Error::ToolCallStreamInvariant { message }
                if message == "received arguments for tool index 0 before any tool-call start"
        ));
    }

    #[tokio::test]
    async fn final_only_tool_stream_emits_start_and_args_for_multiple_calls() {
        let events = stream::iter(vec![
            Ok(ContentEvent::Start {
                prompt_token_ids: vec![1].into(),
                prompt_logprobs: None,
            }),
            Ok(ContentEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: "ignored".to_string(),
            }),
            Ok(ContentEvent::Done {
                prompt_token_count: 1,
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let parser = ScriptedParser {
            push_results: Vec::new(),
            finish_result: ToolParseResult {
                normal_text: String::new(),
                calls: vec![
                    crate::parser::tool::ToolCallDelta {
                        tool_index: 0,
                        name: Some("first".to_string()),
                        arguments: r#"{"a":1}"#.to_string(),
                    },
                    crate::parser::tool::ToolCallDelta {
                        tool_index: 1,
                        name: Some("second".to_string()),
                        arguments: r#"{"b":2}"#.to_string(),
                    },
                ],
            },
        };

        let events = tool_event_stream(events, false, Some(Box::new(parser)))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .unwrap();

        assert!(matches!(events[1], AssistantEvent::ToolCallStart { .. }));
        assert!(matches!(
            events[2],
            AssistantEvent::ToolCallArgumentsDelta { .. }
        ));
        assert!(matches!(events[3], AssistantEvent::ToolCallStart { .. }));
        assert!(matches!(
            events[4],
            AssistantEvent::ToolCallArgumentsDelta { .. }
        ));
        let collected = ChatEventStream::new(
            "req_final_only".to_string(),
            Box::pin(structured_chat_event_stream(stream::iter(
                events.into_iter().map(Ok),
            ))),
        )
        .collect_message()
        .await
        .unwrap();
        let tool_calls = collected.message.tool_calls().collect::<Vec<_>>();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].name, "first");
        assert_eq!(tool_calls[1].name, "second");
    }
}
