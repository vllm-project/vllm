//! Adapts plain assistant text deltas into tool-call-aware assistant updates.
//!
//! This stage runs after reasoning separation and before final block assembly.
//! It only inspects normal assistant text, leaves reasoning deltas untouched,
//! and translates incremental tool parsing output into internal tool-call
//! events while preserving plain-text fallback behavior.

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{StreamExt as _, pin_mut};
use thiserror_ext::AsReport;
use tracing::warn;

use super::{AssistantEvent, ContentEvent, ContentEventStream};
use crate::Result;
use crate::error::Error;
use crate::event::AssistantBlockKind;
use crate::output::generate_tool_call_id;
use crate::parser::tool::{ToolCallDelta, ToolParser, ToolParserOutput};

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

        let mut output = ToolParserOutput::default();
        let parse_result = self.parser.parse_into(&delta, &mut output);

        match parse_result {
            Ok(()) => self.process_parser_output(kind, output, &mut events)?,
            Err(error) => {
                warn!(
                    error = %error.as_report(),
                    "tool parser failed; falling back to plain text deltas"
                );
                // Permanently mark this parser as failed.
                // TODO: we may consider recovering from parsing errors in the future.
                self.parser_failed = true;

                // On parsing failure, we still apply the partial parser output if any, but we close
                // any open tool calls and emit the remaining buffered text as a plain-text delta to
                // preserve as much of the output as possible.
                self.process_parser_output(kind, output, &mut events)?;
                self.open_call_index = None;
                push_text_delta(&mut events, kind, self.parser.reset());
            }
        }

        Ok(events)
    }

    /// Apply one parsed tool output to the current stream state.
    fn process_parser_output(
        &mut self,
        kind: AssistantBlockKind,
        output: ToolParserOutput,
        events: &mut Vec<AssistantEvent>,
    ) -> Result<()> {
        // When we are not currently streaming a tool call, preserve plain
        // text first and then surface any new tool call items.
        if self.open_call_index.is_none() {
            push_text_delta(events, kind, output.normal_text);
            self.process_tool_items(output.calls, events)?;
        } else {
            // Once a tool call is open, prioritize tool deltas first. If the
            // parser emits normal text again, close the tool call and resume
            // plain text output.
            self.process_tool_items(output.calls, events)?;
            if !output.normal_text.is_empty() {
                self.open_call_index = None;
                push_text_delta(events, kind, output.normal_text);
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
                    let id = self
                        .parser
                        .tool_call_id(item.tool_index)
                        .map(str::to_string)
                        .unwrap_or_else(generate_tool_call_id);
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
            Ok(output) => {
                self.process_parser_output(AssistantBlockKind::Text, output, &mut events)?
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

/// Wrap one semantic assistant stream into the internal tool-aware assistant
/// stream.
#[try_stream]
pub(crate) async fn tool_event_stream(
    stream: impl ContentEventStream,
    parser: Option<Box<dyn ToolParser>>,
    mut y: TryYielder<AssistantEvent, Error>,
) -> Result<()> {
    // Without a parser, pass through the input stream unchanged.
    let Some(parser) = parser else {
        pin_mut!(stream);
        while let Some(event) = stream.next().await.transpose()? {
            y.yield_ok(event.into()).await;
        }
        return Ok(());
    };

    pin_mut!(stream);
    let mut state = ToolState::new(parser);

    while let Some(event) = stream.next().await.transpose()? {
        match event {
            ContentEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => {
                y.yield_ok(AssistantEvent::Start {
                    prompt_token_ids,
                    prompt_logprobs,
                })
                .await;
            }
            ContentEvent::TextDelta { kind, delta } => {
                for next in state.process_text_delta(kind, delta)? {
                    y.yield_ok(next).await;
                }
            }
            ContentEvent::LogprobsDelta {
                logprobs,
                token_ids,
            } => {
                y.yield_ok(AssistantEvent::LogprobsDelta {
                    logprobs,
                    token_ids,
                })
                .await;
            }
            ContentEvent::Done {
                usage,
                finish_reason,
                kv_transfer_params,
            } => {
                for next in state.finish()? {
                    y.yield_ok(next).await;
                }

                y.yield_ok(AssistantEvent::Done {
                    usage,
                    finish_reason,
                    kv_transfer_params,
                })
                .await;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    use futures::{StreamExt as _, stream};
    use vllm_llm::FinishReason;
    use vllm_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob};
    use vllm_tool_parser::Result;

    use super::super::{AssistantEvent, ContentEvent};
    use super::tool_event_stream;
    use crate::error::Error;
    use crate::event::{AssistantBlockKind, AssistantMessageExt as _};
    use crate::output::structured::structured_chat_event_stream;
    use crate::parser::tool::{
        DeepSeekV4ToolParser, ToolParser, ToolParserError, ToolParserOutput,
    };
    use crate::request::ChatTool;
    use crate::stream::{ChatEventStream, CollectedAssistantMessage};

    struct FailingParser {
        fail_next: bool,
        buffered: String,
    }

    struct ScriptedParser {
        push_outputs: Vec<ToolParserOutput>,
        finish_output: ToolParserOutput,
    }

    struct PartialThenFailParser {
        buffered: String,
    }

    struct IdScriptedParser {
        output: ToolParserOutput,
        tool_call_id: Option<String>,
    }

    impl ToolParser for FailingParser {
        fn create(_tools: &[ChatTool]) -> vllm_tool_parser::Result<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self {
                fail_next: false,
                buffered: String::new(),
            }))
        }

        fn parse_into(&mut self, chunk: &str, _output: &mut ToolParserOutput) -> Result<()> {
            self.buffered.push_str(chunk);
            if self.fail_next {
                self.fail_next = false;
                return Err(ToolParserError::ParsingFailed {
                    message: "boom".to_string(),
                });
            }

            self.buffered.clear();
            Ok(())
        }

        fn finish(&mut self) -> Result<ToolParserOutput> {
            Ok(ToolParserOutput::default())
        }

        fn reset(&mut self) -> String {
            std::mem::take(&mut self.buffered)
        }
    }

    impl ToolParser for ScriptedParser {
        fn create(_tools: &[ChatTool]) -> vllm_tool_parser::Result<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self {
                push_outputs: Vec::new(),
                finish_output: ToolParserOutput::default(),
            }))
        }

        fn parse_into(&mut self, _chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
            let mut next = self.push_outputs.pop().unwrap_or_default();
            output.normal_text.push_str(&next.normal_text);
            output.calls.append(&mut next.calls);
            Ok(())
        }

        fn finish(&mut self) -> Result<ToolParserOutput> {
            Ok(std::mem::take(&mut self.finish_output))
        }

        fn reset(&mut self) -> String {
            String::new()
        }
    }

    impl ToolParser for IdScriptedParser {
        fn create(_tools: &[ChatTool]) -> vllm_tool_parser::Result<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self {
                output: ToolParserOutput::default(),
                tool_call_id: None,
            }))
        }

        fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
            (tool_index == 0).then_some(self.tool_call_id.as_deref()).flatten()
        }

        fn parse_into(&mut self, _chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
            output.append(std::mem::take(&mut self.output));
            Ok(())
        }

        fn finish(&mut self) -> Result<ToolParserOutput> {
            Ok(ToolParserOutput::default())
        }

        fn reset(&mut self) -> String {
            String::new()
        }
    }

    impl ToolParser for PartialThenFailParser {
        fn create(_tools: &[ChatTool]) -> vllm_tool_parser::Result<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self {
                buffered: String::new(),
            }))
        }

        fn parse_into(&mut self, _chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
            output.calls.extend([
                crate::parser::tool::ToolCallDelta {
                    tool_index: 0,
                    name: Some("get_weather".to_string()),
                    arguments: String::new(),
                },
                crate::parser::tool::ToolCallDelta {
                    tool_index: 0,
                    name: None,
                    arguments: r#"{"location":"SF"}"#.to_string(),
                },
            ]);
            self.buffered.push_str(" trailing text");
            Err(ToolParserError::ParsingFailed {
                message: "boom".to_string(),
            })
        }

        fn finish(&mut self) -> Result<ToolParserOutput> {
            Ok(ToolParserOutput::default())
        }

        fn reset(&mut self) -> String {
            std::mem::take(&mut self.buffered)
        }
    }

    fn deepseek_v4_test_tools() -> Vec<ChatTool> {
        vec![
            ChatTool {
                name: "get_weather".to_string(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" }
                    }
                }),
                strict: None,
            },
            ChatTool {
                name: "add".to_string(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "x": { "type": "integer" },
                        "y": { "type": "integer" }
                    }
                }),
                strict: None,
            },
        ]
    }

    async fn collect_deepseek_v4_message(chunks: Vec<String>) -> CollectedAssistantMessage {
        let events = chunks
            .into_iter()
            .map(|delta| {
                Ok(ContentEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta,
                })
            })
            .chain(std::iter::once(Ok(ContentEvent::Done {
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 1,
                    output_token_count: 1,
                    cached_token_count: 0,
                },
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            })));
        let parser = DeepSeekV4ToolParser::create(&deepseek_v4_test_tools()).unwrap();
        let assistant_events = tool_event_stream(stream::iter(events), Some(parser));
        let chat_events = structured_chat_event_stream(assistant_events);

        ChatEventStream::new("req_deepseek_v4".to_string(), Box::pin(chat_events))
            .collect_message()
            .await
            .unwrap()
    }

    fn message_tool_projection(
        message: &CollectedAssistantMessage,
    ) -> (String, Vec<(String, serde_json::Value)>) {
        (
            message.message.text(),
            message
                .message
                .tool_calls()
                .map(|call| {
                    (
                        call.name.clone(),
                        serde_json::from_str(&call.arguments).unwrap(),
                    )
                })
                .collect(),
        )
    }

    #[tokio::test]
    async fn tool_parser_error_preserves_partial_output_and_flushes_buffer() {
        let events = stream::iter(vec![
            Ok(ContentEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: "ignored".to_string(),
            }),
            Ok(ContentEvent::Done {
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 1,
                    output_token_count: 1,
                    cached_token_count: 0,
                },
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let events = tool_event_stream(
            events,
            Some(Box::new(PartialThenFailParser {
                buffered: String::new(),
            })),
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<crate::Result<Vec<_>>>()
        .unwrap();

        assert!(matches!(
            &events[0],
            AssistantEvent::ToolCallStart { name, .. } if name == "get_weather"
        ));
        assert!(matches!(
            &events[1],
            AssistantEvent::ToolCallArgumentsDelta { delta } if delta == r#"{"location":"SF"}"#
        ));
        assert_eq!(
            events[2],
            AssistantEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: " trailing text".to_string(),
            }
        );
        assert!(matches!(events[3], AssistantEvent::Done { .. }));
    }

    #[tokio::test]
    async fn tool_stream_preserves_parser_provided_tool_call_id() {
        let events = stream::iter(vec![Ok(ContentEvent::TextDelta {
            kind: AssistantBlockKind::Text,
            delta: "ignored".to_string(),
        })]);
        let parser = IdScriptedParser {
            output: ToolParserOutput {
                normal_text: String::new(),
                calls: vec![crate::parser::tool::ToolCallDelta {
                    tool_index: 0,
                    name: Some("get_weather".to_string()),
                    arguments: "{}".to_string(),
                }],
            },
            tool_call_id: Some("functions.get_weather:0".to_string()),
        };

        let events = tool_event_stream(events, Some(Box::new(parser)))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .unwrap();

        assert!(matches!(
            &events[0],
            AssistantEvent::ToolCallStart { id, name }
                if id == "functions.get_weather:0" && name == "get_weather"
        ));
    }

    #[tokio::test]
    async fn tool_stream_generates_tool_call_id_when_parser_omits_one() {
        let events = stream::iter(vec![Ok(ContentEvent::TextDelta {
            kind: AssistantBlockKind::Text,
            delta: "ignored".to_string(),
        })]);
        let parser = IdScriptedParser {
            output: ToolParserOutput {
                normal_text: String::new(),
                calls: vec![crate::parser::tool::ToolCallDelta {
                    tool_index: 0,
                    name: Some("get_weather".to_string()),
                    arguments: "{}".to_string(),
                }],
            },
            tool_call_id: None,
        };

        let events = tool_event_stream(events, Some(Box::new(parser)))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .unwrap();

        assert!(matches!(
            &events[0],
            AssistantEvent::ToolCallStart { id, name }
                if id.starts_with("call_") && name == "get_weather"
        ));
    }

    #[tokio::test]
    async fn real_buffered_parser_error_matches_streaming_and_non_streaming() {
        let prefix = "I will check both.\n";
        let first_tool_call = concat!(
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"location\" string=\"true\">Tokyo</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>",
        );
        let malformed_second_tool_call = concat!(
            "\n<｜DSML｜invoke name=\"add\">\n",
            "not a parameter\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>",
        );
        let streaming_chunks = vec![
            prefix.to_string(),
            first_tool_call.to_string(),
            malformed_second_tool_call.to_string(),
        ];
        let full_output = streaming_chunks.concat();

        let streaming = collect_deepseek_v4_message(streaming_chunks).await;
        let non_streaming = collect_deepseek_v4_message(vec![full_output]).await;

        let expected = (
            format!("{prefix}{malformed_second_tool_call}"),
            vec![(
                "get_weather".to_string(),
                serde_json::json!({ "location": "Tokyo" }),
            )],
        );
        assert_eq!(message_tool_projection(&streaming), expected);
        assert_eq!(message_tool_projection(&non_streaming), expected);
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
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 3,
                    output_token_count: 0,
                    cached_token_count: 0,
                },
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let collected = tool_event_stream(
            events,
            Some(Box::new(FailingParser {
                fail_next: true,
                buffered: String::new(),
            })),
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
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 3,
                        output_token_count: 0,
                        cached_token_count: 0,
                    },
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
    async fn tool_stream_preserves_logprobs_delta() {
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
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 1,
                    output_token_count: 0,
                    cached_token_count: 0,
                },
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);
        let events = tool_event_stream(
            events,
            Some(Box::new(FailingParser {
                fail_next: false,
                buffered: String::new(),
            })),
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
                    usage: vllm_llm::TokenUsage {
                        prompt_token_count: 1,
                        output_token_count: 0,
                        cached_token_count: 0,
                    },
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
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 1,
                    output_token_count: 1,
                    cached_token_count: 0,
                },
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let parser = ScriptedParser {
            push_outputs: vec![ToolParserOutput {
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
            finish_output: ToolParserOutput::default(),
        };

        let err = tool_event_stream(events, Some(Box::new(parser)))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .find_map(|output| output.err())
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
            push_outputs: vec![
                ToolParserOutput {
                    normal_text: String::new(),
                    calls: vec![crate::parser::tool::ToolCallDelta {
                        tool_index: 0,
                        name: None,
                        arguments: "}".to_string(),
                    }],
                },
                ToolParserOutput {
                    normal_text: "plain text".to_string(),
                    calls: Vec::new(),
                },
                ToolParserOutput {
                    normal_text: String::new(),
                    calls: vec![crate::parser::tool::ToolCallDelta {
                        tool_index: 0,
                        name: Some("first".to_string()),
                        arguments: "{".to_string(),
                    }],
                },
            ],
            finish_output: ToolParserOutput::default(),
        };

        let err = tool_event_stream(events, Some(Box::new(parser)))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .find_map(|output| output.err())
            .expect("expected invariant error");

        assert!(matches!(
            err,
            Error::ToolCallStreamInvariant { message }
                if message == "received arguments for tool index 0 before any tool-call start"
        ));
    }

    #[tokio::test]
    async fn tool_stream_emits_start_and_args_for_terminal_text() {
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
                usage: vllm_llm::TokenUsage {
                    prompt_token_count: 1,
                    output_token_count: 1,
                    cached_token_count: 0,
                },
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let parser = ScriptedParser {
            push_outputs: vec![ToolParserOutput {
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
            }],
            finish_output: ToolParserOutput::default(),
        };

        let events = tool_event_stream(events, Some(Box::new(parser)))
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
