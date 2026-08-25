// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Adapts decoded text updates into parsed assistant deltas.
//!
//! This stage sits between low-level token decoding and final block assembly.
//! It drives one unified parser that may emit normal text, reasoning text, or
//! tool-call deltas, then normalizes those parser events into internal
//! assistant events.

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{StreamExt as _, pin_mut};
use thiserror_ext::AsReport;
use tracing::warn;
use vllm_parser::unified::{UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};
use vllm_text::output::DecodedTextEvent;

use crate::Result;
use crate::error::Error;
use crate::event::AssistantBlockKind;
use crate::output::{AssistantEvent, DecodedTextEventStream, generate_tool_call_id};

/// Per-stream unified parsing state.
struct UnifiedParserState {
    /// Parser for the current request stream.
    parser: Box<dyn UnifiedParser>,
    /// Whether unified parsing has already failed for this stream.
    parser_failed: bool,
    /// The parser-local index of the currently open tool call, if any.
    ///
    /// Supported parsers currently emit at most one active tool call at a time.
    /// Change this to an indexed map if a model needs interleaved calls later.
    open_call_index: Option<usize>,
}

impl UnifiedParserState {
    /// Create one fresh unified parsing state for a new streamed response.
    fn new(parser: Box<dyn UnifiedParser>) -> Self {
        Self {
            parser,
            parser_failed: false,
            open_call_index: None,
        }
    }

    /// Initialize parser state once prompt token IDs are available.
    fn initialize(&mut self, prompt_token_ids: &[u32]) {
        if self.parser_failed {
            return;
        }

        match self.parser.initialize(prompt_token_ids) {
            Ok(()) => {}
            Err(error) => {
                warn!(
                    error = %error.as_report(),
                    "failed to initialize unified parser; falling back to plain text deltas"
                );
                self.parser_failed = true;
                self.open_call_index = None;
            }
        }
    }

    /// Convert one decoded text delta into zero or more parsed assistant events.
    fn process_delta(&mut self, delta: String) -> Result<Vec<AssistantEvent>> {
        if self.parser_failed {
            self.open_call_index = None;
            return Ok(text_event(AssistantBlockKind::Text, delta).into_iter().collect());
        }

        let mut output = UnifiedParserOutput::default();
        match self.parser.parse_into(&delta, &mut output) {
            Ok(()) => {
                let mut events = Vec::new();
                self.process_parser_output(output, &mut events)?;
                Ok(events)
            }
            Err(error) => {
                warn!(
                    error = %error.as_report(),
                    "unified parser failed; falling back to plain text deltas"
                );
                self.parser_failed = true;

                let mut events = Vec::new();
                self.process_parser_output(output, &mut events)?;
                self.open_call_index = None;

                let recovered = self.parser.reset();
                if recovered.is_empty() && events.is_empty() {
                    push_text_delta(&mut events, AssistantBlockKind::Text, delta);
                } else {
                    push_text_delta(&mut events, AssistantBlockKind::Text, recovered);
                }
                Ok(events)
            }
        }
    }

    /// Flush parser state at end-of-stream and close any remaining open calls.
    fn finish(&mut self) -> Result<Vec<AssistantEvent>> {
        let mut events = Vec::new();

        if self.parser_failed {
            return Ok(events);
        }

        match self.parser.finish() {
            Ok(output) => self.process_parser_output(output, &mut events)?,
            Err(error) => {
                warn!(
                    error = %error.as_report(),
                    "unified parser finish failed; closing open parser state"
                );
                self.parser_failed = true;
                self.open_call_index = None;
                let recovered = self.parser.reset();
                push_text_delta(&mut events, AssistantBlockKind::Text, recovered);
            }
        }

        Ok(events)
    }

    /// Apply one parsed unified output to the current stream state.
    fn process_parser_output(
        &mut self,
        output: UnifiedParserOutput,
        events: &mut Vec<AssistantEvent>,
    ) -> Result<()> {
        for event in output.events {
            match event {
                UnifiedParserEvent::Text(delta) => {
                    self.open_call_index = None;
                    push_text_delta(events, AssistantBlockKind::Text, delta);
                }
                UnifiedParserEvent::Reasoning(delta) => {
                    self.open_call_index = None;
                    push_text_delta(events, AssistantBlockKind::Reasoning, delta);
                }
                UnifiedParserEvent::ToolCall(item) => {
                    self.process_tool_item(item, events)?;
                }
            }
        }

        Ok(())
    }

    /// Apply one parsed tool-call delta emitted by the parser.
    fn process_tool_item(
        &mut self,
        item: vllm_parser::tool::ToolCallDelta,
        events: &mut Vec<AssistantEvent>,
    ) -> Result<()> {
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
            return Ok(());
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
        Ok(())
    }
}

/// Build one plain text event if `delta` is non-empty.
fn text_event(kind: AssistantBlockKind, delta: String) -> Option<AssistantEvent> {
    if delta.is_empty() {
        return None;
    }
    Some(AssistantEvent::TextDelta { kind, delta })
}

/// Push one plain text delta if it is non-empty.
fn push_text_delta(events: &mut Vec<AssistantEvent>, kind: AssistantBlockKind, delta: String) {
    if let Some(event) = text_event(kind, delta) {
        events.push(event);
    }
}

/// Wrap one decoded-text stream into the internal unified assistant stream.
#[try_stream]
pub(crate) async fn unified_event_stream(
    decoded_stream: impl DecodedTextEventStream,
    parser: Box<dyn UnifiedParser>,
    mut y: TryYielder<AssistantEvent, Error>,
) -> Result<()> {
    pin_mut!(decoded_stream);

    let mut state = UnifiedParserState::new(parser);

    while let Some(event) = decoded_stream.next().await.transpose()? {
        match event {
            DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => {
                state.initialize(&prompt_token_ids);
                y.yield_ok(AssistantEvent::Start {
                    prompt_token_ids,
                    prompt_logprobs,
                })
                .await;
            }
            DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
                finished,
            } => {
                for next in state.process_delta(delta)? {
                    y.yield_ok(next).await;
                }
                if logprobs.is_some() || !token_ids.is_empty() {
                    y.yield_ok(AssistantEvent::LogprobsDelta {
                        logprobs,
                        token_ids,
                    })
                    .await;
                }
                if let Some(finished) = finished {
                    for next in state.finish()? {
                        y.yield_ok(next).await;
                    }
                    y.yield_ok(AssistantEvent::Done {
                        usage: finished.usage,
                        finish_reason: finished.finish_reason,
                        kv_transfer_params: finished.kv_transfer_params,
                        ec_transfer_params: finished.ec_transfer_params,
                    })
                    .await;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;

    use futures::{StreamExt as _, stream};
    use vllm_parser::reasoning::ReasoningError;
    use vllm_parser::tool::{Tool, ToolCallDelta};
    use vllm_parser::unified::{Gemma4UnifiedParser, UnifiedParserError, UnifiedParserOutput};
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::unified_event_stream;
    use crate::event::AssistantBlockKind;
    use crate::output::AssistantEvent;

    enum ScriptedStep {
        Output(UnifiedParserOutput),
        Error {
            committed: UnifiedParserOutput,
            reset_text: String,
        },
    }

    struct ScriptedParser {
        steps: VecDeque<ScriptedStep>,
        reset_text: String,
        tool_call_id: Option<String>,
        finish_error_reset_text: Option<String>,
    }

    impl ScriptedParser {
        fn new(steps: impl IntoIterator<Item = ScriptedStep>) -> Self {
            Self {
                steps: steps.into_iter().collect(),
                reset_text: String::new(),
                tool_call_id: Some("call_test".to_string()),
                finish_error_reset_text: None,
            }
        }

        fn with_finish_error(mut self, reset_text: &str) -> Self {
            self.finish_error_reset_text = Some(reset_text.to_string());
            self
        }
    }

    impl vllm_parser::unified::UnifiedParser for ScriptedParser {
        fn create(
            _tools: &[vllm_parser::tool::Tool],
            _tokenizer: vllm_tokenizer::DynTokenizer,
        ) -> vllm_parser::unified::Result<Box<dyn vllm_parser::unified::UnifiedParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self::new([])))
        }

        fn parse_into(
            &mut self,
            _delta: &str,
            output: &mut UnifiedParserOutput,
        ) -> vllm_parser::unified::Result<()> {
            match self.steps.pop_front().expect("unexpected parser call") {
                ScriptedStep::Output(next) => {
                    output.append(next);
                    Ok(())
                }
                ScriptedStep::Error {
                    committed,
                    reset_text,
                } => {
                    output.append(committed);
                    self.reset_text = reset_text;
                    Err(UnifiedParserError::Reasoning(
                        ReasoningError::MissingToken {
                            token: "<think>".to_string(),
                        },
                    ))
                }
            }
        }

        fn tool_call_id(&self, _tool_index: usize) -> Option<&str> {
            self.tool_call_id.as_deref()
        }

        fn finish(&mut self) -> vllm_parser::unified::Result<UnifiedParserOutput> {
            if let Some(reset_text) = self.finish_error_reset_text.take() {
                self.reset_text = reset_text;
                return Err(UnifiedParserError::Reasoning(
                    ReasoningError::MissingToken {
                        token: "</tool_call>".to_string(),
                    },
                ));
            }
            Ok(UnifiedParserOutput::default())
        }

        fn reset(&mut self) -> String {
            std::mem::take(&mut self.reset_text)
        }
    }

    fn decoded_delta(delta: &str) -> vllm_text::output::DecodedTextEvent {
        vllm_text::output::DecodedTextEvent::TextDelta {
            delta: delta.to_string(),
            token_ids: Vec::new(),
            logprobs: None,
            finished: None,
        }
    }

    fn finished_delta(delta: &str) -> vllm_text::output::DecodedTextEvent {
        vllm_text::output::DecodedTextEvent::TextDelta {
            delta: delta.to_string(),
            token_ids: Vec::new(),
            logprobs: None,
            finished: Some(vllm_text::output::Finished {
                usage: vllm_llm::TokenUsage::default(),
                finish_reason: crate::FinishReason::Stop(None),
                kv_transfer_params: None,
                ec_transfer_params: None,
            }),
        }
    }

    async fn collect(
        parser: ScriptedParser,
        events: Vec<vllm_text::output::DecodedTextEvent>,
    ) -> Vec<AssistantEvent> {
        let stream = stream::iter(events.into_iter().map(Ok));
        unified_event_stream(stream, Box::new(parser))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .unwrap()
    }

    fn text(delta: &str) -> UnifiedParserOutput {
        let mut output = UnifiedParserOutput::default();
        output.push_text(delta.to_string());
        output
    }

    fn reasoning(delta: &str) -> UnifiedParserOutput {
        let mut output = UnifiedParserOutput::default();
        output.push_reasoning(delta.to_string());
        output
    }

    fn tool_call(name: &str, arguments: &str) -> UnifiedParserOutput {
        UnifiedParserOutput {
            events: vec![vllm_parser::unified::UnifiedParserEvent::ToolCall(
                ToolCallDelta {
                    tool_index: 0,
                    name: Some(name.to_string()),
                    arguments: arguments.to_string(),
                },
            )],
        }
    }

    fn tool_call_arguments(arguments: &str) -> UnifiedParserOutput {
        UnifiedParserOutput {
            events: vec![vllm_parser::unified::UnifiedParserEvent::ToolCall(
                ToolCallDelta {
                    tool_index: 0,
                    name: None,
                    arguments: arguments.to_string(),
                },
            )],
        }
    }

    fn combined(first: UnifiedParserOutput, second: UnifiedParserOutput) -> UnifiedParserOutput {
        let mut output = first;
        output.append(second);
        output
    }

    #[tokio::test]
    async fn unified_stream_parses_formatted_tool_call_without_latch() {
        use vllm_parser::tool::{HermesToolParser, ToolParser as _};

        let hermes = HermesToolParser::create(&[]).unwrap();
        let parser = vllm_parser::unified::CombinedParser::new(None, Some(hermes));

        // Regression guard: a formatted call (space before the outer `}`) must not
        // trip the parse-error latch that turns it and every later call into text.
        let d1 = decoded_delta(
            r#"<tool_call>{"name":"get_weather","arguments":{"location":"Paris"} }</tool_call>"#,
        );
        let d2 = finished_delta(
            r#"<tool_call>{"name":"get_time","arguments":{"tz":"UTC"}}</tool_call>"#,
        );

        let stream = stream::iter(vec![d1, d2].into_iter().map(Ok));
        let events = unified_event_stream(stream, Box::new(parser))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .unwrap();

        let names: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                AssistantEvent::ToolCallStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["get_weather", "get_time"]);

        let text_leaks = events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    AssistantEvent::TextDelta {
                        kind: AssistantBlockKind::Text,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            text_leaks, 0,
            "formatted tool call leaked into text: {events:#?}"
        );
    }

    #[tokio::test]
    async fn unified_stream_emits_reasoning_only_deltas() {
        let events = collect(
            ScriptedParser::new([ScriptedStep::Output(reasoning("thinking"))]),
            vec![decoded_delta("raw")],
        )
        .await;

        assert_eq!(
            events,
            vec![AssistantEvent::TextDelta {
                kind: AssistantBlockKind::Reasoning,
                delta: "thinking".to_string(),
            }]
        );
    }

    #[tokio::test]
    async fn unified_stream_emits_tool_only_deltas() {
        let events = collect(
            ScriptedParser::new([ScriptedStep::Output(tool_call(
                "get_weather",
                r#"{"location":"Paris"}"#,
            ))]),
            vec![decoded_delta("raw")],
        )
        .await;

        assert_eq!(
            events,
            vec![
                AssistantEvent::ToolCallStart {
                    id: "call_test".to_string(),
                    name: "get_weather".to_string(),
                },
                AssistantEvent::ToolCallArgumentsDelta {
                    delta: r#"{"location":"Paris"}"#.to_string(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn unified_stream_emits_reasoning_followed_by_tool_call() {
        let events = collect(
            ScriptedParser::new([ScriptedStep::Output(combined(
                reasoning("thinking"),
                tool_call("get_weather", r#"{"location":"Paris"}"#),
            ))]),
            vec![decoded_delta("raw")],
        )
        .await;

        assert_eq!(
            events,
            vec![
                AssistantEvent::TextDelta {
                    kind: AssistantBlockKind::Reasoning,
                    delta: "thinking".to_string(),
                },
                AssistantEvent::ToolCallStart {
                    id: "call_test".to_string(),
                    name: "get_weather".to_string(),
                },
                AssistantEvent::ToolCallArgumentsDelta {
                    delta: r#"{"location":"Paris"}"#.to_string(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn unified_stream_emits_visible_text_followed_by_tool_call() {
        let events = collect(
            ScriptedParser::new([ScriptedStep::Output(combined(
                text("visible "),
                tool_call("get_weather", r#"{"location":"Paris"}"#),
            ))]),
            vec![decoded_delta("raw")],
        )
        .await;

        assert_eq!(
            events,
            vec![
                AssistantEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "visible ".to_string(),
                },
                AssistantEvent::ToolCallStart {
                    id: "call_test".to_string(),
                    name: "get_weather".to_string(),
                },
                AssistantEvent::ToolCallArgumentsDelta {
                    delta: r#"{"location":"Paris"}"#.to_string(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn unified_stream_emits_tool_arguments_before_trailing_text() {
        let events = collect(
            ScriptedParser::new([
                ScriptedStep::Output(tool_call("get_weather", "")),
                ScriptedStep::Output(combined(
                    tool_call_arguments(r#"{"location":"Paris"}"#),
                    text(" done"),
                )),
            ]),
            vec![decoded_delta("start"), decoded_delta("finish")],
        )
        .await;

        assert_eq!(
            events,
            vec![
                AssistantEvent::ToolCallStart {
                    id: "call_test".to_string(),
                    name: "get_weather".to_string(),
                },
                AssistantEvent::ToolCallArgumentsDelta {
                    delta: r#"{"location":"Paris"}"#.to_string(),
                },
                AssistantEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: " done".to_string(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn unified_stream_fallback_keeps_committed_output_and_disables_later_parsing() {
        let events = collect(
            ScriptedParser::new([ScriptedStep::Error {
                committed: text("committed"),
                reset_text: "buffered".to_string(),
            }]),
            vec![decoded_delta("bad"), decoded_delta("later")],
        )
        .await;

        assert_eq!(
            events,
            vec![
                AssistantEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "committed".to_string(),
                },
                AssistantEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "buffered".to_string(),
                },
                AssistantEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "later".to_string(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn unified_stream_finish_error_recovers_buffered_text() {
        let events = collect(
            ScriptedParser::new([ScriptedStep::Output(UnifiedParserOutput::default())])
                .with_finish_error("buffered"),
            vec![finished_delta("")],
        )
        .await;

        assert_eq!(
            events,
            vec![
                AssistantEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "buffered".to_string(),
                },
                AssistantEvent::Done {
                    usage: vllm_llm::TokenUsage::default(),
                    finish_reason: crate::FinishReason::Stop(None),
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                },
            ]
        );
    }

    #[tokio::test]
    async fn unified_stream_recovers_incomplete_gemma4_tool_call_at_eos() {
        let tokenizer = TestTokenizer::new()
            .with_special_token("<|channel>", 256)
            .with_special_token("<channel|>", 257);
        let tools = vec![Tool {
            name: "write_file".to_string(),
            description: None,
            parameters: serde_json::json!({ "type": "object" }),
            strict: None,
        }];
        let parser = Gemma4UnifiedParser::new(&tools, Arc::new(tokenizer)).unwrap();
        let events = vec![
            decoded_delta("<|tool_call>"),
            decoded_delta("call:write_file{"),
            decoded_delta("content:<|\"|>hello "),
            finished_delta("world<|\"|>"),
        ];
        let stream = stream::iter(events.into_iter().map(Ok));
        let events = unified_event_stream(stream, Box::new(parser))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
            .unwrap();

        assert_eq!(
            events,
            vec![
                AssistantEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "<|tool_call>call:write_file{content:<|\"|>hello world<|\"|>"
                        .to_string(),
                },
                AssistantEvent::Done {
                    usage: vllm_llm::TokenUsage::default(),
                    finish_reason: crate::FinishReason::Stop(None),
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                },
            ]
        );
    }
}
