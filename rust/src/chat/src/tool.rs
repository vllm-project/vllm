//! Adapts plain assistant text deltas into tool-call-aware assistant updates.
//!
//! This stage runs after reasoning separation and before final block assembly.
//! It only inspects normal assistant text, leaves reasoning deltas untouched,
//! and translates incremental `tool-parser` output into internal tool-call
//! events while preserving plain-text fallback behavior.

use std::collections::BTreeMap;

use futures::{StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use openai_protocol::common::Tool as OpenAiTool;
use thiserror_ext::AsReport;
use tool_parser::ToolParser;
use tracing::warn;
use uuid::Uuid;

use crate::error::Error;
use crate::event::{AssistantBlockKind, AssistantToolCall};
use crate::pipeline::{AssistantStreamEvent, AssistantStreamEventStream};
use crate::request::ChatRequest;

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
    /// Optional parser for the current model family.
    parser: Option<Box<dyn ToolParser>>,
    /// Whether tool parsing has already failed for this stream.
    parser_failed: bool,
    /// Northbound tool definitions made available to the parser.
    tools: Vec<OpenAiTool>,
    /// Backend model ID, used for parser-specific ID conventions.
    model_id: Option<String>,
    /// Number of historical assistant tool calls in the request.
    history_tool_calls_count: usize,
    /// Open tool calls keyed by the parser's tool index.
    open_calls: BTreeMap<usize, OpenToolCallState>,
}

impl ToolState {
    /// Create one fresh tool-parsing state for a new streamed response.
    fn new(
        request: &ChatRequest,
        parser: Option<Box<dyn ToolParser>>,
        model_id: Option<String>,
    ) -> Self {
        Self {
            parser,
            parser_failed: false,
            tools: request.parser_tools().unwrap_or_default(),
            model_id,
            history_tool_calls_count: request.history_tool_call_count(),
            open_calls: BTreeMap::new(),
        }
    }

    /// Convert one semantic assistant text delta into zero or more tool-aware
    /// internal events.
    async fn process_text_delta(
        &mut self,
        kind: AssistantBlockKind,
        delta: String,
    ) -> Vec<AssistantStreamEvent> {
        let mut events = Vec::new();

        // Only normal assistant text is eligible for tool parsing. Reasoning
        // blocks and plain-text fallback should pass through unchanged.
        if kind != AssistantBlockKind::Text || self.parser.is_none() {
            self.close_all_open_calls(&mut events);
            events.push(AssistantStreamEvent::TextDelta { kind, delta });
            return events;
        }

        let parse_result = {
            let parser = self
                .parser
                .as_mut()
                .expect("tool parser must exist when parsing is enabled");
            parser.parse_incremental(&delta, &self.tools).await
        };

        match parse_result {
            Ok(result) => {
                // When we are not currently streaming a tool call, preserve
                // plain text first and then surface any new tool call items.
                if self.open_calls.is_empty() {
                    push_text_delta(&mut events, kind, result.normal_text);
                    self.process_tool_items(result.calls, &mut events);
                } else {
                    // Once a tool call is open, prioritize tool deltas first.
                    // If the parser emits normal text again, close the tool
                    // call and resume plain text output.
                    self.process_tool_items(result.calls, &mut events);
                    if !result.normal_text.is_empty() {
                        self.close_all_open_calls(&mut events);
                        push_text_delta(&mut events, kind, result.normal_text);
                    }
                }
            }
            Err(error) => {
                if !self.parser_failed {
                    warn!(
                        error = %error.as_report(),
                        model_id = ?self.model_id,
                        "tool parser failed; falling back to plain text deltas"
                    );
                    self.parser_failed = true;
                }
                self.parser = None;
                self.close_all_open_calls(&mut events);
                events.push(AssistantStreamEvent::TextDelta { kind, delta });
            }
        }

        events
    }

    /// Apply one batch of incremental tool-call items emitted by the parser.
    fn process_tool_items(
        &mut self,
        items: Vec<tool_parser::types::ToolCallItem>,
        events: &mut Vec<AssistantStreamEvent>,
    ) {
        for item in items {
            if let Some(name) = item.name {
                // The parser is now advancing a specific tool index, so any
                // previously open sibling calls must be finalized first.
                self.close_calls_not_matching(item.tool_index, events);
                if !self.open_calls.contains_key(&item.tool_index) {
                    let id = generate_tool_call_id(
                        self.model_id.as_deref(),
                        &name,
                        item.tool_index,
                        self.history_tool_calls_count,
                    );
                    self.open_calls.insert(
                        item.tool_index,
                        OpenToolCallState {
                            id: id.clone(),
                            name: name.clone(),
                            arguments: String::new(),
                        },
                    );
                    events.push(AssistantStreamEvent::ToolCallStart { id, name });
                }
            }

            if item.parameters.is_empty() {
                continue;
            }

            let Some(open_call) = self.open_calls.get_mut(&item.tool_index) else {
                continue;
            };
            open_call.arguments.push_str(&item.parameters);
            events.push(AssistantStreamEvent::ToolCallArgumentsDelta {
                id: open_call.id.clone(),
                delta: item.parameters,
            });
        }
    }

    /// Close every open tool call except the one still associated with the
    /// parser's current tool index.
    fn close_calls_not_matching(
        &mut self,
        keep_tool_index: usize,
        events: &mut Vec<AssistantStreamEvent>,
    ) {
        let to_close: Vec<_> = self
            .open_calls
            .keys()
            .copied()
            .filter(|tool_index| *tool_index != keep_tool_index)
            .collect();
        for tool_index in to_close {
            if let Some(open_call) = self.open_calls.remove(&tool_index) {
                events.push(AssistantStreamEvent::ToolCallEnd {
                    call: AssistantToolCall {
                        id: open_call.id,
                        name: open_call.name,
                        arguments: open_call.arguments,
                    },
                });
            }
        }
    }

    /// Close every currently open tool call.
    fn close_all_open_calls(&mut self, events: &mut Vec<AssistantStreamEvent>) {
        let to_close: Vec<_> = self.open_calls.keys().copied().collect();
        for tool_index in to_close {
            if let Some(open_call) = self.open_calls.remove(&tool_index) {
                let _ = tool_index;
                events.push(AssistantStreamEvent::ToolCallEnd {
                    call: AssistantToolCall {
                        id: open_call.id,
                        name: open_call.name,
                        arguments: open_call.arguments,
                    },
                });
            }
        }
    }
}

/// Push one plain-text delta if it is non-empty.
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

/// Generate the northbound tool-call ID for one parsed call.
///
/// Most models use an OpenAI-style `call_<id>` identifier. Kimi-family models
/// instead mirror vLLM / SMG's `functions.{name}:{global_index}` convention,
/// where `global_index` includes historical assistant tool calls from earlier
/// turns.
fn generate_tool_call_id(
    model_id: Option<&str>,
    function_name: &str,
    tool_index: usize,
    history_tool_calls_count: usize,
) -> String {
    if model_id.is_some_and(|model| model.to_ascii_lowercase().contains("kimi")) {
        return format!(
            "functions.{}:{}",
            function_name,
            history_tool_calls_count + tool_index
        );
    }

    format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
}

/// Wrap one semantic assistant stream into the internal tool-aware assistant
/// stream.
#[try_stream(ok = AssistantStreamEvent, error = Error)]
pub(crate) async fn tool_event_stream(
    stream: impl AssistantStreamEventStream,
    request: ChatRequest,
    parser: Option<Box<dyn ToolParser>>,
    model_id: Option<String>,
) {
    pin_mut!(stream);

    let mut state = ToolState::new(&request, parser, model_id);

    while let Some(event) = stream.next().await.transpose()? {
        match event {
            AssistantStreamEvent::Start => yield AssistantStreamEvent::Start,
            AssistantStreamEvent::TextDelta { kind, delta } => {
                for next in state.process_text_delta(kind, delta).await {
                    yield next;
                }
            }
            AssistantStreamEvent::Done {
                token_ids,
                finish_reason,
                stop_reason,
            } => {
                let mut flush_events = Vec::new();
                // Some parsers buffer a trailing arguments fragment and only
                // expose it once streaming is complete.
                if let Some(parser) = state.parser.as_ref()
                    && let Some(remaining) = parser.get_unstreamed_tool_args()
                {
                    state.process_tool_items(remaining, &mut flush_events);
                }
                state.close_all_open_calls(&mut flush_events);
                for next in flush_events {
                    yield next;
                }

                yield AssistantStreamEvent::Done {
                    token_ids,
                    finish_reason,
                    stop_reason,
                };
            }
            AssistantStreamEvent::ToolCallStart { .. }
            | AssistantStreamEvent::ToolCallArgumentsDelta { .. }
            | AssistantStreamEvent::ToolCallEnd { .. } => {
                unreachable!("tool events must not appear before tool parsing stage")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use futures::{StreamExt as _, stream};
    use openai_protocol::common::Tool;
    use tool_parser::ToolParser;
    use tool_parser::errors::{ParserError, ParserResult};
    use tool_parser::types::{StreamingParseResult, ToolCall, ToolCallItem};
    use vllm_engine_core_client::protocol::FinishReason;

    use super::{generate_tool_call_id, tool_event_stream};
    use crate::UserSamplingParams;
    use crate::event::{AssistantBlockKind, AssistantMessageExt as _};
    use crate::pipeline::AssistantStreamEvent;
    use crate::request::{
        ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool, ChatToolChoice,
    };
    use crate::stream::ChatEventStream;
    use crate::structured::structured_chat_event_stream;

    struct FailingParser {
        fail_next: bool,
    }

    #[async_trait]
    impl ToolParser for FailingParser {
        async fn parse_complete(&self, _output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
            Ok((String::new(), Vec::new()))
        }

        async fn parse_incremental(
            &mut self,
            _chunk: &str,
            _tools: &[Tool],
        ) -> ParserResult<StreamingParseResult> {
            if self.fail_next {
                self.fail_next = false;
                return Err(ParserError::ParsingFailed("boom".to_string()));
            }

            Ok(StreamingParseResult::default())
        }

        fn has_tool_markers(&self, _text: &str) -> bool {
            false
        }

        fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
            None
        }
    }

    fn tool_request(request_id: &str) -> ChatRequest {
        ChatRequest {
            request_id: request_id.to_string(),
            messages: vec![ChatMessage::text(ChatRole::User, "Call the tool.")],
            sampling_params: UserSamplingParams::default(),
            chat_options: ChatOptions::default(),
            tools: vec![ChatTool {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }),
                strict: None,
            }],
            tool_choice: ChatToolChoice::Auto,
        }
    }

    #[tokio::test]
    async fn tool_parser_failure_falls_back_to_plain_text() {
        let events = stream::iter(vec![
            Ok(AssistantStreamEvent::Start),
            Ok(AssistantStreamEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: "abc".to_string(),
            }),
            Ok(AssistantStreamEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta: "def".to_string(),
            }),
            Ok(AssistantStreamEvent::Done {
                token_ids: vec![],
                finish_reason: Some(FinishReason::Stop),
                stop_reason: None,
            }),
        ]);

        let collected = tool_event_stream(
            events,
            tool_request("req_fallback"),
            Some(Box::new(FailingParser { fail_next: true })),
            Some("Qwen/Qwen3-32B".to_string()),
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
                AssistantStreamEvent::Start,
                AssistantStreamEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "abc".to_string(),
                },
                AssistantStreamEvent::TextDelta {
                    kind: AssistantBlockKind::Text,
                    delta: "def".to_string(),
                },
                AssistantStreamEvent::Done {
                    token_ids: vec![],
                    finish_reason: Some(FinishReason::Stop),
                    stop_reason: None,
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
        assert_eq!(message.text(), "abcdef");
        assert!(message.tool_calls().next().is_none());
    }

    #[test]
    fn kimi_tool_call_ids_include_history_offset() {
        assert_eq!(
            generate_tool_call_id(Some("MoonshotAI/Kimi-K2-Instruct"), "lookup", 1, 2),
            "functions.lookup:3"
        );
    }
}
