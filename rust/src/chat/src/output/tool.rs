//! Adapts plain assistant text deltas into tool-call-aware assistant updates.
//!
//! This stage runs after reasoning separation and before final block assembly.
//! It only inspects normal assistant text, leaves reasoning deltas untouched,
//! and translates incremental `tool-parser` output into internal tool-call
//! events while preserving plain-text fallback behavior.

use std::collections::{BTreeMap, btree_map};

use futures::{StreamExt as _, pin_mut};
use futures_async_stream::try_stream;
use openai_protocol::common::Tool as OpenAiTool;
use thiserror_ext::AsReport;
use tool_parser::ToolParser;
use tracing::warn;
use uuid::Uuid;

use super::{AssistantEvent, ContentEvent, ContentEventStream};
use crate::error::Error;
use crate::event::{AssistantBlockKind, AssistantToolCall};
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
    /// Parser for the current model family.
    parser: Box<dyn ToolParser>,
    /// Whether tool parsing has already failed for this stream.
    parser_failed: bool,
    /// Northbound tool definitions made available to the parser.
    tools: Vec<OpenAiTool>,
    /// Open tool calls keyed by the parser's tool index.
    open_calls: BTreeMap<usize, OpenToolCallState>,
}

impl ToolState {
    /// Create one fresh tool-parsing state for a new streamed response.
    fn new(request: &ChatRequest, parser: Box<dyn ToolParser>) -> Self {
        Self {
            parser,
            parser_failed: false,
            tools: request.parser_tools().unwrap_or_default(),
            open_calls: BTreeMap::new(),
        }
    }

    /// Convert one semantic assistant text delta into zero or more tool-aware
    /// internal events.
    async fn process_text_delta(
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

        let parse_result = self.parser.parse_incremental(&delta, &self.tools).await;

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

    /// Apply one batch of incremental tool-call items emitted by the parser.
    fn process_tool_items(
        &mut self,
        items: Vec<tool_parser::types::ToolCallItem>,
        events: &mut Vec<AssistantEvent>,
    ) {
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

            if item.parameters.is_empty() {
                // No arguments delta to apply.
                continue;
            }
            let Some(open_call) = self.open_calls.get_mut(&item.tool_index) else {
                continue;
            };
            open_call.arguments.push_str(&item.parameters);

            events.push(AssistantEvent::ToolCallArgumentsDelta {
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

/// Wrap one semantic assistant stream into the internal tool-aware assistant
/// stream.
#[try_stream(ok = AssistantEvent, error = Error)]
pub(crate) async fn tool_event_stream(
    stream: impl ContentEventStream,
    request: ChatRequest,
    parser: Option<Box<dyn ToolParser>>,
) {
    pin_mut!(stream);

    // Without a parser, pass through the input stream unchanged.
    let Some(parser) = parser else {
        while let Some(event) = stream.next().await.transpose()? {
            yield event.into();
        }
        return Ok(());
    };

    let mut state = ToolState::new(&request, parser);

    while let Some(event) = stream.next().await.transpose()? {
        match event {
            ContentEvent::Start => yield AssistantEvent::Start,
            ContentEvent::TextDelta { kind, delta } => {
                for next in state.process_text_delta(kind, delta).await {
                    yield next;
                }
            }
            ContentEvent::Done {
                prompt_token_count,
                token_ids,
                finish_reason,
                stop_reason,
            } => {
                let mut flush_events = Vec::new();
                // Some parsers buffer a trailing arguments fragment and only
                // expose it once streaming is complete.
                if !state.parser_failed
                    && let Some(remaining) = state.parser.get_unstreamed_tool_args()
                {
                    state.process_tool_items(remaining, &mut flush_events);
                }
                state.close_all_open_calls(&mut flush_events);
                for next in flush_events {
                    yield next;
                }

                yield AssistantEvent::Done {
                    prompt_token_count,
                    token_ids,
                    finish_reason,
                    stop_reason,
                };
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

    use super::super::structured::structured_chat_event_stream;
    use super::super::{AssistantEvent, ContentEvent};
    use super::tool_event_stream;
    use crate::UserSamplingParams;
    use crate::event::{AssistantBlockKind, AssistantMessageExt as _};
    use crate::request::{
        ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool, ChatToolChoice,
    };
    use crate::stream::ChatEventStream;

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
            decode_options: Default::default(),
        }
    }

    #[tokio::test]
    async fn tool_parser_failure_falls_back_to_plain_text() {
        let events = stream::iter(vec![
            Ok(ContentEvent::Start),
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
                token_ids: vec![],
                finish_reason: Some(FinishReason::Stop),
                stop_reason: None,
            }),
        ]);

        let collected = tool_event_stream(
            events,
            tool_request("req_fallback"),
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
                AssistantEvent::Start,
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
}
