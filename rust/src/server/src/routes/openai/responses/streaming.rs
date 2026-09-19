// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Streaming event state machine for the Responses API.
//!
//! Maps the structured `vllm-chat` event stream (`ChatEvent`) onto the OpenAI
//! Responses SSE event protocol. Event shapes and emission order mirror the
//! simple (non-Harmony) streaming path of the Python frontend in
//! `vllm/entrypoints/openai/responses/streaming_events.py`, with one
//! deliberate improvement: item IDs stay consistent between streamed events
//! and the terminal `response.completed` payload.

use serde_json::{Map, Value};
use uuid::Uuid;
use vllm_chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall, ChatEvent,
};

use super::convert::build_output_items;
use super::types::{
    AssistantRole, ResponseItemStatus, ResponseOutputContentPart, ResponseOutputItem, TextPart,
};

/// One Responses API SSE event.
///
/// The flat wire shape carries `type` and `sequence_number` next to the
/// event-specific payload (`response`, `item`, `part`, ...), matching how the
/// Python frontend serializes its typed SDK events. `sequence_number` is
/// assigned centrally by the SSE encoder for all events of one request.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResponseStreamEvent {
    /// The wire `type` string (e.g. `response.output_text.delta`).
    event_type: &'static str,
    /// Event-specific payload fields, serialized flat after `type`.
    payload: Map<String, Value>,
}

impl ResponseStreamEvent {
    /// Return the wire event type (used for the SSE `event:` line).
    pub(crate) fn event_type(&self) -> &'static str {
        self.event_type
    }

    /// Serialize the full wire payload including `type` and the assigned
    /// sequence number.
    pub(crate) fn to_json(&self, sequence_number: u64) -> String {
        let mut flattened = Map::with_capacity(self.payload.len() + 2);
        flattened.insert(
            "type".to_string(),
            Value::String(self.event_type.to_string()),
        );
        flattened.insert("sequence_number".to_string(), Value::from(sequence_number));
        flattened.extend(self.payload.clone());
        serde_json::to_string(&Value::Object(flattened))
            .expect("stream event payload must serialize to JSON")
    }
}

/// Build one lifecycle event (`response.created`, `response.in_progress`,
/// `response.completed`, `response.failed`).
pub(crate) fn response_lifecycle_event(
    event_type: &'static str,
    response: &super::types::ResponsesResponse,
) -> ResponseStreamEvent {
    let response = serde_json::to_value(response).expect("response must serialize to JSON");
    ResponseStreamEvent {
        event_type,
        payload: Map::from_iter([("response".to_string(), response)]),
    }
}

/// State machine that maps one `vllm-chat` event stream onto Responses API
/// SSE events.
///
/// Items open lazily on the first non-empty delta so empty blocks never
/// appear in the stream, matching the Python state machine. Item IDs assigned
/// here are matched back onto the final response's output items in
/// [`Self::final_output_items`] so streamed events and the terminal payload
/// agree about IDs.
pub(crate) struct OutputItemStreamer {
    /// Index of the next output item.
    output_index: u32,
    /// The currently open output item, if any.
    current: Option<OpenItem>,
    /// Whether reasoning items appear in the final output. Reasoning deltas
    /// still stream regardless (Python parity), they are only excluded from
    /// the terminal payload.
    include_reasoning: bool,
    /// IDs of streamed items in emission order, together with their block
    /// kind. Used to keep final-response item IDs consistent with the stream.
    streamed_ids: Vec<(AssistantBlockKind, String)>,
}

/// The currently streamed assistant output item.
enum OpenItem {
    Reasoning {
        item_id: String,
        text: String,
    },
    Message {
        item_id: String,
        text: String,
    },
    FunctionCall {
        item_id: String,
        call_id: String,
        name: String,
        arguments: String,
        saw_delta: bool,
    },
}

impl OutputItemStreamer {
    pub(crate) fn new(include_reasoning: bool) -> Self {
        Self {
            output_index: 0,
            current: None,
            include_reasoning,
            streamed_ids: Vec::new(),
        }
    }

    /// Process one chat event, returning the SSE events to emit.
    pub(crate) fn on_event(&mut self, event: &ChatEvent) -> Vec<ResponseStreamEvent> {
        match event {
            ChatEvent::Start { .. } | ChatEvent::LogprobsDelta { .. } | ChatEvent::Done { .. } => {
                vec![]
            }
            ChatEvent::BlockStart { .. } => vec![],
            ChatEvent::BlockDelta { kind, delta, .. } => {
                if delta.is_empty() {
                    return vec![];
                }
                match kind {
                    AssistantBlockKind::Reasoning => {
                        let mut events = self.open_reasoning_if_needed();
                        events.push(self.reasoning_delta(delta.clone()));
                        events
                    }
                    AssistantBlockKind::Text => {
                        let mut events = self.open_message_if_needed();
                        events.push(self.text_delta(delta.clone()));
                        events
                    }
                    // Tool-calls flow through the dedicated events.
                    AssistantBlockKind::ToolCall => vec![],
                }
            }
            ChatEvent::BlockEnd { block, .. } => match block.kind() {
                AssistantBlockKind::Reasoning => self.close_reasoning(block_text(block)),
                AssistantBlockKind::Text => self.close_message(block_text(block)),
                // Tool-call blocks flow through the dedicated events.
                AssistantBlockKind::ToolCall => vec![],
            },
            ChatEvent::ToolCallStart { id, name, .. } => {
                let mut events = self.close_current(None);
                events.push(self.open_function_call(id, name));
                events
            }
            ChatEvent::ToolCallArgumentsDelta { delta, .. } => {
                if delta.is_empty() {
                    return vec![];
                }
                vec![self.function_call_delta(delta.clone())]
            }
            ChatEvent::ToolCallEnd { call, .. } => self.close_current(Some(call)),
        }
    }

    /// Close any still-open item, returning the close events. Defensive: the
    /// chat event contract closes blocks and tool calls explicitly.
    pub(crate) fn on_stream_end(&mut self) -> Vec<ResponseStreamEvent> {
        self.close_current(None)
    }

    /// Build the final response output items, reusing streamed item IDs so
    /// streamed events and the terminal payload agree about IDs.
    pub(crate) fn final_output_items(&self, message: &AssistantMessage) -> Vec<ResponseOutputItem> {
        let mut items = build_output_items(message, self.include_reasoning);
        let mut consumed = vec![false; self.streamed_ids.len()];
        for item in &mut items {
            let kind = match item {
                ResponseOutputItem::Reasoning { .. } => AssistantBlockKind::Reasoning,
                ResponseOutputItem::Message { .. } => AssistantBlockKind::Text,
                ResponseOutputItem::FunctionCall { .. } => AssistantBlockKind::ToolCall,
            };
            let Some(index) = self
                .streamed_ids
                .iter()
                .enumerate()
                .find(|(index, (streamed_kind, _))| *streamed_kind == kind && !consumed[*index])
                .map(|(index, _)| index)
            else {
                continue;
            };
            consumed[index] = true;
            let (_, id) = &self.streamed_ids[index];
            let id = id.clone();
            match item {
                ResponseOutputItem::Message { id: slot, .. }
                | ResponseOutputItem::FunctionCall { id: slot, .. }
                | ResponseOutputItem::Reasoning { id: slot, .. } => *slot = id,
            }
        }
        items
    }

    fn open_reasoning_if_needed(&mut self) -> Vec<ResponseStreamEvent> {
        if self.current.is_some() {
            return vec![];
        }
        let item_id = format!("rs_{}", Uuid::new_v4().simple());
        let item = ResponseOutputItem::Reasoning {
            id: item_id.clone(),
            summary: vec![],
            content: None,
            status: Some(ResponseItemStatus::InProgress),
        };
        if self.include_reasoning {
            self.streamed_ids.push((AssistantBlockKind::Reasoning, item_id.clone()));
        }
        let output_index = self.output_index;
        self.current = Some(OpenItem::Reasoning {
            item_id: item_id.clone(),
            text: String::new(),
        });
        vec![
            output_item_event("response.output_item.added", output_index, item),
            part_event(
                "response.reasoning_part.added",
                output_index,
                &item_id,
                [(
                    "part",
                    serde_json::json!({"type": "reasoning_text", "text": ""}),
                )],
            ),
        ]
    }

    fn open_message_if_needed(&mut self) -> Vec<ResponseStreamEvent> {
        if self.current.is_some() {
            return vec![];
        }
        let item_id = format!("msg_{}", Uuid::new_v4().simple());
        let item = ResponseOutputItem::Message {
            id: item_id.clone(),
            role: AssistantRole,
            status: ResponseItemStatus::InProgress,
            content: vec![],
        };
        self.streamed_ids.push((AssistantBlockKind::Text, item_id.clone()));
        let output_index = self.output_index;
        self.current = Some(OpenItem::Message {
            item_id: item_id.clone(),
            text: String::new(),
        });
        vec![
            output_item_event("response.output_item.added", output_index, item),
            content_part_added(output_index, &item_id),
        ]
    }

    fn open_function_call(&mut self, id: &str, name: &str) -> ResponseStreamEvent {
        let item_id = format!("fc_{}", Uuid::new_v4().simple());
        let call_id = if id.is_empty() {
            format!("call_{}", Uuid::new_v4().simple())
        } else {
            id.to_string()
        };
        self.streamed_ids.push((AssistantBlockKind::ToolCall, item_id.clone()));
        let output_index = self.output_index;
        self.current = Some(OpenItem::FunctionCall {
            item_id: item_id.clone(),
            call_id: call_id.clone(),
            name: name.to_string(),
            arguments: String::new(),
            saw_delta: false,
        });
        let item = ResponseOutputItem::FunctionCall {
            id: item_id,
            call_id,
            name: name.to_string(),
            arguments: String::new(),
            status: Some(ResponseItemStatus::InProgress),
        };
        output_item_event("response.output_item.added", output_index, item)
    }

    fn reasoning_delta(&mut self, delta: String) -> ResponseStreamEvent {
        let Some(OpenItem::Reasoning { item_id, text }) = self.current.as_mut() else {
            unreachable!("reasoning delta requires an open reasoning item");
        };
        text.push_str(&delta);
        part_event(
            "response.reasoning_text.delta",
            self.output_index,
            item_id,
            [("delta", Value::String(delta))],
        )
    }

    fn text_delta(&mut self, delta: String) -> ResponseStreamEvent {
        let Some(OpenItem::Message { item_id, text }) = self.current.as_mut() else {
            unreachable!("text delta requires an open message item");
        };
        text.push_str(&delta);
        part_event(
            "response.output_text.delta",
            self.output_index,
            item_id,
            [
                ("delta", Value::String(delta)),
                ("logprobs", Value::Array(vec![])),
            ],
        )
    }

    fn function_call_delta(&mut self, delta: String) -> ResponseStreamEvent {
        let Some(OpenItem::FunctionCall {
            item_id,
            arguments,
            saw_delta,
            ..
        }) = self.current.as_mut()
        else {
            unreachable!("function call delta requires an open function call item");
        };
        arguments.push_str(&delta);
        *saw_delta = true;
        part_event(
            "response.function_call_arguments.delta",
            self.output_index,
            item_id,
            [("delta", Value::String(delta))],
        )
    }

    /// Close the current item, emitting the done event sequence.
    ///
    /// `final_call` overrides the streamed tool-call payload with the parser's
    /// final tool-call block when available.
    fn close_current(
        &mut self,
        final_call: Option<&AssistantToolCall>,
    ) -> Vec<ResponseStreamEvent> {
        let Some(open) = self.current.take() else {
            return vec![];
        };
        let output_index = self.output_index;
        let events = match open {
            OpenItem::Reasoning { item_id, text } => {
                let part = TextPart::reasoning_text(text.clone());
                let item = ResponseOutputItem::Reasoning {
                    id: item_id.clone(),
                    summary: vec![],
                    content: Some(vec![part.clone()]),
                    status: Some(ResponseItemStatus::Completed),
                };
                vec![
                    part_event(
                        "response.reasoning_text.done",
                        output_index,
                        &item_id,
                        [("text", Value::String(text))],
                    ),
                    part_event(
                        "response.reasoning_part.done",
                        output_index,
                        &item_id,
                        [(
                            "part",
                            serde_json::to_value(part).expect("part must serialize"),
                        )],
                    ),
                    output_item_event("response.output_item.done", output_index, item),
                ]
            }
            OpenItem::Message { item_id, text } => {
                let part = ResponseOutputContentPart::OutputText {
                    text: text.clone(),
                    annotations: vec![],
                    logprobs: None,
                };
                let item = ResponseOutputItem::Message {
                    id: item_id.clone(),
                    role: AssistantRole,
                    status: ResponseItemStatus::Completed,
                    content: vec![part.clone()],
                };
                vec![
                    part_event(
                        "response.output_text.done",
                        output_index,
                        &item_id,
                        [
                            ("text", Value::String(text)),
                            ("logprobs", Value::Array(vec![])),
                        ],
                    ),
                    part_event(
                        "response.content_part.done",
                        output_index,
                        &item_id,
                        [(
                            "part",
                            serde_json::to_value(&part).expect("part must serialize"),
                        )],
                    ),
                    output_item_event("response.output_item.done", output_index, item),
                ]
            }
            OpenItem::FunctionCall {
                item_id,
                call_id,
                name,
                arguments,
                saw_delta,
            } => {
                let (call_id, name, arguments) = match final_call {
                    Some(call) => (
                        if call.id.is_empty() {
                            call_id
                        } else {
                            call.id.clone()
                        },
                        call.name.clone(),
                        call.arguments.clone(),
                    ),
                    None => (call_id, name, arguments),
                };
                let mut events = Vec::new();
                if saw_delta {
                    events.push(part_event(
                        "response.function_call_arguments.done",
                        output_index,
                        &item_id,
                        [
                            ("arguments", Value::String(arguments.clone())),
                            ("name", Value::String(name.clone())),
                        ],
                    ));
                }
                events.push(output_item_event(
                    "response.output_item.done",
                    output_index,
                    ResponseOutputItem::FunctionCall {
                        id: item_id,
                        call_id,
                        name,
                        arguments,
                        status: Some(ResponseItemStatus::Completed),
                    },
                ));
                events
            }
        };
        self.output_index += 1;
        events
    }

    /// Close the reasoning item if it is currently open. The block text
    /// assembled by the parser is authoritative over accumulated deltas.
    fn close_reasoning(&mut self, final_text: Option<&str>) -> Vec<ResponseStreamEvent> {
        if let Some(OpenItem::Reasoning { text, .. }) = self.current.as_mut()
            && let Some(final_text) = final_text
        {
            *text = final_text.to_string();
        }
        if !matches!(self.current, Some(OpenItem::Reasoning { .. })) {
            // Empty reasoning blocks never opened; nothing to close.
            return vec![];
        }
        self.close_current(None)
    }

    /// Close the message item if it is currently open. The block text
    /// assembled by the parser is authoritative over accumulated deltas.
    fn close_message(&mut self, final_text: Option<&str>) -> Vec<ResponseStreamEvent> {
        if let Some(OpenItem::Message { text, .. }) = self.current.as_mut()
            && let Some(final_text) = final_text
        {
            *text = final_text.to_string();
        }
        if !matches!(self.current, Some(OpenItem::Message { .. })) {
            // Empty text blocks never opened; nothing to close.
            return vec![];
        }
        self.close_current(None)
    }
}

/// Extract the text of one text/reasoning block.
fn block_text(block: &AssistantContentBlock) -> Option<&str> {
    match block {
        AssistantContentBlock::Reasoning { text } | AssistantContentBlock::Text { text } => {
            Some(text)
        }
        AssistantContentBlock::ToolCall(_) => None,
    }
}

/// Build one `response.output_item.added`/`response.output_item.done` event.
fn output_item_event(
    event_type: &'static str,
    output_index: u32,
    item: ResponseOutputItem,
) -> ResponseStreamEvent {
    let item = serde_json::to_value(item).expect("output item must serialize to JSON");
    ResponseStreamEvent {
        event_type,
        payload: Map::from_iter([
            ("output_index".to_string(), Value::from(output_index)),
            ("item".to_string(), item),
        ]),
    }
}

/// Build one `response.content_part.added` event.
fn content_part_added(output_index: u32, item_id: &str) -> ResponseStreamEvent {
    part_event(
        "response.content_part.added",
        output_index,
        item_id,
        [(
            "part",
            serde_json::json!({
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            }),
        )],
    )
}

/// Build one part-scoped event carrying the shared
/// `item_id`/`output_index`/`content_index` frame.
fn part_event<const N: usize>(
    event_type: &'static str,
    output_index: u32,
    item_id: &str,
    payload: [(&str, Value); N],
) -> ResponseStreamEvent {
    let mut fields = Map::with_capacity(N + 3);
    fields.insert("item_id".to_string(), Value::String(item_id.to_string()));
    fields.insert("output_index".to_string(), Value::from(output_index));
    fields.insert("content_index".to_string(), Value::from(0u32));
    for (key, value) in payload {
        fields.insert(key.to_string(), value);
    }
    ResponseStreamEvent {
        event_type,
        payload: fields,
    }
}
