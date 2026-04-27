//! Native Harmony output processing for `gpt_oss`.
//!
//! Unlike the default text-first pipeline, this processor consumes `DecodedTextEvent`
//! token IDs directly and lets the official `openai-harmony` parser recover the
//! structured assistant message shape at token granularity.

use std::sync::LazyLock;

use anyhow::Context;
use futures::StreamExt as _;
use futures_async_stream::try_stream;
use openai_harmony::chat::{Content as HarmonyContent, Message as HarmonyMessage, Role};
use openai_harmony::{
    HarmonyEncoding, HarmonyEncodingName, StreamableParser, load_harmony_encoding,
};
use thiserror_ext::AsReport;
use vllm_text::output::DecodedTextEvent;

use crate::Result as ChatResult;
use crate::error::{Error, Result};
use crate::event::AssistantBlockKind;
use crate::output::{
    AssistantEvent, ChatOutputProcessor, DynChatEventStream, DynDecodedTextEventStream,
    generate_tool_call_id,
};
use crate::parser::ParserSelection;
use crate::request::ChatRequest;

/// Request-scoped Harmony output processor used for `model_type == "gpt_oss"`.
///
/// This processor keeps the existing northbound `ChatEvent` shape, but swaps the
/// parsed-assistant backend from generic text/reasoning/tool parsers to the
/// official Harmony token parser.
#[derive(Debug)]
pub struct HarmonyChatOutputProcessor {
    encoding: &'static HarmonyEncoding,
    tool_calls_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct HarmonyGroupKey {
    serial: usize,
    channel: Option<String>,
    recipient: Option<String>,
}

#[derive(Debug)]
struct HarmonyGroup {
    key: HarmonyGroupKey,
    text: String,
}

#[derive(Debug)]
struct OpenHarmonyToolCall {
    recipient: String,
}

struct HarmonyState {
    /// Incremental Harmony parser over assistant token IDs.
    parser: StreamableParser,
    /// Whether tool-call content should surface as structured tool events.
    tool_calls_enabled: bool,
    /// Count of completed visible assistant messages for newline insertion.
    completed_visible_messages: usize,
    /// Count of completed reasoning messages for newline insertion.
    completed_reasoning_messages: usize,
    /// The current visible text/reasoning group, if any.
    current_text_group: Option<HarmonyGroupKey>,
    /// The currently open Harmony tool recipient, if any.
    open_tool_call: Option<OpenHarmonyToolCall>,
}

impl HarmonyChatOutputProcessor {
    /// Build one request-scoped Harmony processor after backend policy checks.
    pub fn new(request: &ChatRequest) -> ChatResult<Self> {
        Ok(Self {
            encoding: harmony_encoding()?,
            tool_calls_enabled: request.tool_parsing_enabled(),
        })
    }
}

/// Validate that the generic parser selections are compatible with native Harmony output parsing.
///
/// `gpt_oss` uses a model-specific token-level parser, so any generic reasoning/tool parser
/// override is rejected instead of being silently ignored.
pub(crate) fn validate_harmony_parser_overrides(
    tool_call_parser: &ParserSelection,
    reasoning_parser: &ParserSelection,
) -> ChatResult<()> {
    validate_harmony_override("tool", tool_call_parser)?;
    validate_harmony_override("reasoning", reasoning_parser)?;
    Ok(())
}

fn validate_harmony_override(kind: &'static str, selection: &ParserSelection) -> ChatResult<()> {
    if matches!(selection, ParserSelection::Auto) {
        return Ok(());
    }

    Err(Error::HarmonyParserOverrideUnsupported {
        kind,
        selection: selection.to_string(),
    })
}

impl ChatOutputProcessor for HarmonyChatOutputProcessor {
    fn process(self: Box<Self>, decoded: DynDecodedTextEventStream) -> Result<DynChatEventStream> {
        let assistant =
            harmony_assistant_event_stream(decoded, self.encoding, self.tool_calls_enabled);
        Ok(crate::output::structured::structured_chat_event_stream(assistant).boxed())
    }
}

impl HarmonyState {
    /// Create one fresh Harmony streaming state for a new assistant response.
    fn new(encoding: HarmonyEncoding, tool_calls_enabled: bool) -> Result<Self> {
        Ok(Self {
            parser: StreamableParser::new(encoding, Some(Role::Assistant))
                .map_err(harmony_output_parsing_error)?,
            tool_calls_enabled,
            completed_visible_messages: 0,
            completed_reasoning_messages: 0,
            current_text_group: None,
            open_tool_call: None,
        })
    }

    fn process_token_ids(&mut self, token_ids: &[u32]) -> Result<Vec<AssistantEvent>> {
        let mut events = Vec::new();
        let mut pending_group: Option<HarmonyGroup> = None;

        for &token_id in token_ids {
            let completed_before = self.parser.messages().len();
            self.parser
                .process(token_id)
                .map_err(harmony_output_parsing_error)?;
            let completed_after = self.parser.messages().len();

            if let Some(delta) = self
                .parser
                .last_content_delta()
                .map_err(harmony_output_parsing_error)?
                .filter(|delta| !delta.is_empty())
            {
                let key = HarmonyGroupKey {
                    serial: completed_after,
                    channel: self.parser.current_channel(),
                    recipient: self.parser.current_recipient(),
                };

                match pending_group.as_mut() {
                    Some(group) if group.key == key => group.text.push_str(&delta),
                    _ => {
                        if let Some(group) = pending_group.take() {
                            self.emit_group(group, &mut events);
                        }
                        pending_group = Some(HarmonyGroup { key, text: delta });
                    }
                }
            }

            if completed_after > completed_before {
                if let Some(group) = pending_group.take() {
                    self.emit_group(group, &mut events);
                }

                for serial in completed_before..completed_after {
                    let key = {
                        let message = &self.parser.messages()[serial];
                        HarmonyGroupKey {
                            serial,
                            channel: message.channel.clone(),
                            recipient: message.recipient.clone(),
                        }
                    };
                    self.handle_completed_message(key);
                }
            }
        }

        if let Some(group) = pending_group {
            self.emit_group(group, &mut events);
        }

        Ok(events)
    }

    /// Flush Harmony parser state at EOS and emit any newly finalized assistant events.
    fn process_eos(&mut self) -> Result<Vec<AssistantEvent>> {
        let completed_before = self.parser.messages().len();
        let pending_key = HarmonyGroupKey {
            serial: completed_before,
            channel: self.parser.current_channel(),
            recipient: self.parser.current_recipient(),
        };
        let pending_content = self
            .parser
            .current_content()
            .map_err(harmony_output_parsing_error)?;

        self.parser
            .process_eos()
            .map_err(harmony_output_parsing_error)?;

        let completed_after = self.parser.messages().len();
        let mut events = Vec::new();

        if completed_after == completed_before {
            return Ok(events);
        }

        let final_message = &self.parser.messages()[completed_before];
        let final_text = harmony_message_text(final_message);
        let tail = final_text
            .strip_prefix(&pending_content)
            .unwrap_or(final_text)
            .to_string();
        if !tail.is_empty() {
            self.emit_group(
                HarmonyGroup {
                    key: pending_key,
                    text: tail,
                },
                &mut events,
            );
        }

        for serial in completed_before..completed_after {
            let key = {
                let message = &self.parser.messages()[serial];
                HarmonyGroupKey {
                    serial,
                    channel: message.channel.clone(),
                    recipient: message.recipient.clone(),
                }
            };
            self.handle_completed_message(key);
        }

        Ok(events)
    }

    /// Flush one coalesced Harmony content group into internal assistant events.
    fn emit_group(&mut self, group: HarmonyGroup, events: &mut Vec<AssistantEvent>) {
        let channel = group.key.channel.as_deref();
        let recipient = group.key.recipient.as_deref();

        if let Some(kind) = text_block_kind(channel, recipient) {
            self.open_tool_call = None;

            if self.current_text_group.as_ref() != Some(&group.key) {
                let needs_newline = match kind {
                    AssistantBlockKind::Text => self.completed_visible_messages > 0,
                    AssistantBlockKind::Reasoning => self.completed_reasoning_messages > 0,
                    AssistantBlockKind::ToolCall => false,
                };

                if needs_newline {
                    events.push(AssistantEvent::TextDelta {
                        kind,
                        delta: "\n".to_string(),
                    });
                }

                self.current_text_group = Some(group.key.clone());
            }

            events.push(AssistantEvent::TextDelta {
                kind,
                delta: group.text,
            });
            return;
        }

        self.current_text_group = None;

        let Some(tool_name) = tool_name(channel, recipient) else {
            return;
        };
        if !self.tool_calls_enabled {
            return;
        }

        let recipient = recipient
            .expect("tool groups always have recipient")
            .to_string();
        let opens_same_call = match self.open_tool_call.as_ref() {
            Some(open_call) => open_call.recipient == recipient,
            None => false,
        };
        if !opens_same_call {
            let id = generate_tool_call_id();
            self.open_tool_call = Some(OpenHarmonyToolCall { recipient });
            events.push(AssistantEvent::ToolCallStart {
                id,
                name: tool_name.to_string(),
            });
        }

        if !group.text.is_empty() {
            events.push(AssistantEvent::ToolCallArgumentsDelta { delta: group.text });
        }
    }

    /// Update newline and open-tool state after one Harmony message completes.
    fn handle_completed_message(&mut self, key: HarmonyGroupKey) {
        if self.current_text_group.as_ref() == Some(&key) {
            self.current_text_group = None;
        }

        let channel = key.channel.as_deref();
        let recipient = key.recipient.as_deref();
        let kind = text_block_kind(channel, recipient);

        if kind == Some(AssistantBlockKind::Text) {
            self.completed_visible_messages += 1;
        } else if kind == Some(AssistantBlockKind::Reasoning) {
            self.completed_reasoning_messages += 1;
        } else if tool_name(channel, recipient).is_some() {
            self.open_tool_call = None;
        }
    }
}

/// Convert decoded token updates into internal assistant events with Harmony parsing.
#[try_stream(ok = AssistantEvent, error = Error)]
async fn harmony_assistant_event_stream(
    decoded: DynDecodedTextEventStream,
    encoding: &'static HarmonyEncoding,
    tool_calls_enabled: bool,
) {
    let mut state = HarmonyState::new(encoding.clone(), tool_calls_enabled)?;
    futures::pin_mut!(decoded);

    while let Some(event) = decoded.next().await.transpose()? {
        match event {
            DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => {
                yield AssistantEvent::Start {
                    prompt_token_ids,
                    prompt_logprobs,
                };
            }
            DecodedTextEvent::TextDelta {
                delta: _, // harmony takes raw token IDs as input, so we ignore text deltas here
                token_ids,
                logprobs,
                finished,
            } => {
                for event in state.process_token_ids(&token_ids)? {
                    yield event;
                }

                if finished.is_some() {
                    for event in state.process_eos()? {
                        yield event;
                    }
                }

                if logprobs.is_some() || !token_ids.is_empty() {
                    yield AssistantEvent::LogprobsDelta {
                        logprobs,
                        token_ids,
                    };
                }

                if let Some(finished) = finished {
                    yield AssistantEvent::Done {
                        prompt_token_count: finished.prompt_token_count,
                        output_token_count: finished.output_token_count,
                        finish_reason: finished.finish_reason,
                        kv_transfer_params: finished.kv_transfer_params,
                    };
                }
            }
        }
    }
}

/// Lazily load the shared GPT-OSS Harmony encoding once per process.
fn harmony_encoding() -> Result<&'static HarmonyEncoding> {
    static ENCODING: LazyLock<anyhow::Result<HarmonyEncoding>> = LazyLock::new(|| {
        load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            .context("failed to load harmony encoding for gpt-oss")
    });

    ENCODING
        .as_ref()
        .map_err(|error| Error::HarmonyOutputParsing {
            error: error.to_report_string().into(),
        })
}

fn harmony_output_parsing_error(
    error: impl Into<Box<dyn std::error::Error + Send + Sync>>,
) -> Error {
    Error::HarmonyOutputParsing {
        error: error.into(),
    }
}

/// Return the decoded text payload from one parsed Harmony message.
fn harmony_message_text(message: &HarmonyMessage) -> &str {
    let [HarmonyContent::Text(text)] = message.content.as_slice() else {
        unreachable!("Harmony parser emits one text content block per parsed message")
    };
    &text.text
}

/// Map one Harmony `(channel, recipient)` pair to a visible assistant block kind.
fn text_block_kind(channel: Option<&str>, recipient: Option<&str>) -> Option<AssistantBlockKind> {
    match (channel, recipient) {
        (Some("final"), _) => Some(AssistantBlockKind::Text),
        (Some("analysis"), None) => Some(AssistantBlockKind::Reasoning),
        (Some("commentary"), None) => Some(AssistantBlockKind::Text),
        _ => None,
    }
}

/// Extract the tool name from a Harmony tool-recipient field, if present.
fn tool_name<'a>(channel: Option<&str>, recipient: Option<&'a str>) -> Option<&'a str> {
    match (channel, recipient) {
        (Some("commentary" | "analysis"), Some(recipient)) => recipient.strip_prefix("functions."),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
