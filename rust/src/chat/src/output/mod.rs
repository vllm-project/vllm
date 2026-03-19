//! Output processing pipeline.

mod decoded;
mod incremental;
mod reasoning;
mod structured;
mod tool;

use futures::Stream;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use subenum::subenum;
use tool_parser::ParserFactory as ToolParserFactory;
use vllm_engine_core_client::protocol::{FinishReason, StopReason};
use vllm_llm::GenerateOutputStream;

use self::decoded::{DecodedTextEvent, decoded_text_event_stream};
use self::reasoning::reasoning_event_stream;
use self::tool::tool_event_stream;
use crate::ChatRequest;
use crate::backend::DynChatBackend;
use crate::error::{Error, Result};
use crate::event::{AssistantBlockKind, AssistantToolCall, ChatEvent};
use crate::output::structured::structured_chat_event_stream;

/// Internal assistant event before final assembly.
///
/// - [`ContentEvent`]: subenum after reasoning parsing, carries only text content.
/// - [`AssistantEvent`]: full event after tool parsing, adds tool-call variants.
#[subenum(ContentEvent)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AssistantEvent {
    #[subenum(ContentEvent)]
    Start,
    #[subenum(ContentEvent)]
    TextDelta {
        kind: AssistantBlockKind,
        delta: String,
    },
    ToolCallStart {
        id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        id: String,
        delta: String,
    },
    ToolCallEnd {
        call: AssistantToolCall,
    },
    #[subenum(ContentEvent)]
    Done {
        prompt_token_count: u32,
        token_ids: Vec<u32>,
        finish_reason: Option<FinishReason>,
        stop_reason: Option<StopReason>,
    },
}

impl ContentEvent {
    /// Convert a [`DecodedTextEvent`] into a [`ContentEvent`] by treating all text as plain
    /// (non-reasoning) content and discarding cumulative fields.
    fn from_decoded_plain_text(event: DecodedTextEvent) -> Self {
        match event {
            DecodedTextEvent::Start => Self::Start,
            DecodedTextEvent::TextDelta { delta, .. } => Self::TextDelta {
                kind: AssistantBlockKind::Text,
                delta,
            },
            DecodedTextEvent::Done {
                prompt_token_count,
                token_ids,
                finish_reason,
                stop_reason,
                ..
            } => Self::Done {
                prompt_token_count,
                token_ids,
                finish_reason,
                stop_reason,
            },
        }
    }
}

trait DecodedTextEventStream = Stream<Item = Result<DecodedTextEvent>> + Send + 'static;
trait ContentEventStream = Stream<Item = Result<ContentEvent>> + Send + 'static;
trait AssistantEventStream = Stream<Item = Result<AssistantEvent>> + Send + 'static;
pub(crate) trait ChatEventStream = Stream<Item = Result<ChatEvent>> + Send + 'static;

/// Transforms a raw generate-output token stream into structured chat events
/// through four sequential stages:
///
/// 1. [`decoded_text_event_stream`] — token-to-text decoding
/// 2. [`reasoning_event_stream`] — reasoning/content separation
/// 3. [`tool_event_stream`] — tool-call parsing
/// 4. [`structured_chat_event_stream`] — final block assembly
pub(crate) fn output_stream(
    request: ChatRequest,
    backend: DynChatBackend,
    raw_stream: GenerateOutputStream,
    model_id: Option<&str>,
    reasoning_parser_factory: &ReasoningParserFactory,
    tool_parser_factory: &ToolParserFactory,
    reasoning_parser_name: Option<&str>,
    tool_call_parser_name: Option<&str>,
) -> Result<impl ChatEventStream> {
    let tool_parser = if request.tool_parsing_enabled() {
        if let Some(name) = tool_call_parser_name {
            // Explicit parser name takes precedence.
            tool_parser_factory
                .registry()
                .create_parser(name)
                .ok_or_else(|| Error::ToolParserUnavailableByName {
                    name: name.to_string(),
                })?
                .into()
        } else if let Some(model_id) = model_id {
            tool_parser_factory
                .registry()
                .create_for_model(model_id)
                .ok_or_else(|| Error::ToolParserUnavailableForModel {
                    model_id: model_id.to_string(),
                })?
                .into()
        } else {
            return Err(Error::ToolParserRequiresModelId);
        }
    } else {
        None
    };
    let reasoning_parser = if let Some(name) = reasoning_parser_name {
        // Explicit parser name takes precedence.
        Some(
            reasoning_parser_factory
                .registry()
                .create_parser(name)
                .ok_or_else(|| Error::ReasoningParserUnavailableByName {
                    name: name.to_string(),
                })?,
        )
    } else {
        model_id.and_then(|model_id| {
            reasoning_parser_factory
                .registry()
                .create_for_model(model_id)
        })
    };

    // Chain the streams together.
    let decoded = decoded_text_event_stream(request.clone(), backend, raw_stream);
    let reasoning = reasoning_event_stream(decoded, reasoning_parser);
    let tool = tool_event_stream(reasoning, request, tool_parser);
    Ok(structured_chat_event_stream(tool))
}
