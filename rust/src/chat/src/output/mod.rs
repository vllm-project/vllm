//! Output processing pipeline.

mod reasoning;
mod structured;
mod tool;

use std::sync::Arc;

use futures::Stream;
use openai_protocol::common::Tool as OpenAiTool;
use subenum::subenum;
use tool_parser::ToolParser;
use vllm_text::output::{DecodedLogprobs, DecodedPromptLogprobs, DecodedTextEvent};

use self::reasoning::reasoning_event_stream;
use self::tool::tool_event_stream;
use crate::FinishReason;
use crate::error::Result;
use crate::event::{AssistantBlockKind, AssistantToolCall, ChatEvent};
use crate::output::structured::structured_chat_event_stream;
use crate::reasoning::ReasoningParser;

/// Internal assistant event before final assembly.
///
/// - [`ContentEvent`]: subenum after reasoning parsing, carries only text content.
/// - [`AssistantEvent`]: full event after tool parsing, adds tool-call variants.
#[subenum(ContentEvent)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AssistantEvent {
    #[subenum(ContentEvent)]
    Start {
        prompt_token_ids: Arc<[u32]>,
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    #[subenum(ContentEvent)]
    TextDelta {
        kind: AssistantBlockKind,
        delta: String,
    },
    /// Per-decoded-update sample metadata: logprobs and/or output token IDs.
    #[subenum(ContentEvent)]
    LogprobsDelta {
        logprobs: Option<DecodedLogprobs>,
        token_ids: Vec<u32>,
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
        prompt_token_count: usize,
        output_token_count: usize,
        finish_reason: FinishReason,
        /// Connector-specific KV transfer parameters for disaggregated serving.
        kv_transfer_params: Option<serde_json::Value>,
    },
}

impl ContentEvent {
    /// Convert a [`DecodedTextEvent`] into one or more [`ContentEvent`] values by treating all text
    /// as plain (non-reasoning) content.
    fn from_decoded_plain_text(event: DecodedTextEvent) -> Vec<Self> {
        match event {
            DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => vec![Self::Start {
                prompt_token_ids,
                prompt_logprobs,
            }],
            DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
                finished,
            } => {
                let mut events = Vec::new();
                if !delta.is_empty() {
                    events.push(Self::TextDelta {
                        kind: AssistantBlockKind::Text,
                        delta,
                    });
                }
                if logprobs.is_some() || !token_ids.is_empty() {
                    events.push(Self::LogprobsDelta {
                        logprobs,
                        token_ids,
                    });
                }
                if let Some(finished) = finished {
                    events.push(Self::Done {
                        prompt_token_count: finished.prompt_token_count,
                        output_token_count: finished.output_token_count,
                        finish_reason: finished.finish_reason,
                        kv_transfer_params: finished.kv_transfer_params,
                    });
                }
                events
            }
        }
    }
}

trait ContentEventStream = Stream<Item = Result<ContentEvent>> + Send + 'static;
trait AssistantEventStream = Stream<Item = Result<AssistantEvent>> + Send + 'static;
pub(crate) trait DecodedTextEventStream = Stream<Item = Result<DecodedTextEvent>> + Send + 'static;
pub(crate) trait ChatEventStream = Stream<Item = Result<ChatEvent>> + Send + 'static;

/// Request-scoped processors that adapt decoded text into structured chat events.
pub(crate) struct OutputProcessors {
    pub(crate) reasoning_parser: Option<Box<dyn ReasoningParser>>,
    pub(crate) parser_tools: Vec<OpenAiTool>,
    pub(crate) tool_parser: Option<Box<dyn ToolParser>>,
}

/// Transforms a raw generate-output token stream into structured chat events
/// through three sequential stages once text decoding has already happened:
///
/// 1. [`reasoning_event_stream`] — reasoning/content separation
/// 2. [`tool_event_stream`] — tool-call parsing
/// 3. [`structured_chat_event_stream`] — final block assembly
pub(crate) fn output_stream(
    intermediate: bool,
    decoded: impl DecodedTextEventStream,
    OutputProcessors {
        reasoning_parser,
        parser_tools,
        tool_parser,
    }: OutputProcessors,
) -> Result<impl ChatEventStream> {
    let reasoning = reasoning_event_stream(decoded, reasoning_parser);
    let tool = tool_event_stream(reasoning, intermediate, parser_tools, tool_parser);
    let structured = structured_chat_event_stream(tool);
    Ok(structured)
}
