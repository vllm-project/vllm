//! Output processing pipeline.

mod reasoning;
mod structured;
mod tool;

use std::sync::Arc;

use futures::Stream;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use subenum::subenum;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::info;
use vllm_text::output::{DecodedLogprobs, DecodedPromptLogprobs, DecodedTextEvent};

use self::reasoning::reasoning_event_stream;
use self::tool::tool_event_stream;
use crate::FinishReason;
use crate::error::{Error, Result};
use crate::event::{AssistantBlockKind, AssistantToolCall, ChatEvent};
use crate::output::structured::structured_chat_event_stream;
use crate::request::{ChatTool, ChatToolChoice};

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

/// Transforms a raw generate-output token stream into structured chat events
/// through three sequential stages once text decoding has already happened:
///
/// 1. [`reasoning_event_stream`] — reasoning/content separation
/// 2. [`tool_event_stream`] — tool-call parsing
/// 3. [`structured_chat_event_stream`] — final block assembly
pub(crate) fn output_stream(
    intermediate: bool,
    tools: Vec<ChatTool>,
    tool_choice: ChatToolChoice,
    decoded: impl DecodedTextEventStream,
    model_id: Option<&str>,
    reasoning_parser_factory: &ReasoningParserFactory,
    tool_parser_factory: &ToolParserFactory,
    reasoning_parser_name: Option<&str>,
    tool_call_parser_name: Option<&str>,
) -> Result<impl ChatEventStream> {
    let tool_parsing_enabled = matches!(tool_choice, ChatToolChoice::Auto) && !tools.is_empty();
    let (tool_parser, parser_tools) = if tool_parsing_enabled {
        let parser = if let Some(name) = tool_call_parser_name {
            // Explicit parser name takes precedence.
            tool_parser_factory
                .registry()
                .create_parser(name)
                .ok_or_else(|| Error::ToolParserUnavailableByName {
                    name: name.to_string(),
                    available_names: Vec::new(),
                })?
        } else if let Some(model_id) = model_id {
            tool_parser_factory
                .registry()
                .create_for_model(model_id)
                .ok_or_else(|| Error::ToolParserUnavailableForModel {
                    model_id: model_id.to_string(),
                })?
        } else {
            return Err(Error::ToolParserRequiresModelId);
        };
        let parser_tools = tools.iter().map(ChatTool::to_openai_tool).collect();
        (Some(parser), parser_tools)
    } else {
        (None, Vec::new())
    };
    let reasoning_parser = if let Some(name) = reasoning_parser_name {
        // Explicit parser name takes precedence.
        Some(
            reasoning_parser_factory
                .registry()
                .create_parser(name)
                .ok_or_else(|| Error::ReasoningParserUnavailableByName {
                    name: name.to_string(),
                    available_names: Vec::new(),
                })?,
        )
    } else {
        model_id.and_then(|model_id| {
            reasoning_parser_factory
                .registry()
                .create_for_model(model_id)
        })
    };

    LOG_ONCE.call_once(|| {
        // TODO: tool-parser doesn't expose its model type
        if let Some(reasoning_parser) = &reasoning_parser {
            let model_type = reasoning_parser.model_type();
            info!(model_type, "using reasoning parser");
        }
    });

    let reasoning = reasoning_event_stream(decoded, reasoning_parser);
    let tool = tool_event_stream(reasoning, intermediate, parser_tools, tool_parser);
    Ok(structured_chat_event_stream(tool))
}

static LOG_ONCE: std::sync::Once = std::sync::Once::new();
