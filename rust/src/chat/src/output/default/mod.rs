//! Default output processing pipeline.

mod reasoning;
mod tool;

use std::sync::Once;

use futures::{Stream, StreamExt as _};
use tracing::info;
use trait_set::trait_set;
use vllm_text::tokenizer::DynTokenizer;

use self::reasoning::reasoning_event_stream;
use self::tool::tool_event_stream;
use super::structured::structured_chat_event_stream;
use crate::error::Result;
use crate::output::{
    AssistantEvent, ChatOutputProcessor, ContentEvent, DynChatEventStream,
    DynDecodedTextEventStream,
};
use crate::parser::ParserSelection;
use crate::parser::reasoning::{ReasoningParser, ReasoningParserFactory};
use crate::parser::tool::{ToolParser, ToolParserFactory};
use crate::request::{ChatRequest, ChatToolChoice};
use crate::{Error, Result as ChatResult};

trait_set! {
    trait ContentEventStream = Stream<Item = Result<ContentEvent>> + Send + 'static;
}

/// Default request-scoped output processor used by Hugging Face style chat
/// backends.
///
/// This implementation assumes the backend already emitted decoded text deltas,
/// then optionally layers reasoning parsing and tool-call parsing before
/// assembling final structured chat events.
pub struct DefaultChatOutputProcessor {
    reasoning_parser: Option<Box<dyn ReasoningParser>>,
    tool_parser: Option<Box<dyn ToolParser>>,
}

impl DefaultChatOutputProcessor {
    /// Build the default output processor and apply any parser-specific request
    /// adjustments.
    ///
    /// Parser resolution happens here so that request validation, prompt
    /// rendering, and streaming all observe the same parser-adjusted
    /// request state.
    pub fn new(
        request: &mut ChatRequest,
        model_id: &str,
        tokenizer: DynTokenizer,
        tool_call_parser: &ParserSelection,
        reasoning_parser: &ParserSelection,
    ) -> ChatResult<Self> {
        let tool_parsing_enabled =
            matches!(request.tool_choice, ChatToolChoice::Auto) && !request.tools.is_empty();
        let tool_parser = if tool_parsing_enabled {
            Some(Self::resolve_tool_parser(
                request,
                model_id,
                tool_call_parser,
            )?)
        } else {
            None
        };
        let reasoning_parser = Self::resolve_optional_reasoning_parser(
            request,
            model_id,
            tokenizer,
            reasoning_parser,
        )?;

        Ok(Self {
            reasoning_parser,
            tool_parser,
        })
    }

    /// Build the plain-text-only default output processor.
    ///
    /// This keeps the default structured chat-event assembly but disables both
    /// reasoning parsing and tool-call parsing completely, so that all
    /// content is treated as opaque text.
    pub fn plain_text_only() -> Self {
        Self {
            reasoning_parser: None,
            tool_parser: None,
        }
    }

    fn resolve_tool_parser(
        request: &mut ChatRequest,
        model_id: &str,
        selection: &ParserSelection,
    ) -> ChatResult<Box<dyn ToolParser>> {
        let factory = ToolParserFactory::global();
        let parser_name = match selection {
            ParserSelection::Auto => factory.resolve_name_for_model(model_id).ok_or_else(|| {
                Error::ParserUnavailableForModel {
                    kind: "tool",
                    model_id: model_id.to_string(),
                }
            })?,
            ParserSelection::None => return Err(Error::ParserDisabled { kind: "tool" }),
            ParserSelection::Explicit(name) => name.as_str(),
        };

        let parser = factory.create(parser_name, &request.tools)?;

        if parser.preserve_special_tokens() {
            request.decode_options.skip_special_tokens = false;
        }

        TOOL_PARSER_LOG_ONCE.call_once(|| info!(parser_name, "using tool parser"));
        Ok(parser)
    }

    fn resolve_optional_reasoning_parser(
        request: &mut ChatRequest,
        model_id: &str,
        tokenizer: DynTokenizer,
        selection: &ParserSelection,
    ) -> ChatResult<Option<Box<dyn ReasoningParser>>> {
        let factory = ReasoningParserFactory::global();
        let parser_name = match selection {
            ParserSelection::Auto => factory.resolve_name_for_model(model_id),
            ParserSelection::None => None,
            ParserSelection::Explicit(name) => Some(name.as_str()),
        };

        let Some(parser_name) = parser_name else {
            REASONING_PARSER_LOG_ONCE.call_once(|| info!("reasoning parsing disabled"));
            return Ok(None);
        };

        let parser = factory.create(parser_name, tokenizer)?;

        if parser.preserve_special_tokens() {
            request.decode_options.skip_special_tokens = false;
        }

        REASONING_PARSER_LOG_ONCE.call_once(|| info!(parser_name, "using reasoning parser"));
        Ok(Some(parser))
    }
}

static TOOL_PARSER_LOG_ONCE: Once = Once::new();
static REASONING_PARSER_LOG_ONCE: Once = Once::new();

impl ChatOutputProcessor for DefaultChatOutputProcessor {
    /// Transforms a raw generate-output token stream into structured chat
    /// events through three sequential stages once text decoding has
    /// already happened:
    ///
    /// 1. [`reasoning_event_stream`] — reasoning/content separation
    /// 2. [`tool_event_stream`] — tool-call parsing
    /// 3. [`structured_chat_event_stream`] — final block assembly
    fn process(self: Box<Self>, decoded: DynDecodedTextEventStream) -> Result<DynChatEventStream> {
        let reasoning = reasoning_event_stream(decoded, self.reasoning_parser);
        let tool = tool_event_stream(reasoning, self.tool_parser);
        let structured = structured_chat_event_stream(tool);

        Ok(structured.boxed())
    }
}
