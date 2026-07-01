//! Default output processing pipeline.

mod structural_tag;
mod unified;

use std::sync::Once;

use futures::StreamExt as _;
use tracing::info;
use vllm_parser::unified::{CombinedParser, UnifiedParser};
use vllm_text::tokenizer::DynTokenizer;

use self::structural_tag::apply_structural_tag_constraint;
use self::unified::unified_event_stream;
use super::structured::structured_chat_event_stream;
use crate::error::Result;
use crate::output::{ChatOutputProcessor, DynChatEventStream, DynDecodedTextEventStream};
use crate::parser::ParserSelection;
use crate::parser::reasoning::{ReasoningParser, ReasoningParserFactory};
use crate::parser::tool::{ToolParser, ToolParserFactory};
use crate::parser::unified::UnifiedParserFactory;
use crate::request::{ChatRequest, ChatTool};
use crate::{Error, Result as ChatResult};

/// Default request-scoped output processor used by Hugging Face style chat
/// backends.
///
/// This implementation assumes the backend already emitted decoded text deltas,
/// then optionally layers unified reasoning and tool-call parsing before
/// assembling final structured chat events.
pub struct DefaultChatOutputProcessor {
    parser: Box<dyn UnifiedParser>,
    parallel_tool_calls: bool,
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
        let parser = if tool_call_parser == reasoning_parser
            && let Some(parser) = Self::resolve_optional_unified_parser(
                &request.tools,
                model_id,
                tokenizer.clone(),
                tool_call_parser,
            )? {
            parser
        } else {
            let tool_parsing_enabled = request.tool_parsing_enabled();
            let tool_parser = if tool_parsing_enabled {
                Some(Self::resolve_tool_parser(
                    &request.tools,
                    model_id,
                    tool_call_parser,
                )?)
            } else {
                None
            };
            let reasoning_parser =
                Self::resolve_optional_reasoning_parser(model_id, tokenizer, reasoning_parser)?;
            Box::new(CombinedParser::new(reasoning_parser, tool_parser)) as Box<dyn UnifiedParser>
        };

        apply_structural_tag_constraint(request, parser.structural_tag_model())?;

        if parser.preserve_special_tokens() {
            request.decode_options.skip_special_tokens = false;
        }

        Ok(Self {
            parser,
            parallel_tool_calls: request.parallel_tool_calls,
        })
    }

    /// Build the plain-text-only default output processor.
    ///
    /// This keeps the default structured chat-event assembly but disables both
    /// reasoning parsing and tool-call parsing completely, so that all
    /// content is treated as opaque text.
    pub fn plain_text_only() -> Self {
        Self {
            parser: Box::new(CombinedParser::plain_text_only()),
            parallel_tool_calls: true,
        }
    }

    fn resolve_tool_parser(
        tools: &[ChatTool],
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

        let parser = factory.create(parser_name, tools)?;

        TOOL_PARSER_LOG_ONCE.call_once(|| info!(parser_name, "using tool parser"));
        Ok(parser)
    }

    fn resolve_optional_unified_parser(
        tools: &[ChatTool],
        model_id: &str,
        tokenizer: DynTokenizer,
        selection: &ParserSelection,
    ) -> ChatResult<Option<Box<dyn UnifiedParser>>> {
        let factory = UnifiedParserFactory::global();
        let parser_name = match selection {
            ParserSelection::Auto => factory.resolve_name_for_model(model_id),
            ParserSelection::None => None,
            ParserSelection::Explicit(name) if factory.contains(name) => Some(name.as_str()),
            ParserSelection::Explicit(_) => None,
        };

        let Some(parser_name) = parser_name else {
            return Ok(None);
        };

        let parser = factory.create(parser_name, tools, tokenizer)?;

        UNIFIED_PARSER_LOG_ONCE.call_once(|| info!(parser_name, "using unified parser"));
        Ok(Some(parser))
    }

    fn resolve_optional_reasoning_parser(
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

        REASONING_PARSER_LOG_ONCE.call_once(|| info!(parser_name, "using reasoning parser"));
        Ok(Some(parser))
    }
}

static TOOL_PARSER_LOG_ONCE: Once = Once::new();
static REASONING_PARSER_LOG_ONCE: Once = Once::new();
static UNIFIED_PARSER_LOG_ONCE: Once = Once::new();

impl ChatOutputProcessor for DefaultChatOutputProcessor {
    /// Transforms a raw generate-output token stream into structured chat
    /// events through two sequential stages once text decoding has
    /// already happened:
    ///
    /// 1. [`unified_event_stream`] — reasoning and tool-call parsing
    /// 2. [`structured_chat_event_stream`] — final block assembly
    fn process(self: Box<Self>, decoded: DynDecodedTextEventStream) -> Result<DynChatEventStream> {
        let parsed = unified_event_stream(decoded, self.parser);
        let structured = structured_chat_event_stream(parsed, self.parallel_tool_calls);

        Ok(structured.boxed())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_tokenizer::Tokenizer;

    use super::DefaultChatOutputProcessor;
    use crate::Error;
    use crate::parser::ParserSelection;
    use crate::request::ChatRequest;

    struct FakeTokenizer;

    impl Tokenizer for FakeTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> vllm_tokenizer::Result<Vec<u32>> {
            Ok(text.chars().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> vllm_tokenizer::Result<String> {
            Ok(token_ids
                .iter()
                .map(|token_id| char::from_u32(*token_id).unwrap_or('\u{FFFD}'))
                .collect())
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<|channel>" => Some(1),
                "<channel|>" => Some(2),
                _ => None,
            }
        }
    }

    fn tokenizer() -> Arc<FakeTokenizer> {
        Arc::new(FakeTokenizer)
    }

    #[test]
    fn equal_explicit_gemma4_uses_unified_parser() {
        let mut request = ChatRequest::for_test();
        let selection = ParserSelection::Explicit("gemma4".to_string());

        DefaultChatOutputProcessor::new(
            &mut request,
            "other-model",
            tokenizer(),
            &selection,
            &selection,
        )
        .unwrap();
    }

    #[test]
    fn auto_auto_gemma4_model_uses_unified_parser() {
        let mut request = ChatRequest::for_test();

        DefaultChatOutputProcessor::new(
            &mut request,
            "google/gemma-4-27b-it",
            tokenizer(),
            &ParserSelection::Auto,
            &ParserSelection::Auto,
        )
        .unwrap();
    }

    #[test]
    fn mixed_gemma4_selection_uses_split_dummy_error() {
        let mut request = ChatRequest::for_test();
        let error = match DefaultChatOutputProcessor::new(
            &mut request,
            "other-model",
            tokenizer(),
            &ParserSelection::Auto,
            &ParserSelection::Explicit("gemma4".to_string()),
        ) {
            Ok(_) => panic!("expected mixed Gemma4 parser selection to fail"),
            Err(error) => error,
        };

        let Error::ParserInitialization { error, .. } = error else {
            panic!("expected parser initialization error");
        };
        assert_eq!(
            error.to_string(),
            "`gemma4` only provides a unified parser; the same reasoning parser and tool parser should be specified together"
        );
    }
}
