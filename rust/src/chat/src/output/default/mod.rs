//! Default output processing pipeline.

mod reasoning;
mod tool;

use std::sync::Once;

use futures::{Stream, StreamExt as _};
use thiserror_ext::AsReport;
use tracing::info;
use trait_set::trait_set;
use vllm_engine_core_client::protocol::{StructuredOutputBackend, StructuredOutputsParams};
use vllm_text::tokenizer::DynTokenizer;
use xgrammar_structural_tag::{
    FunctionDefinition, FunctionToolParam, ToolChoice as StructuralTagToolChoice, ToolParam,
    build_optional_structural_tag,
};

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
        let tool_parsing_enabled = request.tool_parsing_enabled();
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
            reasoning_parser: None,
            tool_parser: None,
            parallel_tool_calls: true,
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

        apply_structural_tag_constraint(request, parser.as_ref())?;

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

fn apply_structural_tag_constraint(
    request: &mut ChatRequest,
    parser: &dyn ToolParser,
) -> ChatResult<()> {
    let Some(model) = parser.structural_tag_model() else {
        return Ok(());
    };
    let Some(tool_choice) = structural_tag_tool_choice(request) else {
        return Ok(());
    };

    let tools = request
        .tools
        .iter()
        .map(|tool| {
            ToolParam::Function(FunctionToolParam::new(FunctionDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: Some(tool.parameters.clone()),
                strict: tool.strict,
            }))
        })
        .collect::<Vec<_>>();

    let tag =
        build_optional_structural_tag(model, &tools, tool_choice, false).map_err(|error| {
            Error::StructuralTag {
                message: error.to_report_string(),
            }
        })?;
    let Some(tag) = tag else {
        return Ok(());
    };
    let structural_tag = tag.to_json_string().map_err(|error| Error::StructuralTag {
        message: error.to_report_string(),
    })?;

    request.sampling_params.structured_outputs = Some(StructuredOutputsParams {
        structural_tag: Some(structural_tag),
        backend: StructuredOutputBackend::Xgrammar,
        ..Default::default()
    });
    Ok(())
}

fn structural_tag_tool_choice(request: &ChatRequest) -> Option<StructuralTagToolChoice> {
    match &request.tool_choice {
        ChatToolChoice::Auto if request.tools.iter().any(|tool| tool.strict == Some(true)) => {
            Some(StructuralTagToolChoice::auto())
        }
        ChatToolChoice::Auto | ChatToolChoice::None => None,
        ChatToolChoice::Required => Some(StructuralTagToolChoice::required()),
        ChatToolChoice::Function { name } => Some(StructuralTagToolChoice::function(name.clone())),
    }
}

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
        let structured = structured_chat_event_stream(tool, self.parallel_tool_calls);

        Ok(structured.boxed())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use vllm_tool_parser::{
        Result as ToolParserResult, StructuralTagModel, Tool, ToolParserOutput,
    };

    use super::*;

    struct StructuralTagParser;

    impl ToolParser for StructuralTagParser {
        fn create(_tools: &[Tool]) -> ToolParserResult<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self))
        }

        fn structural_tag_model(&self) -> Option<StructuralTagModel> {
            Some(StructuralTagModel::Qwen3Coder)
        }

        fn parse_into(
            &mut self,
            _chunk: &str,
            _output: &mut ToolParserOutput,
        ) -> ToolParserResult<()> {
            Ok(())
        }

        fn finish(&mut self) -> ToolParserResult<ToolParserOutput> {
            Ok(ToolParserOutput::default())
        }

        fn reset(&mut self) -> String {
            String::new()
        }
    }

    fn chat_tool(name: &str, strict: Option<bool>) -> Tool {
        Tool {
            name: name.to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
            strict,
        }
    }

    fn request(tool_choice: ChatToolChoice, tools: Vec<Tool>) -> ChatRequest {
        ChatRequest {
            tool_choice,
            tools,
            ..ChatRequest::for_test()
        }
    }

    fn structural_tag_value(request: &ChatRequest) -> Value {
        let params = request
            .sampling_params
            .structured_outputs
            .as_ref()
            .expect("structured outputs should be set");
        assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
        serde_json::from_str(
            params.structural_tag.as_deref().expect("structural_tag should be set"),
        )
        .expect("structural_tag should be valid JSON")
    }

    #[test]
    fn auto_strict_tool_choice_builds_structural_tag() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", Some(true))]);

        apply_structural_tag_constraint(&mut request, &StructuralTagParser)
            .expect("structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        assert!(tag.to_string().contains("search"));
    }

    #[test]
    fn auto_non_strict_tool_choice_skips_structural_tag() {
        let mut request = request(ChatToolChoice::Auto, vec![chat_tool("search", None)]);

        apply_structural_tag_constraint(&mut request, &StructuralTagParser)
            .expect("structural tag decision should succeed");

        assert!(request.sampling_params.structured_outputs.is_none());
    }

    #[test]
    fn required_tool_choice_builds_structural_tag_without_strict_tools() {
        let mut request = request(ChatToolChoice::Required, vec![chat_tool("search", None)]);

        apply_structural_tag_constraint(&mut request, &StructuralTagParser)
            .expect("structural tag should build");

        let tag = structural_tag_value(&request);
        assert_eq!(tag["type"], "structural_tag");
        assert!(tag.to_string().contains("search"));
    }

    #[test]
    fn named_tool_choice_builds_structural_tag_for_named_tool_only() {
        let mut request = request(
            ChatToolChoice::Function {
                name: "lookup".to_string(),
            },
            vec![chat_tool("search", None), chat_tool("lookup", None)],
        );

        apply_structural_tag_constraint(&mut request, &StructuralTagParser)
            .expect("structural tag should build");

        let tag = structural_tag_value(&request).to_string();
        assert!(tag.contains("lookup"));
        assert!(!tag.contains("search"));
    }

    #[test]
    fn none_tool_choice_skips_structural_tag() {
        let mut request = request(ChatToolChoice::None, vec![chat_tool("search", Some(true))]);

        apply_structural_tag_constraint(&mut request, &StructuralTagParser)
            .expect("structural tag decision should succeed");

        assert!(request.sampling_params.structured_outputs.is_none());
    }
}
