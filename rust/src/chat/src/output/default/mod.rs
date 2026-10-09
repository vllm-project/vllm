// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Default output processing pipeline.

mod unified;

use std::sync::{Arc, Once};

use futures::StreamExt as _;
use tracing::info;
use vllm_parser::output_grammar::{BuiltOutputGrammar, OutputGrammarContext};
use vllm_parser::unified::{CombinedParser, UnifiedParser};
use vllm_text::tokenizer::DynTokenizer;
use xgrammar_structural_tag::ToolChoice;
use xgrammar_structural_tag::format::Format;

use self::unified::unified_event_stream;
use super::structural_tag::answer_format;
use super::structured::structured_chat_event_stream;
use crate::error::Result;
use crate::output::{ChatOutputProcessor, DynChatEventStream, DynDecodedTextEventStream};
use crate::parser::reasoning::{ReasoningParser, ReasoningParserFactory};
use crate::parser::tool::{ToolParser, ToolParserFactory};
use crate::parser::unified::{
    HfTemplateError, HfUnifiedParser, ResponseTemplate, UnifiedParserFactory, names,
};
use crate::parser::{ParserSelection, ToolStrictLevel};
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
    /// Request facts the initialized parser needs to build its output grammar.
    /// Absent for the plain-text-only processor, which never builds one.
    grammar_inputs: Option<GrammarInputs>,
}

/// Request-scoped inputs for [`UnifiedParser::build_output_grammar`], captured
/// at construction because the chat request is consumed before the prompt is
/// tokenized.
struct GrammarInputs {
    tools: Vec<ChatTool>,
    tool_choice: ToolChoice,
    tool_strict_level: ToolStrictLevel,
    /// The request's structured-output constraint, normalized for composition
    /// into the output grammar. The original stays on the request until an
    /// output grammar replaces it.
    answer: Option<Format>,
}

impl DefaultChatOutputProcessor {
    /// Build the default output processor and apply any parser-specific request
    /// adjustments.
    ///
    /// Parser resolution happens here so that request validation, prompt
    /// rendering, and streaming all observe the same parser-adjusted
    /// request state. The parser is initialized and its output grammar built
    /// later, once the final prompt token IDs are known.
    pub fn new(
        request: &mut ChatRequest,
        model_id: &str,
        tokenizer: DynTokenizer,
        tool_call_parser: &ParserSelection,
        reasoning_parser: &ParserSelection,
        tool_strict_level: ToolStrictLevel,
    ) -> ChatResult<Self> {
        Self::with_response_template(
            request,
            model_id,
            tokenizer,
            &Err(HfTemplateError::Missing),
            tool_call_parser,
            reasoning_parser,
            tool_strict_level,
        )
    }

    /// Like [`Self::new`], additionally providing the model's `response_template`
    /// for the `hf` parser, or why it is unavailable.
    pub fn with_response_template(
        request: &mut ChatRequest,
        model_id: &str,
        tokenizer: DynTokenizer,
        response_template: &std::result::Result<Arc<ResponseTemplate>, HfTemplateError>,
        tool_call_parser: &ParserSelection,
        reasoning_parser: &ParserSelection,
        tool_strict_level: ToolStrictLevel,
    ) -> ChatResult<Self> {
        let tool_name = tool_call_parser.resolve_tool_name(model_id);
        let reasoning_name = reasoning_parser.resolve_reasoning_name(model_id);
        // `hf` is opt-in: `Auto` never selects it, but a usable template is worth a hint.
        let auto = [tool_call_parser, reasoning_parser]
            .iter()
            .any(|selection| matches!(selection, ParserSelection::Auto));
        if auto && tool_name.is_none() && reasoning_name.is_none() && response_template.is_ok() {
            RESPONSE_TEMPLATE_HINT_ONCE.call_once(|| {
                info!(
                    "the model ships a response_template; pass `--tool-call-parser hf \
                     --reasoning-parser hf` to parse with it"
                );
            });
        }
        let parser = if let Some(parser) = Self::resolve_optional_unified_parser(
            request.tools(),
            tokenizer.clone(),
            response_template,
            tool_name,
            reasoning_name,
        )? {
            parser
        } else {
            let tool_parsing_enabled = request.tool_parsing_enabled();
            let tool_parser = if tool_parsing_enabled {
                Some(Self::resolve_tool_parser(
                    request.tools(),
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

        if parser.preserve_special_tokens() {
            request.decode_options.skip_special_tokens = false;
        }

        Ok(Self {
            parser,
            parallel_tool_calls: request.parallel_tool_calls(),
            grammar_inputs: Some(GrammarInputs {
                tools: request.tools().to_vec(),
                tool_choice: request.tool_choice().into(),
                tool_strict_level,
                answer: request.sampling_params.structured_outputs.as_ref().and_then(answer_format),
            }),
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
            grammar_inputs: None,
        }
    }

    fn resolve_tool_parser(
        tools: &[ChatTool],
        model_id: &str,
        selection: &ParserSelection,
    ) -> ChatResult<Box<dyn ToolParser>> {
        let factory = ToolParserFactory::global();
        let parser_name = match selection {
            ParserSelection::Auto => selection.resolve_tool_name(model_id).ok_or_else(|| {
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
        tokenizer: DynTokenizer,
        response_template: &std::result::Result<Arc<ResponseTemplate>, HfTemplateError>,
        tool_name: Option<&str>,
        reasoning_name: Option<&str>,
    ) -> ChatResult<Option<Box<dyn UnifiedParser>>> {
        let factory = UnifiedParserFactory::global();
        let Some(parser_name) =
            tool_name.into_iter().chain(reasoning_name).find(|name| factory.contains(name))
        else {
            return Ok(None);
        };
        if tool_name != reasoning_name {
            return Err(Error::IncompatibleParserSelections {
                tool: tool_name.unwrap_or("none").to_owned(),
                reasoning: reasoning_name.unwrap_or("none").to_owned(),
            });
        }

        // `hf` is built from the model's template, never by its registry constructor.
        let parser = if parser_name == names::HF {
            let template =
                response_template.as_ref().map_err(|error| Error::ParserInitialization {
                    kind: "unified",
                    name: parser_name.to_string(),
                    error: Box::new(error.clone()),
                })?;
            Box::new(HfUnifiedParser::new(template.clone(), tools, tokenizer))
        } else {
            factory.create(parser_name, tools, tokenizer)?
        };

        UNIFIED_PARSER_LOG_ONCE.call_once(|| info!(parser_name, "using unified parser"));
        Ok(Some(parser))
    }

    fn resolve_optional_reasoning_parser(
        model_id: &str,
        tokenizer: DynTokenizer,
        selection: &ParserSelection,
    ) -> ChatResult<Option<Box<dyn ReasoningParser>>> {
        let factory = ReasoningParserFactory::global();
        let parser_name = selection.resolve_reasoning_name(model_id);

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
static RESPONSE_TEMPLATE_HINT_ONCE: Once = Once::new();

impl ChatOutputProcessor for DefaultChatOutputProcessor {
    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.parser.initialize(prompt_token_ids).map_err(|error| {
            Error::OutputParserInitialization {
                error: Box::new(error),
            }
        })
    }

    fn build_output_grammar(&self) -> Result<Option<BuiltOutputGrammar>> {
        let Some(inputs) = &self.grammar_inputs else {
            return Ok(None);
        };
        self.parser
            .build_output_grammar(&OutputGrammarContext {
                tools: &inputs.tools,
                tool_choice: &inputs.tool_choice,
                tool_strict_level: inputs.tool_strict_level,
                parallel_tool_calls: self.parallel_tool_calls,
                answer: inputs.answer.as_ref(),
            })
            .map_err(|error| Error::OutputGrammar {
                error: Box::new(error),
            })
    }

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

    use thiserror_ext::AsReport as _;
    use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputsParams;
    use vllm_tokenizer::test_utils::TestTokenizer;
    use xgrammar_structural_tag::format::Format;

    use super::DefaultChatOutputProcessor;
    use crate::output::ChatOutputProcessor;
    use crate::parser::unified::{HfTemplateError, ResponseTemplate};
    use crate::parser::{ParserSelection, ToolStrictLevel};
    use crate::request::{ChatRequest, ChatTool, ChatToolChoice, ResolvedToolContext};

    fn tokenizer() -> Arc<TestTokenizer> {
        Arc::new(
            TestTokenizer::new()
                .with_regular_token("<|channel>", 256)
                .with_regular_token("<channel|>", 257),
        )
    }

    #[test]
    fn output_grammar_preserves_tool_strict_level() {
        let build = |level, strict, choice| {
            let tools = vec![ChatTool {
                name: "search".to_string(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "query": { "type": "string" } },
                    "required": ["query"]
                }),
                strict,
                defer_loading: None,
            }];
            let mut request = ChatRequest {
                tool_context: ResolvedToolContext::new(&[], tools, Some(choice), true).unwrap(),
                ..ChatRequest::for_test()
            };
            let mut processor = DefaultChatOutputProcessor::new(
                &mut request,
                "other-model",
                tokenizer(),
                &ParserSelection::Explicit("qwen3_coder".to_string()),
                &ParserSelection::None,
                level,
            )
            .unwrap();
            processor.initialize(&[]).unwrap();
            processor.build_output_grammar().unwrap()
        };

        assert!(build(ToolStrictLevel::Auto, None, ChatToolChoice::Auto).is_none());
        let envelope = build(ToolStrictLevel::Function, None, ChatToolChoice::Auto).unwrap();
        let parameters = build(ToolStrictLevel::Parameter, None, ChatToolChoice::Auto).unwrap();
        let strict = build(ToolStrictLevel::Auto, Some(true), ChatToolChoice::Auto).unwrap();
        assert_eq!(parameters, strict);
        assert_ne!(envelope, parameters);
        assert!(build(ToolStrictLevel::Parameter, None, ChatToolChoice::None).is_none());
    }

    #[test]
    fn answer_constraint_keeps_optional_tool_calls() {
        let tools = vec![ChatTool {
            name: "search".to_string(),
            description: None,
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
            defer_loading: None,
        }];
        let schema = serde_json::json!({"type": "object"});
        let build = |structured_outputs| {
            let mut request = ChatRequest {
                tool_context: ResolvedToolContext::new(
                    &[],
                    tools.clone(),
                    Some(ChatToolChoice::Auto),
                    true,
                )
                .unwrap(),
                ..ChatRequest::for_test()
            };
            request.sampling_params.structured_outputs = structured_outputs;
            let mut processor = DefaultChatOutputProcessor::new(
                &mut request,
                "other-model",
                tokenizer(),
                &ParserSelection::Explicit("qwen3_coder".to_string()),
                &ParserSelection::None,
                ToolStrictLevel::Auto,
            )
            .unwrap();
            processor.initialize(&[]).unwrap();
            processor.build_output_grammar().unwrap()
        };

        // Non-strict `auto` alone needs no grammar; with an answer constraint
        // the grammar holds the answer or a call.
        assert!(build(None).is_none());
        let built = build(Some(StructuredOutputsParams::json(schema.clone()))).unwrap();
        let Format::Or(branches) = built.format else {
            panic!("expected the calls or the answer, got {:?}", built.format);
        };
        assert_eq!(branches.elements.last(), Some(&Format::json_schema(schema)));
    }

    #[test]
    fn tool_grammar_honors_parallel_call_policy() {
        for parallel in [false, true] {
            let tools = vec![ChatTool {
                name: "lookup".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: Some(true),
                defer_loading: None,
            }];
            let mut request = ChatRequest {
                tool_context: ResolvedToolContext::new(
                    &[],
                    tools,
                    Some(ChatToolChoice::Auto),
                    parallel,
                )
                .unwrap(),
                ..ChatRequest::for_test()
            };
            let mut processor = DefaultChatOutputProcessor::new(
                &mut request,
                "other-model",
                tokenizer(),
                &ParserSelection::Explicit("qwen3_coder".to_string()),
                &ParserSelection::None,
                ToolStrictLevel::Auto,
            )
            .unwrap();
            processor.initialize(&[]).unwrap();
            let built = processor.build_output_grammar().unwrap().unwrap();
            let tag =
                serde_json::to_value(xgrammar_structural_tag::StructuralTag::new(built.format))
                    .unwrap();
            assert_eq!(tag["format"]["type"], "triggered_tags");
            assert_eq!(tag["format"]["stop_after_first"], !parallel);
        }
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
            ToolStrictLevel::Auto,
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
            ToolStrictLevel::Auto,
        )
        .unwrap();
    }

    #[test]
    fn auto_and_explicit_gemma4_selections_use_unified_parser() {
        let explicit = ParserSelection::Explicit("gemma4".to_string());
        for (tool, reasoning) in [
            (&ParserSelection::Auto, &explicit),
            (&explicit, &ParserSelection::Auto),
        ] {
            DefaultChatOutputProcessor::new(
                &mut ChatRequest::for_test(),
                "google/gemma-4-27b-it",
                tokenizer(),
                tool,
                reasoning,
                ToolStrictLevel::Auto,
            )
            .unwrap();
        }
    }

    #[test]
    fn conflicting_unified_parser_selections_report_resolved_names() {
        let mut request = ChatRequest::for_test();
        let error = match DefaultChatOutputProcessor::new(
            &mut request,
            "Qwen/Qwen3-8B",
            tokenizer(),
            &ParserSelection::Auto,
            &ParserSelection::Explicit("gemma4".to_string()),
            ToolStrictLevel::Auto,
        ) {
            Ok(_) => panic!("expected mixed Gemma4 parser selection to fail"),
            Err(error) => error,
        };

        expect_test::expect!["unified parsing requires the tool and reasoning selections to resolve to the same parser; resolved tool=qwen3_xml, reasoning=gemma4"]
            .assert_eq(&format!("{error}"));
    }

    type TemplateResult = Result<Arc<ResponseTemplate>, HfTemplateError>;

    fn loaded_template() -> TemplateResult {
        Ok(Arc::new(
            ResponseTemplate::from_json(&serde_json::json!({
                "start_anchor": "<|turn>model\n",
                "fields": {
                    "thinking": {"open": "<|channel>thought\n", "close": "<channel|>"},
                    "content": {"close": "<turn|>"},
                },
            }))
            .unwrap(),
        ))
    }

    fn invalid_template() -> TemplateResult {
        let error = ResponseTemplate::from_json(&serde_json::json!({"fields": {"content": {}}}))
            .unwrap_err();
        Err(error)
    }

    /// Build a processor for `model_id`, returning whether it keeps special
    /// tokens (only parsers that need them, such as `hf`, turn this on).
    fn keeps_special_tokens(
        model_id: &str,
        template: &TemplateResult,
        tool: &ParserSelection,
        reasoning: &ParserSelection,
    ) -> crate::Result<bool> {
        let mut request = ChatRequest::for_test();
        DefaultChatOutputProcessor::with_response_template(
            &mut request,
            model_id,
            tokenizer(),
            template,
            tool,
            reasoning,
            ToolStrictLevel::Auto,
        )?;
        Ok(!request.decode_options.skip_special_tokens)
    }

    #[test]
    fn explicit_hf_uses_the_model_response_template() {
        let hf = ParserSelection::Explicit("hf".to_string());
        assert!(keeps_special_tokens("other-model", &loaded_template(), &hf, &hf).unwrap());
    }

    #[test]
    fn explicit_hf_without_usable_template_fails() {
        let hf = ParserSelection::Explicit("hf".to_string());
        let error = |template| {
            keeps_special_tokens("other-model", &template, &hf, &hf)
                .unwrap_err()
                .to_report_string()
        };
        expect_test::expect!["failed to initialize unified parser `hf`: the model's tokenizer_config.json provides no response_template"]
            .assert_eq(&error(Err(HfTemplateError::Missing)));
        expect_test::expect!["failed to initialize unified parser `hf`: invalid response_template: response_template must define 'start_anchor' or 'start_anchor_pattern'."]
            .assert_eq(&error(invalid_template()));
    }

    #[test]
    fn auto_never_selects_hf() {
        let auto = ParserSelection::Auto;
        // Without a matching parser, parsing stays disabled even with a usable template.
        assert!(!keeps_special_tokens("other-model", &loaded_template(), &auto, &auto).unwrap());
        // An unusable template is not consulted unless `hf` is selected.
        assert!(!keeps_special_tokens("other-model", &invalid_template(), &auto, &auto).unwrap());
        keeps_special_tokens("google/gemma-4-27b-it", &invalid_template(), &auto, &auto).unwrap();
    }

    #[test]
    fn hf_requires_matching_parser_selections() {
        let error = keeps_special_tokens(
            "other-model",
            &loaded_template(),
            &ParserSelection::Explicit("hf".to_string()),
            &ParserSelection::None,
        )
        .unwrap_err();

        expect_test::expect!["unified parsing requires the tool and reasoning selections to resolve to the same parser; resolved tool=hf, reasoning=none"]
            .assert_eq(&format!("{error}"));
    }
}
