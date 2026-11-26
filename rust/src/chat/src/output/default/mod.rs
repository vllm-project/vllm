// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Default output processing pipeline.

mod parsers;
mod unified;

use futures::StreamExt as _;
use vllm_parser::output_grammar::{BuiltOutputGrammar, OutputGrammarContext};
use vllm_parser::unified::{CombinedParser, UnifiedParser};
use xgrammar_structural_tag::ToolChoice;
use xgrammar_structural_tag::format::Format;

pub use self::parsers::ResolvedParsers;
use self::unified::unified_event_stream;
use super::structural_tag::answer_format;
use super::structured::structured_chat_event_stream;
use crate::error::Result;
use crate::output::{ChatOutputProcessor, DynChatEventStream, DynDecodedTextEventStream};
use crate::parser::ToolStrictLevel;
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
    max_tool_calls: Option<usize>,
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
    parallel_tool_calls: bool,
    /// The request's structured-output constraint, normalized for composition
    /// into the output grammar. The original stays on the request until an
    /// output grammar replaces it.
    answer: Option<Format>,
}

impl DefaultChatOutputProcessor {
    /// Build the default output processor and apply any parser-specific request
    /// adjustments.
    ///
    /// The request's parser is built here so that request validation, prompt
    /// rendering, and streaming all observe the same parser-adjusted
    /// request state. The parser is initialized and its output grammar built
    /// later, once the final prompt token IDs are known.
    pub fn new(
        request: &mut ChatRequest,
        parsers: &ResolvedParsers,
        tool_strict_level: Option<ToolStrictLevel>,
    ) -> ChatResult<Self> {
        let parser = parsers.create(request)?;

        if parser.preserve_special_tokens() {
            request.decode_options.skip_special_tokens = Some(false);
        }
        let tool_strict_level =
            tool_strict_level.unwrap_or_else(|| parser.default_tool_strict_level());

        Ok(Self {
            parser,
            max_tool_calls: request.max_tool_calls(),
            grammar_inputs: Some(GrammarInputs {
                tools: request.tools().to_vec(),
                tool_choice: request.tool_choice().into(),
                tool_strict_level,
                parallel_tool_calls: request.parallel_tool_calls(),
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
            max_tool_calls: None,
            grammar_inputs: None,
        }
    }
}

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
                parallel_tool_calls: inputs.parallel_tool_calls,
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
        let structured = structured_chat_event_stream(parsed, self.max_tool_calls);

        Ok(structured.boxed())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::{StreamExt as _, stream};
    use thiserror_ext::AsReport as _;
    use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputsParams;
    use vllm_parser::output_grammar::BuiltOutputGrammar;
    use vllm_text::output::{DecodedTextEvent, Finished};
    use vllm_text::tokenizer::DynTokenizer;
    use vllm_text::{DecodedText, SampledDelta};
    use vllm_tokenizer::test_utils::TestTokenizer;
    use xgrammar_structural_tag::format::Format;

    use super::{DefaultChatOutputProcessor, ResolvedParsers};
    use crate::event::{AssistantMessageExt as _, ChatEvent};
    use crate::output::ChatOutputProcessor;
    use crate::parser::unified::{HfTemplateError, ResponseTemplate};
    use crate::parser::{ParserSelection, ToolStrictLevel};
    use crate::request::{
        AllowedToolsMode, ChatRequest, ChatTool, ChatToolChoice, ResolvedToolContext,
    };

    /// Resolve the selections for `model_id` and build one request's processor.
    fn new_processor(
        request: &mut ChatRequest,
        model_id: &str,
        tokenizer: DynTokenizer,
        tool_call_parser: &ParserSelection,
        reasoning_parser: &ParserSelection,
        tool_strict_level: Option<ToolStrictLevel>,
    ) -> crate::Result<DefaultChatOutputProcessor> {
        new_processor_with_template(
            request,
            model_id,
            tokenizer,
            &Err(HfTemplateError::Missing),
            tool_call_parser,
            reasoning_parser,
            tool_strict_level,
        )
    }

    /// Like [`new_processor`], with the model's `response_template`.
    fn new_processor_with_template(
        request: &mut ChatRequest,
        model_id: &str,
        tokenizer: DynTokenizer,
        response_template: &TemplateResult,
        tool_call_parser: &ParserSelection,
        reasoning_parser: &ParserSelection,
        tool_strict_level: Option<ToolStrictLevel>,
    ) -> crate::Result<DefaultChatOutputProcessor> {
        let parsers = ResolvedParsers::with_response_template(
            model_id,
            tokenizer,
            response_template,
            tool_call_parser,
            reasoning_parser,
        )?;
        DefaultChatOutputProcessor::new(request, &parsers, tool_strict_level)
    }

    fn tokenizer() -> Arc<TestTokenizer> {
        Arc::new(
            TestTokenizer::new()
                .with_regular_token("<|channel>", 256)
                .with_regular_token("<channel|>", 257),
        )
    }

    /// The output grammar for one `search` tool with `strict` under `parser`.
    fn search_grammar(
        parser: &str,
        level: Option<ToolStrictLevel>,
        strict: Option<bool>,
        choice: ChatToolChoice,
    ) -> Option<BuiltOutputGrammar> {
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
        let mut processor = new_processor(
            &mut request,
            "other-model",
            tokenizer(),
            &ParserSelection::Explicit(parser.to_string()),
            &ParserSelection::None,
            level,
        )
        .unwrap();
        processor.initialize(&[]).unwrap();
        processor.build_output_grammar().unwrap()
    }

    #[test]
    fn output_grammar_preserves_tool_strict_level() {
        let build =
            |level, strict, choice| search_grammar("qwen3_coder", Some(level), strict, choice);

        assert!(build(ToolStrictLevel::Auto, None, ChatToolChoice::Auto).is_none());
        let envelope = build(ToolStrictLevel::Function, None, ChatToolChoice::Auto).unwrap();
        let parameters = build(ToolStrictLevel::Parameter, None, ChatToolChoice::Auto).unwrap();
        let strict = build(ToolStrictLevel::Auto, Some(true), ChatToolChoice::Auto).unwrap();
        assert_eq!(parameters, strict);
        assert_ne!(envelope, parameters);
        assert!(build(ToolStrictLevel::Parameter, None, ChatToolChoice::None).is_none());
    }

    #[tokio::test]
    async fn tool_choice_none_drops_parsed_tool_calls() {
        let tools = vec![ChatTool {
            name: "search".to_string(),
            description: None,
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
            defer_loading: None,
        }];
        let gemma4 = ParserSelection::Explicit("gemma4".to_string());
        let gemma4_text = "Sure.<|tool_call>call:search{query:<|\"|>vllm<|\"|>}<tool_call|>";
        let qwen3_coder = ParserSelection::Explicit("qwen3_coder".to_string());
        let qwen3_coder_text = "Sure.<tool_call>\n<function=search>\n<parameter=query>\nvllm\n\
                                </parameter>\n</function>\n</tool_call>";
        let run = async |tool_parser: &ParserSelection,
                         reasoning_parser: &ParserSelection,
                         text: &str,
                         tool_choice| {
            let mut request = ChatRequest {
                tool_context: ResolvedToolContext::new(&[], tools.clone(), Some(tool_choice), true)
                    .unwrap(),
                ..ChatRequest::for_test()
            };
            let mut processor = new_processor(
                &mut request,
                "other-model",
                tokenizer(),
                tool_parser,
                reasoning_parser,
                None,
            )
            .unwrap();
            processor.initialize(&[]).unwrap();
            let decoded = stream::iter([Ok(DecodedTextEvent::TextDelta {
                decoded: DecodedText::unattributed(text),
                sampled: SampledDelta::default(),
                finished: Some(Box::new(Finished {
                    usage: vllm_llm::TokenUsage::default(),
                    finish_reason: crate::FinishReason::Stop(None),
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                    spec_decode_metrics: None,
                    weight_versions: None,
                    routed_experts: None,
                    timestamps: Default::default(),
                })),
            })]);
            let events =
                Box::new(processor).process(decoded.boxed()).unwrap().collect::<Vec<_>>().await;
            let Some(Ok(ChatEvent::Done { message, .. })) = events.last() else {
                panic!("expected done");
            };
            let names = message.tool_calls().map(|call| call.name.clone()).collect::<Vec<_>>();
            (message.text(), names)
        };

        let none = ParserSelection::None;
        let llama3_json = ParserSelection::Explicit("llama3_json".to_string());
        let llama3_json_text = r#"{"name":"search","parameters":{"query":"vllm"}}"#;
        let json_answer = r#"{"answer":42}"#;
        let search = || vec!["search".to_string()];
        for (tool_parser, reasoning_parser, text, auto_content, none_content) in [
            // Python's parser engine parses a marked call and drops it: neither
            // a tool call nor its markup reaches the response, for a unified
            // parser and a split tool parser alike.
            (&gemma4, &gemma4, gemma4_text, "Sure.", "Sure."),
            (&qwen3_coder, &none, qwen3_coder_text, "Sure.", "Sure."),
            // Rust drops a bare JSON call too, where Python's Llama parser
            // keeps it as content.
            (&llama3_json, &none, llama3_json_text, "", ""),
        ] {
            assert_eq!(
                run(tool_parser, reasoning_parser, text, ChatToolChoice::Auto).await,
                (auto_content.to_string(), search())
            );
            assert_eq!(
                run(tool_parser, reasoning_parser, text, ChatToolChoice::None).await,
                (none_content.to_string(), vec![])
            );
            // An empty allowed-tools list allows no call, like `none`.
            let no_tools_allowed = ChatToolChoice::AllowedTools {
                mode: AllowedToolsMode::Auto,
                names: Vec::new(),
            };
            assert_eq!(
                run(tool_parser, reasoning_parser, text, no_tools_allowed).await,
                (none_content.to_string(), vec![])
            );
        }
        // JSON that is not a call stays content either way.
        for tool_choice in [ChatToolChoice::Auto, ChatToolChoice::None] {
            assert_eq!(
                run(&llama3_json, &none, json_answer, tool_choice).await,
                (json_answer.to_string(), vec![])
            );
        }
        // Without a configured tool parser, the markup stays content.
        assert_eq!(
            run(&none, &none, qwen3_coder_text, ChatToolChoice::None).await,
            (qwen3_coder_text.to_string(), vec![])
        );
    }

    #[test]
    fn unset_tool_strict_level_falls_back_to_the_parser_default() {
        let build = |parser, level| search_grammar(parser, level, None, ChatToolChoice::Auto);

        // GLM-4.7 constrains auto tool calls unless the level is set explicitly.
        let function = build("glm47", Some(ToolStrictLevel::Function));
        assert_eq!(build("glm47", None), function);
        assert!(function.is_some());
        assert!(build("glm47", Some(ToolStrictLevel::Auto)).is_none());
        // Other parsers default to `auto`.
        assert!(build("qwen3_coder", None).is_none());
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
        let build = |tool_choice, structured_outputs| {
            let mut request = ChatRequest {
                tool_context: ResolvedToolContext::new(&[], tools.clone(), Some(tool_choice), true)
                    .unwrap(),
                ..ChatRequest::for_test()
            };
            request.sampling_params.structured_outputs = structured_outputs;
            let mut processor = new_processor(
                &mut request,
                "other-model",
                tokenizer(),
                &ParserSelection::Explicit("qwen3_coder".to_string()),
                &ParserSelection::None,
                None,
            )
            .unwrap();
            processor.initialize(&[]).unwrap();
            processor.build_output_grammar().unwrap()
        };

        // Non-strict `auto` alone needs no grammar; with an answer constraint
        // the grammar holds the answer or a call.
        assert!(build(ChatToolChoice::Auto, None).is_none());
        let json = || Some(StructuredOutputsParams::json(schema.clone()));
        let built = build(ChatToolChoice::Auto, json()).unwrap();
        let Format::Or(branches) = built.format else {
            panic!("expected the calls or the answer, got {:?}", built.format);
        };
        assert_eq!(
            branches.elements.last(),
            Some(&Format::json_schema(schema.clone()))
        );
        // `none` builds no calls: the grammar is the answer alone.
        assert_eq!(
            build(ChatToolChoice::None, json()),
            Some(BuiltOutputGrammar::final_output_only(Format::json_schema(
                schema
            )))
        );
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
            let mut processor = new_processor(
                &mut request,
                "other-model",
                tokenizer(),
                &ParserSelection::Explicit("qwen3_coder".to_string()),
                &ParserSelection::None,
                None,
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

        new_processor(
            &mut request,
            "other-model",
            tokenizer(),
            &selection,
            &selection,
            None,
        )
        .unwrap();
    }

    #[test]
    fn auto_auto_gemma4_model_uses_unified_parser() {
        let mut request = ChatRequest::for_test();

        new_processor(
            &mut request,
            "google/gemma-4-27b-it",
            tokenizer(),
            &ParserSelection::Auto,
            &ParserSelection::Auto,
            None,
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
            new_processor(
                &mut ChatRequest::for_test(),
                "google/gemma-4-27b-it",
                tokenizer(),
                tool,
                reasoning,
                None,
            )
            .unwrap();
        }
    }

    #[test]
    fn conflicting_unified_parser_selections_report_resolved_names() {
        let mut request = ChatRequest::for_test();
        let error = match new_processor(
            &mut request,
            "Qwen/Qwen3-8B",
            tokenizer(),
            &ParserSelection::Auto,
            &ParserSelection::Explicit("gemma4".to_string()),
            None,
        ) {
            Ok(_) => panic!("expected mixed Gemma4 parser selection to fail"),
            Err(error) => error,
        };

        expect_test::expect!["unified parsing requires the tool and reasoning selections to resolve to the same parser; resolved tool=qwen3_json, reasoning=gemma4"]
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
        new_processor_with_template(
            &mut request,
            model_id,
            tokenizer(),
            template,
            tool,
            reasoning,
            None,
        )?;
        Ok(request.decode_options.skip_special_tokens == Some(false))
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
    fn explicit_beam_uses_the_model_response_template() {
        let beam = ParserSelection::Explicit("beam".to_string());
        crate::parser::validate_parser_overrides(&beam, &beam).unwrap();
        assert!(keeps_special_tokens("other-model", &loaded_template(), &beam, &beam).unwrap());

        let error =
            keeps_special_tokens("other-model", &Err(HfTemplateError::Missing), &beam, &beam)
                .unwrap_err()
                .to_report_string();
        expect_test::expect!["failed to initialize unified parser `beam`: the model's tokenizer_config.json provides no response_template"]
            .assert_eq(&error);
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
    fn mistral_reasoning_keeps_special_tokens() {
        let mut request = ChatRequest::for_test();
        new_processor_with_template(
            &mut request,
            "mistralai/Mistral-Large-4-1T-A52B",
            Arc::new(
                TestTokenizer::new()
                    .with_special_token("[THINK]", 256)
                    .with_special_token("[/THINK]", 257),
            ),
            &Err(HfTemplateError::Missing),
            &ParserSelection::Auto,
            &ParserSelection::Explicit("mistral".to_string()),
            None,
        )
        .unwrap();
        assert_eq!(request.decode_options.skip_special_tokens, Some(false));
    }

    #[test]
    fn auto_selects_no_reasoning_parser_for_mistral_models() {
        // Most Mistral tokenizers have no `[THINK]` token, so `mistral` stays
        // opt-in rather than failing every request under `auto`.
        let mut request = ChatRequest::for_test();
        new_processor_with_template(
            &mut request,
            "mistralai/Mistral-7B-Instruct-v0.3",
            Arc::new(TestTokenizer::new()),
            &Err(HfTemplateError::Missing),
            &ParserSelection::Auto,
            &ParserSelection::Auto,
            None,
        )
        .unwrap();
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
