// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Adapter that combines reasoning and tool parsers.

use vllm_tokenizer::{DecodedText, DynTokenizer};

use super::{Result, UnifiedParser, UnifiedParserError, UnifiedParserOutput};
use crate::output_grammar::{self, BuiltOutputGrammar, OutputGrammarContext};
use crate::reasoning::ReasoningParser;
use crate::tool::{Tool, ToolParser, ToolParserOutput};

/// Unified parser that composes existing reasoning and tool parsers.
pub struct CombinedParser {
    reasoning: Option<Box<dyn ReasoningParser>>,
    tool: Option<Box<dyn ToolParser>>,
}

impl CombinedParser {
    /// Create a combined parser from optional reasoning and tool parsers.
    pub fn new(
        reasoning: Option<Box<dyn ReasoningParser>>,
        tool: Option<Box<dyn ToolParser>>,
    ) -> Self {
        Self { reasoning, tool }
    }

    /// Create a text-only combined parser.
    pub fn plain_text_only() -> Self {
        Self {
            reasoning: None,
            tool: None,
        }
    }

    fn parse_tool(&mut self, content: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        let Some(tool) = self.tool.as_mut() else {
            output.push_text(content);
            return Ok(());
        };

        // Preserve any tool output that was already produced before the error.
        let mut tool_output = ToolParserOutput::default();
        let result = tool.parse_into(content, &mut tool_output);
        output.append_tool_output(tool_output);
        result?;

        Ok(())
    }

    fn flush_tool(&mut self) -> Result<UnifiedParserOutput> {
        let Some(tool) = self.tool.as_mut() else {
            return Ok(UnifiedParserOutput::default());
        };

        let output = tool.finish()?;
        let mut unified = UnifiedParserOutput::default();
        unified.append_tool_output(output);
        Ok(unified)
    }
}

impl UnifiedParser for CombinedParser {
    fn create(_tools: &[Tool], _tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Err(UnifiedParserError::CombinedParserConstructor)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        if let Some(reasoning) = self.reasoning.as_mut() {
            reasoning.initialize(prompt_token_ids)?;
        }
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        self.reasoning.as_ref().is_some_and(|parser| parser.preserve_special_tokens())
            || self.tool.as_ref().is_some_and(|parser| parser.preserve_special_tokens())
    }

    fn build_output_grammar(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<BuiltOutputGrammar>> {
        let Some(tool) = self.tool.as_ref() else {
            return Ok(None);
        };
        let Some(visible) = tool.build_visible_format(ctx)? else {
            return Ok(None);
        };
        // The reasoning parser may wrap that language with the reasoning phase
        // implied by its initialized (prompt-derived) state. If it does, the
        // grammar describes the stream from the first generated token and the
        // engine can skip its own reasoning gate; if it declines, the grammar
        // covers only the final output and the engine gate stays in charge.
        let wrapped = match self.reasoning.as_ref() {
            Some(reasoning) => reasoning.wrap_visible_format(ctx, &visible)?,
            None => None,
        };
        Ok(Some(match wrapped {
            Some(full) => BuiltOutputGrammar::from_token_zero(full),
            None => BuiltOutputGrammar::final_output_only(visible),
        }))
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.tool.as_ref().and_then(|parser| parser.tool_call_id(tool_index))
    }

    fn parse_into(&mut self, delta: DecodedText, output: &mut UnifiedParserOutput) -> Result<()> {
        let Some(reasoning) = self.reasoning.as_mut() else {
            return self.parse_tool(&delta.text, output);
        };

        let reasoning_delta = reasoning.push(delta)?;
        if let Some(reasoning) = reasoning_delta.reasoning {
            output.push_reasoning(reasoning);
        }
        if let Some(content) = reasoning_delta.content {
            // Content attributions stop at this boundary: the tool parser trait
            // consumes plain text.
            if !content.text.is_empty() {
                self.parse_tool(&content.text, output)?;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        if let Some(reasoning) = self.reasoning.as_mut() {
            let reasoning_delta = reasoning.finish()?;
            if let Some(reasoning) = reasoning_delta.reasoning {
                output.push_reasoning(reasoning);
            }
            if let Some(content) = reasoning_delta.content
                && !content.text.is_empty()
            {
                self.parse_tool(&content.text, &mut output)?;
            }
        }
        output.append(self.flush_tool()?);
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.tool.as_mut().map_or_else(String::new, |parser| parser.reset())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_tokenizer::test_utils::TestTokenizer;
    use vllm_tokenizer::{DecodedText, DynTokenizer, TokenAnchor, TokenAttribution, Tokenizer};
    use xgrammar_structural_tag::ToolChoice;
    use xgrammar_structural_tag::builders::ReasoningMode;
    use xgrammar_structural_tag::format::Format;

    use super::CombinedParser;
    use crate::output_grammar::{
        GrammarCoverage, OutputGrammarContext, full_format_from_builder_for_test,
    };
    use crate::reasoning::{
        DeepSeekR1ReasoningParser, DeepSeekV3ReasoningParser, Glm45ReasoningParser,
        Glm47ReasoningParser, KimiK2ReasoningParser, MiniMaxM2ReasoningParser,
        NemotronV3ReasoningParser, Qwen3ReasoningParser, ReasoningDelta, ReasoningParser,
        Step3ReasoningParser,
    };
    use crate::tool::{
        DeepSeekV3ToolParser, DeepSeekV4ToolParser, DeepSeekV31ToolParser, DeepSeekV32ToolParser,
        Glm47MoeToolParser, KimiK2ToolParser, MinimaxM2ToolParser, Qwen3CoderToolParser,
        Qwen3XmlToolParser, Tool, ToolParser,
    };
    use crate::unified::{UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<think>", 256)
            .with_regular_token("</think>", 257)
    }

    fn test_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
            }),
            strict: None,
        }]
    }

    fn collect(parser: &mut dyn UnifiedParser, chunks: &[&str]) -> UnifiedParserOutput {
        let mut output = UnifiedParserOutput::default();
        for chunk in chunks {
            parser.parse_into(DecodedText::unattributed(*chunk), &mut output).unwrap();
        }
        output.append(parser.finish().unwrap());
        output
    }

    fn assert_builder_parity<R, T>(name: &str)
    where
        R: ReasoningParser + 'static,
        T: ToolParser + 'static,
    {
        // All builders' reasoning=true forms begin inside a reasoning block.
        // Auto starts outside reasoning and allows a generated reasoning block.
        for (mode, prompt) in [
            (ReasoningMode::Enabled, &[256][..]),
            (ReasoningMode::Auto, &[][..]),
        ] {
            for tool_choice in [
                ToolChoice::required(),
                ToolChoice::function("get_weather"),
                ToolChoice::auto(),
            ] {
                let mut tools = test_tools();
                if tool_choice == ToolChoice::auto() {
                    tools[0].strict = Some(true);
                }
                let ctx = OutputGrammarContext {
                    tools: &tools,
                    tool_choice: &tool_choice,
                    tool_strict_level: Default::default(),
                    parallel_tool_calls: true,
                };
                let tool = T::create(&tools).unwrap();
                let expected = full_format_from_builder_for_test(
                    tool.structural_tag_builder().unwrap(),
                    &ctx,
                    mode,
                )
                .unwrap()
                .unwrap();
                let reasoning = R::create(Arc::new(tokenizer()) as DynTokenizer).unwrap();
                let mut parser = CombinedParser::new(Some(reasoning), Some(tool));
                parser.initialize(prompt).unwrap();
                let actual = parser.build_output_grammar(&ctx).unwrap().unwrap();

                assert_eq!(
                    actual.coverage,
                    GrammarCoverage::FromTokenZero,
                    "{name} {mode:?} {tool_choice:?}"
                );
                assert_eq!(
                    normalize_builder_parity(actual.format),
                    normalize_builder_parity(expected),
                    "{name} {mode:?} {tool_choice:?}"
                );
            }
        }
    }

    fn normalize_builder_parity(format: Format) -> Format {
        match format {
            Format::AnyText(mut text) => {
                text.excludes.clear();
                Format::AnyText(text)
            }
            Format::Tag(mut tag) => {
                tag.content = Box::new(normalize_builder_parity(*tag.content));
                Format::Tag(tag)
            }
            Format::Optional(mut optional) => {
                optional.content = Box::new(normalize_builder_parity(*optional.content));
                Format::Optional(optional)
            }
            Format::Sequence(sequence) => {
                let mut elements = Vec::new();
                for element in sequence.elements {
                    match normalize_builder_parity(element) {
                        Format::Sequence(nested) => elements.extend(nested.elements),
                        element => elements.push(element),
                    }
                }
                Format::sequence(elements)
            }
            format => format,
        }
    }

    struct PreserveReasoningParser;

    impl ReasoningParser for PreserveReasoningParser {
        fn create(
            _tokenizer: vllm_tokenizer::DynTokenizer,
        ) -> crate::reasoning::Result<Box<dyn ReasoningParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self))
        }

        fn preserve_special_tokens(&self) -> bool {
            true
        }

        fn push(&mut self, delta: DecodedText) -> crate::reasoning::Result<ReasoningDelta> {
            Ok(ReasoningDelta {
                reasoning: None,
                content: Some(delta),
            })
        }
    }

    struct PreserveToolParser;

    impl ToolParser for PreserveToolParser {
        fn create(_tools: &[Tool]) -> crate::tool::Result<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self))
        }

        fn preserve_special_tokens(&self) -> bool {
            true
        }

        fn parse_into(
            &mut self,
            chunk: &str,
            output: &mut crate::tool::ToolParserOutput,
        ) -> crate::tool::Result<()> {
            output.push_text(chunk);
            Ok(())
        }

        fn finish(&mut self) -> crate::tool::Result<crate::tool::ToolParserOutput> {
            Ok(crate::tool::ToolParserOutput::default())
        }

        fn reset(&mut self) -> String {
            String::new()
        }
    }

    struct PartialThenErrorToolParser;

    impl ToolParser for PartialThenErrorToolParser {
        fn create(_tools: &[Tool]) -> crate::tool::Result<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self))
        }

        fn parse_into(
            &mut self,
            _chunk: &str,
            output: &mut crate::tool::ToolParserOutput,
        ) -> crate::tool::Result<()> {
            output.push_text("committed");
            Err(crate::tool::ToolParserError::ParsingFailed {
                message: "synthetic failure".to_string(),
            })
        }

        fn finish(&mut self) -> crate::tool::Result<crate::tool::ToolParserOutput> {
            Ok(crate::tool::ToolParserOutput::default())
        }

        fn reset(&mut self) -> String {
            String::new()
        }
    }

    #[test]
    fn combined_parser_emits_reasoning_and_text() {
        let tokenizer = Arc::new(tokenizer());
        let reasoning = Qwen3ReasoningParser::create(tokenizer).unwrap();
        let mut parser = CombinedParser::new(Some(reasoning), None);

        let output = collect(&mut parser, &["<think>work</think>answer"]);

        assert_eq!(
            output.events,
            vec![
                UnifiedParserEvent::Reasoning(DecodedText::unattributed("work")),
                UnifiedParserEvent::Text("answer".to_string()),
            ]
        );
    }

    #[test]
    fn combined_parser_reasoning_events_carry_token_attributions() {
        let tokenizer = Arc::new(tokenizer());
        let reasoning = Qwen3ReasoningParser::create(tokenizer).unwrap();
        let mut parser = CombinedParser::new(Some(reasoning), None);

        let chunk = |token_id: u32, text: &str| DecodedText {
            text: text.to_string(),
            attributions: [TokenAttribution {
                token_id,
                anchor: TokenAnchor::Visible { byte_offset: 0 },
            }]
            .into_iter()
            .collect(),
        };

        let mut output = UnifiedParserOutput::default();
        for (token_id, text) in [
            (1, "<think>"),
            (2, "reason"),
            (3, "</think>"),
            (4, "answer"),
        ] {
            parser.parse_into(chunk(token_id, text), &mut output).unwrap();
        }
        output.append(parser.finish().unwrap());

        // The reasoning tokens keep their attributions through the combined
        // parser; marker tokens (1 and 3) are dropped with their spans.
        let reasoning_ids: Vec<u32> = output
            .events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::Reasoning(piece) => Some(piece),
                _ => None,
            })
            .flat_map(|piece| piece.attributions.iter().map(|attr| attr.token_id))
            .collect();
        assert_eq!(reasoning_ids, [2]);
        assert_eq!(
            output.events,
            vec![
                UnifiedParserEvent::Reasoning(DecodedText {
                    text: "reason".to_string(),
                    attributions: [TokenAttribution {
                        token_id: 2,
                        anchor: TokenAnchor::Visible { byte_offset: 0 },
                    }]
                    .into_iter()
                    .collect(),
                }),
                UnifiedParserEvent::Text("answer".to_string()),
            ]
        );
    }

    #[test]
    fn combined_parser_emits_tool_calls_from_visible_content() {
        let tool = Qwen3XmlToolParser::create(&test_tools()).unwrap();
        let mut parser = CombinedParser::new(None, Some(tool));

        let output = collect(
            &mut parser,
            &[r#"<tool_call>
{"name":"get_weather","arguments":{"location":"Paris"}}
</tool_call>"#],
        );

        assert_eq!(
            output.events,
            vec![
                UnifiedParserEvent::ToolCall(crate::tool::ToolCallDelta {
                    tool_index: 0,
                    name: Some("get_weather".to_string()),
                    arguments: String::new(),
                }),
                UnifiedParserEvent::ToolCall(crate::tool::ToolCallDelta {
                    tool_index: 0,
                    name: None,
                    arguments: r#"{"location":"Paris"}"#.to_string(),
                }),
            ]
        );
    }

    #[test]
    fn combined_parser_preserves_tool_output_on_parse_error() {
        let mut parser = CombinedParser::new(None, Some(Box::new(PartialThenErrorToolParser)));
        let mut output = UnifiedParserOutput::default();

        let error = parser.parse_into(DecodedText::unattributed("bad"), &mut output).unwrap_err();

        assert!(matches!(error, crate::unified::UnifiedParserError::Tool(_)));
        assert_eq!(
            output.events,
            vec![UnifiedParserEvent::Text("committed".to_string())]
        );
    }

    #[test]
    fn combined_parser_preserves_special_tokens_when_either_inner_parser_needs_it() {
        let mut parser = CombinedParser::new(Some(Box::new(PreserveReasoningParser)), None);
        assert!(parser.preserve_special_tokens());

        parser = CombinedParser::new(None, Some(Box::new(PreserveToolParser)));
        assert!(parser.preserve_special_tokens());
    }

    #[test]
    fn split_parser_wrappers_match_builder_full_formats() {
        assert_builder_parity::<Qwen3ReasoningParser, Qwen3XmlToolParser>("qwen3");
        assert_builder_parity::<Qwen3ReasoningParser, Qwen3CoderToolParser>("qwen3.5");
        assert_builder_parity::<DeepSeekR1ReasoningParser, DeepSeekV3ToolParser>("deepseek-r1");
        assert_builder_parity::<DeepSeekV3ReasoningParser, DeepSeekV31ToolParser>("deepseek-v3.1");
        assert_builder_parity::<DeepSeekV3ReasoningParser, DeepSeekV32ToolParser>("deepseek-v3.2");
        assert_builder_parity::<DeepSeekV3ReasoningParser, DeepSeekV4ToolParser>("deepseek-v4");
        assert_builder_parity::<Glm47ReasoningParser, Glm47MoeToolParser>("glm-4.7");
        assert_builder_parity::<KimiK2ReasoningParser, KimiK2ToolParser>("kimi-k2");
        assert_builder_parity::<MiniMaxM2ReasoningParser, MinimaxM2ToolParser>("minimax-m2");
    }

    #[test]
    fn outside_reasoning_allows_optional_generated_reasoning() {
        let tools = test_tools();
        let tool_choice = ToolChoice::required();
        let ctx = OutputGrammarContext {
            tools: &tools,
            tool_choice: &tool_choice,
            tool_strict_level: Default::default(),
            parallel_tool_calls: true,
        };
        let reasoning = Qwen3ReasoningParser::create(Arc::new(tokenizer())).unwrap();
        let tool = Qwen3XmlToolParser::create(&tools).unwrap();
        let visible = tool.build_visible_format(&ctx).unwrap().unwrap();
        let mut parser = CombinedParser::new(Some(reasoning), Some(tool));
        parser.initialize(&[]).unwrap();

        let actual = parser.build_output_grammar(&ctx).unwrap().unwrap();
        let expected = Format::sequence(vec![
            Format::optional(Format::sequence(vec![
                Format::tag("<think>", Format::any_text(), "</think>"),
                Format::const_string("\n\n"),
            ])),
            visible,
        ]);

        assert_eq!(actual.coverage, GrammarCoverage::FromTokenZero);
        assert_eq!(actual.format, expected);
    }

    #[test]
    fn glm45_wrapper_keeps_only_generated_start_framing() {
        let tools = test_tools();
        let tool_choice = ToolChoice::required();
        let ctx = OutputGrammarContext {
            tools: &tools,
            tool_choice: &tool_choice,
            tool_strict_level: Default::default(),
            parallel_tool_calls: true,
        };
        let visible = Format::const_string("answer");

        // GLM-4.5/4.6 thinking prompts leave the whole reasoning opener to generation.
        for (prompt, begin) in [(vec![], "\n<think>"), (vec![256], "")] {
            let mut parser = Glm45ReasoningParser::new(Arc::new(tokenizer())).unwrap();
            parser.initialize(&prompt).unwrap();
            let reasoning = Format::sequence(vec![
                Format::tag(begin, Format::any_text(), "</think>"),
                Format::const_string("\n"),
            ]);
            let reasoning = if prompt.is_empty() {
                Format::optional(reasoning)
            } else {
                reasoning
            };
            assert_eq!(
                parser.wrap_visible_format(&ctx, &visible).unwrap(),
                Some(Format::sequence(vec![reasoning, visible.clone()])),
                "prompt {prompt:?}"
            );
        }
    }

    #[test]
    fn delimited_wrappers_use_model_specific_suffixes() {
        let tools = test_tools();
        let tool_choice = ToolChoice::required();
        let ctx = OutputGrammarContext {
            tools: &tools,
            tool_choice: &tool_choice,
            tool_strict_level: Default::default(),
            parallel_tool_calls: true,
        };
        let visible = Format::const_string("visible");

        let mut nemotron = NemotronV3ReasoningParser::new(Arc::new(tokenizer())).unwrap();
        nemotron.initialize(&[256]).unwrap();
        assert_eq!(
            nemotron.wrap_visible_format(&ctx, &visible).unwrap(),
            Some(Format::sequence(vec![
                Format::sequence(vec![
                    Format::tag("", Format::any_text(), "</think>"),
                    Format::const_string("\n"),
                ]),
                visible.clone(),
            ]))
        );

        let mut step3 = Step3ReasoningParser::new(Arc::new(tokenizer())).unwrap();
        step3.initialize(&[256]).unwrap();
        assert_eq!(
            step3.wrap_visible_format(&ctx, &visible).unwrap(),
            Some(Format::sequence(vec![
                Format::tag("", Format::any_text(), "</think>"),
                visible,
            ]))
        );
    }

    #[test]
    fn prompt_closed_reasoning_keeps_only_unconsumed_separator() {
        let tools = test_tools();
        let tool_choice = ToolChoice::required();
        let ctx = OutputGrammarContext {
            tools: &tools,
            tool_choice: &tool_choice,
            tool_strict_level: Default::default(),
            parallel_tool_calls: true,
        };
        let tokenizer = Arc::new(tokenizer());
        let reasoning = Qwen3ReasoningParser::create(tokenizer.clone()).unwrap();
        let tool = Qwen3XmlToolParser::create(&tools).unwrap();
        let mut parser = CombinedParser::new(Some(reasoning), Some(tool));
        parser.initialize(&[]).unwrap();
        let outside = parser.build_output_grammar(&ctx).unwrap().unwrap().format;

        for (prompt, remaining) in [
            ("</think>", "\n\n"),
            ("</think>\n", "\n"),
            ("</think>\n\n", ""),
        ] {
            parser.initialize(&tokenizer.encode(prompt, false).unwrap()).unwrap();
            let actual = parser.build_output_grammar(&ctx).unwrap().unwrap();
            let expected = if remaining.is_empty() {
                outside.clone()
            } else {
                Format::sequence(vec![Format::const_string(remaining), outside.clone()])
            };
            assert_eq!(actual.coverage, GrammarCoverage::FromTokenZero);
            assert_eq!(
                normalize_builder_parity(actual.format),
                normalize_builder_parity(expected),
                "{prompt:?}"
            );
        }
    }
}
