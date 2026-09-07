// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Adapter that combines reasoning and tool parsers.

use vllm_tokenizer::{DecodedText, DynTokenizer};

use super::{Result, UnifiedParser, UnifiedParserError, UnifiedParserOutput};
use crate::reasoning::ReasoningParser;
use crate::tool::{StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

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

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        self.tool.as_ref().and_then(|parser| parser.structural_tag_builder())
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
    use vllm_tokenizer::{DecodedText, TokenAnchor, TokenAttribution};

    use super::CombinedParser;
    use crate::reasoning::{Qwen3ReasoningParser, ReasoningDelta, ReasoningParser};
    use crate::tool::{Qwen3XmlToolParser, Tool, ToolParser};
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
        assert!(parser.structural_tag_builder().is_some());

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
}
