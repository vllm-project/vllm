//! Adapter that combines reasoning and tool parsers.

use crate::reasoning::ReasoningParser;
use crate::tool::{StructuralTagModel, ToolParser, ToolParserOutput};

use super::{Result, UnifiedParser, UnifiedParserOutput};

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
            output.push_text(content.to_string());
            return Ok(());
        };

        let mut tool_output = ToolParserOutput::default();
        tool.parse_into(content, &mut tool_output)?;
        output.append_tool_output(tool_output);
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

    fn structural_tag_model(&self) -> Option<StructuralTagModel> {
        self.tool.as_ref().and_then(|parser| parser.structural_tag_model())
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.tool.as_ref().and_then(|parser| parser.tool_call_id(tool_index))
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        let Some(reasoning) = self.reasoning.as_mut() else {
            return self.parse_tool(delta, output);
        };

        let reasoning_delta = reasoning.push(delta)?;
        if let Some(reasoning) = reasoning_delta.reasoning {
            output.push_reasoning(reasoning);
        }
        if let Some(content) = reasoning_delta.content {
            self.parse_tool(&content, output)?;
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
            if let Some(content) = reasoning_delta.content {
                self.parse_tool(&content, &mut output)?;
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

    use vllm_tokenizer::Tokenizer;

    use super::CombinedParser;
    use crate::reasoning::{Qwen3ReasoningParser, ReasoningDelta, ReasoningParser};
    use crate::tool::{Qwen3XmlToolParser, Tool, ToolParser};
    use crate::unified::{UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};

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
                "<think>" => Some(1),
                "</think>" => Some(2),
                _ => None,
            }
        }
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
            parser.parse_into(chunk, &mut output).unwrap();
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

        fn push(&mut self, delta: &str) -> crate::reasoning::Result<ReasoningDelta> {
            Ok(ReasoningDelta {
                reasoning: None,
                content: Some(delta.to_string()),
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
            output.normal_text.push_str(chunk);
            Ok(())
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
        let tokenizer = Arc::new(FakeTokenizer);
        let reasoning = Qwen3ReasoningParser::create(tokenizer).unwrap();
        let mut parser = CombinedParser::new(Some(reasoning), None);

        let output = collect(&mut parser, &["<think>work</think>answer"]);

        assert_eq!(
            output.events,
            vec![
                UnifiedParserEvent::Reasoning("work".to_string()),
                UnifiedParserEvent::Text("answer".to_string()),
            ]
        );
    }

    #[test]
    fn combined_parser_emits_tool_calls_from_visible_content() {
        let tool = Qwen3XmlToolParser::create(&test_tools()).unwrap();
        let mut parser = CombinedParser::new(None, Some(tool));
        assert!(matches!(
            parser.structural_tag_model(),
            Some(crate::tool::StructuralTagModel::Qwen3)
        ));

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
    fn combined_parser_preserves_special_tokens_when_either_inner_parser_needs_it() {
        let mut parser = CombinedParser::new(Some(Box::new(PreserveReasoningParser)), None);
        assert!(parser.preserve_special_tokens());

        parser = CombinedParser::new(None, Some(Box::new(PreserveToolParser)));
        assert!(parser.preserve_special_tokens());
    }
}
