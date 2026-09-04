// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};

use super::detect_hy_token_suffix;
use crate::output_grammar::{self, BuiltOutputGrammar, OutputGrammarContext};
use crate::reasoning::HyReasoningParser;
use crate::tool::{HyDialect, HyToolMarkers, HyToolParser, Tool};
use crate::unified::{CombinedParser, Result, UnifiedParser, UnifiedParserOutput, token_id};

/// Unified reasoning and tool parser for HY4 output.
pub struct HyV4UnifiedParser {
    inner: CombinedParser,
}

impl HyV4UnifiedParser {
    /// Create a HY4 parser using the suffix encoded in tokenizer added tokens.
    pub fn new(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let suffix = detect_hy_token_suffix(tokenizer.as_ref());
        let markers = HyToolMarkers::new(&suffix, HyDialect::V4);
        for marker in markers.iter() {
            token_id(tokenizer.as_ref(), marker)?;
        }

        let reasoning = HyReasoningParser::new(tokenizer, &suffix)?;
        let tool = HyToolParser::new(tools, &suffix, HyDialect::V4);
        Ok(Self {
            inner: CombinedParser::new(Some(Box::new(reasoning)), Some(Box::new(tool))),
        })
    }
}

impl UnifiedParser for HyV4UnifiedParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Self::new(tools, tokenizer).map(|parser| Box::new(parser) as Box<dyn UnifiedParser>)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids)
    }

    fn preserve_special_tokens(&self) -> bool {
        self.inner.preserve_special_tokens()
    }

    fn build_output_grammar(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<BuiltOutputGrammar>> {
        self.inner.build_output_grammar(ctx)
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.inner.tool_call_id(tool_index)
    }

    fn parse_into(&mut self, delta: DecodedText, output: &mut UnifiedParserOutput) -> Result<()> {
        self.inner.parse_into(delta, output)
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        self.inner.finish()
    }

    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use vllm_tokenizer::{DecodedText, Tokenizer, test_utils::TestTokenizer};
    use xgrammar_structural_tag::{StructuralTag, ToolChoice};

    use super::{HyV4UnifiedParser, UnifiedParser};
    use crate::output_grammar::{GrammarCoverage, OutputGrammarContext};
    use crate::tool::Tool;
    use crate::unified::{UnifiedParserEvent, UnifiedParserOutput};

    fn tokenizer() -> TestTokenizer {
        [
            "<think:opensource>",
            "</think:opensource>",
            "<tool_calls:opensource>",
            "</tool_calls:opensource>",
            "<tool_call:opensource>",
            "</tool_call:opensource>",
            "<arg_key:opensource>",
            "</arg_key:opensource>",
            "<arg_value:opensource>",
            "</arg_value:opensource>",
        ]
        .into_iter()
        .enumerate()
        .fold(TestTokenizer::new(), |tokenizer, (index, token)| {
            tokenizer.with_regular_token(token, 1000 + index as u32)
        })
    }

    fn tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "get_weather".to_string(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                }),
                strict: None,
            },
            Tool {
                name: "get_current_date".to_string(),
                description: None,
                parameters: json!({ "type": "object", "properties": {} }),
                strict: None,
            },
        ]
    }

    #[test]
    fn parses_suffixed_reasoning_and_compact_tool_calls_as_one_stream() {
        let tokenizer = Arc::new(tokenizer());
        let think_start_id = tokenizer.token_to_id("<think:opensource>").unwrap();
        let mut parser = HyV4UnifiedParser::new(&tools(), tokenizer).unwrap();
        parser.initialize(&[think_start_id]).unwrap();

        let chunks = [
            "reasoning</think:open",
            "source>answer<tool_calls:opensource><tool_call:opensource>get_weather",
            "<arg_key:opensource>city</arg_key:opensource>",
            "<arg_value:opensource>Beijing</arg_value:opensource>",
            "</tool_call:opensource><tool_call:opensource>get_current_date",
            "</tool_call:opensource></tool_calls:opensource>",
        ];
        let mut output = UnifiedParserOutput::default();
        for chunk in chunks {
            parser.parse_into(DecodedText::unattributed(chunk), &mut output).unwrap();
        }
        output.append(parser.finish().unwrap());

        assert_eq!(
            output.events,
            vec![
                UnifiedParserEvent::Reasoning(DecodedText::unattributed("reasoning")),
                UnifiedParserEvent::Text("answer".to_string()),
                UnifiedParserEvent::ToolCall(crate::tool::ToolCallDelta {
                    tool_index: 0,
                    name: Some("get_weather".to_string()),
                    arguments: r#"{"city":"Beijing"}"#.to_string(),
                }),
                UnifiedParserEvent::ToolCall(crate::tool::ToolCallDelta {
                    tool_index: 1,
                    name: Some("get_current_date".to_string()),
                    arguments: "{}".to_string(),
                }),
            ]
        );
    }

    #[test]
    fn structural_tag_uses_compact_hy4_skeleton() {
        let tools = tools();
        let parser = HyV4UnifiedParser::new(&tools, Arc::new(tokenizer())).unwrap();
        let tag = parser
            .build_output_grammar(&OutputGrammarContext {
                tools: &tools,
                tool_choice: &ToolChoice::required(),
            })
            .unwrap()
            .unwrap();
        assert_eq!(tag.coverage, GrammarCoverage::FinalOutputOnly);
        let tag = StructuralTag::new(tag.format).to_json_string().unwrap();

        assert!(tag.contains("<tool_calls:opensource>"));
        assert!(tag.contains("<tool_call:opensource>get_weather"));
        assert!(tag.contains("<arg_key:opensource>"));
        assert!(tag.contains("</tool_call:opensource>"));
        assert!(tag.contains("</tool_calls:opensource>"));
        assert!(!tag.contains("tool_sep"));
        assert!(!tag.contains(r#"\n"#));
    }
}
