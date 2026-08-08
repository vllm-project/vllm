// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::DynTokenizer;

use super::{CombinedParser, Result, UnifiedParser, UnifiedParserOutput, token_id};
use crate::reasoning::{
    M3_THINK_END, M3_THINK_START, MiniMaxM3ReasoningParser, ReasoningParser,
    last_reasoning_boundary,
};
use crate::tool::{MinimaxM3ToolParser, StructuralTagBuilder, Tool, ToolParser};

/// Python bridge adapter backed by the existing MiniMax M3 split parsers.
///
/// Rust chat keeps MiniMax M3 in its reasoning and tool parser registries.
pub struct MiniMaxM3CombinedParser {
    inner: CombinedParser,
    tokenizer: DynTokenizer,
    reasoning_start_token_id: u32,
    reasoning_end_token_id: u32,
    initial_in_reasoning: bool,
}

impl MiniMaxM3CombinedParser {
    /// Create the MiniMax M3 adapter exported through the Python bridge.
    pub fn new(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let reasoning_start_token_id = token_id(tokenizer.as_ref(), M3_THINK_START)?;
        let reasoning_end_token_id = token_id(tokenizer.as_ref(), M3_THINK_END)?;
        let reasoning = MiniMaxM3ReasoningParser::create(tokenizer.clone())?;
        let tool = MinimaxM3ToolParser::create(tools)?;
        Ok(Self {
            inner: CombinedParser::new(Some(reasoning), Some(tool)),
            tokenizer,
            reasoning_start_token_id,
            reasoning_end_token_id,
            initial_in_reasoning: false,
        })
    }
}

impl UnifiedParser for MiniMaxM3CombinedParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Self::new(tools, tokenizer).map(|parser| Box::new(parser) as Box<dyn UnifiedParser>)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.initial_in_reasoning = last_reasoning_boundary(
            prompt_token_ids,
            self.reasoning_start_token_id,
            self.reasoning_end_token_id,
            self.tokenizer.as_ref(),
        )
        .unwrap_or(false);
        self.inner.initialize(prompt_token_ids)
    }

    fn preserve_special_tokens(&self) -> bool {
        self.inner.preserve_special_tokens()
    }

    fn reasoning_start_str(&self) -> Option<&str> {
        Some(M3_THINK_START)
    }

    fn reasoning_end_str(&self) -> Option<&str> {
        Some(M3_THINK_END)
    }

    fn is_reasoning_end(&self, input_ids: &[u32]) -> bool {
        for token_id in input_ids.iter().rev() {
            if *token_id == self.reasoning_end_token_id {
                return true;
            }
            if *token_id == self.reasoning_start_token_id {
                return false;
            }
        }
        false
    }

    fn count_reasoning_tokens(&self, input_ids: &[u32]) -> usize {
        let mut depth = usize::from(self.initial_in_reasoning);
        let mut count = 0;
        for token_id in input_ids {
            if *token_id == self.reasoning_start_token_id {
                depth += 1;
            } else if *token_id == self.reasoning_end_token_id {
                depth = depth.saturating_sub(1);
            } else if depth > 0 {
                count += 1;
            }
        }
        count
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        self.inner.structural_tag_builder()
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.inner.tool_call_id(tool_index)
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
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
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::MiniMaxM3CombinedParser;
    use crate::tool::Tool;
    use crate::unified::{UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};

    fn tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
            }),
            strict: None,
        }]
    }

    #[test]
    fn minimax_m3_combines_reasoning_text_and_tool_calls() {
        let tokenizer = Arc::new(
            TestTokenizer::new()
                .with_special_token("<mm:think>", 256)
                .with_special_token("</mm:think>", 257),
        );
        let mut parser = MiniMaxM3CombinedParser::new(&tools(), tokenizer).unwrap();
        parser.initialize(&[256]).unwrap();
        assert_eq!(parser.count_reasoning_tokens(&[101, 102, 257, 103]), 2);

        let mut output = UnifiedParserOutput::default();
        parser
            .parse_into(
                "plan</mm:think>answer\
                 ]<]minimax[>[<tool_call>\
                 ]<]minimax[>[<invoke name=\"get_weather\">\
                 ]<]minimax[>[<city>Paris]<]minimax[>[</city>\
                 ]<]minimax[>[</invoke>\
                 ]<]minimax[>[</tool_call>",
                &mut output,
            )
            .unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(
            output.events,
            vec![
                UnifiedParserEvent::Reasoning("plan".to_string()),
                UnifiedParserEvent::Text("answer".to_string()),
                UnifiedParserEvent::ToolCall(crate::tool::ToolCallDelta {
                    tool_index: 0,
                    name: Some("get_weather".to_string()),
                    arguments: r#"{"city":"Paris"}"#.to_string(),
                },),
            ]
        );
    }
}
