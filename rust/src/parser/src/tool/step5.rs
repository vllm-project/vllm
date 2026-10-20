// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::DecodedText;

use super::qwen_coder::{Qwen3CoderToolParser, QwenCoderConfig};
use super::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

mod structural_tag;

const STEP5_CONFIG: QwenCoderConfig = QwenCoderConfig {
    parser_name: "Step-5",
    tool_call_start: "<tool_call>",
    first_tool_call_start: "<tool_call>",
    next_tool_call_start: "<tool_call>",
    tool_call_end: "</tool_call>",
    trim_parameter_newlines: true,
};

/// Step-5 uses Qwen Coder's inner XML grammar with bare tool-call boundaries.
/// Its template renders scalar null arguments as Python `None`.
pub struct Step5ToolParser {
    inner: Qwen3CoderToolParser,
}

impl ToolParser for Step5ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self {
            inner: Qwen3CoderToolParser::with_config(tools, STEP5_CONFIG),
        }))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(&structural_tag::Step5StructuralTagBuilder)
    }

    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
        self.inner.parse_into(chunk, output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.inner.finish()
    }

    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::tool::ToolParserTestExt as _;
    use crate::tool::test_utils::{collect_stream, split_by_chars};
    use crate::tool::tests::assert_tool_framing_preserves_body_whitespace;

    #[test]
    fn bare_tool_boundaries_preserve_body_whitespace() {
        assert_tool_framing_preserves_body_whitespace::<Step5ToolParser>(
            "",
            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
        );
    }

    #[test]
    fn adjacent_calls_preserve_template_scalar_types_across_chunks() {
        let tools = [Tool {
            name: "convert".into(),
            description: None,
            parameters: json!({"type": "object", "properties": {
                "flag": {"type": "boolean"},
                "empty": {"type": ["string", "null"]},
                "text": {"type": "string"},
                "payload": {"type": "object"}
            }}),
            strict: None,
            defer_loading: None,
        }];
        let call = "<tool_call>\n<function=convert>\n<parameter=flag>\nTrue\n</parameter>\n<parameter=empty>\nNone\n</parameter>\n<parameter=text>\nNone\n</parameter>\n<parameter=payload>\n{\"city\":\"杭州\"}\n</parameter>\n</function>\n</tool_call>";
        let input = format!("answer\n\n{call}{call}");
        let expected = Step5ToolParser::create(&tools).unwrap().parse_complete(&input).unwrap();
        assert_eq!(expected.normal_text(), "answer\n\n");
        assert_eq!(expected.calls().len(), 2);
        for call in expected.calls() {
            assert_eq!(
                serde_json::from_str::<Value>(&call.arguments).unwrap(),
                json!({
                    "flag": true, "empty": null, "text": "None", "payload": {"city": "杭州"}
                })
            );
        }
        for size in 1..=input.chars().count() {
            let mut parser = Step5ToolParser::create(&tools).unwrap();
            assert_eq!(
                collect_stream(parser.as_mut(), &split_by_chars(&input, size)),
                expected
            );
        }
    }
}
