// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::qwen_coder::{Qwen3CoderToolParser, QwenCoderConfig};
use super::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

const MIMO_CONFIG: QwenCoderConfig = QwenCoderConfig {
    parser_name: "MiMo",
    tool_call_start: "<tool_call>",
    first_tool_call_start: "<tool_call>",
    next_tool_call_start: "<tool_call>",
    tool_call_end: "</tool_call>",
    trim_parameter_newlines: false,
};

/// MiMo V2 uses Qwen Coder's XML grammar with unframed content and parameter values.
pub struct MiMoToolParser {
    inner: Qwen3CoderToolParser,
}

impl ToolParser for MiMoToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self {
            inner: Qwen3CoderToolParser::with_config(tools, MIMO_CONFIG),
        }))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        self.inner.structural_tag_builder()
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
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
    use super::*;
    use crate::tool::ToolParserTestExt as _;
    use crate::tool::test_utils::{collect_stream, split_by_chars};
    use crate::tool::tests::assert_tool_framing_preserves_body_whitespace;
    use serde_json::{Value, json};

    #[test]
    fn bare_tool_boundaries_preserve_body_whitespace() {
        assert_tool_framing_preserves_body_whitespace::<MiMoToolParser>(
            "",
            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
        );
    }

    #[test]
    fn adjacent_calls_preserve_parameter_whitespace_and_scalar_types_across_chunks() {
        let tools = [Tool {
            name: "convert".into(),
            description: None,
            strict: None,
            parameters: json!({"type": "object", "properties": {
                "flag": {"type": "boolean"}, "empty": {"type": ["string", "null"]},
                "text": {"type": "string"}, "literal": {"type": "string"}, "payload": {"type": "object"}
            }}),
        }];
        let call = "<tool_call>\n<function=convert>\n<parameter=flag>True</parameter>\n<parameter=empty>None</parameter>\n<parameter=text>\n value\n</parameter>\n<parameter=literal>None</parameter>\n<parameter=payload>{\"city\":\"杭州\"}</parameter>\n</function>\n</tool_call>";
        let input = format!("answer\n\n{call}{call}");
        let expected = MiMoToolParser::create(&tools).unwrap().parse_complete(&input).unwrap();
        assert_eq!(expected.normal_text(), "answer\n\n");
        assert_eq!(expected.calls().len(), 2);
        for call in expected.calls() {
            assert_eq!(
                serde_json::from_str::<Value>(&call.arguments).unwrap(),
                json!({"flag": true, "empty": null, "text": "\n value\n", "literal": "None", "payload": {"city": "杭州"}})
            );
        }
        for size in 1..=input.chars().count() {
            let mut parser = MiMoToolParser::create(&tools).unwrap();
            assert_eq!(
                collect_stream(parser.as_mut(), &split_by_chars(&input, size)),
                expected
            );
        }
    }
}
