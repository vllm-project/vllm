// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{GlmXmlToolParser, Separator};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

/// Tool parser for Ling3 XML-style tool calls.
///
/// Ling3 uses the same XML-style tool-call format as GLM-4.7, with flexible
/// function-name separation (followed by whitespace, a newline, or the first
/// `<arg_key>` tag directly).
pub struct Ling3ToolParser(GlmXmlToolParser);

impl Ling3ToolParser {
    fn new(tools: &[Tool]) -> Self {
        Self(GlmXmlToolParser::new(tools, Separator::Flexible))
    }
}

impl ToolParser for Ling3ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::Glm47.builder())
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.0.parse_into(chunk, output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.0.finish()
    }

    fn reset(&mut self) -> String {
        self.0.reset()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::Ling3ToolParser;
    use crate::tool::ToolParserTestExt as _;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};

    fn ling3_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| format!("<arg_key>{name}</arg_key><arg_value>{value}</arg_value>"))
            .collect::<Vec<_>>()
            .join("");
        format!("<tool_call>{function_name}{params}</tool_call>")
    }

    #[test]
    fn ling3_parse_complete_extracts_single_tool_call() {
        let mut parser = Ling3ToolParser::new(&test_tools());
        let output = format!(
            "Let me search for that.\n{}",
            ling3_tool_call(
                "get_weather",
                &[("city", "Beijing"), ("date", "2024-12-25")]
            )
        );

        let output = parser.parse_complete(&output).unwrap();

        assert_eq!(output.normal_text(), "Let me search for that.\n");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city": "Beijing", "date": "2024-12-25"})
        );
    }

    #[test]
    fn ling3_parse_complete_extracts_tool_call_without_newline() {
        let mut parser = Ling3ToolParser::new(&test_tools());
        let input = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>";

        let output = parser.parse_complete(input).unwrap();

        assert_eq!(output.normal_text(), "");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city": "Beijing"})
        );
    }

    #[test]
    fn ling3_parse_complete_extracts_zero_argument_call() {
        let mut parser = Ling3ToolParser::new(&test_tools());

        let output = parser.parse_complete("<tool_call>ping</tool_call>").unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("ping"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({})
        );
    }

    #[test]
    fn ling3_streaming_extracts_multiple_tool_calls() {
        let mut parser = Ling3ToolParser::new(&test_tools());
        let output = format!(
            "{}{}",
            ling3_tool_call("get_weather", &[("city", "Shanghai")]),
            ling3_tool_call("add", &[("x", "1"), ("y", "2")])
        );

        let chunks = split_by_chars(&output, 7);
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "");
        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[1].name.as_deref(), Some("add"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[1].arguments).unwrap(),
            json!({"x": 1, "y": 2})
        );
    }

    #[test]
    fn ling3_parse_complete_converts_schema_types() {
        let mut parser = Ling3ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&ling3_tool_call(
                "convert",
                &[
                    ("whole", "42"),
                    ("flag", "true"),
                    ("payload", r#"{"nested":{"key":"value"}}"#),
                    ("items", "[1, 2, 3]"),
                    ("empty", ""),
                ],
            ))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "whole": 42,
                "flag": true,
                "payload": {"nested": {"key": "value"}},
                "items": [1, 2, 3],
                "empty": ""
            })
        );
    }

    #[test]
    fn ling3_streaming_handles_split_markers() {
        let mut parser = Ling3ToolParser::new(&test_tools());
        let chunks = [
            "Leading text <tool",
            "_call>get_weather",
            "<arg_key>ci",
            "ty</arg_key><arg_value>Beijing</arg",
            "_value></tool_call> Trailing",
        ];

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "Leading text ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city": "Beijing"})
        );
    }
}
