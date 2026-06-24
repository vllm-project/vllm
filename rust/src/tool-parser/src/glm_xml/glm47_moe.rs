use super::{GlmXmlToolParser, Separator};
use crate::{Result, StructuralTagModel, Tool, ToolParser, ToolParserOutput};

/// Tool parser for GLM-4.7 MoE XML-style tool calls.
///
/// GLM-4.7 reuses the GLM-4.5 parser with a more flexible function-name
/// separator, so the name may be followed by whitespace, a newline, or the
/// first `<arg_key>` tag directly.
pub struct Glm47MoeToolParser(GlmXmlToolParser);

impl Glm47MoeToolParser {
    fn new(tools: &[Tool]) -> Self {
        Self(GlmXmlToolParser::new(tools, Separator::Flexible))
    }
}

impl ToolParser for Glm47MoeToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_model(&self) -> Option<StructuralTagModel> {
        Some(StructuralTagModel::Glm47)
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

    use super::Glm47MoeToolParser;
    use crate::ToolParserTestExt as _;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};

    fn glm47_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| format!("<arg_key>{name}</arg_key><arg_value>{value}</arg_value>"))
            .collect::<Vec<_>>()
            .join("");
        format!("<tool_call>{function_name}{params}</tool_call>")
    }

    #[test]
    fn glm47_parse_complete_extracts_single_tool_call() {
        let mut parser = Glm47MoeToolParser::new(&test_tools());
        let output = format!(
            "Let me search for that.\n{}",
            glm47_tool_call(
                "get_weather",
                &[("city", "Beijing"), ("date", "2024-12-25")]
            )
        );

        let output = parser.parse_complete(&output).unwrap();

        assert_eq!(output.normal_text, "Let me search for that.\n");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls[0].arguments).unwrap(),
            json!({"city": "Beijing", "date": "2024-12-25"})
        );
    }

    #[test]
    fn glm47_streaming_extracts_multiple_tool_calls() {
        let mut parser = Glm47MoeToolParser::new(&test_tools());
        let output = format!(
            "{}{}",
            glm47_tool_call("get_weather", &[("city", "Shanghai")]),
            glm47_tool_call("add", &[("x", "1"), ("y", "2")])
        );

        let chunks = split_by_chars(&output, 7);
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text, "");
        assert_eq!(output.calls.len(), 2);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[1].name.as_deref(), Some("add"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls[1].arguments).unwrap(),
            json!({"x": 1, "y": 2})
        );
    }

    #[test]
    fn glm47_parse_complete_converts_schema_types() {
        let mut parser = Glm47MoeToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&glm47_tool_call(
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
            serde_json::from_str::<Value>(&output.calls[0].arguments).unwrap(),
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
    fn glm47_parse_complete_extracts_zero_argument_call() {
        let mut parser = Glm47MoeToolParser::new(&test_tools());

        let output = parser.parse_complete("<tool_call>add</tool_call>").unwrap();

        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("add"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls[0].arguments).unwrap(),
            json!({})
        );
    }
}
