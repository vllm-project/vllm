use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::parser::tool::{Result, ToolParseResult, ToolParser};
use crate::request::ChatTool;

const QWEN_XML_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Qwen XML",
    start_marker: "<tool_call>",
    end_marker: "</tool_call>",
    marker_whitespace: JsonToolCallWhitespace::Exact("\n"),
    delimiter: None,
    name_key: "name",
    arguments_key: "arguments",
};

/// Tool parser for Qwen XML-wrapped JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>
/// {"name": "get_weather", "arguments": {"location":"Tokyo"}}
/// </tool_call>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
///
/// Note: parallel calls are represented as repeated
/// `<tool_call>...</tool_call>` blocks, not as multiple calls inside one tag.
pub struct Qwen3XmlToolParser {
    inner: JsonToolCallParser,
}

impl Qwen3XmlToolParser {
    /// Create a Qwen XML tool parser.
    fn new(_tools: &[ChatTool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(QWEN_XML_CONFIG),
        }
    }
}

impl ToolParser for Qwen3XmlToolParser {
    /// Create a boxed Qwen XML tool parser.
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Push one decoded text chunk through the Qwen XML parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.inner.push(chunk)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        self.inner.finish()
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::Qwen3XmlToolParser;
    use crate::parser::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::parser::tool::{ToolParseResult, ToolParser};

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(
            "<tool_call>\n{{\"name\": \"{function_name}\", \"arguments\": {arguments}}}\n</tool_call>"
        )
    }

    #[test]
    fn qwen_xml_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn qwen_xml_parse_complete_extracts_raw_json_arguments() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let arguments = r#"{ "location": "Tokyo", "days": "3" }"#;
        let result = parser
            .parse_complete(&format!(
                "Let me check.\n{}",
                build_tool_call("get_weather", arguments)
            ))
            .unwrap();

        assert_eq!(result.normal_text, "Let me check.\n");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn qwen_xml_does_not_validate_or_normalize_arguments() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let result = parser.parse_complete(&build_tool_call("get_weather", arguments)).unwrap();

        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn qwen_xml_streaming_emits_argument_deltas() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let chunks = [
            "<tool_call>",
            "\n{\"name\": \"get_weather\", \"arguments\": ",
            "{\"location\":",
            "\"Beijing\"",
            "}",
            "}\n</tool_call>",
        ];

        let mut result = ToolParseResult::default();
        let mut observed_arguments = Vec::new();
        for chunk in chunks {
            let next = parser.push(chunk).unwrap();
            observed_arguments.extend(
                next.calls
                    .iter()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            result.append(next);
        }
        result.append(parser.finish().unwrap());

        assert_eq!(observed_arguments, ["{\"location\":", "\"Beijing\"", "}"]);
        assert_eq!(
            result.coalesce_calls().calls[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn qwen_xml_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = Qwen3XmlToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text, "hello ");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn qwen_xml_keeps_end_marker_literal_inside_json_string() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let arguments = r#"{"text":"literal </tool_call> inside"}"#;
        let result = parser.parse_complete(&build_tool_call("echo", arguments)).unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn qwen_xml_decodes_escaped_function_name() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let result = parser
            .parse_complete(
                r#"<tool_call>
{"name":"say_\"hi","arguments":{}}
</tool_call>"#,
            )
            .unwrap();

        assert_eq!(result.calls[0].name.as_deref(), Some("say_\"hi"));
    }

    #[test]
    fn qwen_xml_requires_newline_after_tool_call_start() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let input = r#"<tool_call>{"name":"get_weather","arguments":{}}
</tool_call>"#;

        let result = parser.parse_complete(input).unwrap();

        assert_eq!(result.normal_text, input);
        assert!(result.calls.is_empty());
    }

    #[test]
    fn qwen_xml_requires_newline_before_tool_call_end() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let error = parser
            .parse_complete(
                r#"<tool_call>
{"name":"get_weather","arguments":{}}</tool_call>"#,
            )
            .unwrap_err();

        assert!(error.to_report_string().starts_with("tool parser parsing failed:"));
    }

    #[test]
    fn qwen_xml_streaming_extracts_multiple_tool_calls() {
        let input = format!(
            "{}{}",
            build_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            build_tool_call("add", r#"{"x":1,"y":2}"#),
        );
        let chunks = split_by_chars(&input, 7);
        let mut parser = Qwen3XmlToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

        expect![[r#"
            ToolParseResult {
                normal_text: "",
                calls: [
                    ToolCallDelta {
                        tool_index: 0,
                        name: Some(
                            "get_weather",
                        ),
                        arguments: "{\"location\":\"Shanghai\"}",
                    },
                    ToolCallDelta {
                        tool_index: 1,
                        name: Some(
                            "add",
                        ),
                        arguments: "{\"x\":1,\"y\":2}",
                    },
                ],
            }
        "#]]
        .assert_debug_eq(&result);
    }

    #[test]
    fn qwen_xml_finish_fails_incomplete_tool_call() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        parser
            .push(
                r#"<tool_call>
{"name":"get_weather","arguments":{"location""#,
            )
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete Qwen XML tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn qwen_xml_malformed_field_order_fails_fast() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let error = parser
            .push(
                r#"<tool_call>
{"arguments":{},"name":"get_weather"}
</tool_call>"#,
            )
            .unwrap_err();

        expect![[r#"
            tool parser parsing failed: invalid Qwen XML
            expected `name`"#]]
        .assert_eq(&error.to_report_string());
    }
}
