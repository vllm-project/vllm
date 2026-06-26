use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

const LONGCAT_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "LongCat",
    start_marker: "<longcat_tool_call>",
    end_marker: "</longcat_tool_call>",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: None,
    name_key: "name",
    arguments_key: &["arguments"],
};

/// Tool parser for LongCat-Flash XML-wrapped JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <longcat_tool_call>{"name": "get_weather", "arguments": {"location":"Tokyo"}}</longcat_tool_call>
/// ```
///
/// LongCat uses the same JSON object shape as Hermes, with LongCat-specific
/// wrapper tags.
pub struct LongcatToolParser {
    inner: JsonToolCallParser,
}

impl LongcatToolParser {
    /// Create a LongCat tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(LONGCAT_CONFIG),
        }
    }
}

impl ToolParser for LongcatToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
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
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::LongcatToolParser;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(
            r#"<longcat_tool_call>{{"name":"{function_name}","arguments":{arguments}}}</longcat_tool_call>"#
        )
    }

    #[test]
    fn longcat_parse_complete_without_tool_call_keeps_text() {
        let mut parser = LongcatToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn longcat_parse_complete_extracts_raw_json_arguments() {
        let mut parser = LongcatToolParser::new(&test_tools());
        let arguments = r#"{"city":"Tokyo"}"#;
        let output = parser.parse_complete(&build_tool_call("get_weather", arguments)).unwrap();

        assert_eq!(output.normal_text(), "");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn longcat_parse_complete_preserves_surrounding_text() {
        let mut parser = LongcatToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&format!(
                "Let me check.\n{}\nDone.",
                build_tool_call("get_weather", r#"{"city":"Tokyo"}"#)
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "Let me check.\n\nDone.");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn longcat_accepts_newline_after_tool_call_start() {
        let mut parser = LongcatToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"<longcat_tool_call>
{"name":"get_weather","arguments":{}}</longcat_tool_call>"#,
            )
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, "{}");
    }

    #[test]
    fn longcat_parse_complete_handles_json_argument_types() {
        let mut parser = LongcatToolParser::new(&test_tools());
        let arguments = concat!(
            r#"{"string_field":"hello","int_field":42,"float_field":3.14,"#,
            r#""bool_field":true,"null_field":null,"array_field":["a","b"],"#,
            r#""object_field":{"nested":"value"},"empty_array":[],"empty_object":{}}"#
        );
        let output = parser.parse_complete(&build_tool_call("test_function", arguments)).unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn longcat_parse_complete_handles_escaped_strings() {
        let mut parser = LongcatToolParser::new(&test_tools());
        let arguments = r#"{"quoted":"He said \"hello\"","path":"C:\\Users\\file.txt","newline":"line1\nline2"}"#;
        let output = parser.parse_complete(&build_tool_call("test_function", arguments)).unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn longcat_preserves_special_tokens() {
        let parser = LongcatToolParser::new(&test_tools());

        assert!(parser.preserve_special_tokens());
    }

    #[test]
    fn longcat_streaming_emits_argument_deltas() {
        let mut parser = LongcatToolParser::new(&test_tools());
        let chunks = [
            "preface <longcat",
            "_tool_call>{\"name\":\"get_weather\",\"arguments\":",
            "{\"city\":",
            "\"Beijing\"",
            "}",
            "}</longcat_tool_call> suffix",
        ];

        let mut output = ToolParserOutput::default();
        let mut observed_arguments = Vec::new();
        for chunk in chunks {
            let next = parser.parse_chunk(chunk).unwrap();
            observed_arguments.extend(
                next.calls()
                    .iter()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            output.append(next);
        }
        output.append(parser.finish().unwrap());

        assert_eq!(observed_arguments, ["{\"city\":", "\"Beijing\"", "}"]);
        assert_eq!(output.normal_text(), "preface  suffix");
        assert_eq!(
            output.coalesce().calls()[0].arguments,
            r#"{"city":"Beijing"}"#
        );
    }

    #[test]
    fn longcat_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            build_tool_call("get_weather", r#"{"city":"Tokyo"}"#)
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = LongcatToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "hello ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"city":"Tokyo"}"#);
    }

    #[test]
    fn longcat_streaming_extracts_multiple_tool_calls() {
        let input = format!(
            "{}\n{}",
            build_tool_call("get_weather", r#"{"city":"Tokyo"}"#),
            build_tool_call("get_time", r#"{"timezone":"Asia/Tokyo"}"#),
        );
        let chunks = split_by_chars(&input, 7);
        let mut parser = LongcatToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        expect![[r#"
            ToolParserOutput {
                events: [
                    Text(
                        "\n",
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"Tokyo\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "get_time",
                            ),
                            arguments: "{\"timezone\":\"Asia/Tokyo\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn longcat_rejects_non_object_arguments() {
        let mut parser = LongcatToolParser::new(&test_tools());
        let error = parser
            .parse_complete(
                r#"<longcat_tool_call>{"name":"func","arguments":"not a dict"}</longcat_tool_call>"#,
            )
            .unwrap_err();

        assert!(error.to_report_string().contains("JSON object argument"));
    }

    #[test]
    fn longcat_finish_fails_incomplete_tool_call() {
        let mut parser = LongcatToolParser::new(&test_tools());
        parser
            .parse_chunk(r#"<longcat_tool_call>{"name":"func","arguments":{"#)
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete LongCat tool call"]
            .assert_eq(&error.to_report_string());
    }
}
