use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::{Result, Tool, ToolParser, ToolParserOutput};

const ERNIE45_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "ERNIE 4.5",
    start_marker: "<tool_call>",
    end_marker: "</tool_call>",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: None,
    name_key: "name",
    arguments_key: &["arguments"],
};

/// Tool parser for Baidu ERNIE 4.5 `<tool_call>`-wrapped JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>{"name": "get_weather", "arguments": {"location":"Tokyo"}}</tool_call>
/// ```
///
/// The wire format matches Hermes: each call is one `<tool_call>...</tool_call>`
/// block wrapping a `{"name", "arguments"}` object, and parallel calls are
/// repeated blocks rather than multiple calls inside one tag.
///
/// ERNIE 4.5 thinking models also emit `</think>`, `<response>`, and
/// `</response>` blocks alongside tool calls. Those are reasoning/content
/// boundaries handled by the reasoning parser and the response renderer
/// respectively, so this tool parser only frames `<tool_call>` blocks and
/// passes everything else through as `normal_text`.
pub struct Ernie45ToolParser {
    inner: JsonToolCallParser,
}

impl Ernie45ToolParser {
    /// Create an ERNIE 4.5 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(ERNIE45_CONFIG),
        }
    }
}

impl ToolParser for Ernie45ToolParser {
    /// Create a boxed ERNIE 4.5 tool parser.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
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

    use super::Ernie45ToolParser;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(r#"<tool_call>{{"name":"{function_name}","arguments":{arguments}}}</tool_call>"#)
    }

    #[test]
    fn ernie45_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Ernie45ToolParser::new(&test_tools());
        let output = parser.parse_complete("just a plain answer").unwrap();

        assert_eq!(output.normal_text, "just a plain answer");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn ernie45_parse_complete_extracts_raw_json_arguments() {
        let mut parser = Ernie45ToolParser::new(&test_tools());
        let arguments = r#"{ "location": "Tokyo", "days": "3" }"#;
        let output = parser
            .parse_complete(&format!(
                "Let me check.\n{}",
                build_tool_call("get_weather", arguments)
            ))
            .unwrap();

        assert_eq!(output.normal_text, "Let me check.\n");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].tool_index, 0);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[0].arguments, arguments);
    }

    #[test]
    fn ernie45_accepts_newline_after_tool_call_start() {
        let mut parser = Ernie45ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"<tool_call>
{"name":"get_weather","arguments":{}}
</tool_call>"#,
            )
            .unwrap();

        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn ernie45_does_not_validate_or_normalize_arguments() {
        let mut parser = Ernie45ToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let output = parser.parse_complete(&build_tool_call("get_weather", arguments)).unwrap();

        assert_eq!(output.calls[0].arguments, arguments);
    }

    #[test]
    fn ernie45_streaming_emits_argument_deltas() {
        let mut parser = Ernie45ToolParser::new(&test_tools());
        let chunks = [
            "preface <tool",
            "_call>{\"name\":\"get_weather\",\"arguments\":",
            "{\"location\":",
            "\"Beijing\"",
            "}",
            "}</tool_call> suffix",
        ];

        let mut output = ToolParserOutput::default();
        let mut observed_arguments = Vec::new();
        for chunk in chunks {
            let next = parser.parse_chunk(chunk).unwrap();
            observed_arguments.extend(
                next.calls
                    .iter()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            output.append(next);
        }
        output.append(parser.finish().unwrap());

        assert_eq!(observed_arguments, ["{\"location\":", "\"Beijing\"", "}"]);
        assert_eq!(output.normal_text, "preface  suffix");
        assert_eq!(
            output.coalesce_calls().calls[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn ernie45_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = Ernie45ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text, "hello ");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn ernie45_streaming_extracts_multiple_tool_calls() {
        let input = format!(
            "{}{}",
            build_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            build_tool_call("add", r#"{"x":1,"y":2}"#),
        );
        let chunks = split_by_chars(&input, 7);
        let mut parser = Ernie45ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        expect![[r#"
            ToolParserOutput {
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
        .assert_debug_eq(&output);
    }

    #[test]
    fn ernie45_finish_fails_incomplete_tool_call() {
        let mut parser = Ernie45ToolParser::new(&test_tools());
        parser
            .parse_chunk(r#"<tool_call>{"name":"get_weather","arguments":{"location""#)
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete ERNIE 4.5 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn ernie45_does_not_preserve_special_tokens() {
        let parser = Ernie45ToolParser::new(&test_tools());
        assert!(!parser.preserve_special_tokens());
    }
}
