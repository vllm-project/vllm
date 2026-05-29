use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::{Result, Tool, ToolParser, ToolParserOutput};

const GRANITE4_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Granite 4",
    start_marker: "<tool_call>",
    end_marker: "</tool_call>",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: None,
    name_key: "name",
    arguments_key: &["arguments"],
};

/// Tool parser for IBM Granite 4 `<tool_call>`-wrapped JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>{"name": "get_weather", "arguments": {"location":"Tokyo"}}</tool_call>
/// ```
///
/// The wire format matches Hermes: each call is one `<tool_call>...</tool_call>`
/// block wrapping a `{"name", "arguments"}` object, and parallel calls are
/// repeated blocks rather than multiple calls inside one tag. Unlike Hermes,
/// Granite marks the `<tool_call>` / `</tool_call>` sentinels as tokenizer
/// special tokens, so `preserve_special_tokens()` is true (mirroring the Python
/// parser's `adjust_request(skip_special_tokens=False)`).
pub struct Granite4ToolParser {
    inner: JsonToolCallParser,
}

impl Granite4ToolParser {
    /// Create a Granite 4 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(GRANITE4_CONFIG),
        }
    }
}

impl ToolParser for Granite4ToolParser {
    /// Create a boxed Granite 4 tool parser.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Preserve special-token markers while decoding, since `<tool_call>` and
    /// `</tool_call>` are tokenizer special tokens in Granite models.
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

    use super::Granite4ToolParser;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(r#"<tool_call>{{"name":"{function_name}","arguments":{arguments}}}</tool_call>"#)
    }

    #[test]
    fn granite4_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Granite4ToolParser::new(&test_tools());
        let output = parser.parse_complete("just a plain answer").unwrap();

        assert_eq!(output.normal_text, "just a plain answer");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn granite4_parse_complete_extracts_raw_json_arguments() {
        let mut parser = Granite4ToolParser::new(&test_tools());
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
    fn granite4_accepts_newline_after_tool_call_start() {
        let mut parser = Granite4ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"<tool_call>
{"name":"get_weather","arguments":{}}</tool_call>"#,
            )
            .unwrap();

        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn granite4_does_not_validate_or_normalize_arguments() {
        let mut parser = Granite4ToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let output = parser.parse_complete(&build_tool_call("get_weather", arguments)).unwrap();

        assert_eq!(output.calls[0].arguments, arguments);
    }

    #[test]
    fn granite4_streaming_emits_argument_deltas() {
        let mut parser = Granite4ToolParser::new(&test_tools());
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
    fn granite4_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = Granite4ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text, "hello ");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn granite4_streaming_extracts_multiple_tool_calls() {
        let input = format!(
            "{}{}",
            build_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            build_tool_call("add", r#"{"x":1,"y":2}"#),
        );
        let chunks = split_by_chars(&input, 7);
        let mut parser = Granite4ToolParser::new(&test_tools());

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
    fn granite4_finish_fails_incomplete_tool_call() {
        let mut parser = Granite4ToolParser::new(&test_tools());
        parser
            .parse_chunk(r#"<tool_call>{"name":"get_weather","arguments":{"location""#)
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete Granite 4 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn granite4_preserve_special_tokens_is_true() {
        let parser = Granite4ToolParser::new(&test_tools());
        assert!(parser.preserve_special_tokens());
    }
}
