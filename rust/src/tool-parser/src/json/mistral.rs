use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::{Result, Tool, ToolParseResult, ToolParser};

const MISTRAL_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Mistral",
    start_marker: "[TOOL_CALLS] [",
    end_marker: "]",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: Some(","),
    name_key: "name",
    arguments_key: "arguments",
};

/// Tool parser for Mistral JSON-array tool calls.
///
/// Example tool call content:
///
/// ```text
/// [TOOL_CALLS] [{"name": "get_weather", "arguments": {"location":"Tokyo"}}]
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
pub struct MistralToolParser {
    inner: JsonToolCallParser,
}

impl MistralToolParser {
    /// Create a Mistral tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(MISTRAL_CONFIG),
        }
    }
}

impl ToolParser for MistralToolParser {
    /// Create a boxed Mistral tool parser.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Push one decoded text chunk through the Mistral parser.
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

    use super::MistralToolParser;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::{ToolParseResult, ToolParser};

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(r#"{{"name":"{function_name}","arguments":{arguments}}}"#)
    }

    fn build_tool_calls(tool_calls: &[String]) -> String {
        format!("[TOOL_CALLS] [{}]", tool_calls.join(","))
    }

    #[test]
    fn mistral_parse_complete_without_tool_call_keeps_text() {
        let mut parser = MistralToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn mistral_parse_complete_extracts_raw_json_arguments() {
        let mut parser = MistralToolParser::new(&test_tools());
        let arguments = r#"{ "location": "Tokyo", "days": "3" }"#;
        let result = parser
            .parse_complete(&format!(
                "Let me check.\n{}",
                build_tool_calls(&[build_tool_call("get_weather", arguments)])
            ))
            .unwrap();

        assert_eq!(result.normal_text, "Let me check.\n");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn mistral_parse_complete_extracts_pretty_multiple_tool_calls() {
        let mut parser = MistralToolParser::new(&test_tools());
        let result = parser
            .parse_complete(
                r#"I'll help.
[TOOL_CALLS] [
    {"name": "get_weather", "arguments": {"city": "Tokyo", "units": "celsius"}}
    ,
    {"name": "add", "arguments": {"x": 1, "y": 2}}
]"#,
            )
            .unwrap();

        expect![[r#"
            ToolParseResult {
                normal_text: "I'll help.\n",
                calls: [
                    ToolCallDelta {
                        tool_index: 0,
                        name: Some(
                            "get_weather",
                        ),
                        arguments: "{\"city\": \"Tokyo\", \"units\": \"celsius\"}",
                    },
                    ToolCallDelta {
                        tool_index: 1,
                        name: Some(
                            "add",
                        ),
                        arguments: "{\"x\": 1, \"y\": 2}",
                    },
                ],
            }
        "#]]
        .assert_debug_eq(&result);
    }

    #[test]
    fn mistral_does_not_validate_or_normalize_arguments() {
        let mut parser = MistralToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let result = parser
            .parse_complete(&build_tool_calls(&[build_tool_call(
                "get_weather",
                arguments,
            )]))
            .unwrap();

        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn mistral_streaming_emits_argument_deltas() {
        let mut parser = MistralToolParser::new(&test_tools());
        let chunks = [
            "preface [TOOL",
            "_CALLS] [{\"name\":\"get_weather\",\"arguments\":",
            "{\"location\":",
            "\"Beijing\"",
            "}",
            "}] suffix",
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
        assert_eq!(result.normal_text, "preface  suffix");
        assert_eq!(
            result.coalesce_calls().calls[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn mistral_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            build_tool_calls(&[build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)])
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = MistralToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text, "hello ");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn mistral_keeps_array_bracket_literal_inside_json_string() {
        let mut parser = MistralToolParser::new(&test_tools());
        let arguments = r#"{"text":"Array notation: arr[0] = value[1]"}"#;
        let result = parser
            .parse_complete(&build_tool_calls(&[build_tool_call("echo", arguments)]))
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn mistral_finish_fails_incomplete_tool_call() {
        let mut parser = MistralToolParser::new(&test_tools());
        parser
            .push(r#"[TOOL_CALLS] [{"name":"get_weather","arguments":{"location""#)
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete Mistral tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn mistral_malformed_field_order_fails_fast() {
        let mut parser = MistralToolParser::new(&test_tools());
        let error = parser
            .push(r#"[TOOL_CALLS] [{"arguments":{},"name":"get_weather"}]"#)
            .unwrap_err();

        expect![[r#"
            tool parser parsing failed: invalid Mistral
            expected `name`"#]]
        .assert_eq(&error.to_report_string());
    }
}
