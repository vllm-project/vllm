use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::{Result, Tool, ToolParser, ToolParserOutput};

const JAMBA_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Jamba",
    start_marker: "<tool_calls>[",
    end_marker: "]</tool_calls>",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: Some(","),
    name_key: "name",
    arguments_key: &["arguments"],
};

/// Tool parser for Jamba special-token wrapped JSON-array tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_calls>[
///     {"name": "get_current_weather", "arguments": {"city": "Dallas"}}
/// ]</tool_calls>
/// ```
///
/// Jamba wraps a JSON array of `{"name", "arguments"}` objects between the
/// `<tool_calls>` and `</tool_calls>` special tokens. The array's opening `[`
/// and closing `]` are folded into the start and end markers so the shared
/// JSON tool-call core can stream each array element as one tool call, the same
/// way the Mistral parser handles `[TOOL_CALLS] [...]`.
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
///
/// # Divergences from the Python reference
///
/// This Rust port intentionally diverges from
/// `vllm/tool_parsers/jamba_tool_parser.py` in these user-visible ways:
///
/// - **Arguments bytes are forwarded verbatim.** Python re-serializes each
///   `arguments` object with `json.dumps(...)`, which can reorder keys and
///   normalize spacing; this parser streams the raw JSON object text the model
///   emitted, matching the other JSON parsers in this crate (Mistral, Hermes).
/// - **End-marker bytes inside JSON string values are preserved.** Python
///   slices on `split("<tool_calls>")[-1].split("</tool_calls>")[0]`, which
///   truncates regardless of JSON context; this parser scans matched braces and
///   quotes so a literal `]</tool_calls>` inside an arguments string is
///   forwarded intact.
/// - **Truncated tool calls error rather than silently dropping.** Python's
///   streaming wrapper swallows mid-stream errors with
///   `except Exception: return None`; this parser returns an
///   `incomplete Jamba tool call` error from `finish()`, matching the other
///   JSON parsers and Python's non-streaming `json.loads` behavior.
///
/// # Known unaddressed divergences (TODO)
///
/// These stem from shared `JsonToolCallParser` core limitations also documented
/// on the InternLM2 parser, and would require non-local core changes that would
/// affect Hermes / Llama / Mistral / Qwen as well:
///
/// - **Arguments value type.** The shared core requires the `arguments` value to
///   be a JSON object (`take_json_object` rejects anything not starting with
///   `{`). Python's `json.dumps` round-trips `null`, arrays, strings, and numbers
///   verbatim, so a model emitting `"arguments":null` hard-fails here.
/// - **Field order independence.** The header parser requires the JSON keys in
///   the order `name` then `arguments`. Python's `json.loads` + dict access is
///   order-independent, so a model emitting `{"arguments":{...},"name":"foo"}`
///   parses in Python but fails here.
pub struct JambaToolParser {
    inner: JsonToolCallParser,
}

impl JambaToolParser {
    /// Create a Jamba tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(JAMBA_CONFIG),
        }
    }
}

impl ToolParser for JambaToolParser {
    /// Create a boxed Jamba tool parser.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Preserve special-token markers while decoding, since `<tool_calls>` and
    /// `</tool_calls>` are tokenizer special tokens in Jamba models (the Python
    /// parser sets `skip_special_tokens = False` for the same reason).
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    /// Feed one decoded text chunk through the Jamba parser.
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.inner.parse_into(chunk, output)
    }

    /// Flush any buffered partial state at end of stream.
    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.inner.finish()
    }

    /// Clear parser state and return currently uncommitted buffered text.
    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::JambaToolParser;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(r#"{{"name":"{function_name}","arguments":{arguments}}}"#)
    }

    fn build_tool_calls(tool_calls: &[String]) -> String {
        format!("<tool_calls>[{}]</tool_calls>", tool_calls.join(","))
    }

    #[test]
    fn jamba_parse_complete_without_tool_call_keeps_text() {
        let mut parser = JambaToolParser::new(&test_tools());
        let output = parser.parse_complete("This is a test").unwrap();

        assert_eq!(output.normal_text, "This is a test");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn jamba_parse_complete_extracts_single_tool_call() {
        let mut parser = JambaToolParser::new(&test_tools());
        let arguments = r#"{"city":"Dallas","state":"TX","unit":"fahrenheit"}"#;
        let output = parser
            .parse_complete(&build_tool_calls(&[build_tool_call(
                "get_current_weather",
                arguments,
            )]))
            .unwrap();

        assert!(output.normal_text.is_empty());
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].tool_index, 0);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_current_weather"));
        assert_eq!(output.calls[0].arguments, arguments);
    }

    #[test]
    fn jamba_parse_complete_preserves_leading_content() {
        let mut parser = JambaToolParser::new(&test_tools());
        let arguments = r#"{"city":"Dallas"}"#;
        let output = parser
            .parse_complete(&format!(
                "Sure! let me call the tool for you.{}",
                build_tool_calls(&[build_tool_call("get_current_weather", arguments)])
            ))
            .unwrap();

        assert_eq!(output.normal_text, "Sure! let me call the tool for you.");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_current_weather"));
        assert_eq!(output.calls[0].arguments, arguments);
    }

    #[test]
    fn jamba_parse_complete_extracts_pretty_multiple_tool_calls() {
        let mut parser = JambaToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<tool_calls>[\n    {\"name\": \"get_current_weather\", \"arguments\": {\"city\": \"Dallas\", \"state\": \"TX\"}},\n    {\"name\": \"get_current_weather\", \"arguments\": {\"city\": \"Orlando\", \"state\": \"FL\"}}\n]</tool_calls>",
            )
            .unwrap();

        expect![[r#"
            ToolParserOutput {
                normal_text: "",
                calls: [
                    ToolCallDelta {
                        tool_index: 0,
                        name: Some(
                            "get_current_weather",
                        ),
                        arguments: "{\"city\": \"Dallas\", \"state\": \"TX\"}",
                    },
                    ToolCallDelta {
                        tool_index: 1,
                        name: Some(
                            "get_current_weather",
                        ),
                        arguments: "{\"city\": \"Orlando\", \"state\": \"FL\"}",
                    },
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn jamba_does_not_validate_or_normalize_arguments() {
        let mut parser = JambaToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let output = parser
            .parse_complete(&build_tool_calls(&[build_tool_call(
                "get_weather",
                arguments,
            )]))
            .unwrap();

        assert_eq!(output.calls[0].arguments, arguments);
    }

    #[test]
    fn jamba_streaming_emits_argument_deltas() {
        let mut parser = JambaToolParser::new(&test_tools());
        let chunks = [
            "preface <tool",
            "_calls>[{\"name\":\"get_weather\",\"arguments\":",
            "{\"location\":",
            "\"Beijing\"",
            "}",
            "}]</tool_calls> suffix",
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
    fn jamba_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            build_tool_calls(&[build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)])
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = JambaToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text, "hello ");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn jamba_keeps_end_marker_literal_inside_json_string() {
        let mut parser = JambaToolParser::new(&test_tools());
        let arguments = r#"{"text":"literal ]</tool_calls> inside"}"#;
        let output = parser
            .parse_complete(&build_tool_calls(&[build_tool_call("echo", arguments)]))
            .unwrap();

        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].arguments, arguments);
    }

    #[test]
    fn jamba_finish_fails_incomplete_tool_call() {
        let mut parser = JambaToolParser::new(&test_tools());
        parser
            .parse_chunk(r#"<tool_calls>[{"name":"get_weather","arguments":{"location""#)
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete Jamba tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn jamba_malformed_field_order_fails_fast() {
        let mut parser = JambaToolParser::new(&test_tools());
        let error = parser
            .parse_chunk(r#"<tool_calls>[{"arguments":{},"name":"get_weather"}]</tool_calls>"#)
            .unwrap_err();

        expect![[r#"
            tool parser parsing failed: invalid Jamba
            expected `name`"#]]
        .assert_eq(&error.to_report_string());
    }

    #[test]
    fn jamba_preserve_special_tokens_is_true() {
        let parser = JambaToolParser::new(&test_tools());
        assert!(parser.preserve_special_tokens());
    }
}
