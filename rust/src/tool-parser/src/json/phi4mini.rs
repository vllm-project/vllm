use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::{Result, Tool, ToolParser, ToolParserOutput};

const PHI4MINI_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Phi4Mini",
    start_marker: "functools[",
    end_marker: "]",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: Some(","),
    name_key: "name",
    // The Python parser reads `arguments` first and falls back to `parameters`
    // (`raw_function_call["arguments"] if "arguments" in raw_function_call else
    // raw_function_call["parameters"]`). The header parser permits exactly one
    // args key per object, so this candidate order documents Python's
    // preference but does not change single-key parsing.
    arguments_key: &["arguments", "parameters"],
};

/// Tool parser for phi-4-mini models.
///
/// Example tool-call content:
///
/// ```text
/// functools[{"name": "get_weather", "arguments": {"location": "Tokyo"}}]
/// ```
///
/// phi-4-mini emits an array of tool-call objects wrapped in a `functools[..]`
/// envelope. Each object names the function with `name` and carries its
/// arguments under `arguments` (preferred) or `parameters`. Arguments are
/// already OpenAI-style JSON text, so they are streamed as raw argument deltas
/// without schema conversion or JSON normalization.
///
/// # Divergences from the Python reference
///
/// This Rust port intentionally diverges from
/// `vllm/tool_parsers/phi4mini_tool_parser.py` in these user-visible ways:
///
/// - **Streaming is supported.** Python's `extract_tool_calls_streaming`
///   unconditionally returns `None`, so the Python parser never produces
///   incremental tool-call deltas; this parser streams argument deltas through
///   the shared JSON core like the other parsers in this crate.
/// - **Preface text is preserved.** Python's non-streaming path sets
///   `content=None` whenever a `functools[..]` block is found, discarding any
///   text before the tool call; this parser emits that text as `normal_text`.
/// - **Bracket-bearing arguments are not truncated.** Python extracts the block
///   with the non-greedy regex `functools\[(.*?)\]`, which stops at the first
///   `]` and corrupts any argument value containing `]` (e.g. `{"items":[1,2]}`),
///   raising a `JSONDecodeError` and dropping the call; this parser scans
///   matched braces so such arguments are forwarded intact.
/// - **Truncated tool calls error rather than silently dropping.** Python's
///   non-streaming path returns the raw output unchanged on `JSONDecodeError`;
///   this parser returns an `incomplete Phi4Mini tool call` error from
///   `finish()`, matching the other JSON parsers in this crate.
///
/// # Known unaddressed divergences (TODO)
///
/// These Python behaviors are NOT matched because they require non-local
/// changes to the shared `JsonToolCallParser` core (which would also affect
/// Hermes / InternLM2 / Llama / Mistral / Qwen):
///
/// - **Arguments value type.** The shared core requires the arguments value to
///   be a JSON object. Python's `json.dumps(...)` round-trips arrays, strings,
///   numbers, and `null`, so a model emitting `"arguments":null` hard-fails
///   under Rust.
/// - **Field order independence.** The header parser requires `name` before the
///   arguments key; Python's `dict` access is order-independent, so an object
///   emitting the arguments key first parses in Python but fails here.
/// - **Both keys present.** When one object carries both `arguments` and
///   `parameters`, Python prefers `arguments`; this parser accepts the first
///   key encountered and rejects the trailing one as a syntax error.
/// - **Empty array.** `functools[]` with no tool-call objects errors here,
///   because the shared core requires an object after the start marker. Python
///   instead reports `tools_called=true` with no calls — a case its own parity
///   test (`tests/tool_parsers/test_phi4mini_tool_parser.py`) marks `xfail` as
///   a known bug.
pub struct Phi4MiniJsonToolParser {
    inner: JsonToolCallParser,
}

impl Phi4MiniJsonToolParser {
    /// Create a phi-4-mini tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(PHI4MINI_CONFIG),
        }
    }
}

impl ToolParser for Phi4MiniJsonToolParser {
    /// Create a boxed phi-4-mini tool parser.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Feed one decoded text chunk through the phi-4-mini parser.
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

    use super::Phi4MiniJsonToolParser;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::{ToolParser, ToolParserTestExt as _};

    /// Build one phi-4-mini tool-call object: `{"name":..,"<args_key>":<args>}`.
    fn build_call(function_name: &str, args_key: &str, arguments: &str) -> String {
        format!(r#"{{"name":"{function_name}","{args_key}":{arguments}}}"#)
    }

    /// Wrap tool-call objects in the `functools[..]` envelope.
    fn wrap(calls: &[String]) -> String {
        format!("functools[{}]", calls.join(","))
    }

    #[test]
    fn phi4mini_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn phi4mini_parse_complete_extracts_arguments_key() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo","days":"3"}"#;
        let result = parser
            .parse_complete(&wrap(&[build_call("get_weather", "arguments", arguments)]))
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn phi4mini_parse_complete_falls_back_to_parameters_key() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo"}"#;
        let result = parser
            .parse_complete(&wrap(&[build_call("get_weather", "parameters", arguments)]))
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn phi4mini_extracts_multiple_comma_delimited_calls() {
        let input = wrap(&[
            build_call("get_weather", "arguments", r#"{"location":"Shanghai"}"#),
            build_call("add", "arguments", r#"{"x":1,"y":2}"#),
        ]);
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());

        let result = parser.parse_complete(&input).unwrap();

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
        .assert_debug_eq(&result);
    }

    /// Python's `functools\[(.*?)\]` regex stops at the FIRST `]`, so an
    /// array-valued argument such as `{"items":[1,2]}` makes the Python parser
    /// raise `JSONDecodeError` and emit no tool call. The shared JSON core
    /// scans matched braces, so this port forwards array-valued arguments
    /// intact.
    #[test]
    fn phi4mini_array_valued_arguments_are_not_truncated() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let arguments = r#"{"items":[1,2],"flag":true}"#;
        let result = parser
            .parse_complete(&wrap(&[build_call("convert", "arguments", arguments)]))
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, arguments);
    }

    /// Divergence from Python: the Python parser sets `content=None` when a
    /// tool call is found, discarding any preface text. This port preserves
    /// preface text as normal_text, consistent with the other JSON parsers in
    /// this crate.
    #[test]
    fn phi4mini_preserves_text_before_tool_call() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let input = format!(
            "Let me check.\n{}",
            wrap(&[build_call(
                "get_weather",
                "arguments",
                r#"{"location":"Tokyo"}"#
            )])
        );

        let result = parser.parse_complete(&input).unwrap();

        assert_eq!(result.normal_text, "Let me check.\n");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn phi4mini_does_not_validate_or_normalize_arguments() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let result = parser
            .parse_complete(&wrap(&[build_call("get_weather", "arguments", arguments)]))
            .unwrap();

        assert_eq!(result.calls[0].arguments, arguments);
    }

    /// The bundled `tool_chat_template_phi4_mini.jinja` emits objects with
    /// whitespace after `:` and `,` (e.g. `{"name": "f", "arguments": {..}}`).
    /// Confirm the parser handles that real model format and preserves the
    /// inner argument spacing verbatim.
    #[test]
    fn phi4mini_accepts_real_model_whitespace_format() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let input = r#"functools[{"name": "get_weather", "arguments": {"location": "Tokyo"}}]"#;

        let result = parser.parse_complete(input).unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, r#"{"location": "Tokyo"}"#);
    }

    /// Python returns `None` from `extract_tool_calls_streaming` (no streaming
    /// support). This port streams argument deltas through the shared JSON
    /// core — an intentional capability improvement over the Python reference.
    #[test]
    fn phi4mini_streaming_emits_argument_deltas() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let chunks = [
            "preface functo",
            "ols[",
            r#"{"name":"get_weather","arguments":"#,
            r#"{"location":"#,
            r#""Beijing""#,
            r#"}"#,
            r#"}]"#,
            " suffix",
        ];

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text, "preface  suffix");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, r#"{"location":"Beijing"}"#);
    }

    #[test]
    fn phi4mini_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            wrap(&[build_call(
                "get_weather",
                "arguments",
                r#"{"location":"Tokyo"}"#
            )])
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text, "hello ");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn phi4mini_finish_errors_on_truncated_tool_call() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let _ = parser
            .parse_chunk(r#"functools[{"name":"get_weather","arguments":{"location""#)
            .unwrap();
        let error = parser.finish().unwrap_err();

        assert!(
            error.to_report_string().contains("incomplete Phi4Mini tool call"),
            "finish() reports the truncated tool call as incomplete: {}",
            error.to_report_string(),
        );
    }

    #[test]
    fn phi4mini_preserve_special_tokens_is_false() {
        let parser = Phi4MiniJsonToolParser::new(&test_tools());
        assert!(!parser.preserve_special_tokens());
    }

    /// Python's regex parser `xfail`s nested arrays/objects in arguments
    /// (`tests/tool_parsers/test_phi4mini_tool_parser.py` flags
    /// `test_various_data_types` as a known nesting bug). The brace-scanning
    /// core handles them, so this port parses what the Python parser cannot.
    #[test]
    fn phi4mini_parses_nested_arrays_and_objects() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let arguments =
            r#"{"array_field":["a","b","c"],"object_field":{"nested":"value"},"empty_object":{}}"#;
        let result = parser
            .parse_complete(&wrap(&[build_call("convert", "arguments", arguments)]))
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("convert"));
        assert_eq!(result.calls[0].arguments, arguments);
    }

    /// The chat template emits parallel calls as `},\n  {` (comma + newline +
    /// indent). Confirm the `Optional` marker whitespace and `,` delimiter
    /// parse the real multi-call layout.
    #[test]
    fn phi4mini_parses_parallel_calls_in_template_format() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let input = concat!(
            "functools[\n",
            "  {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}},\n",
            "  {\"name\": \"add\", \"arguments\": {\"x\": 1, \"y\": 2}}\n",
            "]"
        );

        let result = parser.parse_complete(input).unwrap();

        assert_eq!(result.calls.len(), 2);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[1].name.as_deref(), Some("add"));
    }

    /// Divergence from Python: an empty `functools[]` array errors here (the
    /// shared core requires an object after the start marker), whereas Python
    /// reports `tools_called=true` with no calls — a case its own parity test
    /// flags as a known bug. See the struct-level docs.
    #[test]
    fn phi4mini_empty_array_errors() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let error = parser.parse_complete("functools[]").unwrap_err();

        assert!(
            error.to_report_string().contains("invalid Phi4Mini"),
            "empty functools[] should error: {}",
            error.to_report_string(),
        );
    }
}
