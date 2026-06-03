use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::{Result, Tool, ToolParser, ToolParserOutput};

const INTERNLM2_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "InternLM2",
    start_marker: "<|action_start|><|plugin|>",
    end_marker: "<|action_end|>",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: None,
    name_key: "name",
    // The Python parser's `get_arguments()` accepts either `parameters` or
    // `arguments` and prefers `parameters` when both are present. This Rust
    // parser uses first-encountered semantics because the header parser only
    // permits one args key per tool-call object; if a future model emits
    // both keys in the same object, the Rust port will accept the first one
    // and reject the trailing one as a syntax error rather than silently
    // shadowing it.
    arguments_key: &["parameters", "arguments"],
};

/// Tool parser for InternLM2 special-token wrapped JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <|action_start|><|plugin|>{"name": "get_weather", "parameters": {"location":"Tokyo"}}<|action_end|>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
///
/// # Divergences from the Python reference
///
/// This Rust port intentionally diverges from
/// `vllm/tool_parsers/internlm2_tool_parser.py` in two user-visible ways:
///
/// - **Parallel tool calls are supported.** Python silently drops every `<|action_start|>` block
///   after the first (`current_tool_id > 0` returns an empty delta); this parser emits every
///   well-formed block with incrementing `tool_index`. Models that legitimately emit multiple
///   action blocks therefore produce more tool calls under Rust than under Python.
/// - **End-marker bytes inside JSON string values are preserved.** Python does
///   `action.split("<|action_end|>")[0]` which truncates regardless of JSON context; this parser
///   scans matched braces and quotes so a literal `<|action_end|>` inside an arguments string is
///   forwarded intact.
/// - **Only whitespace is allowed before the `{`.** Python's non-streaming
///   `action[action.find("{"):]` drops any bytes before the first `{`, but its streaming path has
///   no equivalent and the model format always emits `<|plugin|>{...`; this parser allows only
///   whitespace there, matching the other JSON parsers in this crate.
/// - **Truncated tool calls error rather than silently dropping.** Python's streaming wrapper
///   swallows mid-stream errors with `except Exception: return None` (logging a traceback) while
///   its non-streaming path raises `JSONDecodeError`; this parser returns an `incomplete InternLM2
///   tool call` error from `finish()`, matching the other JSON parsers and Python's non-streaming
///   behavior.
///
/// # Known unaddressed divergences (TODO)
///
/// The following Python behaviors are NOT yet matched. They are deferred to
/// follow-up work because they require non-local changes to the shared
/// `JsonToolCallParser` core that would affect Hermes / Llama / Mistral /
/// Qwen as well. If a real-world InternLM2 deployment hits one of these,
/// prioritize the corresponding fix.
///
/// - **Arguments value type.** The shared core requires the arguments value to be a JSON object
///   (`take_json_object` rejects anything not starting with `{`). Python's
///   `json.dumps(action_dict.get("parameters", ...))` accepts `null`, arrays, strings, and numbers
///   and round-trips them verbatim. Models that legitimately emit `"parameters":null` will hard-
///   fail under Rust.
/// - **Unknown arguments key.** Python falls back to `{}` via `action_dict.get("parameters",
///   action_dict.get("arguments", {}))` when neither key is present; the Rust header parser raises
///   `parsing failed: invalid InternLM2` for any unrecognized key. A model that emits a typo (e.g.
///   `"params"`) breaks the whole response.
/// - **Field order independence.** The header parser requires the JSON keys to appear in the order
///   `name` then arguments key. Python's `json.loads` + `dict.get` is order-independent, so a model
///   emitting `{"parameters":{...},"name":"foo"}` parses in Python but fails in Rust.
pub struct Internlm2ToolParser {
    inner: JsonToolCallParser,
}

impl Internlm2ToolParser {
    /// Create an InternLM2 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(INTERNLM2_CONFIG),
        }
    }
}

impl ToolParser for Internlm2ToolParser {
    /// Create a boxed InternLM2 tool parser.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Preserve special-token markers while decoding, since
    /// `<|action_start|>`, `<|plugin|>`, and `<|action_end|>` are tokenizer
    /// special tokens in InternLM2 models.
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    /// Feed one decoded text chunk through the InternLM2 parser.
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

    use super::Internlm2ToolParser;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    const ACTION_START: &str = "<|action_start|><|plugin|>";
    const ACTION_END: &str = "<|action_end|>";

    fn build_tool_call(function_name: &str, args_key: &str, arguments: &str) -> String {
        format!(
            r#"{ACTION_START}{{"name":"{function_name}","{args_key}":{arguments}}}{ACTION_END}"#
        )
    }

    #[test]
    fn internlm2_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Internlm2ToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn internlm2_parse_complete_extracts_parameters_key() {
        let mut parser = Internlm2ToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo","days":"3"}"#;
        let result = parser
            .parse_complete(&format!(
                "Let me check.\n{}",
                build_tool_call("get_weather", "parameters", arguments)
            ))
            .unwrap();

        assert_eq!(result.normal_text, "Let me check.\n");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn internlm2_parse_complete_extracts_arguments_key_fallback() {
        let mut parser = Internlm2ToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo"}"#;
        let result = parser
            .parse_complete(&build_tool_call("get_weather", "arguments", arguments))
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn internlm2_accepts_whitespace_after_plugin_marker() {
        let mut parser = Internlm2ToolParser::new(&test_tools());
        let result = parser
            .parse_complete(&format!(
                r#"{ACTION_START}
{{"name":"get_weather","parameters":{{}}}}{ACTION_END}"#
            ))
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn internlm2_does_not_validate_or_normalize_arguments() {
        let mut parser = Internlm2ToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let result = parser
            .parse_complete(&build_tool_call("get_weather", "parameters", arguments))
            .unwrap();

        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn internlm2_streaming_emits_argument_deltas() {
        let mut parser = Internlm2ToolParser::new(&test_tools());
        let chunks = [
            "preface <|action",
            "_start|><|plugin|>",
            r#"{"name":"get_weather","parameters":"#,
            r#"{"location":"#,
            r#""Beijing""#,
            r#"}"#,
            r#"}<|action_end|> suffix"#,
        ];

        let mut result = ToolParserOutput::default();
        let mut observed_arguments = Vec::new();
        for chunk in chunks {
            let next = parser.parse_chunk(chunk).unwrap();
            observed_arguments.extend(
                next.calls
                    .iter()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            result.append(next);
        }
        result.append(parser.finish().unwrap());

        assert_eq!(
            observed_arguments,
            [r#"{"location":"#, r#""Beijing""#, r#"}"#]
        );
        assert_eq!(result.normal_text, "preface  suffix");
        assert_eq!(
            result.coalesce_calls().calls[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn internlm2_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            build_tool_call("get_weather", "parameters", r#"{"location":"Tokyo"}"#)
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = Internlm2ToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text, "hello ");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn internlm2_streaming_extracts_multiple_blocks() {
        let input = format!(
            "{}{}",
            build_tool_call("get_weather", "parameters", r#"{"location":"Shanghai"}"#),
            build_tool_call("add", "arguments", r#"{"x":1,"y":2}"#),
        );
        let chunks = split_by_chars(&input, 7);
        let mut parser = Internlm2ToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

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

    #[test]
    fn internlm2_keeps_end_marker_literal_inside_json_string() {
        let mut parser = Internlm2ToolParser::new(&test_tools());
        let arguments = format!(r#"{{"text":"literal {ACTION_END} inside"}}"#);
        let input = build_tool_call("echo", "parameters", &arguments);

        let result = parser.parse_complete(&input).unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn internlm2_finish_errors_on_truncated_tool_call() {
        let mut parser = Internlm2ToolParser::new(&test_tools());
        let pre_finish = parser
            .parse_chunk(&format!(
                r#"{ACTION_START}{{"name":"get_weather","parameters":{{"location""#
            ))
            .unwrap();
        let error = parser.finish().unwrap_err();

        assert_eq!(
            pre_finish.calls[0].name.as_deref(),
            Some("get_weather"),
            "name delta is still emitted from parse_chunk() before truncation",
        );
        assert!(
            error.to_report_string().contains("incomplete InternLM2 tool call"),
            "finish() reports the truncated tool call as incomplete: {}",
            error.to_report_string(),
        );
    }

    #[test]
    fn internlm2_unknown_arguments_key_fails() {
        let mut parser = Internlm2ToolParser::new(&test_tools());
        let input = build_tool_call("get_weather", "params", r#"{"location":"Tokyo"}"#);

        let error = parser.parse_chunk(&input).unwrap_err();

        expect![[r#"
            tool parser parsing failed: invalid InternLM2
            expected `parameters`, `arguments`"#]]
        .assert_eq(&error.to_report_string());
    }

    #[test]
    fn internlm2_preserve_special_tokens_is_true() {
        let parser = Internlm2ToolParser::new(&test_tools());
        assert!(parser.preserve_special_tokens());
    }
}
