// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::HermesToolParser;

/// Tool parser for ERNIE 4.5 XML-wrapped JSON tool calls.
///
/// Example tool call content, as rendered by the ERNIE 4.5 chat templates:
///
/// ```text
/// <tool_call>
/// {"name": "get_weather", "arguments": {"location": "Beijing"}}
/// </tool_call>
/// ```
///
/// This is the same marker-wrapped JSON shape as Hermes (repeated
/// `<tool_call>...</tool_call>` blocks for parallel calls, whitespace around the
/// JSON payload optional), so ERNIE 4.5 currently shares the Hermes parser.
///
/// The thinking-model framing that precedes tool calls (`</think>` and the
/// `<response>...</response>` answer wrapper) is stripped by the `ernie45`
/// reasoning parser, which runs before this parser in the combined pipeline.
pub type Ernie45ToolParser = HermesToolParser;

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::Ernie45ToolParser;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParser, ToolParserTestExt as _};

    /// One tool call exactly as the ERNIE 4.5 chat templates render it.
    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(
            "<tool_call>\n{{\"name\": \"{function_name}\", \"arguments\": {arguments}}}\n</tool_call>\n"
        )
    }

    #[test]
    fn ernie45_parse_complete_extracts_template_shaped_call() {
        let mut parser = Ernie45ToolParser::create(&test_tools()).unwrap();
        let output = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                r#"{"location": "Beijing"}"#,
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "\n");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, r#"{"location": "Beijing"}"#);
    }

    #[test]
    fn ernie45_streaming_extracts_parallel_calls_with_template_framing() {
        // The template separates parallel calls with blank lines.
        let input = format!(
            "{}\n\n{}",
            build_tool_call("get_weather", r#"{"location": "Shanghai"}"#),
            build_tool_call("add", r#"{"x": 1, "y": 2}"#),
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = Ernie45ToolParser::create(&test_tools()).unwrap();

        let output = collect_stream(parser.as_mut(), &chunks);

        expect![[r#"
            ToolParserOutput {
                events: [
                    Text(
                        "\n\n\n\n",
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"location\": \"Shanghai\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "add",
                            ),
                            arguments: "{\"x\": 1, \"y\": 2}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn ernie45_accepts_calls_without_framing_newlines() {
        // The model does not always reproduce the template's newlines.
        let mut parser = Ernie45ToolParser::create(&test_tools()).unwrap();
        let output = parser
            .parse_complete(
                r#"Checking.<tool_call>{"name":"get_weather","arguments":{"location":"Beijing"}}</tool_call>"#,
            )
            .unwrap();

        assert_eq!(output.normal_text(), "Checking.");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Beijing"}"#);
    }
}
