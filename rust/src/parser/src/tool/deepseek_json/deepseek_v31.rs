// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{DeepSeekJsonFormat, DeepSeekJsonToolParser};
use crate::tool::{Result, StructuralTagModel, Tool, ToolParser, ToolParserOutput};

/// Tool parser for DeepSeek V3.1 raw JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
pub struct DeepSeekV31ToolParser(DeepSeekJsonToolParser);

impl DeepSeekV31ToolParser {
    /// Create a DeepSeek V3.1 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self(DeepSeekJsonToolParser::new(DeepSeekJsonFormat::V31))
    }
}

impl ToolParser for DeepSeekV31ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_model(&self) -> Option<StructuralTagModel> {
        Some(StructuralTagModel::DeepSeekV31)
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
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::DeepSeekV31ToolParser;
    use crate::tool::deepseek_json::{
        TOOL_CALL_END, TOOL_CALL_SEPARATOR, TOOL_CALL_START, TOOL_CALLS_END, TOOL_CALLS_START,
    };
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    fn v31_tool_call(function_name: &str, arguments: &str) -> String {
        format!("{TOOL_CALL_START}{function_name}{TOOL_CALL_SEPARATOR}{arguments}{TOOL_CALL_END}")
    }

    fn tool_section(tool_calls: &[String]) -> String {
        format!("{TOOL_CALLS_START}{}{TOOL_CALLS_END}", tool_calls.join(""))
    }

    #[test]
    fn deepseek_v31_parse_complete_without_tool_call_keeps_text() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn deepseek_v31_parse_complete_extracts_raw_json_arguments() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        let arguments = r#"{ "location": "Tokyo", "days": "3" }"#;
        let output = parser
            .parse_complete(&format!(
                "Let me check.{} trailing text",
                tool_section(&[v31_tool_call("get_weather", arguments)])
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "Let me check.");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn deepseek_v31_does_not_validate_or_normalize_arguments() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let output = parser
            .parse_complete(&tool_section(&[v31_tool_call("get_weather", arguments)]))
            .unwrap();

        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn deepseek_v31_streaming_emits_argument_deltas() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        let chunks = [
            TOOL_CALLS_START,
            TOOL_CALL_START,
            "get_weather",
            TOOL_CALL_SEPARATOR,
            "{\"location\":",
            "\"Beijing\"",
            "}",
            TOOL_CALL_END,
            TOOL_CALLS_END,
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

        assert_eq!(observed_arguments, ["{\"location\":", "\"Beijing\"", "}"]);
        assert_eq!(
            output.coalesce().calls()[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn deepseek_v31_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            tool_section(&[v31_tool_call("get_weather", r#"{"location":"Tokyo"}"#)])
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "hello ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn deepseek_v31_keeps_end_marker_literal_inside_json_string() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        let arguments = format!(r#"{{"text":"literal {TOOL_CALL_END} inside"}}"#);
        let input = tool_section(&[v31_tool_call("echo", &arguments)]);

        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn deepseek_v31_streaming_extracts_multiple_tool_calls() {
        let input = tool_section(&[
            v31_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            v31_tool_call("add", r#"{"x":1,"y":2}"#),
        ]);
        let chunks = split_by_chars(&input, 7);
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"location\":\"Shanghai\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "add",
                            ),
                            arguments: "{\"x\":1,\"y\":2}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn deepseek_v31_streaming_drops_eos_after_complete_tool_calls() {
        let input = format!(
            "{}<｜end▁of▁sentence｜>",
            tool_section(&[v31_tool_call("get_weather", r#"{"location":"Tokyo"}"#)])
        );
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &[&input]);

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn deepseek_v31_finish_fails_incomplete_tool_call() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        parser
            .parse_chunk(&format!(
                "{TOOL_CALLS_START}{TOOL_CALL_START}get_weather{TOOL_CALL_SEPARATOR}{{\"location\""
            ))
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete DeepSeek V3.1 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn deepseek_v31_malformed_empty_name_fails_fast() {
        let mut parser = DeepSeekV31ToolParser::new(&test_tools());
        let input = format!("{TOOL_CALLS_START}{TOOL_CALL_START}{TOOL_CALL_SEPARATOR}{{}}");

        let error = parser.parse_chunk(&input).unwrap_err();

        expect![[r#"tool parser parsing failed: near "<｜tool▁sep｜>{}": "#]]
            .assert_eq(&error.to_report_string());
    }
}
