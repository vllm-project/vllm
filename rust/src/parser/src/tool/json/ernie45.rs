// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

const ERNIE45_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "ERNIE 4.5",
    start_marker: "<tool_call>",
    // The chat template separates the answer and each tool call with a blank
    // line, so the framed marker keeps that framing out of the text stream.
    framed_start_marker: Some("\n\n\n<tool_call>"),
    end_marker: "</tool_call>",
    // The Python parser matches `\s*` around the payload, so the newlines the
    // template renders inside the markers are optional here too.
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: None,
    name_key: "name",
    arguments_key: &["arguments"],
};

/// Tool parser for ERNIE 4.5 XML-wrapped JSON tool calls.
///
/// Example tool call content, as rendered by the ERNIE 4.5 chat template:
///
/// ```text
/// <tool_call>
/// {"name": "get_weather", "arguments": {"location": "Beijing"}}
/// </tool_call>
/// ```
///
/// Parallel calls are repeated `<tool_call>...</tool_call>` blocks. Arguments
/// are already OpenAI-style JSON text, so they are streamed as raw argument
/// deltas without schema conversion or JSON normalization.
///
/// The thinking-model framing that precedes tool calls (`</think>` and the
/// `<response>...</response>` answer wrapper) is stripped by the `ernie45`
/// reasoning parser, which runs before this parser in the combined pipeline.
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
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        // ERNIE 4.5 emits the same `<tool_call>{"name": ..., "arguments": ...}`
        // shape that the Hermes structural tag constrains.
        Some(xgrammar_structural_tag::Model::Hermes.builder())
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

    use super::Ernie45ToolParser;
    use crate::tool::ToolParserTestExt as _;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};

    /// One tool call exactly as the ERNIE 4.5 chat template renders it.
    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(
            "\n\n\n<tool_call>\n{{\"name\": \"{function_name}\", \"arguments\": {arguments}}}\n</tool_call>"
        )
    }

    #[test]
    fn ernie45_parse_complete_extracts_template_shaped_call() {
        let mut parser = Ernie45ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                r#"{"location": "Beijing"}"#,
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, r#"{"location": "Beijing"}"#);
    }

    #[test]
    fn ernie45_streaming_extracts_parallel_calls_without_leaking_framing() {
        let input = format!(
            "{}{}",
            build_tool_call("get_weather", r#"{"location": "Shanghai"}"#),
            build_tool_call("add", r#"{"x": 1, "y": 2}"#),
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = Ernie45ToolParser::new(&test_tools());

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
        // The model does not always reproduce the template's framing.
        let mut parser = Ernie45ToolParser::new(&test_tools());
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
