// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::tool::{
    Result, StructuralTagBuilder, Tool, ToolParser, ToolParserEvent, ToolParserOutput,
};

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
/// The template closes every tool call with a newline and separates calls with
/// a blank line; those newlines are framing and are dropped from the text
/// stream. The thinking-model framing that precedes tool calls (`</think>` and
/// the `<response>...</response>` answer wrapper) is stripped by the `ernie45`
/// reasoning parser, which runs before this parser in the combined pipeline.
pub struct Ernie45ToolParser {
    inner: JsonToolCallParser,
    /// Whether the text that follows is right after a tool call, so that its
    /// leading newlines are framing.
    after_tool_call: bool,
}

impl Ernie45ToolParser {
    /// Create an ERNIE 4.5 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(ERNIE45_CONFIG),
            after_tool_call: false,
        }
    }

    /// Drop the newlines framing the text that directly follows a tool call
    /// from the events the inner parser just committed.
    fn drop_framing_after_calls(&mut self, output: &mut ToolParserOutput) {
        let mut index = 0;
        while index < output.events.len() {
            let remove = match &mut output.events[index] {
                ToolParserEvent::ToolCall(_) => {
                    self.after_tool_call = true;
                    false
                }
                ToolParserEvent::Text(text) if self.after_tool_call => {
                    let framing_len = text.len() - text.trim_start_matches('\n').len();
                    text.replace_range(..framing_len, "");
                    if text.is_empty() {
                        true
                    } else {
                        self.after_tool_call = false;
                        false
                    }
                }
                ToolParserEvent::Text(_) => false,
            };
            if remove {
                output.events.remove(index);
            } else {
                index += 1;
            }
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
        // Filter the newly committed events on their own: the caller's output
        // may merge new text into an event it already holds.
        let mut committed = ToolParserOutput::default();
        let result = self.inner.parse_into(chunk, &mut committed);
        self.drop_framing_after_calls(&mut committed);
        output.append(committed);
        result
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = self.inner.finish()?;
        self.drop_framing_after_calls(&mut output);
        self.after_tool_call = false;
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.after_tool_call = false;
        self.inner.reset()
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::Ernie45ToolParser;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParser as _, ToolParserOutput, ToolParserTestExt as _};

    #[test]
    fn ernie45_drops_framing_merged_into_committed_text() {
        // Text the caller already holds is left untouched, and the framing
        // newline is still dropped even though the new text merges into that
        // committed event.
        let mut parser = Ernie45ToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();
        parser
            .parse_into(
                "<tool_call>\n{\"name\": \"add\", \"arguments\": {}}\n</tool_call>",
                &mut output,
            )
            .unwrap();
        output.push_text("committed");

        parser.parse_into("\nDone.", &mut output).unwrap();
        output.append(parser.finish().unwrap());

        let output = output.coalesce();
        assert_eq!(output.normal_text(), "committedDone.");
        assert_eq!(output.calls().len(), 1);
    }

    /// One tool call exactly as the ERNIE 4.5 chat template renders it,
    /// including the blank line that precedes it.
    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(
            "\n\n\n<tool_call>\n{{\"name\": \"{function_name}\", \"arguments\": {arguments}}}\n</tool_call>\n"
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

        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"location\": \"Beijing\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn ernie45_streaming_extracts_parallel_calls_without_leaking_framing() {
        // The template renders `</tool_call>\n` + `\n` + `\n<tool_call>` between
        // calls and `</tool_call>\n` before `<|im_end|>`.
        let input = format!(
            "{}{}",
            build_tool_call("get_weather", r#"{"location": "Shanghai"}"#),
            build_tool_call("add", r#"{"x": 1, "y": 2}"#),
        );
        for chunk_chars in [1, 2, 5, usize::MAX] {
            let chunks = split_by_chars(&input, chunk_chars);
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

        expect![[r#"
            ToolParserOutput {
                events: [
                    Text(
                        "Checking.",
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"location\":\"Beijing\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn ernie45_drops_single_newline_between_calls() {
        // Fewer separator newlines than the template renders are still framing.
        let input = "<tool_call>\n{\"name\": \"add\", \"arguments\": {\"x\": 1}}\n</tool_call>\n<tool_call>\n{\"name\": \"add\", \"arguments\": {\"x\": 2}}\n</tool_call>";
        for chunk_chars in [1, 3, usize::MAX] {
            let chunks = split_by_chars(input, chunk_chars);
            let mut parser = Ernie45ToolParser::new(&test_tools());

            let output = collect_stream(&mut parser, &chunks);

            assert_eq!(output.normal_text(), "", "chunk size {chunk_chars}");
            assert_eq!(output.calls().len(), 2, "chunk size {chunk_chars}");
        }
    }

    #[test]
    fn ernie45_keeps_text_after_tool_calls() {
        // Only the framing newlines are dropped; other trailing text is kept.
        let mut parser = Ernie45ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<tool_call>\n{\"name\": \"add\", \"arguments\": {\"x\": 1}}\n</tool_call>\n\nDone.\n",
            )
            .unwrap();

        assert_eq!(output.normal_text(), "Done.\n");
        assert_eq!(output.calls().len(), 1);
    }
}
