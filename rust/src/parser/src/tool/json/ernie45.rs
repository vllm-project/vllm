// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use xgrammar_structural_tag::format::Format;

use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::output_grammar::{self, OutputGrammarContext};
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

    fn build_visible_format(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<Format>> {
        let format =
            output_grammar::visible_format_from_builder(self.structural_tag_builder(), ctx)?;
        Ok(format.map(with_template_framing))
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

/// Allow the newlines the ERNIE 4.5 template renders around forced tool calls.
///
/// The Hermes tags for `required` and named tool choices must start right at
/// `<tool_call>` and end right after `</tool_call>`, while the model puts a
/// newline after `</think>`, after every `</tool_call>` and between calls.
/// Masking those newlines blocks the model's natural `\n` + EOS ending; under
/// `required` it then prefers another `<tool_call>` to EOS and repeats calls
/// until `max_tokens`.
fn with_template_framing(format: Format) -> Format {
    let Format::TagsWithSeparator(calls) = format else {
        return format;
    };
    if !calls.at_least_one {
        return Format::TagsWithSeparator(calls);
    }
    let newlines = || Format::star(Format::const_string("\n"));
    let call = Format::tags_with_separator(calls.tags, "", true, true);
    let mut elements = vec![newlines(), call.clone()];
    if !calls.stop_after_first {
        elements.push(Format::star(Format::sequence(vec![newlines(), call])));
    }
    elements.push(newlines());
    Format::sequence(elements)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use xgrammar_structural_tag::ToolChoice;

    use super::Ernie45ToolParser;
    use crate::output_grammar::OutputGrammarContext;
    use crate::output_grammar::test_utils::outline;
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

    #[test]
    fn ernie45_forced_tool_grammar_allows_template_newlines() {
        // The model writes `</think>\n\n<tool_call>...</tool_call>\n` and then
        // EOS, so a forced-call grammar must admit those framing newlines.
        let tools = &test_tools()[..1];
        let parser = Ernie45ToolParser::new(tools);
        let outline_for = |tool_choice: ToolChoice, parallel_tool_calls: bool| {
            let ctx = OutputGrammarContext {
                tools,
                tool_choice: &tool_choice,
                tool_strict_level: Default::default(),
                parallel_tool_calls,
            };
            outline(&parser.build_visible_format(&ctx).unwrap().unwrap())
        };

        expect![[r#"
            sequence
              star `\n`
              tags_with_separator `` at_least_one stop_after_first
                tag `<tool_call>\n{"name": "get_weather", "arguments": ` json(any) `}\n</tool_call>`
                tag `<tool_call>{"name": "get_weather", "arguments": ` json(any) `}</tool_call>`
              star
                sequence
                  star `\n`
                  tags_with_separator `` at_least_one stop_after_first
                    tag `<tool_call>\n{"name": "get_weather", "arguments": ` json(any) `}\n</tool_call>`
                    tag `<tool_call>{"name": "get_weather", "arguments": ` json(any) `}</tool_call>`
              star `\n`
        "#]].assert_eq(&outline_for(ToolChoice::required(), true));
        expect![[r#"
            sequence
              star `\n`
              tags_with_separator `` at_least_one stop_after_first
                tag `<tool_call>\n{"name": "get_weather", "arguments": ` json(any) `}\n</tool_call>`
                tag `<tool_call>{"name": "get_weather", "arguments": ` json(any) `}</tool_call>`
              star `\n`
        "#]]
        .assert_eq(&outline_for(ToolChoice::function("get_weather"), true));
    }
}
