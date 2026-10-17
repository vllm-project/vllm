// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use xgrammar_structural_tag::format::Format;

use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::output_grammar::{self, OutputGrammarContext};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

const HERMES_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Hermes",
    start_marker: "<tool_call>",
    framed_start_marker: None,
    end_marker: "</tool_call>",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: None,
    name_key: "name",
    arguments_key: &["arguments"],
};

/// Tool parser for Hermes XML-wrapped JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>{"name": "get_weather", "arguments": {"location":"Tokyo"}}</tool_call>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
///
/// Note: parallel calls are represented as repeated
/// `<tool_call>...</tool_call>` blocks, not as multiple calls inside one tag.
pub struct HermesToolParser {
    inner: JsonToolCallParser,
}

impl HermesToolParser {
    /// Create a Hermes tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(HERMES_CONFIG),
        }
    }
}

impl ToolParser for HermesToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::Hermes.builder())
    }

    fn build_visible_format(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<Format>> {
        let format =
            output_grammar::visible_format_from_builder(self.structural_tag_builder(), ctx)?;
        Ok(format.map(with_template_call_separators))
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

/// Accept repeated `required` calls separated by the newline that Hermes chat
/// templates render between them, as well as back to back, which is all the
/// shared builder accepts.
fn with_template_call_separators(format: Format) -> Format {
    match format {
        Format::TagsWithSeparator(calls) if !calls.stop_after_first => {
            let mut newline_separated = calls.clone();
            newline_separated.separator = "\n".to_owned();
            Format::or(vec![
                Format::TagsWithSeparator(newline_separated),
                Format::TagsWithSeparator(calls),
            ])
        }
        format => format,
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;
    use xgrammar_structural_tag::ToolChoice;

    use super::HermesToolParser;
    use crate::output_grammar::OutputGrammarContext;
    use crate::output_grammar::test_utils::outline;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(r#"<tool_call>{{"name":"{function_name}","arguments":{arguments}}}</tool_call>"#)
    }

    #[test]
    fn hermes_parse_complete_without_tool_call_keeps_text() {
        let mut parser = HermesToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn hermes_parse_complete_extracts_raw_json_arguments() {
        let mut parser = HermesToolParser::new(&test_tools());
        let arguments = r#"{ "location": "Tokyo", "days": "3" }"#;
        let output = parser
            .parse_complete(&format!(
                "Let me check.\n{}",
                build_tool_call("get_weather", arguments)
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "Let me check.\n");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn hermes_accepts_newline_after_tool_call_start() {
        let mut parser = HermesToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"<tool_call>
{"name":"get_weather","arguments":{}}</tool_call>"#,
            )
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn hermes_does_not_validate_or_normalize_arguments() {
        let mut parser = HermesToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let output = parser.parse_complete(&build_tool_call("get_weather", arguments)).unwrap();

        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn hermes_streaming_emits_argument_deltas() {
        let mut parser = HermesToolParser::new(&test_tools());
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
                next.calls()
                    .iter()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            output.append(next);
        }
        output.append(parser.finish().unwrap());

        assert_eq!(observed_arguments, ["{\"location\":", "\"Beijing\"", "}"]);
        assert_eq!(output.normal_text(), "preface  suffix");
        assert_eq!(
            output.coalesce().calls()[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn hermes_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = HermesToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "hello ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn hermes_streaming_extracts_multiple_tool_calls() {
        let input = format!(
            "{}{}",
            build_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            build_tool_call("add", r#"{"x":1,"y":2}"#),
        );
        let chunks = split_by_chars(&input, 7);
        let mut parser = HermesToolParser::new(&test_tools());

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
    fn hermes_finish_fails_incomplete_tool_call() {
        let mut parser = HermesToolParser::new(&test_tools());
        parser
            .parse_chunk(r#"<tool_call>{"name":"get_weather","arguments":{"location""#)
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete Hermes tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn hermes_required_grammar_accepts_template_call_separators() {
        // Hermes templates write `</tool_call>\n<tool_call>` between parallel
        // calls and some fine-tunes write them back to back, so a `required`
        // grammar must admit both.
        let tools = &test_tools()[..1];
        let parser = HermesToolParser::new(tools);
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
            or
              tags_with_separator `\n` at_least_one
                tag `<tool_call>\n{"name": "get_weather", "arguments": ` json(any) `}\n</tool_call>`
                tag `<tool_call>{"name": "get_weather", "arguments": ` json(any) `}</tool_call>`
              tags_with_separator `` at_least_one
                tag `<tool_call>\n{"name": "get_weather", "arguments": ` json(any) `}\n</tool_call>`
                tag `<tool_call>{"name": "get_weather", "arguments": ` json(any) `}</tool_call>`
        "#]]
        .assert_eq(&outline_for(ToolChoice::required(), true));
        expect![[r#"
            tags_with_separator `` at_least_one stop_after_first
              tag `<tool_call>\n{"name": "get_weather", "arguments": ` json(any) `}\n</tool_call>`
              tag `<tool_call>{"name": "get_weather", "arguments": ` json(any) `}</tool_call>`
        "#]]
        .assert_eq(&outline_for(ToolChoice::required(), false));
        expect![[r#"
            tags_with_separator `` at_least_one stop_after_first
              tag `<tool_call>\n{"name": "get_weather", "arguments": ` json(any) `}\n</tool_call>`
              tag `<tool_call>{"name": "get_weather", "arguments": ` json(any) `}</tool_call>`
        "#]]
        .assert_eq(&outline_for(ToolChoice::function("get_weather"), true));
    }
}
