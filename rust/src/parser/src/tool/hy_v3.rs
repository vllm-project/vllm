// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod structural_tag;

use std::sync::Arc;

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, eof, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use self::structural_tag::HyV3StructuralTagBuilder;
use super::parameters::ToolSchemas;
use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len, take_until_marker};
use super::{Result, ToolCallDelta, ToolParser, ToolParserError, ToolParserOutput};
use crate::tool::{StructuralTagBuilder, Tool};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HyV3ToolMarkers {
    tool_calls_start: String,
    tool_calls_end: String,
    tool_call_start: String,
    tool_call_end: String,
    tool_sep: String,
    arg_key_start: String,
    arg_key_end: String,
    arg_value_start: String,
    arg_value_end: String,
}

impl HyV3ToolMarkers {
    pub(crate) fn new(suffix: &str) -> Self {
        Self {
            tool_calls_start: format!("<tool_calls{suffix}>"),
            tool_calls_end: format!("</tool_calls{suffix}>"),
            tool_call_start: format!("<tool_call{suffix}>"),
            tool_call_end: format!("</tool_call{suffix}>"),
            tool_sep: format!("<tool_sep{suffix}>"),
            arg_key_start: format!("<arg_key{suffix}>"),
            arg_key_end: format!("</arg_key{suffix}>"),
            arg_value_start: format!("<arg_value{suffix}>"),
            arg_value_end: format!("</arg_value{suffix}>"),
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &str> {
        [
            self.tool_calls_start.as_str(),
            self.tool_calls_end.as_str(),
            self.tool_call_start.as_str(),
            self.tool_call_end.as_str(),
            self.tool_sep.as_str(),
            self.arg_key_start.as_str(),
            self.arg_key_end.as_str(),
            self.arg_value_start.as_str(),
            self.arg_value_end.as_str(),
        ]
        .into_iter()
    }
}

type HyV3Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum HyV3Mode {
    Text,
    ToolBlock { tool_call_end_scan: MarkerScanState },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HyV3Event {
    Text {
        len: usize,
    },
    ToolBlockStart,
    ToolCall {
        name: String,
        raw_params: Vec<(String, String)>,
    },
    ToolBlockEnd,
    IgnoredRest,
}

/// Tool parser for HY3 XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_calls>
/// <tool_call>get_weather<tool_sep>
/// <arg_key>city</arg_key><arg_value>Beijing</arg_value>
/// </tool_call>
/// </tool_calls>
/// ```
///
/// Arguments are emitted only after a full `<tool_call>` block is parsed.
/// HY3 marker tokens are added-vocabulary tokens rather than tokenizer special
/// tokens, so the default `preserve_special_tokens() == false` is sufficient.
pub(crate) struct HyV3ToolParser {
    buffer: String,
    mode: HyV3Mode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
    markers: Arc<HyV3ToolMarkers>,
    structural_tag_builder: HyV3StructuralTagBuilder,
}

impl HyV3ToolParser {
    /// Create a HY3 tool parser.
    pub(crate) fn new(tools: &[Tool], suffix: &str) -> Self {
        let markers = Arc::new(HyV3ToolMarkers::new(suffix));
        Self {
            buffer: String::new(),
            mode: HyV3Mode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
            markers: Arc::clone(&markers),
            structural_tag_builder: HyV3StructuralTagBuilder::new(markers),
        }
    }

    /// Apply one parsed HY3 event to parser state and output.
    fn apply_event(&mut self, event: HyV3Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            HyV3Event::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            HyV3Event::ToolBlockStart => {
                self.mode = HyV3Mode::ToolBlock {
                    tool_call_end_scan: MarkerScanState::default(),
                };
            }
            HyV3Event::ToolCall { name, raw_params } => {
                let arguments = self.tool_parameters.convert_params_with_schema(&name, raw_params);
                let arguments = serde_json::to_string(&arguments)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                output.push_call(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_tool_count += 1;
            }
            HyV3Event::ToolBlockEnd => self.mode = HyV3Mode::Done,
            HyV3Event::IgnoredRest => {}
        }
        Ok(())
    }
}

impl ToolParser for HyV3ToolParser {
    // Suffix discovery belongs to `HyV3UnifiedParser` so its reasoning and
    // tool delimiters always use the same tokenizer-derived value.
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Err(ToolParserError::DummyUnifiedParser {
            name: "hy_v3".to_string(),
        })
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(&self.structural_tag_builder)
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_hy_v3_event(input, &mut self.mode, &self.markers)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            HyV3Mode::Text => output.push_text(&self.buffer),
            HyV3Mode::ToolBlock { .. } => return Err(parsing_failed!("incomplete HY3 tool call")),
            HyV3Mode::Done => {}
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = HyV3Mode::Text;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

/// Parse a HY3 event for the current parser mode.
fn parse_next_hy_v3_event(
    input: &mut HyV3Input<'_>,
    mode: &mut HyV3Mode,
    markers: &HyV3ToolMarkers,
) -> ModalResult<HyV3Event> {
    match mode {
        HyV3Mode::Text => parse_text_event(input, markers),
        HyV3Mode::ToolBlock { tool_call_end_scan } => {
            parse_tool_block_event(input, tool_call_end_scan, markers)
        }
        HyV3Mode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode HY3 event.
fn parse_text_event(
    input: &mut HyV3Input<'_>,
    markers: &HyV3ToolMarkers,
) -> ModalResult<HyV3Event> {
    alt((
        |input: &mut HyV3Input<'_>| tool_block_start_event(input, markers),
        |input: &mut HyV3Input<'_>| safe_text_event(input, markers),
    ))
    .parse_next(input)
}

/// Parse a HY3 tool-block start marker.
fn tool_block_start_event(
    input: &mut HyV3Input<'_>,
    markers: &HyV3ToolMarkers,
) -> ModalResult<HyV3Event> {
    literal(markers.tool_calls_start.as_str())
        .value(HyV3Event::ToolBlockStart)
        .parse_next(input)
}

/// Parse a safe text run before the next HY3 marker.
fn safe_text_event(input: &mut HyV3Input<'_>, markers: &HyV3ToolMarkers) -> ModalResult<HyV3Event> {
    safe_text_len(input, &markers.tool_calls_start).map(|len| HyV3Event::Text { len })
}

/// Parse one event inside a HY3 tool block.
fn parse_tool_block_event(
    input: &mut HyV3Input<'_>,
    tool_call_end_scan: &mut MarkerScanState,
    markers: &HyV3ToolMarkers,
) -> ModalResult<HyV3Event> {
    alt((
        |input: &mut HyV3Input<'_>| tool_block_end_event(input, markers),
        |input: &mut HyV3Input<'_>| tool_call_event(input, tool_call_end_scan, markers),
    ))
    .parse_next(input)
}

/// Parse a HY3 tool-block end marker.
fn tool_block_end_event(
    input: &mut HyV3Input<'_>,
    markers: &HyV3ToolMarkers,
) -> ModalResult<HyV3Event> {
    (ws0, literal(markers.tool_calls_end.as_str()))
        .value(HyV3Event::ToolBlockEnd)
        .parse_next(input)
}

/// Parse a complete HY3 tool-call block.
fn tool_call_event(
    input: &mut HyV3Input<'_>,
    tool_call_end_scan: &mut MarkerScanState,
    markers: &HyV3ToolMarkers,
) -> ModalResult<HyV3Event> {
    let (name, body) = seq!(
        _: ws0,
        _: literal(markers.tool_call_start.as_str()),
        take_until(0.., markers.tool_sep.as_str()),
        _: literal(markers.tool_sep.as_str()),
        take_until_marker(markers.tool_call_end.as_str(), tool_call_end_scan),
        _: literal(markers.tool_call_end.as_str()),
    )
    .parse_next(input)?;
    let raw_params = parse_tool_call_params(body, markers)?;

    Ok(HyV3Event::ToolCall {
        name: name.trim().to_string(),
        raw_params,
    })
}

/// Parse all parameter blocks inside a complete HY3 tool call.
fn parse_tool_call_params(
    tool_call_body: &str,
    markers: &HyV3ToolMarkers,
) -> ModalResult<Vec<(String, String)>> {
    let mut input = tool_call_body;
    delimited(
        ws0,
        repeat(
            0..,
            terminated(|input: &mut &str| parameter(input, markers), ws0),
        ),
        eof,
    )
    .parse_next(&mut input)
}

/// Parse a HY3 argument key/value block.
fn parameter(input: &mut &str, markers: &HyV3ToolMarkers) -> ModalResult<(String, String)> {
    let (name, value) = seq!(
        _: literal(markers.arg_key_start.as_str()),
        take_until(0.., markers.arg_key_end.as_str()),
        _: literal(markers.arg_key_end.as_str()),
        _: ws0,
        _: literal(markers.arg_value_start.as_str()),
        take_until(0.., markers.arg_value_end.as_str()),
        _: literal(markers.arg_value_end.as_str()),
    )
    .parse_next(input)?;

    Ok((name.trim().to_string(), value.to_string()))
}

/// Parse ignored rest after the HY3 tool block ends.
fn ignored_rest_event(input: &mut HyV3Input<'_>) -> ModalResult<HyV3Event> {
    rest.value(HyV3Event::IgnoredRest).parse_next(input)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::{HyV3ToolParser, ToolParser};
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| format!("<arg_key>{name}</arg_key><arg_value>{value}</arg_value>"))
            .collect::<Vec<_>>()
            .join("\n");
        format!("<tool_call>{function_name}<tool_sep>{params}</tool_call>")
    }

    fn build_tool_calls(tool_calls: &[String]) -> String {
        format!("<tool_calls>\n{}\n</tool_calls>", tool_calls.join("\n"))
    }

    fn parsed_arguments(output: &ToolParserOutput, index: usize) -> Value {
        serde_json::from_str(&output.calls()[index].arguments).unwrap()
    }

    #[test]
    fn hy_v3_does_not_preserve_special_tokens() {
        let parser = HyV3ToolParser::new(&test_tools(), "");

        assert!(!parser.preserve_special_tokens());
    }

    #[test]
    fn hy_v3_parse_complete_without_tool_call_keeps_text() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let output = parser.parse_complete("This is a plain response.").unwrap();

        assert_eq!(output.normal_text(), "This is a plain response.");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn hy_v3_parse_complete_extracts_zero_arg_inline_tool_call() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let output = parser
            .parse_complete(
                "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>",
            )
            .unwrap();

        assert_eq!(output.normal_text(), "");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_current_date"));
        assert_eq!(parsed_arguments(&output, 0), json!({}));
    }

    #[test]
    fn hy_v3_parse_complete_extracts_zero_arg_newline_tool_call() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let output = parser
            .parse_complete(
                "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>",
            )
            .unwrap();

        assert_eq!(output.calls()[0].name.as_deref(), Some("get_current_date"));
        assert_eq!(parsed_arguments(&output, 0), json!({}));
    }

    #[test]
    fn hy_v3_parse_complete_extracts_arguments_on_same_line() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let output = parser
            .parse_complete(
                "<tool_calls><tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Beijing</arg_value><arg_key>date</arg_key><arg_value>2026-03-30</arg_value></tool_call></tool_calls>",
            )
            .unwrap();

        assert_eq!(
            parsed_arguments(&output, 0),
            json!({ "city": "Beijing", "date": "2026-03-30" })
        );
    }

    #[test]
    fn hy_v3_parse_complete_extracts_arguments_with_newlines() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let output = parser
            .parse_complete(&build_tool_calls(&[build_tool_call(
                "get_weather",
                &[("city", "Beijing"), ("date", "2026-03-30")],
            )]))
            .unwrap();

        assert_eq!(
            parsed_arguments(&output, 0),
            json!({ "city": "Beijing", "date": "2026-03-30" })
        );
    }

    #[test]
    fn hy_v3_parse_complete_preserves_prefix_and_ignores_trailing_text() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let output = parser
            .parse_complete(&format!(
                "Checking.{} trailing text",
                build_tool_calls(&[build_tool_call("get_current_date", &[])])
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "Checking.");
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_current_date"));
    }

    #[test]
    fn hy_v3_parse_complete_extracts_multiple_tool_calls_in_one_block() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let output = parser
            .parse_complete(&build_tool_calls(&[
                build_tool_call(
                    "get_weather",
                    &[("city", "Beijing"), ("date", "2026-03-30")],
                ),
                build_tool_call(
                    "get_weather",
                    &[("city", "Hangzhou"), ("date", "2026-03-30")],
                ),
            ]))
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
                            arguments: "{\"city\":\"Beijing\",\"date\":\"2026-03-30\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"Hangzhou\",\"date\":\"2026-03-30\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn hy_v3_parse_complete_converts_schema_types() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let output = parser
            .parse_complete(&build_tool_calls(&[build_tool_call(
                "convert",
                &[
                    ("whole", "5.3"),
                    ("flag", "true"),
                    ("payload", r#"{"k":1}"#),
                    ("items", "[1,2]"),
                ],
            )]))
            .unwrap();

        assert_eq!(
            parsed_arguments(&output, 0),
            json!({
                "whole": 5.3,
                "flag": true,
                "payload": { "k": 1 },
                "items": [1, 2]
            })
        );
    }

    #[test]
    fn hy_v3_streaming_without_tool_call_emits_text_incrementally() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let mut output = ToolParserOutput::default();

        output.append(parser.parse_chunk("This is ").unwrap());
        output.append(parser.parse_chunk("a plain ").unwrap());
        output.append(parser.parse_chunk("response.").unwrap());
        output.append(parser.finish().unwrap());

        assert_eq!(output.normal_text(), "This is a plain response.");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn hy_v3_streaming_extracts_zero_arg_tool_call() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let chunks = [
            "<tool_calls>",
            "\n<tool_call>",
            "get_current_date",
            "<tool_sep>",
            "\n</tool_call>",
            "\n</tool_calls>",
        ];

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_current_date"));
        assert_eq!(parsed_arguments(&output, 0), json!({}));
    }

    #[test]
    fn hy_v3_streaming_extracts_arguments() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let chunks = [
            "<tool_calls>",
            "\n<tool_call>",
            "get_weather",
            "<tool_sep>",
            "\n<arg_key>city</arg_key>",
            "\n<arg_value>Beijing</arg_value>",
            "\n<arg_key>date</arg_key>",
            "\n<arg_value>2026-03-30</arg_value>",
            "\n</tool_call>",
            "\n</tool_calls>",
        ];

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            parsed_arguments(&output, 0),
            json!({ "city": "Beijing", "date": "2026-03-30" })
        );
    }

    #[test]
    fn hy_v3_streaming_preserves_prefix_text() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let chunks = [
            "Checking.",
            "<tool_calls>",
            "\n<tool_call>",
            "get_current_date",
            "<tool_sep>",
            "\n</tool_call>",
            "\n</tool_calls>",
        ];

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "Checking.");
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_current_date"));
    }

    #[test]
    fn hy_v3_streaming_extracts_multiple_tool_calls_in_one_block() {
        let input = build_tool_calls(&[
            build_tool_call(
                "get_weather",
                &[("city", "Beijing"), ("date", "2026-03-30")],
            ),
            build_tool_call(
                "get_weather",
                &[("city", "Hangzhou"), ("date", "2026-03-30")],
            ),
        ]);
        let chunks = split_by_chars(&input, 9);
        let mut parser = HyV3ToolParser::new(&test_tools(), "");

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 2);
        assert_eq!(parsed_arguments(&output, 0)["city"], json!("Beijing"));
        assert_eq!(parsed_arguments(&output, 1)["city"], json!("Hangzhou"));
    }

    #[test]
    fn hy_v3_streaming_handles_start_marker_split_across_chunks() {
        let input = format!(
            "hello {}",
            build_tool_calls(&[build_tool_call("get_weather", &[("city", "Beijing")])])
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = HyV3ToolParser::new(&test_tools(), "");

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "hello ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(parsed_arguments(&output, 0), json!({ "city": "Beijing" }));
    }

    #[test]
    fn hy_v3_streaming_does_not_emit_incomplete_tool_call() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let mut output = ToolParserOutput::default();

        parser
            .parse_into(
                "<tool_calls><tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Bei",
                &mut output,
            )
            .unwrap();

        assert_eq!(output.normal_text(), "");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn hy_v3_finish_fails_incomplete_tool_call() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        parser.parse_chunk("<tool_calls><tool_call>get_weather<tool_sep>").unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete HY3 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn hy_v3_malformed_tool_call_fails_fast() {
        let mut parser = HyV3ToolParser::new(&test_tools(), "");
        let error = parser
            .parse_complete(
                "<tool_calls><tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Beijing</tool_call></tool_calls>",
            )
            .unwrap_err();

        assert!(error.to_report_string().starts_with("tool parser parsing failed:"));
    }
}
