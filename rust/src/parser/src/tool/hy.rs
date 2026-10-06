// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod structural_tag;

use std::sync::Arc;

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, eof, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use self::structural_tag::HyStructuralTagBuilder;
use super::parameters::ToolSchemas;
use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len, take_until_marker};
use super::{Result, ToolCallDelta, ToolParser, ToolParserError, ToolParserOutput};
use crate::tool::{StructuralTagBuilder, Tool};

/// Wire-level HY tool-call dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HyDialect {
    V3,
    V4,
}

impl HyDialect {
    fn separator(self) -> &'static str {
        match self {
            Self::V3 => "\n",
            Self::V4 => "",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::V3 => "HY3",
            Self::V4 => "HY4",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HyToolMarkers {
    tool_calls_start: String,
    tool_calls_end: String,
    tool_call_start: String,
    tool_call_end: String,
    tool_sep: Option<String>,
    arg_key_start: String,
    arg_key_end: String,
    arg_value_start: String,
    arg_value_end: String,
}

impl HyToolMarkers {
    pub(crate) fn new(suffix: &str, dialect: HyDialect) -> Self {
        Self {
            tool_calls_start: format!("<tool_calls{suffix}>"),
            tool_calls_end: format!("</tool_calls{suffix}>"),
            tool_call_start: format!("<tool_call{suffix}>"),
            tool_call_end: format!("</tool_call{suffix}>"),
            tool_sep: (dialect == HyDialect::V3).then(|| format!("<tool_sep{suffix}>")),
            arg_key_start: format!("<arg_key{suffix}>"),
            arg_key_end: format!("</arg_key{suffix}>"),
            arg_value_start: format!("<arg_value{suffix}>"),
            arg_value_end: format!("</arg_value{suffix}>"),
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &str> {
        [
            Some(self.tool_calls_start.as_str()),
            Some(self.tool_calls_end.as_str()),
            Some(self.tool_call_start.as_str()),
            Some(self.tool_call_end.as_str()),
            self.tool_sep.as_deref(),
            Some(self.arg_key_start.as_str()),
            Some(self.arg_key_end.as_str()),
            Some(self.arg_value_start.as_str()),
            Some(self.arg_value_end.as_str()),
        ]
        .into_iter()
        .flatten()
    }
}

type HyInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum HyMode {
    Text,
    ToolBlock { tool_call_end_scan: MarkerScanState },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HyEvent {
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

/// Tool parser for HY XML-style tool calls.
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
/// HY marker tokens are added-vocabulary tokens rather than tokenizer special
/// tokens, so the default `preserve_special_tokens() == false` is sufficient.
pub(crate) struct HyToolParser {
    buffer: String,
    mode: HyMode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
    dialect: HyDialect,
    markers: Arc<HyToolMarkers>,
    structural_tag_builder: HyStructuralTagBuilder,
}

impl HyToolParser {
    /// Create a HY tool parser for one wire dialect.
    pub(crate) fn new(tools: &[Tool], suffix: &str, dialect: HyDialect) -> Self {
        let markers = Arc::new(HyToolMarkers::new(suffix, dialect));
        Self {
            buffer: String::new(),
            mode: HyMode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
            dialect,
            markers: Arc::clone(&markers),
            structural_tag_builder: HyStructuralTagBuilder::new(markers, dialect),
        }
    }

    /// Apply one parsed HY event to parser state and output.
    fn apply_event(&mut self, event: HyEvent, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            HyEvent::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            HyEvent::ToolBlockStart => {
                self.mode = HyMode::ToolBlock {
                    tool_call_end_scan: MarkerScanState::default(),
                };
            }
            HyEvent::ToolCall { name, raw_params } => {
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
            HyEvent::ToolBlockEnd => self.mode = HyMode::Done,
            HyEvent::IgnoredRest => {}
        }
        Ok(())
    }
}

impl ToolParser for HyToolParser {
    // Suffix discovery belongs to the unified HY parser so its reasoning and
    // tool delimiters always use the same tokenizer-derived value.
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Err(ToolParserError::DummyUnifiedParser {
            name: "hy".to_string(),
        })
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(&self.structural_tag_builder)
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_hy_event(input, &mut self.mode, &self.markers, self.dialect)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            HyMode::Text => output.push_text(&self.buffer),
            HyMode::ToolBlock { .. } => {
                return Err(parsing_failed!(
                    "incomplete {} tool call",
                    self.dialect.label()
                ));
            }
            HyMode::Done => {}
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = HyMode::Text;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

/// Parse a HY event for the current parser mode.
fn parse_next_hy_event(
    input: &mut HyInput<'_>,
    mode: &mut HyMode,
    markers: &HyToolMarkers,
    dialect: HyDialect,
) -> ModalResult<HyEvent> {
    match mode {
        HyMode::Text => parse_text_event(input, markers),
        HyMode::ToolBlock { tool_call_end_scan } => {
            parse_tool_block_event(input, tool_call_end_scan, markers, dialect)
        }
        HyMode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode HY event.
fn parse_text_event(input: &mut HyInput<'_>, markers: &HyToolMarkers) -> ModalResult<HyEvent> {
    alt((
        |input: &mut HyInput<'_>| tool_block_start_event(input, markers),
        |input: &mut HyInput<'_>| safe_text_event(input, markers),
    ))
    .parse_next(input)
}

/// Parse a HY tool-block start marker.
fn tool_block_start_event(
    input: &mut HyInput<'_>,
    markers: &HyToolMarkers,
) -> ModalResult<HyEvent> {
    literal(markers.tool_calls_start.as_str())
        .value(HyEvent::ToolBlockStart)
        .parse_next(input)
}

/// Parse a safe text run before the next HY marker.
fn safe_text_event(input: &mut HyInput<'_>, markers: &HyToolMarkers) -> ModalResult<HyEvent> {
    safe_text_len(input, &markers.tool_calls_start).map(|len| HyEvent::Text { len })
}

/// Parse one event inside a HY tool block.
fn parse_tool_block_event(
    input: &mut HyInput<'_>,
    tool_call_end_scan: &mut MarkerScanState,
    markers: &HyToolMarkers,
    dialect: HyDialect,
) -> ModalResult<HyEvent> {
    alt((
        |input: &mut HyInput<'_>| tool_block_end_event(input, markers),
        |input: &mut HyInput<'_>| tool_call_event(input, tool_call_end_scan, markers, dialect),
    ))
    .parse_next(input)
}

/// Parse a HY tool-block end marker.
fn tool_block_end_event(input: &mut HyInput<'_>, markers: &HyToolMarkers) -> ModalResult<HyEvent> {
    (ws0, literal(markers.tool_calls_end.as_str()))
        .value(HyEvent::ToolBlockEnd)
        .parse_next(input)
}

/// Parse a complete HY tool-call block.
fn tool_call_event(
    input: &mut HyInput<'_>,
    tool_call_end_scan: &mut MarkerScanState,
    markers: &HyToolMarkers,
    dialect: HyDialect,
) -> ModalResult<HyEvent> {
    let (body,) = seq!(
        _: ws0,
        _: literal(markers.tool_call_start.as_str()),
        take_until_marker(markers.tool_call_end.as_str(), tool_call_end_scan),
        _: literal(markers.tool_call_end.as_str()),
    )
    .parse_next(input)?;
    let mut body_input = body;
    let (name, params) = parse_tool_call_body(&mut body_input, markers, dialect)?;
    let raw_params = parse_tool_call_params(params, markers)?;

    Ok(HyEvent::ToolCall {
        name: name.trim().to_string(),
        raw_params,
    })
}

/// Parse a complete HY tool-call body according to its dialect.
fn parse_tool_call_body<'i>(
    input: &mut &'i str,
    markers: &HyToolMarkers,
    dialect: HyDialect,
) -> ModalResult<(&'i str, &'i str)> {
    match dialect {
        HyDialect::V3 => {
            let tool_sep = markers.tool_sep.as_deref().expect("HY3 has a tool separator");
            terminated(
                seq!(
                    take_until(0.., tool_sep),
                    _: literal(tool_sep),
                    rest,
                ),
                eof,
            )
            .parse_next(input)
        }
        HyDialect::V4 => terminated(
            alt((
                seq!(take_until(0.., markers.arg_key_start.as_str()), rest,),
                rest.map(|name| (name, "")),
            )),
            eof,
        )
        .parse_next(input),
    }
}

/// Parse all parameter blocks inside a complete HY tool call.
fn parse_tool_call_params(
    tool_call_body: &str,
    markers: &HyToolMarkers,
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

/// Parse a HY argument key/value block.
fn parameter(input: &mut &str, markers: &HyToolMarkers) -> ModalResult<(String, String)> {
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

/// Parse ignored rest after the HY tool block ends.
fn ignored_rest_event(input: &mut HyInput<'_>) -> ModalResult<HyEvent> {
    rest.value(HyEvent::IgnoredRest).parse_next(input)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::{HyDialect, HyToolParser, ToolParser};
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
        let parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);

        assert!(!parser.preserve_special_tokens());
    }

    #[test]
    fn hy_v3_parse_complete_without_tool_call_keeps_text() {
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
        let output = parser.parse_complete("This is a plain response.").unwrap();

        assert_eq!(output.normal_text(), "This is a plain response.");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn hy_v3_parse_complete_extracts_zero_arg_inline_tool_call() {
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);

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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "hello ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(parsed_arguments(&output, 0), json!({ "city": "Beijing" }));
    }

    #[test]
    fn hy_v3_streaming_does_not_emit_incomplete_tool_call() {
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
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
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
        parser.parse_chunk("<tool_calls><tool_call>get_weather<tool_sep>").unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete HY3 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn hy_v3_malformed_tool_call_fails_fast() {
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V3);
        let error = parser
            .parse_complete(
                "<tool_calls><tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Beijing</tool_call></tool_calls>",
            )
            .unwrap_err();

        assert!(error.to_report_string().starts_with("tool parser parsing failed:"));
    }

    #[test]
    fn hy_v4_parse_complete_extracts_compact_calls_with_and_without_arguments() {
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V4);
        let output = parser
            .parse_complete(
                "<tool_calls><tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value><arg_key>date</arg_key><arg_value>2026-03-30</arg_value></tool_call><tool_call>get_current_date</tool_call></tool_calls>",
            )
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
                                "get_current_date",
                            ),
                            arguments: "{}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn hy_v4_streaming_handles_markers_split_across_chunks() {
        let input = "prefix<tool_calls><tool_call>get_weather<arg_key>city</arg_key><arg_value>上海</arg_value></tool_call><tool_call>get_current_date</tool_call></tool_calls>";
        let chunks = split_by_chars(input, 7);
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V4);

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "prefix");
        assert_eq!(output.calls().len(), 2);
        assert_eq!(parsed_arguments(&output, 0), json!({ "city": "上海" }));
        assert_eq!(output.calls()[1].name.as_deref(), Some("get_current_date"));
        assert_eq!(parsed_arguments(&output, 1), json!({}));
    }

    #[test]
    fn hy_v4_finish_reports_its_dialect_for_incomplete_calls() {
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V4);
        parser.parse_chunk("<tool_calls><tool_call>get_weather").unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete HY4 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn hy_v4_malformed_argument_pair_fails_fast() {
        let mut parser = HyToolParser::new(&test_tools(), "", HyDialect::V4);
        let error = parser
            .parse_complete(
                "<tool_calls><tool_call>get_weather<arg_key>city</arg_key>Beijing</tool_call></tool_calls>",
            )
            .unwrap_err();

        assert!(error.to_report_string().starts_with("tool parser parsing failed:"));
    }
}
