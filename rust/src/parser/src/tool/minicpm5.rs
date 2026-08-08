// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::borrow::Cow;

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, eof, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, take_until};

use super::parameters::ToolSchemas;
use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len, take_until_marker};
use super::{Result, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::tool::Tool;

const FUNCTION_START: &str = "<function";
const FUNCTION_END: &str = "</function>";
const PARAM_START: &str = "<param";
const PARAM_END: &str = "</param>";
const ARGUMENTS_START: &str = "<arguments>";
const ARGUMENTS_END: &str = "</arguments>";
const CDATA_START: &str = "<![CDATA[";
const CDATA_END: &str = "]]>";
/// SentencePiece/GPT-style space marker some MiniCPM5 decodes emit.
const TOKENIZER_SPACE: char = '\u{0120}';
/// SentencePiece/GPT-style newline marker some MiniCPM5 decodes emit.
const TOKENIZER_NEWLINE: char = '\u{010a}';

type MiniCpm5Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum MiniCpm5Mode {
    Text,
    Function { end_marker_scan: MarkerScanState },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MiniCpm5Event {
    Text {
        len: usize,
    },
    FunctionStart,
    ToolCall {
        name: String,
        raw_params: Vec<(String, String)>,
    },
}

/// Tool parser for MiniCPM5 XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <function name="get_weather"><param name="city">上海</param></function>
/// ```
///
/// There is no wrapper block: `<function` itself opens a call and the block
/// ends at the first `</function>`. `name` is a quoted XML attribute (either
/// quote style); extra attributes on the function tag are skipped. Parameter
/// values wrapped in `<![CDATA[...]]>` are kept verbatim (including literal
/// `</param>` markup inside); other values are trimmed, deliberately adopting
/// the Python parser's regex-path semantics as the uniform rule. Params may
/// optionally sit under one `<arguments>` wrapper node. Collapsed tags such
/// as `<functionname=` (a real decode artifact of the special-token tags) are
/// accepted by making the attribute whitespace optional. Ported from
/// `vllm/tool_parsers/minicpm5xml_tool_parser.py`; intentional divergences
/// are noted in the PR that added this parser.
///
/// Structured-output tags are intentionally unsupported: the trait default
/// `structural_tag_builder() -> None` matches Python, which has no
/// structural-tag support for MiniCPM5.
pub struct MiniCpm5ToolParser {
    buffer: String,
    mode: MiniCpm5Mode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
}

impl MiniCpm5ToolParser {
    /// Create a MiniCPM5 tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: MiniCpm5Mode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
        }
    }

    /// Apply one parsed MiniCPM5 event to parser state and output.
    fn apply_event(&mut self, event: MiniCpm5Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            MiniCpm5Event::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            MiniCpm5Event::FunctionStart => {
                self.mode = MiniCpm5Mode::Function {
                    end_marker_scan: MarkerScanState::default(),
                };
            }
            MiniCpm5Event::ToolCall { name, raw_params } => {
                self.mode = MiniCpm5Mode::Text;
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
        }
        Ok(())
    }
}

impl ToolParser for MiniCpm5ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// The tool XML tags (`<function`, `</function>`, `<param`, `</param>`)
    /// are tokenizer special tokens in MiniCPM5, so decoding must keep them.
    /// This mirrors the Python parser's `adjust_request` forcing
    /// `skip_special_tokens = False`.
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(&normalize_tokenizer_artifacts(chunk));

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_minicpm5_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        // The mode check sits outside the buffer check: a stream ending right
        // on `<function` consumes the marker into the mode with an empty
        // buffer, and must still fail as an incomplete call.
        if matches!(self.mode, MiniCpm5Mode::Function { .. })
            || self.buffer.starts_with(FUNCTION_START)
        {
            return Err(parsing_failed!("incomplete MiniCPM5 tool call"));
        }
        let mut output = ToolParserOutput::default();
        if !self.buffer.is_empty() {
            output.push_text(&self.buffer);
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        // A started call already consumed `<function` into the mode, so put it
        // back: callers surface the recovered text verbatim after an error.
        let recovered = if matches!(self.mode, MiniCpm5Mode::Function { .. }) {
            let mut recovered = String::with_capacity(FUNCTION_START.len() + self.buffer.len());
            recovered.push_str(FUNCTION_START);
            recovered.push_str(&self.buffer);
            self.buffer.clear();
            recovered
        } else {
            std::mem::take(&mut self.buffer)
        };
        self.mode = MiniCpm5Mode::Text;
        self.emitted_tool_count = 0;
        recovered
    }
}

/// Rewrite the SentencePiece/GPT-style whitespace markers that MiniCPM5
/// decoding can emit around the special-token tags.
///
/// Mirrors Python `_normalize_model_output`. The collapsed-tag rewrite it also
/// performs is unnecessary here because the grammar accepts collapsed tags
/// directly.
fn normalize_tokenizer_artifacts(chunk: &str) -> Cow<'_, str> {
    if !chunk.contains(TOKENIZER_SPACE) && !chunk.contains(TOKENIZER_NEWLINE) {
        return Cow::Borrowed(chunk);
    }
    Cow::Owned(
        chunk
            .replace(TOKENIZER_SPACE, " ")
            .replace(TOKENIZER_NEWLINE, "\n"),
    )
}

/// Parse a MiniCPM5 event for the current parser mode.
fn parse_next_minicpm5_event(
    input: &mut MiniCpm5Input<'_>,
    mode: &mut MiniCpm5Mode,
) -> ModalResult<MiniCpm5Event> {
    match mode {
        MiniCpm5Mode::Text => alt((function_start_event, safe_text_event)).parse_next(input),
        MiniCpm5Mode::Function { end_marker_scan } => function_event(input, end_marker_scan),
    }
}

/// Parse a MiniCPM5 function start marker.
fn function_start_event(input: &mut MiniCpm5Input<'_>) -> ModalResult<MiniCpm5Event> {
    literal(FUNCTION_START).value(MiniCpm5Event::FunctionStart).parse_next(input)
}

/// Parse a safe text run before the next MiniCPM5 marker.
fn safe_text_event(input: &mut MiniCpm5Input<'_>) -> ModalResult<MiniCpm5Event> {
    safe_text_len(input, FUNCTION_START).map(|len| MiniCpm5Event::Text { len })
}

/// Parse a complete MiniCPM5 function block once its end marker arrives.
fn function_event(
    input: &mut MiniCpm5Input<'_>,
    end_marker_scan: &mut MarkerScanState,
) -> ModalResult<MiniCpm5Event> {
    let (body,) = seq!(
        take_until_marker(FUNCTION_END, end_marker_scan),
        _: literal(FUNCTION_END),
    )
    .parse_next(input)?;

    parse_function_body(body)
}

/// Parse a complete function body: `name` attribute, extra attributes, params.
fn parse_function_body(body: &str) -> ModalResult<MiniCpm5Event> {
    let mut input = body;
    // The whitespace before `name=` is optional: `<functionname=` is a real
    // decode artifact of the special-token tag, which Python normalizes to
    // `<function name=`.
    let (name, raw_params) = seq!(
        _: ws0,
        _: (literal("name"), ws0, '=', ws0),
        quoted_attr,
        // Tolerate extra attributes on the function tag, per Python `[^>]*`.
        _: take_until(0.., ">"),
        _: ">",
        _: ws0,
        params_block,
        _: (ws0, eof),
    )
    .parse_next(&mut input)?;

    Ok(MiniCpm5Event::ToolCall {
        name: name.trim().to_string(),
        raw_params,
    })
}

/// Parse params directly or under one `<arguments>` wrapper node.
fn params_block(input: &mut &str) -> ModalResult<Vec<(String, String)>> {
    alt((
        delimited(
            (literal(ARGUMENTS_START), ws0),
            repeat(0.., terminated(parameter, ws0)),
            literal(ARGUMENTS_END),
        ),
        repeat(0.., terminated(parameter, ws0)),
    ))
    .parse_next(input)
}

/// Parse one `<param name="key">value</param>` block.
fn parameter(input: &mut &str) -> ModalResult<(String, String)> {
    let (name, value) = seq!(
        _: literal(PARAM_START),
        // Optional whitespace accepts the collapsed `<paramname=` artifact.
        _: ws0,
        _: (literal("name"), ws0, '=', ws0),
        quoted_attr,
        _: ">",
        param_value,
        _: literal(PARAM_END),
    )
    .parse_next(input)?;

    Ok((name.trim().to_string(), value.to_string()))
}

/// Parse one parameter value: CDATA content verbatim, other values trimmed.
///
/// The CDATA branch scans for `]]>` rather than `</param>`, so a literal
/// `</param>` inside CDATA stays part of the value (the chat template wraps
/// values containing markup in CDATA itself).
fn param_value<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    alt((
        delimited(
            (ws0, literal(CDATA_START)),
            take_until(0.., CDATA_END),
            (literal(CDATA_END), ws0),
        ),
        take_until(0.., PARAM_END).map(unwrap_cdata_or_trim),
    ))
    .parse_next(input)
}

/// Parse a single- or double-quoted attribute value.
fn quoted_attr<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    alt((
        delimited('"', take_until(1.., "\""), '"'),
        delimited('\'', take_until(1.., "'"), '\''),
    ))
    .parse_next(input)
}

/// Keep `<![CDATA[...]]>` content verbatim; trim all other values.
fn unwrap_cdata_or_trim(raw: &str) -> &str {
    let trimmed = raw.trim();
    trimmed
        .strip_prefix(CDATA_START)
        .and_then(|inner| inner.strip_suffix(CDATA_END))
        .unwrap_or(trimmed)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::{MiniCpm5ToolParser, ToolParser};
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| format!("<param name=\"{name}\">{value}</param>"))
            .collect::<Vec<_>>()
            .join("");
        format!("<function name=\"{function_name}\">{params}</function>")
    }

    fn parsed_arguments(output: &ToolParserOutput, index: usize) -> Value {
        serde_json::from_str(&output.calls()[index].arguments).unwrap()
    }

    #[test]
    fn minicpm5_preserves_special_tokens() {
        let parser = MiniCpm5ToolParser::new(&test_tools());

        assert!(parser.preserve_special_tokens());
    }

    #[test]
    fn minicpm5_has_no_structural_tag_builder() {
        let parser = MiniCpm5ToolParser::new(&test_tools());

        assert!(parser.structural_tag_builder().is_none());
    }

    #[test]
    fn minicpm5_parse_complete_without_tool_call_keeps_text() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn minicpm5_parse_complete_extracts_single_tool_call() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&format!(
                "Let me check. {}",
                build_tool_call("get_weather", &[("location", "上海"), ("date", "2026-07-28")])
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "Let me check. ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            parsed_arguments(&output, 0),
            json!({ "location": "上海", "date": "2026-07-28" })
        );
    }

    #[test]
    fn minicpm5_parse_complete_extracts_zero_arg_call() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete("<function name=\"get_weather\"></function>")
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(parsed_arguments(&output, 0), json!({}));
    }

    #[test]
    fn minicpm5_parse_complete_converts_schema_types() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "convert",
                &[
                    ("whole", "5.0"),
                    ("flag", "true"),
                    ("payload", r#"{"nested":true}"#),
                    ("items", "[1,2]"),
                    ("empty", "42"),
                ],
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            parsed_arguments(&output, 0),
            json!({
                "whole": 5.0,
                "flag": true,
                "payload": { "nested": true },
                "items": [1, 2],
                "empty": "42",
            })
        );
    }

    #[test]
    fn minicpm5_parse_complete_accepts_single_quoted_attributes() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<function name='get_weather'><param name='location'>Tokyo</param></function>",
            )
            .unwrap();

        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(parsed_arguments(&output, 0), json!({ "location": "Tokyo" }));
    }

    #[test]
    fn minicpm5_parse_complete_skips_extra_function_attributes() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<function name=\"get_weather\" id=\"call_1\"><param name=\"location\">SF</param></function>",
            )
            .unwrap();

        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(parsed_arguments(&output, 0), json!({ "location": "SF" }));
    }

    #[test]
    fn minicpm5_parse_complete_accepts_collapsed_tags() {
        // `<functionname=` / `<paramname=` are real decode artifacts of the
        // special-token tags; Python normalizes them to the spaced form.
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<functionname=\"get_weather\"><paramname=\"location\">SF</param></function>",
            )
            .unwrap();

        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(parsed_arguments(&output, 0), json!({ "location": "SF" }));
    }

    #[test]
    fn minicpm5_parse_complete_normalizes_tokenizer_space_marker() {
        // Decoding can emit U+0120 in place of the space between the tag and
        // its `name` attribute (Python `test_tokenizer_space_marker`).
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<function\u{0120}name=\"get_weather\"><param\u{0120}name=\"location\">SF</param></function>",
            )
            .unwrap();

        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(parsed_arguments(&output, 0), json!({ "location": "SF" }));
    }

    #[test]
    fn minicpm5_parse_complete_normalizes_tokenizer_newline_marker() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[("location", "北\u{010a}京")],
            ))
            .unwrap();

        assert_eq!(parsed_arguments(&output, 0), json!({ "location": "北\n京" }));
    }

    #[test]
    fn minicpm5_reset_restores_consumed_function_marker() {
        // `<function` is consumed into the parser mode; error recovery reads
        // the buffer back through reset() and must not lose the opener.
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        parser.parse_chunk("ok <function name=\"get_weather\"").unwrap();

        assert_eq!(parser.reset(), "<function name=\"get_weather\"");
    }

    #[test]
    fn minicpm5_reset_restores_bare_consumed_function_marker() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        parser.parse_chunk("ok <function").unwrap();

        assert_eq!(parser.reset(), "<function");
    }

    #[test]
    fn minicpm5_parse_complete_keeps_cdata_verbatim() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<function name=\"echo\"><param name=\"text\"><![CDATA[  line one\nline two  ]]></param></function>",
            )
            .unwrap();

        assert_eq!(
            parsed_arguments(&output, 0),
            json!({ "text": "  line one\nline two  " })
        );
    }

    #[test]
    fn minicpm5_parse_complete_trims_non_cdata_values() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call("get_weather", &[("location", "\n  Tokyo  \n")]))
            .unwrap();

        assert_eq!(parsed_arguments(&output, 0), json!({ "location": "Tokyo" }));
    }

    #[test]
    fn minicpm5_parse_complete_supports_arguments_wrapper() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<function name=\"get_weather\"><arguments><param name=\"location\">SF</param></arguments></function>",
            )
            .unwrap();

        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(parsed_arguments(&output, 0), json!({ "location": "SF" }));
    }

    #[test]
    fn minicpm5_parse_complete_preserves_raw_ampersand_values() {
        // Models emit raw text without XML escaping; values pass through.
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call("echo", &[("text", "a & b < c")]))
            .unwrap();

        assert_eq!(parsed_arguments(&output, 0), json!({ "text": "a & b < c" }));
    }

    #[test]
    fn minicpm5_streaming_extracts_multiple_tool_calls_in_order() {
        let text = format!(
            "First.{}Between.{}Done.",
            build_tool_call("get_weather", &[("location", "SF")]),
            build_tool_call("get_weather", &[("location", "NYC")]),
        );
        let chunks = split_by_chars(&text, 5);
        let mut parser = MiniCpm5ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "First.Between.Done.");
        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(parsed_arguments(&output, 0), json!({ "location": "SF" }));
        assert_eq!(output.calls()[1].tool_index, 1);
        assert_eq!(parsed_arguments(&output, 1), json!({ "location": "NYC" }));
    }

    #[test]
    fn minicpm5_streaming_handles_start_marker_split_across_chunks() {
        let text = build_tool_call("get_weather", &[("location", "SF")]);
        let chunks = split_by_chars(&text, 3);
        let mut parser = MiniCpm5ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert_eq!(parsed_arguments(&output, 0), json!({ "location": "SF" }));
    }

    #[test]
    fn minicpm5_streaming_handles_end_marker_split_across_chunks() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();
        output.append(
            parser
                .parse_chunk("<function name=\"get_weather\"><param name=\"location\">SF</param></func")
                .unwrap(),
        );

        assert!(output.normal_text().is_empty());
        assert!(output.calls().is_empty());

        output.append(parser.parse_chunk("tion>").unwrap());
        output.append(parser.finish().unwrap());
        let output = output.coalesce();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(parsed_arguments(&output, 0), json!({ "location": "SF" }));
    }

    #[test]
    fn minicpm5_streaming_handles_cdata_split_across_chunks() {
        let text =
            "<function name=\"echo\"><param name=\"text\"><![CDATA[chunky value]]></param></function>";
        let chunks = split_by_chars(text, 7);
        let mut parser = MiniCpm5ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(parsed_arguments(&output, 0), json!({ "text": "chunky value" }));
    }

    #[test]
    fn minicpm5_streaming_buffers_long_body_until_end_marker() {
        let long_value = format!("v-{}", "x".repeat(8192));
        let text = build_tool_call("echo", &[("text", &long_value)]);
        let split_at = text.len() - "tion>".len();
        let (body_with_partial_end, end_suffix) = text.split_at(split_at);
        let chunks = split_by_chars(body_with_partial_end, 31);
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();

        for chunk in chunks {
            let chunk_output = parser.parse_chunk(chunk).unwrap();
            assert!(chunk_output.normal_text().is_empty());
            assert!(chunk_output.calls().is_empty());
            output.append(chunk_output);
        }

        output.append(parser.parse_chunk(end_suffix).unwrap());
        output.append(parser.finish().unwrap());
        let output = output.coalesce();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(parsed_arguments(&output, 0), json!({ "text": long_value }));
    }

    #[test]
    fn minicpm5_parse_complete_keeps_param_end_inside_cdata() {
        // The template wraps values containing markup in CDATA; a literal
        // `</param>` inside must stay part of the value.
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<function name=\"echo\"><param name=\"text\"><![CDATA[a</param>b]]></param></function>",
            )
            .unwrap();

        assert_eq!(parsed_arguments(&output, 0), json!({ "text": "a</param>b" }));
    }

    #[test]
    fn minicpm5_parse_complete_accepts_spaced_name_attribute() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<function name = \"get_weather\"><param name = \"location\">SF</param></function>",
            )
            .unwrap();

        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(parsed_arguments(&output, 0), json!({ "location": "SF" }));
    }

    #[test]
    fn minicpm5_finish_fails_stream_ending_on_bare_function_marker() {
        // A stream ending exactly on `<function` consumes the marker into the
        // parser mode with an empty buffer; finish() must still error rather
        // than silently dropping the token.
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let output = parser.parse_chunk("ok <function").unwrap();
        assert_eq!(output.normal_text(), "ok ");

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete MiniCPM5 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn minicpm5_finish_fails_incomplete_tool_call() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        parser
            .parse_chunk("<function name=\"get_weather\"><param name=\"location\">SF</param>")
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete MiniCPM5 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn minicpm5_malformed_missing_name_fails_fast() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let error = parser
            .parse_chunk("<function id=\"call_1\"><param name=\"location\">SF</param></function>")
            .unwrap_err();

        assert!(error.to_report_string().starts_with("tool parser parsing failed:"));
    }

    #[test]
    fn minicpm5_malformed_unclosed_param_fails_fast() {
        let mut parser = MiniCpm5ToolParser::new(&test_tools());
        let error = parser
            .parse_chunk("<function name=\"get_weather\"><param name=\"location\">SF</function>")
            .unwrap_err();

        assert!(error.to_report_string().starts_with("tool parser parsing failed:"));
    }
}
