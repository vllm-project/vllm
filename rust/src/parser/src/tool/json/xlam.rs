// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Tool parser for xLAM (Salesforce) JSON-array tool calls.
//!
//! The model outputs a JSON array of `{"name":..., "arguments":{...}}` objects.
//! Before the array, the model may emit any of the following prefix wrappers
//! (all of which are normalised away as content text):
//!
//! * plain text followed directly by `[`
//! * `[TOOL_CALLS][...]`
//! * `` ```json\n[...]\n``` ``
//! * `<tool_call>[...]</tool_call>`
//! * `</think>...[...]`  (DeepSeek-style reasoning tag)
//!
//! The array body itself uses `{"name":"fn","arguments":{...}}` objects
//! delimited by `,`.

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, seq};
use winnow::error::ModalResult;
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::literal;

use super::{JsonToolCallConfig, JsonToolCallEvent, JsonToolCallWhitespace};
use super::{argument_delta_event, tool_call_header_event};
use crate::tool::utils::{JsonObjectScanState, parse_buffered_event, safe_text_len_mul};
use crate::tool::{Result, Tool, ToolCallDelta, ToolParser, ToolParserEvent, ToolParserOutput};

// All known sentinels that open a tool-call block.
// The array start `[` itself is NOT listed here — we scan for it in Text mode
// via `find_array_start` once we know we are inside a sentinel block, or via
// `safe_text_until_array` for bare-array output.
const TOOL_CALLS_TAG: &str = "[TOOL_CALLS]";
const FENCE_TAG: &str = "```";
const FENCE_JSON_PREFIX: &str = "json";
const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
// After </think> the text is searched for the opening `[`.
const THINK_CLOSE: &str = "</think>";

// All outer sentinels the Text scanner must hold back for.
const ALL_SENTINELS: &[&str] = &[TOOL_CALLS_TAG, FENCE_TAG, TOOL_CALL_OPEN, THINK_CLOSE, "["];

const XLAM_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "xLAM",
    // start_marker / end_marker / marker_whitespace are not used by this parser
    // directly — they are only needed for the shared `JsonToolCallParser` core,
    // which we do NOT use here. We keep this config for the header/argument
    // helpers that do rely on it.
    start_marker: "[",
    end_marker: "]",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: Some(","),
    name_key: "name",
    arguments_key: &["arguments"],
};

/// Parser state machine for xLAM output.
#[derive(Debug, Clone, PartialEq, Eq)]
enum XlamMode {
    /// Scanning plain text; waiting for a sentinel or bare `[`.
    Text,
    /// Inside `[TOOL_CALLS]`, scanning for `[`.
    AfterToolCallsTag,
    /// Inside a `` ``` `` fence, waiting to optionally consume `json\n` before `[`.
    FenceJsonOpt,
    /// Inside `<tool_call>`, scanning for `[`.
    AfterXmlOpen,
    /// After `</think>`, scanning for `[`.
    AfterThink,
    /// Inside the top-level array; reading a tool-call header.
    Header,
    /// Inside the arguments object `{...}` of the current tool call.
    Arguments { json_scan: JsonObjectScanState },
    /// After one `}` closed; expecting `,` (more calls) or `]` (done).
    AfterCall,
    /// The `]` was consumed; consuming trailing whitespace or closing tags.
    AfterArray,
    /// Fully done, any remaining text is plain content.
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum XlamEvent {
    /// Emit `len` bytes of the buffer as plain text.
    Text { len: usize },
    /// One of the sentinel prefixes was consumed.
    SentinelEntered(SentinelKind),
    /// The opening `[` of the JSON array was consumed.
    ArrayStart,
    /// A tool-call header `{"name":"fn","arguments":` was consumed.
    ToolCallHeader { function_name: String },
    /// Raw argument bytes.
    Arguments { len: usize },
    /// The outer `}` of a tool call closed; `,` or `]` follows.
    ToolCallClose,
    /// A `,` delimiter between tool calls was consumed.
    Delimiter,
    /// The closing `]` of the JSON array was consumed.
    ArrayEnd,
    /// A closing tag like `</tool_call>` or ```` ` was consumed and should be dropped.
    ClosingTagDropped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SentinelKind {
    ToolCallsTag,
    Fence,
    XmlOpen,
    Think,
}

/// Tool parser for Salesforce xLAM JSON-array tool calls.
///
/// Example tool call content (bare array):
///
/// ```text
/// [{"name": "get_weather", "arguments": {"city": "Dallas", "state": "TX"}}]
/// ```
///
/// Also handles prefix wrappers:
/// * `[TOOL_CALLS][...]`
/// * `` ```json\n[...]\n``` ``
/// * `<tool_call>[...]</tool_call>`
/// * `</think>...[...]`
pub struct XlamToolParser {
    buffer: String,
    cursor: usize,
    tentative_events: Vec<ToolParserEvent>,
    mode: XlamMode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
}

impl XlamToolParser {
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            cursor: 0,
            tentative_events: Vec::new(),
            mode: XlamMode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    fn is_tentative(&self) -> bool {
        matches!(
            self.mode,
            XlamMode::AfterToolCallsTag
                | XlamMode::FenceJsonOpt
                | XlamMode::AfterXmlOpen
                | XlamMode::AfterThink
                | XlamMode::Header
        )
    }

    fn apply_event(
        &mut self,
        event: XlamEvent,
        output: &mut ToolParserOutput,
        consumed_len: usize,
    ) -> Result<()> {
        let mut text_to_push = None;
        let mut call_to_push = None;

        match event {
            XlamEvent::Text { len: _ } => {
                let text = self.buffer[self.cursor - consumed_len..self.cursor].to_string();
                text_to_push = Some(text);
            }
            XlamEvent::SentinelEntered(kind) => {
                if kind == SentinelKind::Think {
                    text_to_push = Some(THINK_CLOSE.to_string());
                }
                self.mode = match kind {
                    SentinelKind::ToolCallsTag => XlamMode::AfterToolCallsTag,
                    SentinelKind::Fence => XlamMode::FenceJsonOpt,
                    SentinelKind::XmlOpen => XlamMode::AfterXmlOpen,
                    SentinelKind::Think => XlamMode::AfterThink,
                };
            }
            XlamEvent::ArrayStart => {
                self.mode = XlamMode::Header;
            }
            XlamEvent::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = XlamMode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                };
                call_to_push = Some(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            XlamEvent::Arguments { len: _ } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "xLAM arguments without an active tool call"
                    ));
                };
                let arguments = self.buffer[self.cursor - consumed_len..self.cursor].to_string();
                call_to_push = Some(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments,
                });
            }
            XlamEvent::ToolCallClose => {
                self.active_tool_index = None;
                self.mode = XlamMode::AfterCall;
            }
            XlamEvent::Delimiter => {
                self.mode = XlamMode::Header;
            }
            XlamEvent::ArrayEnd => {
                self.mode = XlamMode::AfterArray;
            }
            XlamEvent::ClosingTagDropped => {
                self.mode = XlamMode::Done;
            }
        }

        if self.is_tentative() {
            if let Some(text) = text_to_push {
                self.tentative_events.push(ToolParserEvent::Text(text));
            }
            if let Some(call) = call_to_push {
                self.tentative_events.push(ToolParserEvent::ToolCall(call));
            }
        } else {
            if let Some(text) = text_to_push {
                output.push_text(text);
            }
            if let Some(call) = call_to_push {
                output.push_call(call);
            }
        }

        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = XlamMode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        self.cursor = 0;
        self.tentative_events.clear();
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for XlamToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        // In Done mode flush any trailing text and clear the buffer.
        if matches!(self.mode, XlamMode::Done) {
            output.push_text(&self.buffer[self.cursor..]);
            self.buffer.clear();
            self.cursor = 0;
            return Ok(());
        }

        while let Some((event, consumed_len)) =
            parse_buffered_event(&self.buffer[self.cursor..], |input| {
                parse_next_xlam_event(input, &mut self.mode)
            })?
        {
            self.cursor += consumed_len;
            let was_tentative = self.is_tentative();
            self.apply_event(event, output, consumed_len)?;

            if was_tentative && !self.is_tentative() {
                // We just transitioned out of tentative state (e.g. into Arguments).
                // Flush all tentative events and discard the buffered text up to cursor.
                for ev in self.tentative_events.drain(..) {
                    match ev {
                        ToolParserEvent::Text(text) => output.push_text(text),
                        ToolParserEvent::ToolCall(call) => output.push_call(call),
                    }
                }
                self.buffer.drain(..self.cursor);
                self.cursor = 0;
            } else if !self.is_tentative() {
                // If not tentative, we can safely drain the consumed bytes immediately.
                self.buffer.drain(..self.cursor);
                self.cursor = 0;
            }
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match &self.mode {
            // Nothing was started — the whole buffer is content text.
            XlamMode::Text
            | XlamMode::AfterToolCallsTag
            | XlamMode::FenceJsonOpt
            | XlamMode::AfterXmlOpen
            | XlamMode::AfterThink => {
                output.push_text(&self.buffer);
            }
            XlamMode::AfterArray | XlamMode::Done => {
                output.push_text(&self.buffer[self.cursor..]);
            }
            // Incomplete tool call — hard error.
            XlamMode::Header | XlamMode::Arguments { .. } => {
                return Err(parsing_failed!("incomplete xLAM tool call"));
            }
            // After closing the last `}` we still need to see `]`.
            XlamMode::AfterCall => {
                return Err(parsing_failed!("incomplete xLAM tool call array"));
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        XlamToolParser::reset(self)
    }
}

// ---------------------------------------------------------------------------
// winnow event parsers
// ---------------------------------------------------------------------------

fn parse_next_xlam_event(input: &mut XlamInput<'_>, mode: &mut XlamMode) -> ModalResult<XlamEvent> {
    match mode {
        XlamMode::Text => text_mode_event(input),
        XlamMode::AfterToolCallsTag | XlamMode::AfterXmlOpen | XlamMode::AfterThink => {
            scan_for_array_start(input)
        }
        XlamMode::FenceJsonOpt => fence_json_opt_event(input),
        XlamMode::Header => header_event(input),
        XlamMode::Arguments { json_scan } => arguments_event(input, json_scan),
        XlamMode::AfterCall => after_call_event(input),
        XlamMode::AfterArray => after_array_event(input),
        XlamMode::Done => {
            // Caller should have switched to flush mode before calling us.
            crate::utils::incomplete()
        }
    }
}

type XlamInput<'i> = Partial<&'i str>;

/// In Text mode: emit safe text or recognise a sentinel / bare `[`.
fn text_mode_event(input: &mut XlamInput<'_>) -> ModalResult<XlamEvent> {
    alt((
        // Known sentinels (consumed as plain content text up to the sentinel,
        // then the sentinel itself is consumed and we enter the matching mode).
        // Must come BEFORE literal("[") to avoid partial matches.
        literal(TOOL_CALLS_TAG).value(XlamEvent::SentinelEntered(SentinelKind::ToolCallsTag)),
        literal(FENCE_TAG).value(XlamEvent::SentinelEntered(SentinelKind::Fence)),
        literal(TOOL_CALL_OPEN).value(XlamEvent::SentinelEntered(SentinelKind::XmlOpen)),
        literal(THINK_CLOSE).value(XlamEvent::SentinelEntered(SentinelKind::Think)),
        // Bare JSON array start — no sentinel prefix.
        literal("[").value(XlamEvent::ArrayStart),
        // Safe text before the earliest possible sentinel / `[`.
        |input: &mut XlamInput<'_>| {
            safe_text_len_mul(input, ALL_SENTINELS).map(|len| XlamEvent::Text { len })
        },
    ))
    .parse_next(input)
}

/// After entering any sentinel mode: consume text until `[`.
fn scan_for_array_start(input: &mut XlamInput<'_>) -> ModalResult<XlamEvent> {
    alt((
        literal("[").value(XlamEvent::ArrayStart),
        // Emit content text before `[` (e.g. whitespace, prose after `</think>`).
        |input: &mut XlamInput<'_>| {
            safe_text_len_mul(input, &["["]).map(|len| XlamEvent::Text { len })
        },
    ))
    .parse_next(input)
}

/// After ```, optionally consume `json\n` and ignore it, or just scan for `[`.
fn fence_json_opt_event(input: &mut XlamInput<'_>) -> ModalResult<XlamEvent> {
    alt((
        seq!(_: literal(FENCE_JSON_PREFIX), _: ws0, _: literal("[")).value(XlamEvent::ArrayStart),
        seq!(_: ws0, _: literal("[")).value(XlamEvent::ArrayStart),
        // If it's not immediately `[` or `json[`, fallback to normal scan.
        scan_for_array_start,
    ))
    .parse_next(input)
}

/// Inside the JSON array: parse one tool-call header.
fn header_event(input: &mut XlamInput<'_>) -> ModalResult<XlamEvent> {
    alt((
        // Allow empty array or trailing comma by checking for `]` first.
        seq!(_: ws0, _: literal("]")).value(XlamEvent::ArrayEnd),
        |input: &mut XlamInput<'_>| match tool_call_header_event(input, XLAM_CONFIG)? {
            JsonToolCallEvent::ToolCallHeader { function_name } => {
                Ok(XlamEvent::ToolCallHeader { function_name })
            }
            _ => unreachable!("tool_call_header_event only emits ToolCallHeader"),
        },
    ))
    .parse_next(input)
}

/// Inside the arguments object: stream bytes until `{...}` closes.
fn arguments_event(
    input: &mut XlamInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<XlamEvent> {
    if json_scan.complete() {
        // Outer `}` of the tool-call wrapper.
        seq!(_: ws0, _: literal("}")).value(XlamEvent::ToolCallClose).parse_next(input)
    } else {
        match argument_delta_event(input, json_scan)? {
            JsonToolCallEvent::Arguments { len } => Ok(XlamEvent::Arguments { len }),
            _ => unreachable!("argument_delta_event only emits Arguments"),
        }
    }
}

/// After `}` of a tool call: expect `,` (more) or `]` (end of array).
fn after_call_event(input: &mut XlamInput<'_>) -> ModalResult<XlamEvent> {
    alt((
        seq!(_: ws0, _: literal("]")).value(XlamEvent::ArrayEnd),
        seq!(_: ws0, _: literal(","), _: ws0).value(XlamEvent::Delimiter),
    ))
    .parse_next(input)
}

/// After array ends, drop any closing tags like `</tool_call>` or `\n``` `.
fn after_array_event(input: &mut XlamInput<'_>) -> ModalResult<XlamEvent> {
    alt((
        seq!(_: ws0, _: literal(TOOL_CALL_CLOSE)).value(XlamEvent::ClosingTagDropped),
        seq!(_: ws0, _: literal(FENCE_TAG)).value(XlamEvent::ClosingTagDropped),
        // If neither is found, but we see safe text before a possible marker, emit it.
        |input: &mut XlamInput<'_>| {
            let markers = &[TOOL_CALL_CLOSE, FENCE_TAG];
            if let Ok(len) = safe_text_len_mul(input, markers) {
                return Ok(XlamEvent::Text { len });
            }
            // If we are stuck waiting for a marker but it's not completing,
            // we'll hang in AfterArray until Done flushes it.
            crate::tool::utils::incomplete()
        },
    ))
    .parse_next(input)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::XlamToolParser;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(r#"{{"name":"{function_name}","arguments":{arguments}}}"#)
    }

    fn build_array(calls: &[String]) -> String {
        format!("[{}]", calls.join(","))
    }

    // -----------------------------------------------------------------------
    // parse_complete — no tool calls
    // -----------------------------------------------------------------------

    #[test]
    fn xlam_parse_complete_without_tool_call_keeps_text() {
        let mut parser = XlamToolParser::new(&test_tools());
        let output = parser.parse_complete("This is a test").unwrap();

        assert_eq!(output.normal_text(), "This is a test");
        assert!(output.calls().is_empty());
    }

    // -----------------------------------------------------------------------
    // parse_complete — bare JSON array (no wrapper)
    // -----------------------------------------------------------------------

    #[test]
    fn xlam_parse_complete_bare_array_single_tool() {
        let mut parser = XlamToolParser::new(&test_tools());
        let input = build_array(&[build_tool_call(
            "get_weather",
            r#"{"city":"Dallas","state":"TX","unit":"fahrenheit"}"#,
        )]);
        let output = parser.parse_complete(&input).unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city":"Dallas","state":"TX","unit":"fahrenheit"})
        );
    }

    #[test]
    fn xlam_parse_complete_bare_array_parallel_calls() {
        let mut parser = XlamToolParser::new(&test_tools());
        let input = build_array(&[
            build_tool_call(
                "get_weather",
                r#"{"city":"Dallas","state":"TX","unit":"fahrenheit"}"#,
            ),
            build_tool_call(
                "get_weather",
                r#"{"city":"Orlando","state":"FL","unit":"fahrenheit"}"#,
            ),
        ]);
        let output = parser.parse_complete(&input).unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[1].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city":"Dallas","state":"TX","unit":"fahrenheit"})
        );
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[1].arguments).unwrap(),
            json!({"city":"Orlando","state":"FL","unit":"fahrenheit"})
        );
    }

    // -----------------------------------------------------------------------
    // parse_complete — `</think>` prefix wrapper
    // -----------------------------------------------------------------------

    #[test]
    fn xlam_parse_complete_with_think_tag() {
        let mut parser = XlamToolParser::new(&test_tools());
        let array = build_array(&[build_tool_call(
            "get_weather",
            r#"{"city":"Dallas","state":"TX","unit":"fahrenheit"}"#,
        )]);
        let input = format!("<think>I'll help you with that.</think>{array}");
        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(
            output.normal_text(),
            "<think>I'll help you with that.</think>"
        );
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    // -----------------------------------------------------------------------
    // parse_complete — ````json` fence wrapper
    // -----------------------------------------------------------------------

    #[test]
    fn xlam_parse_complete_with_json_code_fence() {
        let mut parser = XlamToolParser::new(&test_tools());
        let array = build_array(&[build_tool_call(
            "get_weather",
            r#"{"city":"Dallas","state":"TX","unit":"fahrenheit"}"#,
        )]);
        let input = format!("I'll help you with that.\n```json\n{array}\n```");
        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.normal_text(), "I'll help you with that.\n");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn xlam_parse_complete_with_json_code_fence_crlf_and_spaces() {
        let mut parser = XlamToolParser::new(&test_tools());
        let array = build_array(&[build_tool_call(
            "get_weather",
            r#"{"city":"Dallas","state":"TX","unit":"fahrenheit"}"#,
        )]);
        // Code fence with spaces and Windows CRLF
        let input = format!("I'll help you with that.\n```json \r\n  {array}\r\n```");
        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.normal_text(), "I'll help you with that.\n");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    // -----------------------------------------------------------------------
    // parse_complete — `[TOOL_CALLS]` tag wrapper
    // -----------------------------------------------------------------------

    #[test]
    fn xlam_parse_complete_with_tool_calls_tag() {
        let mut parser = XlamToolParser::new(&test_tools());
        let array = build_array(&[build_tool_call(
            "get_weather",
            r#"{"city":"Dallas","state":"TX","unit":"fahrenheit"}"#,
        )]);
        let input = format!("[TOOL_CALLS]{array}");
        let output = parser.parse_complete(&input).unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn xlam_parse_complete_with_tool_calls_tag_and_prefix_text() {
        let mut parser = XlamToolParser::new(&test_tools());
        let array = build_array(&[build_tool_call(
            "get_weather",
            r#"{"city":"Dallas","state":"TX","unit":"fahrenheit"}"#,
        )]);
        let input = format!("I'll check the weather for you.[TOOL_CALLS]{array}");
        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.normal_text(), "I'll check the weather for you.");
        assert_eq!(output.calls().len(), 1);
    }

    // -----------------------------------------------------------------------
    // parse_complete — `<tool_call>...</tool_call>` wrapper
    // -----------------------------------------------------------------------

    #[test]
    fn xlam_parse_complete_with_xml_tool_call_tag() {
        let mut parser = XlamToolParser::new(&test_tools());
        let array = build_array(&[build_tool_call(
            "get_weather",
            r#"{"city":"Dallas","state":"TX","unit":"fahrenheit"}"#,
        )]);
        let input = format!("I'll help you check the weather.<tool_call>{array}</tool_call>");
        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.normal_text(), "I'll help you check the weather.");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    // -----------------------------------------------------------------------
    // Unicode / non-ASCII arguments
    // -----------------------------------------------------------------------

    #[test]
    fn xlam_parse_complete_preserves_unicode_in_arguments() {
        let mut parser = XlamToolParser::new(&test_tools());
        let input = build_array(&[
            build_tool_call("get_weather", r#"{"city":"北京"}"#),
            build_tool_call("get_weather", r#"{"city":"上海"}"#),
        ]);
        let output = parser.parse_complete(&input).unwrap();

        let all_args: String = output.calls().iter().map(|c| c.arguments.as_str()).collect();
        assert!(all_args.contains('北'));
        assert!(all_args.contains('上'));
        assert!(
            !all_args.contains("\\u"),
            "arguments must not be unicode-escaped"
        );
    }

    // -----------------------------------------------------------------------
    // Empty arguments object
    // -----------------------------------------------------------------------

    #[test]
    fn xlam_parse_complete_empty_arguments() {
        let mut parser = XlamToolParser::new(&test_tools());
        let input = build_array(&[build_tool_call("get_weather", "{}")]);
        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({})
        );
    }

    // -----------------------------------------------------------------------
    // Streaming — argument deltas
    // -----------------------------------------------------------------------

    #[test]
    fn xlam_streaming_emits_argument_deltas() {
        let mut parser = XlamToolParser::new(&test_tools());
        let chunks = [
            r#"[{"name":"get_weather","arguments":"#,
            r#"{"city":"#,
            r#""Beijing""#,
            r#"}"#,
            r#"}]"#,
        ];

        let mut output = ToolParserOutput::default();
        let mut observed_args: Vec<String> = Vec::new();
        for chunk in chunks {
            let next = parser.parse_chunk(chunk).unwrap();
            observed_args.extend(
                next.calls().iter().filter(|c| c.name.is_none()).map(|c| c.arguments.clone()),
            );
            output.append(next);
        }
        output.append(parser.finish().unwrap());

        assert_eq!(observed_args, [r#"{"city":"#, r#""Beijing""#, "}"]);
        assert_eq!(
            serde_json::from_str::<Value>(&output.coalesce().calls()[0].arguments).unwrap(),
            json!({"city": "Beijing"})
        );
    }

    #[test]
    fn xlam_streaming_parallel_calls() {
        let input = build_array(&[
            build_tool_call("get_weather", r#"{"city":"Dallas","state":"TX"}"#),
            build_tool_call("get_weather", r#"{"city":"Orlando","state":"FL"}"#),
        ]);
        let chunks = split_by_chars(&input, 7);
        let mut parser = XlamToolParser::new(&test_tools());

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
                            arguments: "{\"city\":\"Dallas\",\"state\":\"TX\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"Orlando\",\"state\":\"FL\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn xlam_streaming_with_think_tag_split_across_chunks() {
        let array = build_array(&[build_tool_call("get_weather", r#"{"city":"Dallas"}"#)]);
        let input = format!("<think>reasoning</think>{array}");
        let chunks = split_by_chars(&input, 5);
        let mut parser = XlamToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "<think>reasoning</think>");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn xlam_streaming_with_tool_calls_tag_split_across_chunks() {
        let array = build_array(&[build_tool_call("get_weather", r#"{"city":"Dallas"}"#)]);
        let input = format!("[TOOL_CALLS]{array}");
        let chunks = split_by_chars(&input, 5);
        let mut parser = XlamToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn xlam_streaming_json_fence_split_across_chunks() {
        let array = build_array(&[build_tool_call("get_weather", r#"{"city":"Dallas"}"#)]);
        let input = format!("```json\n{array}\n```");
        let chunks = split_by_chars(&input, 4);
        let mut parser = XlamToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
    }

    // -----------------------------------------------------------------------
    // Error paths
    // -----------------------------------------------------------------------

    #[test]
    fn xlam_finish_fails_incomplete_tool_call() {
        let mut parser = XlamToolParser::new(&test_tools());
        parser.parse_chunk(r#"[{"name":"get_weather","arguments":{"city""#).unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete xLAM tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn xlam_finish_flushes_dangling_sentinel_without_array() {
        let mut parser = XlamToolParser::new(&test_tools());
        // A sentinel is sent, but the array never starts.
        let mut output = parser.parse_chunk("<tool_call>   ").unwrap();

        output.append(parser.finish().unwrap());
        // Since we fixed eager sentinel consumption, the sentinel is correctly
        // preserved and flushed when the parser finishes without finding a valid array.
        assert_eq!(output.normal_text(), "<tool_call>   ");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn xlam_malformed_field_order_fails_fast() {
        let mut parser = XlamToolParser::new(&test_tools());
        let error = parser.parse_chunk(r#"[{"arguments":{},"name":"get_weather"}]"#).unwrap_err();

        assert!(
            error.to_report_string().contains("xLAM"),
            "error should mention xLAM: {error}"
        );
    }

    #[test]
    fn xlam_parse_complete_bare_array_fallback() {
        let mut parser = XlamToolParser::new(&test_tools());
        // A bare array that is NOT a valid tool call.
        let _input = "The values are [1, 2, 3]";
        // Parse the safe text first (up to the bracket)
        let output1 = parser.parse_chunk("The values are ").unwrap();
        assert_eq!(output1.normal_text(), "The values are ");

        // Parse the bracket and the invalid content
        let _error = parser.parse_chunk("[1, 2, 3]").unwrap_err();

        // The parser should error out, but the uncommitted buffer MUST retain the `[`
        assert_eq!(
            parser.reset(),
            "[1, 2, 3]",
            "The `[` character must be preserved on fallback"
        );
    }
}
