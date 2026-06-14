use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, peek, seq};
use winnow::error::{ModalResult, StrContext};
use winnow::prelude::*;
use winnow::token::literal;

use super::{
    JsonToolCallConfig, JsonToolCallEvent, JsonToolCallWhitespace, JsonToolInput,
    argument_delta_event, tool_call_header_event,
};
use crate::utils::{JsonObjectScanState, incomplete, parse_buffered_event};
use crate::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};

/// Granite 3.0 tool-call marker.
const BOT_TOKEN: &str = "<|tool_call|>";
/// Granite 3.1 tool-call marker.
const BOT_STRING: &str = "<tool_call>";
/// Optional leading markers, tried in order, before the JSON array.
const MARKERS: [&str; 2] = [BOT_TOKEN, BOT_STRING];

#[derive(Debug, Clone, PartialEq, Eq)]
enum GraniteMode {
    Start,
    Passthrough,
    ArrayStart,
    Header,
    Arguments { json_scan: JsonObjectScanState },
    AfterElement,
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GraniteEvent {
    BeginElement,
    ToolCallHeader { function_name: String },
    Arguments { len: usize },
    ElementClose,
    Separator,
    ArrayEnd,
}

/// How the parser should leave `Start` once enough input is buffered.
enum StartCommit {
    /// Not enough buffered yet to decide; wait for more input.
    NeedMore,
    /// The output is plain content; never enter tool parsing.
    Passthrough,
    /// A tool-call array begins; drop `drain_len` leading bytes (whitespace and
    /// the optional marker) so the buffer starts at the JSON array.
    Array { drain_len: usize },
}

/// Tool parser for Granite 3.0/3.1 JSON-array tool calls.
///
/// Example tool call content:
///
/// ```text
/// <|tool_call|>[{"name": "get_weather", "arguments": {"location": "Tokyo"}}]
/// ```
///
/// The leading marker is `<|tool_call|>` (Granite 3.0) or `<tool_call>`
/// (Granite 3.1) and is optional; the payload is a JSON array of
/// `{"name", "arguments"}` objects. Arguments are already OpenAI-style JSON
/// text, so they are streamed as raw argument deltas without schema conversion
/// or JSON normalization. Output that does not begin with the array (after an
/// optional marker and surrounding whitespace) is treated as plain content.
pub struct GraniteToolParser {
    buffer: String,
    mode: GraniteMode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
}

impl GraniteToolParser {
    /// Create a Granite tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: GraniteMode::Start,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    /// Decide whether the buffered output begins a tool-call array.
    ///
    /// Mirrors the Python parser's `strip`/`removeprefix`/`lstrip` gate: leading
    /// whitespace and a single optional marker are skipped, and the output is a
    /// tool call only when the next non-whitespace byte is `[`.
    fn try_commit_start(&self) -> StartCommit {
        let leading_ws = self.buffer.len() - self.buffer.trim_start().len();
        let rest = &self.buffer[leading_ws..];
        if rest.is_empty() {
            return StartCommit::NeedMore;
        }

        if let Some(marker) = MARKERS.iter().find(|marker| rest.starts_with(**marker)) {
            let after_marker = &rest[marker.len()..];
            let after = after_marker.trim_start();
            return if after.is_empty() {
                StartCommit::NeedMore
            } else if after.starts_with('[') {
                // Drain post-marker whitespace here; `trim_start` is broader than the grammar's `multispace0`.
                let ws_after = after_marker.len() - after.len();
                StartCommit::Array {
                    drain_len: leading_ws + marker.len() + ws_after,
                }
            } else {
                StartCommit::Passthrough
            };
        }

        if rest.starts_with('[') {
            StartCommit::Array {
                drain_len: leading_ws,
            }
        } else if is_partial_marker(rest) {
            StartCommit::NeedMore
        } else {
            StartCommit::Passthrough
        }
    }

    /// Apply one parsed Granite event to parser state and output.
    fn apply_event(&mut self, event: GraniteEvent, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            GraniteEvent::BeginElement => self.mode = GraniteMode::Header,
            GraniteEvent::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = GraniteMode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                };
                output.calls.push(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            GraniteEvent::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "Granite arguments without an active tool call"
                    ));
                };
                output.calls.push(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..consumed_len].to_string(),
                });
            }
            GraniteEvent::ElementClose => {
                self.active_tool_index = None;
                self.mode = GraniteMode::AfterElement;
            }
            GraniteEvent::Separator => {
                self.active_tool_index = None;
                self.mode = GraniteMode::Header;
            }
            GraniteEvent::ArrayEnd => self.mode = GraniteMode::Done,
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = GraniteMode::Start;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for GraniteToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        if matches!(self.mode, GraniteMode::Start) {
            match self.try_commit_start() {
                StartCommit::NeedMore => return Ok(()),
                StartCommit::Passthrough => self.mode = GraniteMode::Passthrough,
                StartCommit::Array { drain_len } => {
                    self.buffer.drain(..drain_len);
                    self.mode = GraniteMode::ArrayStart;
                }
            }
        }

        if matches!(self.mode, GraniteMode::Passthrough) {
            output.normal_text.push_str(&self.buffer);
            self.buffer.clear();
            return Ok(());
        }

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_granite_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match &self.mode {
            GraniteMode::Start | GraniteMode::Passthrough => {
                output.normal_text.push_str(&self.buffer);
            }
            GraniteMode::Done if self.buffer.trim().is_empty() => {}
            GraniteMode::ArrayStart
            | GraniteMode::Header
            | GraniteMode::Arguments { .. }
            | GraniteMode::AfterElement => {
                return Err(parsing_failed!("incomplete Granite tool call"));
            }
            GraniteMode::Done => return Err(parsing_failed!("invalid Granite tool call")),
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        GraniteToolParser::reset(self)
    }
}

/// Return whether `text` could still grow into a tool-call marker.
fn is_partial_marker(text: &str) -> bool {
    MARKERS
        .iter()
        .any(|marker| marker.len() > text.len() && marker.starts_with(text))
}

/// Parse a Granite event for the current parser mode.
fn parse_next_granite_event(
    input: &mut JsonToolInput<'_>,
    mode: &mut GraniteMode,
) -> ModalResult<GraniteEvent> {
    match mode {
        GraniteMode::Start | GraniteMode::Passthrough => {
            unreachable!("Granite parser driver must commit before parsing events")
        }
        GraniteMode::ArrayStart => array_start_event(input),
        GraniteMode::Header => granite_tool_call_header_event(input),
        GraniteMode::Arguments { json_scan } => granite_arguments_event(input, json_scan),
        GraniteMode::AfterElement => after_element_event(input),
        GraniteMode::Done => incomplete(),
    }
}

/// Parse the opening `[` of a Granite tool-call array.
fn array_start_event(input: &mut JsonToolInput<'_>) -> ModalResult<GraniteEvent> {
    seq!(_: ws0, _: literal("[")).parse_next(input)?;
    alt((
        seq!(_: ws0, _: literal("]")).value(GraniteEvent::ArrayEnd),
        peek(seq!(_: ws0, _: literal("{"))).value(GraniteEvent::BeginElement),
    ))
    .context(StrContext::Label("Granite"))
    .parse_next(input)
}

/// Parse a Granite tool-call element header before the raw arguments payload.
fn granite_tool_call_header_event(input: &mut JsonToolInput<'_>) -> ModalResult<GraniteEvent> {
    const CONFIG: JsonToolCallConfig = JsonToolCallConfig {
        parser_name: "Granite",
        start_marker: "",
        end_marker: "",
        marker_whitespace: JsonToolCallWhitespace::Optional,
        delimiter: None,
        name_key: "name",
        arguments_key: &["arguments"],
    };

    match tool_call_header_event(input, CONFIG)? {
        JsonToolCallEvent::ToolCallHeader { function_name } => {
            Ok(GraniteEvent::ToolCallHeader { function_name })
        }
        _ => unreachable!("tool_call_header_event only emits ToolCallHeader"),
    }
}

/// Parse one event inside a Granite tool-call arguments payload.
fn granite_arguments_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<GraniteEvent> {
    if json_scan.complete() {
        seq!(_: ws0, _: literal("}"))
            .value(GraniteEvent::ElementClose)
            .parse_next(input)
    } else {
        match argument_delta_event(input, json_scan)? {
            JsonToolCallEvent::Arguments { len } => Ok(GraniteEvent::Arguments { len }),
            _ => unreachable!("argument_delta_event only emits Arguments"),
        }
    }
}

/// Parse the separator or closing bracket after one Granite tool call.
fn after_element_event(input: &mut JsonToolInput<'_>) -> ModalResult<GraniteEvent> {
    alt((
        seq!(_: ws0, _: literal(","), _: ws0).value(GraniteEvent::Separator),
        seq!(_: ws0, _: literal("]")).value(GraniteEvent::ArrayEnd),
    ))
    .context(StrContext::Label("Granite"))
    .parse_next(input)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::GraniteToolParser;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    #[test]
    fn granite_parse_complete_without_tool_call_keeps_text() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let output = parser
            .parse_complete("This is a regular response without any tool calls.")
            .unwrap();

        assert_eq!(
            output.normal_text,
            "This is a regular response without any tool calls."
        );
        assert!(output.calls.is_empty());
    }

    #[test]
    fn granite_parse_complete_extracts_token_marker_call() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"<|tool_call|>[{"name": "get_weather", "arguments": {"location": "Tokyo"}}]"#,
            )
            .unwrap();

        assert_eq!(output.normal_text, "");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].tool_index, 0);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[0].arguments, r#"{"location": "Tokyo"}"#);
    }

    #[test]
    fn granite_parse_complete_accepts_string_marker() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"<tool_call>[{"name":"get_weather","arguments":{"location":"Tokyo"}}]"#,
            )
            .unwrap();

        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn granite_accepts_leading_whitespace_and_space_after_marker() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let output = parser
            .parse_complete("\n  <|tool_call|> [{\"name\":\"get_weather\",\"arguments\":{}}]")
            .unwrap();

        assert_eq!(output.normal_text, "");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[0].arguments, "{}");
    }

    #[test]
    fn granite_handles_unicode_whitespace_after_marker() {
        // A form feed is whitespace to Rust's `trim_start` (and Python's
        // `lstrip`) but not to winnow's `multispace0`; the commit gate must
        // drain it so the array parser still sees a leading `[`.
        let mut parser = GraniteToolParser::new(&test_tools());
        let output = parser
            .parse_complete("<|tool_call|>\u{000c}[{\"name\":\"get_weather\",\"arguments\":{}}]")
            .unwrap();

        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn granite_parses_array_without_marker() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let output = parser
            .parse_complete(r#"[{"name":"get_weather","arguments":{"location":"Tokyo"}}]"#)
            .unwrap();

        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn granite_handles_indented_json_arguments() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let input = "<|tool_call|>[\n    {\n        \"name\": \"get_weather\",\n        \"arguments\": {\n            \"location\": \"Tokyo\"\n        }\n    }\n]";
        let output = parser.parse_complete(input).unwrap();

        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            output.calls[0].arguments,
            "{\n            \"location\": \"Tokyo\"\n        }"
        );
    }

    #[test]
    fn granite_extracts_multiple_calls() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let input = r#"<|tool_call|>[{"name":"get_weather","arguments":{"location":"Shanghai"}},{"name":"add","arguments":{"x":1,"y":2}}]"#;
        let output = parser.parse_complete(input).unwrap();

        expect![[r#"
            ToolParserOutput {
                normal_text: "",
                calls: [
                    ToolCallDelta {
                        tool_index: 0,
                        name: Some(
                            "get_weather",
                        ),
                        arguments: "{\"location\":\"Shanghai\"}",
                    },
                    ToolCallDelta {
                        tool_index: 1,
                        name: Some(
                            "add",
                        ),
                        arguments: "{\"x\":1,\"y\":2}",
                    },
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn granite_empty_array_has_no_calls() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let output = parser.parse_complete("<|tool_call|>[]").unwrap();

        assert_eq!(output.normal_text, "");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn granite_text_starting_like_marker_is_content() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let output = parser.parse_complete("<p>not a tool call</p>").unwrap();

        assert_eq!(output.normal_text, "<p>not a tool call</p>");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn granite_streaming_handles_split_markers() {
        let input = r#"<|tool_call|>[{"name":"get_weather","arguments":{"location":"Tokyo"}}]"#;
        let chunks = split_by_chars(input, 5);
        let mut parser = GraniteToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text, "");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn granite_streaming_emits_argument_deltas() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let chunks = [
            "<|tool_call|>[{\"name\":\"get_weather\",\"arguments\":",
            "{\"location\":",
            "\"Beijing\"",
            "}}]",
        ];

        let mut output = ToolParserOutput::default();
        let mut observed_arguments = Vec::new();
        for chunk in chunks {
            let next = parser.parse_chunk(chunk).unwrap();
            observed_arguments.extend(
                next.calls
                    .iter()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            output.append(next);
        }
        output.append(parser.finish().unwrap());

        assert_eq!(observed_arguments, ["{\"location\":", "\"Beijing\"", "}"]);
        assert_eq!(
            output.coalesce_calls().calls[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn granite_marker_without_array_is_content() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let output = parser.parse_complete("<tool_call> just chatting, not a tool call").unwrap();

        assert_eq!(
            output.normal_text,
            "<tool_call> just chatting, not a tool call"
        );
        assert!(output.calls.is_empty());
    }

    #[test]
    fn granite_preserves_complex_arguments_verbatim() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let arguments = r#"{"quoted":"He said \"hi\" with ] and }","i":42,"f":3.14,"b":true,"nothing":null,"arr":["a","b","c"],"obj":{"nested":"value"},"empty_arr":[],"empty_obj":{}}"#;
        let output = parser
            .parse_complete(&format!(
                r#"<|tool_call|>[{{"name":"convert","arguments":{arguments}}}]"#
            ))
            .unwrap();

        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("convert"));
        assert_eq!(output.calls[0].arguments, arguments);
    }

    #[test]
    fn granite_surrounding_text_disables_tool_parsing() {
        let mut parser = GraniteToolParser::new(&test_tools());
        let input = "Let me check the weather.\n<|tool_call|> [{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"}}]\nDone.";
        let output = parser.parse_complete(input).unwrap();

        assert_eq!(output.normal_text, input);
        assert!(output.calls.is_empty());
    }

    #[test]
    fn granite_finish_fails_incomplete_tool_call() {
        let mut parser = GraniteToolParser::new(&test_tools());
        parser
            .parse_chunk(r#"<|tool_call|>[{"name":"get_weather","arguments":{"location""#)
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete Granite tool call"]
            .assert_eq(&error.to_report_string());
    }
}
