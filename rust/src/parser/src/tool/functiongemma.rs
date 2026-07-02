//! Streaming tool parser for Google FunctionGemma models.
//!
//! Handles the FunctionGemma function-call format:
//!
//! `<start_function_call>call:func_name{key:<escape>value<escape>}<end_function_call>`
//!
//! Original Python implementation:
//! <https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/functiongemma_tool_parser.py>

use serde_json::{Map, Value};
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, eof, opt, separated, seq, terminated};
use winnow::error::{ContextError, ErrMode, ModalResult};
use winnow::prelude::*;
use winnow::stream::{Partial, Stream};
use winnow::token::{literal, take_until, take_while};

use super::utils::{incomplete, parse_buffered_event, partial_prefix_len, safe_text_len_mul};
use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};

const TOOL_CALL_START: &str = "<start_function_call>";
const TOOL_CALL_END: &str = "<end_function_call>";
const CALL_PREFIX: &str = "call:";
const ESCAPE: &str = "<escape>";

type FunctionGemmaInput<'i> = Partial<&'i str>;

/// Streaming scan state for FunctionGemma arguments.
///
/// Tracks position and whether we are inside an `<escape>`-delimited value so
/// that `<end_function_call>` inside a string value is not mistaken for the
/// closing marker.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ArgsScanState {
    scanned_len: usize,
    in_string: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum Mode {
    #[default]
    Text,
    Header,
    Arguments {
        name: String,
        args_scan: ArgsScanState,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum Event {
    Text { len: usize },
    ToolCallStart,
    ToolCallHeader { name: String },
    ToolCall { args: Map<String, Value> },
}

/// Tool parser for Google FunctionGemma models.
///
/// FunctionGemma uses a custom function-call syntax with `<escape>`-delimited
/// string values. Arguments are converted to JSON before emission (similar to
/// the Python parser's `json.loads` pass on each value).
pub struct FunctionGemmaToolParser {
    buffer: String,
    mode: Mode,
    emitted_tool_count: usize,
}

impl FunctionGemmaToolParser {
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: Mode::default(),
            emitted_tool_count: 0,
        }
    }

    fn apply_event(&mut self, event: Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            Event::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            Event::ToolCallStart => self.mode = Mode::Header,
            Event::ToolCallHeader { name } => {
                self.mode = Mode::Arguments {
                    name,
                    args_scan: ArgsScanState::default(),
                };
            }
            Event::ToolCall { args } => {
                let mode = std::mem::replace(&mut self.mode, Mode::Text);
                let Mode::Arguments { name, .. } = mode else {
                    return Err(parsing_failed!(
                        "FunctionGemma arguments without an active tool call"
                    ));
                };
                let arguments = serde_json::to_string(&args)
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

    fn reset(&mut self) -> String {
        let raw = match std::mem::replace(&mut self.mode, Mode::Text) {
            Mode::Text => std::mem::take(&mut self.buffer),
            Mode::Header => {
                format!("{}{}", TOOL_CALL_START, std::mem::take(&mut self.buffer))
            }
            Mode::Arguments { name, .. } => {
                format!(
                    "{}{}{}{{{}",
                    TOOL_CALL_START,
                    CALL_PREFIX,
                    name,
                    std::mem::take(&mut self.buffer)
                )
            }
        };
        self.mode = Mode::Text;
        self.emitted_tool_count = 0;
        raw
    }
}

impl ToolParser for FunctionGemmaToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match &self.mode {
            Mode::Text => output.push_text(std::mem::take(&mut self.buffer)),
            Mode::Header | Mode::Arguments { .. } => {
                return Err(parsing_failed!("incomplete FunctionGemma tool call"));
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        FunctionGemmaToolParser::reset(self)
    }
}

/// Parse one FunctionGemma event from buffered streaming input.
fn parse_next_event(input: &mut FunctionGemmaInput<'_>, mode: &mut Mode) -> ModalResult<Event> {
    match mode {
        Mode::Text => parse_text_event(input),
        Mode::Header => tool_call_header_event(input),
        Mode::Arguments { args_scan, .. } => tool_call_args_event(input, args_scan),
    }
}

/// Parse a text-mode FunctionGemma event.
fn parse_text_event(input: &mut FunctionGemmaInput<'_>) -> ModalResult<Event> {
    alt((tool_call_start_event, safe_text_event)).parse_next(input)
}

/// Parse a FunctionGemma tool-call start marker.
fn tool_call_start_event(input: &mut FunctionGemmaInput<'_>) -> ModalResult<Event> {
    literal(TOOL_CALL_START).value(Event::ToolCallStart).parse_next(input)
}

/// Parse a safe text run before the next FunctionGemma tool-call marker.
fn safe_text_event(input: &mut FunctionGemmaInput<'_>) -> ModalResult<Event> {
    safe_text_len_mul(input, &[TOOL_CALL_START]).map(|len| Event::Text { len })
}

/// Parse a FunctionGemma tool-call header.
fn tool_call_header_event(input: &mut FunctionGemmaInput<'_>) -> ModalResult<Event> {
    let (name,) = seq!(
        _: literal(CALL_PREFIX),
        functiongemma_tool_name,
        _: literal("{"),
    )
    .parse_next(input)?;
    Ok(Event::ToolCallHeader { name })
}

/// Parse a FunctionGemma tool name (word characters before `{`).
fn functiongemma_tool_name(input: &mut FunctionGemmaInput<'_>) -> ModalResult<String> {
    let name = take_until(1.., "{").parse_next(input)?.trim();
    if name.is_empty() {
        return Err(ErrMode::Cut(ContextError::new()));
    }
    Ok(name.to_string())
}

/// Parse complete FunctionGemma tool-call arguments.
fn tool_call_args_event(
    input: &mut FunctionGemmaInput<'_>,
    args_scan: &mut ArgsScanState,
) -> ModalResult<Event> {
    let raw_args = raw_args_until_end_marker(input, args_scan)?;
    let args_input = raw_args.strip_suffix('}').unwrap_or(raw_args);
    let args = parse_functiongemma_args(args_input)?;
    Ok(Event::ToolCall { args })
}

/// Parse raw FunctionGemma arguments through the first end marker outside an
/// `<escape>`-delimited string.
fn raw_args_until_end_marker<'i>(
    input: &mut FunctionGemmaInput<'i>,
    state: &mut ArgsScanState,
) -> ModalResult<&'i str> {
    let text = **input;
    if state.scanned_len > text.len() {
        return incomplete();
    }

    loop {
        let rest = &text[state.scanned_len..];
        if state.in_string {
            let Some(escape_pos) = rest.find(ESCAPE) else {
                state.scanned_len = safe_scan_len(text, state.scanned_len, &[ESCAPE]);
                return incomplete();
            };
            state.scanned_len += escape_pos + ESCAPE.len();
            state.in_string = false;
            continue;
        }

        let next_escape = rest.find(ESCAPE);
        let next_end = rest.find(TOOL_CALL_END);
        match (next_escape, next_end) {
            (Some(esc), Some(end)) if end < esc => {
                let end_pos = state.scanned_len + end;
                state.scanned_len = end_pos + TOOL_CALL_END.len();
                input.next_slice(state.scanned_len);
                return Ok(&text[..end_pos]);
            }
            (Some(esc), _) => {
                state.scanned_len += esc + ESCAPE.len();
                state.in_string = true;
            }
            (None, Some(end)) => {
                let end_pos = state.scanned_len + end;
                state.scanned_len = end_pos + TOOL_CALL_END.len();
                input.next_slice(state.scanned_len);
                return Ok(&text[..end_pos]);
            }
            (None, None) => {
                state.scanned_len =
                    safe_scan_len(text, state.scanned_len, &[ESCAPE, TOOL_CALL_END]);
                return incomplete();
            }
        }
    }
}

/// Return the scan length while holding back a split marker prefix.
fn safe_scan_len(text: &str, start: usize, markers: &[&str]) -> usize {
    let max_partial = markers
        .iter()
        .map(|marker| partial_prefix_len(&text[start..], marker))
        .max()
        .unwrap_or(0);
    text.len() - max_partial
}

/// Parse complete FunctionGemma key-value arguments into a JSON object.
fn parse_functiongemma_args(args: &str) -> ModalResult<Map<String, Value>> {
    let mut input = args;
    terminated(functiongemma_pairs, eof).parse_next(&mut input)
}

/// Parse FunctionGemma key-value pairs.
fn functiongemma_pairs(input: &mut &str) -> ModalResult<Map<String, Value>> {
    let pairs: Vec<(String, Value)> = delimited(
        ws0,
        terminated(
            separated(0.., functiongemma_pair, comma_separator),
            opt(comma_separator),
        ),
        ws0,
    )
    .parse_next(input)?;
    Ok(pairs.into_iter().collect())
}

/// Parse a FunctionGemma key-value pair.
fn functiongemma_pair(input: &mut &str) -> ModalResult<(String, Value)> {
    let (key, value) = seq!(
        _: ws0,
        functiongemma_key,
        _: ws0,
        _: literal(":"),
        _: ws0,
        functiongemma_value,
    )
    .parse_next(input)?;
    Ok((key, value))
}

/// Parse a FunctionGemma bare key (word characters).
fn functiongemma_key(input: &mut &str) -> ModalResult<String> {
    let key = take_while(1.., |ch: char| ch.is_alphanumeric() || ch == '_').parse_next(input)?;
    if key.is_empty() {
        return Err(ErrMode::Cut(ContextError::new()));
    }
    Ok(key.to_string())
}

/// Parse a FunctionGemma `<escape>`-delimited value and convert to JSON.
fn functiongemma_value(input: &mut &str) -> ModalResult<Value> {
    let raw =
        delimited(literal(ESCAPE), take_until(0.., ESCAPE), literal(ESCAPE)).parse_next(input)?;
    Ok(parse_functiongemma_scalar(raw))
}

/// Parse a FunctionGemma comma separator with optional whitespace.
fn comma_separator(input: &mut &str) -> ModalResult<()> {
    delimited(ws0, literal(","), ws0).void().parse_next(input)
}

/// Convert a raw FunctionGemma value string to a JSON value.
///
/// Tries JSON parsing first (for numbers, booleans, null), falling back to
/// treating the value as a plain string. This matches the Python parser's
/// `json.loads(value)` behaviour.
fn parse_functiongemma_scalar(value: &str) -> Value {
    if let Ok(parsed) = serde_json::from_str::<Value>(value) {
        return parsed;
    }
    Value::String(value.to_string())
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::{ESCAPE, FunctionGemmaToolParser, TOOL_CALL_END, TOOL_CALL_START, ToolParser};
    use crate::tool::ToolParserTestExt as _;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};

    fn build_tool_call(function_name: &str, args: &[(&str, &str)]) -> String {
        let args_str = args
            .iter()
            .map(|(key, value)| format!("{key}:{ESCAPE}{value}{ESCAPE}"))
            .collect::<Vec<_>>()
            .join(",");
        format!("{TOOL_CALL_START}call:{function_name}{{{args_str}}}{TOOL_CALL_END}")
    }

    #[test]
    fn parse_complete_without_tool_call_keeps_text() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn parse_complete_extracts_single_tool_call() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call("get_weather", &[("location", "London")]))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "London" })
        );
    }

    #[test]
    fn parse_complete_extracts_multiple_arguments() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[("location", "San Francisco"), ("unit", "celsius")],
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "San Francisco", "unit": "celsius" })
        );
    }

    #[test]
    fn parse_complete_parses_numeric_and_boolean_values() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "configure",
                &[("count", "42"), ("enabled", "true"), ("score", "114.514")],
            ))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "count": 42, "enabled": true, "score": 114.514 })
        );
    }

    #[test]
    fn parse_complete_preserves_prefix_text() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&format!(
                "Let me check. {}",
                build_tool_call("get_weather", &[("location", "Paris")])
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "Let me check. ");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn parse_complete_extracts_multiple_tool_calls() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let input = format!(
            "{}{}",
            build_tool_call("get_weather", &[("location", "London")]),
            build_tool_call("get_time", &[("timezone", "UTC")]),
        );
        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[1].name.as_deref(), Some("get_time"));
    }

    #[test]
    fn parse_complete_handles_empty_arguments() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&format!("{TOOL_CALL_START}call:noop{{}}{TOOL_CALL_END}"))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("noop"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({})
        );
    }

    #[test]
    fn parse_complete_rejects_incomplete_tool_call() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let error = parser
            .parse_complete(&format!(
                "{TOOL_CALL_START}call:get_weather{{location:{ESCAPE}London"
            ))
            .unwrap_err();

        assert!(error.to_report_string().contains("incomplete FunctionGemma tool call"));
    }

    #[test]
    fn streaming_emits_complete_tool_call() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                TOOL_CALL_START,
                "call:get_weather{",
                &format!("location:{ESCAPE}Paris{ESCAPE}"),
                "}",
                TOOL_CALL_END,
            ],
        );

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "Paris" })
        );
    }

    #[test]
    fn streaming_waits_for_complete_tool_call() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let mut output =
            parser.parse_chunk(&format!("{TOOL_CALL_START}call:get_weather{{")).unwrap();
        assert!(output.calls().is_empty());

        output.append(parser.parse_chunk(&format!("location:{ESCAPE}Paris{ESCAPE}}}")).unwrap());
        assert!(output.calls().is_empty());

        output.append(parser.parse_chunk(TOOL_CALL_END).unwrap());
        let output = output.coalesce();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn streaming_handles_marker_split_across_chunks() {
        let text = build_tool_call("get_weather", &[("location", "NYC")]);
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &split_by_chars(&text, 5));

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "NYC" })
        );
    }

    #[test]
    fn streaming_handles_end_marker_inside_string_value() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                TOOL_CALL_START,
                "call:log{",
                &format!("message:{ESCAPE}call ended {TOOL_CALL_END} ok{ESCAPE}"),
                "}",
                TOOL_CALL_END,
            ],
        );

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("log"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "message": format!("call ended {} ok", TOOL_CALL_END) })
        );
    }

    #[test]
    fn streaming_emits_text_before_and_after_tool_call() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                "I'll check. ",
                TOOL_CALL_START,
                "call:get_weather{",
                &format!("location:{ESCAPE}London{ESCAPE}"),
                "}",
                TOOL_CALL_END,
                " Done.",
            ],
        );

        assert_eq!(output.normal_text(), "I'll check.  Done.");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn streaming_extracts_multiple_tool_calls() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let input = format!(
            "{}{}",
            build_tool_call("get_weather", &[("location", "Shanghai")]),
            build_tool_call("add", &[("x", "1"), ("y", "2")]),
        );
        let output = collect_stream(&mut parser, &split_by_chars(&input, 7));

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
    fn finish_incomplete_tool_call_reports_error() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        parser.parse_chunk(&format!("{TOOL_CALL_START}call:search{{")).unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete FunctionGemma tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn preserve_special_tokens_is_true() {
        let parser = FunctionGemmaToolParser::new(&test_tools());
        assert!(parser.preserve_special_tokens());
    }

    #[test]
    fn reset_reconstructs_buffered_text() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        for chunk in [
            TOOL_CALL_START,
            "call:get_weather{",
            &format!("location:{ESCAPE}London{ESCAPE}"),
        ] {
            parser.parse_chunk(chunk).unwrap();
        }

        let raw = parser.reset();

        assert_eq!(
            raw,
            format!("{TOOL_CALL_START}call:get_weather{{location:{ESCAPE}London{ESCAPE}")
        );
    }

    #[test]
    fn reset_reconstructs_after_parse_error() {
        let mut parser = FunctionGemmaToolParser::new(&test_tools());
        let input = format!("{TOOL_CALL_START}call:bad{{broken}}{TOOL_CALL_END}");

        let _error = parser.parse_chunk(&input).unwrap_err();
        let raw = parser.reset();

        assert_eq!(raw, input);
    }
}
