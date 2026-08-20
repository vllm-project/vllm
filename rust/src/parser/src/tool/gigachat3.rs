use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, seq};
use winnow::error::StrContext;
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::literal;

use super::utils::{
    JsonObjectScanState, json_str, parse_buffered_event, safe_text_len_mul, take_json_object,
};
use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};

// GigaChat 3.0 format:
//   [content]<|message_sep|>\n\nfunction call<|role_sep|>\n{"name":"...","arguments":{...}}
// GigaChat 3.1 format:
//   [content]<|function_call|>{"name":"...","arguments":{...}}
//
// Content delimiters mark the boundary between normal text and the tool-call
// preamble. Both formats share the same JSON payload structure with `name` and
// `arguments` keys.
//
// The ? separators in the GigaChat 3.0 format above represent special-token
// boundary bytes produced by the tokenizer when `skip_special_tokens` is
// disabled. These may decode as newlines (`\n`) or as null bytes (`\0`),
// depending on the tokenizer implementation. The parser normalises `\0`
// to `\n` on input so downstream marker matching uses a single expected
// delimiter form.

const MSG_SEP_DELIMITER: &str = "<|message_sep|>\n\n";
const FUNCTION_CALL_MARKER_V3: &str = "function call<|role_sep|>\n";
const FUNCTION_CALL_MARKER_V31: &str = "<|function_call|>";

/// All markers that can appear in GigaChat 3.x tool-call output, ordered for
/// `safe_text_len_mul` scanning in text mode.
const TEXT_MARKERS: &[&str] = &[
    MSG_SEP_DELIMITER,
    FUNCTION_CALL_MARKER_V31,
    FUNCTION_CALL_MARKER_V3,
];

type GigaChat3Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    /// Collecting normal text before a tool call.
    Text,
    /// Parsing the JSON header `{"name":"...","arguments":` after a start marker.
    ToolCallHeader,
    /// Streaming the raw JSON arguments object payload.
    ToolCallArguments { json_scan: JsonObjectScanState },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Event {
    /// Safe text run consumed from the buffer.
    Text { len: usize },
    /// A tool-call start marker was consumed.
    ToolCallStart,
    /// The function `name` key and value were parsed.
    ToolCallHeader { function_name: String },
    /// A fragment of raw JSON arguments text.
    Arguments { len: usize },
    /// The outer JSON close-brace was consumed; tool call is complete.
    ToolCallEnd,
}

/// Tool parser for GigaChat 3.x special-token wrapped JSON tool calls.
///
/// Example GigaChat 3.0 tool call:
///
/// ```text
/// Let me check that.<|message_sep|>
///
/// function call<|role_sep|>
/// {"name":"get_weather","arguments":{"location":"Moscow"}}
/// ```
///
/// Example GigaChat 3.1 tool call:
///
/// ```text
/// Let me check that.<|function_call|>{"name":"get_weather","arguments":{"location":"Moscow"}}
/// ```
///
/// GigaChat 3.x marker tokens are added-vocabulary tokens stored via
/// `tokenizer.add_tokens(...)`. They must survive decoding, so
/// `preserve_special_tokens()` returns `true`.
pub struct GigaChat3ToolParser {
    buffer: String,
    mode: Mode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
}

impl GigaChat3ToolParser {
    /// Create a GigaChat3 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: Mode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    /// Apply one parsed event to parser state and output.
    fn apply_event(&mut self, event: Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            Event::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            Event::ToolCallStart => {
                self.mode = Mode::ToolCallHeader;
            }
            Event::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = Mode::ToolCallArguments {
                    json_scan: JsonObjectScanState::default(),
                };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            Event::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!("arguments without an active tool call"));
                };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..consumed_len].to_string(),
                });
            }
            Event::ToolCallEnd => {
                self.active_tool_index = None;
                self.mode = Mode::Text;
            }
        }
        Ok(())
    }
}

impl ToolParser for GigaChat3ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// GigaChat 3.x markers are added-vocabulary tokens that must survive
    /// decoding. The Python parser sets `skip_special_tokens = False` in
    /// `adjust_request()`.
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        // When `preserve_special_tokens` is active (always true for
        // GigaChat 3.x), the tokenizer may decode special-token boundaries
        // as null bytes instead of newlines.  Normalise them so downstream
        // marker constants match regardless of the tokenizer variant.
        let sanitised: String = chunk.replace('\0', "\n");
        self.buffer.push_str(&sanitised);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_gigachat3_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match &self.mode {
            Mode::Text => output.push_text(&self.buffer),
            Mode::ToolCallHeader | Mode::ToolCallArguments { .. } => {
                return Err(parsing_failed!("incomplete GigaChat3 tool call"));
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = Mode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

/// Parse a GigaChat3 event for the current parser mode.
fn parse_next_gigachat3_event(
    input: &mut GigaChat3Input<'_>,
    mode: &mut Mode,
) -> ModalResult<Event> {
    match mode {
        Mode::Text => parse_text_event(input),
        Mode::ToolCallHeader => parse_tool_call_header_event(input),
        Mode::ToolCallArguments { json_scan } => parse_arguments_event(input, json_scan),
    }
}

/// Parse a text-mode event: safe text, content delimiter, or tool-call start.
fn parse_text_event(input: &mut GigaChat3Input<'_>) -> ModalResult<Event> {
    alt((
        content_delimiter_event,
        tool_call_start_v31_event,
        tool_call_start_v3_event,
        safe_text_event,
    ))
    .parse_next(input)
}

/// Parse safe text before any GigaChat3 marker.
fn safe_text_event(input: &mut GigaChat3Input<'_>) -> ModalResult<Event> {
    safe_text_len_mul(input, TEXT_MARKERS).map(|len| Event::Text { len })
}

/// Parse and silently consume the GigaChat 3.0 content delimiter.
fn content_delimiter_event(input: &mut GigaChat3Input<'_>) -> ModalResult<Event> {
    literal(MSG_SEP_DELIMITER).value(Event::Text { len: 0 }).parse_next(input)
}

/// Parse the GigaChat 3.1 tool-call start marker.
fn tool_call_start_v31_event(input: &mut GigaChat3Input<'_>) -> ModalResult<Event> {
    literal(FUNCTION_CALL_MARKER_V31).value(Event::ToolCallStart).parse_next(input)
}

/// Parse the GigaChat 3.0 tool-call start marker.
fn tool_call_start_v3_event(input: &mut GigaChat3Input<'_>) -> ModalResult<Event> {
    literal(FUNCTION_CALL_MARKER_V3).value(Event::ToolCallStart).parse_next(input)
}

/// Parse a JSON tool-call header.
fn parse_tool_call_header_event(input: &mut GigaChat3Input<'_>) -> ModalResult<Event> {
    let (function_name,) = seq!(
        _: ws0,
        _: literal("{"),
        _: ws0,
        _: literal(r#""name""#),
        _: ws0,
        _: literal(":"),
        _: ws0,
        json_str,
        _: ws0,
        _: literal(","),
        _: ws0,
        _: literal(r#""arguments""#),
        _: ws0,
        _: literal(":"),
        _: ws0,
    )
    .context(StrContext::Label("GigaChat3"))
    .parse_next(input)?;

    Ok(Event::ToolCallHeader { function_name })
}

/// Parse one event inside the raw JSON arguments payload.
///
/// After the inner arguments JSON object closes, consumes the trailing `}`
/// that closes the outer tool-call JSON object.
fn parse_arguments_event(
    input: &mut GigaChat3Input<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<Event> {
    if json_scan.complete() {
        return tool_call_close_event(input);
    }
    argument_delta_event(input, json_scan)
}

/// Consume the closing `}` of the outer JSON object after arguments complete.
fn tool_call_close_event(input: &mut GigaChat3Input<'_>) -> ModalResult<Event> {
    seq!(_: ws0, _: literal("}")).value(Event::ToolCallEnd).parse_next(input)
}

/// Parse a raw JSON arguments delta.
fn argument_delta_event(
    input: &mut GigaChat3Input<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<Event> {
    take_json_object(input, json_scan).map(|len| Event::Arguments { len })
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::GigaChat3ToolParser;
    use crate::tool::{
        ToolParser, ToolParserOutput, ToolParserTestExt as _,
        test_utils::{collect_stream, split_by_chars, test_tools},
    };

    const MSG_SEP: &str = "<|message_sep|>\n\n";
    const TOOL_HEADER_V3: &str = "function call<|role_sep|>\n";
    const TOOL_HEADER_V31: &str = "<|function_call|>";
    /// Null-byte token-boundary markers (the variant some tokenizers produce).
    const MSG_SEP_NUL: &str = "<|message_sep|>\0\0";
    const TOOL_HEADER_V3_NUL: &str = "function call<|role_sep|>\0";

    fn build_tool_call(name: &str, arguments: &str) -> String {
        format!(r#"{{"name":"{name}","arguments":{arguments}}}"#)
    }

    fn build_gigachat3_output(prefix: &str, name: &str, arguments: &str) -> String {
        format!(
            "{prefix}{MSG_SEP}{TOOL_HEADER_V3}{}",
            build_tool_call(name, arguments)
        )
    }

    fn build_gigachat31_output(prefix: &str, name: &str, arguments: &str) -> String {
        format!(
            "{prefix}{TOOL_HEADER_V31}{}",
            build_tool_call(name, arguments)
        )
    }

    // --- preserve_special_tokens ---

    #[test]
    fn preserves_special_tokens() {
        let parser = GigaChat3ToolParser::new(&test_tools());
        assert!(parser.preserve_special_tokens());
    }

    // --- null-byte token separators ---

    #[test]
    fn gigachat3_parse_complete_handles_null_byte_separators_v3() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let input = format!(
            "Let me check.{MSG_SEP_NUL}{TOOL_HEADER_V3_NUL}{{\"name\":\"get_weather\",\"arguments\":{{\"location\":\"Moscow\"}}}}"
        );
        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Moscow"}"#);
        assert_eq!(output.normal_text(), "Let me check.");
    }

    #[test]
    fn gigachat3_streaming_handles_null_byte_separators_v3() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let input = format!(
            "hello {MSG_SEP_NUL}{TOOL_HEADER_V3_NUL}{{\"name\":\"echo\",\"arguments\":{{\"x\":1}}}}"
        );
        let chunks = split_by_chars(&input, 5);
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "hello ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"x":1}"#);
    }

    // --- complete (non-streaming) ---

    #[test]
    fn parse_complete_without_tool_call_keeps_text() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let output = parser.parse_complete("How can I help?").unwrap();

        assert_eq!(output.normal_text(), "How can I help?");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn parse_complete_extracts_simple_tool_call_gigachat3() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_gigachat3_output(
                "",
                "get_weather",
                r#"{"location":"Moscow"}"#,
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Moscow"}"#);
    }

    #[test]
    fn parse_complete_extracts_simple_tool_call_gigachat31() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_gigachat31_output(
                "",
                "get_weather",
                r#"{"location":"Moscow"}"#,
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Moscow"}"#);
    }

    #[test]
    fn parse_complete_preserves_prefix_text() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let prefix = "Let me check that for you.";
        let arguments = r#"{"location":"Moscow"}"#;

        let output = parser
            .parse_complete(&build_gigachat3_output(prefix, "get_weather", arguments))
            .unwrap();

        assert_eq!(output.normal_text(), prefix);
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn parse_complete_preserves_prefix_text_gigachat31() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let prefix = "Let me check that for you.";
        let arguments = r#"{"location":"Moscow"}"#;

        let output = parser
            .parse_complete(&build_gigachat31_output(prefix, "get_weather", arguments))
            .unwrap();

        assert_eq!(output.normal_text(), prefix);
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn parse_complete_handles_no_arguments() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let output =
            parser.parse_complete(&build_gigachat3_output("", "get_weather", "{}")).unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, "{}");
    }

    #[test]
    fn parse_complete_handles_nested_arguments() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let arguments = r#"{"nested":{"key":"value"},"list":[1,2]}"#;
        let output = parser
            .parse_complete(&build_gigachat3_output("", "convert", arguments))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn does_not_validate_or_normalize_arguments() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let arguments = r#"{"location":"Moscow",}"#;
        let output = parser
            .parse_complete(&build_gigachat3_output("", "get_weather", arguments))
            .unwrap();

        assert_eq!(output.calls()[0].arguments, arguments);
    }

    #[test]
    fn parse_complete_handles_trailing_eos() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&format!(
                "{}</s>",
                build_gigachat3_output("", "get_weather", r#"{"location":"Moscow"}"#)
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Moscow"}"#);
    }

    // --- streaming ---

    #[test]
    fn streaming_emits_text_without_tool_call() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();

        output.append(parser.parse_chunk("How ").unwrap());
        output.append(parser.parse_chunk("can I ").unwrap());
        output.append(parser.parse_chunk("help?").unwrap());
        output.append(parser.finish().unwrap());

        assert_eq!(output.normal_text(), "How can I help?");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn streaming_preserves_prefix_text_gigachat3() {
        let input =
            build_gigachat3_output("Let me check.", "get_weather", r#"{"location":"Moscow"}"#);
        let chunks = split_by_chars(&input, 9);
        let mut parser = GigaChat3ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "Let me check.");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Moscow"}"#);
    }

    #[test]
    fn streaming_preserves_prefix_text_gigachat31() {
        let input =
            build_gigachat31_output("Let me check.", "get_weather", r#"{"location":"Moscow"}"#);
        let chunks = split_by_chars(&input, 9);
        let mut parser = GigaChat3ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "Let me check.");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn streaming_emits_argument_deltas() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let chunks = [
            "hello ",
            MSG_SEP,
            TOOL_HEADER_V3,
            r#"{"name":"get_weather","arguments":"#,
            r#"{"location":"#,
            r#""Moscow""#,
            r#"}"#,
            r#"}"#,
            " suffix",
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

        assert_eq!(
            observed_arguments,
            [r#"{"location":"#, r#""Moscow""#, r#"}"#]
        );
        assert_eq!(output.normal_text(), "hello  suffix");
        assert_eq!(
            output.coalesce().calls()[0].arguments,
            r#"{"location":"Moscow"}"#
        );
    }

    #[test]
    fn streaming_emits_argument_deltas_gigachat31() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let chunks = [
            "hello ",
            TOOL_HEADER_V31,
            r#"{"name":"get_weather","arguments":"#,
            r#"{"location":"#,
            r#""Moscow""#,
            r#"}"#,
            r#"}"#,
            " suffix",
        ];

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "hello  suffix");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Moscow"}"#);
    }

    #[test]
    fn streaming_extracts_multiple_tool_calls() {
        let input = format!(
            "{}{}",
            build_gigachat3_output("", "get_weather", r#"{"location":"Shanghai"}"#),
            build_gigachat3_output("", "add", r#"{"x":1,"y":2}"#),
        );
        let chunks = split_by_chars(&input, 11);
        let mut parser = GigaChat3ToolParser::new(&test_tools());

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
    fn streaming_handles_split_markers() {
        let input = build_gigachat3_output("hello ", "get_weather", r#"{"location":"Tokyo"}"#);
        let chunks = split_by_chars(&input, 5);
        let mut parser = GigaChat3ToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "hello ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn streaming_handles_embedded_marker_in_string() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let arguments = format!(r#"{{"text":"literal {} inside"}}"#, TOOL_HEADER_V3);
        let input = build_gigachat3_output("", "echo", &arguments);

        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, arguments);
    }

    // --- error paths ---

    #[test]
    fn finish_errors_on_incomplete_tool_call() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        parser
            .parse_chunk(&format!(
                "{MSG_SEP}{TOOL_HEADER_V3}{{\"name\":\"get_weather\",\"arguments\":{{\"location\""
            ))
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete GigaChat3 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn finish_errors_on_incomplete_tool_call_header() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        parser
            .parse_chunk(&format!("{TOOL_HEADER_V31}{{\"name\":\"get_weather\""))
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete GigaChat3 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn streaming_gigachat31_extracts_simple_tool_call() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let chunks = [
            TOOL_HEADER_V31,
            r#"{"name":"get_weather","arguments":{"location":"Moscow"}}"#,
        ];

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Moscow"}"#);
    }

    #[test]
    fn streaming_gigachat3_with_eos() {
        let mut parser = GigaChat3ToolParser::new(&test_tools());
        let chunks = [
            "I'll check that for you.",
            MSG_SEP,
            TOOL_HEADER_V3,
            r#"{"name":"get_weather","arguments":{"location":"Moscow"}}"#,
            "</s>",
        ];

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "I'll check that for you.</s>");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"location":"Moscow"}"#);
    }
}
