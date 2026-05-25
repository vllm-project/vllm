use winnow::ascii::{digit1, multispace0 as ws0};
use winnow::combinator::{alt, eof, repeat, seq};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until, take_while};

use super::utils::{JsonObjectScanState, parse_buffered_event, safe_text_len, take_json_object};
use super::{Result, ToolCallDelta, ToolParseResult, ToolParser};
use crate::Tool;

const TOOL_CALLS_START: &str = "<|tool_calls_section_begin|>";
const TOOL_CALLS_END: &str = "<|tool_calls_section_end|>";
const TOOL_CALL_START: &str = "<|tool_call_begin|>";
const TOOL_CALL_END: &str = "<|tool_call_end|>";
const TOOL_CALL_ARGUMENT_START: &str = "<|tool_call_argument_begin|>";

type KimiK2Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum KimiK2Mode {
    Text,
    ToolBlock,
    Header,
    Arguments { json_scan: JsonObjectScanState },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum KimiK2Event {
    Text {
        len: usize,
    },
    ToolCallsStart,
    ToolCallStart,
    ToolCallHeader {
        function_name: String,
        function_index: usize,
    },
    Arguments {
        len: usize,
    },
    ToolCallEnd,
    ToolCallsEnd,
    IgnoredRest,
}

/// Tool parser for Kimi K2 token-delimited tool calls.
///
/// Example tool call content:
///
/// ```text
/// <|tool_calls_section_begin|>
/// <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|>
/// <|tool_calls_section_end|>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
pub struct KimiK2ToolParser {
    buffer: String,
    mode: KimiK2Mode,
    active_tool_index: Option<usize>,
}

impl KimiK2ToolParser {
    /// Create a Kimi K2 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: KimiK2Mode::Text,
            active_tool_index: None,
        }
    }

    /// Apply one parsed Kimi K2 event to parser state and output.
    fn apply_event(&mut self, event: KimiK2Event, result: &mut ToolParseResult) -> Result<()> {
        match event {
            KimiK2Event::Text { len: consumed_len } => {
                result.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            KimiK2Event::ToolCallsStart => self.mode = KimiK2Mode::ToolBlock,
            KimiK2Event::ToolCallStart => self.mode = KimiK2Mode::Header,
            KimiK2Event::ToolCallHeader {
                function_name,
                function_index,
            } => {
                let tool_index = function_index;
                self.active_tool_index = Some(tool_index);
                self.mode = KimiK2Mode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                };
                result.calls.push(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            KimiK2Event::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "Kimi K2 arguments without an active tool call"
                    ));
                };
                result.calls.push(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..consumed_len].to_string(),
                });
            }
            KimiK2Event::ToolCallEnd => {
                self.active_tool_index = None;
                self.mode = KimiK2Mode::ToolBlock;
            }
            KimiK2Event::ToolCallsEnd => {
                self.active_tool_index = None;
                self.mode = KimiK2Mode::Done;
            }
            KimiK2Event::IgnoredRest => {}
        }
        Ok(())
    }

    /// Reset all streaming state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.mode = KimiK2Mode::Text;
        self.active_tool_index = None;
    }
}

impl ToolParser for KimiK2ToolParser {
    /// Create a boxed Kimi K2 tool parser.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Preserve Kimi K2 special-token markers while decoding.
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    /// Push one decoded text chunk through the Kimi K2 parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_kimi_k2_event(input, &mut self.mode)
        })? {
            self.apply_event(event, &mut result)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(result)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        let mut result = ToolParseResult::default();
        match &self.mode {
            KimiK2Mode::Text => result.normal_text.push_str(&self.buffer),
            KimiK2Mode::ToolBlock | KimiK2Mode::Done => {}
            KimiK2Mode::Header | KimiK2Mode::Arguments { .. } => {
                return Err(parsing_failed!("incomplete Kimi K2 tool call"));
            }
        }
        self.reset();
        Ok(result)
    }
}

/// Parse a Kimi K2 event for the current parser mode.
fn parse_next_kimi_k2_event(
    input: &mut KimiK2Input<'_>,
    mode: &mut KimiK2Mode,
) -> ModalResult<KimiK2Event> {
    match mode {
        KimiK2Mode::Text => parse_text_event(input),
        KimiK2Mode::ToolBlock => parse_tool_block_event(input),
        KimiK2Mode::Header => tool_call_header_event(input),
        KimiK2Mode::Arguments { json_scan } => parse_arguments_event(input, json_scan),
        KimiK2Mode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode Kimi K2 event.
fn parse_text_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    alt((tool_calls_start_event, safe_text_event)).parse_next(input)
}

/// Parse one event inside the Kimi K2 tool-calls section.
fn parse_tool_block_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    alt((tool_calls_end_event, tool_call_start_event)).parse_next(input)
}

/// Parse one event inside a Kimi K2 tool-call arguments payload.
fn parse_arguments_event(
    input: &mut KimiK2Input<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<KimiK2Event> {
    if json_scan.complete() {
        tool_call_end_event(input)
    } else {
        argument_delta_event(input, json_scan)
    }
}

/// Parse a Kimi K2 tool-calls section start marker.
fn tool_calls_start_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    literal(TOOL_CALLS_START).value(KimiK2Event::ToolCallsStart).parse_next(input)
}

/// Parse a Kimi K2 tool-calls section end marker.
fn tool_calls_end_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    (ws0, literal(TOOL_CALLS_END))
        .value(KimiK2Event::ToolCallsEnd)
        .parse_next(input)
}

/// Parse a Kimi K2 tool-call start marker.
fn tool_call_start_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    (ws0, literal(TOOL_CALL_START))
        .value(KimiK2Event::ToolCallStart)
        .parse_next(input)
}

/// Parse a Kimi K2 tool-call end marker.
fn tool_call_end_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    literal(TOOL_CALL_END).value(KimiK2Event::ToolCallEnd).parse_next(input)
}

/// Parse a Kimi K2 tool-call header before the argument marker.
fn tool_call_header_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    let (header, _) = (
        take_until(1.., TOOL_CALL_ARGUMENT_START),
        literal(TOOL_CALL_ARGUMENT_START),
    )
        .parse_next(input)?;

    let mut header_input = header;
    let (header, _, _) = (tool_header, ws0, eof).parse_next(&mut header_input)?;

    Ok(KimiK2Event::ToolCallHeader {
        function_name: header.function_name,
        function_index: header.function_index,
    })
}

/// Parse a Kimi K2 raw JSON arguments delta.
fn argument_delta_event(
    input: &mut KimiK2Input<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<KimiK2Event> {
    take_json_object(input, json_scan).map(|len| KimiK2Event::Arguments { len })
}

/// Parse a safe text run before the next Kimi K2 tool-calls section.
fn safe_text_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    safe_text_len(input, TOOL_CALLS_START).map(|len| KimiK2Event::Text { len })
}

/// Parse ignored rest after the Kimi K2 tool-calls section ends.
fn ignored_rest_event(input: &mut KimiK2Input<'_>) -> ModalResult<KimiK2Event> {
    rest.value(KimiK2Event::IgnoredRest).parse_next(input)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KimiK2ToolHeader {
    function_name: String,
    function_index: usize,
}

/// Parse a Kimi K2 tool-call header.
fn tool_header(input: &mut &str) -> ModalResult<KimiK2ToolHeader> {
    let (function_name, function_index) = seq!(
        _: ws0,
        _: namespace_prefix,
        tool_name_segment,
        _: literal(":"),
        tool_call_index,
    )
    .parse_next(input)?;

    Ok(KimiK2ToolHeader {
        function_name: function_name.to_string(),
        function_index,
    })
}

/// Parse Kimi K2 namespace segments before the final tool name.
fn namespace_prefix(input: &mut &str) -> ModalResult<()> {
    repeat(0.., namespace_segment).parse_next(input)
}

/// Parse a Kimi K2 namespace segment.
fn namespace_segment<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    let (segment, _) = (tool_name_segment, literal(".")).parse_next(input)?;
    Ok(segment)
}

/// Parse a Kimi K2 tool name segment.
fn tool_name_segment<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    take_while(1.., |ch: char| {
        !ch.is_whitespace() && ch != '<' && ch != ':' && ch != '.'
    })
    .parse_next(input)
}

/// Parse a Kimi K2 tool-call index.
fn tool_call_index(input: &mut &str) -> ModalResult<usize> {
    digit1.parse_to().parse_next(input)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::{
        KimiK2ToolParser, TOOL_CALL_ARGUMENT_START, TOOL_CALL_END, TOOL_CALL_START, TOOL_CALLS_END,
        TOOL_CALLS_START, ToolParser, tool_header,
    };
    use crate::ToolParseResult;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};

    fn build_tool_call(function_name: &str, index: usize, arguments: &str) -> String {
        format!(
            "{TOOL_CALL_START}functions.{function_name}:{index}{TOOL_CALL_ARGUMENT_START}{arguments}{TOOL_CALL_END}"
        )
    }

    fn build_tool_section(tool_calls: &[String]) -> String {
        format!("{TOOL_CALLS_START}{}{TOOL_CALLS_END}", tool_calls.join(""))
    }

    #[test]
    fn kimi_k2_parse_complete_without_tool_call_keeps_text() {
        let mut parser = KimiK2ToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn kimi_k2_parse_complete_extracts_raw_json_arguments() {
        let mut parser = KimiK2ToolParser::new(&test_tools());
        let arguments = r#"{ "location": "NYC", "days": "3" }"#;
        let result = parser
            .parse_complete(&format!(
                "Checking. {} trailing text",
                build_tool_section(&[build_tool_call("get_weather", 0, arguments)])
            ))
            .unwrap();

        assert_eq!(result.normal_text, "Checking. ");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn kimi_k2_does_not_validate_or_normalize_arguments() {
        let mut parser = KimiK2ToolParser::new(&test_tools());
        let arguments = r#"{"location":"NYC",}"#;
        let result = parser
            .parse_complete(&build_tool_section(&[build_tool_call(
                "get_weather",
                0,
                arguments,
            )]))
            .unwrap();

        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn kimi_k2_streaming_emits_argument_deltas() {
        let mut parser = KimiK2ToolParser::new(&test_tools());
        let chunks = [
            TOOL_CALLS_START,
            TOOL_CALL_START,
            "functions.get_weather:0",
            TOOL_CALL_ARGUMENT_START,
            "{\"location\":",
            "\"Paris\"",
            "}",
            TOOL_CALL_END,
            TOOL_CALLS_END,
        ];

        let mut result = ToolParseResult::default();
        let mut observed_arguments = Vec::new();
        for chunk in chunks {
            let next = parser.push(chunk).unwrap();
            observed_arguments.extend(
                next.calls
                    .iter()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            result.append(next);
        }
        result.append(parser.finish().unwrap());

        assert_eq!(observed_arguments, ["{\"location\":", "\"Paris\"", "}"]);
        let result = result.coalesce_calls();
        assert_eq!(result.calls[0].arguments, r#"{"location":"Paris"}"#);
    }

    #[test]
    fn kimi_k2_streaming_holds_back_split_markers() {
        let mut parser = KimiK2ToolParser::new(&test_tools());
        let chunks = [
            "hello <|tool_calls",
            "_section_begin|>",
            TOOL_CALL_START,
            "functions.get_weather:0",
            TOOL_CALL_ARGUMENT_START,
            r#"{"location":"NYC"}"#,
            "<|tool_call",
            "_end|>",
            TOOL_CALLS_END,
        ];

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text, "hello ");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn kimi_k2_keeps_end_marker_literal_inside_json_string() {
        let mut parser = KimiK2ToolParser::new(&test_tools());
        let arguments = format!(r#"{{"text":"literal {TOOL_CALL_END} inside"}}"#);
        let input = build_tool_section(&[build_tool_call("echo", 0, &arguments)]);

        let result = parser.parse_complete(&input).unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn kimi_k2_streaming_keeps_split_end_marker_literal_inside_json_string() {
        let mut parser = KimiK2ToolParser::new(&test_tools());
        let chunks = [
            TOOL_CALLS_START,
            TOOL_CALL_START,
            "functions.echo:0",
            TOOL_CALL_ARGUMENT_START,
            r#"{"text":"literal <|tool"#,
            r#"_call_end|> inside"}"#,
            TOOL_CALL_END,
            TOOL_CALLS_END,
        ];

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.calls.len(), 1);
        assert_eq!(
            result.calls[0].arguments,
            r#"{"text":"literal <|tool_call_end|> inside"}"#
        );
    }

    #[test]
    fn kimi_k2_streaming_extracts_multiple_tool_calls() {
        let mut parser = KimiK2ToolParser::new(&test_tools());
        let input = build_tool_section(&[
            build_tool_call("get_weather", 0, r#"{"location":"Shanghai"}"#),
            build_tool_call("add", 1, r#"{"x":1,"y":2}"#),
        ]);

        let chunks = split_by_chars(&input, 7);
        let result = collect_stream(&mut parser, &chunks);

        expect![[r#"
            ToolParseResult {
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
        .assert_debug_eq(&result);
    }

    #[test]
    fn kimi_k2_accepts_non_functions_header_prefix() {
        let mut parser = KimiK2ToolParser::new(&test_tools());
        let input = format!(
            "{TOOL_CALLS_START}{TOOL_CALL_START}api.tools.search:42{TOOL_CALL_ARGUMENT_START}{{}}{TOOL_CALL_END}{TOOL_CALLS_END}"
        );

        let result = parser.parse_complete(&input).unwrap();

        assert_eq!(result.calls[0].tool_index, 42);
        assert_eq!(result.calls[0].name.as_deref(), Some("search"));
        assert_eq!(result.calls[0].arguments, "{}");
    }

    #[test]
    fn kimi_k2_tool_header_parses_namespace_function_and_index() {
        let mut input = "api.tools.search:42";
        let header = tool_header(&mut input).unwrap();

        expect![[r#"
            KimiK2ToolHeader {
                function_name: "search",
                function_index: 42,
            }
        "#]]
        .assert_debug_eq(&header);
    }

    #[test]
    fn kimi_k2_finish_fails_incomplete_tool_call() {
        let mut parser = KimiK2ToolParser::new(&test_tools());
        parser
            .push(&format!(
                "{TOOL_CALLS_START}{TOOL_CALL_START}functions.get_weather:0{TOOL_CALL_ARGUMENT_START}{{\"location\""
            ))
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete Kimi K2 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn kimi_k2_malformed_header_fails_fast() {
        let mut parser = KimiK2ToolParser::new(&test_tools());
        let input =
            format!("{TOOL_CALLS_START}{TOOL_CALL_START}get_weather{TOOL_CALL_ARGUMENT_START}{{}}");

        let error = parser.push(&input).unwrap_err();

        expect!["tool parser parsing failed: "].assert_eq(&error.to_report_string());
    }
}
