use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, seq};
use winnow::error::ModalResult;
use winnow::prelude::*;
use winnow::token::literal;

use super::{
    JsonToolCallConfig, JsonToolCallEvent, JsonToolCallWhitespace, JsonToolInput,
    argument_delta_event, tool_call_header_event,
};
use crate::utils::{JsonObjectScanState, parse_buffered_event, safe_text_len};
use crate::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};

/// Per-call marker (plain generated text, not a special token).
const FUNCTION_CALL: &str = "<function_call>";

#[derive(Debug, Clone, PartialEq, Eq)]
enum Granite20bFcMode {
    Text,
    Header,
    Arguments {
        json_scan: JsonObjectScanState,
    },
    /// Seek the next marker, dropping any inter-call / trailing text.
    BetweenCalls,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Granite20bFcEvent {
    Text { len: usize },
    SkipText,
    ToolCallStart,
    ToolCallHeader { function_name: String },
    Arguments { len: usize },
    ToolCallEnd,
}

/// Tool parser for the `granite-20b-functioncalling` model.
///
/// Example tool call content:
///
/// ```text
/// <function_call> {"name": "get_weather", "arguments": {"city": "Tokyo"}}
/// ```
///
/// Each call is introduced by a `<function_call>` marker (repeated for parallel
/// calls; no closing marker); arguments are streamed verbatim, and only the text
/// before the first marker is content. Unlike `granite4`, `<function_call>` is
/// plain generated text, so `preserve_special_tokens` stays `false`.
pub struct Granite20bFcToolParser {
    buffer: String,
    mode: Granite20bFcMode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
}

impl Granite20bFcToolParser {
    /// Create a Granite 20B FC tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: Granite20bFcMode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    /// Apply one parsed event to parser state and output.
    fn apply_event(
        &mut self,
        event: Granite20bFcEvent,
        output: &mut ToolParserOutput,
    ) -> Result<()> {
        match event {
            Granite20bFcEvent::Text { len } => output.normal_text.push_str(&self.buffer[..len]),
            // Inter-call / trailing text after the first call is dropped, not content.
            Granite20bFcEvent::SkipText => {}
            Granite20bFcEvent::ToolCallStart => self.mode = Granite20bFcMode::Header,
            Granite20bFcEvent::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = Granite20bFcMode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                };
                output.calls.push(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            Granite20bFcEvent::Arguments { len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "Granite 20B FC arguments without an active tool call"
                    ));
                };
                output.calls.push(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..len].to_string(),
                });
            }
            Granite20bFcEvent::ToolCallEnd => {
                self.active_tool_index = None;
                self.mode = Granite20bFcMode::BetweenCalls;
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = Granite20bFcMode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for Granite20bFcToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_granite_20b_fc_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match &self.mode {
            Granite20bFcMode::Text => output.normal_text.push_str(&self.buffer),
            // Trailing/inter-call text after the last call is dropped.
            Granite20bFcMode::BetweenCalls => {}
            Granite20bFcMode::Header | Granite20bFcMode::Arguments { .. } => {
                return Err(parsing_failed!("incomplete Granite 20B FC tool call"));
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        Granite20bFcToolParser::reset(self)
    }
}

/// Parse one event for the current parser mode.
fn parse_next_granite_20b_fc_event(
    input: &mut JsonToolInput<'_>,
    mode: &mut Granite20bFcMode,
) -> ModalResult<Granite20bFcEvent> {
    match mode {
        Granite20bFcMode::Text => text_event(input),
        Granite20bFcMode::Header => header_event(input),
        Granite20bFcMode::Arguments { json_scan } => arguments_event(input, json_scan),
        Granite20bFcMode::BetweenCalls => between_calls_event(input),
    }
}

/// Emit leading content, or start the first call on `<function_call>`.
fn text_event(input: &mut JsonToolInput<'_>) -> ModalResult<Granite20bFcEvent> {
    alt((
        |input: &mut JsonToolInput<'_>| {
            seq!(_: literal(FUNCTION_CALL), _: ws0)
                .value(Granite20bFcEvent::ToolCallStart)
                .parse_next(input)
        },
        |input: &mut JsonToolInput<'_>| {
            safe_text_len(input, FUNCTION_CALL).map(|len| Granite20bFcEvent::Text { len })
        },
    ))
    .parse_next(input)
}

/// Parse the `{"name":"X","arguments":` header before the value.
fn header_event(input: &mut JsonToolInput<'_>) -> ModalResult<Granite20bFcEvent> {
    const CONFIG: JsonToolCallConfig = JsonToolCallConfig {
        parser_name: "Granite 20B FC",
        start_marker: "",
        end_marker: "",
        marker_whitespace: JsonToolCallWhitespace::Optional,
        delimiter: None,
        name_key: "name",
        arguments_key: &["arguments"],
    };

    match tool_call_header_event(input, CONFIG)? {
        JsonToolCallEvent::ToolCallHeader { function_name } => {
            Ok(Granite20bFcEvent::ToolCallHeader { function_name })
        }
        _ => unreachable!("tool_call_header_event only emits ToolCallHeader"),
    }
}

/// Stream the arguments object, then consume the call object's closing brace.
fn arguments_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<Granite20bFcEvent> {
    if json_scan.complete() {
        seq!(_: ws0, _: literal("}"))
            .value(Granite20bFcEvent::ToolCallEnd)
            .parse_next(input)
    } else {
        match argument_delta_event(input, json_scan)? {
            JsonToolCallEvent::Arguments { len } => Ok(Granite20bFcEvent::Arguments { len }),
            _ => unreachable!("argument_delta_event only emits Arguments"),
        }
    }
}

/// Start the next `<function_call>`, or drop any inter-call / trailing text before it.
fn between_calls_event(input: &mut JsonToolInput<'_>) -> ModalResult<Granite20bFcEvent> {
    alt((
        |input: &mut JsonToolInput<'_>| {
            seq!(_: literal(FUNCTION_CALL), _: ws0)
                .value(Granite20bFcEvent::ToolCallStart)
                .parse_next(input)
        },
        |input: &mut JsonToolInput<'_>| {
            safe_text_len(input, FUNCTION_CALL).map(|_| Granite20bFcEvent::SkipText)
        },
    ))
    .parse_next(input)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::Granite20bFcToolParser;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    #[test]
    fn granite_20b_fc_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Granite20bFcToolParser::new(&test_tools());
        let output = parser.parse_complete("Just a normal response, no tools here.").unwrap();

        assert_eq!(output.normal_text, "Just a normal response, no tools here.");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn granite_20b_fc_parse_complete_single_call() {
        let mut parser = Granite20bFcToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"<function_call> {"name": "get_weather", "arguments": {"city": "Tokyo"}}"#,
            )
            .unwrap();

        assert_eq!(output.normal_text, "");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[0].arguments, r#"{"city": "Tokyo"}"#);
    }

    #[test]
    fn granite_20b_fc_parse_complete_parallel_calls() {
        let mut parser = Granite20bFcToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<function_call> {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n\
                 <function_call> {\"name\": \"get_time\", \"arguments\": {\"timezone\": \"Asia/Tokyo\"}}",
            )
            .unwrap();

        assert_eq!(output.normal_text, "");
        assert_eq!(output.calls.len(), 2);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[0].arguments, r#"{"city": "Tokyo"}"#);
        assert_eq!(output.calls[1].name.as_deref(), Some("get_time"));
        assert_eq!(output.calls[1].arguments, r#"{"timezone": "Asia/Tokyo"}"#);
    }

    #[test]
    fn granite_20b_fc_drops_text_between_and_after_calls() {
        let mut parser = Granite20bFcToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<function_call> {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}} and then <function_call> {\"name\": \"get_time\", \"arguments\": {}} trailing words",
            )
            .unwrap();

        // Matches Python: both calls returned; inter-call ("and then") and
        // trailing ("trailing words") text is dropped (content is prefix-only).
        assert_eq!(output.normal_text, "");
        assert_eq!(output.calls.len(), 2);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[1].name.as_deref(), Some("get_time"));
    }

    #[test]
    fn granite_20b_fc_extracts_leading_content() {
        let mut parser = Granite20bFcToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "Let me check the weather for you.\n\
                 <function_call> {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}",
            )
            .unwrap();

        assert_eq!(output.normal_text, "Let me check the weather for you.\n");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[0].arguments, r#"{"city": "Tokyo"}"#);
    }

    #[test]
    fn granite_20b_fc_parse_complete_preserves_complex_arguments_verbatim() {
        let mut parser = Granite20bFcToolParser::new(&test_tools());
        let arguments = r#"{"quoted":"He said \"hi\" with ] and }","i":42,"f":3.14,"b":true,"nothing":null,"arr":["a","b","c"],"obj":{"nested":"value"},"empty_arr":[],"empty_obj":{}}"#;
        let output = parser
            .parse_complete(&format!(
                r#"<function_call> {{"name":"test_function","arguments":{arguments}}}"#
            ))
            .unwrap();

        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("test_function"));
        assert_eq!(output.calls[0].arguments, arguments);
    }

    #[test]
    fn granite_20b_fc_handles_multiline_json_arguments() {
        let mut parser = Granite20bFcToolParser::new(&test_tools());
        let arguments = "{\n    \"city\": \"Tokyo\",\n    \"days\": 3\n  }";
        let input = format!(
            "<function_call> {{\n  \"name\": \"get_weather\",\n  \"arguments\": {arguments}\n}}"
        );
        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[0].arguments, arguments);
    }

    #[test]
    fn granite_20b_fc_parse_complete_empty_arguments() {
        let mut parser = Granite20bFcToolParser::new(&test_tools());
        let output = parser
            .parse_complete(r#"<function_call> {"name": "refresh", "arguments": {}}"#)
            .unwrap();

        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("refresh"));
        assert_eq!(output.calls[0].arguments, "{}");
    }

    #[test]
    fn granite_20b_fc_streaming_handles_split_markers() {
        let input = r#"<function_call> {"name":"get_weather","arguments":{"city":"Tokyo"}}"#;
        let chunks = split_by_chars(input, 5);
        let mut parser = Granite20bFcToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text, "");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[0].arguments, r#"{"city":"Tokyo"}"#);
    }

    #[test]
    fn granite_20b_fc_streaming_emits_object_argument_deltas() {
        let mut parser = Granite20bFcToolParser::new(&test_tools());
        let chunks = [
            r#"<function_call> {"name":"get_weather","arguments":"#,
            r#"{"city":"#,
            r#""Beijing""#,
            r#"}}"#,
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

        assert_eq!(observed_arguments, [r#"{"city":"#, r#""Beijing""#, r#"}"#]);
        assert_eq!(
            output.coalesce_calls().calls[0].arguments,
            r#"{"city":"Beijing"}"#
        );
    }

    #[test]
    fn granite_20b_fc_finish_fails_incomplete_tool_call() {
        let mut parser = Granite20bFcToolParser::new(&test_tools());
        parser
            .parse_chunk(r#"<function_call> {"name":"get_weather","arguments":{"city""#)
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete Granite 20B FC tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn granite_20b_fc_rejects_array_arguments() {
        let mut parser = Granite20bFcToolParser::new(&test_tools());
        let error = parser
            .parse_chunk(r#"<function_call> [{"name": "f", "arguments": {}}]"#)
            .unwrap_err();

        expect!["tool parser parsing failed: invalid Granite 20B FC"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn granite_20b_fc_preserve_special_tokens_is_false() {
        let parser = Granite20bFcToolParser::new(&test_tools());
        assert!(!parser.preserve_special_tokens());
    }
}
