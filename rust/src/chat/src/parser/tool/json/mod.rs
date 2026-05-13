//! Shared parser core for JSON tool calls wrapped by text markers.

pub use hermes::HermesToolParser;
pub use llama::Llama3JsonToolParser;
pub use mistral::MistralToolParser;
pub use qwen::Qwen3XmlToolParser;

mod hermes;
mod llama;
mod mistral;
mod qwen;

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, seq};
use winnow::error::{ModalResult, StrContext, StrContextValue};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::literal;

use super::utils::{
    JsonObjectScanState, json_str, parse_buffered_event, safe_text_len, take_json_object,
};
use super::{Result, ToolCallDelta, ToolParseResult, ToolParserError, parsing_failed};

type JsonToolInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, Copy)]
struct JsonToolCallConfig {
    parser_name: &'static str,
    start_marker: &'static str,
    end_marker: &'static str,
    marker_whitespace: JsonToolCallWhitespace,
    delimiter: Option<&'static str>,
    name_key: &'static str,
    arguments_key: &'static str,
}

#[derive(Debug, Clone, Copy)]
enum JsonToolCallWhitespace {
    Optional,
    Exact(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonToolCallMode {
    Text,
    Header,
    Arguments { json_scan: JsonObjectScanState },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonToolCallEvent {
    Text { len: usize },
    ToolCallStart,
    ToolCallHeader { function_name: String },
    Arguments { len: usize },
    ToolCallDelimiter,
    ToolCallEnd,
}

/// Tool parser core for marker-wrapped JSON tool calls.
#[derive(Debug)]
struct JsonToolCallParser {
    config: JsonToolCallConfig,
    buffer: String,
    mode: JsonToolCallMode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
}

impl JsonToolCallParser {
    /// Create a marker-wrapped JSON tool-call parser.
    fn new(config: JsonToolCallConfig) -> Self {
        Self {
            config,
            buffer: String::new(),
            mode: JsonToolCallMode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    /// Push one decoded text chunk through the JSON tool-call parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();
        let config = self.config;

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_json_tool_call_event(input, &mut self.mode, config)
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
            JsonToolCallMode::Text => result.normal_text.push_str(&self.buffer),
            JsonToolCallMode::Header | JsonToolCallMode::Arguments { .. } => {
                return Err(parsing_failed!(
                    "incomplete {} tool call",
                    self.config.parser_name
                ));
            }
        }
        self.reset();
        Ok(result)
    }

    /// Apply one parsed JSON tool-call event to parser state and output.
    fn apply_event(
        &mut self,
        event: JsonToolCallEvent,
        result: &mut ToolParseResult,
    ) -> Result<()> {
        match event {
            JsonToolCallEvent::Text { len: consumed_len } => {
                result.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            JsonToolCallEvent::ToolCallStart => self.mode = JsonToolCallMode::Header,
            JsonToolCallEvent::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = JsonToolCallMode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                };
                result.calls.push(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            JsonToolCallEvent::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "{} arguments without an active tool call",
                        self.config.parser_name
                    ));
                };
                result.calls.push(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..consumed_len].to_string(),
                });
            }
            JsonToolCallEvent::ToolCallDelimiter => {
                self.active_tool_index = None;
                self.mode = JsonToolCallMode::Header;
            }
            JsonToolCallEvent::ToolCallEnd => {
                self.active_tool_index = None;
                self.mode = JsonToolCallMode::Text;
            }
        }
        Ok(())
    }

    /// Reset all streaming state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.mode = JsonToolCallMode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
    }
}

/// Parse a JSON tool-call event for the current parser mode.
fn parse_next_json_tool_call_event(
    input: &mut JsonToolInput<'_>,
    mode: &mut JsonToolCallMode,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    match mode {
        JsonToolCallMode::Text => parse_text_event(input, config),
        JsonToolCallMode::Header => tool_call_header_event(input, config),
        JsonToolCallMode::Arguments { json_scan } => {
            parse_arguments_event(input, json_scan, config)
        }
    }
}

/// Parse a text-mode JSON tool-call event.
fn parse_text_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    alt((
        |input: &mut JsonToolInput<'_>| tool_call_start_event(input, config),
        |input: &mut JsonToolInput<'_>| safe_text_event(input, config),
    ))
    .parse_next(input)
}

/// Parse a marker-wrapped JSON tool-call start marker.
fn tool_call_start_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    seq!(
        _: literal(config.start_marker),
        _: |input: &mut JsonToolInput<'_>| marker_whitespace(input, config),
    )
    .value(JsonToolCallEvent::ToolCallStart)
    .parse_next(input)
}

/// Parse a marker-wrapped JSON tool-call header before the raw arguments
/// payload.
fn tool_call_header_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    let (function_name,) = seq!(
        _: ws0,
        _: literal("{"),
        _: ws0,
        _: |input: &mut JsonToolInput<'_>| json_key(input, config.name_key),
        _: ws0,
        _: literal(":"),
        _: ws0,
        json_str,
        _: ws0,
        _: literal(","),
        _: ws0,
        _: |input: &mut JsonToolInput<'_>| json_key(input, config.arguments_key),
        _: ws0,
        _: literal(":"),
        _: ws0,
    )
    .context(StrContext::Label(config.parser_name))
    .parse_next(input)?;

    Ok(JsonToolCallEvent::ToolCallHeader { function_name })
}

/// Parse a configured JSON object key.
fn json_key(input: &mut JsonToolInput<'_>, key: &'static str) -> ModalResult<()> {
    seq!(
        _: literal("\""),
        _: literal(key).context(StrContext::Expected(StrContextValue::StringLiteral(key))),
        _: literal("\""),
    )
    .void()
    .parse_next(input)
}

/// Parse one event inside a marker-wrapped JSON tool-call arguments payload.
fn parse_arguments_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    if json_scan.complete() {
        tool_call_close_event(input, config)
    } else {
        argument_delta_event(input, json_scan)
    }
}

/// Parse a raw JSON arguments delta.
fn argument_delta_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<JsonToolCallEvent> {
    take_json_object(input, json_scan).map(|len| JsonToolCallEvent::Arguments { len })
}

/// Parse a marker-wrapped JSON tool-call close marker.
fn tool_call_close_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    let _ = literal("}").parse_next(input)?;

    match config.delimiter {
        Some(delimiter) => alt((
            |input: &mut JsonToolInput<'_>| tool_call_end_event(input, config),
            |input: &mut JsonToolInput<'_>| tool_call_delimiter_event(input, delimiter),
        ))
        .parse_next(input),
        None => tool_call_end_event(input, config),
    }
}

/// Parse a marker-wrapped JSON tool-call end marker.
fn tool_call_end_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    seq!(
        _: |input: &mut JsonToolInput<'_>| marker_whitespace(input, config),
        _: literal(config.end_marker),
    )
    .value(JsonToolCallEvent::ToolCallEnd)
    .parse_next(input)
}

/// Parse a delimiter between JSON tool calls inside one marker block.
fn tool_call_delimiter_event(
    input: &mut JsonToolInput<'_>,
    delimiter: &'static str,
) -> ModalResult<JsonToolCallEvent> {
    seq!(
        _: ws0,
        _: literal(delimiter),
        _: ws0,
    )
    .value(JsonToolCallEvent::ToolCallDelimiter)
    .parse_next(input)
}

/// Parse configured whitespace around a marker-wrapped JSON tool call.
fn marker_whitespace(input: &mut JsonToolInput<'_>, config: JsonToolCallConfig) -> ModalResult<()> {
    match config.marker_whitespace {
        JsonToolCallWhitespace::Optional => ws0.void().parse_next(input),
        JsonToolCallWhitespace::Exact(whitespace) => literal(whitespace).void().parse_next(input),
    }
}

/// Parse a safe text run before the next marker-wrapped JSON tool call.
fn safe_text_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    safe_text_len(input, config.start_marker).map(|len| JsonToolCallEvent::Text { len })
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
    use crate::parser::tool::ToolParseResult;

    const DELIMITED_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
        parser_name: "Delimited JSON",
        start_marker: "<tool_calls>",
        end_marker: "</tool_calls>",
        marker_whitespace: JsonToolCallWhitespace::Optional,
        delimiter: Some("<"),
        name_key: "function",
        arguments_key: "parameters",
    };

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(r#"{{"function":"{function_name}","parameters":{arguments}}}"#)
    }

    fn build_tool_calls(tool_calls: &[String]) -> String {
        format!("<tool_calls>{}</tool_calls>", tool_calls.join(" <\n"))
    }

    fn collect_chunks(parser: &mut JsonToolCallParser, chunks: &[&str]) -> ToolParseResult {
        let mut result = ToolParseResult::default();
        for chunk in chunks {
            result.append(parser.push(chunk).unwrap());
        }
        result.append(parser.finish().unwrap());
        result.coalesce_calls()
    }

    #[test]
    fn json_tool_call_delimiter_extracts_multiple_calls_in_one_block() {
        let input = build_tool_calls(&[
            build_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            build_tool_call("add", r#"{"x":1,"y":2}"#),
        ]);
        let mut parser = JsonToolCallParser::new(DELIMITED_CONFIG);

        let result = collect_chunks(&mut parser, &[&input]);

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
    fn json_tool_call_delimiter_can_arrive_in_later_chunk() {
        let mut parser = JsonToolCallParser::new(DELIMITED_CONFIG);
        let chunks = [
            r#"<tool_calls>{"function":"get_weather","parameters":{"location":"Shanghai"}}"#,
            " <\n",
            r#"{"function":"add","parameters":{"x":1,"y":2}}"#,
            "</tool_calls>",
        ];

        let result = collect_chunks(&mut parser, &chunks);

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
    fn json_tool_call_end_marker_wins_over_delimiter_prefix() {
        let mut parser = JsonToolCallParser::new(DELIMITED_CONFIG);
        let chunks = [
            r#"<tool_calls>{"function":"get_weather","parameters":{"location":"Shanghai"}}"#,
            " ",
            "</tool_calls> trailing text",
        ];

        let result = collect_chunks(&mut parser, &chunks);

        expect![[r#"
            ToolParseResult {
                normal_text: " trailing text",
                calls: [
                    ToolCallDelta {
                        tool_index: 0,
                        name: Some(
                            "get_weather",
                        ),
                        arguments: "{\"location\":\"Shanghai\"}",
                    },
                ],
            }
        "#]]
        .assert_debug_eq(&result);
    }
}
