mod deepseek_v3;
mod deepseek_v31;

pub use deepseek_v3::DeepSeekV3ToolParser;
pub use deepseek_v31::DeepSeekV31ToolParser;
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, seq};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use super::utils::{JsonObjectScanState, parse_buffered_event, safe_text_len, take_json_object};
use super::{Result, ToolCallDelta, ToolParseResult};

pub(super) const TOOL_CALLS_START: &str = "<｜tool▁calls▁begin｜>";
pub(super) const TOOL_CALLS_END: &str = "<｜tool▁calls▁end｜>";
pub(super) const TOOL_CALL_START: &str = "<｜tool▁call▁begin｜>";
pub(super) const TOOL_CALL_END: &str = "<｜tool▁call▁end｜>";
pub(super) const TOOL_CALL_SEPARATOR: &str = "<｜tool▁sep｜>";
pub(super) const V3_JSON_START: &str = "\n```json\n";
pub(super) const V3_ARGUMENT_END: &str = "\n```<｜tool▁call▁end｜>";

type DeepSeekJsonInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeepSeekJsonFormat {
    V3,
    V31,
}

impl DeepSeekJsonFormat {
    /// Return the parser name used in diagnostics.
    const fn parser_name(self) -> &'static str {
        match self {
            Self::V3 => "DeepSeek V3",
            Self::V31 => "DeepSeek V3.1",
        }
    }

    /// Return the marker that closes the raw JSON arguments payload.
    const fn argument_end_marker(self) -> &'static str {
        match self {
            Self::V3 => V3_ARGUMENT_END,
            Self::V31 => TOOL_CALL_END,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeepSeekJsonMode {
    Text,
    ToolBlock,
    Header,
    Arguments { json_scan: JsonObjectScanState },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeepSeekJsonEvent {
    Text { len: usize },
    ToolCallsStart,
    ToolCallStart,
    ToolCallHeader { function_name: String },
    Arguments { len: usize },
    ToolCallEnd,
    ToolCallsEnd,
    IgnoredRest,
}

/// Tool parser core for DeepSeek JSON-argument tool calls.
struct DeepSeekJsonToolParser {
    buffer: String,
    mode: DeepSeekJsonMode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
    format: DeepSeekJsonFormat,
}

impl DeepSeekJsonToolParser {
    /// Create a parser for one DeepSeek JSON-argument format.
    fn new(format: DeepSeekJsonFormat) -> Self {
        Self {
            buffer: String::new(),
            mode: DeepSeekJsonMode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
            format,
        }
    }

    /// Apply one parsed DeepSeek JSON event to parser state and output.
    fn apply_event(
        &mut self,
        event: DeepSeekJsonEvent,
        result: &mut ToolParseResult,
    ) -> Result<()> {
        match event {
            DeepSeekJsonEvent::Text { len: consumed_len } => {
                result.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            DeepSeekJsonEvent::ToolCallsStart => self.mode = DeepSeekJsonMode::ToolBlock,
            DeepSeekJsonEvent::ToolCallStart => self.mode = DeepSeekJsonMode::Header,
            DeepSeekJsonEvent::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = DeepSeekJsonMode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                };
                result.calls.push(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            DeepSeekJsonEvent::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "{} arguments without an active tool call",
                        self.format.parser_name()
                    ));
                };
                result.calls.push(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..consumed_len].to_string(),
                });
            }
            DeepSeekJsonEvent::ToolCallEnd => {
                self.active_tool_index = None;
                self.mode = DeepSeekJsonMode::ToolBlock;
            }
            DeepSeekJsonEvent::ToolCallsEnd => {
                self.active_tool_index = None;
                self.mode = DeepSeekJsonMode::Done;
            }
            DeepSeekJsonEvent::IgnoredRest => {}
        }
        Ok(())
    }

    /// Push one decoded text chunk through the DeepSeek JSON parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_deepseek_json_event(input, &mut self.mode, self.format)
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
            DeepSeekJsonMode::Text => result.normal_text.push_str(&self.buffer),
            DeepSeekJsonMode::ToolBlock | DeepSeekJsonMode::Done => {}
            DeepSeekJsonMode::Header | DeepSeekJsonMode::Arguments { .. } => {
                return Err(parsing_failed!(
                    "incomplete {} tool call",
                    self.format.parser_name()
                ));
            }
        }
        self.reset();
        Ok(result)
    }

    /// Reset all streaming state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.mode = DeepSeekJsonMode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
    }
}

/// Parse a DeepSeek JSON event for the current parser mode.
fn parse_next_deepseek_json_event(
    input: &mut DeepSeekJsonInput<'_>,
    mode: &mut DeepSeekJsonMode,
    format: DeepSeekJsonFormat,
) -> ModalResult<DeepSeekJsonEvent> {
    match mode {
        DeepSeekJsonMode::Text => parse_text_event(input),
        DeepSeekJsonMode::ToolBlock => parse_tool_block_event(input),
        DeepSeekJsonMode::Header => tool_call_header_event(input, format),
        DeepSeekJsonMode::Arguments { json_scan } => {
            parse_arguments_event(input, format, json_scan)
        }
        DeepSeekJsonMode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode DeepSeek JSON event.
fn parse_text_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    alt((tool_calls_start_event, safe_text_event)).parse_next(input)
}

/// Parse one event inside the DeepSeek tool-calls section.
fn parse_tool_block_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    ws0.void().parse_next(input)?;
    alt((tool_calls_end_event, tool_call_start_event)).parse_next(input)
}

/// Parse one event inside a DeepSeek tool-call arguments payload.
fn parse_arguments_event(
    input: &mut DeepSeekJsonInput<'_>,
    format: DeepSeekJsonFormat,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<DeepSeekJsonEvent> {
    if json_scan.complete() {
        tool_call_end_event(input, format)
    } else {
        argument_delta_event(input, json_scan)
    }
}

/// Parse a DeepSeek tool-calls start marker.
fn tool_calls_start_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    literal(TOOL_CALLS_START)
        .value(DeepSeekJsonEvent::ToolCallsStart)
        .parse_next(input)
}

/// Parse a DeepSeek tool-calls end marker.
fn tool_calls_end_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    literal(TOOL_CALLS_END).value(DeepSeekJsonEvent::ToolCallsEnd).parse_next(input)
}

/// Parse a DeepSeek tool-call start marker.
fn tool_call_start_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    literal(TOOL_CALL_START)
        .value(DeepSeekJsonEvent::ToolCallStart)
        .parse_next(input)
}

/// Parse a DeepSeek tool-call end marker.
fn tool_call_end_event(
    input: &mut DeepSeekJsonInput<'_>,
    format: DeepSeekJsonFormat,
) -> ModalResult<DeepSeekJsonEvent> {
    literal(format.argument_end_marker())
        .value(DeepSeekJsonEvent::ToolCallEnd)
        .parse_next(input)
}

/// Parse a DeepSeek tool-call header before the JSON arguments payload.
fn tool_call_header_event(
    input: &mut DeepSeekJsonInput<'_>,
    format: DeepSeekJsonFormat,
) -> ModalResult<DeepSeekJsonEvent> {
    match format {
        DeepSeekJsonFormat::V3 => v3_tool_call_header_event(input),
        DeepSeekJsonFormat::V31 => v31_tool_call_header_event(input),
    }
}

/// Parse a DeepSeek V3 tool-call header.
fn v3_tool_call_header_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    let name = seq!(
        _: literal("function"),
        _: literal(TOOL_CALL_SEPARATOR),
        take_until(1.., V3_JSON_START),
        _: literal(V3_JSON_START),
    )
    .parse_next(input)?;

    Ok(DeepSeekJsonEvent::ToolCallHeader {
        function_name: name.0.trim().to_string(),
    })
}

/// Parse a DeepSeek V3.1 tool-call header.
fn v31_tool_call_header_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    let (name, _) = (
        take_until(1.., TOOL_CALL_SEPARATOR),
        literal(TOOL_CALL_SEPARATOR),
    )
        .parse_next(input)?;

    Ok(DeepSeekJsonEvent::ToolCallHeader {
        function_name: name.trim().to_string(),
    })
}

/// Parse a DeepSeek raw JSON arguments delta.
fn argument_delta_event(
    input: &mut DeepSeekJsonInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<DeepSeekJsonEvent> {
    take_json_object(input, json_scan).map(|len| DeepSeekJsonEvent::Arguments { len })
}

/// Parse a safe text run before the next DeepSeek tool-calls section.
fn safe_text_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    safe_text_len(input, TOOL_CALLS_START).map(|len| DeepSeekJsonEvent::Text { len })
}

/// Parse ignored rest after the DeepSeek tool-calls section ends.
fn ignored_rest_event(input: &mut DeepSeekJsonInput<'_>) -> ModalResult<DeepSeekJsonEvent> {
    rest.value(DeepSeekJsonEvent::IgnoredRest).parse_next(input)
}
