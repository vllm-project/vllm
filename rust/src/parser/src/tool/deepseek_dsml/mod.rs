// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, eof, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use super::parameters::ToolSchemas;
use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len, take_until_marker};
use super::{Result, ToolCallDelta, ToolParserOutput};
use crate::tool::Tool;

mod deepseek_v32;
mod deepseek_v4;

pub use deepseek_v4::DeepSeekV4ToolParser;
pub use deepseek_v32::DeepSeekV32ToolParser;

const INVOKE_START: &str = "<｜DSML｜invoke";
const INVOKE_END: &str = "</｜DSML｜invoke>";
const PARAMETER_START: &str = "<｜DSML｜parameter";
const PARAMETER_END: &str = "</｜DSML｜parameter>";

type DsmlInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, Copy)]
struct DsmlTokens {
    tool_calls_start: &'static str,
    tool_calls_end: &'static str,
}

impl DsmlTokens {
    const V32: Self = Self {
        tool_calls_start: "<｜DSML｜function_calls>",
        tool_calls_end: "</｜DSML｜function_calls>",
    };
    const V4: Self = Self {
        tool_calls_start: "<｜DSML｜tool_calls>",
        tool_calls_end: "</｜DSML｜tool_calls>",
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DsmlMode {
    Text,
    ToolBlock { invoke_end_scan: MarkerScanState },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DsmlEvent {
    Text {
        len: usize,
    },
    ToolCallsStart,
    Invoke {
        name: String,
        raw_params: Vec<DsmlParameter>,
    },
    ToolCallsEnd,
    IgnoredRest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DsmlParameter {
    name: String,
    value: String,
    is_string: bool,
}

/// Tool parser core for DeepSeek DSML tool calls.
struct DeepSeekDsmlToolParser {
    buffer: String,
    mode: DsmlMode,
    emitted_invoke_count: usize,
    tool_parameters: ToolSchemas,
    tokens: DsmlTokens,
}

impl DeepSeekDsmlToolParser {
    /// Create a parser with DSML tokens for one DeepSeek format.
    fn new(tools: &[Tool], tokens: DsmlTokens) -> Self {
        Self {
            buffer: String::new(),
            mode: DsmlMode::Text,
            emitted_invoke_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
            tokens,
        }
    }

    /// Apply one parsed DSML event to parser state and output.
    fn apply_event(&mut self, event: DsmlEvent, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            DsmlEvent::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            DsmlEvent::ToolCallsStart => {
                self.mode = DsmlMode::ToolBlock {
                    invoke_end_scan: MarkerScanState::default(),
                };
            }
            DsmlEvent::Invoke { name, raw_params } => {
                let mut arguments = serde_json::Map::with_capacity(raw_params.len());
                for param in raw_params {
                    let value = if param.is_string {
                        serde_json::Value::String(param.value)
                    } else {
                        self.tool_parameters.convert_param_with_schema(
                            &name,
                            &param.name,
                            param.value,
                        )
                    };
                    arguments.insert(param.name, value);
                }
                let arguments = serde_json::to_string(&arguments)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                output.push_call(ToolCallDelta {
                    tool_index: self.emitted_invoke_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_invoke_count += 1;
            }
            DsmlEvent::ToolCallsEnd => self.mode = DsmlMode::Done,
            DsmlEvent::IgnoredRest => {}
        };
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = DsmlMode::Text;
        self.emitted_invoke_count = 0;
        std::mem::take(&mut self.buffer)
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        // Extract tool calls from streaming model output.
        //
        // Uses a buffer-until-complete-invoke strategy: text is buffered until
        // a complete invoke block is available, then parsed and emitted in one
        // shot.
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_dsml_event(input, &mut self.mode, self.tokens)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            DsmlMode::Text => output.push_text(&self.buffer),
            DsmlMode::Done => {}
            DsmlMode::ToolBlock { .. } => {
                return Err(parsing_failed!("incomplete DeepSeek DSML tool call"));
            }
        }
        let _ = self.reset();
        Ok(output)
    }
}

/// Parse a DSML event for the current parser mode.
fn parse_next_dsml_event(
    input: &mut DsmlInput<'_>,
    mode: &mut DsmlMode,
    tokens: DsmlTokens,
) -> ModalResult<DsmlEvent> {
    match mode {
        DsmlMode::Text => parse_text_event(input, tokens),
        DsmlMode::ToolBlock { invoke_end_scan } => {
            parse_tool_block_event(input, tokens, invoke_end_scan)
        }
        DsmlMode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode DSML event.
fn parse_text_event(input: &mut DsmlInput<'_>, tokens: DsmlTokens) -> ModalResult<DsmlEvent> {
    alt((
        |input: &mut DsmlInput<'_>| tool_calls_start_event(input, tokens),
        |input: &mut DsmlInput<'_>| safe_text_event(input, tokens),
    ))
    .parse_next(input)
}

/// Parse a tool-block DSML event.
fn parse_tool_block_event(
    input: &mut DsmlInput<'_>,
    tokens: DsmlTokens,
    invoke_end_scan: &mut MarkerScanState,
) -> ModalResult<DsmlEvent> {
    ws0.void().parse_next(input)?;
    alt((
        |input: &mut DsmlInput<'_>| invoke_event(input, invoke_end_scan),
        |input: &mut DsmlInput<'_>| tool_calls_end_event(input, tokens),
    ))
    .parse_next(input)
}

/// Parse a DSML function-calls start marker.
fn tool_calls_start_event(input: &mut DsmlInput<'_>, tokens: DsmlTokens) -> ModalResult<DsmlEvent> {
    literal(tokens.tool_calls_start)
        .value(DsmlEvent::ToolCallsStart)
        .parse_next(input)
}

/// Parse a DSML function-calls end marker.
fn tool_calls_end_event(input: &mut DsmlInput<'_>, tokens: DsmlTokens) -> ModalResult<DsmlEvent> {
    literal(tokens.tool_calls_end).value(DsmlEvent::ToolCallsEnd).parse_next(input)
}

/// Parse a trailing rest after DSML function calls.
fn ignored_rest_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    rest.value(DsmlEvent::IgnoredRest).parse_next(input)
}

/// Parse a safe text run before the next DSML marker.
fn safe_text_event(input: &mut DsmlInput<'_>, tokens: DsmlTokens) -> ModalResult<DsmlEvent> {
    safe_text_len(input, tokens.tool_calls_start).map(|len| DsmlEvent::Text { len })
}

/// Parse a DSML invoke block.
fn invoke_event(
    input: &mut DsmlInput<'_>,
    invoke_end_scan: &mut MarkerScanState,
) -> ModalResult<DsmlEvent> {
    let (name, body) = seq!(
        _: literal(INVOKE_START),
        _: ws1,
        dsml_name_attr,
        _: ws0,
        _: ">",
        take_until_marker(INVOKE_END, invoke_end_scan),
        _: literal(INVOKE_END),
    )
    .parse_next(input)?;
    let raw_params = parse_invoke_params(body)?;
    Ok(DsmlEvent::Invoke {
        name: name.to_string(),
        raw_params,
    })
}

/// Parse a DSML invoke body.
fn parse_invoke_params(invoke_body: &str) -> ModalResult<Vec<DsmlParameter>> {
    let mut input = invoke_body;
    delimited(ws0, repeat(0.., terminated(parse_parameter, ws0)), eof).parse_next(&mut input)
}

/// Parse a DSML parameter block.
fn parse_parameter(input: &mut &str) -> ModalResult<DsmlParameter> {
    seq! {DsmlParameter {
        _: literal(PARAMETER_START),
        _: ws1,
        name: name_attr.map(|name: &str| name.to_string()),
        _: ws1,
        is_string: string_attr.map(|value| value == "true"),
        _: ws0,
        _: ">",
        value: take_until(0.., PARAMETER_END).map(str::to_string),
        _: literal(PARAMETER_END),
    }}
    .parse_next(input)
}

/// Parse a name attribute.
fn name_attr<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    delimited("name=\"", take_until(1.., "\""), "\"").parse_next(input)
}

/// Parse a string attribute.
fn string_attr<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    delimited("string=\"", alt(("true", "false")), "\"").parse_next(input)
}

/// Parse a DSML name attribute.
fn dsml_name_attr<'i>(input: &mut DsmlInput<'i>) -> ModalResult<&'i str> {
    delimited("name=\"", take_until(1.., "\""), "\"").parse_next(input)
}
