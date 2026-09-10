// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, take_until};

use super::parameters::ToolSchemas;
use super::utils::{parse_buffered_event, partial_prefix_len};
use super::{Result, ToolCallDelta, ToolParserOutput};
use crate::tool::Tool;

mod deepseek_v32;
mod deepseek_v4;
mod deepseek_v41;

pub use deepseek_v4::DeepSeekV4ToolParser;
pub use deepseek_v32::DeepSeekV32ToolParser;
pub use deepseek_v41::DeepSeekV41ToolParser;

type DsmlInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, Copy)]
struct DsmlTokens {
    tool_calls_start: &'static str,
    framed_tool_calls_start: &'static str,
    tool_calls_end: &'static str,
    invoke_start: &'static str,
    invoke_end: &'static str,
    parameter_start: &'static str,
    parameter_end: &'static str,
}

impl DsmlTokens {
    const V32: Self = Self {
        tool_calls_start: "<｜DSML｜function_calls>",
        framed_tool_calls_start: "\n\n<｜DSML｜function_calls>",
        tool_calls_end: "</｜DSML｜function_calls>",
        invoke_start: "<｜DSML｜invoke",
        invoke_end: "</｜DSML｜invoke>",
        parameter_start: "<｜DSML｜parameter",
        parameter_end: "</｜DSML｜parameter>",
    };
    const V4: Self = Self {
        tool_calls_start: "<｜DSML｜tool_calls>",
        framed_tool_calls_start: "\n\n<｜DSML｜tool_calls>",
        tool_calls_end: "</｜DSML｜tool_calls>",
        ..Self::V32
    };
    const V41: Self = Self {
        tool_calls_start: "<｜DSML｜ calls>",
        framed_tool_calls_start: "\n\n<｜DSML｜ calls>",
        tool_calls_end: "</｜DSML｜ calls>",
        invoke_start: "<｜DSML｜ invoke",
        invoke_end: "</｜DSML｜ invoke>",
        parameter_start: "<｜DSML｜ parameter",
        parameter_end: "</｜DSML｜ parameter>",
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DsmlMode {
    Text,
    ToolBlock,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamMode {
    String,
    Raw,
    Buffered,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveParam {
    name: String,
    mode: ParamMode,
    buffered: String,
}

/// Tool parser core for DeepSeek DSML tool calls.
struct DeepSeekDsmlToolParser {
    buffer: String,
    mode: DsmlMode,
    emitted_invoke_count: usize,
    active_tool_index: Option<usize>,
    active_tool_name: Option<String>,
    args_started: bool,
    active_param: Option<ActiveParam>,
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
            active_tool_index: None,
            active_tool_name: None,
            args_started: false,
            active_param: None,
            tool_parameters: ToolSchemas::from_tools(tools),
            tokens,
        }
    }

    /// Emit one function-call update.
    fn push_call(
        &self,
        output: &mut ToolParserOutput,
        index: usize,
        name: Option<String>,
        arguments: impl Into<String>,
    ) {
        output.push_call(ToolCallDelta {
            tool_index: index,
            name,
            arguments: arguments.into(),
        });
    }

    /// Emit the JSON object key prefix for one parameter.
    fn push_param_prefix(
        &mut self,
        output: &mut ToolParserOutput,
        index: usize,
        name: &str,
        as_string: bool,
    ) -> Result<()> {
        let key = serde_json::to_string(name)
            .map_err(|error| parsing_failed!("failed to serialize parameter name: {}", error))?;
        let separator = if self.args_started { ',' } else { '{' };
        self.args_started = true;
        let quote = if as_string { "\"" } else { "" };
        self.push_call(output, index, None, format!("{separator}{key}:{quote}"));
        Ok(())
    }

    /// Emit a complete schema-converted parameter value.
    fn push_buffered_param(
        &mut self,
        output: &mut ToolParserOutput,
        index: usize,
        tool_name: &str,
        param: ActiveParam,
    ) -> Result<()> {
        let value =
            self.tool_parameters
                .convert_param_with_schema(tool_name, &param.name, param.buffered);
        self.push_param_prefix(output, index, &param.name, false)?;
        let value = serde_json::to_string(&value)
            .map_err(|error| parsing_failed!("failed to serialize argument value: {}", error))?;
        self.push_call(output, index, None, value);
        Ok(())
    }

    /// JSON-escape a string fragment without adding surrounding quotes.
    fn escape_string_fragment(value: &str) -> Result<String> {
        let encoded = serde_json::to_string(value)
            .map_err(|error| parsing_failed!("failed to serialize string argument: {}", error))?;
        Ok(encoded[1..encoded.len() - 1].to_string())
    }

    /// Remove leading DSML whitespace from the buffered tool block.
    fn trim_tool_whitespace(&mut self) -> bool {
        let trimmed = self.buffer.trim_start();
        let len = self.buffer.len() - trimmed.len();
        if len == 0 {
            return false;
        }
        self.buffer.drain(..len);
        true
    }

    /// Start one invoke and emit its metadata immediately.
    fn begin_invoke(&mut self, name: String, output: &mut ToolParserOutput) {
        let index = self.emitted_invoke_count;
        self.emitted_invoke_count += 1;
        self.active_tool_index = Some(index);
        self.active_tool_name = Some(name.clone());
        self.args_started = false;
        self.active_param = None;
        self.push_call(output, index, Some(name), "");
    }

    /// Finish one invoke by closing its streamed JSON object.
    fn finish_invoke(&mut self, output: &mut ToolParserOutput) {
        let Some(index) = self.active_tool_index.take() else {
            return;
        };
        self.push_call(
            output,
            index,
            None,
            if self.args_started { "}" } else { "{}" },
        );
        self.active_tool_name = None;
        self.args_started = false;
        self.active_param = None;
    }

    /// Consume plain text until a complete or partial tool-call marker.
    fn process_text(&mut self, output: &mut ToolParserOutput) -> bool {
        let markers = [
            self.tokens.framed_tool_calls_start,
            self.tokens.tool_calls_start,
        ];
        if let Some((start, marker)) = markers
            .iter()
            .filter_map(|marker| self.buffer.find(*marker).map(|start| (start, *marker)))
            .min_by_key(|(start, _)| *start)
        {
            if start > 0 {
                output.push_text(self.buffer[..start].to_string());
                self.buffer.drain(..start);
                return true;
            }
            self.buffer.drain(..marker.len());
            self.mode = DsmlMode::ToolBlock;
            return true;
        }

        let keep = markers
            .iter()
            .map(|marker| partial_prefix_len(&self.buffer, marker))
            .max()
            .unwrap_or(0);
        let emit = self.buffer.len().saturating_sub(keep);
        if emit == 0 {
            return false;
        }
        output.push_text(self.buffer[..emit].to_string());
        self.buffer.drain(..emit);
        true
    }

    /// Consume the next invoke header or the tool-calls closing marker.
    fn process_between_invokes(&mut self, output: &mut ToolParserOutput) -> Result<bool> {
        if self.trim_tool_whitespace() {
            return Ok(true);
        }
        if self.buffer.is_empty() {
            return Ok(false);
        }

        if self.buffer.starts_with(self.tokens.tool_calls_end) {
            self.buffer.drain(..self.tokens.tool_calls_end.len());
            self.mode = DsmlMode::Done;
            return Ok(true);
        }
        if self.tokens.tool_calls_end.starts_with(&self.buffer) {
            return Ok(false);
        }

        let parsed =
            parse_buffered_event(&self.buffer, |input| parse_invoke_start(input, self.tokens))?;
        let Some((name, consumed)) = parsed else {
            return Ok(false);
        };
        self.buffer.drain(..consumed);
        self.begin_invoke(name, output);
        Ok(true)
    }

    /// Consume one active parameter's content incrementally.
    fn process_active_param(&mut self, output: &mut ToolParserOutput) -> Result<bool> {
        let Some(index) = self.active_tool_index else {
            return Ok(false);
        };
        let Some(mode) = self.active_param.as_ref().map(|param| param.mode) else {
            return Ok(false);
        };

        if let Some(end) = self.buffer.find(self.tokens.parameter_end) {
            let raw = self.buffer[..end].to_string();
            self.buffer.drain(..end + self.tokens.parameter_end.len());
            match mode {
                ParamMode::String => {
                    let arguments = Self::escape_string_fragment(&raw)? + "\"";
                    self.push_call(output, index, None, arguments);
                    self.active_param = None;
                }
                ParamMode::Raw => {
                    self.push_call(output, index, None, raw);
                    self.active_param = None;
                }
                ParamMode::Buffered => {
                    let mut param = self.active_param.take().expect("active parameter exists");
                    param.buffered.push_str(&raw);
                    let tool_name = self.active_tool_name.clone().unwrap_or_default();
                    self.push_buffered_param(output, index, &tool_name, param)?;
                }
            }
            return Ok(true);
        }

        let keep = partial_prefix_len(&self.buffer, self.tokens.parameter_end);
        let emit = self.buffer.len().saturating_sub(keep);
        if emit == 0 {
            return Ok(false);
        }
        let raw = self.buffer[..emit].to_string();
        self.buffer.drain(..emit);
        match mode {
            ParamMode::String => {
                self.push_call(output, index, None, Self::escape_string_fragment(&raw)?);
            }
            ParamMode::Raw => self.push_call(output, index, None, raw),
            ParamMode::Buffered => {
                if let Some(param) = self.active_param.as_mut() {
                    param.buffered.push_str(&raw);
                }
            }
        }
        Ok(true)
    }

    /// Consume an invoke closing marker or begin its next parameter.
    fn process_invoke(&mut self, output: &mut ToolParserOutput) -> Result<bool> {
        if self.active_param.is_some() {
            return self.process_active_param(output);
        }
        if self.trim_tool_whitespace() {
            return Ok(true);
        }
        if self.buffer.is_empty() {
            return Ok(false);
        }

        if self.buffer.starts_with(self.tokens.invoke_end) {
            self.buffer.drain(..self.tokens.invoke_end.len());
            self.finish_invoke(output);
            return Ok(true);
        }
        if self.tokens.invoke_end.starts_with(&self.buffer) {
            return Ok(false);
        }

        let parsed = parse_buffered_event(&self.buffer, |input| {
            parse_parameter_start(input, self.tokens)
        })?;
        let Some(((name, is_string), consumed)) = parsed else {
            return Ok(false);
        };
        self.buffer.drain(..consumed);

        let index = self.active_tool_index.expect("active tool exists");
        let tool_name = self.active_tool_name.clone().unwrap_or_default();
        let mode = if is_string {
            self.push_param_prefix(output, index, &name, true)?;
            ParamMode::String
        } else if self.tool_parameters.can_stream_raw_param(&tool_name, &name) {
            self.push_param_prefix(output, index, &name, false)?;
            ParamMode::Raw
        } else {
            ParamMode::Buffered
        };
        self.active_param = Some(ActiveParam {
            name,
            mode,
            buffered: String::new(),
        });
        Ok(true)
    }

    fn reset(&mut self) -> String {
        self.mode = DsmlMode::Text;
        self.emitted_invoke_count = 0;
        self.active_tool_index = None;
        self.active_tool_name = None;
        self.args_started = false;
        self.active_param = None;
        std::mem::take(&mut self.buffer)
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        // Extract tool calls from streaming model output. DSML framing is
        // buffered, while stable function metadata and argument fragments are
        // emitted as soon as they can no longer change.
        self.buffer.push_str(chunk);

        loop {
            let progressed = match self.mode {
                DsmlMode::Text => self.process_text(output),
                DsmlMode::ToolBlock if self.active_tool_index.is_none() => {
                    self.process_between_invokes(output)?
                }
                DsmlMode::ToolBlock => self.process_invoke(output)?,
                DsmlMode::Done => false,
            };
            if !progressed {
                break;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            DsmlMode::Text => output.push_text(&self.buffer),
            DsmlMode::Done => {}
            DsmlMode::ToolBlock => {
                return Err(parsing_failed!("incomplete DeepSeek DSML tool call"));
            }
        }
        let _ = self.reset();
        Ok(output)
    }
}

/// Parse a DSML invoke start tag.
fn parse_invoke_start(input: &mut DsmlInput<'_>, tokens: DsmlTokens) -> ModalResult<String> {
    literal(tokens.invoke_start).parse_next(input)?;
    ws1.void().parse_next(input)?;
    let name = dsml_name_attr(input)?.to_string();
    ws0.void().parse_next(input)?;
    literal(">").parse_next(input)?;
    Ok(name)
}

/// Parse a DSML parameter start tag.
fn parse_parameter_start(
    input: &mut DsmlInput<'_>,
    tokens: DsmlTokens,
) -> ModalResult<(String, bool)> {
    literal(tokens.parameter_start).parse_next(input)?;
    ws1.void().parse_next(input)?;
    let name = dsml_name_attr(input)?.to_string();
    ws1.void().parse_next(input)?;
    let is_string = dsml_string_attr(input)? == "true";
    ws0.void().parse_next(input)?;
    literal(">").parse_next(input)?;
    Ok((name, is_string))
}

/// Parse a DSML name attribute.
fn dsml_name_attr<'i>(input: &mut DsmlInput<'i>) -> ModalResult<&'i str> {
    delimited("name=\"", take_until(1.., "\""), "\"").parse_next(input)
}

/// Parse a DSML string attribute.
fn dsml_string_attr<'i>(input: &mut DsmlInput<'i>) -> ModalResult<&'i str> {
    delimited("string=\"", alt(("true", "false")), "\"").parse_next(input)
}
