// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::Serialize;
use vllm_tokenizer::{DecodedText, Tokenizer};
use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, opt, preceded, seq};
use winnow::prelude::*;
use winnow::token::{rest, take_until};

use super::parameters::ToolSchemas;
use super::utils::{
    Attributed, AttributedExt, Marker, SpecialToken, parse_buffered_event_attributed,
    safe_text_len_mul,
};
use super::{Result, ToolCallDelta, ToolParserError, ToolParserEvent, ToolParserOutput};
use crate::tool::Tool;

mod deepseek_v32;
mod deepseek_v4;
mod deepseek_v41;

pub use deepseek_v4::DeepSeekV4ToolParser;
pub use deepseek_v32::DeepSeekV32ToolParser;
pub use deepseek_v41::DeepSeekV41ToolParser;

type DsmlInput<'i> = Attributed<'i, DsmlMarkers>;

/// The added token inside every DSML marker.
const DSML: &str = "｜DSML｜";

/// Spellings of the DSML markers of one DeepSeek format.
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

/// DSML markers resolved against the tokenizer.
///
/// In every marker only the inner `｜DSML｜` is an added token, so each marker
/// is guarded in the middle: a marker counts only where the dedicated token
/// produced its `｜DSML｜`, and an ordinary-token spelling stays text. A
/// tokenizer without the token (DeepSeek V3.2-Exp spells DSML with ordinary
/// tokens) leaves the markers matched by spelling.
#[derive(Debug)]
struct DsmlMarkers {
    tool_calls_start: Marker,
    framed_tool_calls_start: Marker,
    tool_calls_end: Marker,
    invoke_start: Marker,
    invoke_end: Marker,
    /// Parameter markers, matched by spelling.
    // TODO: build the parameter markers with `marker` too, so they match by token
    // identity, together with the Kimi K3 call body.
    parameter_start: Marker,
    parameter_end: Marker,
}

impl DsmlMarkers {
    /// Resolve `tokens` against the `｜DSML｜` token of `tokenizer`.
    fn new(tokens: DsmlTokens, tokenizer: &dyn Tokenizer) -> Self {
        let dsml = tokenizer.token_to_id(DSML).map(|id| SpecialToken {
            text: DSML.to_string(),
            id,
        });
        let marker = |spelling: &str| {
            let Some(dsml) = &dsml else {
                return Marker::text(spelling);
            };
            let (before, after) =
                spelling.split_once(DSML).expect("every DSML marker contains `｜DSML｜`");
            Marker::text(before).then_special(dsml).then_text(after)
        };
        Self {
            tool_calls_start: marker(tokens.tool_calls_start),
            framed_tool_calls_start: marker(tokens.framed_tool_calls_start),
            tool_calls_end: marker(tokens.tool_calls_end),
            invoke_start: marker(tokens.invoke_start),
            invoke_end: marker(tokens.invoke_end),
            parameter_start: Marker::text(tokens.parameter_start),
            parameter_end: Marker::text(tokens.parameter_end),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DsmlMode {
    Text,
    ToolBlock,
    /// Inside an invoke whose name was emitted.
    Invoke(DsmlInvoke),
    Done,
}

/// An invoke whose arguments stream as its parameters arrive.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DsmlInvoke {
    tool_index: usize,
    name: String,
    /// Whether a parameter opened the arguments object.
    has_params: bool,
    param: Option<DsmlParameter>,
}

impl DsmlInvoke {
    /// The text that closes the arguments object.
    fn closing(&self) -> &'static str {
        if self.has_params { "}" } else { "{}" }
    }
}

/// A parameter whose value is still arriving.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DsmlParameter {
    name: String,
    /// Raw value held for schema conversion; `None` for a string value, which streams.
    raw_value: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DsmlEvent {
    Text { len: usize },
    ToolCallsStart,
    InvokeStart { name: String },
    ParameterStart { name: String, is_string: bool },
    ParameterValue { len: usize },
    ParameterEnd,
    InvokeEnd,
    ToolCallsEnd,
    IgnoredRest,
}

/// Tool parser core for DeepSeek DSML tool calls.
///
/// Requires attributed input when the markers are guarded: fed unattributed
/// text, every DSML marker is plain text and no tool call is parsed.
struct DeepSeekDsmlToolParser {
    buffer: DecodedText,
    mode: DsmlMode,
    emitted_invoke_count: usize,
    tool_parameters: ToolSchemas,
    markers: DsmlMarkers,
}

impl DeepSeekDsmlToolParser {
    /// Create a parser with DSML tokens for one DeepSeek format.
    fn new(tools: &[Tool], tokenizer: &dyn Tokenizer, tokens: DsmlTokens) -> Self {
        Self {
            buffer: DecodedText::default(),
            mode: DsmlMode::Text,
            emitted_invoke_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
            markers: DsmlMarkers::new(tokens, tokenizer),
        }
    }

    /// Apply one parsed DSML event to parser state and output.
    fn apply_event(&mut self, event: DsmlEvent, output: &mut ToolParserOutput) -> Result<()> {
        match (&mut self.mode, event) {
            (DsmlMode::Text, DsmlEvent::Text { len: consumed_len }) => {
                output.push_text(&self.buffer.text[..consumed_len]);
            }
            (mode @ DsmlMode::Text, DsmlEvent::ToolCallsStart) => *mode = DsmlMode::ToolBlock,
            (mode @ DsmlMode::ToolBlock, DsmlEvent::InvokeStart { name }) => {
                let tool_index = self.emitted_invoke_count;
                self.emitted_invoke_count += 1;
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(name.clone()),
                    arguments: String::new(),
                });
                *mode = DsmlMode::Invoke(DsmlInvoke {
                    tool_index,
                    name,
                    has_params: false,
                    param: None,
                });
            }
            (DsmlMode::Invoke(invoke), DsmlEvent::ParameterStart { name, is_string }) => {
                let opener = if invoke.has_params { "," } else { "{" };
                let quote = if is_string { "\"" } else { "" };
                let arguments = format!("{opener}{}:{quote}", to_json(&name)?);
                invoke.has_params = true;
                invoke.param = Some(DsmlParameter {
                    name,
                    raw_value: (!is_string).then(String::new),
                });
                push_arguments(output, invoke.tool_index, &arguments);
            }
            (DsmlMode::Invoke(invoke), DsmlEvent::ParameterValue { len }) => {
                let value = &self.buffer.text[..len];
                let Some(param) = &mut invoke.param else {
                    unreachable!("a DSML parameter value outside a parameter");
                };
                match &mut param.raw_value {
                    Some(raw_value) => raw_value.push_str(value),
                    None => {
                        let quoted = to_json(value)?;
                        push_arguments(output, invoke.tool_index, &quoted[1..quoted.len() - 1]);
                    }
                }
            }
            (DsmlMode::Invoke(invoke), DsmlEvent::ParameterEnd) => {
                let Some(param) = invoke.param.take() else {
                    unreachable!("a DSML parameter end outside a parameter");
                };
                match param.raw_value {
                    None => push_arguments(output, invoke.tool_index, "\""),
                    Some(raw_value) => {
                        let value = self.tool_parameters.convert_param_with_schema(
                            &invoke.name,
                            &param.name,
                            raw_value,
                        );
                        push_arguments(output, invoke.tool_index, &to_json(&value)?);
                    }
                }
            }
            (DsmlMode::Invoke(invoke), DsmlEvent::InvokeEnd) => {
                push_arguments(output, invoke.tool_index, invoke.closing());
                self.mode = DsmlMode::ToolBlock;
            }
            // The model left out the invoke end marker; Python closes the call here too.
            (DsmlMode::Invoke(invoke), DsmlEvent::ToolCallsEnd) => {
                push_arguments(output, invoke.tool_index, invoke.closing());
                self.mode = DsmlMode::Done;
            }
            (mode @ DsmlMode::ToolBlock, DsmlEvent::ToolCallsEnd) => *mode = DsmlMode::Done,
            (DsmlMode::Done, DsmlEvent::IgnoredRest) => {}
            (mode, event) => unreachable!("DSML event {event:?} in mode {mode:?}"),
        };
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = DsmlMode::Text;
        self.emitted_invoke_count = 0;
        std::mem::take(&mut self.buffer).text
    }

    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
        // Extract tool calls from streaming model output.
        self.buffer.append(chunk);

        while let Some((event, consumed_len)) =
            parse_buffered_event_attributed(&self.buffer, &self.markers, |input| {
                parse_next_dsml_event(input, &self.mode)
            })
            .inspect_err(|_| self.close_arguments_between_parameters(output))?
        {
            self.apply_event(event, output)?;
            let _ = self.buffer.drain_prefix(consumed_len);
        }

        Ok(())
    }

    /// On a parse failure between parameters, close the arguments; a cut-off value stays open.
    fn close_arguments_between_parameters(&self, output: &mut ToolParserOutput) {
        if let DsmlMode::Invoke(invoke @ DsmlInvoke { param: None, .. }) = &self.mode {
            push_arguments(output, invoke.tool_index, invoke.closing());
        }
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            DsmlMode::Text => output.push_text(&self.buffer.text),
            DsmlMode::Done => {}
            DsmlMode::ToolBlock | DsmlMode::Invoke(_) => {
                return Err(ToolParserError::IncompleteToolCall {
                    parser: "DeepSeek DSML",
                });
            }
        }
        let _ = self.reset();
        Ok(output)
    }
}

/// Parse a DSML event for the current parser mode.
fn parse_next_dsml_event(input: &mut DsmlInput<'_>, mode: &DsmlMode) -> ModalResult<DsmlEvent> {
    match mode {
        DsmlMode::Text => parse_text_event(input),
        DsmlMode::ToolBlock => parse_tool_block_event(input),
        DsmlMode::Invoke(DsmlInvoke { param: None, .. }) => parse_invoke_body_event(input),
        DsmlMode::Invoke(DsmlInvoke { param: Some(_), .. }) => parse_parameter_value_event(input),
        DsmlMode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode DSML event.
fn parse_text_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    alt((tool_calls_start_event, safe_text_event)).parse_next(input)
}

/// Parse a tool-block DSML event.
fn parse_tool_block_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    ws0.void().parse_next(input)?;
    alt((invoke_start_event, tool_calls_end_event)).parse_next(input)
}

/// Parse the next DSML parameter header, or an invoke or function-calls end marker.
fn parse_invoke_body_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    ws0.void().parse_next(input)?;
    alt((
        parameter_start_event,
        invoke_end_event,
        tool_calls_end_event,
    ))
    .parse_next(input)
}

/// Parse a DSML parameter end marker, or a value run before it or an invoke end.
fn parse_parameter_value_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    let markers = input.markers();
    alt((
        (&markers.parameter_end).value(DsmlEvent::ParameterEnd),
        |input: &mut DsmlInput<'_>| {
            safe_text_len_mul(input, &[&markers.parameter_end, &markers.invoke_end])
                .map(|len| DsmlEvent::ParameterValue { len })
        },
    ))
    .parse_next(input)
}

/// Parse a DSML function-calls start marker.
fn tool_calls_start_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    let markers = input.markers();
    alt((&markers.framed_tool_calls_start, &markers.tool_calls_start))
        .value(DsmlEvent::ToolCallsStart)
        .parse_next(input)
}

/// Parse a DSML function-calls end marker.
fn tool_calls_end_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    (&input.markers().tool_calls_end)
        .value(DsmlEvent::ToolCallsEnd)
        .parse_next(input)
}

/// Parse a trailing rest after DSML function calls.
fn ignored_rest_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    rest.value(DsmlEvent::IgnoredRest).parse_next(input)
}

/// Parse a safe text run before the next DSML marker.
fn safe_text_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    let markers = input.markers();
    safe_text_len_mul(
        input,
        &[&markers.framed_tool_calls_start, &markers.tool_calls_start],
    )
    .map(|len| DsmlEvent::Text { len })
}

/// Parse a DSML invoke header.
fn invoke_start_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    let markers = input.markers();
    seq! {DsmlEvent::InvokeStart {
        _: &markers.invoke_start,
        _: ws1,
        name: name_attr.map(str::to_string),
        _: ws0,
        _: ">",
    }}
    .parse_next(input)
}

/// Parse a DSML invoke end marker.
fn invoke_end_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    (&input.markers().invoke_end).value(DsmlEvent::InvokeEnd).parse_next(input)
}

/// Parse a DSML parameter header.
///
/// The `string` attribute is optional: the model sometimes omits it, and
/// failing the whole invoke would leak the block into the text output. A
/// missing or non-`"true"` value is treated like `string="false"`.
fn parameter_start_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    let markers = input.markers();
    seq! {DsmlEvent::ParameterStart {
        _: &markers.parameter_start,
        _: ws1,
        name: name_attr.map(str::to_string),
        is_string: opt(preceded(ws1, string_attr)).map(|value| value == Some("true")),
        _: ws0,
        _: ">",
    }}
    .parse_next(input)
}

/// Parse a name attribute.
fn name_attr<'i>(input: &mut DsmlInput<'i>) -> ModalResult<&'i str> {
    delimited("name=\"", take_until(1.., "\""), "\"").parse_next(input)
}

/// Parse a string attribute.
fn string_attr<'i>(input: &mut DsmlInput<'i>) -> ModalResult<&'i str> {
    delimited("string=\"", take_until(0.., "\""), "\"").parse_next(input)
}

/// Append `arguments` to a call, merging into the chunk's last delta for the same call.
fn push_arguments(output: &mut ToolParserOutput, tool_index: usize, arguments: &str) {
    if let Some(ToolParserEvent::ToolCall(call)) = output.events.last_mut()
        && call.tool_index == tool_index
    {
        call.arguments.push_str(arguments);
    } else {
        output.push_call(ToolCallDelta {
            tool_index,
            name: None,
            arguments: arguments.to_string(),
        });
    }
}

/// Serialize `value` as compact JSON.
fn to_json<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    serde_json::to_string(value)
        .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))
}

#[cfg(test)]
mod test_utils {
    use vllm_tokenizer::Tokenizer as _;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::DSML;
    use crate::tool::test_utils::{collect_attributed_stream, decode_ids};
    use crate::tool::{Result, ToolParser, ToolParserOutput};

    /// The DeepSeek tokenizers have `｜DSML｜` as an added token and a special
    /// end-of-sentence token.
    pub(super) fn tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token(DSML, 256)
            .with_special_token("<｜end▁of▁sentence｜>", 257)
    }

    /// Encode with the added tokens, so DSML markers are structural.
    pub(super) fn dedicated(text: &str) -> Vec<u32> {
        tokenizer().encode(text, false).unwrap()
    }

    /// Encode as ordinary bytes only, so DSML marker spellings are text.
    pub(super) fn ordinary(text: &str) -> Vec<u32> {
        tokenizer().encode_ordinary(text).unwrap()
    }

    /// Run token IDs through the real incremental decoder.
    pub(super) fn decode(ids: &[u32]) -> vllm_tokenizer::DecodedText {
        decode_ids(&tokenizer(), ids)
    }

    /// Stream the structural encoding of the concatenated `chunks`, split at
    /// the chunk boundaries, and coalesce the tool-call deltas.
    pub(super) fn collect_stream<T: ToolParser + ?Sized>(
        parser: &mut T,
        chunks: &[&str],
    ) -> ToolParserOutput {
        collect_attributed_stream(parser, &tokenizer(), chunks)
    }

    /// `ToolParserTestExt` over structurally encoded input.
    #[easy_ext::ext(DsmlParserTestExt)]
    pub(super) impl<T: ToolParser + ?Sized> T {
        /// Feed one structurally encoded text delta.
        fn parse_chunk(&mut self, chunk: &str) -> Result<ToolParserOutput> {
            let mut output = ToolParserOutput::default();
            self.parse_into(decode(&dedicated(chunk)), &mut output)?;
            Ok(output)
        }

        /// Parse complete structurally encoded output.
        fn parse_complete(&mut self, text: &str) -> Result<ToolParserOutput> {
            let mut output = self.parse_chunk(text)?;
            output.append(self.finish()?);
            Ok(output.coalesce())
        }
    }
}
