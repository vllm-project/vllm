// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, Tokenizer};
use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, opt, preceded, seq};
use winnow::prelude::*;
use winnow::token::{rest, take_until};

use super::parameters::ToolSchemas;
use super::utils::{
    Attributed, AttributedExt, Marker, MarkerScanState, SpecialToken,
    parse_buffered_event_attributed, safe_text_len, safe_text_len_mul, take_until_marker,
};
use super::{JsonStyle, OpenToolCall, Result, ToolParserError, ToolParserOutput};
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
            parameter_start: marker(tokens.parameter_start),
            parameter_end: marker(tokens.parameter_end),
        }
    }
}

#[derive(Debug)]
enum DsmlMode {
    Text,
    /// Inside the tool-call block, between invokes.
    ToolBlock,
    /// Inside an invoke.
    Invoke {
        call: OpenToolCall,
        body: InvokeBody,
    },
    Done,
}

impl DsmlMode {
    /// The open invoke and the position inside it, for an event inside an
    /// invoke.
    fn invoke_mut(&mut self) -> Result<(&mut OpenToolCall, &mut InvokeBody)> {
        match self {
            DsmlMode::Invoke { call, body } => Ok((call, body)),
            _ => Err(parsing_failed!(
                "DeepSeek DSML invoke event outside an invoke"
            )),
        }
    }
}

/// Where the parser is inside an invoke.
#[derive(Debug)]
enum InvokeBody {
    /// Between parameters.
    Parameters,
    /// Inside a string parameter value that streams as it arrives.
    Parameter,
    /// Inside a parameter value that is converted once complete.
    BufferedParameter {
        name: String,
        conversion: ValueConversion,
        value_end_scan: MarkerScanState,
    },
}

/// How a buffered parameter value is converted once complete.
#[derive(Debug, Clone, Copy)]
enum ValueConversion {
    /// `string="false"`: JSON text, kept when valid and written as a string
    /// otherwise.
    Json,
    /// No `string` attribute: converted through the tool's schema.
    Schema,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DsmlEvent {
    Text {
        len: usize,
    },
    ToolCallsStart,
    InvokeStart {
        name: String,
    },
    /// A parameter opens. `is_string` tells whether its `string` attribute is
    /// `"true"`, or is `None` when the attribute is missing.
    ParameterStart {
        name: String,
        is_string: Option<bool>,
    },
    /// The first `len` buffered bytes continue a streamed parameter value.
    ValueText {
        len: usize,
    },
    ParameterEnd,
    /// The complete value of a buffered parameter.
    BufferedValue {
        value: String,
    },
    InvokeEnd,
    ToolCallsEnd,
    IgnoredRest,
}

/// Tool parser core for DeepSeek DSML tool calls.
///
/// DeepSeek's encoder writes a `string="true"` value as raw text, which
/// streams as it arrives, escaped into a JSON string. Any other value is JSON
/// text, whose validity is known only at its end, so it is written once
/// complete: as a string when it is not valid JSON, like the Python parser,
/// and otherwise as spelled, like DeepSeek's reference stream decoder:
/// <https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/stream/output_parser.rs#L553-L566>
/// A parameter without the attribute streams only when its schema keeps text
/// verbatim, and is otherwise converted through the schema once complete.
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
        match event {
            DsmlEvent::Text { len: consumed_len } => {
                output.push_text(&self.buffer.text[..consumed_len]);
            }
            DsmlEvent::ToolCallsStart => self.mode = DsmlMode::ToolBlock,
            DsmlEvent::InvokeStart { name } => {
                // The template renders JSON values with Python separators.
                let call = OpenToolCall::new(name, self.emitted_invoke_count, JsonStyle::Spaced);
                self.emitted_invoke_count += 1;
                self.mode = DsmlMode::Invoke {
                    call,
                    body: InvokeBody::Parameters,
                };
            }
            DsmlEvent::ParameterStart { name, is_string } => {
                let (call, body) = self.mode.invoke_mut()?;
                let conversion = match is_string {
                    Some(true) => None,
                    Some(false) => Some(ValueConversion::Json),
                    None => (!self.tool_parameters.is_verbatim_string(&call.name, &name))
                        .then_some(ValueConversion::Schema),
                };
                *body = match conversion {
                    Some(conversion) => InvokeBody::BufferedParameter {
                        name,
                        conversion,
                        value_end_scan: MarkerScanState::default(),
                    },
                    None => {
                        call.arguments.open_string(&name);
                        InvokeBody::Parameter
                    }
                };
            }
            DsmlEvent::ValueText { len } => {
                let (call, _) = self.mode.invoke_mut()?;
                call.arguments.push(&self.buffer.text[..len]);
            }
            DsmlEvent::ParameterEnd => {
                let (call, body) = self.mode.invoke_mut()?;
                call.arguments.close();
                *body = InvokeBody::Parameters;
            }
            DsmlEvent::BufferedValue { value } => {
                let (call, body) = self.mode.invoke_mut()?;
                let InvokeBody::BufferedParameter {
                    name, conversion, ..
                } = std::mem::replace(body, InvokeBody::Parameters)
                else {
                    return Err(parsing_failed!(
                        "DeepSeek DSML buffered value outside a buffered parameter"
                    ));
                };
                match conversion {
                    ValueConversion::Json => call.arguments.write_json(&name, &value),
                    ValueConversion::Schema => {
                        let value = self
                            .tool_parameters
                            .convert_param_with_schema(&call.name, &name, value);
                        call.arguments.write_value(&name, &value);
                    }
                }
            }
            DsmlEvent::InvokeEnd => {
                let DsmlMode::Invoke { call, .. } =
                    std::mem::replace(&mut self.mode, DsmlMode::ToolBlock)
                else {
                    return Err(parsing_failed!(
                        "DeepSeek DSML invoke end outside an invoke"
                    ));
                };
                output.push_call(call.finish());
            }
            DsmlEvent::ToolCallsEnd => self.mode = DsmlMode::Done,
            DsmlEvent::IgnoredRest => {}
        }
        if let DsmlMode::Invoke { call, .. } = &mut self.mode
            && let Some(delta) = call.take_delta()
        {
            output.push_call(delta);
        }
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
                parse_next_dsml_event(input, &mut self.mode)
            })?
        {
            self.apply_event(event, output)?;
            let _ = self.buffer.drain_prefix(consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            DsmlMode::Text => output.push_text(&self.buffer.text),
            DsmlMode::Done => {}
            DsmlMode::ToolBlock | DsmlMode::Invoke { .. } => {
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
fn parse_next_dsml_event(input: &mut DsmlInput<'_>, mode: &mut DsmlMode) -> ModalResult<DsmlEvent> {
    match mode {
        DsmlMode::Text => parse_text_event(input),
        DsmlMode::ToolBlock => parse_tool_block_event(input),
        DsmlMode::Invoke { body, .. } => match body {
            InvokeBody::Parameters => parse_invoke_event(input),
            InvokeBody::Parameter => parse_parameter_value_event(input),
            InvokeBody::BufferedParameter { value_end_scan, .. } => {
                buffered_value_event(input, value_end_scan)
            }
        },
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

/// Parse an invoke-body DSML event.
fn parse_invoke_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    ws0.void().parse_next(input)?;
    alt((parameter_start_event, invoke_end_event)).parse_next(input)
}

/// Parse a streamed parameter-value DSML event.
fn parse_parameter_value_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    alt((parameter_end_event, value_text_event)).parse_next(input)
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

/// Parse a DSML invoke start tag.
fn invoke_start_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    let markers = input.markers();
    let (name,) = seq!(
        _: &markers.invoke_start,
        _: ws1,
        name_attr,
        _: ws0,
        _: ">",
    )
    .parse_next(input)?;
    Ok(DsmlEvent::InvokeStart {
        name: name.to_string(),
    })
}

/// Parse a DSML invoke end marker.
fn invoke_end_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    (&input.markers().invoke_end).value(DsmlEvent::InvokeEnd).parse_next(input)
}

/// Parse a DSML parameter start tag.
///
/// The `string` attribute is optional: the model sometimes omits it, and
/// failing the whole invoke would leak the block into the text output. A
/// missing value leaves the parameter to its schema; a non-`"true"` value is
/// treated like `string="false"`.
fn parameter_start_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    let markers = input.markers();
    let (name, string) = seq!(
        _: &markers.parameter_start,
        _: ws1,
        name_attr,
        opt(preceded(ws1, string_attr)),
        _: ws0,
        _: ">",
    )
    .parse_next(input)?;
    Ok(DsmlEvent::ParameterStart {
        name: name.to_string(),
        is_string: string.map(|string| string == "true"),
    })
}

/// Parse a DSML parameter end marker.
fn parameter_end_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    (&input.markers().parameter_end)
        .value(DsmlEvent::ParameterEnd)
        .parse_next(input)
}

/// Parse a safe run of streamed parameter value before the DSML parameter end
/// marker.
fn value_text_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    let markers = input.markers();
    safe_text_len(input, &markers.parameter_end).map(|len| DsmlEvent::ValueText { len })
}

/// Parse a complete buffered parameter value through the DSML parameter end
/// marker.
fn buffered_value_event(
    input: &mut DsmlInput<'_>,
    value_end_scan: &mut MarkerScanState,
) -> ModalResult<DsmlEvent> {
    let markers = input.markers();
    let (value,) = seq!(
        take_until_marker(&markers.parameter_end, value_end_scan),
        _: &markers.parameter_end,
    )
    .parse_next(input)?;
    Ok(DsmlEvent::BufferedValue {
        value: value.to_string(),
    })
}

/// Parse a name attribute.
fn name_attr<'i>(input: &mut DsmlInput<'i>) -> ModalResult<&'i str> {
    delimited("name=\"", take_until(1.., "\""), "\"").parse_next(input)
}

/// Parse a string attribute.
fn string_attr<'i>(input: &mut DsmlInput<'i>) -> ModalResult<&'i str> {
    delimited("string=\"", take_until(0.., "\""), "\"").parse_next(input)
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
