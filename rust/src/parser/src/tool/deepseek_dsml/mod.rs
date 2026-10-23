// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, Tokenizer};
use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, eof, opt, preceded, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::token::{literal, rest, take_until};

use super::parameters::ToolSchemas;
use super::utils::{
    Attributed, AttributedExt, Marker, MarkerScanState, SpecialToken,
    parse_buffered_event_attributed, safe_text_len_mul, take_until_marker,
};
use super::{Result, ToolCallDelta, ToolParserOutput};
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
    /// Parameter markers, matched by spelling inside a complete invoke body.
    parameter_start: &'static str,
    parameter_end: &'static str,
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
            parameter_start: tokens.parameter_start,
            parameter_end: tokens.parameter_end,
        }
    }
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
        std::mem::take(&mut self.buffer).text
    }

    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
        // Extract tool calls from streaming model output.
        //
        // Uses a buffer-until-complete-invoke strategy: text is buffered until
        // a complete invoke block is available, then parsed and emitted in one
        // shot.
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
            // A held-back tail that already spells part of a tool-call opener,
            // such as `\n\n<｜DSML｜ ca` cut by max_tokens, is an unfinished
            // tool call rather than text.
            DsmlMode::Text if is_unfinished_opener(&self.buffer.text, &self.markers) => {
                return Err(incomplete_tool_call!("incomplete DeepSeek DSML tool call"));
            }
            DsmlMode::Text => output.push_text(&self.buffer.text),
            DsmlMode::Done => {}
            DsmlMode::ToolBlock { .. } => {
                return Err(incomplete_tool_call!("incomplete DeepSeek DSML tool call"));
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
        DsmlMode::ToolBlock { invoke_end_scan } => parse_tool_block_event(input, invoke_end_scan),
        DsmlMode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode DSML event.
fn parse_text_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    alt((tool_calls_start_event, safe_text_event)).parse_next(input)
}

/// Parse a tool-block DSML event.
fn parse_tool_block_event(
    input: &mut DsmlInput<'_>,
    invoke_end_scan: &mut MarkerScanState,
) -> ModalResult<DsmlEvent> {
    ws0.void().parse_next(input)?;
    alt((
        |input: &mut DsmlInput<'_>| invoke_event(input, invoke_end_scan),
        tool_calls_end_event,
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

/// Whether `text` is a proper prefix of a tool-call opener that already
/// contains `<｜`.
///
/// Text mode holds back such a tail until it can tell whether an opener
/// follows. Openers match only on a genuine `｜DSML｜` token, so a spelling
/// made of ordinary tokens has already been emitted as text by then. A
/// held-back tail without `<｜`, such as a trailing newline, is text.
fn is_unfinished_opener(text: &str, markers: &DsmlMarkers) -> bool {
    text.contains("<｜")
        && [&markers.framed_tool_calls_start, &markers.tool_calls_start]
            .iter()
            .any(|opener| opener.as_str().len() > text.len() && opener.as_str().starts_with(text))
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

/// Parse a DSML invoke block.
fn invoke_event(
    input: &mut DsmlInput<'_>,
    invoke_end_scan: &mut MarkerScanState,
) -> ModalResult<DsmlEvent> {
    let markers = input.markers();
    let (name, body) = seq!(
        _: &markers.invoke_start,
        _: ws1,
        dsml_name_attr,
        _: ws0,
        _: ">",
        take_until_marker(&markers.invoke_end, invoke_end_scan),
        _: &markers.invoke_end,
    )
    .parse_next(input)?;
    // TODO: parse the body from the attributed input, so parameter markers are
    // matched by token identity too, together with the Kimi K3 call body.
    let raw_params = parse_invoke_params(body, markers)?;
    Ok(DsmlEvent::Invoke {
        name: name.to_string(),
        raw_params,
    })
}

/// Parse a DSML invoke body.
fn parse_invoke_params(
    invoke_body: &str,
    markers: &DsmlMarkers,
) -> ModalResult<Vec<DsmlParameter>> {
    let mut input = invoke_body;
    delimited(
        ws0,
        repeat(
            0..,
            terminated(|input: &mut &str| parse_parameter(input, markers), ws0),
        ),
        eof,
    )
    .parse_next(&mut input)
}

/// Parse a DSML parameter block.
///
/// The `string` attribute is optional: the model sometimes omits it, and
/// failing the whole invoke would leak the block into the text output. A
/// missing or non-`"true"` value is treated like `string="false"`.
fn parse_parameter(input: &mut &str, markers: &DsmlMarkers) -> ModalResult<DsmlParameter> {
    seq! {DsmlParameter {
        _: literal(markers.parameter_start),
        _: ws1,
        name: name_attr.map(|name: &str| name.to_string()),
        is_string: opt(preceded(ws1, string_attr)).map(|value| value == Some("true")),
        _: ws0,
        _: ">",
        value: take_until(0.., markers.parameter_end).map(str::to_string),
        _: literal(markers.parameter_end),
    }}
    .parse_next(input)
}

/// Parse a name attribute.
fn name_attr<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    delimited("name=\"", take_until(1.., "\""), "\"").parse_next(input)
}

/// Parse a string attribute.
fn string_attr<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    delimited("string=\"", take_until(0.., "\""), "\"").parse_next(input)
}

/// Parse a DSML name attribute.
fn dsml_name_attr<'i>(input: &mut DsmlInput<'i>) -> ModalResult<&'i str> {
    delimited("name=\"", take_until(1.., "\""), "\"").parse_next(input)
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
