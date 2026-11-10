// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer, Tokenizer};
use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, opt, peek, seq};
use winnow::error::{ContextError, ErrMode};
use winnow::prelude::*;
use winnow::stream::Stream;
use winnow::token::{literal, rest, take_until};

use super::parameters::{ParamElement, ParamInput, ToolSchemas};
use super::utils::{
    Attributed, AttributedExt, Marker, MarkerScanState, MarkerStream, Scan, SpecialToken,
    incomplete, parse_buffered_event_attributed, safe_text_len, scan_markers, take_until_marker,
};
use super::{JsonStyle, OpenToolCall, Result, ToolParser, ToolParserError, ToolParserOutput};
use crate::tool::Tool;
use crate::utils::recursion::ParserRecursionGuard;

const NAMESPACE: &str = "]<]minimax[>[";
const TOOL_CALL_START: &str = "]<]minimax[>[<tool_call>";
const TOOL_CALL_END: &str = "]<]minimax[>[</tool_call>";
const INVOKE_START: &str = "]<]minimax[>[<invoke";
const INVOKE_END: &str = "]<]minimax[>[</invoke>";
const ELEMENT_START: &str = "]<]minimax[>[<";
const ELEMENT_END_START: &str = "]<]minimax[>[</";
const MIXED_TEXT_FIELD: &str = "$text";

type MinimaxM3Input<'i> = Attributed<'i, MinimaxM3Markers>;

/// MiniMax M3 markers resolved against the tokenizer.
///
/// Every structural tag begins with the namespace marker, an added token, so
/// each marker is guarded on it: a tag counts only where the dedicated token
/// produced its namespace marker, and an ordinary-token spelling stays text. A
/// tokenizer without the token leaves the markers matched by spelling.
#[derive(Debug)]
struct MinimaxM3Markers {
    /// The namespace token, if the tokenizer has it.
    namespace_token: Option<SpecialToken>,
    namespace: Marker,
    tool_call_start: Marker,
    tool_call_end: Marker,
    invoke_start: Marker,
    invoke_end: Marker,
    element_start: Marker,
    element_end_start: Marker,
}

impl MinimaxM3Markers {
    /// Resolve the markers against the namespace token of `tokenizer`.
    fn new(tokenizer: &dyn Tokenizer) -> Self {
        let namespace_token = tokenizer.token_to_id(NAMESPACE).map(|id| SpecialToken {
            text: NAMESPACE.to_string(),
            id,
        });
        let marker = |spelling: &str| tag_marker(namespace_token.as_ref(), spelling);
        Self {
            namespace: marker(NAMESPACE),
            tool_call_start: marker(TOOL_CALL_START),
            tool_call_end: marker(TOOL_CALL_END),
            invoke_start: marker(INVOKE_START),
            invoke_end: marker(INVOKE_END),
            element_start: marker(ELEMENT_START),
            element_end_start: marker(ELEMENT_END_START),
            namespace_token,
        }
    }

    /// The close tag of element `name`.
    fn close_tag(&self, name: &str) -> Marker {
        tag_marker(
            self.namespace_token.as_ref(),
            &format!("{ELEMENT_END_START}{name}>"),
        )
    }
}

/// The marker `spelling`, which begins with the namespace marker, guarded on
/// `namespace` when the tokenizer has it.
fn tag_marker(namespace: Option<&SpecialToken>, spelling: &str) -> Marker {
    let rest = spelling
        .strip_prefix(NAMESPACE)
        .expect("every MiniMax M3 tag begins with the namespace marker");
    match namespace {
        Some(namespace) => Marker::special(namespace).then_text(rest),
        None => Marker::text(spelling),
    }
}

#[derive(Debug)]
enum MinimaxM3Mode {
    Text,
    /// Inside the tool block, between invokes.
    ToolBlock,
    /// Inside an invoke.
    Invoke {
        call: OpenToolCall,
        body: InvokeBody,
    },
    Done,
}

impl MinimaxM3Mode {
    /// The open invoke and the position inside it, for an event inside an
    /// invoke.
    fn invoke_mut(&mut self) -> Result<(&mut OpenToolCall, &mut InvokeBody)> {
        match self {
            MinimaxM3Mode::Invoke { call, body } => Ok((call, body)),
            _ => Err(parsing_failed!("MiniMax M3 invoke event outside an invoke")),
        }
    }
}

/// Where the parser is inside an invoke.
#[derive(Debug)]
enum InvokeBody {
    /// Between parameters.
    Parameters {
        /// Scan for the invoke end once ordinary text ends the parameters.
        invoke_end_scan: MarkerScanState,
    },
    /// Inside a top-level parameter whose text streams as it arrives.
    Parameter { close_tag: Marker },
    /// Inside a parameter that is converted through the tool's schema once its
    /// element closes.
    BufferedParameter {
        name: String,
        body_scan: ElementBodyScanState,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MinimaxM3Event {
    Text {
        len: usize,
    },
    ToolBlockStart,
    InvokeStart {
        name: String,
    },
    ParameterStart {
        name: String,
    },
    /// The first `len` buffered bytes continue a streamed parameter value.
    ValueText {
        len: usize,
    },
    ParameterEnd,
    /// The complete value of a buffered parameter.
    BufferedValue {
        value: ParamInput,
    },
    InvokeEnd,
    ToolBlockEnd,
    IgnoredRest,
}

/// Streaming scan state for [`take_element_body`], so a resumed scan neither
/// rescans the buffered body nor recounts its child tags.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ElementBodyScanState {
    /// Offset relative to the body start from which the scan resumes.
    scan_start: usize,
    /// Child elements opened before `scan_start` and not yet closed.
    depth: usize,
}

/// Tool parser for MiniMax M3 namespace-delimited XML-style tool calls.
///
/// Example tool call content with recursive parameters:
///
/// ```text
/// ]<]minimax[>[<tool_call>
/// ]<]minimax[>[<invoke name="create_order">
/// ]<]minimax[>[<user_id>42]<]minimax[>[</user_id>
/// ]<]minimax[>[<shipping>
/// ]<]minimax[>[<city>Singapore]<]minimax[>[</city>
/// ]<]minimax[>[<zip>018956]<]minimax[>[</zip>
/// ]<]minimax[>[</shipping>
/// ]<]minimax[>[<items>
/// ]<]minimax[>[<item>
/// ]<]minimax[>[<sku>book-001]<]minimax[>[</sku>
/// ]<]minimax[>[<qty>2]<]minimax[>[</qty>
/// ]<]minimax[>[</item>
/// ]<]minimax[>[</items>
/// ]<]minimax[>[</invoke>
/// ]<]minimax[>[</tool_call>
/// ```
///
/// With a schema where `shipping` is an object and `items` is an array of
/// objects, recursive parameter conversion produces:
///
/// ```json
/// {
///   "user_id": 42,
///   "shipping": {
///     "city": "Singapore",
///     "zip": 18956
///   },
///   "items": [
///     {
///       "sku": "book-001",
///       "qty": 2
///     }
///   ]
/// }
/// ```
///
/// MiniMax M3 emits the namespace marker `]<]minimax[>[` before each structural
/// tag. A top-level parameter whose schema is exactly `string` streams as its
/// text arrives; any other parameter, nested elements included, is converted
/// through the schema once its element closes.
///
/// A tag counts only where the namespace token produced its namespace marker,
/// so the format spelled with ordinary tokens (the model quoting it) stays
/// text.
pub struct MinimaxM3ToolParser {
    buffer: DecodedText,
    mode: MinimaxM3Mode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
    markers: MinimaxM3Markers,
}

impl MinimaxM3ToolParser {
    /// Create a MiniMax M3 tool parser.
    pub fn new(tools: &[Tool], tokenizer: &dyn Tokenizer) -> Self {
        Self {
            buffer: DecodedText::default(),
            mode: MinimaxM3Mode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
            markers: MinimaxM3Markers::new(tokenizer),
        }
    }

    /// Apply one parsed MiniMax M3 event to parser state and output.
    fn apply_event(&mut self, event: MinimaxM3Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            MinimaxM3Event::Text { len: consumed_len } => {
                output.push_text(&self.buffer.text[..consumed_len]);
            }
            MinimaxM3Event::ToolBlockStart => self.mode = MinimaxM3Mode::ToolBlock,
            MinimaxM3Event::InvokeStart { name } => {
                // The template renders elements, not JSON, so the no-JSON default applies.
                let call = OpenToolCall::new(name, self.emitted_tool_count, JsonStyle::Spaced);
                self.emitted_tool_count += 1;
                self.mode = MinimaxM3Mode::Invoke {
                    call,
                    body: InvokeBody::Parameters {
                        invoke_end_scan: MarkerScanState::default(),
                    },
                };
            }
            MinimaxM3Event::ParameterStart { name } => {
                let (call, body) = self.mode.invoke_mut()?;
                *body = if self.tool_parameters.is_verbatim_string(&call.name, &name) {
                    call.arguments.open_string(&name);
                    InvokeBody::Parameter {
                        close_tag: self.markers.close_tag(&name),
                    }
                } else {
                    InvokeBody::BufferedParameter {
                        name,
                        body_scan: ElementBodyScanState::default(),
                    }
                };
            }
            MinimaxM3Event::ValueText { len } => {
                let (call, _) = self.mode.invoke_mut()?;
                call.arguments.push(&self.buffer.text[..len]);
            }
            MinimaxM3Event::ParameterEnd => {
                let (call, body) = self.mode.invoke_mut()?;
                call.arguments.close();
                *body = InvokeBody::Parameters {
                    invoke_end_scan: MarkerScanState::default(),
                };
            }
            MinimaxM3Event::BufferedValue { value } => {
                let (call, body) = self.mode.invoke_mut()?;
                let parameters = InvokeBody::Parameters {
                    invoke_end_scan: MarkerScanState::default(),
                };
                let InvokeBody::BufferedParameter { name, .. } =
                    std::mem::replace(body, parameters)
                else {
                    return Err(parsing_failed!(
                        "MiniMax M3 buffered value outside a buffered parameter"
                    ));
                };
                let value =
                    self.tool_parameters.convert_param_with_schema(&call.name, &name, value);
                call.arguments.write_value(&name, &value);
            }
            MinimaxM3Event::InvokeEnd => {
                let MinimaxM3Mode::Invoke { call, .. } =
                    std::mem::replace(&mut self.mode, MinimaxM3Mode::ToolBlock)
                else {
                    return Err(parsing_failed!("MiniMax M3 invoke end outside an invoke"));
                };
                output.push_call(call.finish());
            }
            MinimaxM3Event::ToolBlockEnd => self.mode = MinimaxM3Mode::Done,
            MinimaxM3Event::IgnoredRest => {}
        }
        if let MinimaxM3Mode::Invoke { call, .. } = &mut self.mode
            && let Some(delta) = call.take_delta()
        {
            output.push_call(delta);
        }
        Ok(())
    }
}

impl ToolParser for MinimaxM3ToolParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools, tokenizer.as_ref())))
    }

    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.append(chunk);

        while let Some((event, consumed_len)) =
            parse_buffered_event_attributed(&self.buffer, &self.markers, |input| {
                parse_next_minimax_m3_event(input, &mut self.mode)
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
            MinimaxM3Mode::Text => {
                output.push_text(&self.buffer.text);
            }
            MinimaxM3Mode::ToolBlock => {
                if !self.buffer.text.trim_start().is_empty() {
                    return Err(ToolParserError::IncompleteToolCall {
                        parser: "MiniMax M3",
                    });
                }
            }
            MinimaxM3Mode::Invoke { .. } => {
                return Err(ToolParserError::IncompleteToolCall {
                    parser: "MiniMax M3",
                });
            }
            MinimaxM3Mode::Done => {}
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = MinimaxM3Mode::Text;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer).text
    }
}

/// Parse a MiniMax M3 event for the current parser mode.
fn parse_next_minimax_m3_event(
    input: &mut MinimaxM3Input<'_>,
    mode: &mut MinimaxM3Mode,
) -> ModalResult<MinimaxM3Event> {
    match mode {
        MinimaxM3Mode::Text => parse_text_event(input),
        MinimaxM3Mode::ToolBlock => parse_tool_block_event(input),
        MinimaxM3Mode::Invoke { body, .. } => match body {
            InvokeBody::Parameters { invoke_end_scan } => {
                parse_invoke_event(input, invoke_end_scan)
            }
            InvokeBody::Parameter { close_tag } => parse_parameter_value_event(input, close_tag),
            InvokeBody::BufferedParameter { name, body_scan } => {
                buffered_value_event(input, name, body_scan)
            }
        },
        MinimaxM3Mode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode MiniMax M3 event.
fn parse_text_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    alt((tool_block_start_event, safe_text_event)).parse_next(input)
}

/// Parse a MiniMax M3 tool-block start marker.
fn tool_block_start_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    (&input.markers().tool_call_start)
        .value(MinimaxM3Event::ToolBlockStart)
        .parse_next(input)
}

/// Parse a safe text run before the next MiniMax M3 marker.
fn safe_text_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    let markers = input.markers();
    safe_text_len(input, &markers.tool_call_start).map(|len| MinimaxM3Event::Text { len })
}

/// Parse one event inside a MiniMax M3 tool block.
fn parse_tool_block_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    alt((tool_block_end_event, invoke_start_event)).parse_next(input)
}

/// Parse a MiniMax M3 tool-block end marker.
fn tool_block_end_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    let markers = input.markers();
    (ws0, &markers.tool_call_end)
        .value(MinimaxM3Event::ToolBlockEnd)
        .parse_next(input)
}

/// Parse a MiniMax M3 invoke start tag.
fn invoke_start_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    let markers = input.markers();
    let (name,) = seq!(
        _: ws0,
        _: &markers.invoke_start,
        _: (ws1, literal("name=")),
        partial_attr_value,
        _: literal(">"),
    )
    .parse_next(input)?;

    Ok(MinimaxM3Event::InvokeStart {
        name: name.trim().to_string(),
    })
}

/// Parse one event inside a MiniMax M3 invoke, between parameters.
fn parse_invoke_event(
    input: &mut MinimaxM3Input<'_>,
    invoke_end_scan: &mut MarkerScanState,
) -> ModalResult<MinimaxM3Event> {
    ws0.void().parse_next(input)?;
    let markers = input.markers();
    alt((
        (&markers.invoke_end).value(MinimaxM3Event::InvokeEnd),
        open_element_tag.map(|name: &str| MinimaxM3Event::ParameterStart {
            name: name.to_string(),
        }),
        |input: &mut MinimaxM3Input<'_>| skipped_invoke_rest_event(input, invoke_end_scan),
    ))
    .parse_next(input)
}

/// Skip ordinary text at a parameter boundary through the invoke end.
///
/// Be tolerant: ordinary text at an invoke parameter boundary ends this invoke.
/// Keep parsed parameters and drop the remaining invoke body.
fn skipped_invoke_rest_event(
    input: &mut MinimaxM3Input<'_>,
    invoke_end_scan: &mut MarkerScanState,
) -> ModalResult<MinimaxM3Event> {
    let markers = input.markers();
    if opt(peek(&markers.namespace)).parse_next(input)?.is_some() {
        return malformed();
    }
    seq!(
        _: take_until_marker(&markers.invoke_end, invoke_end_scan),
        _: &markers.invoke_end,
    )
    .value(MinimaxM3Event::InvokeEnd)
    .parse_next(input)
}

/// Parse a streamed value event inside a top-level string parameter.
///
/// A child element makes the value malformed, since its text already streamed.
fn parse_parameter_value_event(
    input: &mut MinimaxM3Input<'_>,
    close_tag: &Marker,
) -> ModalResult<MinimaxM3Event> {
    alt((
        close_tag.value(MinimaxM3Event::ParameterEnd),
        |input: &mut MinimaxM3Input<'_>| {
            let markers = input.markers();
            safe_text_len(input, &markers.namespace).map(|len| MinimaxM3Event::ValueText { len })
        },
    ))
    .parse_next(input)
}

/// Parse the complete element of a buffered parameter into its value.
fn buffered_value_event<'i>(
    input: &mut MinimaxM3Input<'i>,
    name: &str,
    body_scan: &mut ElementBodyScanState,
) -> ModalResult<MinimaxM3Event> {
    let close_tag = input.markers().close_tag(name);
    // Wait for the whole element, then parse its body once.
    let element = input.checkpoint();
    (
        |input: &mut MinimaxM3Input<'i>| take_element_body(input, body_scan),
        &close_tag,
    )
        .void()
        .parse_next(input)?;
    input.reset(&element);
    let _guard = ParserRecursionGuard::enter()?;
    let value = element_body(input, name)?;
    (&close_tag).void().parse_next(input)?;

    Ok(MinimaxM3Event::BufferedValue { value })
}

/// Parse an element body up to the close tag matching its open tag, counting
/// nested child tags, and leave the close tag for the caller.
///
/// Like [`take_until_marker`], it consumes nothing until that close tag has
/// arrived, and keeps its progress in `state` across chunks. Only genuine
/// namespace markers count as tags.
fn take_element_body<'i>(
    input: &mut MinimaxM3Input<'i>,
    state: &mut ElementBodyScanState,
) -> ModalResult<&'i str> {
    let markers = input.markers();
    // An element start is a prefix of an element end, so the end goes first:
    // index 0 is a close tag.
    let tags = [&markers.element_end_start, &markers.element_start];
    let start = input.offset();
    loop {
        let (at, index) = match scan_markers(input, &tags, start + state.scan_start)? {
            Scan::Found { at, index } => (at, index),
            Scan::Pending(at) | Scan::Clear(at) => {
                state.scan_start = at - start;
                return incomplete();
            }
        };
        match (index, state.depth) {
            // A close tag still arriving is rescanned from here.
            (0, 0) => {
                state.scan_start = at - start;
                return Ok(input.next_slice(at - start));
            }
            (0, _) => state.depth -= 1,
            _ => state.depth += 1,
        }
        state.scan_start = at - start + tags[index].as_str().len();
    }
}

/// Parse a MiniMax M3 parameter element.
fn parameter_element(input: &mut MinimaxM3Input<'_>) -> ModalResult<ParamElement> {
    let name = open_element_tag(input)?.to_string();
    let _guard = ParserRecursionGuard::enter()?;
    let value = element_body(input, &name)?;
    close_element_tag(input, &name)?;
    Ok(ParamElement { name, value })
}

/// Parse a MiniMax M3 opening element tag.
fn open_element_tag<'i>(input: &mut MinimaxM3Input<'i>) -> ModalResult<&'i str> {
    let markers = input.markers();
    let (name,) = seq!(
        _: &markers.element_start,
        take_until(1.., ">"),
        _: literal(">"),
    )
    .parse_next(input)?;

    if name.starts_with('/') || name.trim().is_empty() {
        return malformed();
    }

    Ok(name)
}

/// Parse a MiniMax M3 closing element tag.
fn close_element_tag(input: &mut MinimaxM3Input<'_>, name: &str) -> ModalResult<()> {
    (&input.markers().close_tag(name)).void().parse_next(input)
}

/// Parse the body of one MiniMax M3 element.
fn element_body(input: &mut MinimaxM3Input<'_>, closing_name: &str) -> ModalResult<ParamInput> {
    let markers = input.markers();
    let close_tag = markers.close_tag(closing_name);
    let mut text = String::new();
    let mut elements = Vec::new();

    loop {
        text.push_str(text_until_namespace(input)?);

        if opt(peek(&close_tag)).parse_next(input)?.is_some() {
            // Close tag reached, end of element body.
            break;
        }
        if opt(peek(&markers.element_start)).parse_next(input)?.is_some() {
            // Child element start reached, parse child element recursively.
            elements.push(parameter_element(input)?);
            continue;
        }
        // Unexpected namespace marker.
        return malformed();
    }

    if elements.is_empty() {
        Ok(ParamInput::Text(text))
    } else {
        if !text.trim().is_empty() {
            push_mixed_text_element(&mut elements, text);
        }
        Ok(ParamInput::Elements(elements))
    }
}

/// Parse text until the next MiniMax M3 namespace marker.
fn text_until_namespace<'i>(input: &mut MinimaxM3Input<'i>) -> ModalResult<&'i str> {
    let markers = input.markers();
    take_until_marker(&markers.namespace, &mut MarkerScanState::default()).parse_next(input)
}

/// Preserve mixed text content under a reserved object field.
///
/// By default, the field name is `$text`, but if that collides with an existing
/// child element name, prepend `$` until there is no collision.
fn push_mixed_text_element(elements: &mut Vec<ParamElement>, text: String) {
    let mut name = MIXED_TEXT_FIELD.to_string();
    while elements.iter().any(|element| element.name == name) {
        name.insert(0, '$');
    }
    elements.push(ParamElement {
        name,
        value: ParamInput::Text(text),
    });
}

/// Parse a quoted or unquoted XML attribute value from partial streaming input.
fn partial_attr_value<'i>(input: &mut MinimaxM3Input<'i>) -> ModalResult<&'i str> {
    alt((
        delimited(literal("\""), take_until(1.., "\""), literal("\"")),
        delimited(literal("'"), take_until(1.., "'"), literal("'")),
        take_until(1.., ">"),
    ))
    .parse_next(input)
}

/// Parse ignored rest after the MiniMax M3 tool block ends.
fn ignored_rest_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    rest.value(MinimaxM3Event::IgnoredRest).parse_next(input)
}

fn malformed<T>() -> ModalResult<T> {
    Err(ErrMode::Cut(ContextError::new()))
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use vllm_tokenizer::Tokenizer as _;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::{
        ELEMENT_END_START, ELEMENT_START, INVOKE_END, INVOKE_START, MinimaxM3ToolParser, NAMESPACE,
        TOOL_CALL_END, TOOL_CALL_START, ToolParser,
    };
    use crate::tool::test_utils::{
        collect_attributed_stream, decode_ids, split_by_chars, test_tools,
    };
    use crate::tool::{Result, Tool, ToolParserEvent, ToolParserOutput};
    use crate::utils::recursion::MAX_PARSER_RECURSION_DEPTH;

    /// The MiniMax M3 tokenizer has the namespace marker as an added token.
    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new().with_regular_token(NAMESPACE, 256)
    }

    /// Encode with the added token, so namespace markers are structural.
    fn dedicated(text: &str) -> Vec<u32> {
        tokenizer().encode(text, false).unwrap()
    }

    /// Encode as ordinary bytes only, so namespace marker spellings are text.
    fn ordinary(text: &str) -> Vec<u32> {
        tokenizer().encode_ordinary(text).unwrap()
    }

    /// Run token IDs through the real incremental decoder.
    fn decode(ids: &[u32]) -> vllm_tokenizer::DecodedText {
        decode_ids(&tokenizer(), ids)
    }

    /// Stream the structural encoding of the concatenated `chunks`, split at
    /// the chunk boundaries, and coalesce the tool-call deltas.
    fn collect_stream<T: ToolParser + ?Sized>(parser: &mut T, chunks: &[&str]) -> ToolParserOutput {
        collect_attributed_stream(parser, &tokenizer(), chunks)
    }

    fn parser() -> MinimaxM3ToolParser {
        MinimaxM3ToolParser::new(&m3_test_tools(), &tokenizer())
    }

    /// `ToolParserTestExt` over structurally encoded input.
    #[easy_ext::ext(M3ParserTestExt)]
    impl<T: ToolParser + ?Sized> T {
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

    fn element(name: &str, body: &str) -> String {
        format!("{ELEMENT_START}{name}>{body}{ELEMENT_END_START}{name}>")
    }

    fn invoke(function_name: &str, body: &str) -> String {
        format!("{INVOKE_START} name=\"{function_name}\">{body}{INVOKE_END}")
    }

    fn build_tool_block(invokes: &[(&str, String)]) -> String {
        let invokes = invokes
            .iter()
            .map(|(function_name, body)| invoke(function_name, body))
            .collect::<Vec<_>>()
            .join("\n");
        format!("{TOOL_CALL_START}\n{invokes}\n{TOOL_CALL_END}")
    }

    fn m3_test_tools() -> Vec<Tool> {
        let mut tools = test_tools();
        tools.push(Tool {
            name: "create_order".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "user_id": { "type": "integer" },
                    "urgent": { "type": "boolean" },
                    "note": { "type": "string" },
                    "shipping": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" },
                            "zip": { "type": "integer" }
                        }
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": { "type": "string" },
                                "qty": { "type": "integer" }
                            }
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": { "type": "integer" }
                    },
                    "duplicate_demo": {
                        "type": "object",
                        "properties": {
                            "tag": { "type": "string" }
                        }
                    },
                    "schema_mismatch_array": {
                        "type": "array",
                        "items": { "type": "integer" }
                    }
                }
            }),
            strict: None,
            defer_loading: None,
        });
        tools
    }

    fn order_arguments() -> String {
        let shipping = element(
            "shipping",
            &format!(
                "{}{}",
                element("city", "Singapore"),
                element("zip", "018956")
            ),
        );
        let first_item = element(
            "item",
            &format!("{}{}", element("sku", "book-001"), element("qty", "2")),
        );
        let second_item = element(
            "item",
            &format!("{}{}", element("sku", "pen-007"), element("qty", "5")),
        );
        let items = element("items", &format!("{first_item}{second_item}"));
        let metadata = element(
            "metadata",
            &format!("{}{}", element("score", "42"), element("rank", "7")),
        );
        let duplicate_demo = element(
            "duplicate_demo",
            &format!("{}{}", element("tag", "a"), element("tag", "b")),
        );
        let schema_mismatch_array = element(
            "schema_mismatch_array",
            &format!("{}{}", element("x", "1"), element("x", "2")),
        );

        [
            element("user_id", "42"),
            element("urgent", "true"),
            element("note", "Please leave at front desk."),
            shipping,
            items,
            metadata,
            duplicate_demo,
            schema_mismatch_array,
            element(
                "unknown_struct",
                &format!("{}{}", element("a", "1"), element("a", "2")),
            ),
        ]
        .join("")
    }

    fn nested_element(depth: usize) -> String {
        format!(
            "{}leaf{}",
            format!("{ELEMENT_START}node>").repeat(depth),
            format!("{ELEMENT_END_START}node>").repeat(depth)
        )
    }

    #[test]
    fn minimax_m3_recursion_limit_accepts_boundary_and_rejects_next_depth() {
        let parse = |depth| {
            parser().parse_complete(&build_tool_block(&[("convert", nested_element(depth))]))
        };
        assert!(parse(MAX_PARSER_RECURSION_DEPTH).is_ok());
        assert!(parse(MAX_PARSER_RECURSION_DEPTH + 1).is_err());
    }

    #[test]
    fn minimax_m3_parse_complete_without_tool_call_keeps_text() {
        let mut parser = parser();
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn minimax_m3_parse_complete_extracts_single_tool_call() {
        let mut parser = parser();
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "get_weather",
                format!("{}{}", element("city", "Seattle"), element("days", "5")),
            )]))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle", "days": 5 })
        );
    }

    #[test]
    fn minimax_m3_parse_complete_preserves_prefix_and_ignores_trailing_text() {
        let mut parser = parser();
        let output = format!(
            "Let me check. {} This trailing text is ignored.",
            build_tool_block(&[("get_weather", element("city", "Seattle"))])
        );
        let output = parser.parse_complete(&output).unwrap();

        assert_eq!(output.normal_text(), "Let me check. ");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn minimax_m3_parse_complete_extracts_multiple_invokes() {
        let mut parser = parser();
        let output = parser
            .parse_complete(&build_tool_block(&[
                ("get_weather", element("city", "Seattle")),
                ("get_weather", element("city", "NYC")),
            ]))
            .unwrap();

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[1].tool_index, 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[1].arguments).unwrap(),
            json!({ "city": "NYC" })
        );
    }

    #[test]
    fn minimax_m3_invoke_body_junk_drops_rest_of_invoke() {
        let mut parser = parser();
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "get_weather",
                [
                    element("city", "Seattle"),
                    "I need to use the city above.".to_string(),
                    element("days", "5"),
                ]
                .join(""),
            )]))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
    }

    #[test]
    fn minimax_m3_parse_complete_converts_schema_types() {
        let mut parser = parser();
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "convert",
                [
                    element("whole", "5.0"),
                    element("flag", "true"),
                    element("payload", r#"{"nested":true}"#),
                    element("items", "[1,2]"),
                    element("empty", "42"),
                ]
                .join(""),
            )]))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "whole": 5.0,
                "flag": true,
                "payload": { "nested": true },
                "items": [1, 2],
                "empty": "42",
            })
        );
    }

    #[test]
    fn minimax_m3_parse_complete_converts_nested_arguments() {
        let mut parser = parser();
        let output = parser
            .parse_complete(&build_tool_block(&[("create_order", order_arguments())]))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "user_id": 42,
                "urgent": true,
                "note": "Please leave at front desk.",
                "shipping": {
                    "city": "Singapore",
                    "zip": 18956
                },
                "items": [
                    {
                        "sku": "book-001",
                        "qty": 2
                    },
                    {
                        "sku": "pen-007",
                        "qty": 5
                    }
                ],
                "metadata": {
                    "score": 42,
                    "rank": 7
                },
                "duplicate_demo": {
                    "tag": ["a", "b"]
                },
                "schema_mismatch_array": [1, 2],
                "unknown_struct": {
                    "a": ["1", "2"]
                }
            })
        );
    }

    #[test]
    fn minimax_m3_parse_complete_handles_multiline_leaf_parameters() {
        let mut parser = parser();
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "calculate_area",
                [
                    element("shape", "\nrectangle\n"),
                    element("dimensions", r#"{"width":10,"height":20}"#),
                    element("precision", "2"),
                ]
                .join(""),
            )]))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "shape": "\nrectangle\n",
                "dimensions": { "width": 10, "height": 20 },
                "precision": 2,
            })
        );
    }

    #[test]
    fn minimax_m3_streaming_extracts_single_tool_call() {
        let mut parser = parser();
        let output = collect_stream(
            &mut parser,
            &[
                TOOL_CALL_START,
                &invoke("get_weather", &element("city", "Seattle")),
                TOOL_CALL_END,
            ],
        );

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
    }

    #[test]
    fn minimax_m3_streaming_preserves_prefix_text() {
        let mut parser = parser();
        let output = collect_stream(
            &mut parser,
            &[
                "Let me check. ",
                TOOL_CALL_START,
                &invoke("get_weather", &element("city", "Seattle")),
                TOOL_CALL_END,
            ],
        );

        assert_eq!(output.normal_text(), "Let me check. ");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn minimax_m3_streaming_preserves_ordered_events() {
        let mut parser = parser();
        let output = collect_stream(
            &mut parser,
            &[
                "Let me check. ",
                TOOL_CALL_START,
                &invoke("get_weather", &element("city", "Seattle")),
                TOOL_CALL_END,
            ],
        );

        assert_eq!(output.events.len(), 2);
        assert_eq!(
            output.events[0],
            ToolParserEvent::Text("Let me check. ".to_string())
        );
        let ToolParserEvent::ToolCall(call) = &output.events[1] else {
            panic!("expected tool-call event");
        };
        assert_eq!(call.name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
    }

    #[test]
    fn minimax_m3_streaming_without_tool_call_emits_text_incrementally() {
        let mut parser = parser();
        let output = collect_stream(&mut parser, &["Hello, ", "world!"]);

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn minimax_m3_streaming_handles_marker_split_across_chunks() {
        let text = build_tool_block(&[("get_weather", element("city", "Seattle"))]);
        let chunks = split_by_chars(&text, 3);
        let mut parser = parser();
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert!(output.normal_text().is_empty());
    }

    #[test]
    fn minimax_m3_streaming_extracts_multiple_invokes_in_order() {
        let text = build_tool_block(&[
            ("get_weather", element("city", "Seattle")),
            ("get_weather", element("city", "NYC")),
        ]);
        let chunks = split_by_chars(&text, 7);
        let mut parser = parser();
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[1].tool_index, 1);
    }

    #[test]
    fn minimax_m3_streams_string_parameters_before_the_invoke_closes() {
        let mut parser = parser();
        let mut output = ToolParserOutput::default();

        // The call and its arguments so far, after each chunk.
        let mut progress = Vec::new();
        for chunk in [
            format!("{TOOL_CALL_START}\n{INVOKE_START} name=\"create_order\">"),
            format!("{ELEMENT_START}note>first line\nsec"),
            format!("ond{ELEMENT_END_START}note>"),
            format!("{ELEMENT_START}shipping>{}", element("city", "Singapore")),
            format!("{ELEMENT_END_START}shipping>{}", element("user_id", "42")),
            format!("{INVOKE_END}\n{TOOL_CALL_END}"),
        ] {
            output.append(parser.parse_chunk(&chunk).unwrap());
            progress.push(match output.clone().coalesce().calls().first() {
                Some(call) => format!("{}({})", call.name.as_deref().unwrap(), call.arguments),
                None => "no call".to_string(),
            });
        }

        expect![[r#"
            create_order()
            create_order({"note": "first line\nsec)
            create_order({"note": "first line\nsecond")
            create_order({"note": "first line\nsecond")
            create_order({"note": "first line\nsecond", "shipping": {"city": "Singapore"}, "user_id": 42)
            create_order({"note": "first line\nsecond", "shipping": {"city": "Singapore"}, "user_id": 42})"#]].assert_eq(&progress.join("\n"));
    }

    #[test]
    fn minimax_m3_child_element_in_string_parameter_is_malformed() {
        // The string value streams as it arrives, so it cannot become an object.
        let mut parser = parser();
        let body = element("note", &format!("text {}", element("child", "value")));

        assert!(parser.parse_complete(&build_tool_block(&[("create_order", body)])).is_err());
    }

    #[test]
    fn minimax_m3_streaming_matches_parse_complete_at_every_chunk_size() {
        let text = build_tool_block(&[("create_order", order_arguments())]);
        let complete = parser().parse_complete(&text).unwrap();
        for size in 1..=8 {
            let mut parser = parser();
            let streamed = collect_stream(&mut parser, &split_by_chars(&text, size));
            assert_eq!(streamed, complete, "chunk size {size}");
        }
    }

    #[test]
    fn minimax_m3_streaming_ignores_text_after_tool_block() {
        let text = format!(
            "{} ignored",
            build_tool_block(&[("get_weather", element("city", "Seattle"))])
        );
        let chunks = split_by_chars(&text, 5);
        let mut parser = parser();
        let output = collect_stream(&mut parser, &chunks);

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn minimax_m3_finish_fails_incomplete_tool_call() {
        let mut parser = parser();
        parser
            .parse_chunk(&format!(
                "{TOOL_CALL_START}{INVOKE_START} name=\"get_weather\">"
            ))
            .unwrap();

        assert!(parser.finish().is_err());
    }

    #[test]
    fn minimax_m3_finish_recovers_after_bare_tool_block_start() {
        let mut parser = parser();
        parser.parse_chunk(TOOL_CALL_START).unwrap();

        let output = parser.finish().unwrap();
        assert!(output.normal_text().is_empty());
        assert!(output.calls().is_empty());
    }

    #[test]
    fn minimax_m3_finish_recovers_completed_invoke_with_whitespace_tail() {
        let mut parser = parser();
        let output = parser
            .parse_complete(&format!(
                "{}\n{}\n  \n",
                TOOL_CALL_START,
                invoke("get_weather", &element("city", "Seattle"))
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
    }

    #[test]
    fn minimax_m3_finish_fails_partial_outer_end_marker() {
        let text = format!(
            "{TOOL_CALL_START}\n{}\n{TOOL_CALL_END}",
            invoke("get_weather", &element("city", "Seattle")),
        );
        // Stop three bytes into the namespace token of the end marker.
        let partial = decode(&dedicated(&text)).drain_prefix(text.len() - TOOL_CALL_END.len() + 3);
        let mut parser = parser();
        parser.parse_into(partial, &mut ToolParserOutput::default()).unwrap();

        assert!(parser.finish().is_err());
    }

    #[test]
    fn minimax_m3_malformed_tool_call_fails_fast() {
        let mut parser = parser();
        let error = parser
            .parse_chunk(&format!(
                "{TOOL_CALL_START}{ELEMENT_START}bad>{TOOL_CALL_END}"
            ))
            .unwrap_err();

        expect![[
            r#"tool parser parsing failed: near "]<]minimax[>[<bad>]<]minimax[>[</tool_call>": "#
        ]]
        .assert_eq(&error.to_report_string());
    }

    #[test]
    fn minimax_m3_mixed_content_is_preserved_as_text_field() {
        let mut parser = parser();
        let body = element(
            "payload",
            &format!("text before {} text after", element("child", "value")),
        );
        let output = parser.parse_complete(&build_tool_block(&[("convert", body)])).unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "payload": {
                    "child": "value",
                    "$text": "text before  text after"
                }
            })
        );
    }

    #[test]
    fn minimax_m3_mixed_text_field_avoids_child_name_collision() {
        let mut parser = parser();
        let body = element(
            "payload",
            &format!(
                "text{}{}",
                element("$text", "child text"),
                element("child", "value")
            ),
        );
        let output = parser.parse_complete(&build_tool_block(&[("convert", body)])).unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "payload": {
                    "$text": "child text",
                    "$$text": "text",
                    "child": "value"
                }
            })
        );
    }

    /// Feed `ids` decoded once and split at the byte offsets `splits`, then
    /// finish and coalesce.
    fn stream_ids(ids: &[u32], splits: &[usize]) -> ToolParserOutput {
        let mut parser = parser();
        let mut rest = decode(ids);
        let mut output = ToolParserOutput::default();
        let mut consumed = 0;
        for &split in splits {
            parser.parse_into(rest.drain_prefix(split - consumed), &mut output).unwrap();
            consumed = split;
        }
        parser.parse_into(rest, &mut output).unwrap();
        output.append(parser.finish().unwrap());
        output.coalesce()
    }

    /// Stream `ids` split at every char boundary and return the arguments of
    /// each run's single call.
    fn arguments_at_every_split(ids: &[u32]) -> Vec<Value> {
        let text = decode(ids).text;
        text.char_indices()
            .map(|(at, _)| {
                let output = stream_ids(ids, &[at]);
                assert_eq!(output.calls().len(), 1, "split {at}");
                serde_json::from_str(&output.calls()[0].arguments).unwrap()
            })
            .collect()
    }

    #[test]
    fn minimax_m3_ordinary_marker_spelling_inside_string_parameter_stays_value() {
        let quoted = format!("{ELEMENT_END_START}note>{ELEMENT_START}x>");
        let ids = [
            dedicated(&format!(
                "{TOOL_CALL_START}\n{INVOKE_START} name=\"create_order\">{ELEMENT_START}note>a"
            )),
            ordinary(&quoted),
            dedicated(&format!(
                "b{ELEMENT_END_START}note>{INVOKE_END}\n{TOOL_CALL_END}"
            )),
        ]
        .concat();

        for arguments in arguments_at_every_split(&ids) {
            assert_eq!(arguments, json!({ "note": format!("a{quoted}b") }));
        }
    }

    #[test]
    fn minimax_m3_ordinary_marker_spelling_inside_buffered_parameter_stays_value() {
        let quoted = format!("{ELEMENT_END_START}shipping>{ELEMENT_START}x>");
        let ids = [
            dedicated(&format!(
                "{TOOL_CALL_START}\n{INVOKE_START} name=\"create_order\">{ELEMENT_START}shipping>{ELEMENT_START}city>a"
            )),
            ordinary(&quoted),
            dedicated(&format!(
                "b{ELEMENT_END_START}city>{ELEMENT_END_START}shipping>{INVOKE_END}\n{TOOL_CALL_END}"
            )),
        ]
        .concat();

        for arguments in arguments_at_every_split(&ids) {
            assert_eq!(
                arguments,
                json!({ "shipping": { "city": format!("a{quoted}b") } })
            );
        }
    }

    #[test]
    fn minimax_m3_ordinary_partial_marker_streams_immediately() {
        let mut parser = parser();
        let mut output = ToolParserOutput::default();
        let ids = [dedicated("Text "), ordinary("ending in ]<]mini")].concat();
        parser.parse_into(decode(&ids), &mut output).unwrap();

        // Not held back: ordinary text cannot grow into the namespace token.
        assert_eq!(output.normal_text(), "Text ending in ]<]mini");
    }

    #[test]
    fn minimax_m3_ordinary_partial_marker_in_string_parameter_streams_immediately() {
        let mut parser = parser();
        let mut output = ToolParserOutput::default();
        let ids = [
            dedicated(&format!(
                "{TOOL_CALL_START}\n{INVOKE_START} name=\"create_order\">{ELEMENT_START}note>"
            )),
            ordinary("half ]<]mini"),
        ]
        .concat();
        parser.parse_into(decode(&ids), &mut output).unwrap();

        // Not held back: ordinary text cannot grow into the namespace token.
        let output = output.coalesce();
        assert_eq!(output.calls()[0].arguments, r#"{"note": "half ]<]mini"#);
    }

    #[test]
    fn minimax_m3_ordinary_tool_call_markup_stays_text() {
        let call = build_tool_block(&[("get_weather", element("city", "Seattle"))]);
        let ids = [dedicated("Write "), ordinary(&call), dedicated(" here.")].concat();

        let output = stream_ids(&ids, &[]);
        assert_eq!(output.normal_text(), format!("Write {call} here."));
        assert!(output.calls().is_empty());
    }

    #[test]
    fn minimax_m3_without_namespace_token_matches_markers_by_spelling() {
        let tokenizer = TestTokenizer::new();
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools(), &tokenizer);
        let call = build_tool_block(&[("get_weather", element("city", "Seattle"))]);
        let ids = tokenizer.encode_ordinary(&call).unwrap();

        let mut output = ToolParserOutput::default();
        parser.parse_into(decode_ids(&tokenizer, &ids), &mut output).unwrap();
        output.append(parser.finish().unwrap());
        let output = output.coalesce();
        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
    }
}
