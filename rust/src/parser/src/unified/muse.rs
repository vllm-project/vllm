// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser for the Muse chat format (Muse Spark and Muse Glimmer).
//!
//! The generation prompt ends with `<|start|>assistant`; the completion body is
//! a sequence of recipient-routed segments forming one assistant turn:
//!
//! ```text
//!  to=self<|message|>REASONING<|eom|><|start|>assistant
//!  to=commentary<|message|>VISIBLE<|eom|><|start|>assistant
//!  to=FUNCTION<|message|>
//!  <atem:function_calls>
//!  <atem:invoke name="FUNCTION">
//!  <atem:parameter name="KEY">VALUE</atem:parameter>
//!  </atem:invoke>
//!  </atem:function_calls><|eom|><|start|>assistant
//!  to=user<|message|>FINAL<|eot|>
//! ```
//!
//! `self` bodies are reasoning, `commentary`/`user` bodies are visible text,
//! and any other recipient names a tool whose body is one ATEM block with one
//! or more `<atem:invoke>` calls. Each invoke streams as one tool call: its
//! name once the open tag is complete, then its arguments JSON object in
//! parameter order. A parameter whose schema admits only a string retains its
//! raw text, which streams as it arrives, escaped into a JSON string. Any other
//! value is decoded once complete: JSON when it decodes to a type the schema
//! admits, and its raw text otherwise. A parameter repeated within one invoke
//! keeps its first value, which may already be on the wire.
//!
//! The model sometimes opens the next segment without closing the current one
//! with `<|eom|>`, e.g. writing `to=FUNCTION<|message|>` straight after its
//! reasoning when calling a tool with empty arguments. Since `<|start|>` and
//! `<|message|>` never occur inside a body, either one also ends the current
//! segment and starts the next. It sometimes omits the last
//! `</atem:parameter>` too, so a parameter value can also end where its invoke
//! closes.

use std::collections::{HashMap, HashSet};

use serde_json::Value;
use vllm_tokenizer::{DecodedText, DynTokenizer};
use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, eof, opt, peek, preceded, seq};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_till, take_until, take_while};

use super::{Result, UnifiedParser, UnifiedParserError, UnifiedParserOutput, token_id};
use crate::schema::{JsonType, OptionSource, SchemaRoot};
use crate::tool::{JsonStyle, OpenToolCall, Tool};
use crate::unified::parsing_failed;
use crate::utils::{
    MarkerScanState, parse_buffered_event, safe_text_len_mul, take_until_marker,
    take_until_marker_mul,
};

const START: &str = "<|start|>";
const MESSAGE: &str = "<|message|>";
const EOM: &str = "<|eom|>";
const EOT: &str = "<|eot|>";

const ASSISTANT_START: &str = "<|start|>assistant";
const RECIPIENT_PREFIX: &str = "to=";
const ATEM_OPEN: &str = "<atem:function_calls>";
const ATEM_CLOSE: &str = "</atem:function_calls>";
const INVOKE_OPEN: &str = "<atem:invoke";
const INVOKE_CLOSE: &str = "</atem:invoke>";
const PARAMETER_OPEN: &str = "<atem:parameter";
const PARAMETER_CLOSE: &str = "</atem:parameter>";

/// Markers that end a segment body run: the boundary markers, or the possible
/// start of the next segment's header when `<|eom|>` is missing.
const BODY_MARKERS: &[&str] = &[EOM, EOT, START, MESSAGE, RECIPIENT_PREFIX];

/// Markers that end a parameter value: its close tag, or the invoke close that
/// ends a last value missing it, after one separating newline the value drops.
const VALUE_END_MARKERS: &[&str] = &[PARAMETER_CLOSE, "\n</atem:invoke>", INVOKE_CLOSE];

/// Recipient whose segments are dropped: `developer` is a prompt role, not a
/// valid assistant recipient.
const DEVELOPER_RECIPIENT: &str = "developer";

type MuseInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum MuseEvent {
    Text,
    Reasoning,
    /// Structural noise consumed without emitting anything.
    Skip,
    /// A complete segment routing header; `None` for a bare `<|message|>`
    /// without recipient.
    SegmentHeader {
        recipient: Option<String>,
    },
    /// `<|eom|>`: the current segment ends and another segment may follow.
    SegmentEnd,
    /// `<|eot|>`: the assistant turn ends.
    TurnEnd,
    /// `<atem:function_calls>`: the ATEM block of a tool segment opens.
    BlockOpen,
    /// A complete `<atem:invoke>` open tag.
    InvokeOpen {
        name: String,
    },
    /// A complete `<atem:parameter>` open tag.
    ParameterOpen {
        key: String,
    },
    /// The drained text continues the open string parameter value.
    ParameterText,
    /// `</atem:parameter>` after a string parameter value.
    ParameterClose,
    /// The complete raw value of a buffered parameter, through its close tag;
    /// `closes_invoke` when `</atem:invoke>` ended it instead.
    BufferedValue {
        value: String,
        closes_invoke: bool,
    },
    /// `</atem:invoke>`, which also ends a string parameter value missing its
    /// close tag.
    InvokeClose,
    /// `</atem:function_calls>`: the ATEM block closes.
    BlockClose,
}

#[derive(Debug)]
enum MuseMode {
    /// Awaiting the ` to=RECIPIENT<|message|>` routing header of the next
    /// segment; the full `<|start|>assistant to=…` form is also accepted.
    Header { scan: MarkerScanState },
    /// Inside a `to=self` segment (private reasoning), or a bare `<|message|>`
    /// segment without recipient.
    Reasoning,
    /// Inside a `to=commentary`/`to=user` segment (visible text).
    Text,
    /// Inside a tool segment, streaming its ATEM block.
    Tool { block: AtemBlock },
    /// Dropping text until the next segment boundary: noise after a complete
    /// ATEM block, or the body of a `to=developer` segment.
    Skip,
    /// After `<|eot|>`: ignore the rest (EOS leakage guard).
    Done,
}

/// Where the parser is inside the ATEM block of a tool segment.
#[derive(Debug)]
enum AtemBlock {
    /// Before `<atem:function_calls>`.
    Start,
    /// Between invokes; `any` tells whether one was already closed, since a
    /// block holds at least one.
    Invokes { any: bool },
    /// Inside one invoke, streaming it as a tool call.
    Invoke {
        call: OpenToolCall,
        body: InvokeBody,
    },
}

/// Where the parser is inside one `<atem:invoke>`.
#[derive(Debug)]
enum InvokeBody {
    /// Between parameters.
    Parameters,
    /// Inside a string parameter value, which streams as it arrives.
    Parameter,
    /// Inside another parameter value, which is decoded once complete.
    BufferedParameter {
        key: String,
        value_end_scan: MarkerScanState,
    },
}

impl MuseMode {
    /// The position inside the ATEM block, for an event inside a tool segment.
    fn block_mut(&mut self) -> Result<&mut AtemBlock> {
        match self {
            MuseMode::Tool { block } => Ok(block),
            _ => Err(parsing_failed!("Muse ATEM event outside a tool segment")),
        }
    }

    /// The open invoke's call and the position inside it, for an event inside
    /// an `<atem:invoke>`.
    fn invoke_mut(&mut self) -> Result<(&mut OpenToolCall, &mut InvokeBody)> {
        match self {
            MuseMode::Tool {
                block: AtemBlock::Invoke { call, body },
            } => Ok((call, body)),
            _ => Err(parsing_failed!("Muse invoke event outside an invoke")),
        }
    }
}

impl Default for MuseMode {
    fn default() -> Self {
        Self::Header {
            scan: MarkerScanState::default(),
        }
    }
}

/// Unified parser for Muse recipient routing and ATEM tool calls.
pub struct MuseUnifiedParser {
    buffer: DecodedText,
    mode: MuseMode,
    /// Names of the tools registered on the request.
    tool_names: HashSet<String>,
    parameters: HashMap<String, Value>,
    /// Number of calls emitted in the current response.
    emitted_call_count: usize,
}

impl MuseUnifiedParser {
    /// Create a Muse parser.
    pub fn new(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        token_id(tokenizer.as_ref(), START)?;
        token_id(tokenizer.as_ref(), MESSAGE)?;
        token_id(tokenizer.as_ref(), EOM)?;
        token_id(tokenizer.as_ref(), EOT)?;

        Ok(Self {
            buffer: DecodedText::default(),
            mode: MuseMode::default(),
            tool_names: tools.iter().map(|tool| tool.name.clone()).collect(),
            parameters: tools
                .iter()
                .map(|tool| (tool.name.clone(), tool.parameters.clone()))
                .collect(),
            emitted_call_count: 0,
        })
    }

    fn apply_event(
        &mut self,
        event: MuseEvent,
        piece: DecodedText,
        output: &mut UnifiedParserOutput,
    ) -> Result<()> {
        match event {
            MuseEvent::Text => output.push_text(piece.text),
            MuseEvent::Reasoning => output.push_reasoning(piece),
            MuseEvent::Skip => {}
            MuseEvent::SegmentHeader { recipient } => {
                self.mode = match recipient.as_deref() {
                    None | Some("self") => MuseMode::Reasoning,
                    Some("commentary" | "user") => MuseMode::Text,
                    Some(DEVELOPER_RECIPIENT) => MuseMode::Skip,
                    Some(_) => MuseMode::Tool {
                        block: AtemBlock::Start,
                    },
                };
            }
            MuseEvent::SegmentEnd => {
                self.mode = MuseMode::Header {
                    scan: MarkerScanState::default(),
                };
            }
            MuseEvent::TurnEnd => self.mode = MuseMode::Done,
            MuseEvent::BlockOpen => *self.mode.block_mut()? = AtemBlock::Invokes { any: false },
            MuseEvent::InvokeOpen { name } => {
                let name = normalize_tool_name(name, &self.tool_names);
                let tool_index = self.emitted_call_count;
                self.emitted_call_count += 1;
                *self.mode.block_mut()? = AtemBlock::Invoke {
                    // The template renders JSON values with Python separators.
                    call: OpenToolCall::new(name, tool_index, JsonStyle::Spaced),
                    body: InvokeBody::Parameters,
                };
            }
            MuseEvent::ParameterOpen { key } => {
                let (call, body) = self.mode.invoke_mut()?;
                let parameters = self.parameters.get(&call.name).unwrap_or(&Value::Null);
                if admits_only_string(parameters, &key) {
                    call.arguments.open_string(&key);
                    *body = InvokeBody::Parameter;
                } else {
                    *body = InvokeBody::BufferedParameter {
                        key,
                        value_end_scan: MarkerScanState::default(),
                    };
                }
            }
            MuseEvent::ParameterText => self.mode.invoke_mut()?.0.arguments.push(&piece.text),
            MuseEvent::ParameterClose => {
                let (call, body) = self.mode.invoke_mut()?;
                call.arguments.close();
                *body = InvokeBody::Parameters;
            }
            MuseEvent::BufferedValue {
                value,
                closes_invoke,
            } => {
                let (call, body) = self.mode.invoke_mut()?;
                let InvokeBody::BufferedParameter { key, .. } =
                    std::mem::replace(body, InvokeBody::Parameters)
                else {
                    return Err(parsing_failed!("Muse parameter value outside a parameter"));
                };
                let parameters = self.parameters.get(&call.name).unwrap_or(&Value::Null);
                let value = convert_parameter_value(parameters, &key, value);
                call.arguments.write_value(&key, &value);
                if closes_invoke {
                    self.close_invoke(output)?;
                }
            }
            MuseEvent::InvokeClose => self.close_invoke(output)?,
            MuseEvent::BlockClose => self.mode = MuseMode::Skip,
        }
        if let MuseMode::Tool {
            block: AtemBlock::Invoke { call, .. },
        } = &mut self.mode
            && let Some(delta) = call.take_delta()
        {
            output.push_call(delta);
        }
        Ok(())
    }

    /// Close the open invoke and emit the rest of its call.
    fn close_invoke(&mut self, output: &mut UnifiedParserOutput) -> Result<()> {
        let block = std::mem::replace(self.mode.block_mut()?, AtemBlock::Invokes { any: true });
        let AtemBlock::Invoke { mut call, body } = block else {
            return Err(parsing_failed!("Muse invoke close outside an invoke"));
        };
        // A string value missing its `</atem:parameter>` ends here.
        if let InvokeBody::Parameter = body {
            call.arguments.close();
        }
        output.push_call(call.finish());
        Ok(())
    }
}

impl UnifiedParser for MuseUnifiedParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Self::new(tools, tokenizer).map(|parser| Box::new(parser) as Box<dyn UnifiedParser>)
    }

    fn initialize(&mut self, _prompt_token_ids: &[u32]) -> Result<()> {
        // The generation prompt always ends with `<|start|>assistant`, so the
        // completion starts with the bare ` to=RECIPIENT<|message|>` header
        // continuation; no prefill channel detection is needed.
        self.reset();
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    // TODO: add a `structural_tag_builder` covering the full output envelope
    // (self/commentary/user recipients and ATEM tool calls) for strict tool
    // calling and JSON response formats.

    fn parse_into(&mut self, delta: DecodedText, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffer.append(delta);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer.text, |input| {
            parse_next_muse_event(input, &mut self.mode)
        })? {
            let piece = self.buffer.drain_prefix(consumed_len);
            self.apply_event(event, piece, output)?;
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();

        match &self.mode {
            // A truncated header (or marker-free plain text) falls back to
            // visible text, matching how other unified parsers treat
            // incomplete output.
            MuseMode::Header { .. } | MuseMode::Text => {
                output.push_text(self.buffer.take().text);
            }
            MuseMode::Reasoning => output.push_reasoning(self.buffer.take()),
            MuseMode::Skip | MuseMode::Done => self.buffer.clear(),
            MuseMode::Tool { .. } => {
                return Err(UnifiedParserError::IncompleteToolCall { parser: "Muse" });
            }
        }

        self.mode = MuseMode::default();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = MuseMode::default();
        self.emitted_call_count = 0;
        self.buffer.take().text
    }
}

/// Map an emitted ATEM invoke name back to a registered tool name.
///
/// The Muse Glimmer template lists a registered bare name `NAME` as the
/// namespace recipient `"NAME.*"`, so the model may call it as `NAME.NAME`, or
/// fill the `*` with a parameter, a verb or nothing (`NAME.city`, `NAME.`).
/// Collapse to `NAME` only when `NAME` is registered and the emitted name is
/// not; for a suffix other than `NAME`, also only when no registered name lives
/// under `NAME.`, since the suffix may then mean a missing sibling. Matching on
/// the trailing segment alone could dispatch `weather.get` to a registered
/// `calendar.get`. Same rule as the Python `MuseToolParser._normalize_name`.
fn normalize_tool_name(name: String, tool_names: &HashSet<String>) -> String {
    if tool_names.contains(&name) {
        return name;
    }
    let Some((head, tail)) = name.split_once('.') else {
        return name;
    };
    // Python never registers an empty name, so an emitted `.x` stays as-is.
    let has_sibling = || {
        tool_names
            .iter()
            .any(|n| n.strip_prefix(head).is_some_and(|r| r.starts_with('.')))
    };
    if !head.is_empty() && tool_names.contains(head) && (head == tail || !has_sibling()) {
        tracing::debug!(emitted = %name, mapped = head, "muse: mapped emitted tool name");
        return head.to_string();
    }
    name
}

/// Parse one Muse event from buffered streaming input.
fn parse_next_muse_event(input: &mut MuseInput<'_>, mode: &mut MuseMode) -> ModalResult<MuseEvent> {
    match mode {
        MuseMode::Header { scan } => alt((
            literal(EOT).value(MuseEvent::TurnEnd),
            |input: &mut MuseInput<'_>| header_event(input, scan),
        ))
        .parse_next(input),
        MuseMode::Reasoning => body_event(input, MuseEvent::Reasoning),
        MuseMode::Text => body_event(input, MuseEvent::Text),
        MuseMode::Tool { block } => parse_atem_event(input, block),
        MuseMode::Skip => body_event(input, MuseEvent::Skip),
        MuseMode::Done => parse_done_event(input),
    }
}

/// Parse one segment routing header up to and including `<|message|>`.
fn header_event<'i>(
    input: &mut MuseInput<'i>,
    scan: &mut MarkerScanState,
) -> ModalResult<MuseEvent> {
    let (header,) = seq!(
        take_until_marker(MESSAGE, scan),
        _: literal(MESSAGE),
    )
    .parse_next(input)?;
    let recipient = parse_segment_recipient(header)?;

    Ok(MuseEvent::SegmentHeader { recipient })
}

/// Parse an event inside a segment body, emitting body text as `body`.
fn body_event(input: &mut MuseInput<'_>, body: MuseEvent) -> ModalResult<MuseEvent> {
    alt((
        literal(EOM).value(MuseEvent::SegmentEnd),
        literal(EOT).value(MuseEvent::TurnEnd),
        inline_header_event,
        // A `to=` not followed by `NAME<|message|>` is ordinary body text.
        literal(RECIPIENT_PREFIX).value(body.clone()),
        safe_body_event.value(body),
    ))
    .parse_next(input)
}

/// Parse safe body text before the next body marker.
fn safe_body_event(input: &mut MuseInput<'_>) -> ModalResult<usize> {
    safe_text_len_mul(input, BODY_MARKERS)
}

/// Parse the header of a segment opened inside a body whose `<|eom|>` is
/// missing: `<|start|>assistant to=NAME<|message|>`, a bare
/// `to=NAME<|message|>`, or a recipient-less `<|message|>`.
fn inline_header_event(input: &mut MuseInput<'_>) -> ModalResult<MuseEvent> {
    alt((
        preceded(peek(literal(START)), |input: &mut MuseInput<'_>| {
            header_event(input, &mut MarkerScanState::default())
        }),
        delimited(
            literal(RECIPIENT_PREFIX),
            take_while(1.., is_recipient_char),
            literal(MESSAGE),
        )
        .map(|recipient: &str| MuseEvent::SegmentHeader {
            recipient: Some(recipient.to_string()),
        }),
        literal(MESSAGE).value(MuseEvent::SegmentHeader { recipient: None }),
    ))
    .parse_next(input)
}

/// Parse one event of the ATEM block at position `block`.
///
/// Any structure other than the expected tags fails the block: it is
/// determinably malformed, and calls emitted before cannot be retracted.
fn parse_atem_event(input: &mut MuseInput<'_>, block: &mut AtemBlock) -> ModalResult<MuseEvent> {
    let block_close = literal(ATEM_CLOSE).value(MuseEvent::BlockClose);
    match block {
        AtemBlock::Start => atem_structure(preceded(
            ws0,
            literal(ATEM_OPEN).value(MuseEvent::BlockOpen),
        ))
        .parse_next(input),
        AtemBlock::Invokes { any: false } => {
            atem_structure(preceded(ws0, invoke_open_event)).parse_next(input)
        }
        AtemBlock::Invokes { any: true } => {
            atem_structure(preceded(ws0, alt((invoke_open_event, block_close)))).parse_next(input)
        }
        AtemBlock::Invoke { body, .. } => match body {
            InvokeBody::Parameters => atem_structure(preceded(
                ws0,
                alt((
                    parameter_open_event,
                    literal(INVOKE_CLOSE).value(MuseEvent::InvokeClose),
                )),
            ))
            .parse_next(input),
            InvokeBody::Parameter => alt((
                literal(PARAMETER_CLOSE).value(MuseEvent::ParameterClose),
                unclosed_value_invoke_close.value(MuseEvent::InvokeClose),
                |input: &mut MuseInput<'_>| {
                    safe_text_len_mul(input, VALUE_END_MARKERS).map(|_| MuseEvent::ParameterText)
                },
            ))
            .parse_next(input),
            InvokeBody::BufferedParameter { value_end_scan, .. } => {
                let (value, closes_invoke) = seq!(
                    take_until_marker_mul(VALUE_END_MARKERS, value_end_scan),
                    alt((
                        literal(PARAMETER_CLOSE).value(false),
                        unclosed_value_invoke_close.value(true),
                    )),
                )
                .parse_next(input)?;
                Ok(MuseEvent::BufferedValue {
                    value: value.to_string(),
                    closes_invoke,
                })
            }
        },
    }
}

/// Fail a mismatched ATEM tag as a malformed block, while still waiting on
/// input that may yet complete it.
fn atem_structure<'i>(
    mut parser: impl Parser<MuseInput<'i>, MuseEvent, ErrMode<ContextError>>,
) -> impl Parser<MuseInput<'i>, MuseEvent, ErrMode<ContextError>> {
    move |input: &mut MuseInput<'i>| {
        parser.parse_next(input).map_err(|error| match error {
            ErrMode::Backtrack(_) => muse_error("Muse ATEM block"),
            error => error,
        })
    }
}

/// Parse the `</atem:invoke>` that ends a last parameter value missing its
/// `</atem:parameter>`, with the one newline before it that the value drops.
fn unclosed_value_invoke_close(input: &mut MuseInput<'_>) -> ModalResult<()> {
    (opt(literal("\n")), literal(INVOKE_CLOSE)).void().parse_next(input)
}

/// Parse an `<atem:invoke>` open tag into its tool name.
fn invoke_open_event(input: &mut MuseInput<'_>) -> ModalResult<MuseEvent> {
    let (name,) = seq!(
        _: (literal(INVOKE_OPEN), ws1, literal("name"), literal("=")),
        atem_attr_value,
        _: (take_until(0.., ">"), literal(">")),
    )
    .parse_next(input)?;

    Ok(MuseEvent::InvokeOpen { name })
}

/// Parse an `<atem:parameter>` open tag into its key.
fn parameter_open_event(input: &mut MuseInput<'_>) -> ModalResult<MuseEvent> {
    let (key,) = seq!(
        _: (literal(PARAMETER_OPEN), ws1, literal("name"), literal("=")),
        atem_attr_value,
        _: (take_until(0.., ">"), literal(">")),
    )
    .parse_next(input)?;

    Ok(MuseEvent::ParameterOpen { key })
}

/// Ignore everything after the assistant turn ended.
fn parse_done_event(input: &mut MuseInput<'_>) -> ModalResult<MuseEvent> {
    rest.value(MuseEvent::Skip).parse_next(input)
}

/// Parse a complete segment routing header into its recipient name, or `None`
/// for a header without recipient.
///
/// Accepted shapes: ` to=NAME` (generation-prompt continuation) and
/// `<|start|>assistant to=NAME` (full segment start), each optionally without
/// the recipient.
fn parse_segment_recipient(header: &str) -> ModalResult<Option<String>> {
    let header = header.strip_prefix(ASSISTANT_START).unwrap_or(header).trim();
    if header.is_empty() {
        return Ok(None);
    }

    let mut input = header;
    let parsed: ModalResult<(&str,)> = seq!(
        _: literal(RECIPIENT_PREFIX),
        take_while(1.., is_recipient_char),
        _: eof,
    )
    .parse_next(&mut input);
    let (recipient,) = parsed.map_err(|_| muse_error("Muse segment header"))?;

    Ok(Some(recipient.to_string()))
}

/// One recipient name character, per the Python parser's `[A-Za-z0-9_.-]+`.
fn is_recipient_char(char: char) -> bool {
    char.is_ascii_alphanumeric() || matches!(char, '_' | '.' | '-')
}

/// Parse one ATEM attribute value, double/single-quoted or bare.
fn atem_attr_value(input: &mut MuseInput<'_>) -> ModalResult<String> {
    alt((
        delimited(literal("\""), take_till(0.., '"'), literal("\"")),
        delimited(literal("'"), take_till(0.., '\''), literal("'")),
        take_while(1.., is_recipient_char),
    ))
    .map(str::to_string)
    .parse_next(input)
}

/// Decode one ATEM parameter value: JSON when it parses, raw string otherwise.
///
/// The raw value is used verbatim — whitespace inside string values is
/// significant ("spaces for string values are not stripped").
fn decode_parameter_value(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Decode a parameter value, keeping the raw text when its schema admits a
/// string unless the value decodes to another admitted type (`null` for
/// `Optional[str]`, `3` for `anyOf: [integer, string]`).
fn convert_parameter_value(parameters: &Value, key: &str, raw: String) -> Value {
    let decoded = decode_parameter_value(&raw);
    let types = parameter_types(parameters, key);
    match JsonType::of_value(&decoded) {
        _ if !types.contains(&JsonType::String) => decoded,
        JsonType::Integer if types.contains(&JsonType::Number) => decoded,
        ty if ty != JsonType::String && types.contains(&ty) => decoded,
        _ => Value::String(raw),
    }
}

/// Whether the schema of parameter `key` admits only a string, so that
/// [`convert_parameter_value`] keeps its raw text whatever it spells and the
/// value can stream before it is complete.
fn admits_only_string(parameters: &Value, key: &str) -> bool {
    let types = parameter_types(parameters, key);
    !types.is_empty() && types.iter().all(|ty| *ty == JsonType::String)
}

/// Types the schema of parameter `key` admits; empty when unconstrained.
fn parameter_types(parameters: &Value, key: &str) -> Vec<JsonType> {
    let root = SchemaRoot::new(parameters);
    root.resolved()
        .get("properties")
        .and_then(|properties| properties.get(key))
        .or_else(|| root.resolved().get("additionalProperties"))
        .and_then(|schema| schema_types(root, schema))
        .unwrap_or_default()
}

/// Types `schema` admits (none for a non-`string` type name), or `None` if unconstrained.
fn schema_types<'a>(root: SchemaRoot<'a>, schema: &'a Value) -> Option<Vec<JsonType>> {
    // Only the exact type name `string` keeps the raw text, as before.
    if let Some(name) = schema.get("type").and_then(Value::as_str) {
        let string = (name == "string").then_some(JsonType::String);
        return Some(string.into_iter().collect());
    }
    // `options` admits any value under a multi-schema `allOf`; intersect the
    // constrained branches so `[{$ref}, {description}]` keeps the referenced type.
    if let Some(schemas @ [_, _, ..]) =
        schema.get("allOf").and_then(Value::as_array).map(Vec::as_slice)
    {
        return schemas
            .iter()
            .filter_map(|schema| schema_types(root, schema))
            .map(|mut types| {
                // A `number` branch also admits integers.
                if types.contains(&JsonType::Number) {
                    types.push(JsonType::Integer);
                }
                types
            })
            .reduce(|both, types| both.into_iter().filter(|ty| types.contains(ty)).collect());
    }
    // An untyped option leaves the value unconstrained; `nullable` adds a null one.
    root.options(schema)
        .into_iter()
        .map(|option| match option.source {
            OptionSource::Any if option.ty != JsonType::Null => None,
            _ => Some(option.ty),
        })
        .collect()
}

/// Build a cut error for determinably malformed Muse structure.
fn muse_error(label: &'static str) -> ErrMode<ContextError> {
    let mut error = ContextError::new();
    error.push(StrContext::Label(label));
    ErrMode::Cut(error)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use expect_test::expect;
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;
    use vllm_tokenizer::{DecodedText, test_utils::TestTokenizer};

    use super::{EOM, EOT, MESSAGE, MuseUnifiedParser, START, convert_parameter_value};
    use crate::tool::{Tool, ToolCallDelta};
    use crate::unified::{
        UnifiedParser, UnifiedParserError, UnifiedParserEvent, UnifiedParserOutput,
    };

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_special_token(START, 200022)
            .with_special_token(MESSAGE, 200023)
            .with_special_token(EOM, 200007)
            .with_special_token(EOT, 200008)
    }

    trait UnifiedParserTestExt {
        fn parse_chunk(&mut self, chunk: &str) -> super::Result<UnifiedParserOutput>;
        fn parse_complete(&mut self, text: &str) -> super::Result<UnifiedParserOutput>;
    }

    impl UnifiedParserTestExt for MuseUnifiedParser {
        fn parse_chunk(&mut self, chunk: &str) -> super::Result<UnifiedParserOutput> {
            let mut output = UnifiedParserOutput::default();
            self.parse_into(DecodedText::unattributed(chunk), &mut output)?;
            Ok(output)
        }

        fn parse_complete(&mut self, text: &str) -> super::Result<UnifiedParserOutput> {
            let mut output = self.parse_chunk(text)?;
            output.append(self.finish()?);
            Ok(output)
        }
    }

    trait UnifiedOutputTestExt {
        fn normal_text(&self) -> String;
        fn reasoning_text(&self) -> String;
        fn calls(&self) -> Vec<ToolCallDelta>;
        fn merged_events(&self) -> Vec<UnifiedParserEvent>;
    }

    impl UnifiedOutputTestExt for UnifiedParserOutput {
        fn normal_text(&self) -> String {
            self.events
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect()
        }

        fn reasoning_text(&self) -> String {
            self.events
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::Reasoning(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect()
        }

        /// Tool calls with their argument deltas merged.
        fn calls(&self) -> Vec<ToolCallDelta> {
            let mut calls = Vec::<ToolCallDelta>::new();
            for event in &self.events {
                let UnifiedParserEvent::ToolCall(delta) = event else {
                    continue;
                };
                match calls.iter_mut().find(|call| call.tool_index == delta.tool_index) {
                    Some(call) => call.arguments.push_str(&delta.arguments),
                    None => calls.push(delta.clone()),
                }
            }
            calls
        }

        /// Events with each tool call's consecutive deltas merged, which
        /// depend on how the input was chunked.
        fn merged_events(&self) -> Vec<UnifiedParserEvent> {
            let mut events = Vec::<UnifiedParserEvent>::new();
            for event in &self.events {
                if let (
                    Some(UnifiedParserEvent::ToolCall(call)),
                    UnifiedParserEvent::ToolCall(delta),
                ) = (events.last_mut(), event)
                    && call.tool_index == delta.tool_index
                {
                    call.arguments.push_str(&delta.arguments);
                    continue;
                }
                events.push(event.clone());
            }
            events
        }
    }

    fn test_parser() -> MuseUnifiedParser {
        MuseUnifiedParser::new(&[], Arc::new(tokenizer())).unwrap()
    }

    fn collect_stream(parser: &mut MuseUnifiedParser, chunks: &[&str]) -> UnifiedParserOutput {
        let mut output = UnifiedParserOutput::default();
        for chunk in chunks {
            output.append(parser.parse_chunk(chunk).unwrap());
        }
        output.append(parser.finish().unwrap());
        output
    }

    /// Split `text` into small chunks to stress marker-split handling.
    fn char_chunks(text: &str, size: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        chars.chunks(size).map(|chunk| chunk.iter().collect()).collect()
    }

    /// Every two-way char-boundary split of `text`.
    fn two_way_splits(text: &str) -> Vec<(String, String)> {
        text.char_indices()
            .map(|(index, _)| index)
            .chain([text.len()])
            .map(|index| (text[..index].to_string(), text[index..].to_string()))
            .collect()
    }

    /// A `<|start|>assistant`-prefixed segment header followed by its body and
    /// the `<|eom|>` continuation boundary.
    fn segment(recipient: &str, body: &str) -> String {
        format!("{START}assistant to={recipient}{MESSAGE}{body}{EOM}")
    }

    /// One ATEM block with one invoke per (name, params) entry.
    fn atem(invokes: &[(&str, &[(&str, &str)])]) -> String {
        let mut block = String::from("<atem:function_calls>\n");
        for (name, params) in invokes {
            block.push_str(&format!("<atem:invoke name=\"{name}\">\n"));
            for (key, value) in *params {
                block.push_str(&format!(
                    "<atem:parameter name=\"{key}\">{value}</atem:parameter>\n"
                ));
            }
            block.push_str("</atem:invoke>\n");
        }
        block.push_str("</atem:function_calls>");
        block
    }

    /// A completion exercising every recipient kind and both boundary markers.
    fn full_completion() -> String {
        let commentary = segment("commentary", "working");
        let calls = atem(&[
            (
                "get_weather",
                &[("location", "Shanghai"), ("days", "1.5")][..],
            ),
            ("get_time", &[][..]),
        ]);
        format!(
            " to=self{MESSAGE}plan it{EOM}{commentary}\
             {START}assistant to=get_weather{MESSAGE}{calls}{EOM}\
             {START}assistant to=user{MESSAGE}done{EOT}"
        )
    }

    #[test]
    fn muse_create_requires_structural_tokens() {
        let error = match MuseUnifiedParser::new(&[], Arc::new(TestTokenizer::new())) {
            Ok(_) => panic!("expected missing token error"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            UnifiedParserError::MissingToken { token } if token == START
        ));
    }

    #[test]
    fn muse_preserves_special_tokens() {
        assert!(test_parser().preserve_special_tokens());
    }

    #[test]
    fn muse_parses_self_commentary_user_segments() {
        let text = format!(
            " to=self{MESSAGE}think step{EOM}{}\
             {START}assistant to=user{MESSAGE}final answer{EOT}",
            segment("commentary", "intermediate"),
        );

        let output = test_parser().parse_complete(&text).unwrap();

        assert_eq!(output.reasoning_text(), "think step");
        // Visible segments concatenate without any implicit separator.
        assert_eq!(output.normal_text(), "intermediatefinal answer");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn muse_tolerates_full_header_on_first_segment() {
        let text = format!("{START}assistant to=user{MESSAGE}answer{EOT}");

        let output = test_parser().parse_complete(&text).unwrap();

        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn muse_eot_ends_reasoning_segment() {
        let text = format!(" to=self{MESSAGE}only reasoning{EOT}");

        let output = test_parser().parse_complete(&text).unwrap();

        assert_eq!(output.reasoning_text(), "only reasoning");
        assert!(output.normal_text().is_empty());
    }

    #[test]
    fn muse_parses_tool_call_with_typed_and_string_parameters() {
        let text = format!(
            " to=get_weather{MESSAGE}{}{EOT}",
            atem(
                &[(
                    "get_weather",
                    &[
                        ("location", "Shanghai"),
                        ("days", "1.5"),
                        ("detailed", "true"),
                        ("filters", r#"{"kind":"rain"}"#),
                        ("hours", "[8,20]"),
                        ("note", "not json {"),
                    ][..],
                )][..]
            )
        );

        let output = test_parser().parse_complete(&text).unwrap();

        let calls = output.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_index, 0);
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].arguments).unwrap(),
            json!({
                "location": "Shanghai",
                "days": 1.5,
                "detailed": true,
                "filters": { "kind": "rain" },
                "hours": [8, 20],
                "note": "not json {",
            })
        );
    }

    #[test]
    fn muse_preserves_schema_string_parameters_in_batch_and_streaming() {
        let tools = [Tool {
            name: "echo".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "note": {"$ref": "#/$defs/Note"},
                    "count": {"type": "integer"},
                    "items": {"type": "array", "items": {"type": "integer"}}
                },
                "$defs": {"Note": {"type": "string"}}
            }),
            strict: None,
            defer_loading: None,
        }];
        for value in [
            "25",
            "true",
            "null",
            "[1,2]",
            r#"{"x":1}"#,
            "3%",
            "1,500",
            " 25 ",
        ] {
            let text = format!(
                " to=echo.echo{MESSAGE}{}{EOT}",
                atem(&[(
                    "echo.echo",
                    &[
                        ("text", value),
                        ("note", value),
                        ("count", "25"),
                        ("items", "[1,2]")
                    ]
                )])
            );
            let mut parser = MuseUnifiedParser::new(&tools, Arc::new(tokenizer())).unwrap();
            let batch = parser.parse_complete(&text).unwrap();
            let calls = batch.calls();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name.as_deref(), Some("echo"));
            assert_eq!(
                serde_json::from_str::<Value>(&calls[0].arguments).unwrap(),
                json!({"text": value, "note": value, "count": 25, "items": [1, 2]})
            );
            for size in [1, 17, text.len()] {
                parser.reset();
                let chunks = char_chunks(&text, size);
                let chunks: Vec<_> = chunks.iter().map(String::as_str).collect();
                assert_eq!(
                    collect_stream(&mut parser, &chunks).merged_events(),
                    batch.merged_events()
                );
            }
        }
    }

    #[test]
    fn muse_arguments_preserve_order_and_number_formatting() {
        let text = format!(
            " to=add{MESSAGE}{}{EOT}",
            atem(
                &[(
                    "add",
                    &[("y", "1.0"), ("x", "2"), ("items", r#"["left","right"]"#)][..],
                )][..]
            )
        );

        let output = test_parser().parse_complete(&text).unwrap();

        assert_eq!(
            output.calls()[0].arguments,
            r#"{"y": 1.0, "x": 2, "items": ["left", "right"]}"#
        );
    }

    #[test]
    fn muse_parses_multiple_invokes_in_one_block() {
        let text = format!(
            " to=multi{MESSAGE}{}{EOT}",
            atem(&[
                ("first", &[("a", "1")][..]),
                ("second", &[][..]),
                ("third", &[("b", "x")][..]),
            ])
        );

        let output = test_parser().parse_complete(&text).unwrap();

        let calls = output.calls();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].tool_index, 0);
        assert_eq!(calls[0].name.as_deref(), Some("first"));
        assert_eq!(calls[0].arguments, r#"{"a": 1}"#);
        assert_eq!(calls[1].tool_index, 1);
        assert_eq!(calls[1].name.as_deref(), Some("second"));
        assert_eq!(calls[1].arguments, "{}");
        assert_eq!(calls[2].tool_index, 2);
        assert_eq!(calls[2].name.as_deref(), Some("third"));
    }

    fn shell_tool() -> Tool {
        Tool {
            name: "shell".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "note": {"type": ["string", "null"]}
                }
            }),
            strict: None,
            defer_loading: None,
        }
    }

    #[test]
    fn muse_tool_call_streams_before_close_tag() {
        let mut parser = MuseUnifiedParser::new(&[shell_tool()], Arc::new(tokenizer())).unwrap();
        let mut output = UnifiedParserOutput::default();

        // The call and its arguments so far, after each chunk.
        let mut progress = Vec::new();
        for chunk in [
            &format!(" to=shell{MESSAGE}<atem:function_calls>\n"),
            "<atem:invoke name=\"shell\">\n",
            "<atem:parameter name=\"command\">ls \"a",
            " b\"</atem:para",
            "meter>\n<atem:parameter name=\"timeout\">3",
            "0</atem:parameter>\n",
            "<atem:parameter name=\"note\">null</atem:parameter>\n",
            "</atem:invoke>\n",
            "</atem:function_calls>",
        ] {
            output.append(parser.parse_chunk(chunk).unwrap());
            progress.push(match output.calls().first() {
                Some(call) => format!("{}({})", call.name.as_deref().unwrap(), call.arguments),
                None => "no call".to_string(),
            });
        }

        // A string-only parameter streams; `timeout` and the nullable `note`
        // are decoded once complete.
        expect![[r#"
            no call
            shell()
            shell({"command": "ls \"a)
            shell({"command": "ls \"a b\")
            shell({"command": "ls \"a b\"")
            shell({"command": "ls \"a b\"", "timeout": 30)
            shell({"command": "ls \"a b\"", "timeout": 30, "note": null)
            shell({"command": "ls \"a b\"", "timeout": 30, "note": null})
            shell({"command": "ls \"a b\"", "timeout": 30, "note": null})"#]]
        .assert_eq(&progress.join("\n"));
    }

    #[test]
    fn muse_string_parameter_keeps_atem_tags_in_its_value() {
        // Only `</atem:parameter>` and `</atem:invoke>` end a value, so a string
        // value may spell the other ATEM tags, including the block close.
        let value = "<atem:invoke name=\"x\"><atem:parameter></atem:function_calls>";
        let text = format!(
            " to=shell{MESSAGE}{}{EOT}",
            atem(&[("shell", &[("command", value)][..])])
        );
        let mut parser = MuseUnifiedParser::new(&[shell_tool()], Arc::new(tokenizer())).unwrap();

        let output = parser.parse_complete(&text).unwrap();

        let calls = output.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].arguments).unwrap(),
            json!({ "command": value })
        );
    }

    #[test]
    fn muse_unclosed_last_parameter_ends_at_invoke_close() {
        let text = format!(
            " to=cite{MESSAGE}<atem:function_calls>\n<atem:invoke name=\"cite\">\n\
             <atem:parameter name=\"chunk_ids\">[\"07657059-389d-4601\"]</atem:invoke>\n\
             </atem:function_calls>{EOT}"
        );

        let output = test_parser().parse_complete(&text).unwrap();

        let calls = output.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].arguments).unwrap(),
            json!({ "chunk_ids": ["07657059-389d-4601"] })
        );
    }

    #[test]
    fn muse_unclosed_last_parameter_after_closed_one() {
        let text = format!(
            " to=search{MESSAGE}<atem:function_calls>\n<atem:invoke name=\"search\">\n\
             <atem:parameter name=\"query\">foosball bracket</atem:parameter>\n\
             <atem:parameter name=\"limit\">10\n\
             </atem:invoke>\n</atem:function_calls>{EOT}"
        );

        let output = test_parser().parse_complete(&text).unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "query": "foosball bracket", "limit": 10 })
        );
    }

    #[test]
    fn muse_unclosed_last_parameter_stays_inside_its_invoke() {
        let text = format!(
            " to=cite{MESSAGE}<atem:function_calls>\n<atem:invoke name=\"cite\">\n\
             <atem:parameter name=\"chunk_ids\">[\"07657059-389d-4601\"]</atem:invoke>\n\
             <atem:invoke name=\"search\">\n\
             <atem:parameter name=\"query\">foosball bracket</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls>{EOT}"
        );

        let output = test_parser().parse_complete(&text).unwrap();

        let calls = output.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].arguments).unwrap(),
            json!({ "chunk_ids": ["07657059-389d-4601"] })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&calls[1].arguments).unwrap(),
            json!({ "query": "foosball bracket" })
        );
    }

    #[test]
    fn muse_unclosed_last_string_parameter_drops_separator_newline() {
        let text = format!(
            " to=search{MESSAGE}<atem:function_calls>\n<atem:invoke name=\"search\">\n\
             <atem:parameter name=\"query\">foosball bracket\n\
             </atem:invoke>\n</atem:function_calls>{EOT}"
        );

        let output = test_parser().parse_complete(&text).unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "query": "foosball bracket" })
        );
    }

    #[test]
    fn muse_unclosed_last_parameter_streams_and_keeps_later_segments() {
        // The streamed string value holds back its trailing newline until the
        // invoke close shows it is the separator, and the turn goes on after
        // the block.
        let text = format!(
            " to=shell{MESSAGE}<atem:function_calls>\n<atem:invoke name=\"shell\">\n\
             <atem:parameter name=\"command\">ls\n\
             </atem:invoke>\n<atem:invoke name=\"shell\">\n\
             <atem:parameter name=\"timeout\">5</atem:invoke>\n\
             </atem:function_calls>{EOM}{START}assistant to=user{MESSAGE}Final answer.{EOT}"
        );
        let parser = || MuseUnifiedParser::new(&[shell_tool()], Arc::new(tokenizer())).unwrap();
        let batch = parser().parse_complete(&text).unwrap();

        let calls = batch.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, r#"{"command": "ls"}"#);
        assert_eq!(calls[1].arguments, r#"{"timeout": 5}"#);
        assert_eq!(batch.normal_text(), "Final answer.");
        for (first, second) in two_way_splits(&text) {
            let output = collect_stream(&mut parser(), &[&first, &second]);
            assert_eq!(
                output.merged_events(),
                batch.merged_events(),
                "split: {first:?} + {second:?}"
            );
        }
    }

    #[test]
    fn muse_repeated_parameter_keeps_first_value() {
        // The first value may already be on the wire, so a later one is dropped.
        let text = format!(
            " to=shell{MESSAGE}{}{EOT}",
            atem(&[(
                "shell",
                &[
                    ("command", "ls"),
                    ("timeout", "1"),
                    ("command", "rm"),
                    ("timeout", "2")
                ][..]
            )])
        );
        let mut parser = MuseUnifiedParser::new(&[shell_tool()], Arc::new(tokenizer())).unwrap();

        let output = parser.parse_complete(&text).unwrap();

        assert_eq!(
            output.calls()[0].arguments,
            r#"{"command": "ls", "timeout": 1}"#
        );
    }

    #[test]
    fn muse_streaming_tool_calls_match_batch_for_every_marker_split() {
        let text = format!(
            " to=shell{MESSAGE}{}{EOM}{START}assistant to=user{MESSAGE}done{EOT}",
            atem(&[
                (
                    "shell",
                    &[("command", "echo \"</atem:par\" \\ é"), ("timeout", "5")][..]
                ),
                ("shell", &[][..]),
            ])
        );
        let parser = || MuseUnifiedParser::new(&[shell_tool()], Arc::new(tokenizer())).unwrap();
        let batch = parser().parse_complete(&text).unwrap();
        assert_eq!(batch.calls().len(), 2);

        for (first, second) in two_way_splits(&text) {
            let output = collect_stream(&mut parser(), &[&first, &second]);
            assert_eq!(
                output.merged_events(),
                batch.merged_events(),
                "split: {first:?} + {second:?}"
            );
        }
    }

    #[test]
    fn muse_atem_block_without_invoke_fails() {
        let mut parser = test_parser();
        let error = parser
            .parse_chunk(&format!(
                " to=calc{MESSAGE}<atem:function_calls>\n</atem:function_calls>"
            ))
            .unwrap_err();

        assert!(error.to_report_string().contains("Muse ATEM block"));
    }

    #[test]
    fn muse_parses_consecutive_tool_segments_across_eom() {
        let text = format!(
            "{}{START}assistant to=add{MESSAGE}{}{EOT}",
            segment(
                "get_weather",
                &atem(&[("get_weather", &[("location", "SF")][..])])
            ),
            atem(&[("add", &[("x", "2")][..])]),
        );

        let output = test_parser().parse_complete(&text).unwrap();

        let calls = output.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool_index, 0);
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(calls[1].tool_index, 1);
        assert_eq!(calls[1].name.as_deref(), Some("add"));
    }

    #[test]
    fn muse_streaming_matches_batch_for_every_marker_split() {
        assert_streaming_matches_batch(&full_completion());
    }

    /// Stream `text` under every two-way split and several fixed chunk sizes,
    /// asserting each run emits the same events as one batch parse.
    fn assert_streaming_matches_batch(text: &str) {
        let batch = test_parser().parse_complete(text).unwrap();

        // Exhaustive two-way splits cover every marker cut at every boundary.
        for (first, second) in two_way_splits(text) {
            let output = collect_stream(&mut test_parser(), &[&first, &second]);
            assert_eq!(
                output.merged_events(),
                batch.merged_events(),
                "split: {first:?} + {second:?}"
            );
        }

        // Small fixed-size chunks split longer markers across 3+ chunks.
        for size in 1..=9 {
            let chunks = char_chunks(text, size);
            let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let output = collect_stream(&mut test_parser(), &chunk_refs);
            assert_eq!(
                output.merged_events(),
                batch.merged_events(),
                "chunk size {size}"
            );
        }
    }

    #[test]
    fn muse_streaming_emits_text_incrementally() {
        let mut parser = test_parser();
        let mut output = parser.parse_chunk(" to=user").unwrap();
        assert!(output.events.is_empty());

        output.append(parser.parse_chunk(&format!("{MESSAGE}Hel")).unwrap());
        assert_eq!(output.normal_text(), "Hel");

        output.append(parser.parse_chunk(&format!("lo{EOT}")).unwrap());
        assert_eq!(output.normal_text(), "Hello");
    }

    #[test]
    fn muse_ignores_output_after_turn_end() {
        let text = format!(" to=user{MESSAGE}answer{EOT}junk{EOM}more");

        let output = test_parser().parse_complete(&text).unwrap();

        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn muse_bare_eot_between_segments_ends_turn() {
        let text = format!("{}{EOT}", segment("commentary", "note"));

        let output = test_parser().parse_complete(&text).unwrap();

        assert_eq!(output.normal_text(), "note");
    }

    #[test]
    fn muse_finish_flushes_unclosed_reasoning() {
        let mut parser = test_parser();
        let mut output = parser.parse_chunk(&format!(" to=self{MESSAGE}still thinking")).unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(output.reasoning_text(), "still thinking");
        assert!(output.normal_text().is_empty());
    }

    #[test]
    fn muse_finish_flushes_unclosed_text() {
        let mut parser = test_parser();
        let mut output = parser.parse_chunk(&format!(" to=user{MESSAGE}partial answer")).unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(output.normal_text(), "partial answer");
    }

    #[test]
    fn muse_finish_flushes_marker_free_text_as_text() {
        let output = test_parser().parse_complete("plain answer").unwrap();

        assert_eq!(output.normal_text(), "plain answer");
    }

    #[test]
    fn muse_finish_after_closed_tool_block_keeps_calls() {
        // The terminal `<|eot|>` may already be stripped by the serving layer
        // (e.g. stop-token handling); a complete ATEM block still commits.
        let mut parser = test_parser();
        let mut output = parser
            .parse_chunk(&format!(
                " to=calc{MESSAGE}{}",
                atem(&[("calc", &[("x", "42")][..])])
            ))
            .unwrap();
        output.append(parser.finish().unwrap());

        let calls = output.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_deref(), Some("calc"));
        assert_eq!(calls[0].arguments, r#"{"x": 42}"#);
    }

    #[test]
    fn muse_finish_fails_mid_tool_block() {
        let mut parser = test_parser();
        parser
            .parse_chunk(&format!(
                " to=calc{MESSAGE}<atem:function_calls>\n<atem:invoke"
            ))
            .unwrap();

        let error = parser.finish().unwrap_err();

        assert!(error.to_report_string().contains("incomplete Muse tool call"));
    }

    #[test]
    fn muse_malformed_segment_header_fails() {
        let mut parser = test_parser();
        let error = parser.parse_chunk(&format!(" junk header{MESSAGE}body")).unwrap_err();

        assert!(error.to_report_string().contains("Muse segment header"));
    }

    #[test]
    fn muse_malformed_atem_block_fails() {
        let mut parser = test_parser();
        let error = parser
            .parse_chunk(&format!(
                " to=calc{MESSAGE}<atem:function_calls>garbage</atem:function_calls>"
            ))
            .unwrap_err();

        assert!(error.to_report_string().contains("Muse ATEM block"));
    }

    #[test]
    fn muse_reset_returns_buffered_text() {
        let mut parser = test_parser();
        let mut output = parser.parse_chunk(&format!(" to=user{MESSAGE}answer<|eo")).unwrap();
        assert_eq!(output.normal_text(), "answer");

        let raw = parser.reset();

        assert_eq!(raw, "<|eo");

        output = parser.parse_chunk(&format!(" to=user{MESSAGE}again")).unwrap();
        output.append(parser.finish().unwrap());
        assert_eq!(output.normal_text(), "again");
    }

    #[test]
    fn muse_bare_tool_header_closes_reasoning_without_eom() {
        // The model may skip `<|eom|>` and write the tool header right after
        // its reasoning, e.g. for a tool call with empty arguments.
        let text = format!(
            " to=self{MESSAGE}I should check the time. to=get_time{MESSAGE}{}{EOM}\
             {START}assistant to=user{MESSAGE}It is noon.{EOT}",
            atem(&[("get_time", &[][..])]),
        );

        let output = test_parser().parse_complete(&text).unwrap();

        assert_eq!(output.reasoning_text(), "I should check the time. ");
        let calls = output.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_deref(), Some("get_time"));
        assert_eq!(calls[0].arguments, "{}");
        assert_eq!(output.normal_text(), "It is noon.");
        assert_streaming_matches_batch(&text);
    }

    #[test]
    fn muse_full_header_closes_segments_without_eom() {
        let text = format!(
            " to=self{MESSAGE}plan{START}assistant to=calc{MESSAGE}{}\
             {START}assistant to=user{MESSAGE}done{EOT}",
            atem(&[("calc", &[("x", "1")][..])]),
        );

        let output = test_parser().parse_complete(&text).unwrap();

        assert_eq!(output.reasoning_text(), "plan");
        assert_eq!(output.calls()[0].name.as_deref(), Some("calc"));
        assert_eq!(output.normal_text(), "done");
        assert_streaming_matches_batch(&text);
    }

    #[test]
    fn muse_recipient_mentions_in_body_stay_text() {
        let text = format!(
            " to=self{MESSAGE}send it to=user later, too{EOM}\
             {START}assistant to=user{MESSAGE}auto to=x{EOT}"
        );

        let output = test_parser().parse_complete(&text).unwrap();

        assert_eq!(output.reasoning_text(), "send it to=user later, too");
        assert_eq!(output.normal_text(), "auto to=x");
        assert_streaming_matches_batch(&text);
    }

    #[test]
    fn muse_recipient_less_segment_is_reasoning() {
        let text = format!(
            "{MESSAGE}thinking{EOM}{START}assistant to=user{MESSAGE}answer{MESSAGE}more{EOT}"
        );

        let output = test_parser().parse_complete(&text).unwrap();

        assert_eq!(output.reasoning_text(), "thinkingmore");
        assert_eq!(output.normal_text(), "answer");
        assert_streaming_matches_batch(&text);
    }

    #[test]
    fn muse_mixed_channels_match_python_routing() {
        let text = concat!(
            "<|message|>Think.<|eom|>",
            "<|start|>assistant to=commentary<|message|>Checking. ",
            "to=developer<|message|>internal<|eom|>",
            "<|start|>assistant to=weather.get<|message|>",
            "<atem:function_calls><atem:invoke name=\"weather.get\">",
            "<atem:parameter name=\"city\">Paris</atem:parameter>",
            "</atem:invoke></atem:function_calls><|eom|>",
            "<|start|>assistant to=self<|message|>Recheck.<|eom|>",
            "<|start|>assistant to=user<|message|>Done.<|eot|>",
        );
        let output = test_parser().parse_complete(text).unwrap();
        expect_test::expect![[r#"
            [
                Reasoning(
                    DecodedText {
                        text: "Think.",
                        attributions: [],
                    },
                ),
                Text(
                    "Checking. ",
                ),
                ToolCall(
                    ToolCallDelta {
                        tool_index: 0,
                        name: Some(
                            "weather.get",
                        ),
                        arguments: "{\"city\": \"Paris\"}",
                    },
                ),
                Reasoning(
                    DecodedText {
                        text: "Recheck.",
                        attributions: [],
                    },
                ),
                Text(
                    "Done.",
                ),
            ]
        "#]]
        .assert_debug_eq(&output.merged_events());
        assert_streaming_matches_batch(text);
    }

    #[test]
    fn muse_drops_developer_segment() {
        let text = format!(
            " to=developer{MESSAGE}not for the user{EOM}\
             {START}assistant to=user{MESSAGE}answer{EOT}"
        );

        let output = test_parser().parse_complete(&text).unwrap();

        assert!(output.reasoning_text().is_empty());
        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn muse_collapses_doubled_namespace_tool_name() {
        let tools = [
            "get_weather",
            "calendar.get",
            "Bash",
            "Read",
            "Glob",
            "forecast",
            "forecast.daily",
            "Edit",
            "Edit.Edit",
            "",
        ]
        .map(|name| Tool {
            name: name.to_string(),
            description: None,
            parameters: json!({"type": "object"}),
            strict: None,
            defer_loading: None,
        });
        let mut parser = MuseUnifiedParser::new(&tools, Arc::new(tokenizer())).unwrap();
        let emitted = [
            "get_weather.get_weather",
            "calendar.get",
            "weather.get",
            "calendar.calendar",
            // A bare tool with any other suffix collapses too...
            "Bash.command",
            "Read.read",
            "Bash.execute",
            "Glob.",
            // ...unless a registered sibling may be what the suffix names.
            "forecast.weekly",
            "forecast.forecast",
            "forecast.daily",
            // An exact match wins over collapsing the doubled form.
            "Edit.Edit",
            ".x",
        ];
        let invokes: Vec<_> = emitted.iter().map(|name| (*name, &[][..])).collect();
        let text = format!(" to=get_weather.get_weather{MESSAGE}{}", atem(&invokes));

        let output = parser.parse_complete(&text).unwrap();

        let names: Vec<_> = output.calls().into_iter().filter_map(|call| call.name).collect();
        assert_eq!(
            names,
            [
                "get_weather",
                "calendar.get",
                "weather.get",
                "calendar.calendar",
                "Bash",
                "Read",
                "Bash",
                "Glob",
                "forecast.weekly",
                "forecast",
                "forecast.daily",
                "Edit.Edit",
                ".x",
            ]
        );
    }

    #[test]
    fn muse_collapsed_tool_name_uses_its_parameter_types() {
        let tools = [Tool {
            name: "Bash".to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": {
                "command": {"type": "string"}, "n": {"type": "integer"}}}),
            strict: None,
            defer_loading: None,
        }];
        let mut parser = MuseUnifiedParser::new(&tools, Arc::new(tokenizer())).unwrap();
        let text = format!(
            " to=Bash.command{MESSAGE}{}",
            atem(&[("Bash.command", &[("command", "007"), ("n", "7")][..])])
        );

        let calls = parser.parse_complete(&text).unwrap().calls();

        assert_eq!(calls[0].name.as_deref(), Some("Bash"));
        assert_eq!(calls[0].arguments, r#"{"command": "007", "n": 7}"#);
    }

    #[test]
    fn muse_parameter_schema_types() {
        let nullable = json!({"anyOf": [{"type": "string"}, {"type": "null"}]});
        let body = "{\n  \"v\": \"1.4.0\"\n}";
        let cases = json!([
            // schemas admitting a string keep the raw text...
            [{"type": "string"}, "1.10", "1.10"],
            [nullable, "1984", "1984"],
            [nullable, body, body],
            [nullable, r#""q""#, r#""q""#],
            [{"oneOf": [{"type": "string"}, {"type": "null"}]}, "true", "true"],
            [{"type": ["string", "null"]}, "1984", "1984"],
            [{"enum": ["1.10", "2.0"]}, "1.10", "1.10"],
            [{"const": "1.10"}, "1.10", "1.10"],
            [{"$ref": "#/$defs/S"}, "1984", "1984"],
            [{"allOf": [{"$ref": "#/$defs/S"}]}, "1.10", "1.10"],
            [{"allOf": [{"$ref": "#/$defs/S"}, {"description": "d"}]}, "1.10", "1.10"],
            [{"allOf": [nullable, {"type": "string"}]}, "null", "null"],
            [{"anyOf": [{"type": "integer"}, {"type": "string"}]}, "1.10", "1.10"],
            [{"type": ["number", "string"]}, "true", "true"],
            // ...unless the value decodes to another admitted type
            [nullable, "null", null],
            [{"anyOf": [{"type": "integer"}, {"type": "string"}]}, "25", 25],
            [{"type": ["number", "string"]}, "25", 25],
            [{"enum": [1, "a"]}, "1", 1],
            [{"type": ["number", "string"], "enum": [1, "x"]}, "1", 1],
            [{"allOf": [{"type": ["number", "string"]}, {"enum": [1, "x"]}]}, "1", 1],
            [{"allOf": [{"type": ["integer", "string"]}, {"type": ["number", "string"]}]}, "5", 5],
            // no string admitted, or no type information: decode as before
            [{"type": "integer"}, "1.10", 1.1],
            [{"description": "untyped"}, "25", 25],
            [{"anyOf": [{"type": "string"}, {}]}, r#""q""#, "q"],
            [{"type": "integer", "enum": ["1"]}, "1", 1],
        ]);
        let defs = json!({"S": {"type": "string"}});
        for case in cases.as_array().unwrap() {
            let (schema, raw) = (&case[0], case[1].as_str().unwrap());
            let parameters =
                json!({"properties": {"v": schema}, "additionalProperties": schema, "$defs": defs});
            for key in ["v", "other"] {
                let value = convert_parameter_value(&parameters, key, raw.to_string());
                assert_eq!(value, case[2], "{schema} {key} {raw:?}");
            }
        }
    }
}
