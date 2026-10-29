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
//! or more `<atem:invoke>` calls. Values are JSON-decoded when possible and
//! kept as raw strings otherwise; parameters whose schema admits a string keep
//! their raw text unless it decodes to another admitted type. Each invoke's
//! arguments are reassembled as a compact JSON object preserving parameter
//! order.
//!
//! The model sometimes opens the next segment without closing the current one
//! with `<|eom|>`, e.g. writing `to=FUNCTION<|message|>` straight after its
//! reasoning when calling a tool with empty arguments. Since `<|start|>` and
//! `<|message|>` never occur inside a body, either one also ends the current
//! segment and starts the next.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};
use vllm_tokenizer::{DecodedText, DynTokenizer};
use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, eof, peek, preceded, repeat, seq};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_till, take_until, take_while};

use super::{Result, UnifiedParser, UnifiedParserOutput, token_id};
use crate::tool::{Tool, ToolCallDelta};
use crate::unified::parsing_failed;
use crate::utils::{MarkerScanState, parse_buffered_event, safe_text_len_mul, take_until_marker};

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

/// Recipient whose segments are dropped: `developer` is a prompt role, not a
/// valid assistant recipient.
const DEVELOPER_RECIPIENT: &str = "developer";

type MuseInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct MuseToolCall {
    name: String,
    parameters: Vec<(String, String)>,
}

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
    /// One complete ATEM block parsed into raw calls.
    ToolCalls {
        calls: Vec<MuseToolCall>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MuseMode {
    /// Awaiting the ` to=RECIPIENT<|message|>` routing header of the next
    /// segment; the full `<|start|>assistant to=…` form is also accepted.
    Header { scan: MarkerScanState },
    /// Inside a `to=self` segment (private reasoning), or a bare `<|message|>`
    /// segment without recipient.
    Reasoning,
    /// Inside a `to=commentary`/`to=user` segment (visible text).
    Text,
    /// Inside a tool segment, buffering one ATEM block through its close tag.
    Tool { scan: MarkerScanState },
    /// Dropping text until the next segment boundary: noise after a complete
    /// ATEM block, or the body of a `to=developer` segment.
    Skip,
    /// After `<|eot|>`: ignore the rest (EOS leakage guard).
    Done,
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
    /// JSON types admitted by each registered tool's parameters.
    parameter_types: HashMap<String, ParameterTypes>,
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
            parameter_types: tools
                .iter()
                .map(|tool| (tool.name.clone(), ParameterTypes::new(&tool.parameters)))
                .collect(),
            emitted_call_count: 0,
        })
    }

    fn apply_event(
        &mut self,
        event: MuseEvent,
        piece: DecodedText,
        output: &mut UnifiedParserOutput,
    ) {
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
                        scan: MarkerScanState::default(),
                    },
                };
            }
            MuseEvent::SegmentEnd => {
                self.mode = MuseMode::Header {
                    scan: MarkerScanState::default(),
                };
            }
            MuseEvent::TurnEnd => self.mode = MuseMode::Done,
            MuseEvent::ToolCalls { calls } => {
                for call in calls {
                    let name = normalize_tool_name(call.name, &self.tool_names);
                    let types = self.parameter_types.get(&name);
                    let arguments: Map<String, Value> = call
                        .parameters
                        .into_iter()
                        .map(|(key, raw)| {
                            let allowed = types.and_then(|types| types.get(&key));
                            let value = convert_parameter_value(raw, allowed);
                            (key, value)
                        })
                        .collect();
                    let arguments = Value::Object(arguments).to_string();
                    let tool_index = self.emitted_call_count;
                    self.emitted_call_count += 1;
                    output.push_call(ToolCallDelta {
                        tool_index,
                        name: Some(name),
                        arguments,
                    });
                }
                self.mode = MuseMode::Skip;
            }
        }
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
            self.apply_event(event, piece, output);
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
                return Err(parsing_failed!("incomplete Muse tool call"));
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
/// namespace recipient `"NAME.*"`, so the model may call it as `NAME.NAME`.
/// Collapse only that doubled form, and only when `NAME` is registered and the
/// emitted name is not: matching on the trailing segment alone could dispatch
/// `weather.get` to a registered `calendar.get`.
fn normalize_tool_name(name: String, tool_names: &HashSet<String>) -> String {
    if tool_names.contains(&name) {
        return name;
    }
    match name.split_once('.') {
        Some((head, tail)) if head == tail && tool_names.contains(head) => head.to_string(),
        _ => name,
    }
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
        MuseMode::Tool { scan } => tool_block_event(input, scan),
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

/// Parse a buffered ATEM block through `</atem:function_calls>` into completed calls.
// TODO: calls commit at block granularity; the canonical wire shape is one
// invoke per block (the template renders each tool call as its own segment),
// so this matches call-at-a-time emission. Revisit per-invoke emission only if
// multi-invoke blocks ever matter.
fn tool_block_event(
    input: &mut MuseInput<'_>,
    scan: &mut MarkerScanState,
) -> ModalResult<MuseEvent> {
    let (body,) = seq!(
        take_until_marker(ATEM_CLOSE, scan),
        _: literal(ATEM_CLOSE),
    )
    .parse_next(input)?;
    let calls = parse_atem_block(body)?;

    Ok(MuseEvent::ToolCalls { calls })
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

/// Parse one complete ATEM block body (up to `</atem:function_calls>`) into
/// raw calls, one per `<atem:invoke>`.
fn parse_atem_block(body: &str) -> ModalResult<Vec<MuseToolCall>> {
    let mut input = body;
    delimited(
        (ws0, literal(ATEM_OPEN)),
        repeat(1.., atem_invoke),
        (ws0, eof),
    )
    .parse_next(&mut input)
    .map_err(|_| muse_error("Muse ATEM block"))
}

/// Parse one `<atem:invoke>` element into its name and raw parameters.
fn atem_invoke(input: &mut &str) -> ModalResult<MuseToolCall> {
    let (name, parameters): (String, Vec<(String, String)>) = seq!(
        _: (ws0, literal(INVOKE_OPEN), ws1, literal("name"), literal("=")),
        atem_attr_value,
        _: (take_until(0.., ">"), literal(">")),
        repeat(0.., atem_parameter),
        _: (ws0, literal(INVOKE_CLOSE)),
    )
    .parse_next(input)?;

    Ok(MuseToolCall { name, parameters })
}

/// Parse one `<atem:parameter>` element into its key/value pair.
fn atem_parameter(input: &mut &str) -> ModalResult<(String, String)> {
    let (name, raw) = seq!(
        _: (ws0, literal(PARAMETER_OPEN), ws1, literal("name"), literal("=")),
        atem_attr_value,
        _: (take_until(0.., ">"), literal(">")),
        take_until(0.., PARAMETER_CLOSE),
        _: literal(PARAMETER_CLOSE),
    )
    .parse_next(input)?;

    Ok((name, raw.to_string()))
}

/// Parse one ATEM attribute value, double/single-quoted or bare.
fn atem_attr_value(input: &mut &str) -> ModalResult<String> {
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

/// JSON types a schema admits, or `None` if it admits any.
type AllowedTypes = Option<HashSet<String>>;

/// JSON types admitted by one tool's top-level parameters.
#[derive(Debug, Default)]
struct ParameterTypes {
    /// Types admitted by each parameter listed in `properties`.
    properties: HashMap<String, AllowedTypes>,
    /// Types admitted by any other parameter (`additionalProperties`).
    additional: AllowedTypes,
}

impl ParameterTypes {
    /// Resolve the types admitted by each parameter of a tool's `parameters`
    /// schema, sharing one `$ref` cache across the whole tool.
    fn new(parameters: &Value) -> Self {
        let mut refs = HashMap::new();
        let properties = parameters
            .get("properties")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .map(|(name, schema)| (name.clone(), schema_types(schema, parameters, &mut refs)))
            .collect();
        let additional = parameters
            .get("additionalProperties")
            .and_then(|schema| schema_types(schema, parameters, &mut refs));
        Self {
            properties,
            additional,
        }
    }

    /// Types admitted by parameter `key`, or `None` if unconstrained.
    fn get(&self, key: &str) -> Option<&HashSet<String>> {
        self.properties.get(key).unwrap_or(&self.additional).as_ref()
    }
}

/// JSON Schema type name of a decoded JSON value.
fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_f64() => "number",
        Value::Number(_) => "integer",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Resolve a local JSON pointer such as `#/$defs/Name` against `root`.
fn resolve_ref<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    reference.strip_prefix("#/")?.split('/').try_fold(root, |node, part| {
        node.as_object()?.get(&part.replace("~1", "/").replace("~0", "~"))
    })
}

/// JSON types a parameter schema admits, or `None` if it admits any.
///
/// Covers `type` (name or list), `const`, `enum`, local `$ref`,
/// `anyOf`/`oneOf` (union) and `allOf` (intersection), so that e.g.
/// pydantic's `Optional[str]` (`anyOf: [string, null]`) counts as a string.
/// `refs` memoizes each `$ref` (a ref still being resolved counts as
/// unconstrained), keeping the walk linear in the schema size even when
/// `$defs` reference each other many times or cyclically.
fn schema_types(
    schema: &Value,
    root: &Value,
    refs: &mut HashMap<String, AllowedTypes>,
) -> AllowedTypes {
    let schema = schema.as_object()?;
    let mut constraints = Vec::new();
    match schema.get("type") {
        Some(Value::String(declared)) => constraints.push(Some(HashSet::from([declared.clone()]))),
        Some(Value::Array(declared)) => constraints.push(Some(
            declared.iter().filter_map(Value::as_str).map(str::to_owned).collect(),
        )),
        _ => {}
    }
    if let Some(value) = schema.get("const") {
        constraints.push(Some(HashSet::from([json_type(value).to_owned()])));
    }
    if let Some(Value::Array(values)) = schema.get("enum") {
        constraints.push(Some(
            values.iter().map(|value| json_type(value).to_owned()).collect(),
        ));
    }
    if let Some(Value::String(reference)) = schema.get("$ref") {
        if !refs.contains_key(reference) {
            refs.insert(reference.clone(), None);
            let types =
                resolve_ref(root, reference).and_then(|target| schema_types(target, root, refs));
            refs.insert(reference.clone(), types);
        }
        constraints.push(refs[reference].clone());
    }
    for key in ["anyOf", "oneOf"] {
        let Some(Value::Array(branches)) = schema.get(key) else {
            continue;
        };
        if branches.is_empty() {
            continue;
        }
        // A branch without type information makes the whole union
        // unconstrained; the remaining branches are not walked.
        let union = branches.iter().try_fold(HashSet::new(), |mut union, branch| {
            union.extend(schema_types(branch, root, refs)?);
            Some(union)
        });
        if union.is_some() {
            constraints.push(union);
        }
    }
    if let Some(Value::Array(subschemas)) = schema.get("allOf") {
        constraints.extend(subschemas.iter().map(|subschema| schema_types(subschema, root, refs)));
    }
    // Intersect, treating "integer" as a subset of "number".
    constraints.into_iter().flatten().reduce(|allowed, types| {
        let mut both = &allowed & &types;
        if (allowed.contains("number") && types.contains("integer"))
            || (allowed.contains("integer") && types.contains("number"))
        {
            both.insert("integer".to_string());
        }
        both
    })
}

/// Decode a parameter value, keeping the raw text where a string is meant.
///
/// When the schema admits a string, the raw text is kept unless it decodes to
/// another type the schema also admits (`null` for `Optional[str]`, `3` for
/// `anyOf: [integer, string]`).
fn convert_parameter_value(raw: String, allowed: Option<&HashSet<String>>) -> Value {
    let decoded = decode_parameter_value(&raw);
    let Some(allowed) = allowed.filter(|allowed| allowed.contains("string")) else {
        return decoded;
    };
    let kind = match json_type(&decoded) {
        "integer" if allowed.contains("number") => "number",
        kind => kind,
    };
    if kind != "string" && allowed.contains(kind) {
        decoded
    } else {
        Value::String(raw)
    }
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

    use serde_json::{Map, Value, json};
    use thiserror_ext::AsReport;
    use vllm_tokenizer::{DecodedText, test_utils::TestTokenizer};

    use super::{EOM, EOT, MESSAGE, MuseUnifiedParser, START};
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

        fn calls(&self) -> Vec<ToolCallDelta> {
            self.events
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect()
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
                    "count": {"type": "integer"},
                    "items": {"type": "array", "items": {"type": "integer"}}
                }
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
                    &[("text", value), ("count", "25"), ("items", "[1,2]")]
                )])
            );
            let mut parser = MuseUnifiedParser::new(&tools, Arc::new(tokenizer())).unwrap();
            let batch = parser.parse_complete(&text).unwrap();
            let calls = batch.calls();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name.as_deref(), Some("echo"));
            assert_eq!(
                serde_json::from_str::<Value>(&calls[0].arguments).unwrap(),
                json!({"text": value, "count": 25, "items": [1, 2]})
            );
            for size in [1, 17, text.len()] {
                parser.reset();
                let chunks = char_chunks(&text, size);
                let chunks: Vec<_> = chunks.iter().map(String::as_str).collect();
                assert_eq!(collect_stream(&mut parser, &chunks), batch);
            }
        }
    }

    #[test]
    fn muse_arguments_preserve_order_and_compact_number_formatting() {
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
            r#"{"y":1.0,"x":2,"items":["left","right"]}"#
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
        assert_eq!(calls[0].arguments, r#"{"a":1}"#);
        assert_eq!(calls[1].tool_index, 1);
        assert_eq!(calls[1].name.as_deref(), Some("second"));
        assert_eq!(calls[1].arguments, "{}");
        assert_eq!(calls[2].tool_index, 2);
        assert_eq!(calls[2].name.as_deref(), Some("third"));
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
            assert_eq!(output.events, batch.events, "split: {first:?} + {second:?}");
        }

        // Small fixed-size chunks split longer markers across 3+ chunks.
        for size in 1..=9 {
            let chunks = char_chunks(text, size);
            let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let output = collect_stream(&mut test_parser(), &chunk_refs);
            assert_eq!(output.events, batch.events, "chunk size {size}");
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
        assert_eq!(calls[0].arguments, r#"{"x":42}"#);
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
            UnifiedParserOutput {
                events: [
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
                            arguments: "{\"city\":\"Paris\"}",
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
                ],
            }
        "#]]
        .assert_debug_eq(&output);
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
        let tools = ["get_weather", "calendar.get"].map(|name| Tool {
            name: name.to_string(),
            description: None,
            parameters: json!({"type": "object"}),
            strict: None,
            defer_loading: None,
        });
        let mut parser = MuseUnifiedParser::new(&tools, Arc::new(tokenizer())).unwrap();
        let text = format!(
            " to=get_weather.get_weather{MESSAGE}{}",
            atem(&[
                ("get_weather.get_weather", &[][..]),
                ("calendar.get", &[][..]),
                ("weather.get", &[][..]),
                ("calendar.calendar", &[][..]),
            ])
        );

        let output = parser.parse_complete(&text).unwrap();

        let names: Vec<_> = output.calls().into_iter().filter_map(|call| call.name).collect();
        assert_eq!(
            names,
            [
                "get_weather",
                "calendar.get",
                "weather.get",
                "calendar.calendar"
            ]
        );
    }

    /// Parse one `v` parameter of tool `f`, asserting streaming matches batch.
    fn parse_param(parameters: Value, value: &str) -> Value {
        let tools = [Tool {
            name: "f".to_string(),
            description: None,
            parameters,
            strict: None,
            defer_loading: None,
        }];
        let mut parser = MuseUnifiedParser::new(&tools, Arc::new(tokenizer())).unwrap();
        let text = format!(" to=f{MESSAGE}{}{EOT}", atem(&[("f", &[("v", value)][..])]));
        let batch = parser.parse_complete(&text).unwrap();
        for size in [1, 7, text.len()] {
            parser.reset();
            let chunks = char_chunks(&text, size);
            let chunks: Vec<_> = chunks.iter().map(String::as_str).collect();
            assert_eq!(collect_stream(&mut parser, &chunks), batch);
        }
        let calls = batch.calls();
        assert_eq!(calls.len(), 1);
        let mut arguments: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        arguments["v"].take()
    }

    #[test]
    fn muse_string_schema_forms_keep_raw_text() {
        let string_def = json!({"S": {"type": "string"}});
        let schemas = [
            json!({"properties": {"v": {"type": "string"}}}),
            json!({"properties": {"v": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}),
            json!({"properties": {"v": {"oneOf": [{"type": "string"}, {"type": "null"}]}}}),
            json!({"properties": {"v": {"type": ["string", "null"]}}}),
            json!({"properties": {"v": {"enum": ["1.10", "2.0"]}}}),
            json!({"properties": {"v": {"const": "1.10"}}}),
            json!({"properties": {"v": {"$ref": "#/$defs/S"}}, "$defs": string_def}),
            json!({"properties": {"v": {"allOf": [{"$ref": "#/$defs/S"}]}}, "$defs": string_def}),
            json!({"properties": {"v": {"allOf": [{"type": "string"}, {"minLength": 1}]}}}),
            json!({"properties": {}, "additionalProperties": {"type": "string"}}),
        ];
        let values = [
            "1.10",
            "1984",
            "true",
            r#""q""#,
            "{\n  \"version\": \"1.4.0\",\n  \"build\": 7\n}",
        ];
        for schema in schemas {
            for value in values {
                assert_eq!(
                    parse_param(schema.clone(), value),
                    json!(value),
                    "{schema} {value:?}"
                );
            }
        }
    }

    #[test]
    fn muse_string_union_schemas() {
        let cases = [
            // nullable string: null is still null, anything else stays text
            (
                json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
                "null",
                json!(null),
            ),
            (json!({"type": ["string", "null"]}), "null", json!(null)),
            // a string-or-X union keeps a value of the other admitted type
            (
                json!({"anyOf": [{"type": "integer"}, {"type": "string"}]}),
                "25",
                json!(25),
            ),
            (
                json!({"anyOf": [{"type": "integer"}, {"type": "string"}]}),
                "1.10",
                json!("1.10"),
            ),
            (json!({"type": ["number", "string"]}), "25", json!(25)),
            (json!({"enum": [1, "a"]}), "1", json!(1)),
            (
                json!({"type": ["number", "string"], "enum": [1, "x"]}),
                "1",
                json!(1),
            ),
            // no string admitted, or no type information: decode as before
            (
                json!({"anyOf": [{"type": "number"}, {"type": "null"}]}),
                "1.10",
                json!(1.1),
            ),
            (json!({"description": "untyped"}), "25", json!(25)),
            (json!({"anyOf": [{"type": "string"}, {}]}), "25", json!(25)),
        ];
        for (schema, value, expected) in cases {
            let parameters = json!({"properties": {"v": schema}});
            assert_eq!(
                parse_param(parameters, value),
                expected,
                "{schema} {value:?}"
            );
        }
    }

    #[test]
    fn muse_ref_cycles_and_shared_refs_resolve_quickly() {
        // $ref cycles terminate; shared $refs are walked once, not once per path.
        let cyclic = json!({
            "properties": {"v": {"$ref": "#/$defs/A"}},
            "$defs": {
                "A": {"anyOf": [{"$ref": "#/$defs/B"}, {"type": "string"}]},
                "B": {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}},
            },
        });
        assert_eq!(parse_param(cyclic, "1.10"), json!("1.10"));
        // Each def references the next one twice: 2**60 paths without memoization.
        let mut defs: Map<String, Value> = (0..60)
            .map(|i| {
                let next = json!({"$ref": format!("#/$defs/D{}", i + 1)});
                (format!("D{i}"), json!({"anyOf": [next, next]}))
            })
            .collect();
        defs.insert("D60".to_string(), json!({"type": "string"}));
        let chain = json!({"properties": {"v": {"$ref": "#/$defs/D0"}}, "$defs": defs});
        assert_eq!(parse_param(chain, "1.10"), json!("1.10"));
    }
}
