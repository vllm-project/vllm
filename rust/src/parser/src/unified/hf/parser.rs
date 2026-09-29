// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Streaming executor for a compiled [`ResponseTemplate`].
//!
//! Follows `ResponseParser` in `response_parser.py`: a flat region state
//! machine that, outside explicit regions, watches every explicit open plus the
//! implicit region's close, and inside an explicit region watches only that
//! region's close. Instead of region events and an aggregated dict, it emits
//! ordered unified parser events:
//!
//! - text and reasoning stream as they arrive, with `strip` applied so that the
//!   concatenated deltas equal Transformers' stripped value. For a field without
//!   `repeats`, stripping spans all its occurrences: an interrupted field (text,
//!   tool call, more text) streams every occurrence, where Transformers keeps only
//!   the last;
//! - a tool call starts at its opener when the opener captures the function
//!   name, and its arguments are emitted when the region closes.

use std::mem::take;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Map, Value};
use thiserror_ext::AsReport as _;
use vllm_tokenizer::{DecodedText, DynTokenizer, TokenAnchor, TokenAttribution};
use winnow::error::{ErrMode, ModalResult};
use winnow::stream::{Partial, Stream};

use super::pattern::{Captures, Resolution};
use super::template::{
    FieldName, Region, RegionKind, Repeat, ResponseTemplate, TextRegion, TextRole, ToolCallRegion,
    Watch, WatchKind, WatchSet,
};
use crate::tool::{Tool, ToolCallDelta, ToolSchemas};
use crate::unified::{
    Result, UnifiedParser, UnifiedParserError, UnifiedParserOutput, parsing_failed,
};
use crate::utils::{incomplete, parse_buffered_event, safe_text_len_mul};

/// Prompt tail windows, in tokens, searched for the start anchor before
/// decoding the whole prompt.
const PROMPT_WINDOWS: [usize; 3] = [64, 512, 4096];

/// Unified parser executing a checkpoint's `response_template`.
///
/// Original Python implementation:
/// <https://github.com/huggingface/transformers/blob/6d43ab4008/src/transformers/utils/chat_parsing/response_parser.py>
pub struct HfUnifiedParser {
    template: Arc<ResponseTemplate>,
    tool_schemas: ToolSchemas,
    tokenizer: DynTokenizer,
    buffer: DecodedText,
    mode: HfMode,
    /// Whether each region has closed an occurrence.
    closed: Vec<bool>,
    /// Strip state per region: per occurrence with `repeats`, per stream otherwise.
    strip: Vec<StripState>,
    emitted_tool_count: usize,
}

/// Parser mode: the region currently receiving text.
enum HfMode {
    /// No implicit field is declared: text outside regions is discarded.
    Discard,
    /// The implicit region, opened lazily on its first byte.
    Implicit(Occurrence),
    /// An explicit region, opened by its open boundary.
    Explicit(Occurrence),
}

/// One occurrence of a region.
struct Occurrence {
    region: usize,
    /// Whether the occurrence has content to close (`_opened`).
    opened: bool,
    captures: Captures,
    /// Opener text, returned by [`UnifiedParser::reset`].
    raw_open: String,
    /// Buffered body of a tool-call region.
    body: String,
    /// Index of the tool call started at the opener, if any.
    active_tool_index: Option<usize>,
    /// `join` separator still to emit before this occurrence's content.
    separator: Option<String>,
}

/// Strip-compatible streaming state of a text or reasoning occurrence.
#[derive(Default)]
struct StripState {
    enabled: bool,
    /// Whether a non-whitespace character has been seen.
    started: bool,
    /// Trailing whitespace held until more text arrives or the region closes.
    held: DecodedText,
}

/// One parsed event of the buffered input.
enum HfEvent {
    /// Text routed into the current region.
    Text,
    /// A boundary of the current mode's watch list.
    Boundary { watch: Watch, captures: Captures },
}

/// A committable boundary match at the current position.
struct BoundaryMatch {
    watch: Watch,
    len: usize,
    captures: Captures,
}

impl BoundaryMatch {
    /// Preference order of `_scan`: longest first, opens before closes, then by
    /// field name.
    fn key(&self, template: &ResponseTemplate) -> (std::cmp::Reverse<usize>, bool, FieldName) {
        (
            std::cmp::Reverse(self.len),
            self.watch.kind == WatchKind::Close,
            template.regions[self.watch.region].name,
        )
    }
}

/// How the event loop treats its input.
#[derive(Clone, Copy)]
enum Phase {
    /// Prompt text after the start anchor: state transitions only, no events.
    Prefill,
    /// Generated text; more may follow.
    Stream,
    /// Generated text at the end of the stream.
    Eof,
}

impl Phase {
    fn emits(self) -> bool {
        !matches!(self, Self::Prefill)
    }
}

impl HfUnifiedParser {
    /// Create a parser for one request.
    pub fn new(template: Arc<ResponseTemplate>, tools: &[Tool], tokenizer: DynTokenizer) -> Self {
        let mut parser = Self {
            closed: vec![false; template.regions.len()],
            strip: template.regions.iter().map(StripState::new).collect(),
            mode: HfMode::Discard,
            template,
            tool_schemas: ToolSchemas::from_tools(tools),
            tokenizer,
            buffer: DecodedText::default(),
            emitted_tool_count: 0,
        };
        parser.reset_to_implicit();
        parser
    }

    /// Clear all per-stream state.
    fn reset_state(&mut self) {
        self.buffer.clear();
        self.closed.fill(false);
        self.strip = self.template.regions.iter().map(StripState::new).collect();
        self.emitted_tool_count = 0;
        self.reset_to_implicit();
    }

    /// Return to the implicit region, or discard text when there is none.
    fn reset_to_implicit(&mut self) {
        self.mode = match self.template.implicit {
            Some(region) => {
                HfMode::Implicit(self.occurrence(region, false, Captures::new(), String::new()))
            }
            None => HfMode::Discard,
        };
    }

    /// Start a new occurrence of `region`.
    fn occurrence(
        &mut self,
        region: usize,
        opened: bool,
        captures: Captures,
        raw_open: String,
    ) -> Occurrence {
        let spec = &self.template.regions[region];
        let mut separator = None;
        if let RegionKind::Text(text) = &spec.kind {
            if let Repeat::Join(join) = &text.repeat
                && !join.is_empty()
                && self.closed[region]
            {
                separator = Some(join.clone());
            }
            if text.repeat != Repeat::Once {
                self.strip[region] = StripState::new(spec);
            }
        }
        Occurrence {
            region,
            opened,
            captures,
            raw_open,
            body: String::new(),
            active_tool_index: None,
            separator,
        }
    }

    /// Parse buffered input until more is needed.
    fn drive(&mut self, output: &mut UnifiedParserOutput, phase: Phase) -> Result<()> {
        let template = Arc::clone(&self.template);
        loop {
            let watch = match &self.mode {
                HfMode::Explicit(occurrence) => &template.regions[occurrence.region].close_watch,
                HfMode::Discard | HfMode::Implicit(_) => &template.idle_watch,
            };
            let eof = matches!(phase, Phase::Eof);
            let Some((step, consumed_len)) = parse_buffered_event(&self.buffer.text, |input| {
                parse_next_event(input, &template, watch, eof)
            })?
            else {
                return Ok(());
            };
            let piece = self.buffer.drain_prefix(consumed_len);
            match step {
                HfEvent::Text => self.route(piece, output, phase),
                HfEvent::Boundary { watch, captures } => {
                    self.close_current(output, phase)?;
                    match watch.kind {
                        WatchKind::Open => {
                            self.open_explicit(watch.region, captures, piece.text, output, phase)
                        }
                        // The implicit region's close: start a fresh implicit occurrence.
                        WatchKind::Close => {}
                    }
                }
            }
        }
    }

    /// Route a text piece into the current region.
    fn route(&mut self, piece: DecodedText, output: &mut UnifiedParserOutput, phase: Phase) {
        let (occurrence, region) = match &mut self.mode {
            HfMode::Discard => return,
            HfMode::Implicit(occurrence) | HfMode::Explicit(occurrence) => {
                let region = &self.template.regions[occurrence.region];
                (occurrence, region)
            }
        };
        let RegionKind::Text(text) = &region.kind else {
            if !piece.text.is_empty() {
                occurrence.opened = true;
                occurrence.body.push_str(&piece.text);
            }
            return;
        };
        let reasoning = text.role == TextRole::Reasoning;
        if piece.text.is_empty() {
            // Zero-width tokens still count as reasoning.
            if reasoning && phase.emits() {
                output.push_reasoning(piece);
            }
            return;
        }
        occurrence.opened = true;

        let strip = &mut self.strip[occurrence.region];
        if !phase.emits() {
            // Prompt text is never emitted; it only decides whether later
            // leading whitespace is still stripped.
            strip.started |= !piece.text.trim_start().is_empty();
            return;
        }
        let (visible, dropped) = strip.feed(piece);
        if reasoning {
            push_dropped_reasoning(output, dropped);
        }
        if visible.text.is_empty() {
            return;
        }
        if let Some(separator) = occurrence.separator.take() {
            push_text(output, text.role, DecodedText::unattributed(separator));
        }
        push_text(output, text.role, visible);
    }

    /// Open an explicit region.
    fn open_explicit(
        &mut self,
        region: usize,
        captures: Captures,
        raw_open: String,
        output: &mut UnifiedParserOutput,
        phase: Phase,
    ) {
        let mut occurrence = self.occurrence(region, true, captures, raw_open);
        let opener_name = match &self.template.regions[region].kind {
            RegionKind::ToolCalls(ToolCallRegion {
                name_group: Some(group),
                ..
            }) => occurrence.captures.get(group).and_then(Value::as_str).map(str::to_string),
            _ => None,
        };
        if let Some(name) = opener_name
            && phase.emits()
        {
            let tool_index = self.allocate_tool_index();
            occurrence.active_tool_index = Some(tool_index);
            output.push_call(ToolCallDelta {
                tool_index,
                name: Some(name),
                arguments: String::new(),
            });
        }
        self.mode = HfMode::Explicit(occurrence);
    }

    /// Close the current region and reset to the implicit region.
    ///
    /// Skipped (aside from the reset) when the current region never opened --
    /// avoids vacuous open/close pairs at every explicit boundary.
    fn close_current(&mut self, output: &mut UnifiedParserOutput, phase: Phase) -> Result<()> {
        let template = Arc::clone(&self.template);
        let mode = std::mem::replace(&mut self.mode, HfMode::Discard);
        let occurrence = match mode {
            HfMode::Implicit(occurrence) | HfMode::Explicit(occurrence) if occurrence.opened => {
                Some(occurrence)
            }
            _ => None,
        };
        // Without `repeats`, trailing whitespace stays held in case the field
        // continues; it is dropped at the end of the stream.
        let held = occurrence
            .as_ref()
            .filter(|occurrence| {
                matches!(&template.regions[occurrence.region].kind,
                    RegionKind::Text(text) if text.repeat != Repeat::Once)
            })
            .map(|occurrence| take(&mut self.strip[occurrence.region].held));
        if let Some(occurrence) = &occurrence {
            self.closed[occurrence.region] = true;
        }
        self.reset_to_implicit();
        // Regions closed inside the prompt contribute nothing to the output.
        let Some(occurrence) = occurrence.filter(|_| phase.emits()) else {
            return Ok(());
        };

        let region = &template.regions[occurrence.region];
        match &region.kind {
            RegionKind::Text(text) => {
                if text.role == TextRole::Reasoning
                    && let Some(held) = held
                {
                    push_dropped_reasoning(output, held);
                }
                // An empty occurrence still joins: `previous + join + ""`.
                if let Some(separator) = occurrence.separator {
                    push_text(output, text.role, DecodedText::unattributed(separator));
                }
            }
            RegionKind::ToolCalls(calls) => {
                let parsed = calls.content.parse(&occurrence.body)?;
                let value = match &calls.transform {
                    Some(transform) => {
                        transform.apply(region.name.as_str(), parsed, &occurrence.captures)?
                    }
                    None => parsed,
                };
                self.emit_calls(value, occurrence.active_tool_index, output)?;
            }
        }
        Ok(())
    }

    /// Emit the tool calls of a closed tool-call region.
    // TODO: stream arguments incrementally instead of once at region close; the
    // dialects (e.g. Gemma 4's unquoted keys and `<|"|>` strings, `xml-inline`)
    // need incremental conversion rather than raw passthrough.
    fn emit_calls(
        &mut self,
        value: Value,
        started: Option<usize>,
        output: &mut UnifiedParserOutput,
    ) -> Result<()> {
        let calls = match value {
            Value::Array(calls) => calls,
            call => vec![call],
        };
        for (position, call) in calls.into_iter().enumerate() {
            let (name, arguments) = call_parts(call, &self.tool_schemas)?;
            let (tool_index, name) = match (position, started) {
                (0, Some(tool_index)) => (tool_index, None),
                _ => (self.allocate_tool_index(), Some(name)),
            };
            output.push_call(ToolCallDelta {
                tool_index,
                name,
                arguments,
            });
        }
        Ok(())
    }

    fn allocate_tool_index(&mut self) -> usize {
        let tool_index = self.emitted_tool_count;
        self.emitted_tool_count += 1;
        tool_index
    }

    /// Return the prompt text after the last start-anchor match, if any.
    ///
    /// Searches geometrically growing tail windows before decoding the whole
    /// prompt. A decoded suffix can differ from the full decode only in its first
    /// character, so a match is accepted only after that character unless the
    /// window is the whole prompt.
    fn prompt_remainder(&self, prompt_token_ids: &[u32]) -> Result<Option<String>> {
        let len = prompt_token_ids.len();
        let windows = PROMPT_WINDOWS.into_iter().filter(|window| *window < len).chain([len]);
        for window in windows {
            let text = self.tokenizer.decode(&prompt_token_ids[len - window..], false).map_err(
                |error| parsing_failed!("failed to decode prompt: {}", error.as_report()),
            )?;
            let whole = window == len;
            if let Some((start, end)) = self.template.last_start_anchor(&text) {
                let first_char = text.chars().next().map_or(0, char::len_utf8);
                if whole || start >= first_char {
                    return Ok(Some(text[end..].to_string()));
                }
            }
        }
        Ok(None)
    }
}

impl StripState {
    fn new(region: &Region) -> Self {
        Self {
            enabled: matches!(
                region.kind,
                RegionKind::Text(TextRegion { strip: true, .. })
            ),
            ..Self::default()
        }
    }

    /// Split a piece into its visible part and whitespace dropped by `strip`.
    ///
    /// Leading whitespace is dropped until the first non-whitespace character;
    /// trailing whitespace is held until more non-whitespace text arrives (then
    /// released) or the region closes (then dropped).
    fn feed(&mut self, mut piece: DecodedText) -> (DecodedText, DecodedText) {
        if !self.enabled {
            return (piece, DecodedText::default());
        }
        let mut dropped = DecodedText::default();
        if !self.started {
            let lead = piece.text.len() - piece.text.trim_start().len();
            if lead == piece.text.len() {
                return (DecodedText::default(), piece);
            }
            dropped = piece.drain_prefix(lead);
            self.started = true;
        }
        let body_len = piece.text.trim_end().len();
        if body_len == 0 {
            self.held.append(piece);
            return (DecodedText::default(), dropped);
        }
        let head = piece.drain_prefix(body_len);
        let mut visible = take(&mut self.held);
        visible.append(head);
        self.held = piece;
        (visible, dropped)
    }
}

/// Push visible text of a text region.
fn push_text(output: &mut UnifiedParserOutput, role: TextRole, piece: DecodedText) {
    match role {
        TextRole::Content => output.push_text(piece.text),
        TextRole::Reasoning => output.push_reasoning(piece),
    }
}

/// Keep the tokens of stripped reasoning whitespace in the reasoning count.
fn push_dropped_reasoning(output: &mut UnifiedParserOutput, dropped: DecodedText) {
    if dropped.attributions.is_empty() {
        return;
    }
    output.push_reasoning(DecodedText {
        text: String::new(),
        attributions: dropped
            .attributions
            .into_iter()
            .map(|attribution| TokenAttribution {
                token_id: attribution.token_id,
                anchor: TokenAnchor::ZeroWidth { byte_offset: 0 },
            })
            .collect(),
    });
}

/// One tool call produced by a tool-call region, after its transform.
#[derive(Deserialize)]
#[serde(untagged)]
enum ToolCallValue {
    /// The OpenAI shape `{"type": "function", "function": {...}}`.
    Wrapped { function: FunctionValue },
    /// The bare `{"name", "arguments"}` shape.
    Bare(FunctionValue),
}

#[derive(Deserialize)]
struct FunctionValue {
    name: String,
    /// Missing or `null` arguments become `{}`.
    #[serde(default)]
    arguments: Option<ArgumentsValue>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ArgumentsValue {
    /// An argument object, typed by the tool schema.
    Object(Map<String, Value>),
    /// Pre-serialized or non-JSON arguments (e.g. an `allow_non_json`
    /// fallback), passed through unchanged.
    Text(String),
    /// Any other JSON value, serialized as is.
    Other(Value),
}

/// Extract the function name and serialized arguments of one tool-call value.
///
/// String argument values are converted by the calling tool's parameter
/// schema, as for the other tool parsers.
fn call_parts(call: Value, tool_schemas: &ToolSchemas) -> Result<(String, String)> {
    let (ToolCallValue::Wrapped { function } | ToolCallValue::Bare(function)) =
        ToolCallValue::deserialize(&call).map_err(|_| {
            parsing_failed!(
                "tool call must be {{\"function\": {{\"name\": ..., \"arguments\": ...}}}} or \
                 {{\"name\": ..., \"arguments\": ...}} with a string name, got {call}"
            )
        })?;
    let FunctionValue { name, arguments } = function;
    let arguments = match arguments {
        None => "{}".to_string(),
        Some(ArgumentsValue::Text(arguments)) => arguments,
        Some(ArgumentsValue::Object(mut arguments)) => {
            convert_text_arguments(tool_schemas, &name, &mut arguments);
            serialize_arguments(&arguments)?
        }
        Some(ArgumentsValue::Other(arguments)) => serialize_arguments(&arguments)?,
    };
    Ok((name, arguments))
}

/// Convert the argument values a content parser left as raw text by the tool
/// schema: top-level strings, and string lists collected from duplicate keys
/// (`merge_duplicates`).
///
/// Deeper values are the content parser's own JSON and are kept, as in
/// Transformers' `_coerce_tool_calls`.
fn convert_text_arguments(
    tool_schemas: &ToolSchemas,
    function_name: &str,
    arguments: &mut Map<String, Value>,
) {
    for (name, value) in arguments.iter_mut() {
        let values = match value {
            Value::Array(items) => items.as_mut_slice(),
            value => std::slice::from_mut(value),
        };
        for value in values {
            if let Value::String(text) = value {
                *value = tool_schemas.convert_param_with_schema(function_name, name, take(text));
            }
        }
    }
}

/// Serialize tool-call arguments to JSON text.
fn serialize_arguments(arguments: &impl serde::Serialize) -> Result<String> {
    serde_json::to_string(arguments).map_err(|error| {
        parsing_failed!("failed to serialize tool arguments: {}", error.as_report())
    })
}

/// Parse one event: safe text before the next marker, or a boundary at it.
fn parse_next_event(
    input: &mut Partial<&str>,
    template: &ResponseTemplate,
    watch: &WatchSet,
    eof: bool,
) -> ModalResult<HfEvent> {
    let markers: Vec<&str> = watch.markers.iter().map(String::as_str).collect();
    match safe_text_len_mul(input, &markers) {
        Ok(0) => {}
        Ok(_) => return Ok(HfEvent::Text),
        // At the end of the stream a partial marker is plain text.
        Err(ErrMode::Incomplete(_)) if eof => {
            let len = input.eof_offset();
            input.next_slice(len);
            return Ok(HfEvent::Text);
        }
        Err(error) => return Err(error),
    }

    // Any pending boundary blocks the position; otherwise the preferred match wins.
    let text = **input;
    let mut best: Option<BoundaryMatch> = None;
    for &watch in &watch.boundaries {
        match template.boundary(watch).resolve(text, eof) {
            Resolution::Pending => return incomplete(),
            Resolution::NoMatch => {}
            Resolution::Matched { len, captures } => {
                let candidate = BoundaryMatch {
                    watch,
                    len,
                    captures,
                };
                if best.as_ref().is_none_or(|best| candidate.key(template) < best.key(template)) {
                    best = Some(candidate);
                }
            }
        }
    }

    match best {
        Some(BoundaryMatch {
            watch,
            len,
            captures,
        }) => {
            input.next_slice(len);
            Ok(HfEvent::Boundary { watch, captures })
        }
        None => {
            // Not a boundary: the marker's first character is plain text.
            let len = text.chars().next().map_or(0, char::len_utf8);
            input.next_slice(len);
            Ok(HfEvent::Text)
        }
    }
}

impl UnifiedParser for HfUnifiedParser {
    fn create(_tools: &[Tool], _tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Err(UnifiedParserError::NoNamedConstructor {
            parser: "hf",
            built_from: "the checkpoint's response_template",
        })
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.reset_state();
        if prompt_token_ids.is_empty() {
            return Ok(());
        }
        let Some(remainder) = self.prompt_remainder(prompt_token_ids)? else {
            self.template.warn_missing_anchor_once();
            return Ok(());
        };
        // A partial boundary at the end of the prompt stays buffered, so the
        // generated text can complete it.
        self.buffer = DecodedText::unattributed(remainder);
        self.drive(&mut UnifiedParserOutput::default(), Phase::Prefill)
    }

    fn preserve_special_tokens(&self) -> bool {
        // Templates spell boundaries with special tokens and expect decoding
        // with them kept.
        true
    }

    // TODO: derive a structural-tag output grammar from literal-bounded templates.

    fn parse_into(&mut self, delta: DecodedText, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffer.append(delta);
        self.drive(output, Phase::Stream)
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        self.drive(&mut output, Phase::Eof)?;
        // Trailing zero-width tokens carry no text for the step loop to consume.
        let rest = self.buffer.take();
        self.route(rest, &mut output, Phase::Eof);
        self.close_current(&mut output, Phase::Eof)?;
        for (region, strip) in self.template.regions.iter().zip(&mut self.strip) {
            let held = take(&mut strip.held);
            if matches!(&region.kind, RegionKind::Text(text) if text.role == TextRole::Reasoning) {
                push_dropped_reasoning(&mut output, held);
            }
        }

        let missing: Vec<_> = self
            .template
            .regions
            .iter()
            .zip(&self.closed)
            .filter(|(region, closed)| !region.optional && !**closed)
            .map(|(region, _)| region.name.as_str())
            .collect();
        if !missing.is_empty() {
            return Err(parsing_failed!(
                "Required response_template fields missing from parsed output: {missing:?}"
            ));
        }

        self.reset_state();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        let mut raw = String::new();
        if let HfMode::Implicit(occurrence) | HfMode::Explicit(occurrence) =
            std::mem::replace(&mut self.mode, HfMode::Discard)
        {
            raw.push_str(&occurrence.raw_open);
            raw.push_str(&occurrence.body);
            raw.push_str(&self.strip[occurrence.region].held.text);
        }
        raw.push_str(&self.buffer.take().text);
        self.reset_state();
        raw
    }
}
