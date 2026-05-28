use serde_json::{Map, Number, Value};
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, opt, separated, seq, terminated};
use winnow::error::{ContextError, ErrMode, ModalResult};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, take_till, take_until};

use super::utils::{parse_buffered_event, safe_text_len};
use super::{Result, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::Tool;

const TOOL_CALL_START: &str = "<|tool_call>";
const TOOL_CALL_END: &str = "<tool_call|>";
const STRING_DELIM: &str = "<|\"|>";
const CALL_PREFIX: &str = "call:";

type Gemma4Input<'i> = Partial<&'i str>;

/// Tool parser for Google Gemma4 models.
///
/// Original Python implementation:
/// <https://github.com/vllm-project/vllm/blob/bf45e6d0a558da2b8d7b60efb07b4aa394f3b60b/vllm/tool_parsers/gemma4_tool_parser.py>
///
/// Handles the Gemma4 function call format:
///
/// `<|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>`
///
/// Arguments are emitted only after a full Gemma4 tool call is parsed.
pub struct Gemma4ToolParser {
    buffer: String,
    emitted_tool_count: usize,
    scan: Gemma4ScanState,
}

/// Streaming state for the Gemma4 tool call body.
///
/// Tracks how far the args body has already been scanned across `parse_into`
/// calls so that successive chunks only walk the newly arrived bytes, keeping
/// the total scan cost linear in stream length rather than quadratic in chunk
/// count.
#[derive(Debug, Clone, Default)]
struct Gemma4ScanState {
    phase: Gemma4ScanPhase,
    /// Function name parsed from the tool call header.
    /// Populated when transitioning into `InArgs`.
    name: String,
    /// Byte offset in `buffer` where the args body starts (right after the
    /// opening `{`).
    args_body_start: usize,
    /// Byte offset in `buffer` where the args body's closing `}` sits.
    /// Populated when transitioning into `AfterArgs`.
    args_body_end: usize,
    /// Byte offset in `buffer` up to which the args body has already been
    /// scanned. Each byte is visited at most once across `parse_into` calls.
    scan_offset: usize,
    /// Grammar-aware position within the args body. Tracking key/value/string
    /// context (rather than blindly counting braces) is required because
    /// Gemma4 keys (`take_till(':')`) and bare values (`take_till(',}]')`) can
    /// legally contain `{`, `[`, `]` and the `<|"|>` delimiter as content.
    mode: ArgsScanMode,
    /// Stack of currently open containers. The args object itself is pushed
    /// when entering `InArgs`; the body is fully closed once this empties.
    containers: Vec<Container>,
    /// Number of `STRING_DELIM` bytes matched at the current position, used to
    /// detect string delimiters across chunk splits (both when deciding
    /// whether a value start is a string and when seeking a string's close).
    string_marker_progress: usize,
}

/// A currently open container within the Gemma4 args body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

/// Grammar-aware scan position within the Gemma4 args body.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ArgsScanMode {
    /// In an object, before a key (just after `{` or `,`). Whitespace is
    /// skipped; `}` closes the object; anything else begins a key.
    #[default]
    ObjectKeyStart,
    /// Consuming a bare key until `:`.
    Key,
    /// After `:` in an object — expecting a value.
    ValueStart,
    /// In an array, before an element (just after `[` or `,`). Whitespace is
    /// skipped; `]` closes the array; anything else begins a value.
    ArrayElemStart,
    /// Matched some `STRING_DELIM` bytes at a value start; deciding whether the
    /// value is a delimited string or a bare value that merely begins with `<`.
    ValueStringDetect,
    /// Inside a `<|"|>`-delimited string, seeking the closing delimiter.
    StringBody,
    /// Inside a bare scalar value, until `,`, `}` or `]`.
    BareValue,
    /// After a complete value — expecting `,`, `}` or `]`.
    AfterValue,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Gemma4ScanPhase {
    /// Not inside a tool call: looking for the start marker or emitting safe
    /// text.
    #[default]
    Outside,
    /// Inside the args body, incrementally scanning until matched braces close.
    InArgs,
    /// Args body fully matched; awaiting the `<tool_call|>` end marker.
    AfterArgs,
}

impl Gemma4ToolParser {
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            emitted_tool_count: 0,
            scan: Gemma4ScanState::default(),
        }
    }

    fn reset_scan(&mut self) {
        self.scan = Gemma4ScanState::default();
    }

    /// Drop the bytes belonging to the current failed tool call so the
    /// remainder is reprocessed from `Outside`. Without this, the header
    /// would keep re-matching on every subsequent `parse_into` and re-fail in
    /// a loop. `drain_to` controls how much is discarded: the `AfterArgs`
    /// JSON-error path passes past the verified end marker (so we don't leak
    /// it as text); the `AfterArgs` bad-tail path passes just past the closing
    /// brace. (The `InArgs` structural-error path does not use this helper — it
    /// fails closed by emitting the whole buffer as text; see `drive`.)
    fn discard_failed_call(&mut self, drain_to: usize) {
        self.buffer.drain(..drain_to);
        self.reset_scan();
    }

    /// Recover from a parse error encountered in `Outside` (e.g. a
    /// whitespace-only tool name that triggers a `Cut` in
    /// `gemma4_tool_name`). The only parser in `parse_outside_event` that
    /// can hard-fail is `tool_call_header_event`, and it can only do so
    /// after `literal(TOOL_CALL_START)` has matched — so the buffer
    /// provably starts with `TOOL_CALL_START` here, and dropping its
    /// leading `<` stays on a char boundary. The next iteration's
    /// `safe_text_event` will then resync on the next real
    /// `TOOL_CALL_START`, preserving any same-chunk valid call.
    fn discard_failed_outside(&mut self) {
        debug_assert!(
            self.buffer.starts_with(TOOL_CALL_START),
            "discard_failed_outside expects an in-flight start marker"
        );
        if !self.buffer.is_empty() {
            self.buffer.drain(..1);
        }
        self.reset_scan();
    }

    /// Drive the streaming state machine over the current buffer until
    /// `NeedMore` breaks the loop (no full event is available without more
    /// input). Malformed calls are recovered inline — the relevant `discard_*`
    /// helper drains the failed bytes (or they are emitted as text) and the
    /// loop continues — so a parse pass keeps any text and calls already
    /// accumulated this round instead of discarding them. Returns `Err` only on
    /// a genuine internal failure (argument re-serialization), which is
    /// infallible in practice.
    fn drive(&mut self, output: &mut ToolParserOutput) -> Result<()> {
        loop {
            match self.scan.phase {
                Gemma4ScanPhase::Outside => {
                    let parsed = match parse_buffered_event(&self.buffer, parse_outside_event) {
                        Ok(parsed) => parsed,
                        Err(_) => {
                            // Malformed header (e.g. whitespace-only tool name).
                            // Drop the leading `<` so the next iteration's
                            // `safe_text_event` resyncs on the next real start
                            // marker, surfacing the broken bytes as text rather
                            // than discarding the whole parse result.
                            self.discard_failed_outside();
                            continue;
                        }
                    };
                    let Some((event, consumed_len)) = parsed else {
                        break;
                    };
                    match event {
                        OutsideEvent::Text { len } => {
                            output.normal_text.push_str(&self.buffer[..len]);
                            self.buffer.drain(..len);
                        }
                        OutsideEvent::ToolCallHeader { name } => {
                            // `consumed_len` covers `<|tool_call>call:NAME{`,
                            // so the args body starts at exactly that offset
                            // and we keep the header bytes in `buffer` until
                            // the full call is emitted (simplifies offset
                            // bookkeeping and reset on error).
                            self.scan.phase = Gemma4ScanPhase::InArgs;
                            self.scan.name = name;
                            self.scan.args_body_start = consumed_len;
                            self.scan.scan_offset = consumed_len;
                            self.scan.mode = ArgsScanMode::ObjectKeyStart;
                            self.scan.containers.clear();
                            self.scan.containers.push(Container::Object);
                            self.scan.string_marker_progress = 0;
                        }
                    }
                }
                Gemma4ScanPhase::InArgs => {
                    let outcome = match scan_args_body(&self.buffer, &mut self.scan) {
                        Ok(outcome) => outcome,
                        Err(_) => {
                            // Fail closed: a structurally malformed args body
                            // (mismatched braces/brackets, missing separator)
                            // cannot be trusted. Surface the entire buffered
                            // region as plain text rather than re-scanning it
                            // for tool calls — a `<|tool_call>` embedded in a
                            // broken payload must never be promoted to a real
                            // call. Any same-chunk bytes after the bad call are
                            // intentionally treated as text too; recovering a
                            // trailing valid call is not worth the risk of
                            // emitting a spurious one.
                            output.normal_text.push_str(&self.buffer);
                            self.buffer.clear();
                            self.reset_scan();
                            break;
                        }
                    };
                    match outcome {
                        ArgsScanOutcome::Closed => {
                            self.scan.phase = Gemma4ScanPhase::AfterArgs;
                        }
                        ArgsScanOutcome::NeedMore => break,
                    }
                }
                Gemma4ScanPhase::AfterArgs => {
                    let after_brace = self.scan.args_body_end + 1;
                    let remaining = &self.buffer[after_brace..];
                    if remaining.starts_with(TOOL_CALL_END) {
                        let args_text =
                            &self.buffer[self.scan.args_body_start..self.scan.args_body_end];
                        let args = match parse_args_text(args_text) {
                            Ok(args) => args,
                            Err(_) => {
                                // Body has balanced delimiters but is not valid
                                // Gemma4 grammar. Drain past the verified end
                                // marker (so the `<tool_call|>` bytes don't leak
                                // as text) and continue: this keeps any text
                                // already accumulated this pass and lets a
                                // following valid call still parse.
                                self.discard_failed_call(after_brace + TOOL_CALL_END.len());
                                continue;
                            }
                        };
                        // `serde_json::to_string(&Map<String, Value>)` is
                        // infallible in practice (all keys are strings), but
                        // we surface any error rather than panic.
                        let arguments = serde_json::to_string(&args).map_err(|error| {
                            parsing_failed!("failed to serialize arguments: {}", error)
                        })?;
                        let name = std::mem::take(&mut self.scan.name);
                        output.calls.push(ToolCallDelta {
                            tool_index: self.emitted_tool_count,
                            name: Some(name),
                            arguments,
                        });
                        self.emitted_tool_count += 1;
                        let drain_len = after_brace + TOOL_CALL_END.len();
                        self.buffer.drain(..drain_len);
                        self.reset_scan();
                    } else if TOOL_CALL_END.starts_with(remaining) {
                        // `remaining` is a genuine (possibly empty) prefix of
                        // the end marker — wait for more bytes.
                        break;
                    } else {
                        // The bytes after the closing `}` can never grow into
                        // `TOOL_CALL_END`, so this call has no end marker. Drain
                        // the header and body (keeping the short trailing bytes,
                        // which flow back as text) and continue rather than
                        // discarding the whole parse result.
                        self.discard_failed_call(after_brace);
                        continue;
                    }
                }
            }
        }

        Ok(())
    }
}

impl ToolParser for Gemma4ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);
        self.drive(output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();

        // `drive` recovers from malformed calls inline (draining or emitting
        // them as text and continuing), so a single call drains the buffer
        // down to at most an incomplete trailing call. Whatever it leaves is
        // resolved by the phase check below.
        self.drive(&mut output)?;

        match self.scan.phase {
            Gemma4ScanPhase::Outside => {
                // Anything still in the buffer here is either plain text we
                // were holding back for a potential start marker, or a
                // partial start marker that never completed. Both should be
                // surfaced as normal text — except for a full start-marker
                // prefix with no body, which we treat as an incomplete tool
                // call to match the previous behavior.
                //
                // On error we leave the buffer intact (per the `finish`
                // contract) so the caller may recover it with `reset()`.
                if self.buffer.starts_with(TOOL_CALL_START) {
                    return Err(parsing_failed!("incomplete Gemma4 tool call"));
                }
                if !self.buffer.is_empty() {
                    output.normal_text.push_str(&self.buffer);
                }
            }
            Gemma4ScanPhase::InArgs | Gemma4ScanPhase::AfterArgs => {
                return Err(parsing_failed!("incomplete Gemma4 tool call"));
            }
        }

        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.emitted_tool_count = 0;
        self.scan = Gemma4ScanState::default();
        std::mem::take(&mut self.buffer)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum OutsideEvent {
    Text { len: usize },
    ToolCallHeader { name: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArgsScanOutcome {
    /// The top-level args object closed. `scan.args_body_end` is set to the
    /// byte offset of the closing `}` in `buffer`.
    Closed,
    /// Reached end of buffer without closing — need more input.
    NeedMore,
}

/// Parse one event when not currently inside a tool call body.
fn parse_outside_event(input: &mut Gemma4Input<'_>) -> ModalResult<OutsideEvent> {
    alt((tool_call_header_event, safe_text_event)).parse_next(input)
}

/// Parse the header part of a Gemma4 tool call: `<|tool_call>call:NAME{`.
///
/// The opening `{` is consumed so the caller can begin scanning the args body
/// from the next byte without re-checking the brace.
fn tool_call_header_event(input: &mut Gemma4Input<'_>) -> ModalResult<OutsideEvent> {
    let name = seq!(
        _: literal(TOOL_CALL_START),
        _: literal(CALL_PREFIX),
        gemma4_tool_name,
        _: literal("{"),
    )
    .parse_next(input)?
    .0;
    Ok(OutsideEvent::ToolCallHeader { name })
}

/// Parse a Gemma4 tool name.
fn gemma4_tool_name(input: &mut Gemma4Input<'_>) -> ModalResult<String> {
    let name = take_until(1.., "{").parse_next(input)?.trim();
    if name.is_empty() {
        return Err(ErrMode::Cut(ContextError::new()));
    }
    Ok(name.to_string())
}

/// Parse a safe text run before the next Gemma4 start marker.
fn safe_text_event(input: &mut Gemma4Input<'_>) -> ModalResult<OutsideEvent> {
    safe_text_len(input, TOOL_CALL_START).map(|len| OutsideEvent::Text { len })
}

/// Incrementally scan the Gemma4 args body in `buffer`, starting from
/// `state.scan_offset` and updating `state` in place so that subsequent calls
/// continue from where the previous one stopped.
///
/// This is the core of the linear-time streaming behavior: each byte is
/// inspected at most once across all `parse_into` invocations for a given tool
/// call.
///
/// The scanner tracks grammar context (key vs. value vs. string vs. bare
/// scalar) so that `{`, `[`, `]` and the `<|"|>` delimiter are treated as
/// structural only where the grammar does — in particular a bare value or key
/// may *contain* those bytes as content.
///
/// Known limitation: a single forward pass cannot reproduce the backtracking
/// in `gemma4_value`'s `alt((string, object, array, bare_value))`. When a
/// value's first non-whitespace byte is `{`, `[` or `<|"|>` the scanner commits
/// to a container/string; if that token does not actually close, the grammar
/// would have fallen back to a bare scalar, but the scanner instead keeps
/// waiting for a close and the call is reported as incomplete. The same root
/// cause affects object-start tokens: because a key runs until `:`, the
/// scanner cannot tell an object-closing `}` from a key that *is* or *begins
/// with* `}`/`,` (e.g. `{}:1}` parses as `{"}":1}` and `{,}` as `{}`), so it
/// closes early and the call is dropped. These are pathological shapes a model
/// is unlikely to emit; crucially the scanner never promotes a body the
/// grammar would reject into a tool call.
fn scan_args_body(buffer: &str, state: &mut Gemma4ScanState) -> Result<ArgsScanOutcome> {
    let bytes = buffer.as_bytes();
    let delim = STRING_DELIM.as_bytes();

    let mut index = state.scan_offset;

    while index < bytes.len() {
        let byte = bytes[index];
        match state.mode {
            ArgsScanMode::ObjectKeyStart => {
                if is_args_ws(byte) {
                    index += 1;
                } else if byte == b'}' {
                    // Empty object or trailing comma: close this object.
                    pop_object(state)?;
                    index += 1;
                    if state.containers.is_empty() {
                        state.args_body_end = index - 1;
                        state.scan_offset = index;
                        return Ok(ArgsScanOutcome::Closed);
                    }
                    state.mode = ArgsScanMode::AfterValue;
                } else {
                    // Begin a key; reprocess this byte as key content.
                    state.mode = ArgsScanMode::Key;
                }
            }
            ArgsScanMode::Key => {
                // A key runs until `:` and may contain any other byte
                // (including `{`, `}`, `[`, `]`, `,` and `<|"|>`).
                index += 1;
                if byte == b':' {
                    state.mode = ArgsScanMode::ValueStart;
                }
            }
            ArgsScanMode::ValueStart => {
                if is_args_ws(byte) {
                    index += 1;
                } else {
                    index += scan_value_start(state, byte);
                }
            }
            ArgsScanMode::ArrayElemStart => {
                if is_args_ws(byte) {
                    index += 1;
                } else if byte == b']' {
                    // Empty array or trailing comma: close this array.
                    pop_array(state)?;
                    index += 1;
                    state.mode = ArgsScanMode::AfterValue;
                } else {
                    index += scan_value_start(state, byte);
                }
            }
            ArgsScanMode::ValueStringDetect => {
                if byte == delim[state.string_marker_progress] {
                    index += 1;
                    state.string_marker_progress += 1;
                    if state.string_marker_progress == delim.len() {
                        state.string_marker_progress = 0;
                        state.mode = ArgsScanMode::StringBody;
                    }
                } else {
                    // The `<`-prefix was not a string delimiter, so this is a
                    // bare value beginning with `<`. The consumed prefix bytes
                    // are bare content; reprocess the breaking byte.
                    state.string_marker_progress = 0;
                    state.mode = ArgsScanMode::BareValue;
                }
            }
            ArgsScanMode::StringBody => {
                index += 1;
                if byte == delim[state.string_marker_progress] {
                    state.string_marker_progress += 1;
                    if state.string_marker_progress == delim.len() {
                        state.string_marker_progress = 0;
                        state.mode = ArgsScanMode::AfterValue;
                    }
                } else {
                    // KMP restart: `STRING_DELIM` (`<|"|>`) has no proper prefix
                    // that is also a suffix, so the only viable restart is
                    // matching this byte against `delim[0]`. Revisit if
                    // `STRING_DELIM` ever changes.
                    state.string_marker_progress = usize::from(byte == delim[0]);
                }
            }
            ArgsScanMode::BareValue => {
                // A bare scalar runs until `,`, `}` or `]`; everything else
                // (including `{`, `[` and `<|"|>`) is content.
                if matches!(byte, b',' | b'}' | b']') {
                    state.mode = ArgsScanMode::AfterValue;
                } else {
                    index += 1;
                }
            }
            ArgsScanMode::AfterValue => {
                if is_args_ws(byte) {
                    index += 1;
                } else if byte == b',' {
                    index += 1;
                    state.mode = match state.containers.last() {
                        Some(Container::Object) => ArgsScanMode::ObjectKeyStart,
                        Some(Container::Array) => ArgsScanMode::ArrayElemStart,
                        None => {
                            return Err(parsing_failed!(
                                "unexpected `,` outside any Gemma4 container"
                            ));
                        }
                    };
                } else if byte == b'}' {
                    pop_object(state)?;
                    index += 1;
                    if state.containers.is_empty() {
                        state.args_body_end = index - 1;
                        state.scan_offset = index;
                        return Ok(ArgsScanOutcome::Closed);
                    }
                    state.mode = ArgsScanMode::AfterValue;
                } else if byte == b']' {
                    pop_array(state)?;
                    index += 1;
                    state.mode = ArgsScanMode::AfterValue;
                } else {
                    return Err(parsing_failed!(
                        "expected `,`, `}}` or `]` after a Gemma4 value"
                    ));
                }
            }
        }
    }

    state.scan_offset = bytes.len();
    Ok(ArgsScanOutcome::NeedMore)
}

/// Dispatch on the first non-whitespace byte of a value position, updating
/// `state.mode`. Returns how many bytes were consumed: `1` for a structural
/// open (`{`, `[`) or a string delimiter start, `0` for a bare scalar whose
/// first byte must be reprocessed in `BareValue`.
fn scan_value_start(state: &mut Gemma4ScanState, byte: u8) -> usize {
    let delim = STRING_DELIM.as_bytes();
    match byte {
        b'{' => {
            state.containers.push(Container::Object);
            state.mode = ArgsScanMode::ObjectKeyStart;
            1
        }
        b'[' => {
            state.containers.push(Container::Array);
            state.mode = ArgsScanMode::ArrayElemStart;
            1
        }
        _ if byte == delim[0] => {
            state.mode = ArgsScanMode::ValueStringDetect;
            state.string_marker_progress = 1;
            1
        }
        _ => {
            state.mode = ArgsScanMode::BareValue;
            0
        }
    }
}

/// Pop an object container, erroring on a `}`/`[` mismatch or underflow.
fn pop_object(state: &mut Gemma4ScanState) -> Result<()> {
    match state.containers.last() {
        Some(Container::Object) => {
            state.containers.pop();
            Ok(())
        }
        Some(Container::Array) => Err(parsing_failed!(
            "`}}` closing a Gemma4 array opened with `[`"
        )),
        None => Err(parsing_failed!("unbalanced `}}` in Gemma4 args")),
    }
}

/// Pop an array container, erroring on a `]`/`{` mismatch or underflow.
fn pop_array(state: &mut Gemma4ScanState) -> Result<()> {
    match state.containers.last() {
        Some(Container::Array) => {
            state.containers.pop();
            Ok(())
        }
        Some(Container::Object) => Err(parsing_failed!(
            "`]` closing a Gemma4 object opened with `{{`"
        )),
        None => Err(parsing_failed!("unbalanced `]` in Gemma4 args")),
    }
}

/// Whitespace as recognized between Gemma4 tokens (matching `multispace0`).
fn is_args_ws(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\r' | b'\n')
}

/// Parse the args body text (between the opening and closing braces) once it
/// has been fully buffered. Runs in linear time over the body.
fn parse_args_text(args_text: &str) -> Result<Map<String, Value>> {
    let mut input = Partial::new(args_text);
    let _ = input.complete();
    match terminated(gemma4_args, winnow::combinator::eof).parse_next(&mut input) {
        Ok(value) => Ok(value),
        Err(ErrMode::Incomplete(_)) => Err(parsing_failed!("incomplete Gemma4 arguments")),
        Err(ErrMode::Backtrack(error) | ErrMode::Cut(error)) => Err(parsing_failed!("{}", error)),
    }
}

/// Parse Gemma4's custom key-value argument object content.
fn gemma4_args(input: &mut Gemma4Input<'_>) -> ModalResult<Map<String, Value>> {
    let pairs: Vec<(String, Value)> = delimited(
        ws0,
        terminated(
            separated(0.., gemma4_pair, comma_separator),
            opt(comma_separator),
        ),
        ws0,
    )
    .parse_next(input)?;
    Ok(pairs.into_iter().collect())
}

/// Parse a Gemma4 key-value pair.
fn gemma4_pair(input: &mut Gemma4Input<'_>) -> ModalResult<(String, Value)> {
    let (key, value) = seq!(
        _: ws0,
        gemma4_key,
        _: ws0,
        _: literal(":"),
        _: ws0,
        gemma4_value,
    )
    .parse_next(input)?;
    Ok((key, value))
}

/// Parse a Gemma4 bare key.
fn gemma4_key(input: &mut Gemma4Input<'_>) -> ModalResult<String> {
    let key = take_till(1.., |char: char| char == ':').parse_next(input)?.trim();
    if key.is_empty() {
        return Err(ErrMode::Cut(ContextError::new()));
    }
    Ok(key.to_string())
}

/// Parse a Gemma4 value.
fn gemma4_value(input: &mut Gemma4Input<'_>) -> ModalResult<Value> {
    alt((
        gemma4_string.map(|value: &str| Value::String(value.to_string())),
        gemma4_object.map(Value::Object),
        gemma4_array_value.map(Value::Array),
        gemma4_bare_value,
    ))
    .parse_next(input)
}

/// Parse a Gemma4 string delimited by `<|"|>`.
fn gemma4_string<'i>(input: &mut Gemma4Input<'i>) -> ModalResult<&'i str> {
    delimited(
        literal(STRING_DELIM),
        take_until(0.., STRING_DELIM),
        literal(STRING_DELIM),
    )
    .parse_next(input)
}

/// Parse a nested Gemma4 object.
fn gemma4_object(input: &mut Gemma4Input<'_>) -> ModalResult<Map<String, Value>> {
    delimited(literal("{"), gemma4_args, literal("}")).parse_next(input)
}

/// Parse a Gemma4 array value.
fn gemma4_array_value(input: &mut Gemma4Input<'_>) -> ModalResult<Vec<Value>> {
    delimited(literal("["), gemma4_array_content, literal("]")).parse_next(input)
}

/// Parse Gemma4 array content.
fn gemma4_array_content(input: &mut Gemma4Input<'_>) -> ModalResult<Vec<Value>> {
    delimited(
        ws0,
        terminated(
            separated(0.., gemma4_value, comma_separator),
            opt(comma_separator),
        ),
        ws0,
    )
    .parse_next(input)
}

/// Parse a Gemma4 bare scalar.
fn gemma4_bare_value(input: &mut Gemma4Input<'_>) -> ModalResult<Value> {
    take_till(1.., |char: char| matches!(char, ',' | '}' | ']'))
        .map(parse_gemma4_scalar)
        .parse_next(input)
}

/// Parse a Gemma4 comma separator.
fn comma_separator(input: &mut Gemma4Input<'_>) -> ModalResult<()> {
    delimited(ws0, literal(","), ws0).void().parse_next(input)
}

fn parse_gemma4_scalar(value: &str) -> Value {
    let value = value.trim();
    if value.is_empty() {
        return Value::String(String::new());
    }
    if value == "true" {
        return Value::Bool(true);
    }
    if value == "false" {
        return Value::Bool(false);
    }
    if matches!(value, "null" | "none" | "nil" | "NULL" | "None" | "NIL") {
        return Value::Null;
    }
    if value.contains('.') {
        if let Ok(parsed) = value.parse::<f64>()
            && let Some(number) = Number::from_f64(parsed)
        {
            return Value::Number(number);
        }
    } else if let Ok(parsed) = value.parse::<i64>() {
        return Value::Number(Number::from(parsed));
    }

    Value::String(value.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;
    use winnow::combinator::{eof, terminated};
    use winnow::error::ErrMode;
    use winnow::prelude::*;
    use winnow::stream::Partial;

    use super::{
        ArgsScanMode, Container, Gemma4ScanPhase, Gemma4ScanState, Gemma4ToolParser, ToolCallDelta,
        ToolParser, ToolParserOutput, gemma4_args, gemma4_array_content, scan_args_body,
    };
    use crate::{Tool, ToolParserTestExt as _};

    fn parse_gemma4_args(args: &str) -> super::Result<serde_json::Map<String, Value>> {
        let mut input = Partial::new(args);
        let _ = input.complete();
        match terminated(gemma4_args, eof).parse_next(&mut input) {
            Ok(value) => Ok(value),
            Err(ErrMode::Incomplete(_)) => Err(parsing_failed!("incomplete Gemma4 arguments")),
            Err(ErrMode::Backtrack(error) | ErrMode::Cut(error)) => {
                Err(parsing_failed!("{}", error))
            }
        }
    }

    fn parse_gemma4_array(array: &str) -> super::Result<Vec<Value>> {
        let mut input = Partial::new(array);
        let _ = input.complete();
        match terminated(gemma4_array_content, eof).parse_next(&mut input) {
            Ok(value) => Ok(value),
            Err(ErrMode::Incomplete(_)) => Err(parsing_failed!("incomplete Gemma4 array")),
            Err(ErrMode::Backtrack(error) | ErrMode::Cut(error)) => {
                Err(parsing_failed!("{}", error))
            }
        }
    }

    fn test_tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "get_weather".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "get_time".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "write_file".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "Edit".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "search".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "set".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "get_status".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "todowrite".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
        ]
    }

    fn collect_stream(chunks: &[&str]) -> ToolParserOutput {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();
        for chunk in chunks {
            output.append(parser.parse_chunk(chunk).unwrap());
        }
        output.append(parser.finish().unwrap());
        output.coalesce_calls()
    }

    fn first_call(output: &ToolParserOutput) -> &ToolCallDelta {
        output.calls.first().expect("expected one tool call")
    }

    #[test]
    fn gemma4_parse_args_handles_scalars_and_nested_values() {
        let parsed = parse_gemma4_args(
            "name:<|\"|>test<|\"|>,count:42,active:true,score:114.514,nested:{inner:<|\"|>value<|\"|>},items:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]",
        )
        .unwrap();

        assert_eq!(
            Value::Object(parsed),
            json!({
                "name": "test",
                "count": 42,
                "active": true,
                "score": 114.514,
                "nested": { "inner": "value" },
                "items": ["a", "b"],
            })
        );
    }

    #[test]
    fn gemma4_parse_args_handles_empty_arguments() {
        let parsed = parse_gemma4_args("").unwrap();
        assert_eq!(Value::Object(parsed), json!({}));
    }

    #[test]
    fn gemma4_parse_array_handles_bare_values() {
        let parsed = parse_gemma4_array("42,true,114.514").unwrap();
        assert_eq!(Value::Array(parsed), json!([42, true, 114.514]));
    }

    #[test]
    fn gemma4_parse_complete_extracts_single_tool_call() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let output = parser
            .parse_complete("<|tool_call>call:get_weather{location:<|\"|>London<|\"|>}<tool_call|>")
            .unwrap();

        assert!(output.normal_text.is_empty());
        assert_eq!(output.calls.len(), 1);
        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "London" })
        );
    }

    #[test]
    fn gemma4_parse_complete_rejects_incomplete_tool_call() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let error = parser
            .parse_complete("<|tool_call>call:get_weather{location:<|\"|>London")
            .unwrap_err();

        assert!(error.to_report_string().contains("incomplete Gemma4 tool call"));
    }

    #[test]
    fn gemma4_streaming_basic_single_tool_call() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:get_weather{",
            "location:<|\"|>Paris",
            ", France",
            "<|\"|>}",
            "<tool_call|>",
        ]);

        assert!(output.normal_text.is_empty());
        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "Paris, France" })
        );
    }

    #[test]
    fn gemma4_streaming_text_before_and_after_tool_call() {
        let output = collect_stream(&[
            "Let me check ",
            "the weather. ",
            "<|tool_call>",
            "call:get_weather{",
            "location:<|\"|>London<|\"|>}",
            "<tool_call|><",
            "div>",
        ]);

        assert_eq!(output.normal_text, "Let me check the weather. <div>");
        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "London" })
        );
    }

    #[test]
    fn gemma4_streaming_waits_for_complete_tool_call() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();

        for chunk in [
            "<|tool_call>",
            "call:get_weather{",
            "location:<|\"|>Paris<|\"|>}",
        ] {
            output.append(parser.parse_chunk(chunk).unwrap());
            assert!(output.calls.is_empty());
        }

        output.append(parser.parse_chunk("<tool_call|>").unwrap());
        let output = output.coalesce_calls();

        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "Paris" })
        );
    }

    #[test]
    fn gemma4_streaming_handles_boolean_split_across_chunks() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:search{input:{all:tru",
            "e}}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&output).name.as_deref(), Some("search"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "input": { "all": true } })
        );
    }

    #[test]
    fn gemma4_streaming_handles_false_split_across_chunks() {
        let output = collect_stream(&["<|tool_call>", "call:set{flag:fals", "e}", "<tool_call|>"]);

        assert_eq!(first_call(&output).name.as_deref(), Some("set"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "flag": false })
        );
    }

    #[test]
    fn gemma4_streaming_handles_number_split_across_chunks() {
        let output = collect_stream(&["<|tool_call>", "call:set{count:4", "2}", "<tool_call|>"]);

        assert_eq!(first_call(&output).name.as_deref(), Some("set"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "count": 42 })
        );
    }

    #[test]
    fn gemma4_streaming_handles_split_string_delimiter() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:todowrite{",
            "content:<|\"|>Buy milk<|",
            "\"|>}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&output).name.as_deref(), Some("todowrite"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "content": "Buy milk" })
        );
        assert!(!first_call(&output).arguments.contains("<|"));
    }

    #[test]
    fn gemma4_streaming_handles_end_marker_literal_inside_string() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:todowrite{",
            "content:<|\"|>literal }<tool_call|> inside",
            "<|\"|>}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&output).name.as_deref(), Some("todowrite"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "content": "literal }<tool_call|> inside" })
        );
    }

    #[test]
    fn gemma4_streaming_handles_html_argument_without_duplication() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:write_file{",
            "path:<|\"|>index.html<|\"|>,",
            "content:<|\"|><!DOCTYPE html>\n<",
            "html lang=\"zh-CN\">\n<",
            "head>\n    <",
            "meta charset=\"UTF-8\">\n    <",
            "meta name=\"viewport\" content=\"width=device-width\">\n",
            "<|\"|>}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&output).name.as_deref(), Some("write_file"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({
                "path": "index.html",
                "content": "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width\">\n",
            })
        );
    }

    #[test]
    fn gemma4_streaming_trailing_bare_bool_is_not_duplicated() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:Edit{",
            "file_path:<|\"|>src/env.py<|\"|>,",
            "old_string:<|\"|>old_val<|\"|>,",
            "new_string:<|\"|>new_val<|\"|>,",
            "replace_all:",
            "false}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&output).name.as_deref(), Some("Edit"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({
                "file_path": "src/env.py",
                "old_string": "old_val",
                "new_string": "new_val",
                "replace_all": false,
            })
        );
        assert_eq!(
            first_call(&output).arguments.matches("replace_all").count(),
            1
        );
    }

    #[test]
    fn gemma4_finish_flushes_partial_start_marker_as_text() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let mut output = parser.parse_chunk("<").unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(output.normal_text, "<");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn gemma4_finish_rejects_complete_args_without_end_marker() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        for chunk in ["<|tool_call>", "call:get_status{}"] {
            parser.parse_chunk(chunk).unwrap();
        }

        let error = parser.finish().unwrap_err();

        assert!(error.to_report_string().contains("incomplete Gemma4 tool call"));
    }

    /// Drive a 10 KiB string argument one byte at a time. Functional check
    /// that streaming reassembles the payload — see
    /// `gemma4_scan_args_body_visits_each_body_byte_exactly_once` for the
    /// actual linear-time invariant.
    #[test]
    fn gemma4_streaming_byte_at_a_time_large_string_argument() {
        let payload: String = (0..10_240).map(|i| char::from(b'a' + (i % 26) as u8)).collect();
        let full =
            format!("<|tool_call>call:write_file{{content:<|\"|>{payload}<|\"|>}}<tool_call|>");

        let mut parser = Gemma4ToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();
        for byte_index in 0..full.len() {
            // Use `is_char_boundary` to keep chunks UTF-8 safe; payload is ASCII.
            if !full.is_char_boundary(byte_index) || !full.is_char_boundary(byte_index + 1) {
                continue;
            }
            output.append(parser.parse_chunk(&full[byte_index..byte_index + 1]).unwrap());
        }
        output.append(parser.finish().unwrap());
        let output = output.coalesce_calls();

        assert_eq!(first_call(&output).name.as_deref(), Some("write_file"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "content": payload })
        );
    }

    /// Linear-time invariant: when the args body arrives one byte per
    /// `scan_args_body` call, `scan_offset` must advance by exactly one byte
    /// each time. An O(n²) implementation would either reset `scan_offset`
    /// or rescan from the start, both of which this asserts against.
    #[test]
    fn gemma4_scan_args_body_visits_each_body_byte_exactly_once() {
        let body = "a:<|\"|>x<|\"|>}";
        let mut buffer = String::from("{");
        let mut state = Gemma4ScanState {
            phase: Gemma4ScanPhase::InArgs,
            args_body_start: 1,
            scan_offset: 1,
            containers: vec![Container::Object],
            ..Gemma4ScanState::default()
        };

        let mut closed = false;
        for (i, byte) in body.bytes().enumerate() {
            let scan_offset_before = state.scan_offset;
            buffer.push(byte as char);
            let outcome = scan_args_body(&buffer, &mut state).unwrap();
            assert_eq!(
                state.scan_offset,
                scan_offset_before + 1,
                "scan_offset must advance by 1 on each new byte (iter {i})"
            );
            match outcome {
                super::ArgsScanOutcome::Closed => {
                    closed = true;
                    assert_eq!(i, body.len() - 1, "closed before consuming all bytes");
                }
                super::ArgsScanOutcome::NeedMore => {
                    assert!(i < body.len() - 1, "no NeedMore expected on final byte");
                }
            }
        }
        assert!(closed);
        assert_eq!(state.args_body_end, buffer.len() - 1);
    }

    /// Direct test of the scanner state: feeding the body in two halves must
    /// produce the same result as a single pass, with `scan_offset` reflecting
    /// the work done in each call.
    #[test]
    fn gemma4_scan_args_body_is_resumable_across_chunks() {
        let mut buffer = String::from("{");
        let mut state = Gemma4ScanState {
            phase: Gemma4ScanPhase::InArgs,
            args_body_start: 1,
            scan_offset: 1,
            containers: vec![Container::Object],
            ..Gemma4ScanState::default()
        };

        // First half: opens a string but does not close the top-level object.
        buffer.push_str("a:<|\"|>val");
        let outcome = scan_args_body(&buffer, &mut state).unwrap();
        assert_eq!(outcome, super::ArgsScanOutcome::NeedMore);
        assert_eq!(state.scan_offset, buffer.len());
        assert_eq!(state.mode, ArgsScanMode::StringBody);

        // Second half: closes string then the object.
        buffer.push_str("<|\"|>}");
        let outcome = scan_args_body(&buffer, &mut state).unwrap();
        assert_eq!(outcome, super::ArgsScanOutcome::Closed);
        assert!(state.containers.is_empty());
        assert_eq!(state.args_body_end, buffer.len() - 1);
    }

    /// A structurally malformed args body fails closed: the `<|tool_call>`
    /// markup is surfaced as plain text (fail-loud, not silent-drop) and the
    /// scanner resets to `Outside`, so a *separate* later push with a valid
    /// call still parses normally.
    #[test]
    fn gemma4_malformed_call_fails_closed_then_recovers_on_later_push() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        // `]` closing the object (opened with `{`) is a structural mismatch in
        // `scan_args_body` (which runs before `parse_args_text`).
        let output = parser.parse_chunk("<|tool_call>call:get_weather{bad:]}<tool_call|>").unwrap();
        assert!(
            output.calls.is_empty(),
            "malformed call must not emit a tool call"
        );
        assert!(output.normal_text.contains("bad:]"));
        assert!(output.normal_text.contains("<|tool_call>"));
        assert_eq!(parser.scan.phase, Gemma4ScanPhase::Outside);
        assert!(parser.buffer.is_empty());

        let output = parser
            .parse_chunk("<|tool_call>call:get_weather{location:<|\"|>NY<|\"|>}<tool_call|>")
            .unwrap()
            .coalesce_calls();
        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "NY" })
        );
    }

    /// Fail-closed recovery deliberately does not try to salvage a valid call
    /// that arrives in the *same chunk* after a malformed one: the whole
    /// region is emitted as text. This trades a recoverable trailing call for
    /// the guarantee that a `<|tool_call>` embedded in a broken payload can
    /// never be promoted to a spurious tool call.
    #[test]
    fn gemma4_malformed_call_swallows_same_chunk_tail_as_text() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let output = parser
            .parse_chunk(concat!(
                "<|tool_call>call:get_weather{bad:]}<tool_call|>",
                "<|tool_call>call:get_status{}<tool_call|>",
            ))
            .unwrap();
        assert!(
            output.calls.is_empty(),
            "no call should be emitted from the bad chunk"
        );
        // Both the malformed call and the trailing valid markup come out as text.
        assert!(output.normal_text.contains("bad:]"));
        assert!(output.normal_text.contains("get_status"));
        assert_eq!(parser.scan.phase, Gemma4ScanPhase::Outside);
        assert!(parser.buffer.is_empty());
    }

    /// Outside-phase header errors (e.g. empty/whitespace tool name) recover
    /// inline within the same `parse_into`, not by looping: the broken header
    /// bytes are surfaced as text and a valid call later in the chunk still
    /// emits.
    #[test]
    fn gemma4_outside_header_error_recovers_inline() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        // Whitespace-only tool name → `gemma4_tool_name` returns `Cut`; the
        // bad header resyncs as text instead of erroring out of the parse.
        let output = parser
            .parse_chunk(concat!(
                "<|tool_call>call:   {}<tool_call|>",
                "<|tool_call>call:get_status{}<tool_call|>",
            ))
            .unwrap()
            .coalesce_calls();
        assert_eq!(first_call(&output).name.as_deref(), Some("get_status"));
        assert_eq!(parser.scan.phase, Gemma4ScanPhase::Outside);
    }

    /// A malformed call that reaches `AfterArgs` (balanced delimiters + a
    /// verified end marker, but invalid grammar) is recovered inline: it is
    /// drained and a following valid call in the same chunk still emits,
    /// without erroring out of the parse or relying on a `finish` recovery
    /// loop.
    #[test]
    fn gemma4_afterargs_error_recovers_following_call_inline() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let output = parser
            .parse_chunk(concat!(
                "<|tool_call>call:get_weather{:}<tool_call|>",
                "<|tool_call>call:get_status{}<tool_call|>",
            ))
            .unwrap()
            .coalesce_calls();
        assert_eq!(first_call(&output).name.as_deref(), Some("get_status"));
    }

    /// When `parse_args_text` fails inside `AfterArgs`, the verified
    /// `<tool_call|>` end marker must be drained along with the bad body
    /// rather than leaking as normal text.
    #[test]
    fn gemma4_parse_args_error_does_not_leak_end_marker_as_text() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        // Args body has balanced braces (so `scan_args_body` succeeds) but
        // is not valid Gemma4 grammar (the empty key fails `gemma4_key`).
        // Append trailing text after the end marker so we can prove that
        // the drain stopped *exactly* at the end marker rather than
        // accidentally clearing it via overshoot.
        let output = parser
            .parse_chunk("<|tool_call>call:get_weather{:}<tool_call|>trailing")
            .unwrap();
        assert!(output.calls.is_empty());
        assert!(!output.normal_text.contains("<tool_call|>"));
        assert!(output.normal_text.contains("trailing"));
    }

    /// Regression for normal text preceding a malformed call: text already
    /// accumulated earlier in the same chunk must not be lost when a later
    /// call in the chunk fails to parse. (Previously the parse returned `Err`
    /// and the caller-visible result — including that text — was discarded.)
    #[test]
    fn gemma4_text_before_malformed_call_is_not_lost() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let output = parser
            .parse_chunk("prefix <|tool_call>call:get_weather{:}<tool_call|> suffix")
            .unwrap();
        assert!(output.calls.is_empty());
        assert!(output.normal_text.contains("prefix "));
        assert!(output.normal_text.contains(" suffix"));
        assert!(!output.normal_text.contains("<tool_call|>"));
    }

    /// A chunk full of back-to-back structurally malformed calls fails closed
    /// without emitting any tool call or looping: the first scan error surfaces
    /// the whole buffered region as text and clears it.
    #[test]
    fn gemma4_many_consecutive_malformed_calls_fail_closed() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let output = parser
            .parse_chunk(concat!(
                "<|tool_call>call:get_weather{a:]}<tool_call|>",
                "<|tool_call>call:get_weather{b:]}<tool_call|>",
                "<|tool_call>call:get_weather{c:]}<tool_call|>",
            ))
            .unwrap();

        assert!(output.calls.is_empty(), "no malformed call should emit");
        // All three malformed bodies come out as text.
        assert!(output.normal_text.contains("a:]"));
        assert!(output.normal_text.contains("b:]"));
        assert!(output.normal_text.contains("c:]"));
        assert_eq!(parser.scan.phase, Gemma4ScanPhase::Outside);
        assert!(parser.buffer.is_empty());

        // finish() is a no-op on the empty buffer.
        let tail = parser.finish().unwrap();
        assert!(tail.calls.is_empty());
    }

    /// Quoted braces inside a `<|"|>` string must not affect depth tracking.
    #[test]
    fn gemma4_scan_args_body_ignores_braces_inside_string() {
        let mut buffer = String::from("{");
        let mut state = Gemma4ScanState {
            phase: Gemma4ScanPhase::InArgs,
            args_body_start: 1,
            scan_offset: 1,
            containers: vec![Container::Object],
            ..Gemma4ScanState::default()
        };
        buffer.push_str("k:<|\"|>}}}}}<|\"|>}");

        let outcome = scan_args_body(&buffer, &mut state).unwrap();

        assert_eq!(outcome, super::ArgsScanOutcome::Closed);
        assert!(state.containers.is_empty());
    }

    /// Oracle test: the incremental scanner plus `parse_args_text` must agree
    /// with the full `gemma4_args` grammar on bodies where bare keys/values
    /// legally contain structural bytes (`{`, `[`, `]`) or the `<|"|>`
    /// delimiter — exactly the inputs the previous depth-counting scanner
    /// mishandled. For each body, a full Gemma4 call is streamed end-to-end and
    /// its parsed arguments are compared against the grammar applied directly
    /// to the same body text.
    #[test]
    fn gemma4_scanner_matches_full_grammar_on_pathological_bare_values() {
        let bodies = [
            "key:a<|\"|>b",                // bare value containing one delimiter
            "key:a<|\"|>b<|\"|>c",         // bare value containing two delimiters
            "key:a{b",                     // bare value containing `{`
            "key:a[b",                     // bare value containing `[`
            "key:a]b",                     // bare value: `]` actually terminates it
            "a:1,b:<|\"|>x<|\"|>",         // mixed bare scalar + string
            "outer:{inner:a{b}",           // nested object, brace inside bare value
            "list:[a{b,c]",                // array, brace inside a bare element
            "s:<|\"|>has{}[]braces<|\"|>", // string containing structural bytes
            "k:<|\"|>v<|\"|>,",            // trailing comma after a string value
        ];

        for body in bodies {
            let oracle = parse_gemma4_args(body);
            let call = format!("<|tool_call>call:get_weather{{{body}}}<tool_call|>");
            let output = collect_stream(&[&call]);

            match oracle {
                Ok(expected) => {
                    assert_eq!(output.calls.len(), 1, "body `{body}` should emit one call");
                    assert_eq!(
                        serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
                        Value::Object(expected),
                        "streamed args for body `{body}` must match the full grammar",
                    );
                }
                Err(_) => {
                    assert!(
                        output.calls.is_empty(),
                        "body `{body}` is rejected by the grammar; streaming must not emit a call",
                    );
                }
            }
        }
    }

    /// Documents the known limitation: a single forward scan cannot reproduce
    /// `gemma4_value`'s `alt` backtracking, so bodies whose value begins with
    /// an unmatched `{`/`[`/`<|"|>` — and the `{,}` empty-object form — are
    /// reported as incomplete even though the full grammar accepts them. The
    /// safety-critical guarantee still holds: no tool call is emitted. If a
    /// future change adds backtracking support, update this test.
    #[test]
    fn gemma4_known_backtracking_limitation_drops_pathological_bodies() {
        let bodies = [
            "a:{",           // grammar: {"a":"{"} (bare `{`); scanner: incomplete
            "a:[",           // grammar: {"a":"["}
            "a:<|\"|>",      // grammar: {"a":"<|\"|>"} (lone delimiter, bare)
            ",",             // grammar: {} (empty object via stray comma)
            "<|\"|>:<|\"|>", // grammar: {"<|\"|>":"<|\"|>"}
            "}:1",           // grammar: {"}":1} (a key may begin with `}`)
        ];

        for body in bodies {
            assert!(
                parse_gemma4_args(body).is_ok(),
                "precondition: grammar accepts `{body}`",
            );
            let call = format!("<|tool_call>call:get_weather{{{body}}}<tool_call|>");
            // The body is dropped either by failing closed to text (parse Ok,
            // no call) or by being reported incomplete (finish Err); both are
            // acceptable, what matters is that no tool call is ever emitted.
            let mut parser = Gemma4ToolParser::new(&test_tools());
            let mut calls = 0;
            if let Ok(output) = parser.parse_chunk(&call) {
                calls += output.calls.len();
            }
            if let Ok(output) = parser.finish() {
                calls += output.calls.len();
            }
            assert_eq!(
                calls, 0,
                "known limitation: body `{body}` must not emit a call"
            );
        }
    }
}
