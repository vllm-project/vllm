//! Config-driven unified parser: a shared boundary state machine parameterized
//! by a per-model [`ParserFormat`] implemented on zero-sized marker types.
//!
//! The engine owns a fixed five-state boundary FSM (plus a terminal `Done`
//! state) and all streaming mechanics: buffering, partial-marker holdback,
//! event application, raw-text reconstruction for [`UnifiedParser::reset`],
//! and end-of-stream semantics. A format contributes only:
//!
//! - boundary marker literals (with `Option` encoding which constructs exist),
//! - a winnow grammar for the tool-call header and, for custom argument
//!   syntaxes, the complete argument body,
//! - a resumable scanner that delimits the raw argument body and chooses its
//!   channel ([`ArgsStep`]): streamed verbatim for JSON-native bodies, or
//!   buffered for one-shot conversion.
//!
//! Formats never define states or transitions. A format whose behavior does
//! not fit the fixed graph plus the small policy surface below should be
//! written as a standalone parser instead of growing this engine.
//!
//! Because every [`ParserFormat`] item is an associated constant or a static
//! method, a model parser is a zero-sized marker type plus a type alias in its
//! own format module (see [`gemma4`]):
//!
//! ```ignore
//! pub struct Gemma4Format;
//! impl ParserFormat for Gemma4Format { /* consts + two grammar fns */ }
//! pub type Gemma4ConfigDrivenParser = ConfigDrivenParser<Gemma4Format>;
//! ```
//!
//! Monomorphization then compiles `ConfigDrivenParser<F>` down to the
//! equivalent of a hand-written per-model parser: literals are inlined and
//! there is no dynamic dispatch on the format.

mod gemma4;

use std::marker::PhantomData;

pub use gemma4::{Gemma4ConfigDrivenParser, Gemma4Format};
use serde_json::{Map, Value};
use vllm_tokenizer::DynTokenizer;
use winnow::error::{ContextError, ErrMode, ModalResult};
use winnow::prelude::*;
use winnow::stream::{Partial, Stream};
use winnow::token::{literal, rest};

use super::{Result, UnifiedParser, UnifiedParserError, UnifiedParserOutput};
use crate::reasoning::last_reasoning_boundary;
use crate::tool::{Tool, ToolCallDelta, ToolSchema, ToolSchemas};
use crate::unified::parsing_failed;
use crate::utils::{
    JsonObjectScanState, MarkerScanState, parse_buffered_event, safe_text_len_mul,
    take_json_object, take_until_marker,
};

/// Partial streaming input over buffered decoded text.
pub type Input<'i> = Partial<&'i str>;

/// Per-model format description consumed by [`ConfigDrivenParser`].
///
/// All items are associated constants or static methods, so implementors are
/// zero-sized marker types and the engine monomorphizes per format.
pub trait ParserFormat: 'static {
    /// Human-readable format name used in error messages.
    const NAME: &'static str;

    /// Reasoning block opener (e.g. `<|channel>thought\n`), or `None` when the
    /// format has no reasoning markers. Free to span multiple tokens.
    const REASONING_START: Option<&'static str>;

    /// Reasoning block closer, or `None` when the format has no reasoning.
    const REASONING_END: Option<&'static str>;

    /// Start marker of one tool call. Consumed by the engine before
    /// [`ParserFormat::tool_header`] runs.
    const CALL_START: &'static str;

    /// End marker of one tool call. Scan target of [`ParserFormat::ArgsScan`].
    const CALL_END: &'static str;

    /// Section markers around consecutive tool calls (e.g. Qwen3's
    /// `<tool_call>`/`</tool_call>` around `<function=...>` blocks), or `None`
    /// when each `CALL_START`..`CALL_END` stands alone.
    const SECTION: Option<(&'static str, &'static str)> = None;

    /// Swallow all output after the section closes, like the `IgnoredRest`
    /// event in standalone parsers. Only meaningful with `SECTION`.
    const SUPPRESS_TEXT_AFTER_SECTION: bool = false;

    /// Token texts of the reasoning boundary markers, used only to derive the
    /// initial parser state from prompt token IDs (never for stream matching).
    const REASONING_BOUNDARY_TOKENS: Option<(&'static str, &'static str)> = None;

    /// Whether decoded output must keep tokenizer special tokens.
    const PRESERVE_SPECIAL_TOKENS: bool = true;

    /// Whether to normalize the request tools' parameter JSON schemas at
    /// creation for schema-aware argument conversion. When `false`, the
    /// engine skips schema construction and `tool_args` resolves to the
    /// empty schema (all raw values keep their string-level interpretation).
    const USES_TOOL_SCHEMAS: bool = false;

    /// Resumable scanner delimiting the raw argument body.
    type ArgsScan: ArgsScan;

    /// Parse the tool-call header after `CALL_START`: the tool name plus
    /// everything up to the start of the raw argument body.
    fn tool_header(input: &mut Input<'_>) -> ModalResult<String>;

    /// Parse one complete buffered argument body (end marker already
    /// stripped) into a JSON object. Never invoked for scanners that stream
    /// ([`ArgsStep::Streamed`]).
    ///
    /// `schema` is the parameter schema resolved for this call's tool name —
    /// the empty schema unless `USES_TOOL_SCHEMAS` is enabled.
    ///
    /// The default parses the body as a JSON object. Note this normalizes
    /// the text on re-serialization; JSON-native formats that need byte
    /// fidelity should use a streaming scanner such as [`JsonArgsScan`]
    /// instead, which bypasses this method entirely.
    fn tool_args(schema: &ToolSchema, body: &str) -> ModalResult<Map<String, Value>> {
        let _ = schema;
        serde_json::from_str(body).map_err(|_| ErrMode::Cut(ContextError::new()))
    }
}

/// One step of argument scanning, choosing the call's argument channel.
///
/// - incremental `Streamed`: JSON-native argument bodies (e.g. Kimi K2);
/// - one final `Streamed` (`fragment` = whole body, `complete: true`):
///   validate-first formats that buffer, then pass through verbatim;
/// - `Buffered`: custom argument syntaxes converted by
///   [`ParserFormat::tool_args`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgsStep<'i> {
    /// Raw-JSON channel: the argument body is already wire-format JSON, so
    /// `fragment` is emitted verbatim as an argument delta right away.
    ///
    /// - `fragment` must be the consumed prefix of this step's input.
    /// - `complete` ends the call; the scanner has then also consumed the end
    ///   marker.
    Streamed { fragment: &'i str, complete: bool },
    /// Converted channel: the complete raw body of a custom argument syntax,
    /// handed to [`ParserFormat::tool_args`] exactly once. The scanner has
    /// consumed through the end marker.
    Buffered { body: &'i str },
}

/// Resumable scanner that delimits one call's raw argument body.
///
/// On incomplete input it must record a resume position so repeated scans
/// stay linear in the argument length. A scanner statically commits to one
/// [`ArgsStep`] channel; mixing them within a call is a protocol violation.
pub trait ArgsScan: Default + Send {
    /// Advance the scan over buffered `input`, resuming from prior progress.
    // TODO(bugen): can we simply retrieve the length from offset of `input`?
    fn scan<'i>(&mut self, input: &mut Input<'i>, end: &str) -> ModalResult<ArgsStep<'i>>;
}

/// Default argument scanner: a plain resumable marker scan with no lexical
/// structure awareness, buffering the complete body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlainArgsScan(MarkerScanState);

impl ArgsScan for PlainArgsScan {
    fn scan<'i>(&mut self, input: &mut Input<'i>, end: &str) -> ModalResult<ArgsStep<'i>> {
        let body = take_until_marker(end, &mut self.0).parse_next(input)?;
        input.next_slice(end.len());
        Ok(ArgsStep::Buffered { body })
    }
}

/// Streaming scanner for JSON-native argument bodies: emits raw fragments as
/// the top-level JSON object is scanned (lexically, string-aware), then
/// consumes the end marker once the object closes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JsonArgsScan(JsonObjectScanState);

impl ArgsScan for JsonArgsScan {
    fn scan<'i>(&mut self, input: &mut Input<'i>, end: &str) -> ModalResult<ArgsStep<'i>> {
        if self.0.complete() {
            let _matched: &str = literal(end).parse_next(input)?;
            return Ok(ArgsStep::Streamed {
                fragment: "",
                complete: true,
            });
        }
        let text = **input;
        let len = take_json_object(input, &mut self.0)?;
        Ok(ArgsStep::Streamed {
            fragment: &text[..len],
            complete: false,
        })
    }
}

/// Boundary marker recognized by the fixed transition graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Boundary {
    ReasoningStart,
    ReasoningEnd,
    SectionStart,
    SectionEnd,
    CallStart,
}

/// One parsed engine event over buffered streaming input.
#[derive(Debug, Clone, PartialEq)]
enum Event {
    /// A safe run of visible text (value is the consumed buffer prefix).
    Text,
    /// A safe run of reasoning text (value is the consumed buffer prefix).
    Reasoning,
    /// Consumed input that produces no output (between-call noise, suppressed
    /// trailing text).
    Ignored,
    /// A boundary marker driving a fixed state transition.
    Boundary(Boundary),
    /// A complete tool-call header.
    CallHeader { name: String },
    /// A raw streamed argument fragment (value is the leading `len` bytes of
    /// the consumed buffer prefix); `complete` ends the call.
    ArgsFragment { len: usize, complete: bool },
    /// A complete buffered tool call with converted arguments.
    CallComplete { args: Map<String, Value> },
}

/// Engine parsing state. The state set and transition graph are fixed;
/// `Option`-typed format constants decide which edges are reachable.
enum Mode<F: ParserFormat> {
    /// Visible assistant text.
    Text,
    /// Inside a reasoning block.
    Reasoning,
    /// Inside the section, between tool calls. Unreachable without `SECTION`.
    ToolBetween,
    /// After `CALL_START`, parsing the tool-call header.
    ToolHeader,
    /// Inside the raw argument body, scanning for `CALL_END`.
    ToolArgs {
        /// Tool name from the header; taken by the first emitted delta.
        name: Option<String>,
        /// Raw header text consumed by `tool_header`, kept for `reset()`.
        raw_header: String,
        scan: F::ArgsScan,
    },
    /// Section closed with `SUPPRESS_TEXT_AFTER_SECTION`: swallow the rest.
    Done,
}

/// Boundary marker candidates watched in one engine mode, precomputed from
/// the format constants so per-event parsing builds nothing.
struct ModeMarkers {
    /// Marker texts tried in order; also the safe-content fallback scan set.
    texts: Vec<&'static str>,
    /// Transition triggered by each marker, index-aligned with `texts`.
    boundaries: Vec<Boundary>,
}

impl ModeMarkers {
    fn new(candidates: impl IntoIterator<Item = (&'static str, Boundary)>) -> Self {
        let (texts, boundaries) = candidates.into_iter().unzip();
        Self { texts, boundaries }
    }
}

/// Per-mode marker candidates derived once from a [`ParserFormat`].
struct ModeMarkerSet {
    text: ModeMarkers,
    reasoning: ModeMarkers,
    between: ModeMarkers,
}

impl ModeMarkerSet {
    fn of<F: ParserFormat>() -> Self {
        // The tool-entry marker watched in text and reasoning modes: the
        // section opener when the format has one, the call start otherwise.
        let tool_entry = match F::SECTION {
            Some((section_start, _)) => (section_start, Boundary::SectionStart),
            None => (F::CALL_START, Boundary::CallStart),
        };
        let reasoning_start = F::REASONING_START.map(|marker| (marker, Boundary::ReasoningStart));
        // Tool entry inside reasoning implicitly ends it: the transition to a
        // tool state simply stops routing text to the reasoning channel.
        let reasoning_end = F::REASONING_END.map(|marker| (marker, Boundary::ReasoningEnd));
        let between = F::SECTION.map(|(_, section_end)| {
            [
                (F::CALL_START, Boundary::CallStart),
                (section_end, Boundary::SectionEnd),
            ]
        });

        Self {
            text: ModeMarkers::new(reasoning_start.into_iter().chain([tool_entry])),
            reasoning: ModeMarkers::new(reasoning_end.into_iter().chain([tool_entry])),
            between: ModeMarkers::new(between.into_iter().flatten()),
        }
    }
}

/// Unified parser driven by a [`ParserFormat`] marker type.
///
/// Owns all streaming state; the format contributes constants and grammar.
pub struct ConfigDrivenParser<F: ParserFormat> {
    buffer: String,
    mode: Mode<F>,
    markers: ModeMarkerSet,
    tool_schemas: ToolSchemas,
    emitted_tool_count: usize,
    tokenizer: DynTokenizer,
    /// Resolved reasoning boundary token IDs for prompt-state derivation.
    boundary_ids: Option<(u32, u32)>,
    _format: PhantomData<fn() -> F>,
}

impl<F: ParserFormat> ConfigDrivenParser<F> {
    /// Create a parser for one request stream.
    pub fn new(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let tool_schemas = if F::USES_TOOL_SCHEMAS {
            ToolSchemas::from_tools(tools)
        } else {
            ToolSchemas::default()
        };

        let boundary_ids = match F::REASONING_BOUNDARY_TOKENS {
            Some((start, end)) => {
                let start_id = tokenizer.token_to_id(start).ok_or_else(|| {
                    UnifiedParserError::MissingToken {
                        token: start.to_string(),
                    }
                })?;
                let end_id =
                    tokenizer.token_to_id(end).ok_or_else(|| UnifiedParserError::MissingToken {
                        token: end.to_string(),
                    })?;
                Some((start_id, end_id))
            }
            None => None,
        };

        Ok(Self {
            buffer: String::new(),
            mode: Mode::Text,
            markers: ModeMarkerSet::of::<F>(),
            tool_schemas,
            emitted_tool_count: 0,
            tokenizer,
            boundary_ids,
            _format: PhantomData,
        })
    }

    fn apply_event(
        &mut self,
        event: Event,
        consumed_len: usize,
        output: &mut UnifiedParserOutput,
    ) -> Result<()> {
        match event {
            Event::Text => output.push_text(self.buffer[..consumed_len].to_string()),
            Event::Reasoning => output.push_reasoning(self.buffer[..consumed_len].to_string()),
            Event::Ignored => {}
            Event::Boundary(boundary) => match boundary {
                Boundary::ReasoningStart => self.mode = Mode::Reasoning,
                Boundary::ReasoningEnd => self.mode = Mode::Text,
                Boundary::SectionStart => self.mode = Mode::ToolBetween,
                Boundary::SectionEnd => {
                    self.mode = if F::SUPPRESS_TEXT_AFTER_SECTION {
                        Mode::Done
                    } else {
                        Mode::Text
                    };
                }
                Boundary::CallStart => self.mode = Mode::ToolHeader,
            },
            Event::CallHeader { name } => {
                self.mode = Mode::ToolArgs {
                    name: Some(name),
                    raw_header: self.buffer[..consumed_len].to_string(),
                    scan: F::ArgsScan::default(),
                };
            }
            Event::ArgsFragment { len, complete } => {
                let Mode::ToolArgs { name, .. } = &mut self.mode else {
                    return Err(parsing_failed!(
                        "{} argument fragment without an active tool call",
                        F::NAME
                    ));
                };
                let name = name.take();
                if name.is_some() || len > 0 {
                    output.push_call(ToolCallDelta {
                        tool_index: self.emitted_tool_count,
                        name,
                        arguments: self.buffer[..len].to_string(),
                    });
                }
                if complete {
                    self.emitted_tool_count += 1;
                    self.mode = if F::SECTION.is_some() {
                        Mode::ToolBetween
                    } else {
                        Mode::Text
                    };
                }
            }
            Event::CallComplete { args } => {
                let mode = std::mem::replace(&mut self.mode, Mode::Text);
                let Mode::ToolArgs {
                    name: Some(name), ..
                } = mode
                else {
                    return Err(parsing_failed!(
                        "{} buffered arguments without an active tool call",
                        F::NAME
                    ));
                };
                let arguments = serde_json::to_string(&args)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                output.push_call(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_tool_count += 1;
                self.mode = if F::SECTION.is_some() {
                    Mode::ToolBetween
                } else {
                    Mode::Text
                };
            }
        }
        Ok(())
    }

    fn initialize_mode(&mut self, prompt_token_ids: &[u32]) {
        let in_reasoning = self.boundary_ids.and_then(|(start_id, end_id)| {
            last_reasoning_boundary(prompt_token_ids, start_id, end_id, self.tokenizer.as_ref())
        });
        self.mode = match in_reasoning {
            Some(true) => Mode::Reasoning,
            Some(false) | None => Mode::Text,
        };
    }

    fn reset(&mut self) -> String {
        let raw = match std::mem::replace(&mut self.mode, Mode::Text) {
            Mode::Text | Mode::Done => std::mem::take(&mut self.buffer),
            Mode::Reasoning => {
                // Reasoning mode is only reachable through this marker (or a
                // prompt already inside reasoning, where there is no marker
                // to restore; the opener default keeps reset lossless for the
                // common in-stream case).
                let start = F::REASONING_START.unwrap_or("");
                format!("{}{}", start, std::mem::take(&mut self.buffer))
            }
            // The section opener was already consumed alongside previously
            // emitted calls, so only the unconsumed buffer remains.
            Mode::ToolBetween => std::mem::take(&mut self.buffer),
            Mode::ToolHeader => {
                format!("{}{}", F::CALL_START, std::mem::take(&mut self.buffer))
            }
            Mode::ToolArgs { raw_header, .. } => {
                format!(
                    "{}{}{}",
                    F::CALL_START,
                    raw_header,
                    std::mem::take(&mut self.buffer)
                )
            }
        };
        self.mode = Mode::Text;
        self.emitted_tool_count = 0;
        raw
    }
}

impl<F: ParserFormat> UnifiedParser for ConfigDrivenParser<F> {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Self::new(tools, tokenizer).map(|parser| Box::new(parser) as Box<dyn UnifiedParser>)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.buffer.clear();
        self.emitted_tool_count = 0;
        self.initialize_mode(prompt_token_ids);
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        F::PRESERVE_SPECIAL_TOKENS
    }

    fn parse_into(&mut self, chunk: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = {
            parse_buffered_event(&self.buffer, |input| {
                parse_next_event::<F>(input, &mut self.mode, &self.markers, &self.tool_schemas)
            })?
        } {
            self.apply_event(event, consumed_len, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();

        match &self.mode {
            Mode::Text => output.push_text(std::mem::take(&mut self.buffer)),
            Mode::Reasoning => output.push_reasoning(std::mem::take(&mut self.buffer)),
            // All calls were emitted; an unclosed section at end of stream is
            // tolerated and any buffered between-call noise is dropped.
            Mode::ToolBetween | Mode::Done => {}
            Mode::ToolHeader | Mode::ToolArgs { .. } => {
                return Err(parsing_failed!("incomplete {} tool call", F::NAME));
            }
        }

        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        ConfigDrivenParser::reset(self)
    }
}

/// Parse one engine event from buffered streaming input.
///
/// In `ToolBetween`, non-marker text is separator noise (typically
/// whitespace) and is consumed without output.
fn parse_next_event<F: ParserFormat>(
    input: &mut Input<'_>,
    mode: &mut Mode<F>,
    markers: &ModeMarkerSet,
    schemas: &ToolSchemas,
) -> ModalResult<Event> {
    match mode {
        Mode::Text => boundary_or_content(input, &markers.text, Event::Text),
        Mode::Reasoning => boundary_or_content(input, &markers.reasoning, Event::Reasoning),
        Mode::ToolBetween => boundary_or_content(input, &markers.between, Event::Ignored),
        Mode::ToolHeader => F::tool_header(input).map(|name| Event::CallHeader { name }),
        Mode::ToolArgs { name, scan, .. } => {
            // A streamed call takes `name` with its first fragment; the schema
            // is only consulted on the buffered path, where `name` is intact.
            let schema = schemas.resolve(name.as_deref().unwrap_or(""));
            args_event::<F>(input, scan, schema)
        }
        Mode::Done => rest.value(Event::Ignored).parse_next(input),
    }
}

/// Parse the next argument event: a raw streamed fragment, or the complete
/// buffered body converted through [`ParserFormat::tool_args`].
fn args_event<F: ParserFormat>(
    input: &mut Input<'_>,
    scan: &mut F::ArgsScan,
    schema: &ToolSchema,
) -> ModalResult<Event> {
    match scan.scan(input, F::CALL_END)? {
        ArgsStep::Streamed { fragment, complete } => Ok(Event::ArgsFragment {
            len: fragment.len(),
            complete,
        }),
        ArgsStep::Buffered { body } => {
            let args = F::tool_args(schema, body)?;
            Ok(Event::CallComplete { args })
        }
    }
}

/// Match one of the mode's marker candidates at the cursor (in order), or
/// emit a safe content run held back before the earliest possible marker.
fn boundary_or_content(
    input: &mut Input<'_>,
    markers: &ModeMarkers,
    content: Event,
) -> ModalResult<Event> {
    // Engine invariant: every reachable mode watches at least one marker
    // (`ToolBetween` without `SECTION` is unreachable).
    debug_assert!(!markers.texts.is_empty());
    for (marker, boundary) in markers.texts.iter().zip(&markers.boundaries) {
        let matched: ModalResult<&str> = literal(*marker).parse_next(input);
        match matched {
            Ok(_) => return Ok(Event::Boundary(*boundary)),
            Err(ErrMode::Incomplete(needed)) => return Err(ErrMode::Incomplete(needed)),
            Err(_) => {}
        }
    }

    safe_text_len_mul(input, &markers.texts).map(|_| content)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_tokenizer::test_utils::TestTokenizer;
    use winnow::combinator::seq;
    use winnow::error::ModalResult;
    use winnow::prelude::*;
    use winnow::token::{literal, take_until};

    use super::{ConfigDrivenParser, Input, JsonArgsScan, ParserFormat};
    use crate::unified::{UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};

    /// Minimal JSON-native test format: `<jtool:NAME>{...}</jtool>`.
    ///
    /// Uses the streaming [`JsonArgsScan`] and the default `tool_args`
    /// (which is never invoked on the streamed channel).
    struct TestJsonFormat;

    impl ParserFormat for TestJsonFormat {
        const NAME: &'static str = "TestJson";
        const REASONING_START: Option<&'static str> = None;
        const REASONING_END: Option<&'static str> = None;
        const CALL_START: &'static str = "<jtool:";
        const CALL_END: &'static str = "</jtool>";

        type ArgsScan = JsonArgsScan;

        fn tool_header(input: &mut Input<'_>) -> ModalResult<String> {
            let (name,): (&str,) = seq!(take_until(1.., ">"), _: literal(">")).parse_next(input)?;
            Ok(name.to_string())
        }
    }

    type TestJsonParser = ConfigDrivenParser<TestJsonFormat>;

    fn test_parser() -> TestJsonParser {
        TestJsonParser::new(&[], Arc::new(TestTokenizer::new())).unwrap()
    }

    fn parse_stream(chunks: &[&str]) -> UnifiedParserOutput {
        let mut parser = test_parser();
        let mut output = UnifiedParserOutput::default();
        for chunk in chunks {
            let mut step = UnifiedParserOutput::default();
            parser.parse_into(chunk, &mut step).unwrap();
            output.append(step);
        }
        output.append(parser.finish().unwrap());
        output
    }

    fn split_chunks(text: &str, size: usize) -> Vec<&str> {
        let mut chunks = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let mut end = size.min(rest.len());
            while !rest.is_char_boundary(end) {
                end += 1;
            }
            let (chunk, tail) = rest.split_at(end);
            chunks.push(chunk);
            rest = tail;
        }
        chunks
    }

    fn normal_text(output: &UnifiedParserOutput) -> String {
        output
            .events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Merge streamed tool-call deltas by tool index.
    fn merged_calls(output: &UnifiedParserOutput) -> Vec<(usize, Option<String>, String)> {
        let mut calls: Vec<(usize, Option<String>, String)> = Vec::new();
        for event in &output.events {
            let UnifiedParserEvent::ToolCall(call) = event else {
                continue;
            };
            match calls.iter_mut().find(|(index, ..)| *index == call.tool_index) {
                Some((_, name, arguments)) => {
                    if name.is_none() {
                        *name = call.name.clone();
                    }
                    arguments.push_str(&call.arguments);
                }
                None => calls.push((call.tool_index, call.name.clone(), call.arguments.clone())),
            }
        }
        calls
    }

    /// Fidelity-sensitive JSON: exponent and trailing-zero number forms plus
    /// irregular whitespace, all of which a parse/re-serialize would destroy.
    const FIDELITY_JSON: &str = "{\"b\": 1e2,  \"a\": \"x\",\n  \"n\": 0.10}";

    #[test]
    fn json_args_pass_through_verbatim() {
        let input = format!("Check: <jtool:get_weather>{FIDELITY_JSON}</jtool> done");
        let output = parse_stream(&[&input]);

        assert_eq!(normal_text(&output), "Check:  done");
        let calls = merged_calls(&output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1.as_deref(), Some("get_weather"));
        assert_eq!(calls[0].2, FIDELITY_JSON);
    }

    #[test]
    fn json_args_stream_incrementally_with_name_on_first_delta() {
        let input = format!("<jtool:search>{FIDELITY_JSON}</jtool>");
        let chunks = split_chunks(&input, 5);
        let output = parse_stream(&chunks);

        let deltas: Vec<_> = output
            .events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect();
        assert!(deltas.len() > 1, "expected streamed argument deltas");
        assert_eq!(deltas[0].name.as_deref(), Some("search"));
        assert!(deltas[1..].iter().all(|delta| delta.name.is_none()));

        let calls = merged_calls(&output);
        assert_eq!(calls[0].2, FIDELITY_JSON);
    }

    #[test]
    fn json_args_emit_before_call_completes() {
        let mut parser = test_parser();
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("<jtool:search>{\"a\": \"lo", &mut output).unwrap();

        let calls = merged_calls(&output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1.as_deref(), Some("search"));
        assert!(calls[0].2.starts_with("{\"a\""));

        let mut rest = UnifiedParserOutput::default();
        parser.parse_into("ng\"}</jtool>", &mut rest).unwrap();
        output.append(rest);
        output.append(parser.finish().unwrap());
        assert_eq!(merged_calls(&output)[0].2, "{\"a\": \"long\"}");
    }

    #[test]
    fn json_args_empty_object() {
        let output = parse_stream(&["<jtool:noop>{}</jtool>"]);
        let calls = merged_calls(&output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1.as_deref(), Some("noop"));
        assert_eq!(calls[0].2, "{}");
    }

    #[test]
    fn json_args_end_marker_literal_inside_string() {
        let json = "{\"s\": \"</jtool> inside\"}";
        let input = format!("<jtool:echo>{json}</jtool>");
        let output = parse_stream(&[&input]);

        let calls = merged_calls(&output);
        assert_eq!(calls[0].2, json);
        assert!(normal_text(&output).is_empty());
    }

    #[test]
    fn json_args_two_calls_use_distinct_indices() {
        let output = parse_stream(&["<jtool:a>{}</jtool><jtool:b>{\"k\": 1}</jtool>"]);
        let calls = merged_calls(&output);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], (0, Some("a".to_string()), "{}".to_string()));
        assert_eq!(
            calls[1],
            (1, Some("b".to_string()), "{\"k\": 1}".to_string())
        );
    }

    #[test]
    fn json_args_incomplete_errors_at_finish() {
        let mut parser = test_parser();
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("<jtool:x>{\"a\": 1", &mut output).unwrap();

        let error = parser.finish().unwrap_err();
        assert!(error.to_string().contains("incomplete TestJson tool call"));
    }

    #[test]
    fn json_args_garbage_before_end_marker_errors() {
        let mut parser = test_parser();
        let mut output = UnifiedParserOutput::default();
        let result = parser.parse_into("<jtool:x>{} junk</jtool>", &mut output);
        assert!(result.is_err());
    }
}
