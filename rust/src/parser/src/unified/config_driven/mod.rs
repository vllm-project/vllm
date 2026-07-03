//! Config-driven unified parser: a shared boundary state machine parameterized
//! by a per-model [`ParserFormat`] implemented on zero-sized marker types.
//!
//! The engine owns a fixed five-state boundary FSM (plus a terminal `Done`
//! state) and all streaming mechanics: buffering, partial-marker holdback,
//! event application, raw-text reconstruction for [`UnifiedParser::reset`],
//! and end-of-stream semantics. A format contributes only:
//!
//! - boundary marker literals (with `Option` encoding which constructs exist),
//! - a winnow grammar for the tool-call header and the complete argument body,
//! - a resumable scanner locating the end of the raw argument body.
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
use winnow::error::{ErrMode, ModalResult};
use winnow::prelude::*;
use winnow::stream::{Partial, Stream};
use winnow::token::{literal, rest};

use super::{Result, UnifiedParser, UnifiedParserError, UnifiedParserOutput};
use crate::reasoning::last_reasoning_boundary;
use crate::tool::{Tool, ToolCallDelta};
use crate::unified::parsing_failed;
use crate::utils::{MarkerScanState, parse_buffered_event, safe_text_len_mul, take_until_marker};

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

    /// Wrapper markers around consecutive tool calls (e.g. Qwen3's
    /// `<tool_call>`/`</tool_call>` around `<function=...>` blocks), or `None`
    /// when each `CALL_START`..`CALL_END` stands alone.
    const WRAPPER: Option<(&'static str, &'static str)> = None;

    /// Swallow all output after the wrapper closes, like the `IgnoredRest`
    /// event in standalone parsers. Only meaningful with `WRAPPER`.
    const SUPPRESS_TEXT_AFTER_WRAPPER: bool = false;

    /// Token texts of the reasoning boundary markers, used only to derive the
    /// initial parser state from prompt token IDs (never for stream matching).
    const REASONING_BOUNDARY_TOKENS: Option<(&'static str, &'static str)> = None;

    /// Whether decoded output must keep tokenizer special tokens.
    const PRESERVE_SPECIAL_TOKENS: bool = true;

    /// Resumable scanner locating `CALL_END` for the raw argument body.
    type ArgsScan: ArgsEndScan;

    /// Parse the tool-call header after `CALL_START`: the tool name plus
    /// everything up to the start of the raw argument body.
    fn tool_header(input: &mut Input<'_>) -> ModalResult<String>;

    /// Parse one complete raw argument body (end marker already stripped)
    /// into a JSON object.
    fn tool_args(body: &str) -> ModalResult<Map<String, Value>>;
}

/// Resumable scanner that locates a tool-call end marker in buffered input.
///
/// On success the scanner consumes through the end marker and returns the
/// body before it. On incomplete input it must record a resume position so
/// repeated scans stay linear in the argument length.
pub trait ArgsEndScan: Default + Send {
    /// Scan buffered `input` for `end`, resuming from prior progress.
    fn scan<'i>(&mut self, input: &mut Input<'i>, end: &str) -> ModalResult<&'i str>;
}

/// Default argument scanner: a plain resumable marker scan with no lexical
/// structure awareness.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlainArgsScan(MarkerScanState);

impl ArgsEndScan for PlainArgsScan {
    fn scan<'i>(&mut self, input: &mut Input<'i>, end: &str) -> ModalResult<&'i str> {
        let body = take_until_marker(end, &mut self.0).parse_next(input)?;
        input.next_slice(end.len());
        Ok(body)
    }
}

/// Boundary marker recognized by the fixed transition graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Boundary {
    ReasoningStart,
    ReasoningEnd,
    WrapperStart,
    WrapperEnd,
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
    /// A complete tool call with parsed arguments.
    CallComplete { args: Map<String, Value> },
}

/// Engine parsing state. The state set and transition graph are fixed;
/// `Option`-typed format constants decide which edges are reachable.
enum Mode<F: ParserFormat> {
    /// Visible assistant text.
    Text,
    /// Inside a reasoning block.
    Reasoning,
    /// Inside the wrapper, between tool calls. Unreachable without `WRAPPER`.
    ToolBetween,
    /// After `CALL_START`, parsing the tool-call header.
    ToolHeader,
    /// Inside the raw argument body, scanning for `CALL_END`.
    ToolArgs {
        name: String,
        /// Raw header text consumed by `tool_header`, kept for `reset()`.
        raw_header: String,
        scan: F::ArgsScan,
    },
    /// Wrapper closed with `SUPPRESS_TEXT_AFTER_WRAPPER`: swallow the rest.
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
        // wrapper opener when the format has one, the call start otherwise.
        let tool_entry = match F::WRAPPER {
            Some((wrapper_start, _)) => (wrapper_start, Boundary::WrapperStart),
            None => (F::CALL_START, Boundary::CallStart),
        };
        let reasoning_start = F::REASONING_START.map(|marker| (marker, Boundary::ReasoningStart));
        // Tool entry inside reasoning implicitly ends it: the transition to a
        // tool state simply stops routing text to the reasoning channel.
        let reasoning_end = F::REASONING_END.map(|marker| (marker, Boundary::ReasoningEnd));
        let between = F::WRAPPER.map(|(_, wrapper_end)| {
            [
                (F::CALL_START, Boundary::CallStart),
                (wrapper_end, Boundary::WrapperEnd),
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
    emitted_tool_count: usize,
    tokenizer: DynTokenizer,
    /// Resolved reasoning boundary token IDs for prompt-state derivation.
    boundary_ids: Option<(u32, u32)>,
    _format: PhantomData<fn() -> F>,
}

impl<F: ParserFormat> ConfigDrivenParser<F> {
    /// Create a parser for one request stream.
    pub fn new(_tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
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
                Boundary::WrapperStart => self.mode = Mode::ToolBetween,
                Boundary::WrapperEnd => {
                    self.mode = if F::SUPPRESS_TEXT_AFTER_WRAPPER {
                        Mode::Done
                    } else {
                        Mode::Text
                    };
                }
                Boundary::CallStart => self.mode = Mode::ToolHeader,
            },
            Event::CallHeader { name } => {
                self.mode = Mode::ToolArgs {
                    name,
                    raw_header: self.buffer[..consumed_len].to_string(),
                    scan: F::ArgsScan::default(),
                };
            }
            Event::CallComplete { args } => {
                let mode = std::mem::replace(&mut self.mode, Mode::Text);
                let Mode::ToolArgs { name, .. } = mode else {
                    return Err(parsing_failed!(
                        "{} arguments without an active tool call",
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
                self.mode = if F::WRAPPER.is_some() {
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
            // The wrapper opener was already consumed alongside previously
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
                parse_next_event::<F>(input, &mut self.mode, &self.markers)
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
            // All calls were emitted; an unclosed wrapper at end of stream is
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
) -> ModalResult<Event> {
    match mode {
        Mode::Text => boundary_or_content(input, &markers.text, Event::Text),
        Mode::Reasoning => boundary_or_content(input, &markers.reasoning, Event::Reasoning),
        Mode::ToolBetween => boundary_or_content(input, &markers.between, Event::Ignored),
        Mode::ToolHeader => F::tool_header(input).map(|name| Event::CallHeader { name }),
        Mode::ToolArgs { scan, .. } => args_event::<F>(input, scan),
        Mode::Done => rest.value(Event::Ignored).parse_next(input),
    }
}

/// Parse complete tool-call arguments once the end marker is located.
fn args_event<F: ParserFormat>(
    input: &mut Input<'_>,
    scan: &mut F::ArgsScan,
) -> ModalResult<Event> {
    let body = scan.scan(input, F::CALL_END)?;
    let args = F::tool_args(body)?;
    Ok(Event::CallComplete { args })
}

/// Match one of the mode's marker candidates at the cursor (in order), or
/// emit a safe content run held back before the earliest possible marker.
fn boundary_or_content(
    input: &mut Input<'_>,
    markers: &ModeMarkers,
    content: Event,
) -> ModalResult<Event> {
    // Engine invariant: every reachable mode watches at least one marker
    // (`ToolBetween` without `WRAPPER` is unreachable).
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
