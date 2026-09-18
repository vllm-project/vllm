// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser for the Muse Glimmer chat format (a Harmony dialect).
//!
//! Original Python implementations:
//! - `vllm/reasoning/muse_glimmer_reasoning_parser.py`
//! - `vllm/tool_parsers/muse_glimmer_tool_parser.py`
//!
//! One assistant turn is a sequence of channels framed by the special tokens
//! `<|start|>`, `<|message|>`, `<|eom|>`, and `<|eot|>`:
//!
//! ```text
//! <|start|>assistant to=self<|message|>...reasoning...<|eom|>
//! <|start|>assistant to=get_weather<|message|><atem:function_calls>
//! <atem:invoke name="get_weather">
//! <atem:parameter name="city">Paris</atem:parameter>
//! </atem:invoke>
//! </atem:function_calls><|eom|>
//! <|start|>assistant to=user<|message|>...final answer...<|eot|>
//! ```
//!
//! The generation prompt ends with `<|start|>assistant`, so the FIRST channel
//! header of a turn is emitted bare (e.g. ` to=self<|message|>`) without the
//! `<|start|>assistant` prefix. `to=self` bodies are reasoning, `to=user` and
//! untagged (`<|message|>`) bodies are visible content, and any other
//! recipient is a tool call whose body is ATEM XML. See the module-level docs
//! of the Python parsers for the full format contract.

mod structural_tag;

use serde_json::{Map, Value};
use vllm_tokenizer::{DecodedText, DynTokenizer};
use winnow::combinator::{alt, delimited, eof, not, opt, peek, preceded, seq, terminated};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext};
use winnow::prelude::*;
use winnow::stream::{Partial, Stream};
use winnow::token::{literal, rest, take_until, take_while};

use self::structural_tag::MUSE_GLIMMER_STRUCTURAL_TAG_BUILDER;
use super::{Result, ScopedStructuralTagBuilder, UnifiedParser, UnifiedParserOutput, token_id};
use crate::tool::{Tool, ToolCallDelta};
use crate::unified::parsing_failed;
use crate::utils::{
    MarkerScanState, incomplete, max_partial_prefix_len, parse_buffered_event, partial_prefix_len,
    safe_text_len_mul, take_until_marker_mul,
};

const START: &str = "<|start|>";
const MESSAGE: &str = "<|message|>";
const EOM: &str = "<|eom|>";
const EOT: &str = "<|eot|>";
const ASSISTANT: &str = "assistant";

const ATEM_PREFIX: &str = "<atem:";
const FUNCTION_CALLS_OPEN: &str = "<atem:function_calls>";
const FUNCTION_CALLS_CLOSE: &str = "</atem:function_calls>";
const INVOKE_OPEN: &str = "<atem:invoke";
const INVOKE_CLOSE: &str = "</atem:invoke>";
const PARAMETER_OPEN: &str = "<atem:parameter";
const PARAMETER_CLOSE: &str = "</atem:parameter>";

/// Markers that interrupt any channel body: the channel closes and the framed
/// header start (a framed header is authoritative anywhere).
const BODY_STOP_MARKERS: &[&str] = &[EOM, EOT, START];
/// Content bodies additionally stop at ATEM openers, so a tool block surfaced
/// inside a content channel can be reclassified.
const CONTENT_STOP_MARKERS: &[&str] = &[EOM, EOT, START, FUNCTION_CALLS_OPEN, INVOKE_OPEN];
/// Markers whose trailing partial fragments are held back in an open body.
const BODY_HOLD_BACK_MARKERS: &[&str] = &[EOM, EOT, START, MESSAGE];
/// Content also holds partial ATEM openers, so the markup never streams as
/// content before reclassification decides.
const CONTENT_HOLD_BACK_MARKERS: &[&str] =
    &[EOM, EOT, START, MESSAGE, FUNCTION_CALLS_OPEN, INVOKE_OPEN];
/// Markers that interrupt noise while waiting for the next channel header
/// (`<|message|>` included: a bare one opens an untagged content channel).
const IDLE_STOP_MARKERS: &[&str] = &[START, MESSAGE, EOM, EOT];
/// Markers that interrupt skippable noise inside a tool channel.
const TOOL_NOISE_MARKERS: &[&str] = &[
    EOM,
    EOT,
    START,
    FUNCTION_CALLS_OPEN,
    FUNCTION_CALLS_CLOSE,
    INVOKE_OPEN,
];
/// Markers that end an invoke body: its close, or a complete framing marker
/// (real ATEM bodies never contain framing, so the `<|eom|>`/`<|eot|>` must be
/// allowed to close the channel); a bare `<|` that is no marker is body text
/// (Python parity).
const INVOKE_BODY_STOP_MARKERS: &[&str] = &[INVOKE_CLOSE, EOM, EOT, START];

type MuseGlimmerInput<'i> = Partial<&'i str>;

/// Maximum length in bytes of a header-candidate run held across deltas: a
/// recipient name, the whitespace inside a header, or an ATEM attribute run.
/// A run that outgrows the cap can never be a header, so the candidate fails
/// definitively (Backtrack, not Incomplete) and the body-text fallback
/// consumes it — deterministic per content, so chunking invariance holds, and
/// no unbounded run is held and re-scanned on every delta.
const MAX_CANDIDATE_LEN: usize = 1024;

/// Parse a run of at least `min` `pred` chars like `take_while`, but fail
/// definitively once the run exceeds [`MAX_CANDIDATE_LEN`] bytes instead of
/// waiting for a terminator that may never come.
fn capped_run<'i>(
    min: usize,
    pred: impl Fn(char) -> bool + 'i,
) -> impl Parser<MuseGlimmerInput<'i>, &'i str, ErrMode<ContextError>> + 'i {
    move |input: &mut MuseGlimmerInput<'i>| {
        let text = **input;
        let mut len = 0;
        for c in text.chars() {
            if !pred(c) {
                if len < min {
                    return Err(ErrMode::Backtrack(ContextError::new()));
                }
                input.next_slice(len);
                return Ok(&text[..len]);
            }
            len += c.len_utf8();
            if len > MAX_CANDIDATE_LEN {
                return Err(ErrMode::Backtrack(ContextError::new()));
            }
        }
        // The run reached the end of the partial input and may still grow.
        incomplete()
    }
}

/// Which channel a header recipient opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelKind {
    /// `to=self`: reasoning.
    Reasoning,
    /// `to=user` or an untagged `<|message|>`: visible content. Only an
    /// untagged body may be reclassified into a tool channel: a `to=user`
    /// final answer must never yield a real tool call (Python contract).
    /// Note the untagged reclassification deliberately differs from the
    /// Python fallback, which scans unframed output only when NO channel
    /// header exists at all; the Rust parser instead reclassifies an ATEM
    /// block inside an untagged channel (and never scans headerless text).
    Content { reclassify: bool },
    /// Any other recipient: an ATEM tool call.
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MuseGlimmerEvent {
    /// Body text for the current content channel.
    Text,
    /// Body text for the current reasoning channel.
    Reasoning,
    /// Structural noise consumed without emitting anything.
    Skip,
    /// A channel header opened a new channel. Tool channels entered through a
    /// real header (framed or bare) are strict.
    ChannelOpen(ChannelKind),
    /// `<|eom|>` closed the current channel.
    ChannelClose,
    /// `<|eot|>` ended the turn; everything after it is ignored.
    TurnEnd,
    /// A complete `<atem:invoke>` block inside a tool channel.
    Invoke {
        name: Option<String>,
        arguments: String,
    },
    /// An ATEM block reclassified from a content body into a tool channel;
    /// carries its first complete invoke.
    AtemToolChannel {
        name: Option<String>,
        arguments: String,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum MuseGlimmerMode {
    /// Between channels (also the turn start): waiting for the next header.
    /// Text without any framing falls through as visible content, so
    /// marker-free output still streams.
    #[default]
    Idle,
    /// Inside a `to=self` reasoning channel.
    Reasoning,
    /// Inside a `to=user` or untagged content channel; see
    /// [`ChannelKind::Content`] for `reclassify`.
    Content { reclassify: bool },
    /// Inside a tool channel. `strict` is set when the channel was entered
    /// through a real `to=<tool>` header rather than reclassified content.
    Tool { strict: bool },
    /// After `<|eot|>`: ignore the rest of the stream.
    Done,
}

/// Unified parser for Muse Glimmer `self` / `user` / tool channels.
pub struct MuseGlimmerUnifiedParser {
    buffer: DecodedText,
    mode: MuseGlimmerMode,
    /// Checkpoint of the close-marker scan over an invoke block buffered at
    /// the cursor, so re-parsing the block on every delta does not rescan the
    /// whole body. Offsets are cursor-relative: reset whenever bytes are
    /// consumed.
    invoke_scan: MarkerScanState,
    /// Number of tool calls emitted in the current response.
    emitted_call_count: usize,
    /// Whether any reasoning bytes were emitted in the current response.
    reasoning_emitted: bool,
    /// Whether the next reasoning text opens a later `to=self` block and must
    /// first be separated from earlier reasoning by `"\n"`. Set lazily at
    /// channel open so a block that receives no text emits no separator.
    pending_reasoning_sep: bool,
    /// Channel of a prompt tail `assistant to=RECIPIENT` prefilled without its
    /// `<|message|>`: the turn's first bare untagged header completes that
    /// header, so it opens this kind instead of untagged content. A framed
    /// header instead abandons the prefilled channel.
    prefilled_kind: Option<ChannelKind>,
    /// Names of the tools registered on the request (for name normalization).
    registered_names: Vec<String>,
    /// Whether the buffer position is a legal bare-header position (see
    /// [`BareHeaderAnchor`]). Without this, a bare header glued to a
    /// non-whitespace byte (`xto=calc<|message|>`) would parse differently
    /// depending on whether a delta boundary falls between them: whole-input
    /// only recognizes ws-anchored headers, while a buffer-start `to=` could
    /// not see the byte before it.
    bare_header_anchor: BareHeaderAnchor,
    tokenizer: DynTokenizer,
    start_token_id: u32,
}

/// Whether the buffer position is a legal bare-header position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BareHeaderAnchor {
    /// Stream start or right after a structural marker event (a channel close,
    /// header, or call): whitespace before the header belongs to the header.
    Structural,
    /// Right after body text ending in whitespace (already committed as text):
    /// a bare header starts exactly at `to=` / `<|message|>`. The whitespace
    /// class includes `\n` (Python `_OPEN_TAIL_HEADER_RE` anchors on `[\s]`).
    AfterWhitespace,
    /// After body text ending in a non-whitespace byte: a bare `to=…` or
    /// `<|message|>` here is literal body text.
    None,
}

impl MuseGlimmerUnifiedParser {
    /// Create a Muse Glimmer parser.
    pub fn new(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let start_token_id = token_id(tokenizer.as_ref(), START)?;

        Ok(Self {
            buffer: DecodedText::default(),
            mode: MuseGlimmerMode::default(),
            invoke_scan: MarkerScanState::default(),
            emitted_call_count: 0,
            reasoning_emitted: false,
            pending_reasoning_sep: false,
            prefilled_kind: None,
            registered_names: tools.iter().map(|tool| tool.name.clone()).collect(),
            bare_header_anchor: BareHeaderAnchor::Structural,
            tokenizer,
            start_token_id,
        })
    }

    /// Detect the prefilled generation channel from the prompt tail.
    ///
    /// `add_generation_prompt` ends the prompt with `<|start|>assistant` (or,
    /// with `continue_final_message`, a partial channel), so generation may
    /// start inside a channel without re-emitting its header. Locate the last
    /// `<|start|>` token and decode the tail after it.
    ///
    /// A prefilled tool channel seeds Tool mode, but body text already present
    /// in the prompt is NOT re-fed to the parser: a `continue_final_message`
    /// that stops mid-tool-call parses nothing, matching the Python parser's
    /// documented limitation.
    fn initialize_mode(&mut self, prompt_token_ids: &[u32]) {
        self.mode = MuseGlimmerMode::Idle;
        self.prefilled_kind = None;
        // Anything but an open prefilled channel body ends the prompt at a
        // header boundary: a legal bare-header position.
        self.bare_header_anchor = BareHeaderAnchor::Structural;

        let Some(start_pos) = prompt_token_ids.iter().rposition(|&id| id == self.start_token_id)
        else {
            return;
        };
        let Ok(tail) = self.tokenizer.decode(
            &prompt_token_ids[start_pos + 1..],
            /* skip_special_tokens */ false,
        ) else {
            return;
        };

        let mut tail_input = tail.as_str();
        let parsed: ModalResult<(Option<String>, Option<&str>)> = seq!(
            _: take_while(0.., char::is_whitespace),
            _: literal(ASSISTANT),
            _: take_while(0.., is_inline_ws),
            opt(preceded(literal("to="), complete_recipient_name)),
            alt((preceded(literal(MESSAGE), rest).map(Some), eof.value(None))),
        )
        .parse_next(&mut tail_input);
        // Not an assistant header, or one malformed before `<|message|>`
        // (`assistant\n`, an empty `to=`): nothing is prefilled.
        let Ok((recipient, body)) = parsed else {
            return;
        };
        // A header cut before `<|message|>` only fixes the recipient: the
        // generation completes it with a bare `<|message|>`.
        let Some(body) = body else {
            self.prefilled_kind = recipient.as_deref().map(|name| classify_recipient(Some(name)));
            return;
        };
        // A channel already closed in the prompt does not seed a mode: the
        // next header starts fresh from Idle.
        if body.contains(EOM) || body.contains(EOT) {
            return;
        }
        // Generation continues the prefilled body: a bare header is legal only
        // where the prompt's last byte leaves a bare-header position (an empty
        // body ends right after the `<|message|>` marker, a structural one).
        self.bare_header_anchor = match body.chars().next_back() {
            None => BareHeaderAnchor::Structural,
            Some(c) if c.is_whitespace() => BareHeaderAnchor::AfterWhitespace,
            Some(_) => BareHeaderAnchor::None,
        };
        self.mode = match classify_recipient(recipient.as_deref()) {
            ChannelKind::Reasoning => MuseGlimmerMode::Reasoning,
            ChannelKind::Content { reclassify } => MuseGlimmerMode::Content { reclassify },
            ChannelKind::Tool => MuseGlimmerMode::Tool { strict: true },
        };
    }

    fn apply_event(
        &mut self,
        event: MuseGlimmerEvent,
        piece: DecodedText,
        output: &mut UnifiedParserOutput,
    ) -> Result<()> {
        // The next event's bare-header position depends on what this event
        // committed: marker/header/call spans are structural; body text leaves
        // a position only when it ends in whitespace.
        match &event {
            MuseGlimmerEvent::Text | MuseGlimmerEvent::Reasoning => {
                if let Some(last) = piece.text.chars().next_back() {
                    self.bare_header_anchor = if last.is_whitespace() {
                        BareHeaderAnchor::AfterWhitespace
                    } else {
                        BareHeaderAnchor::None
                    };
                }
            }
            _ => self.bare_header_anchor = BareHeaderAnchor::Structural,
        }
        match event {
            MuseGlimmerEvent::Text => output.push_text(piece.text),
            MuseGlimmerEvent::Reasoning => self.push_reasoning_text(piece, output),
            // Marker and noise spans are drained and dropped with their tokens.
            MuseGlimmerEvent::Skip => {}
            MuseGlimmerEvent::ChannelOpen(kind) => {
                // Only a bare header (no `<|start|>`) completes the prefill.
                let kind = match (self.prefilled_kind.take(), kind) {
                    (Some(prefilled), ChannelKind::Content { reclassify: true })
                        if !piece.text.starts_with(START) =>
                    {
                        prefilled
                    }
                    (_, kind) => kind,
                };
                self.open_channel(kind);
            }
            MuseGlimmerEvent::ChannelClose => self.mode = MuseGlimmerMode::Idle,
            MuseGlimmerEvent::TurnEnd => self.mode = MuseGlimmerMode::Done,
            MuseGlimmerEvent::Invoke { name, arguments } => {
                self.emit_invoke(name, arguments, output);
            }
            MuseGlimmerEvent::AtemToolChannel { name, arguments } => {
                self.mode = MuseGlimmerMode::Tool { strict: false };
                self.emit_invoke(name, arguments, output);
            }
        }
        Ok(())
    }

    /// Open a channel, arming the lazy `"\n"` separator between repeated
    /// `to=self` blocks: the separator is emitted only when the block's first
    /// reasoning text arrives, so a block that receives no text (e.g. an
    /// abandoned reasoning prefill) emits nothing.
    fn open_channel(&mut self, kind: ChannelKind) {
        self.mode = match kind {
            ChannelKind::Reasoning => {
                self.pending_reasoning_sep = self.reasoning_emitted;
                MuseGlimmerMode::Reasoning
            }
            ChannelKind::Content { reclassify } => MuseGlimmerMode::Content { reclassify },
            ChannelKind::Tool => MuseGlimmerMode::Tool { strict: true },
        };
    }

    /// Push reasoning body text, separating a later `to=self` block's first
    /// text from earlier reasoning by `"\n"`.
    fn push_reasoning_text(&mut self, piece: DecodedText, output: &mut UnifiedParserOutput) {
        if piece.text.is_empty() {
            output.push_reasoning(piece);
            return;
        }
        if self.pending_reasoning_sep {
            output.push_reasoning(DecodedText::unattributed("\n"));
            self.pending_reasoning_sep = false;
        }
        self.reasoning_emitted = true;
        output.push_reasoning(piece);
    }

    /// Emit one completed invoke as a tool call.
    fn emit_invoke(
        &mut self,
        name: Option<String>,
        arguments: String,
        output: &mut UnifiedParserOutput,
    ) {
        // An invoke without a `name` attribute is skipped (Python parity).
        let Some(name) = name else {
            return;
        };
        let name = normalize_name(&name, &self.registered_names);
        let tool_index = self.emitted_call_count;
        self.emitted_call_count += 1;
        output.push_call(ToolCallDelta {
            tool_index,
            name: Some(name),
            arguments,
        });
    }

    fn reset_state(&mut self) -> String {
        self.mode = MuseGlimmerMode::Idle;
        self.invoke_scan.reset();
        self.emitted_call_count = 0;
        self.reasoning_emitted = false;
        self.pending_reasoning_sep = false;
        self.prefilled_kind = None;
        self.bare_header_anchor = BareHeaderAnchor::Structural;
        self.buffer.take().text
    }
}

impl UnifiedParser for MuseGlimmerUnifiedParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Self::new(tools, tokenizer).map(|parser| Box::new(parser) as Box<dyn UnifiedParser>)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.buffer.clear();
        self.invoke_scan.reset();
        self.emitted_call_count = 0;
        self.reasoning_emitted = false;
        self.pending_reasoning_sep = false;
        self.initialize_mode(prompt_token_ids);
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn scoped_structural_tag_builder(&self) -> Option<&dyn ScopedStructuralTagBuilder> {
        Some(&MUSE_GLIMMER_STRUCTURAL_TAG_BUILDER)
    }

    // The legacy Python `muse_glimmer` reasoner only recognizes tool channels
    // (not `to=user`), and this parser's whole-generation structural tags
    // must apply from token 0 — so the engine must not be given a reasoning
    // parser for this model.
    fn forwards_engine_reasoning_parser() -> bool
    where
        Self: Sized,
    {
        false
    }

    fn parse_into(&mut self, delta: DecodedText, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffer.append(delta);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer.text, |input| {
            parse_next_muse_glimmer_event(
                input,
                &mut self.mode,
                &mut self.invoke_scan,
                self.bare_header_anchor,
            )
        })? {
            let piece = self.buffer.drain_prefix(consumed_len);
            self.invoke_scan.reset();
            self.apply_event(event, piece, output)?;
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();

        match self.mode {
            MuseGlimmerMode::Idle | MuseGlimmerMode::Content { .. } => {
                // The stream ended: a trailing ` to=…` fragment can no longer
                // grow into a header, so it is flushed; a trailing COMPLETE
                // anchored bare header (a channel that never got a body) and
                // trailing truncated framing stay dropped.
                let text = strip_complete_bare_header(
                    strip_trailing_truncated_framing(&self.buffer.text),
                    self.bare_header_anchor,
                )
                .to_string();
                self.buffer.clear();
                output.push_text(text);
            }
            MuseGlimmerMode::Reasoning => {
                let len = strip_complete_bare_header(
                    strip_trailing_truncated_framing(&self.buffer.text),
                    self.bare_header_anchor,
                )
                .len();
                let piece = self.buffer.drain_prefix(len);
                self.buffer.clear();
                self.push_reasoning_text(piece, &mut output);
            }
            // A tool channel truncated between complete calls loses only its
            // closing markers — possibly cut mid-marker; keep the calls
            // already emitted. Anything more is an incomplete call.
            MuseGlimmerMode::Tool { strict: true } => {
                let text = &self.buffer.text;
                let held = max_partial_prefix_len(text, TOOL_NOISE_MARKERS)
                    .max(partial_prefix_len(text, INVOKE_CLOSE));
                if !text[..text.len() - held].trim().is_empty() {
                    return Err(parsing_failed!("incomplete Muse Glimmer tool call"));
                }
                self.buffer.clear();
            }
            // A reclassified channel scans like a tool channel: leftover text
            // after the last complete invoke is dropped, as it is mid-stream.
            MuseGlimmerMode::Tool { strict: false } => self.buffer.clear(),
            MuseGlimmerMode::Done => self.buffer.clear(),
        }

        self.reset_state();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.reset_state()
    }
}

/// Parse one Muse Glimmer event from buffered streaming input.
fn parse_next_muse_glimmer_event(
    input: &mut MuseGlimmerInput<'_>,
    mode: &mut MuseGlimmerMode,
    invoke_scan: &mut MarkerScanState,
    bare_header_anchor: BareHeaderAnchor,
) -> ModalResult<MuseGlimmerEvent> {
    match mode {
        MuseGlimmerMode::Idle => parse_idle_event(input, bare_header_anchor),
        MuseGlimmerMode::Reasoning => parse_reasoning_event(input, bare_header_anchor),
        MuseGlimmerMode::Content { reclassify } => {
            parse_content_event(input, *reclassify, invoke_scan, bare_header_anchor)
        }
        MuseGlimmerMode::Tool { .. } => parse_tool_event(input, invoke_scan),
        MuseGlimmerMode::Done => parse_done_event(input),
    }
}

/// Run a bare-header parser only at a legal bare-header position.
fn at_bare_header_position<'i, O>(
    anchor: BareHeaderAnchor,
    mut parser: impl FnMut(&mut MuseGlimmerInput<'i>) -> ModalResult<O>,
) -> impl FnMut(&mut MuseGlimmerInput<'i>) -> ModalResult<O> {
    move |input: &mut MuseGlimmerInput<'i>| {
        if anchor == BareHeaderAnchor::None {
            Err(ErrMode::Backtrack(ContextError::new()))
        } else {
            parser(input)
        }
    }
}

/// Parse an event while waiting for the next channel header.
fn parse_idle_event(
    input: &mut MuseGlimmerInput<'_>,
    bare_header_anchor: BareHeaderAnchor,
) -> ModalResult<MuseGlimmerEvent> {
    alt((
        framed_header_event,
        |input: &mut MuseGlimmerInput<'_>| anchored_bare_header_event(input, bare_header_anchor),
        // A stray close between channels is structural noise.
        literal(EOM).value(MuseGlimmerEvent::Skip),
        literal(EOT).value(MuseGlimmerEvent::TurnEnd),
        // A bare `<|message|>` not at a bare-header position is literal text.
        literal(MESSAGE).value(MuseGlimmerEvent::Text),
        // A `<|start|>` that does not begin a valid framed header is literal text.
        literal(START).value(MuseGlimmerEvent::Text),
        safe_idle_text_event,
    ))
    .parse_next(input)
}

/// Parse an event inside a reasoning (`to=self`) channel.
fn parse_reasoning_event(
    input: &mut MuseGlimmerInput<'_>,
    bare_header_anchor: BareHeaderAnchor,
) -> ModalResult<MuseGlimmerEvent> {
    alt((
        framed_header_event,
        at_bare_header_position(bare_header_anchor, bare_tool_switch_event),
        literal(EOM).value(MuseGlimmerEvent::ChannelClose),
        literal(EOT).value(MuseGlimmerEvent::TurnEnd),
        at_bare_header_position(bare_header_anchor, |input: &mut MuseGlimmerInput<'_>| {
            failed_bare_header_text(input).map(|_| MuseGlimmerEvent::Reasoning)
        }),
        literal(START).value(MuseGlimmerEvent::Reasoning),
        safe_reasoning_text_event,
    ))
    .parse_next(input)
}

/// Parse an event inside a content (`to=user` or untagged) channel. Only an
/// untagged channel reclassifies ATEM blocks (see [`ChannelKind::Content`]).
fn parse_content_event(
    input: &mut MuseGlimmerInput<'_>,
    reclassify: bool,
    invoke_scan: &mut MarkerScanState,
    bare_header_anchor: BareHeaderAnchor,
) -> ModalResult<MuseGlimmerEvent> {
    if !reclassify {
        return alt((
            framed_header_event,
            at_bare_header_position(bare_header_anchor, bare_tool_switch_event),
            literal(EOM).value(MuseGlimmerEvent::ChannelClose),
            literal(EOT).value(MuseGlimmerEvent::TurnEnd),
            at_bare_header_position(bare_header_anchor, |input: &mut MuseGlimmerInput<'_>| {
                failed_bare_header_text(input).map(|_| MuseGlimmerEvent::Text)
            }),
            literal(START).value(MuseGlimmerEvent::Text),
            safe_tagged_content_text_event,
        ))
        .parse_next(input);
    }
    alt((
        framed_header_event,
        at_bare_header_position(bare_header_anchor, bare_tool_switch_event),
        literal(EOM).value(MuseGlimmerEvent::ChannelClose),
        literal(EOT).value(MuseGlimmerEvent::TurnEnd),
        |input: &mut MuseGlimmerInput<'_>| atem_tool_channel_event(input, invoke_scan),
        // The ATEM opener is literal content when no complete invoke follows.
        alt((literal(FUNCTION_CALLS_OPEN), literal(INVOKE_OPEN))).value(MuseGlimmerEvent::Text),
        at_bare_header_position(bare_header_anchor, |input: &mut MuseGlimmerInput<'_>| {
            failed_bare_header_text(input).map(|_| MuseGlimmerEvent::Text)
        }),
        literal(START).value(MuseGlimmerEvent::Text),
        safe_content_text_event,
    ))
    .parse_next(input)
}

/// Parse an event inside a tool channel: wrapper markers and stray text are
/// skipped (Python scans with regex findall), complete invokes become calls.
fn parse_tool_event(
    input: &mut MuseGlimmerInput<'_>,
    invoke_scan: &mut MarkerScanState,
) -> ModalResult<MuseGlimmerEvent> {
    alt((
        // A framed header is authoritative anywhere, closing the tool channel.
        framed_header_event,
        literal(EOM).value(MuseGlimmerEvent::ChannelClose),
        literal(EOT).value(MuseGlimmerEvent::TurnEnd),
        literal(FUNCTION_CALLS_OPEN).value(MuseGlimmerEvent::Skip),
        literal(FUNCTION_CALLS_CLOSE).value(MuseGlimmerEvent::Skip),
        |input: &mut MuseGlimmerInput<'_>| tool_invoke_event(input, invoke_scan),
        // A definitively-rejected marker at the cursor (an invoke whose body
        // quotes framing, an `<atem:invoke`-prefixed word, a `<|start|>` that
        // is no header) must still be consumed: the noise scanner cannot skip
        // past a marker sitting at offset 0, and zero consumption would stall
        // the parser for the rest of the stream.
        literal(INVOKE_OPEN).value(MuseGlimmerEvent::Skip),
        literal(START).value(MuseGlimmerEvent::Skip),
        skip_tool_noise_event,
    ))
    .parse_next(input)
}

/// Ignore everything after the turn ended (EOS leakage guard).
fn parse_done_event(input: &mut MuseGlimmerInput<'_>) -> ModalResult<MuseGlimmerEvent> {
    rest.value(MuseGlimmerEvent::Skip).parse_next(input)
}

/// Parse a framed channel header: `<|start|>` + `\s*` + `assistant` + the
/// bare header tail (see [`bare_header_event`]).
fn framed_header_event(input: &mut MuseGlimmerInput<'_>) -> ModalResult<MuseGlimmerEvent> {
    preceded(
        (
            literal(START),
            capped_run(0, is_ascii_multispace),
            literal(ASSISTANT),
        ),
        bare_header_event,
    )
    .parse_next(input)
}

/// Parse a bare channel header at channel boundaries: `[^\S\n]*` + optional
/// `to=RECIPIENT` + `<|message|>`. The first channel of a turn starts bare
/// right after the prompt's trailing `<|start|>assistant`. Callers must gate
/// this on the parser's bare-header anchor (see
/// [`MuseGlimmerUnifiedParser::bare_header_anchor`]); a framed-header tail
/// needs no gate because a framed header is authoritative anywhere.
fn bare_header_event(input: &mut MuseGlimmerInput<'_>) -> ModalResult<MuseGlimmerEvent> {
    preceded(capped_run(0, is_inline_ws), bare_header_tail_event).parse_next(input)
}

/// Parse a bare channel header without its leading-whitespace run: after body
/// text the whitespace is already committed as text, so the header starts
/// exactly at `to=` / `<|message|>`.
fn bare_header_tail_event(input: &mut MuseGlimmerInput<'_>) -> ModalResult<MuseGlimmerEvent> {
    let (recipient,) = seq!(
        opt(preceded(literal("to="), recipient_name)),
        _: literal(MESSAGE),
    )
    .parse_next(input)?;
    Ok(MuseGlimmerEvent::ChannelOpen(classify_recipient(
        recipient.as_deref(),
    )))
}

/// Parse a bare channel header at a legal bare-header position: leading
/// whitespace belongs to the header only at a structural position (stream
/// start or right after a marker); after body text it is already committed
/// text, so the header starts exactly at `to=` / `<|message|>`.
fn anchored_bare_header_event(
    input: &mut MuseGlimmerInput<'_>,
    anchor: BareHeaderAnchor,
) -> ModalResult<MuseGlimmerEvent> {
    match anchor {
        BareHeaderAnchor::Structural => bare_header_event(input),
        BareHeaderAnchor::AfterWhitespace => bare_header_tail_event(input),
        BareHeaderAnchor::None => Err(ErrMode::Backtrack(ContextError::new())),
    }
}

/// Parse a bare `to=RECIPIENT<|message|>` header appearing mid-body.
fn bare_recipient_header(input: &mut MuseGlimmerInput<'_>) -> ModalResult<String> {
    delimited(literal("to="), recipient_name, literal(MESSAGE)).parse_next(input)
}

/// Parse a bare `to=<tool><|message|>` header that is immediately followed by
/// `<atem:`: the model defect of an unterminated reasoning/content body closed
/// by a bare tool header (deterministic for empty-argument calls).
fn bare_tool_switch_event(input: &mut MuseGlimmerInput<'_>) -> ModalResult<MuseGlimmerEvent> {
    // A bare self/user header mid-body is literal text, not a switch.
    let is_tool = |recipient: &str| !matches!(recipient, "self" | "user");
    terminated(
        bare_recipient_header.verify(is_tool),
        peek(literal(ATEM_PREFIX)),
    )
    .value(MuseGlimmerEvent::ChannelOpen(ChannelKind::Tool))
    .parse_next(input)
}

/// Consume a bare `to=RECIPIENT<|message|>` header that cannot be a tool
/// switch (recipient is `self`/`user`, or no ATEM block follows) as literal
/// body text: a quoted header must not truncate the body.
fn failed_bare_header_text(input: &mut MuseGlimmerInput<'_>) -> ModalResult<()> {
    terminated(bare_recipient_header, peek(not(literal(ATEM_PREFIX))))
        .void()
        .parse_next(input)
}

/// Reclassify an ATEM block inside a content body as a tool channel, emitting
/// its first complete invoke. Commits only once a complete
/// `<atem:invoke>…</atem:invoke>` is ahead; otherwise the parse either holds
/// (incomplete) or fails definitively and the opener is literal content.
/// Whitespace between the wrapper and the invoke is structural only when the
/// wrapper matched; whitespace before a bare invoke stays content.
fn atem_tool_channel_event(
    input: &mut MuseGlimmerInput<'_>,
    invoke_scan: &mut MarkerScanState,
) -> ModalResult<MuseGlimmerEvent> {
    let ((name, arguments),) = seq!(
        _: opt(preceded(
            literal(FUNCTION_CALLS_OPEN),
            capped_run(0, is_ascii_multispace),
        )),
        |input: &mut MuseGlimmerInput<'_>| invoke_block(input, invoke_scan),
    )
    .parse_next(input)?;
    Ok(MuseGlimmerEvent::AtemToolChannel { name, arguments })
}

/// Parse one complete `<atem:invoke name="N">…</atem:invoke>` block into a call.
fn tool_invoke_event(
    input: &mut MuseGlimmerInput<'_>,
    invoke_scan: &mut MarkerScanState,
) -> ModalResult<MuseGlimmerEvent> {
    invoke_block(input, invoke_scan)
        .map(|(name, arguments)| MuseGlimmerEvent::Invoke { name, arguments })
}

/// Parse one complete `<atem:invoke name="N">…</atem:invoke>` block into its
/// name attribute and arguments JSON object string.
///
/// A framing marker (`<|…|>`) inside the block fails the parse: real ATEM
/// bodies never contain framing, so such markup was quoted body text and the
/// `<|eom|>`/`<|eot|>` after it must still be allowed to close the channel.
///
/// The block is buffered whole and re-parsed from the cursor on every delta,
/// so the body scan resumes from `invoke_scan` instead of rescanning it.
fn invoke_block(
    input: &mut MuseGlimmerInput<'_>,
    invoke_scan: &mut MarkerScanState,
) -> ModalResult<(Option<String>, String)> {
    let (attrs, body) = seq!(
        _: literal(INVOKE_OPEN),
        // `\b` after `invoke`: the tag name must not run into an identifier.
        _: peek(not(capped_run(1, |c: char| c.is_ascii_alphanumeric() || c == '_'))),
        atem_tag_attrs,
        _: literal(">"),
        // Stops at the close or at a framing marker; only the close may follow.
        take_until_marker_mul(INVOKE_BODY_STOP_MARKERS, invoke_scan),
        _: literal(INVOKE_CLOSE),
    )
    .parse_next(input)?;
    Ok((atem_name_attr(attrs), parse_invoke_arguments(body)?))
}

/// Parse an ATEM opener tag's attributes up to `>`, rejecting `<` so a framing
/// marker cannot be swallowed into a tag.
fn atem_tag_attrs<'i>(input: &mut MuseGlimmerInput<'i>) -> ModalResult<&'i str> {
    capped_run(0, |c: char| c != '>' && c != '<').parse_next(input)
}

/// Parse safe text while waiting for the next channel header.
fn safe_idle_text_event(input: &mut MuseGlimmerInput<'_>) -> ModalResult<MuseGlimmerEvent> {
    safe_body_text_len(input, IDLE_STOP_MARKERS, BODY_HOLD_BACK_MARKERS)
        .map(|_| MuseGlimmerEvent::Text)
}

/// Parse safe reasoning text before the next channel marker. Reasoning bodies
/// never reclassify: quoted ATEM markup stays reasoning text.
fn safe_reasoning_text_event(input: &mut MuseGlimmerInput<'_>) -> ModalResult<MuseGlimmerEvent> {
    safe_body_text_len(input, BODY_STOP_MARKERS, BODY_HOLD_BACK_MARKERS)
        .map(|_| MuseGlimmerEvent::Reasoning)
}

/// Parse safe content text, additionally stopping at ATEM openers so they can
/// be reclassified into a tool channel.
fn safe_content_text_event(input: &mut MuseGlimmerInput<'_>) -> ModalResult<MuseGlimmerEvent> {
    safe_body_text_len(input, CONTENT_STOP_MARKERS, CONTENT_HOLD_BACK_MARKERS)
        .map(|_| MuseGlimmerEvent::Text)
}

/// Parse safe `to=user` content text: ATEM markup never reclassifies here, so
/// it streams as plain content.
fn safe_tagged_content_text_event(
    input: &mut MuseGlimmerInput<'_>,
) -> ModalResult<MuseGlimmerEvent> {
    safe_body_text_len(input, BODY_STOP_MARKERS, BODY_HOLD_BACK_MARKERS)
        .map(|_| MuseGlimmerEvent::Text)
}

/// Skip non-content noise between invokes in a tool channel.
fn skip_tool_noise_event(input: &mut MuseGlimmerInput<'_>) -> ModalResult<MuseGlimmerEvent> {
    safe_text_len_mul(input, TOOL_NOISE_MARKERS).map(|_| MuseGlimmerEvent::Skip)
}

/// Parse safe body text before the next structural candidate, returning its
/// length in bytes and advancing the input.
///
/// The scan stops at the earliest `stop_markers` match or bare-header
/// candidate (whitespace directly followed by `to=`, the same anchor the tail
/// holdback uses — chunked and whole-input parses must agree on what is a
/// header), so the header alternatives get first chance at it. With no
/// candidate in sight, the tail holdback keeps partials of `hold_markers` and
/// any trailing ` to=…` fragment that could still grow into a header.
fn safe_body_text_len(
    input: &mut MuseGlimmerInput<'_>,
    stop_markers: &[&str],
    hold_markers: &[&str],
) -> ModalResult<usize> {
    let text = **input;
    if text.is_empty() {
        return incomplete();
    }

    let mut stop = text.len();
    for marker in stop_markers {
        if let Some(index) = text.find(marker) {
            stop = stop.min(index);
        }
    }
    for (index, _) in text.match_indices("to=") {
        if index >= stop {
            break;
        }
        if index > 0 && text[..index].chars().next_back().is_some_and(char::is_whitespace) {
            stop = index;
            break;
        }
    }
    if stop < text.len() {
        if stop == 0 {
            // A structural alternative ahead of this scanner must consume it.
            return incomplete();
        }
        input.next_slice(stop);
        return Ok(stop);
    }

    // Iterate the holdback to a fixpoint: trimming a partial marker can expose
    // a trailing ` to=…` fragment (" to=skill<") and vice versa. One marker
    // strip with nothing else to strip is settled: a marker's third byte is a
    // letter, so of a run of marker starts only the last can still grow into a
    // marker and the earlier ones are text. The fragment scan resumes at the
    // previous strip point, so one call is O(n) overall.
    let mut emit_len = text.len();
    loop {
        let marker_hold = max_partial_prefix_len(&text[..emit_len], hold_markers);
        let body_end = emit_len - marker_hold;
        let fragment_hold = open_tail_to_fragment_len(&text[..body_end]);
        emit_len = body_end - fragment_hold;
        if fragment_hold == 0 {
            break;
        }
    }
    if emit_len == 0 {
        return incomplete();
    }
    input.next_slice(emit_len);
    Ok(emit_len)
}

/// Length of a trailing ` to=`-in-progress fragment (` t`, ` to`, ` to=NAME*`)
/// that could still grow into a bare channel header (Python `_OPEN_TAIL_HEADER_RE`).
fn open_tail_to_fragment_len(text: &str) -> usize {
    // Only the last whitespace can anchor such a fragment: the recipient
    // charset excludes whitespace, so an earlier anchor's suffix cannot match.
    let Some((ws_index, ws_char)) = text.char_indices().rev().find(|(_, c)| c.is_whitespace())
    else {
        return 0;
    };
    let suffix = &text[ws_index + ws_char.len_utf8()..];
    let holds = matches!(suffix, "t" | "to")
        || suffix
            .strip_prefix("to=")
            .is_some_and(|name| name.chars().all(|c| !c.is_whitespace() && c != '<'));
    if holds { text.len() - ws_index } else { 0 }
}

/// Strip a trailing COMPLETE bare header (`to=RECIPIENT<|message|>`) from a
/// finished body when the buffer is at a legal bare-header position: the
/// header opened a channel that never got a body, so it is dropped rather
/// than flushed as text (Python parity: its header-bounded regexes drop it).
/// Partial fragments still flush.
fn strip_complete_bare_header(text: &str, anchor: BareHeaderAnchor) -> &str {
    if anchor == BareHeaderAnchor::None {
        return text;
    }
    let mut input = text;
    let parsed: ModalResult<()> = seq!(
        _: literal("to="),
        _: complete_recipient_name,
        _: literal(MESSAGE),
        _: eof,
    )
    .void()
    .parse_next(&mut input);
    if parsed.is_ok() { "" } else { text }
}

/// Strip trailing truncated framing from a finished body: a partial structural
/// marker, or a complete `<|start|>` whose framed header was cut off before
/// its `<|message|>` (e.g. `<|start|>assist` at a max_tokens stop).
fn strip_trailing_truncated_framing(text: &str) -> &str {
    let text = &text[..text.len() - max_partial_prefix_len(text, BODY_HOLD_BACK_MARKERS)];
    let Some(start) = text.rfind(START) else {
        return text;
    };
    let mut header = MuseGlimmerInput::new(&text[start..]);
    match framed_header_event(&mut header) {
        Err(ErrMode::Incomplete(_)) => &text[..start],
        _ => text,
    }
}

/// Parse a channel recipient name (`[A-Za-z0-9_.\-]+`) from streaming input.
fn recipient_name(input: &mut MuseGlimmerInput<'_>) -> ModalResult<String> {
    capped_run(1, is_recipient_char).map(str::to_string).parse_next(input)
}

/// Parse a channel recipient name from a complete (non-streaming) input.
fn complete_recipient_name(input: &mut &str) -> ModalResult<String> {
    take_while(1.., is_recipient_char).map(str::to_string).parse_next(input)
}

/// Classify a channel recipient: `self` is reasoning, `user` or an absent
/// recipient is visible content, anything else is a tool call.
fn classify_recipient(recipient: Option<&str>) -> ChannelKind {
    match recipient {
        Some("self") => ChannelKind::Reasoning,
        Some("user") => ChannelKind::Content { reclassify: false },
        None => ChannelKind::Content { reclassify: true },
        Some(_) => ChannelKind::Tool,
    }
}

/// Whether `c` is whitespace other than a newline (`[^\S\n]` in the grammar).
fn is_inline_ws(c: char) -> bool {
    c.is_whitespace() && c != '\n'
}

/// Whether `c` is ASCII whitespace (winnow's `multispace0` charset).
fn is_ascii_multispace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}

/// Whether `c` is a channel recipient character (`[A-Za-z0-9_.\-]`).
fn is_recipient_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')
}

/// Extract the `name="…"` attribute from an ATEM opener tag (Python `_NAME_RE`).
fn atem_name_attr(attrs: &str) -> Option<String> {
    let mut offset = 0;
    while let Some(found) = attrs[offset..].find("name=\"") {
        let start = offset + found;
        // `\bname`: the attribute name must not be a suffix of a longer word.
        let is_boundary = start == 0
            || attrs[..start]
                .chars()
                .next_back()
                .is_some_and(|c| !(c.is_ascii_alphanumeric() || c == '_'));
        if !is_boundary {
            offset = start + 1;
            continue;
        }
        let value_start = start + "name=\"".len();
        let value_end = attrs[value_start..].find('"')? + value_start;
        // `[^"]+`: an empty name attribute does not count; a later one may.
        if value_end > value_start {
            return Some(attrs[value_start..value_end].to_string());
        }
        offset = value_end + 1;
    }
    None
}

/// Parse one `<atem:parameter name="K">value</atem:parameter>` block.
fn parameter_pair(input: &mut &str) -> ModalResult<(String, Value)> {
    let (attrs, raw) = seq!(
        _: literal(PARAMETER_OPEN),
        // `\b` after `parameter`: the tag name must not run into an identifier.
        _: peek(not(take_while(1.., |c: char| c.is_ascii_alphanumeric() || c == '_'))),
        take_until(0.., ">"),
        _: literal(">"),
        take_until(0.., PARAMETER_CLOSE),
        _: literal(PARAMETER_CLOSE),
    )
    .parse_next(input)?;
    // A parameter without a `name` attribute is skipped (Python parity).
    let Some(key) = atem_name_attr(attrs) else {
        return Err(ErrMode::Backtrack(ContextError::new()));
    };
    Ok((key, decode_value(raw)))
}

/// Parse all parameter blocks in one invoke body into a JSON object string,
/// preserving emission order. Non-parameter text is skipped (Python findall
/// parity), so whitespace between blocks and stray noise are dropped.
fn parse_invoke_arguments(body: &str) -> ModalResult<String> {
    let mut pairs: Vec<(String, Value)> = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find(PARAMETER_OPEN) {
        // `seq!` does not rewind a failed parse, so resume the scan from the
        // opener's own position, not from wherever the attempt stopped.
        let attempt = &rest[start..];
        rest = attempt;
        match parameter_pair.parse_next(&mut rest) {
            Ok(pair) => pairs.push(pair),
            // Skip a malformed opener so one bad parameter loses only itself.
            Err(_) => rest = &attempt[PARAMETER_OPEN.len()..],
        }
    }
    let arguments = pairs.into_iter().collect::<Map<String, Value>>();
    serde_json::to_string(&arguments).map_err(|_| atem_error("Muse Glimmer tool arguments"))
}

/// Decode one parameter value: JSON when possible, else the raw string
/// (Python `_decode_value`: `x-parser: json` with `allow_non_json: True`).
fn decode_value(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Map an emitted ATEM invoke name back to a registered tool name (Python
/// `_normalize_name`).
fn normalize_name(emitted: &str, registered_names: &[String]) -> String {
    if registered_names.is_empty() || registered_names.iter().any(|name| name == emitted) {
        return emitted.to_string();
    }
    // A client-registered bare name renders as `name.*`, and the model duly
    // emits `name.name`; collapse that doubled form when the head is
    // registered. Anything else passes through unchanged: matching on the
    // trailing segment alone could silently dispatch the wrong tool.
    if let Some((head, tail)) = emitted.split_once('.')
        && head == tail
        && registered_names.iter().any(|name| name == head)
    {
        return head.to_string();
    }
    emitted.to_string()
}

/// Build a cut error for determinably malformed ATEM structure.
fn atem_error(label: &'static str) -> ErrMode<ContextError> {
    let mut error = ContextError::new();
    error.push(StrContext::Label(label));
    ErrMode::Cut(error)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::{Value, json};
    use thiserror_ext::AsReport;
    use vllm_tokenizer::Tokenizer as _;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::{ATEM_PREFIX, EOT, MuseGlimmerUnifiedParser, START};
    use crate::tool::Tool;
    use crate::unified::test_utils::{
        UnifiedOutputTestExt, UnifiedParserTestExt, char_chunks, collect_stream, first_call,
    };
    use crate::unified::{UnifiedParser, UnifiedParserError, UnifiedParserOutput};

    // The real Muse Glimmer tokenizer ids for the framing tokens.
    const START_ID: u32 = 200022;
    const MESSAGE_ID: u32 = 200023;
    const EOM_ID: u32 = 200024;
    const EOT_ID: u32 = 200025;

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_special_token(START, START_ID)
            .with_special_token("<|message|>", MESSAGE_ID)
            .with_special_token("<|eom|>", EOM_ID)
            .with_special_token(EOT, EOT_ID)
    }

    fn test_parser() -> MuseGlimmerUnifiedParser {
        MuseGlimmerUnifiedParser::new(&[], Arc::new(tokenizer())).unwrap()
    }

    fn test_parser_with_tools(names: &[&str]) -> MuseGlimmerUnifiedParser {
        let tools: Vec<Tool> = names
            .iter()
            .map(|name| Tool {
                name: (*name).to_string(),
                description: None,
                parameters: json!({}),
                strict: None,
            })
            .collect();
        MuseGlimmerUnifiedParser::new(&tools, Arc::new(tokenizer())).unwrap()
    }

    /// Concatenated events must be identical whether the turn arrives whole or
    /// chunked at 1/3/7 chars.
    fn assert_chunking_invariant(text: &str) -> UnifiedParserOutput {
        let whole = collect_stream(&mut test_parser(), &[text]);
        for size in [1, 3, 7] {
            let chunks = char_chunks(text, size);
            let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let streamed = collect_stream(&mut test_parser(), &chunk_refs);
            assert_eq!(streamed, whole, "chunk size {size}");
        }
        whole
    }

    fn param(key: &str, value: &str) -> String {
        format!("<atem:parameter name=\"{key}\">{value}</atem:parameter>\n")
    }

    fn invoke(name: &str, params: &str) -> String {
        format!("<atem:invoke name=\"{name}\">\n{params}</atem:invoke>\n")
    }

    /// One framed tool channel invoking `name` (header recipient = tool name).
    fn tool_channel(name: &str, params: &str, close: &str) -> String {
        format!(
            "<|start|>assistant to={name}<|message|><atem:function_calls>\n{}</atem:function_calls>{close}",
            invoke(name, params)
        )
    }

    #[test]
    fn muse_glimmer_create_requires_start_token() {
        let error = match MuseGlimmerUnifiedParser::new(&[], Arc::new(TestTokenizer::new())) {
            Ok(_) => panic!("expected missing token error"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            UnifiedParserError::MissingToken { token } if token == START
        ));
    }

    #[test]
    fn muse_glimmer_answer_only_turn() {
        let output = assert_chunking_invariant(" to=user<|message|>Just a direct answer.<|eot|>");

        assert_eq!(output.normal_text(), "Just a direct answer.");
        assert!(output.reasoning_text().is_empty());
        assert!(output.calls().is_empty());
    }

    #[test]
    fn muse_glimmer_reasoning_then_answer() {
        let output = assert_chunking_invariant(
            " to=self<|message|>Let me think step by step about the sum.<|eom|>\
             <|start|>assistant to=user<|message|>The answer is 42.<|eot|>",
        );

        assert_eq!(
            output.reasoning_text(),
            "Let me think step by step about the sum."
        );
        assert_eq!(output.normal_text(), "The answer is 42.");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn muse_glimmer_reasoning_then_tool_call() {
        let output = assert_chunking_invariant(
            " to=self<|message|>I should read the hostname.<|eom|>\
             <|start|>assistant to=read.read<|message|>\
             <atem:function_calls>\n<atem:invoke name=\"read.read\">\n\
             <atem:parameter name=\"path\">/etc/hostname</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls><|eot|>",
        );

        assert_eq!(output.reasoning_text(), "I should read the hostname.");
        assert!(output.normal_text().is_empty());
        let call = first_call(&output);
        assert_eq!(call.tool_index, 0);
        assert_eq!(call.name.as_deref(), Some("read.read"));
        assert_eq!(call.arguments, r#"{"path":"/etc/hostname"}"#);
    }

    #[test]
    fn muse_glimmer_reasoning_tool_reasoning_interleave_joins_with_newline() {
        let output = assert_chunking_invariant(
            " to=self<|message|>first thoughts<|eom|>\
             <|start|>assistant to=weather.get<|message|>\
             <atem:function_calls>\n<atem:invoke name=\"weather.get\">\n\
             <atem:parameter name=\"city\">Paris</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls><|eom|>\
             <|start|>assistant to=self<|message|>second thoughts<|eom|>\
             <|start|>assistant to=user<|message|>done<|eot|>",
        );

        assert_eq!(output.reasoning_text(), "first thoughts\nsecond thoughts");
        assert_eq!(output.normal_text(), "done");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(first_call(&output).arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn muse_glimmer_unterminated_reasoning_switches_on_bare_tool_header() {
        // The model defect: a `to=self` body that is never closed by `<|eom|>`
        // followed by a bare tool header (deterministic for empty-argument
        // calls). The whitespace before the header stays reasoning text.
        let output = assert_chunking_invariant(
            " to=self<|message|>think then to=weather.get<|message|>\
             <atem:function_calls>\n<atem:invoke name=\"weather.get\">\n\
             </atem:invoke>\n</atem:function_calls><|eot|>",
        );

        assert_eq!(output.reasoning_text(), "think then ");
        assert!(output.normal_text().is_empty());
        let call = first_call(&output);
        assert_eq!(call.name.as_deref(), Some("weather.get"));
        assert_eq!(call.arguments, "{}");
    }

    #[test]
    fn muse_glimmer_glued_bare_tool_header_is_body_text() {
        // A bare header glued to a non-whitespace byte is not at a bare-header
        // position, so it stays body text -- and a chunk boundary right before
        // `to=` must not change that.
        let output = assert_chunking_invariant(
            " to=self<|message|>think xto=calc<|message|>\
             <atem:invoke name=\"calc\"></atem:invoke><|eom|>\
             <|start|>assistant to=user<|message|>done<|eot|>",
        );

        assert!(output.calls().is_empty());
        assert_eq!(
            output.reasoning_text(),
            "think xto=calc<|message|><atem:invoke name=\"calc\"></atem:invoke>"
        );
        assert_eq!(output.normal_text(), "done");
    }

    #[test]
    fn muse_glimmer_glued_bare_tool_header_split_before_to() {
        // The chunk-boundary case of the model defect: the header is glued to
        // the preceding non-whitespace byte, so no channel switch happens.
        let mut parser = test_parser();
        let mut output = parser.parse_chunk(" to=self<|message|>think x").unwrap();
        output.append(
            parser
                .parse_chunk("to=calc<|message|><atem:invoke name=\"calc\"></atem:invoke><|eom|>")
                .unwrap(),
        );
        output.append(
            parser.parse_chunk("<|start|>assistant to=user<|message|>done<|eot|>").unwrap(),
        );
        output.append(parser.finish().unwrap());

        assert!(output.calls().is_empty());
        assert_eq!(
            output.reasoning_text(),
            "think xto=calc<|message|><atem:invoke name=\"calc\"></atem:invoke>"
        );
        assert_eq!(output.normal_text(), "done");
    }

    #[test]
    fn muse_glimmer_idle_glued_bare_headers_are_literal_text() {
        let output =
            assert_chunking_invariant(" to=user<|message|>a<|eom|>xto=self<|message|>R<|eom|>");

        assert_eq!(output.normal_text(), "axto=self<|message|>R");
        assert!(output.reasoning_text().is_empty());
    }

    #[test]
    fn muse_glimmer_idle_whitespace_before_bare_header_stays_text() {
        // The whitespace before a bare header is body text: whether it shares
        // a chunk with the header or not must not change the output.
        let output = assert_chunking_invariant(" to=user<|message|>a<|eom|>x <|message|>b<|eot|>");
        assert_eq!(output.normal_text(), "ax b");

        let mut parser = test_parser();
        let mut output = parser.parse_chunk(" to=user<|message|>a<|eom|>x").unwrap();
        output.append(parser.parse_chunk(" ").unwrap());
        output.append(parser.parse_chunk("<|message|>b<|eot|>").unwrap());
        output.append(parser.finish().unwrap());
        assert_eq!(output.normal_text(), "ax b");
    }

    #[test]
    fn muse_glimmer_bare_header_right_after_marker_still_fires() {
        // A marker event is a legal bare-header position even without any
        // intervening whitespace.
        let output = assert_chunking_invariant(
            " to=user<|message|>a<|eom|>to=self<|message|>r<|eom|>\
             <|start|>assistant to=user<|message|>b<|eot|>",
        );

        assert_eq!(output.normal_text(), "ab");
        assert_eq!(output.reasoning_text(), "r");
    }

    #[test]
    fn muse_glimmer_marker_run_holdback_stays_linear() {
        // A run of marker starts holds back only its last `<|`: per delta the
        // holdback is a single pass, not a fixpoint rescan over the whole run
        // (which made "<|" * n in 1-byte deltas effectively cubic).
        let text = format!(
            " to=user<|message|>a{}<|eom|>tail<|eot|>",
            "<|".repeat(2048)
        );
        let chunks = char_chunks(&text, 1);
        let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let streamed = collect_stream(&mut test_parser(), &chunk_refs);
        let whole = collect_stream(&mut test_parser(), &[&text]);

        assert_eq!(streamed, whole);
        assert_eq!(whole.normal_text(), format!("a{}tail", "<|".repeat(2048)));
    }

    #[test]
    fn muse_glimmer_holdback_marker_and_to_fragment_compose() {
        // A ` to=…` fragment behind a partial marker stays held until the
        // marker question resolves.
        let output = assert_chunking_invariant(" to=user<|message|>v to=skill<|eom|><|eot|>");
        assert_eq!(output.normal_text(), "v to=skill");

        // At stream end the partial marker is dropped and the fragment flushes.
        let mut parser = test_parser();
        let mut output = parser.parse_chunk(" to=user<|message|>v to=skill<|").unwrap();
        output.append(parser.finish().unwrap());
        assert_eq!(output.normal_text(), "v to=skill");
    }

    #[test]
    fn muse_glimmer_overlong_recipient_is_body_text() {
        // A recipient run longer than the candidate cap can never be a header.
        let text = format!(" to={}<|message|>x<|eot|>", "a".repeat(2048));
        let output = assert_chunking_invariant(&text);

        assert!(output.calls().is_empty());
        assert_eq!(
            output.normal_text(),
            format!(" to={}<|message|>x", "a".repeat(2048))
        );
    }

    #[test]
    fn muse_glimmer_overlong_invoke_attribute_is_content() {
        let body = format!(
            "<atem:invoke name=\"{}\">unclosed</atem:invoke>",
            "a".repeat(2048)
        );
        let output = assert_chunking_invariant(&format!("<|message|>{body}<|eot|>"));

        assert!(output.calls().is_empty());
        assert_eq!(output.normal_text(), body);
    }

    #[test]
    fn muse_glimmer_overlong_invoke_word_is_content() {
        let body = format!("<atem:invoke{}></atem:invoke>", "a".repeat(2048));
        let output = assert_chunking_invariant(&format!("<|message|>x{body}<|eot|>"));

        assert!(output.calls().is_empty());
        assert_eq!(output.normal_text(), format!("x{body}"));
    }

    #[test]
    fn muse_glimmer_overlong_whitespace_run_is_text() {
        let text = format!(
            " to=user<|message|>a<|eom|>{}to=self<|message|>r<|eom|>\
             <|start|>assistant to=user<|message|>b<|eot|>",
            " ".repeat(2048)
        );
        let output = assert_chunking_invariant(&text);

        assert_eq!(output.normal_text(), format!("a{}b", " ".repeat(2048)));
        assert_eq!(output.reasoning_text(), "r");
    }

    #[test]
    fn muse_glimmer_overlong_candidate_streams_instead_of_holding() {
        // Once a candidate run exceeds the cap it is text and streams out; it
        // is not held (and re-scanned per delta) waiting for a terminator
        // that may never come.
        let mut parser = test_parser();
        let output = parser
            .parse_chunk(&format!(
                "<|message|><atem:invoke name=\"{}\"",
                "a".repeat(4096)
            ))
            .unwrap();

        assert!(!output.normal_text().is_empty());
    }

    #[test]
    fn muse_glimmer_whitespace_run_before_bare_header_after_text_stays_text() {
        // Whitespace between body text and a bare header is body text; only a
        // structural position (stream start, after a marker) lets the header
        // absorb leading whitespace.
        let output = assert_chunking_invariant(
            " to=user<|message|>a<|eom|>x  to=self<|message|>r<|eom|>\
             <|start|>assistant to=user<|message|>b<|eot|>",
        );

        assert_eq!(output.normal_text(), "ax  b");
        assert_eq!(output.reasoning_text(), "r");
    }

    #[test]
    fn muse_glimmer_marker_anchored_bare_header_absorbs_leading_whitespace() {
        let output = assert_chunking_invariant(
            " to=user<|message|>a<|eom|> to=self<|message|>r<|eom|>\
             <|start|>assistant to=user<|message|>b<|eot|>",
        );

        assert_eq!(output.normal_text(), "ab");
        assert_eq!(output.reasoning_text(), "r");
    }

    #[test]
    fn muse_glimmer_whitespace_before_wrapperless_invoke_stays_content() {
        // Whitespace between content and a wrapper-less reclassified invoke is
        // content; only whitespace inside the wrapper is structural.
        let output = assert_chunking_invariant(
            "<|message|>pre <atem:invoke name=\"calc\"></atem:invoke><|eom|>\
             <|start|>assistant to=user<|message|>done<|eot|>",
        );
        assert_eq!(output.normal_text(), "pre done");
        assert_eq!(first_call(&output).name.as_deref(), Some("calc"));

        let output = assert_chunking_invariant(
            "<|message|>pre <atem:function_calls>\n<atem:invoke name=\"calc\"></atem:invoke>\n\
             </atem:function_calls><|eom|><|start|>assistant to=user<|message|>done<|eot|>",
        );
        assert_eq!(output.normal_text(), "pre done");
        assert_eq!(first_call(&output).name.as_deref(), Some("calc"));
    }

    #[test]
    fn muse_glimmer_finish_drops_complete_bare_header_without_body() {
        // A complete bare header at stream end opened a channel that never got
        // a body: it is dropped, not flushed as text (Python parity).
        let mut parser = test_parser();
        let mut output =
            parser.parse_chunk(" to=self<|message|>thinking to=calc<|message|>").unwrap();
        output.append(parser.finish().unwrap());
        assert_eq!(output.reasoning_text(), "thinking ");
        assert!(output.calls().is_empty());

        // A partial header fragment still flushes as text.
        let mut parser = test_parser();
        let mut output = parser.parse_chunk(" to=self<|message|>thinking to=calc").unwrap();
        output.append(parser.finish().unwrap());
        assert_eq!(output.reasoning_text(), "thinking to=calc");

        // A glued header was never a header: it is body text, flushed as such.
        let mut parser = test_parser();
        let mut output =
            parser.parse_chunk(" to=self<|message|>thinking xto=calc<|message|>").unwrap();
        output.append(parser.finish().unwrap());
        assert_eq!(output.reasoning_text(), "thinking xto=calc<|message|>");
    }

    #[test]
    fn muse_glimmer_parameter_tag_word_boundary() {
        // `<atem:parameterx …>` is not a parameter tag (the same `\b` guard
        // the invoke opener has).
        let text = format!(
            "<|start|>assistant to=calc<|message|>\
             <atem:function_calls>\n<atem:invoke name=\"calc\">\n{}{}</atem:invoke>\n\
             </atem:function_calls><|eot|>",
            "<atem:parameterx name=\"bad\">1</atem:parameter>\n",
            param("x", "1"),
        );
        let output = assert_chunking_invariant(&text);

        assert_eq!(first_call(&output).arguments, r#"{"x":1}"#);
    }

    #[test]
    fn muse_glimmer_quoted_bare_header_without_atem_stays_reasoning() {
        let output = assert_chunking_invariant(
            " to=self<|message|>I write to=weather.get<|message|> to start a call, \
             but to=user<|message|> is the answer channel.<|eom|>\
             <|start|>assistant to=user<|message|>noted<|eot|>",
        );

        assert_eq!(
            output.reasoning_text(),
            "I write to=weather.get<|message|> to start a call, \
             but to=user<|message|> is the answer channel."
        );
        assert!(output.calls().is_empty());
        assert_eq!(output.normal_text(), "noted");
    }

    #[test]
    fn muse_glimmer_framed_user_header_closes_unterminated_reasoning() {
        let output = assert_chunking_invariant(
            " to=self<|message|>some thinking\
             <|start|>assistant to=user<|message|>the answer<|eot|>",
        );

        assert_eq!(output.reasoning_text(), "some thinking");
        assert_eq!(output.normal_text(), "the answer");
    }

    #[test]
    fn muse_glimmer_quoted_start_marker_is_literal_body_text() {
        let output = assert_chunking_invariant(
            " to=user<|message|>the marker <|start|> quoted here is text<|eot|>",
        );

        assert_eq!(
            output.normal_text(),
            "the marker <|start|> quoted here is text"
        );
    }

    #[test]
    fn muse_glimmer_initialize_closed_reasoning_prefill_starts_idle() {
        let mut parser = test_parser();
        let prompt = tokenizer()
            .encode(
                "<|start|>user<|message|>hi<|eom|>\
                 <|start|>assistant to=self<|message|>Prior.<|eom|>",
                false,
            )
            .unwrap();
        parser.initialize(&prompt).unwrap();

        // The prompt's reasoning channel is closed: it must not seed reasoning
        // mode, and the bare `to=user` header starts a content channel.
        let output = parser.parse_complete(" to=user<|message|>answer<|eot|>").unwrap();

        assert_eq!(output.normal_text(), "answer");
        assert!(output.reasoning_text().is_empty());
    }

    #[test]
    fn muse_glimmer_initialize_open_user_prefill_starts_in_content() {
        let mut parser = test_parser();
        let prompt = tokenizer()
            .encode(
                "<|start|>user<|message|>hi<|eom|><|start|>assistant to=user<|message|>",
                false,
            )
            .unwrap();
        parser.initialize(&prompt).unwrap();

        let output = parser.parse_complete("the answer<|eot|>").unwrap();

        assert_eq!(output.normal_text(), "the answer");
        assert!(output.reasoning_text().is_empty());
    }

    #[test]
    fn muse_glimmer_abandoned_reasoning_prefill_emits_no_leading_separator() {
        // The prefilled reasoning block produced no reasoning bytes, so a
        // framed `to=self` header that abandons it must not emit a leading
        // "\n".
        let mut parser = test_parser();
        let prompt = tokenizer()
            .encode(
                "<|start|>user<|message|>hi<|eom|><|start|>assistant to=self<|message|>Prior.",
                false,
            )
            .unwrap();
        parser.initialize(&prompt).unwrap();

        let output = parser
            .parse_complete(
                "<|start|>assistant to=self<|message|>more<|eom|>\
                 <|start|>assistant to=user<|message|>done<|eot|>",
            )
            .unwrap();

        assert_eq!(output.reasoning_text(), "more");
        assert_eq!(output.normal_text(), "done");
    }

    #[test]
    fn muse_glimmer_empty_reasoning_block_emits_no_separator() {
        let output = assert_chunking_invariant(
            " to=self<|message|>a<|eom|><|start|>assistant to=self<|message|><|eom|>\
             <|start|>assistant to=self<|message|>b<|eom|>\
             <|start|>assistant to=user<|message|>done<|eot|>",
        );

        assert_eq!(output.reasoning_text(), "a\nb");
        assert_eq!(output.normal_text(), "done");
    }

    #[test]
    fn muse_glimmer_many_reasoning_blocks_get_lazy_separators() {
        let mut text = String::new();
        for i in 0..1000 {
            text.push_str(&format!(" to=self<|message|>block{i}<|eom|>"));
        }
        text.push_str("<|start|>assistant to=user<|message|>done<|eot|>");
        let output = assert_chunking_invariant(&text);

        let expected: Vec<String> = (0..1000).map(|i| format!("block{i}")).collect();
        assert_eq!(output.reasoning_text(), expected.join("\n"));
        assert_eq!(output.normal_text(), "done");
    }

    #[test]
    fn muse_glimmer_initialize_open_reasoning_prefill_continues_reasoning() {
        let mut parser = test_parser();
        let prompt = tokenizer()
            .encode(
                "<|start|>user<|message|>hi<|eom|><|start|>assistant to=self<|message|>Prior.",
                false,
            )
            .unwrap();
        parser.initialize(&prompt).unwrap();

        // The prefilled body is prompt text and is not re-emitted; the next
        // `to=self` block is separated from it by "\n".
        let output = parser
            .parse_complete(
                " continued<|eom|><|start|>assistant to=self<|message|>more<|eom|>\
                 <|start|>assistant to=user<|message|>done<|eot|>",
            )
            .unwrap();

        assert_eq!(output.reasoning_text(), " continued\nmore");
        assert_eq!(output.normal_text(), "done");
    }

    #[test]
    fn muse_glimmer_initialize_open_tool_prefill_starts_in_tool_channel() {
        let mut parser = test_parser();
        let prompt = tokenizer()
            .encode(
                "<|start|>user<|message|>hi<|eom|><|start|>assistant to=weather.get<|message|>",
                false,
            )
            .unwrap();
        parser.initialize(&prompt).unwrap();

        let output = parser
            .parse_complete(
                "<atem:function_calls>\n<atem:invoke name=\"weather.get\">\n</atem:invoke>\n\
                 </atem:function_calls><|eot|>",
            )
            .unwrap();

        let call = first_call(&output);
        assert_eq!(call.name.as_deref(), Some("weather.get"));
        assert_eq!(call.arguments, "{}");
    }

    #[test]
    fn muse_glimmer_initialize_recipient_only_tool_prefill_opens_strict_tool_channel() {
        let prompt = tokenizer()
            .encode(
                "<|start|>user<|message|>hi<|eom|><|start|>assistant to=weather.get",
                false,
            )
            .unwrap();
        let body = "<|message|><atem:function_calls>\n<atem:invoke name=\"weather.get\">\n";

        let mut parser = test_parser();
        parser.initialize(&prompt).unwrap();
        let output = parser
            .parse_complete(&format!(
                "{body}</atem:invoke>\n</atem:function_calls><|eot|>"
            ))
            .unwrap();

        // The bare `<|message|>` completes the prefilled tool header rather
        // than opening untagged content.
        let call = first_call(&output);
        assert_eq!(call.name.as_deref(), Some("weather.get"));
        assert_eq!(call.arguments, "{}");
        assert!(output.normal_text().is_empty());

        // Strict, unlike a reclassified channel: truncation mid-call is an error.
        let mut parser = test_parser();
        parser.initialize(&prompt).unwrap();
        parser.parse_chunk(body).unwrap();
        let error = parser.finish().unwrap_err();
        assert!(error.to_report_string().contains("incomplete Muse Glimmer tool call"));
    }

    #[test]
    fn muse_glimmer_initialize_recipient_only_reasoning_prefill_opens_reasoning() {
        let mut parser = test_parser();
        let prompt = tokenizer()
            .encode(
                "<|start|>user<|message|>hi<|eom|><|start|>assistant to=self",
                false,
            )
            .unwrap();
        parser.initialize(&prompt).unwrap();

        let output = parser
            .parse_complete(
                "<|message|>thinking<|eom|><|start|>assistant to=self<|message|>more<|eom|>\
                 <|start|>assistant to=user<|message|>done<|eot|>",
            )
            .unwrap();

        // The completed prefill is the first reasoning block, so the next one
        // is separated from it by "\n".
        assert_eq!(output.reasoning_text(), "thinking\nmore");
        assert_eq!(output.normal_text(), "done");
    }

    #[test]
    fn muse_glimmer_initialize_recipient_only_prefill_yields_to_framed_header() {
        let mut parser = test_parser();
        let prompt = tokenizer()
            .encode(
                "<|start|>user<|message|>hi<|eom|><|start|>assistant to=weather.get",
                false,
            )
            .unwrap();
        parser.initialize(&prompt).unwrap();

        // The model abandons the forced header; the framed one it emits
        // instead is authoritative, not a completion of the prefill.
        let output = parser
            .parse_complete("<|start|>assistant<|message|>Sorry, I cannot do that.<|eot|>")
            .unwrap();

        assert_eq!(output.normal_text(), "Sorry, I cannot do that.");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn muse_glimmer_atem_in_untagged_content_reclassifies_to_tool_channel() {
        let output = assert_chunking_invariant(
            "<|message|>Some preamble <|start|>assistant is not it, \
             <atem:function_calls>\n<atem:invoke name=\"weather.get\">\n\
             <atem:parameter name=\"city\">Paris</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls><|eom|>\
             <|start|>assistant to=user<|message|>the answer<|eot|>",
        );

        // No ATEM markup leaks as content, and the preamble keeps its text.
        let content = output.normal_text();
        assert!(!content.contains(ATEM_PREFIX));
        assert_eq!(
            content,
            "Some preamble <|start|>assistant is not it, the answer"
        );
        let call = first_call(&output);
        assert_eq!(call.name.as_deref(), Some("weather.get"));
        assert_eq!(call.arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn muse_glimmer_content_atem_without_complete_invoke_stays_content() {
        let output = assert_chunking_invariant(
            "<|message|>look: <atem:function_calls> explains calls, \
             and <atem:invoke names a block<|eot|>",
        );

        assert_eq!(
            output.normal_text(),
            "look: <atem:function_calls> explains calls, and <atem:invoke names a block"
        );
        assert!(output.calls().is_empty());
    }

    #[test]
    fn muse_glimmer_one_delta_spanning_answer_and_tool_channel() {
        let mut parser = test_parser();
        let output = parser
            .parse_complete(
                " to=user<|message|>answer before <|eom|>\
                 <|start|>assistant to=weather.get<|message|>\
                 <atem:function_calls>\n<atem:invoke name=\"weather.get\">\n\
                 <atem:parameter name=\"city\">Paris</atem:parameter>\n\
                 </atem:invoke>\n</atem:function_calls><|eot|>",
            )
            .unwrap();

        assert_eq!(output.normal_text(), "answer before ");
        let call = first_call(&output);
        assert_eq!(call.name.as_deref(), Some("weather.get"));
        assert_eq!(call.arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn muse_glimmer_two_tool_channels_in_one_delta() {
        let mut parser = test_parser();
        let text = format!(
            " to=self<|message|>need two calls<|eom|>{}{}",
            tool_channel("math.add", &param("a", "1"), "<|eom|>"),
            tool_channel("math.mul", &param("a", "3"), EOT),
        );
        let output = parser.parse_complete(&text).unwrap();

        let calls = output.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool_index, 0);
        assert_eq!(calls[0].name.as_deref(), Some("math.add"));
        assert_eq!(calls[0].arguments, r#"{"a":1}"#);
        assert_eq!(calls[1].tool_index, 1);
        assert_eq!(calls[1].name.as_deref(), Some("math.mul"));
        assert_eq!(calls[1].arguments, r#"{"a":3}"#);
    }

    #[test]
    fn muse_glimmer_typed_parameter_values_decode() {
        let text = format!(
            "<|start|>assistant to=api.call<|message|>\
             <atem:function_calls>\n<atem:invoke name=\"api.call\">\n{}{}{}{}</atem:invoke>\n\
             </atem:function_calls><|eot|>",
            param("payload", r#"{"nested":[1,2,3]}"#),
            param("flag", "true"),
            param("ratio", "1.5"),
            param("greeting", "héllo"),
        );

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        let call = first_call(&output);
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).unwrap(),
            json!({
                "payload": { "nested": [1, 2, 3] },
                "flag": true,
                "ratio": 1.5,
                "greeting": "héllo",
            })
        );
        // Non-ASCII text is not escaped.
        assert!(call.arguments.contains("héllo"));
        // Parameters stay in emission order.
        assert!(call.arguments.starts_with(r#"{"payload":"#));
    }

    #[test]
    fn muse_glimmer_malformed_parameter_value_falls_back_to_raw_string() {
        let text = format!(
            "<|start|>assistant to=calc<|message|>\
             <atem:function_calls>\n<atem:invoke name=\"calc\">\n{}</atem:invoke>\n\
             </atem:function_calls><|eot|>",
            param("expr", "not a number"),
        );

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        assert_eq!(first_call(&output).arguments, r#"{"expr":"not a number"}"#);
    }

    #[test]
    fn muse_glimmer_invoke_without_name_is_skipped() {
        let text = format!(
            "<|start|>assistant to=calc<|message|>\
             <atem:function_calls>\n<atem:invoke>\n{}</atem:invoke>\n{}</atem:function_calls><|eot|>",
            param("x", "1"),
            invoke("real", ""),
        );

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        let calls = output.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_index, 0);
        assert_eq!(calls[0].name.as_deref(), Some("real"));
    }

    #[test]
    fn muse_glimmer_empty_name_attribute_defers_to_a_later_one() {
        let text = "<|start|>assistant to=real<|message|>\
             <atem:function_calls>\n<atem:invoke name=\"\" name=\"real\">\n\
             <atem:parameter name=\"\" name=\"k\">v</atem:parameter>\n</atem:invoke>\n\
             </atem:function_calls><|eot|>";

        let mut parser = test_parser();
        let output = parser.parse_complete(text).unwrap();

        let call = first_call(&output);
        assert_eq!(call.name.as_deref(), Some("real"));
        assert_eq!(call.arguments, r#"{"k":"v"}"#);
    }

    #[test]
    fn muse_glimmer_empty_invoke_yields_empty_object_arguments() {
        let text = tool_channel("weather.get", "", EOT);

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        assert_eq!(first_call(&output).arguments, "{}");
    }

    #[test]
    fn muse_glimmer_doubled_bare_name_collapses_when_registered() {
        let mut parser = test_parser_with_tools(&["get_weather"]);
        let output = parser
            .parse_complete(&tool_channel("get_weather.get_weather", "", EOT))
            .unwrap();

        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn muse_glimmer_namespaced_and_unknown_names_pass_through() {
        let mut parser = test_parser_with_tools(&["get_weather"]);
        let output = parser
            .parse_complete(&format!(
                "{}{}",
                tool_channel("foo.get_weather", "", "<|eom|>"),
                tool_channel("weather.get", "", EOT),
            ))
            .unwrap();

        // Suffix-only matching is not safe: `foo.get_weather` stays as emitted.
        assert_eq!(output.calls()[0].name.as_deref(), Some("foo.get_weather"));
        assert_eq!(output.calls()[1].name.as_deref(), Some("weather.get"));
    }

    #[test]
    fn muse_glimmer_doubled_name_passes_through_without_registered_tools() {
        let mut parser = test_parser();
        let output = parser
            .parse_complete(&tool_channel("get_weather.get_weather", "", EOT))
            .unwrap();

        assert_eq!(
            first_call(&output).name.as_deref(),
            Some("get_weather.get_weather")
        );
    }

    #[test]
    fn muse_glimmer_ignores_output_after_turn_end() {
        let output = assert_chunking_invariant(
            " to=user<|message|>answer<|eot|>garbage to=self<|message|>more<|eom|>",
        );

        assert_eq!(output.normal_text(), "answer");
        assert!(output.reasoning_text().is_empty());
    }

    #[test]
    fn muse_glimmer_finish_flushes_held_back_to_fragment() {
        let mut parser = test_parser();
        let mut output = parser.parse_chunk(" to=self<|message|>thinking hard about to").unwrap();
        output.append(parser.finish().unwrap());

        // The held-back ` to` fragment can no longer grow into a header.
        assert_eq!(output.reasoning_text(), "thinking hard about to");
    }

    #[test]
    fn muse_glimmer_finish_strips_trailing_partial_marker() {
        let mut parser = test_parser();
        let mut output = parser.parse_chunk(" to=user<|message|>answer<|eo").unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn muse_glimmer_finish_fails_mid_tool_call() {
        let mut parser = test_parser();
        parser
            .parse_chunk(
                " to=weather.get<|message|><atem:function_calls>\n\
                 <atem:invoke name=\"weather.get\">\n",
            )
            .unwrap();

        let error = parser.finish().unwrap_err();

        assert!(error.to_report_string().contains("incomplete Muse Glimmer tool call"));
    }

    #[test]
    fn muse_glimmer_finish_after_truncated_channel_keeps_complete_calls() {
        let mut parser = test_parser();
        let mut output = parser
            .parse_chunk(
                " to=weather.get<|message|><atem:function_calls>\n\
                 <atem:invoke name=\"weather.get\">\n</atem:invoke>\n",
            )
            .unwrap();
        output.append(parser.finish().unwrap());

        let call = first_call(&output);
        assert_eq!(call.name.as_deref(), Some("weather.get"));
        assert_eq!(call.arguments, "{}");
    }

    #[test]
    fn muse_glimmer_plain_text_falls_through_as_text() {
        let output = collect_stream(&mut test_parser(), &["plain ", "answer"]);

        assert_eq!(output.normal_text(), "plain answer");
        assert!(output.reasoning_text().is_empty());
        assert!(output.calls().is_empty());
    }

    #[test]
    fn muse_glimmer_streaming_emits_text_incrementally() {
        let mut parser = test_parser();

        let first = parser.parse_chunk(" to=user<|message|>Hel").unwrap();
        assert_eq!(first.normal_text(), "Hel");

        // A trailing partial marker is held back, not emitted.
        let second = parser.parse_chunk("lo<|eo").unwrap();
        assert_eq!(second.normal_text(), "lo");

        // `<|eom|>` closes the channel; text after it has no framing and falls
        // through as content.
        let third = parser.parse_chunk("m|>rest<|eot|>").unwrap();
        assert_eq!(third.normal_text(), "rest");

        let tail = parser.finish().unwrap();
        assert_eq!(tail.normal_text(), "");
    }

    #[test]
    fn muse_glimmer_reset_returns_buffered_text() {
        let mut parser = test_parser();
        parser.parse_chunk(" to=user<|message|>answer<|eo").unwrap();

        let raw = parser.reset();

        assert_eq!(raw, "<|eo");
    }

    #[test]
    fn muse_glimmer_finish_fully_resets_cross_delta_state() {
        // A parser reused after finish() (without initialize/reset) must
        // behave exactly like a fresh one: mode, anchor, and counters all
        // reset, so a bare header at the new stream start fires. Stream 1
        // deliberately ends on non-whitespace with no held fragment: a stale
        // anchor would keep the bare header from firing (no call), unlike a
        // fresh parser.
        let reused = {
            let mut parser = test_parser();
            parser.parse_chunk(" to=self<|message|>think").unwrap();
            parser.finish().unwrap();
            parser
                .parse_chunk("to=calc<|message|><atem:invoke name=\"calc\"></atem:invoke><|eot|>")
                .unwrap()
        };
        let fresh = test_parser()
            .parse_complete("to=calc<|message|><atem:invoke name=\"calc\"></atem:invoke><|eot|>")
            .unwrap();

        assert_eq!(reused, fresh);
        assert_eq!(reused.calls().len(), 1);
        assert_eq!(reused.calls()[0].name.as_deref(), Some("calc"));
    }

    #[test]
    fn muse_glimmer_nameless_parameter_is_skipped_without_panic() {
        let text = format!(
            "<|start|>assistant to=calc<|message|>\
             <atem:function_calls>\n<atem:invoke name=\"calc\">\n\
             <atem:parameter>oops</atem:parameter>\n{}</atem:invoke>\n\
             </atem:function_calls><|eot|>",
            param("x", "1"),
        );

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        assert_eq!(first_call(&output).arguments, r#"{"x":1}"#);
    }

    #[test]
    fn muse_glimmer_bare_marker_prefix_in_parameter_value_is_text() {
        // Only a complete framing marker is structural inside an invoke
        // (Python `_INVOKE_RE` parity).
        let text = format!(
            "{}<|start|>assistant to=user<|message|>done<|eot|>",
            tool_channel("calc", &param("code", "a <| b"), "<|eom|>"),
        );
        let output = assert_chunking_invariant(&text);

        assert_eq!(first_call(&output).arguments, r#"{"code":"a <| b"}"#);
        assert_eq!(output.normal_text(), "done");
    }

    #[test]
    fn muse_glimmer_long_parameter_value_round_trips_through_small_chunks() {
        // The invoke is buffered whole, so a ~20KB value re-parses on every
        // delta: the resumable body scan must not change what comes out.
        let value: String = (0..1_200).map(|i| format!("line {i}: a <| b\n")).collect();
        let output =
            assert_chunking_invariant(&tool_channel("notes.write", &param("body", &value), EOT));

        assert_eq!(
            first_call(&output).arguments,
            json!({ "body": value }).to_string()
        );
    }

    #[test]
    fn muse_glimmer_tool_channel_recovers_from_rejected_invoke_opener() {
        let text = format!(
            "<|start|>assistant to=calc<|message|><atem:function_calls>\n\
             note <atem:invoker> here\n{}</atem:function_calls><|eom|>\
             <|start|>assistant to=user<|message|>done<|eot|>",
            invoke("calc", &param("x", "1")),
        );
        let output = assert_chunking_invariant(&text);

        assert_eq!(first_call(&output).arguments, r#"{"x":1}"#);
        assert_eq!(output.normal_text(), "done");
    }

    #[test]
    fn muse_glimmer_quoted_framing_in_invoke_lets_channel_close() {
        let output = assert_chunking_invariant(
            "<|start|>assistant to=calc<|message|><atem:function_calls>\n\
             <atem:invoke name=\"calc\">\ntext<|eom|>\
             <|start|>assistant to=user<|message|>done<|eot|>",
        );

        assert!(output.calls().is_empty());
        assert_eq!(output.normal_text(), "done");
    }

    #[test]
    fn muse_glimmer_newline_anchored_bare_tool_header_switches() {
        let output = assert_chunking_invariant(
            " to=self<|message|>think\nto=weather.get<|message|>\
             <atem:function_calls>\n<atem:invoke name=\"weather.get\">\n\
             </atem:invoke>\n</atem:function_calls><|eot|>",
        );

        assert_eq!(output.reasoning_text(), "think\n");
        assert_eq!(first_call(&output).name.as_deref(), Some("weather.get"));
    }

    #[test]
    fn muse_glimmer_atem_in_user_answer_stays_content() {
        // Python contract: an invoke echoed inside a `to=user` final answer
        // is never a real tool call.
        let body = format!(
            "Example: <atem:function_calls>\n{}</atem:function_calls> like that.",
            invoke("get_weather", &param("city", "Paris")),
        );
        let output = assert_chunking_invariant(&format!(" to=user<|message|>{body}<|eot|>"));

        assert!(output.calls().is_empty());
        assert_eq!(output.normal_text(), body);
    }

    #[test]
    fn muse_glimmer_finish_after_truncated_closing_marker_keeps_complete_calls() {
        let mut parser = test_parser();
        let mut output = parser
            .parse_chunk(
                " to=weather.get<|message|><atem:function_calls>\n\
                 <atem:invoke name=\"weather.get\">\n</atem:invoke>\n</atem:func",
            )
            .unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(first_call(&output).name.as_deref(), Some("weather.get"));
    }

    #[test]
    fn muse_glimmer_finish_strips_truncated_framed_header() {
        let mut parser = test_parser();
        let mut output = parser.parse_chunk(" to=user<|message|>answer<|start|>assist").unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(output.normal_text(), "answer");
    }
}
