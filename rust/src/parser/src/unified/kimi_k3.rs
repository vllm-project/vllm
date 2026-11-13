// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser for the Kimi K3 (XTML) chat format.
//!
//! Original Python implementations:
//! - `vllm/reasoning/kimi_k3_reasoning_parser.py`
//! - `vllm/tool_parsers/kimi_k3_tool_parser.py`
//!
//! K3 wraps one assistant message into XTML channels built from the dedicated
//! special tokens `<|open|>`, `<|close|>`, and `<|sep|>`:
//!
//! ```text
//! <|open|>think<|sep|>reasoning<|close|>think<|sep|>
//! <|open|>response<|sep|>visible answer<|close|>response<|sep|>
//! <|open|>tools<|sep|>
//!   <|open|>call tool="get_weather" index="1"<|sep|>
//!     <|open|>argument key="city" type="string"<|sep|>Hangzhou<|close|>argument<|sep|>
//!   <|close|>call<|sep|>
//! <|close|>tools<|sep|>
//! <|close|>message<|sep|>
//! ```
//!
//! In chat serving the generation prompt ends with `<|open|>think<|sep|>`
//! (thinking) or `<|open|>response<|sep|>` (instruct), so the model output
//! starts *inside* that channel without re-emitting the open tag;
//! [`UnifiedParser::initialize`] detects this from the prompt token IDs.
//!
//! Channel markers are structural only when their `<|open|>` / `<|close|>` /
//! `<|sep|>` bytes come from the dedicated special tokens. The model can spell
//! the same text with ordinary BPE tokens (for example while quoting a
//! configuration snippet in its reasoning); such text stays content in the
//! current channel. The parser therefore expects token-attributed input from
//! the incremental detokenizer: fed unattributed text, it treats every marker
//! as content. Channel names such as `think` are ordinary text either way.
//!
//! Arguments mirror the renderer's type tagging (inverse encoding). A
//! `type="string"` value is raw text, which streams as it arrives, escaped into
//! a JSON string. Any other value is JSON text, whose validity is known only at
//! its end, so it is written once complete: as spelled when it is valid JSON,
//! and otherwise as a JSON string of the text, like the Python parser. A raw
//! `json` block is the whole arguments object and streams through unmodified.
//! Attribute values reverse the renderer escaping (`&quot;` before `&amp;`).
//!
//! String argument and response bodies are emitted raw, so in text a value that
//! literally contains `<|close|>argument<|sep|>` or `<|close|>response<|sep|>`
//! is indistinguishable from a real closing marker, a known limitation of the
//! Python parser. With token attribution only the special tokens close a block.

mod grammar;

use vllm_tokenizer::{DecodedText, DynTokenizer, Tokenizer};
use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, eof, preceded, repeat, seq, terminated};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext};
use winnow::prelude::*;
use winnow::token::{literal, rest, take_till, take_while};

use super::{Result, UnifiedParser, UnifiedParserError, UnifiedParserOutput, special_token};
use crate::output_grammar::{self, BuiltOutputGrammar, OutputGrammarContext};
use crate::tool::{JsonStyle, OpenToolCall, Tool};
use crate::unified::parsing_failed;
use crate::utils::{
    Attributed, AttributedExt as _, Marker, MarkerScanState, SpecialToken,
    parse_buffered_event_attributed, safe_text_len, safe_text_len_mul, take_until_marker,
};

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
const END_OF_MSG: &str = "<|end_of_msg|>";

const THINK_OPEN: &str = "<|open|>think<|sep|>";
const THINK_CLOSE: &str = "<|close|>think<|sep|>";
const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";
const CALL_CLOSE: &str = "<|close|>call<|sep|>";
const ARG_OPEN: &str = "<|open|>argument";
const ARG_CLOSE: &str = "<|close|>argument<|sep|>";
const JSON_OPEN: &str = "<|open|>json";
const JSON_CLOSE: &str = "<|close|>json<|sep|>";

/// Channel tags are a couple of text tokens; longer `<|open|>…<|sep|>` spans in
/// the prompt tail (attribute-bearing message opens, message bodies) never name
/// a generation channel.
const MAX_PREFILL_TAG_TOKENS: usize = 8;

type KimiK3Input<'i> = Attributed<'i, Markers>;

/// The K3 structural special tokens, resolved once from the active tokenizer.
struct SpecialTokens {
    open: SpecialToken,
    close: SpecialToken,
    sep: SpecialToken,
    end_of_msg: SpecialToken,
}

impl SpecialTokens {
    fn new(tokenizer: &dyn Tokenizer) -> Result<Self> {
        Ok(Self {
            open: special_token(tokenizer, OPEN)?,
            close: special_token(tokenizer, CLOSE)?,
            sep: special_token(tokenizer, SEP)?,
            end_of_msg: special_token(tokenizer, END_OF_MSG)?,
        })
    }
}

/// The K3 channel markers, with their special-token segments guarded by the
/// active tokenizer's IDs.
///
/// One definition serves both the `alt` branches that consume a marker and the
/// safe-text scans that stop in front of one, so the two can never disagree on
/// whether a spelling is structural.
#[derive(Debug)]
struct Markers {
    think_open: Marker,
    think_close: Marker,
    response_open: Marker,
    response_close: Marker,
    tools_open: Marker,
    tools_close: Marker,
    message_close: Marker,
    end_of_msg: Marker,
    /// `<|open|>call`; the tag attributes and the terminating `<|sep|>` follow.
    call_open: Marker,
    call_close: Marker,
    /// `<|open|>argument`; the tag attributes and the terminating `<|sep|>`
    /// follow.
    argument_open: Marker,
    argument_close: Marker,
    /// `<|open|>json`; the tag attributes and the terminating `<|sep|>` follow.
    json_open: Marker,
    json_close: Marker,
    sep: Marker,
}

impl Markers {
    fn new(tokens: &SpecialTokens) -> Self {
        let open =
            |tag: &str| Marker::special(&tokens.open).then_text(tag).then_special(&tokens.sep);
        let close =
            |tag: &str| Marker::special(&tokens.close).then_text(tag).then_special(&tokens.sep);

        Self {
            think_open: open("think"),
            think_close: close("think"),
            response_open: open("response"),
            response_close: close("response"),
            tools_open: open("tools"),
            tools_close: close("tools"),
            message_close: close("message"),
            end_of_msg: Marker::special(&tokens.end_of_msg),
            call_open: Marker::special(&tokens.open).then_text("call"),
            call_close: close("call"),
            argument_open: Marker::special(&tokens.open).then_text("argument"),
            argument_close: close("argument"),
            json_open: Marker::special(&tokens.open).then_text("json"),
            json_close: close("json"),
            sep: Marker::special(&tokens.sep),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum KimiK3Event {
    Text,
    Reasoning,
    /// Structural noise consumed without emitting anything.
    Skip,
    ThinkOpen,
    ThinkClose,
    ResponseOpen,
    ResponseClose,
    ToolsOpen,
    ToolsClose,
    /// The assistant message closed; everything after it is ignored.
    MessageEnd,
    CallOpen {
        name: String,
    },
    /// An `argument` block opens; `is_string` tells whether its `type`
    /// attribute is `string`, the default.
    ArgumentOpen {
        key: String,
        is_string: bool,
    },
    /// The drained text continues the open string argument value.
    ArgumentText,
    ArgumentClose,
    /// The complete value of a typed (non-string) argument, through its close
    /// marker.
    TypedValue {
        value: String,
    },
    /// A raw `json` block, the whole arguments object, opens.
    JsonOpen,
    /// The drained text continues the raw `json` block.
    JsonText,
    JsonClose,
    CallClose,
}

#[derive(Debug, Default)]
enum KimiK3Mode {
    /// Before any channel opens: channel opens are expected, but raw text
    /// falls through as visible text so marker-free output still streams.
    #[default]
    Idle,
    /// Inside the `think` channel the model opened.
    Reasoning,
    /// Inside the `think` channel the prompt opened, before the model emitted
    /// any channel marker. The model may skip thinking and open `response`
    /// directly, which also ends this channel; a generated `think` open moves
    /// to [`KimiK3Mode::Reasoning`].
    PrefilledReasoning,
    /// Inside the `response` channel.
    Response,
    /// After the `response` (or `tools`) channel closed: only a `tools`
    /// channel or the message close may follow, and noise is never content.
    Epilogue,
    /// Inside the `tools` channel, between `call` blocks.
    Tools,
    /// Inside one `call` block, streaming its arguments.
    Call {
        /// The call being emitted, or `None` for a call without a tool name,
        /// which is parsed but dropped.
        call: Option<OpenToolCall>,
        body: CallBody,
    },
    /// After the message closed: ignore the rest (EOS leakage guard).
    Done,
}

/// Where the parser is inside a `call` block.
#[derive(Debug)]
enum CallBody {
    /// Before the first argument, where a raw `json` block may open instead.
    Start,
    /// Between `argument` blocks.
    Arguments,
    /// Inside a string `argument` value, which streams as it arrives.
    Argument,
    /// Inside a typed `argument` value, which is written once complete.
    TypedArgument {
        key: String,
        value_end_scan: MarkerScanState,
    },
    /// Inside a raw `json` block.
    Json,
    /// After a raw `json` block.
    AfterJson,
}

/// Unified parser for Kimi K3 XTML think / response / tools channels.
pub struct KimiK3UnifiedParser {
    buffer: DecodedText,
    mode: KimiK3Mode,
    markers: Markers,
    /// Number of calls emitted in the current response.
    emitted_call_count: usize,
    tokenizer: DynTokenizer,
    special_tokens: SpecialTokens,
}

impl KimiK3UnifiedParser {
    /// Create a Kimi K3 parser expecting token-attributed input.
    pub fn new(_tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let special_tokens = SpecialTokens::new(tokenizer.as_ref())?;

        Ok(Self {
            buffer: DecodedText::default(),
            mode: KimiK3Mode::default(),
            markers: Markers::new(&special_tokens),
            emitted_call_count: 0,
            tokenizer,
            special_tokens,
        })
    }

    /// Detect the prefilled generation channel from the prompt tail.
    ///
    /// `add_generation_prompt` ends the prompt with `<|open|>think<|sep|>` or
    /// `<|open|>response<|sep|>`, so generation starts inside that channel and
    /// never re-emits the open tag. Locate the last `<|open|>…<|sep|>` pair and
    /// decode the short tag between the two structural tokens.
    fn initialize_mode(&mut self, prompt_token_ids: &[u32]) {
        self.mode = KimiK3Mode::Idle;

        let Some(sep_pos) =
            prompt_token_ids.iter().rposition(|&id| id == self.special_tokens.sep.id)
        else {
            return;
        };
        let Some(open_pos) = prompt_token_ids[..sep_pos]
            .iter()
            .rposition(|&id| id == self.special_tokens.open.id)
        else {
            return;
        };
        let tag_ids = &prompt_token_ids[open_pos + 1..sep_pos];
        if tag_ids.is_empty() || tag_ids.len() > MAX_PREFILL_TAG_TOKENS {
            return;
        }
        let Ok(tag) = self.tokenizer.decode(tag_ids, /* skip_special_tokens */ false) else {
            return;
        };
        self.mode = match tag.trim() {
            "think" => KimiK3Mode::PrefilledReasoning,
            "response" => KimiK3Mode::Response,
            _ => KimiK3Mode::Idle,
        };
    }

    fn apply_event(
        &mut self,
        event: KimiK3Event,
        piece: DecodedText,
        output: &mut UnifiedParserOutput,
    ) -> Result<()> {
        match event {
            KimiK3Event::Text => output.push_text(piece.text),
            KimiK3Event::Reasoning => output.push_reasoning(piece),
            // Marker and noise spans are drained and dropped with their tokens.
            KimiK3Event::Skip => {}
            KimiK3Event::ThinkOpen => self.mode = KimiK3Mode::Reasoning,
            KimiK3Event::ThinkClose => self.mode = KimiK3Mode::Idle,
            KimiK3Event::ResponseOpen => self.mode = KimiK3Mode::Response,
            KimiK3Event::ResponseClose => self.mode = KimiK3Mode::Epilogue,
            KimiK3Event::ToolsOpen => self.mode = KimiK3Mode::Tools,
            KimiK3Event::ToolsClose => self.mode = KimiK3Mode::Epilogue,
            KimiK3Event::MessageEnd => self.mode = KimiK3Mode::Done,
            KimiK3Event::CallOpen { name } => {
                // An empty/garbage call block without a tool name is dropped,
                // matching the Python parser.
                let call = (!name.is_empty()).then(|| {
                    let tool_index = self.emitted_call_count;
                    self.emitted_call_count += 1;
                    // The template renders compact JSON values.
                    OpenToolCall::new(name, tool_index, JsonStyle::Compact)
                });
                self.mode = KimiK3Mode::Call {
                    call,
                    body: CallBody::Start,
                };
            }
            KimiK3Event::ArgumentOpen { key, is_string } => {
                let (call, body) = self.call_mut()?;
                if is_string {
                    *body = CallBody::Argument;
                    if let Some(call) = call {
                        call.arguments.open_string(&key);
                    }
                } else {
                    *body = CallBody::TypedArgument {
                        key,
                        value_end_scan: MarkerScanState::default(),
                    };
                }
            }
            KimiK3Event::ArgumentText => {
                if let (Some(call), _) = self.call_mut()? {
                    call.arguments.push(&piece.text);
                }
            }
            KimiK3Event::ArgumentClose => {
                let (call, body) = self.call_mut()?;
                *body = CallBody::Arguments;
                if let Some(call) = call {
                    call.arguments.close();
                }
            }
            KimiK3Event::TypedValue { value } => {
                let (call, body) = self.call_mut()?;
                let CallBody::TypedArgument { key, .. } =
                    std::mem::replace(body, CallBody::Arguments)
                else {
                    return Err(parsing_failed!(
                        "Kimi K3 typed value outside a typed argument"
                    ));
                };
                if let Some(call) = call {
                    call.arguments.write_json(&key, &value);
                }
            }
            KimiK3Event::JsonOpen => *self.call_mut()?.1 = CallBody::Json,
            KimiK3Event::JsonText => {
                if let (Some(call), _) = self.call_mut()? {
                    call.arguments.push_raw_arguments(&piece.text);
                }
            }
            KimiK3Event::JsonClose => *self.call_mut()?.1 = CallBody::AfterJson,
            KimiK3Event::CallClose => {
                let mode = std::mem::replace(&mut self.mode, KimiK3Mode::Tools);
                let KimiK3Mode::Call { call, body, .. } = mode else {
                    return Err(parsing_failed!(
                        "Kimi K3 call completion without an active tool call"
                    ));
                };
                // A raw `json` block already is the whole arguments object.
                if let Some(call) = call
                    && !matches!(body, CallBody::AfterJson)
                {
                    output.push_call(call.finish());
                }
            }
        }
        if let KimiK3Mode::Call {
            call: Some(call), ..
        } = &mut self.mode
            && let Some(delta) = call.take_delta()
        {
            output.push_call(delta);
        }
        Ok(())
    }

    /// The active call and the position inside its block, for an event inside
    /// a `call` block.
    fn call_mut(&mut self) -> Result<(Option<&mut OpenToolCall>, &mut CallBody)> {
        match &mut self.mode {
            KimiK3Mode::Call { call, body, .. } => Ok((call.as_mut(), body)),
            _ => Err(parsing_failed!(
                "Kimi K3 call event without an active tool call"
            )),
        }
    }

    fn reset_state(&mut self) -> String {
        self.mode = KimiK3Mode::Idle;
        self.emitted_call_count = 0;
        self.buffer.take().text
    }
}

impl UnifiedParser for KimiK3UnifiedParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Self::new(tools, tokenizer).map(|parser| Box::new(parser) as Box<dyn UnifiedParser>)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.buffer.clear();
        self.emitted_call_count = 0;
        self.initialize_mode(prompt_token_ids);
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn build_output_grammar(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<BuiltOutputGrammar>> {
        grammar::build_output_grammar(&self.mode, ctx)
    }

    fn parse_into(&mut self, delta: DecodedText, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffer.append(delta);

        while let Some((event, consumed_len)) =
            parse_buffered_event_attributed(&self.buffer, &self.markers, |input| {
                parse_next_kimi_k3_event(input, &mut self.mode)
            })?
        {
            let piece = self.buffer.drain_prefix(consumed_len);
            self.apply_event(event, piece, output)?;
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();

        match &self.mode {
            KimiK3Mode::Idle | KimiK3Mode::Response => {
                output.push_text(self.buffer.take().text);
            }
            KimiK3Mode::Reasoning | KimiK3Mode::PrefilledReasoning => {
                output.push_reasoning(self.buffer.take());
            }
            KimiK3Mode::Epilogue | KimiK3Mode::Done => self.buffer.clear(),
            // A tools channel truncated between complete calls loses only its
            // closing markers; keep the calls already emitted.
            KimiK3Mode::Tools if self.buffer.text.is_empty() => {}
            KimiK3Mode::Tools | KimiK3Mode::Call { .. } => {
                return Err(UnifiedParserError::IncompleteToolCall { parser: "Kimi K3" });
            }
        }

        self.mode = KimiK3Mode::Idle;
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.reset_state()
    }
}

/// Parse one Kimi K3 event from buffered streaming input.
fn parse_next_kimi_k3_event(
    input: &mut KimiK3Input<'_>,
    mode: &mut KimiK3Mode,
) -> ModalResult<KimiK3Event> {
    match mode {
        KimiK3Mode::Idle => parse_idle_event(input),
        KimiK3Mode::Reasoning => parse_reasoning_event(input),
        KimiK3Mode::PrefilledReasoning => parse_prefilled_reasoning_event(input),
        KimiK3Mode::Response => parse_response_event(input),
        KimiK3Mode::Epilogue => parse_epilogue_event(input),
        KimiK3Mode::Tools => parse_tools_event(input),
        KimiK3Mode::Call { body, .. } => parse_call_event(input, body),
        KimiK3Mode::Done => parse_done_event(input),
    }
}

/// Parse an event while waiting for the next channel open.
fn parse_idle_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    alt((
        markers.think_open.value(KimiK3Event::ThinkOpen),
        markers.response_open.value(KimiK3Event::ResponseOpen),
        markers.tools_open.value(KimiK3Event::ToolsOpen),
        message_end_event,
        safe_idle_text_event,
    ))
    .parse_next(input)
}

/// Parse an event inside the `think` channel.
fn parse_reasoning_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    alt((
        markers.think_close.value(KimiK3Event::ThinkClose),
        // `<|end_of_msg|>` can reach the parser under `ignore_eos` or
        // `include_stop_str_in_output`; never leak it into reasoning.
        markers.end_of_msg.value(KimiK3Event::MessageEnd),
        safe_reasoning_event,
    ))
    .parse_next(input)
}

/// Parse an event inside the `think` channel the prompt opened.
///
/// Like Python, a `response` open ends reasoning only while the model has not
/// emitted its own `think` open.
fn parse_prefilled_reasoning_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    alt((
        markers.think_close.value(KimiK3Event::ThinkClose),
        markers.think_open.value(KimiK3Event::ThinkOpen),
        markers.response_open.value(KimiK3Event::ResponseOpen),
        markers.end_of_msg.value(KimiK3Event::MessageEnd),
        safe_prefilled_reasoning_event,
    ))
    .parse_next(input)
}

/// Parse an event inside the `response` channel.
fn parse_response_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    alt((
        markers.response_close.value(KimiK3Event::ResponseClose),
        // The response body also implicitly ends at a `tools` channel.
        markers.tools_open.value(KimiK3Event::ToolsOpen),
        message_end_event,
        safe_response_text_event,
    ))
    .parse_next(input)
}

/// Parse an event after the response channel closed.
fn parse_epilogue_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    alt((
        markers.tools_open.value(KimiK3Event::ToolsOpen),
        message_end_event,
        skip_epilogue_noise_event,
    ))
    .parse_next(input)
}

/// Parse an event inside the `tools` channel, between `call` blocks.
fn parse_tools_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    alt((
        call_open_event,
        markers.tools_close.value(KimiK3Event::ToolsClose),
        // Defensive: an unterminated tools channel still ends with the message.
        message_end_event,
        skip_tools_noise_event,
    ))
    .parse_next(input)
}

/// Parse a message close or end-of-message marker.
fn message_end_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    alt((&markers.message_close, &markers.end_of_msg))
        .value(KimiK3Event::MessageEnd)
        .parse_next(input)
}

/// Ignore everything after the assistant message closed.
fn parse_done_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    rest.value(KimiK3Event::Skip).parse_next(input)
}

/// Parse safe text while waiting for the next channel marker.
fn safe_idle_text_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    safe_text_len_mul(
        input,
        &[
            &markers.think_open,
            &markers.response_open,
            &markers.tools_open,
            &markers.message_close,
            &markers.end_of_msg,
        ],
    )
    .map(|_| KimiK3Event::Text)
}

/// Parse safe reasoning before the think close marker.
fn safe_reasoning_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    safe_text_len_mul(input, &[&markers.think_close, &markers.end_of_msg])
        .map(|_| KimiK3Event::Reasoning)
}

/// Parse safe reasoning before the next marker of a prefilled `think`
/// channel.
fn safe_prefilled_reasoning_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    safe_text_len_mul(
        input,
        &[
            &markers.think_close,
            &markers.think_open,
            &markers.response_open,
            &markers.end_of_msg,
        ],
    )
    .map(|_| KimiK3Event::Reasoning)
}

/// Parse safe response text before the next channel marker.
fn safe_response_text_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    safe_text_len_mul(
        input,
        &[
            &markers.response_close,
            &markers.tools_open,
            &markers.message_close,
            &markers.end_of_msg,
        ],
    )
    .map(|_| KimiK3Event::Text)
}

/// Skip non-content noise after the response channel closed.
fn skip_epilogue_noise_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    safe_text_len_mul(
        input,
        &[
            &markers.tools_open,
            &markers.message_close,
            &markers.end_of_msg,
        ],
    )
    .map(|_| KimiK3Event::Skip)
}

/// Skip non-content noise between `call` blocks.
fn skip_tools_noise_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    safe_text_len_mul(
        input,
        &[
            &markers.call_open,
            &markers.tools_close,
            &markers.message_close,
            &markers.end_of_msg,
        ],
    )
    .map(|_| KimiK3Event::Skip)
}

/// Parse a `call` open tag into its tool name.
///
/// The attributes end at the structural `<|sep|>`; an ordinary spelling of
/// `<|sep|>` inside an attribute value stays attribute text.
fn call_open_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    let mut scan = MarkerScanState::default();
    let (attrs,) = seq!(
        _: &markers.call_open,
        take_until_marker(&markers.sep, &mut scan),
        _: &markers.sep,
    )
    .parse_next(input)?;
    let attrs = parse_tag_attrs(attrs)?;

    Ok(KimiK3Event::CallOpen {
        name: attr_value(&attrs, "tool").unwrap_or_default().to_string(),
    })
}

/// Parse an event inside a `call` block at position `body`.
fn parse_call_event(input: &mut KimiK3Input<'_>, body: &mut CallBody) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    let call_close = markers.call_close.value(KimiK3Event::CallClose);
    match body {
        CallBody::Start => {
            preceded(ws0, alt((argument_open_event, json_open_event, call_close))).parse_next(input)
        }
        CallBody::Arguments => {
            preceded(ws0, alt((argument_open_event, call_close))).parse_next(input)
        }
        CallBody::Argument => alt((
            markers.argument_close.value(KimiK3Event::ArgumentClose),
            safe_argument_text_event,
        ))
        .parse_next(input),
        CallBody::TypedArgument { value_end_scan, .. } => typed_value_event(input, value_end_scan),
        CallBody::Json => alt((
            markers.json_close.value(KimiK3Event::JsonClose),
            safe_json_text_event,
        ))
        .parse_next(input),
        CallBody::AfterJson => preceded(ws0, call_close).parse_next(input),
    }
}

/// Parse an `argument` open tag into its key and whether its value is a
/// string.
///
/// A `string` value, the default, is raw text, and a value of any other type is
/// JSON text. The attributes end at the structural `<|sep|>`.
fn argument_open_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    let mut scan = MarkerScanState::default();
    let (attrs,) = seq!(
        _: &markers.argument_open,
        take_until_marker(&markers.sep, &mut scan),
        _: &markers.sep,
    )
    .parse_next(input)?;
    let attrs = parse_tag_attrs(attrs)?;

    Ok(KimiK3Event::ArgumentOpen {
        key: attr_value(&attrs, "key").unwrap_or_default().to_string(),
        is_string: attr_value(&attrs, "type").unwrap_or("string") == "string",
    })
}

/// Parse a raw `json` block open tag; its attributes (`type="object"`) are
/// unused on decode.
fn json_open_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    let mut scan = MarkerScanState::default();
    seq!(
        _: &markers.json_open,
        _: take_until_marker(&markers.sep, &mut scan),
        _: &markers.sep,
    )
    .value(KimiK3Event::JsonOpen)
    .parse_next(input)
}

/// Parse safe argument value text before the argument close marker.
fn safe_argument_text_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    safe_text_len(input, &markers.argument_close).map(|_| KimiK3Event::ArgumentText)
}

/// Parse a complete typed argument value through the argument close marker.
fn typed_value_event(
    input: &mut KimiK3Input<'_>,
    value_end_scan: &mut MarkerScanState,
) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    let (value,) = seq!(
        take_until_marker(&markers.argument_close, value_end_scan),
        _: &markers.argument_close,
    )
    .parse_next(input)?;
    Ok(KimiK3Event::TypedValue {
        value: value.to_string(),
    })
}

/// Parse safe raw `json` block text before the block close marker.
fn safe_json_text_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let markers = input.markers();
    safe_text_len(input, &markers.json_close).map(|_| KimiK3Event::JsonText)
}

/// Parse a complete XTML attribute string like ` tool="get_weather" index="1"`.
fn parse_tag_attrs(attrs: &str) -> ModalResult<Vec<(String, String)>> {
    let mut input = attrs;
    terminated(repeat(0.., preceded(ws1, tag_attr)), (ws0, eof))
        .parse_next(&mut input)
        .map_err(|_| xtml_error("XTML tag attributes"))
}

/// Parse one XTML `key="value"` attribute pair.
fn tag_attr(input: &mut &str) -> ModalResult<(String, String)> {
    seq!(
        take_while(1.., |char: char| char.is_alphanumeric() || char == '_').map(str::to_string),
        _: literal("=\""),
        take_till(0.., '"').map(unescape_attr_value),
        _: literal("\""),
    )
    .parse_next(input)
}

/// Reverse XTML attribute escaping: `&quot;` first, then `&amp;` (the inverse
/// of the encode order).
fn unescape_attr_value(value: &str) -> String {
    value.replace("&quot;", "\"").replace("&amp;", "&")
}

/// Look up one parsed attribute value by key.
fn attr_value<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs.iter().find(|(name, _)| name == key).map(|(_, value)| value.as_str())
}

/// Build a cut error for determinably malformed XTML structure.
fn xtml_error(label: &'static str) -> ErrMode<ContextError> {
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
    use vllm_tokenizer::DecodedText;
    use vllm_tokenizer::Tokenizer as _;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::{
        END_OF_MSG, KimiK3UnifiedParser, OPEN, RESPONSE_CLOSE, RESPONSE_OPEN, SEP, THINK_CLOSE,
        THINK_OPEN, TOOLS_CLOSE, TOOLS_OPEN,
    };
    use crate::tool::ToolCallDelta;
    use crate::unified::{
        UnifiedParser, UnifiedParserError, UnifiedParserEvent, UnifiedParserOutput,
    };

    const OPEN_ID: u32 = 256;
    const CLOSE_ID: u32 = 257;
    const SEP_ID: u32 = 258;
    const END_OF_MSG_ID: u32 = 259;

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_special_token(OPEN, OPEN_ID)
            .with_special_token("<|close|>", CLOSE_ID)
            .with_special_token(SEP, SEP_ID)
            .with_special_token(END_OF_MSG, END_OF_MSG_ID)
    }

    trait UnifiedParserTestExt {
        fn parse_chunk(&mut self, chunk: &str) -> super::Result<UnifiedParserOutput>;
        fn parse_complete(&mut self, text: &str) -> super::Result<UnifiedParserOutput>;
    }

    impl UnifiedParserTestExt for KimiK3UnifiedParser {
        /// Parse `chunk` as the detokenizer would deliver it: markers spelled in
        /// full are structural special tokens, everything else ordinary text.
        fn parse_chunk(&mut self, chunk: &str) -> super::Result<UnifiedParserOutput> {
            let mut output = UnifiedParserOutput::default();
            self.parse_into(decode_ids(&structural(chunk)), &mut output)?;
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
    }

    /// The parser as constructed in production: markers must be spelled by
    /// their special tokens.
    fn test_parser() -> KimiK3UnifiedParser {
        KimiK3UnifiedParser::new(&[], Arc::new(tokenizer())).unwrap()
    }

    /// Run token IDs through the real incremental decoder to obtain attributed
    /// input, exactly as the frontend does.
    fn decode_ids(ids: &[u32]) -> DecodedText {
        let tokenizer = tokenizer();
        let mut decoder = tokenizer.create_decode_stream(&[], false, 0);
        for &id in ids {
            decoder.push_token(id).unwrap();
        }
        decoder.flush(None).unwrap().1
    }

    /// Encode with the special tokens, so markers are structural.
    fn structural(text: &str) -> Vec<u32> {
        tokenizer().encode(text, false).unwrap()
    }

    /// Encode as ordinary bytes only, so marker spellings are content.
    fn ordinary(text: &str) -> Vec<u32> {
        tokenizer().encode_ordinary(text).unwrap()
    }

    /// Stream the structural encoding of the concatenated `chunks`, split at the
    /// chunk boundaries, so a marker split across chunks keeps its token anchor.
    fn collect_stream(parser: &mut KimiK3UnifiedParser, chunks: &[&str]) -> UnifiedParserOutput {
        let mut rest = decode_ids(&structural(&chunks.concat()));
        let mut output = UnifiedParserOutput::default();
        for chunk in chunks {
            let piece = rest.drain_prefix(chunk.len());
            parser.parse_into(piece, &mut output).unwrap();
        }
        output.append(parser.finish().unwrap());
        output
    }

    /// Split `text` into small chunks to stress marker-split handling.
    fn char_chunks(text: &str, size: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        chars.chunks(size).map(|chunk| chunk.iter().collect()).collect()
    }

    fn arg(key: &str, arg_type: &str, value: &str) -> String {
        format!(
            "{OPEN}argument key=\"{key}\" type=\"{arg_type}\"{SEP}{value}<|close|>argument{SEP}"
        )
    }

    fn call(attrs: &str, body: &str) -> String {
        format!("{OPEN}call {attrs}{SEP}{body}<|close|>call{SEP}")
    }

    fn thinking_output(reasoning: &str, response: &str, tools_body: &str) -> String {
        let mut output = format!("{THINK_OPEN}{reasoning}{THINK_CLOSE}");
        output.push_str(&format!("{RESPONSE_OPEN}{response}{RESPONSE_CLOSE}"));
        if !tools_body.is_empty() {
            output.push_str(&format!("{TOOLS_OPEN}{tools_body}{TOOLS_CLOSE}"));
        }
        output.push_str("<|close|>message<|sep|>");
        output
    }

    fn first_call(output: &UnifiedParserOutput) -> ToolCallDelta {
        output.calls().first().expect("expected one tool call").clone()
    }

    #[test]
    fn kimi_k3_create_requires_structural_tokens() {
        let error = match KimiK3UnifiedParser::new(&[], Arc::new(TestTokenizer::new())) {
            Ok(_) => panic!("expected missing token error"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            UnifiedParserError::MissingToken { token } if token == OPEN
        ));
    }

    #[test]
    fn kimi_k3_parses_reasoning_response_and_typed_tool_call() {
        let body = [
            arg("city", "string", "Hangzhou"),
            arg("days", "number", "1.5"),
            arg("detailed", "boolean", "true"),
            arg("filters", "object", r#"{"kind":"rain"}"#),
            arg("hours", "array", "[8,20]"),
        ]
        .concat();
        let text = thinking_output(
            "Need the weather tool.",
            "I'll check.",
            &call("tool=\"get_weather\" index=\"1\"", &body),
        );

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        assert_eq!(output.reasoning_text(), "Need the weather tool.");
        assert_eq!(output.normal_text(), "I'll check.");
        let call = first_call(&output);
        assert_eq!(call.tool_index, 0);
        assert_eq!(call.name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).unwrap(),
            json!({
                "city": "Hangzhou",
                "days": 1.5,
                "detailed": true,
                "filters": { "kind": "rain" },
                "hours": [8, 20],
            })
        );
    }

    #[test]
    fn kimi_k3_arguments_preserve_order_and_number_formatting() {
        let body = [
            arg("y", "number", "1.0"),
            arg("x", "number", "2"),
            arg("items", "array", r#"["left","right"]"#),
        ]
        .concat();
        let text = thinking_output("t", "", &call("tool=\"add\" index=\"1\"", &body));

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        assert_eq!(
            first_call(&output).arguments,
            r#"{"y":1.0,"x":2,"items":["left","right"]}"#
        );
    }

    #[test]
    fn kimi_k3_streaming_splits_markers_across_chunks() {
        let text = thinking_output(
            "step by step",
            "the answer",
            &call("tool=\"calc\" index=\"1\"", &arg("x", "number", "42")),
        );

        for size in [1, 3, 7] {
            let chunks = char_chunks(&text, size);
            let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let output = collect_stream(&mut test_parser(), &chunk_refs);

            assert_eq!(output.reasoning_text(), "step by step", "chunk size {size}");
            assert_eq!(output.normal_text(), "the answer", "chunk size {size}");
            assert_eq!(first_call(&output).name.as_deref(), Some("calc"));
            assert_eq!(first_call(&output).arguments, r#"{"x":42}"#);
        }
    }

    #[test]
    fn kimi_k3_streaming_emits_text_incrementally() {
        let mut parser = test_parser();
        let prompt = tokenizer().encode("<|open|>response<|sep|>", false).unwrap();
        parser.initialize(&prompt).unwrap();

        let first = parser.parse_chunk("Hel").unwrap();
        assert_eq!(first.normal_text(), "Hel");

        let second = parser.parse_chunk("lo<|close|>resp").unwrap();
        assert_eq!(second.normal_text(), "lo");

        let mut output = parser.parse_chunk("onse<|sep|>").unwrap();
        output.append(parser.finish().unwrap());
        assert_eq!(output.normal_text(), "");
    }

    #[test]
    fn kimi_k3_initialize_think_prefill_starts_in_reasoning() {
        let mut parser = test_parser();
        let prompt = tokenizer()
            .encode(
                "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>",
                false,
            )
            .unwrap();
        parser.initialize(&prompt).unwrap();

        let output = parser
            .parse_complete(&format!(
                "reasoning{THINK_CLOSE}{RESPONSE_OPEN}answer{RESPONSE_CLOSE}<|close|>message{SEP}"
            ))
            .unwrap();

        assert_eq!(output.reasoning_text(), "reasoning");
        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn kimi_k3_response_open_ends_prefilled_reasoning() {
        let prompt = structural("<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>");
        let text = format!(
            "{RESPONSE_OPEN}answer{RESPONSE_CLOSE}{TOOLS_OPEN}{}{TOOLS_CLOSE}<|close|>message{SEP}",
            call("tool=\"calc\" index=\"1\"", &arg("x", "number", "42")),
        );

        for size in [1, 3, 7, text.len()] {
            let chunks = char_chunks(&text, size);
            let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let mut parser = test_parser();
            parser.initialize(&prompt).unwrap();
            let output = collect_stream(&mut parser, &chunk_refs);

            assert_eq!(output.reasoning_text(), "", "chunk size {size}");
            assert_eq!(output.normal_text(), "answer", "chunk size {size}");
            assert_eq!(first_call(&output).arguments, r#"{"x":42}"#);
        }
    }

    #[test]
    fn kimi_k3_response_open_stays_in_explicitly_opened_reasoning() {
        let mut parser = test_parser();
        start_in_reasoning(&mut parser);
        let output = parser
            .parse_complete(&format!(
                "{THINK_OPEN}draft {RESPONSE_OPEN}x{THINK_CLOSE}{RESPONSE_OPEN}answer{RESPONSE_CLOSE}"
            ))
            .unwrap();

        assert_eq!(output.reasoning_text(), format!("draft {RESPONSE_OPEN}x"));
        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn kimi_k3_initialize_response_prefill_starts_in_response() {
        let mut parser = test_parser();
        let prompt = tokenizer()
            .encode(
                "<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>",
                false,
            )
            .unwrap();
        parser.initialize(&prompt).unwrap();

        let output = parser
            .parse_complete(&format!("answer{RESPONSE_CLOSE}<|close|>message{SEP}"))
            .unwrap();

        assert_eq!(output.normal_text(), "answer");
        assert!(output.reasoning_text().is_empty());
    }

    #[test]
    fn kimi_k3_initialize_message_open_prefill_starts_idle() {
        let mut parser = test_parser();
        let prompt =
            tokenizer().encode("<|open|>message role=\"assistant\"<|sep|>", false).unwrap();
        parser.initialize(&prompt).unwrap();

        let output = parser
            .parse_complete(&format!("{THINK_OPEN}reason{THINK_CLOSE}{RESPONSE_OPEN}hi"))
            .unwrap();

        assert_eq!(output.reasoning_text(), "reason");
        assert_eq!(output.normal_text(), "hi");
    }

    #[test]
    fn kimi_k3_plain_text_falls_through_as_text() {
        let output = collect_stream(&mut test_parser(), &["plain ", "answer"]);

        assert_eq!(output.normal_text(), "plain answer");
        assert!(output.reasoning_text().is_empty());
        assert!(output.calls().is_empty());
    }

    #[test]
    fn kimi_k3_tool_call_streams_before_close_marker() {
        let mut parser = test_parser();
        let mut output = UnifiedParserOutput::default();

        // The call and its arguments so far, after each chunk.
        let mut progress = Vec::new();
        for chunk in [
            TOOLS_OPEN,
            "<|open|>call tool=\"calc\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"note\" type=\"string\"<|sep|>half",
            " \"done\"<|close|>argument<|sep|>",
            &arg("x", "number", "1"),
            "<|close|>call<|sep|>",
        ] {
            output.append(parser.parse_chunk(chunk).unwrap());
            progress.push(match output.calls().first() {
                Some(call) => format!("{}({})", call.name.as_deref().unwrap(), call.arguments),
                None => "no call".to_string(),
            });
        }

        expect![[r#"
            no call
            calc()
            calc({"note":"half)
            calc({"note":"half \"done\"")
            calc({"note":"half \"done\"","x":1)
            calc({"note":"half \"done\"","x":1})"#]]
        .assert_eq(&progress.join("\n"));
    }

    #[test]
    fn kimi_k3_parses_multiple_tool_calls() {
        let tools_body = format!(
            "{}{}",
            call(
                "tool=\"get_weather\" index=\"1\"",
                &arg("city", "string", "SF")
            ),
            call("tool=\"get_time\" index=\"2\"", ""),
        );
        let text = thinking_output("t", "r", &tools_body);

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        let calls = output.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool_index, 0);
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(calls[0].arguments, r#"{"city":"SF"}"#);
        assert_eq!(calls[1].tool_index, 1);
        assert_eq!(calls[1].name.as_deref(), Some("get_time"));
        assert_eq!(calls[1].arguments, "{}");
    }

    #[test]
    fn kimi_k3_json_block_arguments_pass_through_raw() {
        // Spacing and key order must survive unmodified: raw `json` blocks are
        // not validated or normalized.
        let raw = r#"{"b": 1,  "a": [2 , 3]}"#;
        let body = format!("{OPEN}json type=\"object\"{SEP}{raw}<|close|>json{SEP}");
        let text = thinking_output("t", "", &call("tool=\"run\" index=\"1\"", &body));

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        assert_eq!(first_call(&output).arguments, raw);
    }

    #[test]
    fn kimi_k3_string_argument_passes_raw_text_through() {
        let value = "line one\nline two {\"not\": \"json\"} & <tags>";
        let text = thinking_output(
            "t",
            "",
            &call(
                "tool=\"write\" index=\"1\"",
                &arg("content", "string", value),
            ),
        );

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "content": value })
        );
    }

    #[test]
    fn kimi_k3_typed_arguments_keep_json_spelling_or_fall_back_to_strings() {
        // Non-string values are JSON text: valid text keeps its spelling, and
        // malformed or empty text becomes a string, as in the Python parser.
        let body = [
            arg("x", "number", "1.50"),
            arg("filters", "object", r#"{"kind": "rain"}"#),
            arg("y", "number", "not a number"),
            arg("z", "boolean", ""),
        ]
        .concat();
        let text = thinking_output("t", "", &call("tool=\"calc\" index=\"1\"", &body));

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        assert_eq!(
            first_call(&output).arguments,
            r#"{"x":1.50,"filters":{"kind": "rain"},"y":"not a number","z":""}"#
        );
    }

    #[test]
    fn kimi_k3_ordinary_argument_close_inside_string_value_stays_value() {
        let quoted = "<|close|>argument<|sep|><|close|>call<|sep|>";
        let mut ids = structural(&format!(
            "{TOOLS_OPEN}{OPEN}call tool=\"write\" index=\"1\"{SEP}\
             {OPEN}argument key=\"content\" type=\"string\"{SEP}a"
        ));
        ids.extend(ordinary(quoted));
        ids.extend(structural(&format!(
            "b<|close|>argument{SEP}<|close|>call{SEP}{TOOLS_CLOSE}"
        )));

        let output = parse_decoded(&mut test_parser(), decode_ids(&ids));

        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "content": format!("a{quoted}b") })
        );
    }

    #[test]
    fn kimi_k3_attribute_values_are_unescaped() {
        let text = thinking_output(
            "t",
            "",
            &call(
                "tool=\"a&quot;b&amp;c\" index=\"1\"",
                &arg("key", "string", "value"),
            ),
        );

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        assert_eq!(first_call(&output).name.as_deref(), Some("a\"b&c"));
    }

    #[test]
    fn kimi_k3_call_without_tool_name_is_dropped() {
        let tools_body = format!(
            "{}{}",
            call("index=\"1\"", &arg("x", "number", "1")),
            call("tool=\"real\" index=\"2\"", ""),
        );
        let text = thinking_output("t", "", &tools_body);

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        let calls = output.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_index, 0);
        assert_eq!(calls[0].name.as_deref(), Some("real"));
    }

    #[test]
    fn kimi_k3_tool_indices_ignore_xtml_index_attribute() {
        let tools_body = format!(
            "{}{}{}",
            call("tool=\"first\" index=\"3\"", ""),
            call("tool=\"second\"", ""),
            call("tool=\"third\" index=\"x\"", ""),
        );
        let text = thinking_output("t", "", &tools_body);

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        assert_eq!(
            output.calls().iter().map(|call| call.tool_index).collect::<Vec<_>>(),
            [0, 1, 2]
        );
    }

    #[test]
    fn kimi_k3_ignores_output_after_message_close() {
        let mut parser = test_parser();
        let output = parser
            .parse_complete(&format!(
                "{RESPONSE_OPEN}answer{RESPONSE_CLOSE}<|close|>message{SEP}junk{END_OF_MSG}"
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn kimi_k3_epilogue_noise_is_not_content() {
        let text = format!(
            "{RESPONSE_OPEN}answer{RESPONSE_CLOSE}\n{TOOLS_OPEN}{}{TOOLS_CLOSE}\n<|close|>message{SEP}",
            call("tool=\"calc\" index=\"1\"", ""),
        );

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        assert_eq!(output.normal_text(), "answer");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn kimi_k3_finish_flushes_unclosed_reasoning() {
        let mut parser = test_parser();
        let mut output = parser.parse_chunk(&format!("{THINK_OPEN}still thinking")).unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(output.reasoning_text(), "still thinking");
        assert!(output.normal_text().is_empty());
    }

    #[test]
    fn kimi_k3_finish_flushes_partial_marker_as_text() {
        let mut parser = test_parser();
        let mut output = parser.parse_chunk("answer<|clo").unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(output.normal_text(), "answer<|clo");
    }

    #[test]
    fn kimi_k3_finish_fails_mid_tool_call() {
        let mut parser = test_parser();
        parser
            .parse_chunk(&format!(
                "{TOOLS_OPEN}<|open|>call tool=\"calc\" index=\"1\"<|sep|>{}",
                arg("x", "number", "1")
            ))
            .unwrap();

        let error = parser.finish().unwrap_err();

        assert!(error.to_report_string().contains("incomplete Kimi K3 tool call"));
    }

    #[test]
    fn kimi_k3_finish_after_truncated_tools_keeps_complete_calls() {
        let mut parser = test_parser();
        let mut output = parser
            .parse_chunk(&format!(
                "{TOOLS_OPEN}{}",
                call("tool=\"calc\" index=\"1\"", &arg("x", "number", "1"))
            ))
            .unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(first_call(&output).name.as_deref(), Some("calc"));
    }

    #[test]
    fn kimi_k3_malformed_call_attributes_fail_fast() {
        let mut parser = test_parser();
        let error = parser
            .parse_chunk(&format!("{TOOLS_OPEN}<|open|>call garbage attrs<|sep|>"))
            .unwrap_err();

        assert!(error.to_report_string().contains("XTML tag attributes"));
    }

    #[test]
    fn kimi_k3_empty_response_channel_emits_nothing() {
        let text = thinking_output("t", "", &call("tool=\"calc\" index=\"1\"", ""));

        let mut parser = test_parser();
        let output = parser.parse_complete(&text).unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.reasoning_text(), "t");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn kimi_k3_reset_returns_buffered_text() {
        let mut parser = test_parser();
        let prompt = tokenizer().encode("<|open|>response<|sep|>", false).unwrap();
        parser.initialize(&prompt).unwrap();
        parser.parse_chunk("answer<|close|>resp").unwrap();

        let raw = parser.reset();

        assert_eq!(raw, "<|close|>resp");
    }

    // --- Ordinary marker spellings ------------------------------------------

    fn start_in_reasoning(parser: &mut KimiK3UnifiedParser) {
        parser.initialize(&structural(THINK_OPEN)).unwrap();
    }

    fn parse_decoded(
        parser: &mut KimiK3UnifiedParser,
        decoded: DecodedText,
    ) -> UnifiedParserOutput {
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(decoded, &mut output).unwrap();
        output.append(parser.finish().unwrap());
        output
    }

    #[test]
    fn kimi_k3_markers_spell_the_documented_strings() {
        let special_tokens = super::SpecialTokens::new(&tokenizer()).unwrap();
        let markers = super::Markers::new(&special_tokens);

        assert_eq!(markers.think_open.as_str(), THINK_OPEN);
        assert_eq!(markers.think_close.as_str(), THINK_CLOSE);
        assert_eq!(markers.response_open.as_str(), RESPONSE_OPEN);
        assert_eq!(markers.response_close.as_str(), RESPONSE_CLOSE);
        assert_eq!(markers.tools_open.as_str(), TOOLS_OPEN);
        assert_eq!(markers.tools_close.as_str(), TOOLS_CLOSE);
        assert_eq!(markers.message_close.as_str(), super::MESSAGE_CLOSE);
        assert_eq!(markers.call_close.as_str(), super::CALL_CLOSE);
        assert_eq!(markers.end_of_msg.as_str(), END_OF_MSG);
    }

    #[test]
    fn kimi_k3_ordinary_close_marker_text_stays_in_reasoning() {
        // The production incident: the model spells `<|close|>think<|sep|>` with
        // ordinary tokens while reasoning, then really closes the channel later.
        let mut ids = ordinary(THINK_CLOSE);
        ids.extend(structural(&format!(
            " after{THINK_CLOSE}{RESPONSE_OPEN}answer{RESPONSE_CLOSE}<|close|>message{SEP}"
        )));
        let decoded = decode_ids(&ids);
        assert!(
            decoded.text.starts_with(THINK_CLOSE),
            "same bytes as the structural marker"
        );

        let mut parser = test_parser();
        start_in_reasoning(&mut parser);
        let output = parse_decoded(&mut parser, decoded);

        assert_eq!(output.reasoning_text(), format!("{THINK_CLOSE} after"));
        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn kimi_k3_ordinary_open_marker_text_stays_in_reasoning() {
        let mut ids = structural("config: ");
        ids.extend(ordinary(THINK_OPEN));
        ids.extend(structural(&format!(
            "{THINK_CLOSE}{RESPONSE_OPEN}ok{RESPONSE_CLOSE}"
        )));

        let mut parser = test_parser();
        start_in_reasoning(&mut parser);
        let output = parse_decoded(&mut parser, decode_ids(&ids));

        assert_eq!(output.reasoning_text(), format!("config: {THINK_OPEN}"));
        assert_eq!(output.normal_text(), "ok");
    }

    #[test]
    fn kimi_k3_attributed_streaming_splits_at_every_char_boundary() {
        // A lookalike close inside reasoning, then genuine markers and a call.
        let mut ids = structural(THINK_OPEN);
        ids.extend(ordinary(THINK_CLOSE));
        ids.extend(structural(&format!(
            " step{THINK_CLOSE}{RESPONSE_OPEN}the answer{RESPONSE_CLOSE}{TOOLS_OPEN}{}{TOOLS_CLOSE}<|close|>message{SEP}",
            call("tool=\"calc\" index=\"1\"", &arg("x", "number", "42")),
        )));
        let full = decode_ids(&ids);

        for split in (0..=full.text.len()).filter(|&index| full.text.is_char_boundary(index)) {
            let mut rest = full.clone();
            let head = rest.drain_prefix(split);
            let mut parser = test_parser();
            let mut output = UnifiedParserOutput::default();
            parser.parse_into(head, &mut output).unwrap();
            parser.parse_into(rest, &mut output).unwrap();
            output.append(parser.finish().unwrap());

            assert_eq!(
                output.reasoning_text(),
                format!("{THINK_CLOSE} step"),
                "split {split}"
            );
            assert_eq!(output.normal_text(), "the answer", "split {split}");
            assert_eq!(
                first_call(&output).arguments,
                r#"{"x":42}"#,
                "split {split}"
            );
        }
    }

    #[test]
    fn kimi_k3_incomplete_utf8_before_structural_close_still_closes() {
        // A stray incomplete byte right before the close marker: the decoder
        // anchors the special token at its own spelling, so the boundary holds.
        let mut ids = vec![0xe4];
        ids.extend(structural(&format!(
            "{THINK_CLOSE}{RESPONSE_OPEN}ok{RESPONSE_CLOSE}"
        )));

        let mut parser = test_parser();
        start_in_reasoning(&mut parser);
        let output = parse_decoded(&mut parser, decode_ids(&ids));

        assert_eq!(output.reasoning_text(), "\u{FFFD}");
        assert_eq!(output.normal_text(), "ok");
    }

    #[test]
    fn kimi_k3_ordinary_sep_text_inside_call_attributes_stays_attribute_text() {
        let mut ids = structural(&format!("{TOOLS_OPEN}{OPEN}call tool=\"a"));
        ids.extend(ordinary(SEP));
        ids.extend(structural(&format!(
            "b\" index=\"1\"{SEP}{}<|close|>call{SEP}{TOOLS_CLOSE}",
            arg("x", "number", "1")
        )));

        let output = parse_decoded(&mut test_parser(), decode_ids(&ids));

        assert_eq!(first_call(&output).name, Some(format!("a{SEP}b")));
        assert_eq!(first_call(&output).arguments, r#"{"x":1}"#);
    }
}
