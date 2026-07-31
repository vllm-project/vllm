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
//! Argument decoding mirrors the renderer's type tagging (inverse encoding):
//! `type="string"` values pass the raw text through, other types are
//! JSON-decoded, and a raw `json` block is passed through unmodified. Attribute
//! values reverse the renderer escaping (`&quot;` before `&amp;`).
//!
//! Known limitation (shared with the Python parser): string argument and
//! response bodies are emitted raw, so a value that literally contains
//! `<|close|>argument<|sep|>` or `<|close|>response<|sep|>` is
//! indistinguishable from a real closing marker.

mod structural_tag;

pub use structural_tag::KimiK3StructuralTagBuilder;

use serde_json::Value;
use vllm_tokenizer::DynTokenizer;
use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, eof, preceded, repeat, seq, terminated};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_till, take_until, take_while};

use self::structural_tag::KIMI_K3_STRUCTURAL_TAG_BUILDER;
use super::{Result, UnifiedParser, UnifiedParserOutput, token_id};
use crate::tool::{StructuralTagBuilder, Tool, ToolCallDelta};
use crate::unified::parsing_failed;
use crate::utils::{parse_buffered_event, safe_text_len, safe_text_len_mul};

const OPEN: &str = "<|open|>";
const SEP: &str = "<|sep|>";
const END_OF_MSG: &str = "<|end_of_msg|>";

const THINK_OPEN: &str = "<|open|>think<|sep|>";
const THINK_CLOSE: &str = "<|close|>think<|sep|>";
const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";
const CALL_OPEN: &str = "<|open|>call";
const CALL_CLOSE: &str = "<|close|>call<|sep|>";
const ARG_OPEN: &str = "<|open|>argument";
const ARG_CLOSE: &str = "<|close|>argument<|sep|>";
const JSON_OPEN: &str = "<|open|>json";
const JSON_CLOSE: &str = "<|close|>json<|sep|>";

const IDLE_MARKERS: &[&str] = &[
    THINK_OPEN,
    RESPONSE_OPEN,
    TOOLS_OPEN,
    MESSAGE_CLOSE,
    END_OF_MSG,
];
const REASONING_MARKERS: &[&str] = &[THINK_CLOSE, END_OF_MSG];
const RESPONSE_MARKERS: &[&str] = &[RESPONSE_CLOSE, TOOLS_OPEN, MESSAGE_CLOSE, END_OF_MSG];
const EPILOGUE_MARKERS: &[&str] = &[TOOLS_OPEN, MESSAGE_CLOSE, END_OF_MSG];
const TOOLS_MARKERS: &[&str] = &[CALL_OPEN, TOOLS_CLOSE, MESSAGE_CLOSE, END_OF_MSG];
const CALL_BODY_MARKERS: &[&str] = &[ARG_OPEN, JSON_OPEN, CALL_CLOSE, MESSAGE_CLOSE, END_OF_MSG];

/// Channel tags are a couple of text tokens; longer `<|open|>…<|sep|>` spans in
/// the prompt tail (attribute-bearing message opens, message bodies) never name
/// a generation channel.
const MAX_PREFILL_TAG_TOKENS: usize = 8;

type KimiK3Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum KimiK3Event {
    Text {
        len: usize,
    },
    Reasoning {
        len: usize,
    },
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
        index: Option<String>,
    },
    /// A typed `argument` block opened.
    ArgumentOpen {
        key: String,
        arg_type: String,
    },
    /// A raw `json` block opened; its body streams through unmodified.
    JsonOpen,
    /// Safe argument-value text; interpreted per the current [`CallStage`].
    ValueText {
        len: usize,
    },
    /// An argument/json block closed. Scalar typed blocks carry their
    /// buffered raw value; incrementally streamed values carry `None`.
    ArgumentEnd {
        raw: Option<String>,
    },
    CallEnd,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum KimiK3Mode {
    /// Before any channel opens: channel opens are expected, but raw text
    /// falls through as visible text so marker-free output still streams.
    #[default]
    Idle,
    /// Inside the `think` channel.
    Reasoning,
    /// Inside the `response` channel.
    Response,
    /// After the `response` (or `tools`) channel closed: only a `tools`
    /// channel or the message close may follow, and noise is never content.
    Epilogue,
    /// Inside the `tools` channel, between `call` blocks.
    Tools,
    /// Inside one `call` block; arguments are emitted incrementally.
    Call(CallState),
    /// After the message closed: ignore the rest (EOS leakage guard).
    Done,
}

/// Streaming state of one in-flight Kimi K3 `call` block.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct CallState {
    /// Output index assigned to this call at open time.
    tool_index: usize,
    /// The XTML `index` attribute (informational only; output indices are
    /// dense per-turn).
    index: Option<String>,
    /// Swallow the call without emitting (empty tool name).
    dropped: bool,
    /// Whether any typed-argument fragment went out yet; drives the `{`/`,`
    /// separator and the closing fragment.
    arg_emitted: bool,
    /// The call body is one raw `json` block passed through verbatim, so no
    /// braces are added around its fragments.
    raw_json: bool,
    /// Position within the call body.
    stage: CallStage,
}

/// Position within a `call` body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum CallStage {
    /// Between `argument`/`json` blocks, or before the call close.
    #[default]
    BetweenBlocks,
    /// Inside a `type="string"` value, streamed as it grows: the opening
    /// `"` was emitted when the block opened and stays open between chunks.
    StringValue,
    /// Inside a non-string typed value, buffered until the block closes:
    /// partial scalars are never valid JSON, so they emit as one fragment.
    ScalarValue { key: String, arg_type: String },
    /// Inside a raw `json` body, streamed through unmodified.
    JsonValue,
}

/// Unified parser for Kimi K3 XTML think / response / tools channels.
pub struct KimiK3UnifiedParser {
    buffer: String,
    mode: KimiK3Mode,
    /// Number of calls emitted in the current response.
    emitted_call_count: usize,
    tokenizer: DynTokenizer,
    open_token_id: u32,
    sep_token_id: u32,
}

impl KimiK3UnifiedParser {
    /// Create a Kimi K3 parser.
    pub fn new(_tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let open_token_id = token_id(tokenizer.as_ref(), OPEN)?;
        let sep_token_id = token_id(tokenizer.as_ref(), SEP)?;

        Ok(Self {
            buffer: String::new(),
            mode: KimiK3Mode::default(),
            emitted_call_count: 0,
            tokenizer,
            open_token_id,
            sep_token_id,
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

        let Some(sep_pos) = prompt_token_ids.iter().rposition(|&id| id == self.sep_token_id) else {
            return;
        };
        let Some(open_pos) =
            prompt_token_ids[..sep_pos].iter().rposition(|&id| id == self.open_token_id)
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
            "think" => KimiK3Mode::Reasoning,
            "response" => KimiK3Mode::Response,
            _ => KimiK3Mode::Idle,
        };
    }

    fn apply_event(&mut self, event: KimiK3Event, output: &mut UnifiedParserOutput) -> Result<()> {
        match event {
            KimiK3Event::Text { len } => output.push_text(self.buffer[..len].to_string()),
            KimiK3Event::Reasoning { len } => {
                output.push_reasoning(self.buffer[..len].to_string());
            }
            KimiK3Event::Skip => {}
            KimiK3Event::ThinkOpen => self.mode = KimiK3Mode::Reasoning,
            KimiK3Event::ThinkClose => self.mode = KimiK3Mode::Idle,
            KimiK3Event::ResponseOpen => self.mode = KimiK3Mode::Response,
            KimiK3Event::ResponseClose => self.mode = KimiK3Mode::Epilogue,
            KimiK3Event::ToolsOpen => self.mode = KimiK3Mode::Tools,
            KimiK3Event::ToolsClose => self.mode = KimiK3Mode::Epilogue,
            KimiK3Event::MessageEnd => {
                if let KimiK3Mode::Call(state) = &self.mode {
                    // A truncated call closed by the message end: best-effort
                    // close its arguments like `finish` would.
                    push_call_close(state, output);
                }
                self.mode = KimiK3Mode::Done;
            }
            KimiK3Event::CallOpen { name, index } => {
                let dropped = name.is_empty();
                let tool_index = self.emitted_call_count;
                if !dropped {
                    // Incremental contract: the function name is emitted (with
                    // empty arguments) as soon as the call header parses,
                    // before any argument fragment.
                    output.push_call(ToolCallDelta {
                        tool_index,
                        name: Some(name),
                        arguments: String::new(),
                    });
                }
                self.mode = KimiK3Mode::Call(CallState {
                    tool_index,
                    index,
                    dropped,
                    ..CallState::default()
                });
            }
            KimiK3Event::ArgumentOpen { key, arg_type } => {
                let state = self.active_call()?;
                if state.raw_json {
                    return Err(parsing_failed!(
                        "Kimi K3 mixed raw json and typed argument blocks"
                    ));
                }
                if !state.dropped && arg_type == "string" {
                    let key_json = serde_json::to_string(&key).map_err(|error| {
                        parsing_failed!("failed to serialize argument key: {}", error)
                    })?;
                    let separator = if state.arg_emitted { "," } else { "{" };
                    // The string value itself streams in the following deltas;
                    // its JSON quote stays open until the block closes.
                    output.push_call(ToolCallDelta {
                        tool_index: state.tool_index,
                        name: None,
                        arguments: format!("{separator}{key_json}:\""),
                    });
                    state.arg_emitted = true;
                }
                state.stage = if arg_type == "string" {
                    CallStage::StringValue
                } else {
                    CallStage::ScalarValue { key, arg_type }
                };
            }
            KimiK3Event::JsonOpen => {
                let state = self.active_call()?;
                if state.arg_emitted {
                    return Err(parsing_failed!(
                        "Kimi K3 mixed typed argument and raw json blocks"
                    ));
                }
                state.raw_json = true;
                state.stage = CallStage::JsonValue;
            }
            KimiK3Event::ValueText { len } => {
                // Snapshot what we need from the call state before borrowing
                // the buffer to build the streamed arguments fragment.
                let (tool_index, in_string) = {
                    let state = self.active_call()?;
                    if state.dropped {
                        return Ok(());
                    }
                    match state.stage {
                        CallStage::StringValue => (state.tool_index, true),
                        CallStage::JsonValue => (state.tool_index, false),
                        _ => {
                            return Err(parsing_failed!("Kimi K3 value text outside a value"));
                        }
                    }
                };
                let fragment = if in_string {
                    escape_json_contents(&self.buffer[..len]).map_err(|error| {
                        parsing_failed!("failed to escape argument text: {}", error)
                    })?
                } else {
                    self.buffer[..len].to_string()
                };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: fragment,
                });
            }
            KimiK3Event::ArgumentEnd { raw } => {
                let state = self.active_call()?;
                let stage = std::mem::take(&mut state.stage);
                if state.dropped {
                    return Ok(());
                }
                match stage {
                    // Close the quote left open over the streamed string value.
                    CallStage::StringValue => output.push_call(ToolCallDelta {
                        tool_index: state.tool_index,
                        name: None,
                        arguments: "\"".to_string(),
                    }),
                    CallStage::ScalarValue { key, arg_type } => {
                        let raw = raw.ok_or_else(|| {
                            parsing_failed!("Kimi K3 scalar argument without its value")
                        })?;
                        let value = decode_argument_value(&arg_type, &raw);
                        let key_json = serde_json::to_string(&key).map_err(|error| {
                            parsing_failed!("failed to serialize argument key: {}", error)
                        })?;
                        let value_json = serde_json::to_string(&value).map_err(|error| {
                            parsing_failed!("failed to serialize arguments: {}", error)
                        })?;
                        let separator = if state.arg_emitted { "," } else { "{" };
                        output.push_call(ToolCallDelta {
                            tool_index: state.tool_index,
                            name: None,
                            arguments: format!("{separator}{key_json}:{value_json}"),
                        });
                        state.arg_emitted = true;
                    }
                    // Raw json bodies stream through; nothing to close.
                    CallStage::JsonValue => {}
                    CallStage::BetweenBlocks => {
                        return Err(parsing_failed!(
                            "Kimi K3 argument close without an open argument"
                        ));
                    }
                }
            }
            KimiK3Event::CallEnd => {
                let mode = std::mem::replace(&mut self.mode, KimiK3Mode::Tools);
                let KimiK3Mode::Call(state) = mode else {
                    return Err(parsing_failed!(
                        "Kimi K3 call close without an active tool call"
                    ));
                };
                // An empty/garbage call block without a tool name is dropped
                // (and never consumes an output index), matching the Python
                // parser.
                if !state.dropped {
                    self.emitted_call_count += 1;
                    push_call_close(&state, output);
                }
            }
        }
        Ok(())
    }

    /// The state of the currently open `call` block.
    fn active_call(&mut self) -> Result<&mut CallState> {
        match &mut self.mode {
            KimiK3Mode::Call(state) => Ok(state),
            _ => Err(parsing_failed!(
                "Kimi K3 call event without an active tool call"
            )),
        }
    }

    fn reset_state(&mut self) -> String {
        self.mode = KimiK3Mode::Idle;
        self.emitted_call_count = 0;
        std::mem::take(&mut self.buffer)
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

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(&KIMI_K3_STRUCTURAL_TAG_BUILDER)
    }

    fn parse_into(&mut self, chunk: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_kimi_k3_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();

        match &self.mode {
            KimiK3Mode::Idle | KimiK3Mode::Response => {
                output.push_text(std::mem::take(&mut self.buffer));
            }
            KimiK3Mode::Reasoning => output.push_reasoning(std::mem::take(&mut self.buffer)),
            KimiK3Mode::Epilogue | KimiK3Mode::Done | KimiK3Mode::Tools => {
                // A tools channel truncated between calls loses only its
                // closing markers or an incomplete call header; keep the calls
                // already emitted.
                self.buffer.clear();
            }
            KimiK3Mode::Call(state) => {
                // A call truncated mid-flight: flush the in-flight value and
                // best-effort close the arguments, so the caller still gets a
                // parseable partial tool call instead of an error.
                if !state.dropped {
                    match &state.stage {
                        CallStage::StringValue => {
                            // The value streamed as it arrived; only a held
                            // back partial-marker tail remains here.
                            if !self.buffer.is_empty() {
                                let rest = escape_json_contents(&self.buffer).map_err(|error| {
                                    parsing_failed!("failed to escape argument text: {}", error)
                                })?;
                                output.push_call(ToolCallDelta {
                                    tool_index: state.tool_index,
                                    name: None,
                                    arguments: rest,
                                });
                            }
                        }
                        CallStage::JsonValue => {
                            if !self.buffer.is_empty() {
                                output.push_call(ToolCallDelta {
                                    tool_index: state.tool_index,
                                    name: None,
                                    arguments: std::mem::take(&mut self.buffer),
                                });
                            }
                        }
                        // A buffered scalar's partial value is not guaranteed
                        // valid JSON and is dropped; keep fully-received blocks.
                        CallStage::ScalarValue { .. } | CallStage::BetweenBlocks => {}
                    }
                    push_call_close(state, &mut output);
                }
                self.buffer.clear();
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
        KimiK3Mode::Response => parse_response_event(input),
        KimiK3Mode::Epilogue => parse_epilogue_event(input),
        KimiK3Mode::Tools => parse_tools_event(input),
        KimiK3Mode::Call(state) => parse_call_event(input, state),
        KimiK3Mode::Done => parse_done_event(input),
    }
}

/// Parse an event while waiting for the next channel open.
fn parse_idle_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        literal(THINK_OPEN).value(KimiK3Event::ThinkOpen),
        literal(RESPONSE_OPEN).value(KimiK3Event::ResponseOpen),
        literal(TOOLS_OPEN).value(KimiK3Event::ToolsOpen),
        message_end_event,
        safe_idle_text_event,
    ))
    .parse_next(input)
}

/// Parse an event inside the `think` channel.
fn parse_reasoning_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        literal(THINK_CLOSE).value(KimiK3Event::ThinkClose),
        // `<|end_of_msg|>` can reach the parser under `ignore_eos` or
        // `include_stop_str_in_output`; never leak it into reasoning.
        literal(END_OF_MSG).value(KimiK3Event::MessageEnd),
        safe_reasoning_event,
    ))
    .parse_next(input)
}

/// Parse an event inside the `response` channel.
fn parse_response_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        literal(RESPONSE_CLOSE).value(KimiK3Event::ResponseClose),
        // The response body also implicitly ends at a `tools` channel.
        literal(TOOLS_OPEN).value(KimiK3Event::ToolsOpen),
        message_end_event,
        safe_response_text_event,
    ))
    .parse_next(input)
}

/// Parse an event after the response channel closed.
fn parse_epilogue_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        literal(TOOLS_OPEN).value(KimiK3Event::ToolsOpen),
        message_end_event,
        skip_epilogue_noise_event,
    ))
    .parse_next(input)
}

/// Parse an event inside the `tools` channel, between `call` blocks.
fn parse_tools_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        call_open_event,
        literal(TOOLS_CLOSE).value(KimiK3Event::ToolsClose),
        // Defensive: an unterminated tools channel still ends with the message.
        message_end_event,
        skip_tools_noise_event,
    ))
    .parse_next(input)
}

/// Parse a message close or end-of-message marker.
fn message_end_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((literal(MESSAGE_CLOSE), literal(END_OF_MSG)))
        .value(KimiK3Event::MessageEnd)
        .parse_next(input)
}

/// Ignore everything after the assistant message closed.
fn parse_done_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    rest.value(KimiK3Event::Skip).parse_next(input)
}

/// Parse safe text while waiting for the next channel marker.
fn safe_idle_text_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, IDLE_MARKERS).map(|len| KimiK3Event::Text { len })
}

/// Parse safe reasoning before the think close marker.
fn safe_reasoning_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, REASONING_MARKERS).map(|len| KimiK3Event::Reasoning { len })
}

/// Parse safe response text before the next channel marker.
fn safe_response_text_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, RESPONSE_MARKERS).map(|len| KimiK3Event::Text { len })
}

/// Skip non-content noise after the response channel closed.
fn skip_epilogue_noise_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, EPILOGUE_MARKERS).map(|_| KimiK3Event::Skip)
}

/// Skip non-content noise between `call` blocks.
fn skip_tools_noise_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, TOOLS_MARKERS).map(|_| KimiK3Event::Skip)
}

/// Parse a `call` open tag into its tool name and one-based index.
fn call_open_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let (attrs,) = seq!(
        _: literal(CALL_OPEN),
        take_until(0.., SEP),
        _: literal(SEP),
    )
    .parse_next(input)?;
    let attrs = parse_tag_attrs(attrs)?;

    Ok(KimiK3Event::CallOpen {
        name: attr_value(&attrs, "tool").unwrap_or_default().to_string(),
        index: attr_value(&attrs, "index")
            .filter(|index| !index.is_empty())
            .map(str::to_string),
    })
}

/// Parse one event inside a `call` block for the current stage.
fn parse_call_event(input: &mut KimiK3Input<'_>, state: &CallState) -> ModalResult<KimiK3Event> {
    match &state.stage {
        CallStage::BetweenBlocks => alt((
            argument_open_event,
            json_open_event,
            literal(CALL_CLOSE).value(KimiK3Event::CallEnd),
            // Defensive: an unterminated call still ends with the message.
            message_end_event,
            skip_call_noise_event,
        ))
        .parse_next(input),
        CallStage::StringValue => alt((
            literal(ARG_CLOSE).value(KimiK3Event::ArgumentEnd { raw: None }),
            string_value_text_event,
        ))
        .parse_next(input),
        CallStage::ScalarValue { .. } => scalar_value_end_event(input),
        CallStage::JsonValue => alt((
            literal(JSON_CLOSE).value(KimiK3Event::ArgumentEnd { raw: None }),
            json_value_text_event,
        ))
        .parse_next(input),
    }
}

/// Parse an `argument` open tag into its key and type tag.
fn argument_open_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let (attrs,) = seq!(
        _: literal(ARG_OPEN),
        take_until(0.., SEP),
        _: literal(SEP),
    )
    .parse_next(input)?;
    let attrs = parse_tag_attrs(attrs)?;

    Ok(KimiK3Event::ArgumentOpen {
        key: attr_value(&attrs, "key").unwrap_or_default().to_string(),
        arg_type: attr_value(&attrs, "type").unwrap_or("string").to_string(),
    })
}

/// Parse a raw `json` block open tag; its attrs (`type="object"`) are unused
/// on decode.
fn json_open_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    seq!(
        _: literal(JSON_OPEN),
        _: take_until(0.., SEP),
        _: literal(SEP),
    )
    .parse_next(input)?;

    Ok(KimiK3Event::JsonOpen)
}

/// Decode one typed argument value per its XTML `type` tag.
///
/// `string` values pass the raw text through (the renderer emits them
/// unescaped); other types are JSON-decoded, falling back to the raw text on
/// malformed payloads so one quirky value does not fail the whole call.
fn decode_argument_value(arg_type: &str, raw: &str) -> Value {
    if arg_type == "string" {
        return Value::String(raw.to_string());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Parse safe text of a streamed string-argument value; the run stops (with
/// partial-marker holdback) before the argument close marker.
fn string_value_text_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len(input, ARG_CLOSE).map(|len| KimiK3Event::ValueText { len })
}

/// Parse safe text of a raw `json` body; passed through unmodified.
fn json_value_text_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len(input, JSON_CLOSE).map(|len| KimiK3Event::ValueText { len })
}

/// Skip non-content noise between the blocks of one `call` body.
fn skip_call_noise_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, CALL_BODY_MARKERS).map(|_| KimiK3Event::Skip)
}

/// Parse a buffered non-string argument value through its close marker;
/// partial scalars are never valid JSON, so they buffer until the block
/// closes and then emit as one fragment.
fn scalar_value_end_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let (raw,) = seq!(
        take_until(0.., ARG_CLOSE),
        _: literal(ARG_CLOSE),
    )
    .parse_next(input)?;

    Ok(KimiK3Event::ArgumentEnd {
        raw: Some(raw.to_string()),
    })
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

/// Escape raw argument text as JSON string *contents* (no surrounding
/// quotes). serde_json escapes each char independently, so concatenating
/// escaped fragments equals escaping the whole string.
fn escape_json_contents(text: &str) -> std::result::Result<String, serde_json::Error> {
    let quoted = serde_json::to_string(text)?;
    Ok(quoted[1..quoted.len() - 1].to_string())
}

/// Push the closing fragment of an in-flight call's arguments object.
///
/// Best-effort on truncation: a mid-string value gets its quote and the
/// object closed, a partially buffered scalar is skipped, and a raw `json`
/// body (already streamed verbatim) needs no closing.
fn push_call_close(state: &CallState, output: &mut UnifiedParserOutput) {
    if state.dropped {
        return;
    }
    let closing = match &state.stage {
        CallStage::StringValue => "\"}",
        CallStage::BetweenBlocks | CallStage::JsonValue if state.raw_json => return,
        _ if state.arg_emitted => "}",
        _ => "{}",
    };
    output.push_call(ToolCallDelta {
        tool_index: state.tool_index,
        name: None,
        arguments: closing.to_string(),
    });
}

/// Build a cut error for determinably malformed XTML structure.
fn xtml_error(label: &'static str) -> ErrMode<ContextError> {
    let mut error = ContextError::new();
    error.push(StrContext::Label(label));
    ErrMode::Cut(error)
}

#[cfg(test)]
mod tests;
