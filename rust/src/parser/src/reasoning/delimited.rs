// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer, Tokenizer};
use winnow::Parser;
use winnow::combinator::alt;
use winnow::error::ErrMode;
use winnow::stream::StreamIsPartial;
use xgrammar_structural_tag::format::Format;

use super::{ReasoningDelta, ReasoningError, Result};
use crate::utils::{Marker, SpecialToken, attributed_with_markers, safe_text_len_mul};

/// Build a delimited reasoning parser with fixed text at each boundary.
pub(crate) struct DelimitedReasoningParserBuilder {
    tokenizer: DynTokenizer,
    start_token: String,
    end_token: String,
    before_start: &'static str,
    after_start: &'static str,
    before_end: &'static str,
    after_end: &'static str,
}

impl DelimitedReasoningParserBuilder {
    /// Configure the tokenizer and reasoning delimiters.
    pub(crate) fn new(
        tokenizer: DynTokenizer,
        start_token: impl Into<String>,
        end_token: impl Into<String>,
    ) -> Self {
        Self {
            tokenizer,
            start_token: start_token.into(),
            end_token: end_token.into(),
            before_start: "",
            after_start: "",
            before_end: "",
            after_end: "",
        }
    }

    /// Include a fixed prefix when it immediately precedes the start marker.
    pub(crate) fn with_before_start(mut self, before_start: &'static str) -> Self {
        self.before_start = before_start;
        self
    }

    /// Consume fixed text immediately after the start marker.
    pub(crate) fn with_after_start(mut self, after_start: &'static str) -> Self {
        self.after_start = after_start;
        self
    }

    /// Include a fixed prefix when it immediately precedes the end marker.
    pub(crate) fn with_before_end(mut self, before_end: &'static str) -> Self {
        self.before_end = before_end;
        self
    }

    /// Consume fixed text immediately after the end marker.
    pub(crate) fn with_after_end(mut self, after_end: &'static str) -> Self {
        self.after_end = after_end;
        self
    }

    /// Create one delimited parser state machine.
    pub(crate) fn build(self) -> Result<DelimitedReasoningParser> {
        let start_token_id = self.tokenizer.token_to_id(&self.start_token).ok_or_else(|| {
            ReasoningError::MissingToken {
                token: self.start_token.clone(),
            }
        })?;
        let end_token_id = self.tokenizer.token_to_id(&self.end_token).ok_or_else(|| {
            ReasoningError::MissingToken {
                token: self.end_token.clone(),
            }
        })?;

        let delimiter = |token: String, id: u32, before: &str| {
            let token = SpecialToken { text: token, id };
            (
                Marker::special(&token),
                Marker::text(before).then_special(&token),
            )
        };
        let (start, framed_start) = delimiter(self.start_token, start_token_id, self.before_start);
        let (end, framed_end) = delimiter(self.end_token, end_token_id, self.before_end);

        Ok(DelimitedReasoningParser {
            tokenizer: self.tokenizer,
            start,
            framed_start,
            end,
            framed_end,
            start_token_id,
            end_token_id,
            after_start: self.after_start,
            after_end: self.after_end,
            current_in_reasoning: false,
            buffer: DecodedText::default(),
            pending_after: "",
        })
    }
}

/// Shared incremental state machine for tag-delimited reasoning protocols.
///
/// This helper is intentionally not a public parser type. Model-family parser
/// wrappers own one `DelimitedReasoningParser` internally and expose the
/// request-facing [`super::ReasoningParser`] trait.
///
/// The shared state machine stays generic by deriving its initial
/// `current_in_reasoning` state from the prompt token boundary instead of
/// hardcoding model-family conventions. That means families with the same
/// delimiters can often reuse this implementation even if their chat templates
/// prefill different prompts.
///
/// A delimiter counts only where its dedicated token produced it: the same
/// bytes spelled by ordinary tokens (the model quoting `</think>`) stay body
/// text, as in the Python parsers' streaming path. The parser therefore needs
/// attributed input; fed unattributed text, it never crosses a delimiter.
pub(crate) struct DelimitedReasoningParser {
    tokenizer: DynTokenizer,
    start: Marker,
    framed_start: Marker,
    end: Marker,
    framed_end: Marker,
    start_token_id: u32,
    end_token_id: u32,
    after_start: &'static str,
    after_end: &'static str,

    // Mutable state.
    current_in_reasoning: bool,
    buffer: DecodedText,
    pending_after: &'static str,
}

impl DelimitedReasoningParser {
    /// Initialize the starting state and remaining framing from prompt tokens.
    pub(crate) fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.buffer = DecodedText::default();
        self.current_in_reasoning = false;
        self.pending_after = "";
        if let Some((index, in_reasoning)) = last_reasoning_boundary_position(
            prompt_token_ids,
            self.start_token_id,
            self.end_token_id,
            self.tokenizer.as_ref(),
        ) {
            self.current_in_reasoning = in_reasoning;
            self.pending_after = if in_reasoning {
                self.after_start
            } else {
                self.after_end
            };
            if !self.pending_after.is_empty() && index + 1 < prompt_token_ids.len() {
                let tail = self.tokenizer.decode(&prompt_token_ids[index + 1..], false)?;
                self.pending_after = self.pending_after.strip_prefix(&tail).unwrap_or("");
            }
        }
        Ok(())
    }

    /// Return whether the parser is currently inside a reasoning section.
    pub(crate) fn in_reasoning(&self) -> bool {
        self.current_in_reasoning
    }

    /// The end delimiter, guarded by its token.
    pub(crate) fn end_marker(&self) -> &Marker {
        &self.end
    }

    /// Wrap visible output from the initialized reasoning and framing state.
    pub(crate) fn wrap_visible_format(&self, visible: &Format) -> Format {
        let begin = if self.current_in_reasoning {
            "".to_string()
        } else {
            self.framed_start.as_str().to_string()
        };
        let mut reasoning = Format::tag(begin, Format::any_text(), self.end.as_str());
        if !self.after_end.is_empty() {
            reasoning = Format::sequence(vec![reasoning, Format::const_string(self.after_end)]);
        }
        if !self.current_in_reasoning {
            reasoning = Format::optional(reasoning);
            if !self.pending_after.is_empty() {
                return Format::sequence(vec![
                    Format::const_string(self.pending_after),
                    reasoning,
                    visible.clone(),
                ]);
            }
        }

        Format::sequence(vec![reasoning, visible.clone()])
    }

    /// Parse one decoded text delta and return its reasoning/content split.
    pub(crate) fn push(&mut self, delta: DecodedText) -> ReasoningDelta {
        self.buffer.append(delta);
        self.parse_buffer(false)
    }

    /// Flush partial framing and delimiters as body text at end of stream.
    pub(crate) fn finish(&mut self) -> ReasoningDelta {
        self.parse_buffer(true)
    }

    /// Reasoning and content pieces keep the attributions of the tokens that
    /// produced them; delimiter marker spans are drained and dropped, keeping
    /// marker tokens out of any count.
    fn parse_buffer(&mut self, finishing: bool) -> ReasoningDelta {
        let mut delta = ReasoningDelta::default();
        loop {
            if !self.pending_after.is_empty() {
                if self.buffer.text.starts_with(self.pending_after) {
                    let _ = self.buffer.drain_prefix(self.pending_after.len());
                } else if !finishing && self.pending_after.starts_with(&self.buffer.text) {
                    break;
                }
                // A mismatch preserves the candidate as body text.
                self.pending_after = "";
            }

            let (marker, framed_marker) = if self.current_in_reasoning {
                (&self.end, &self.framed_end)
            } else {
                (&self.start, &self.framed_start)
            };
            let mut input = attributed_with_markers(&self.buffer, &());
            if finishing {
                // Partial delimiters at the end of the stream are body text.
                let _ = input.complete();
            }
            let boundary = alt((framed_marker, marker)).map(str::len).parse_next(&mut input);
            let body_len = match boundary {
                Ok(boundary_len) => {
                    // Zero-width tokens before the delimiter stay in the current state.
                    let body = self.buffer.drain_prefix(0);
                    self.push_body(&mut delta, body);
                    let _ = self.buffer.drain_prefix(boundary_len);
                    self.current_in_reasoning = !self.current_in_reasoning;
                    self.pending_after = if self.current_in_reasoning {
                        self.after_start
                    } else {
                        self.after_end
                    };
                    continue;
                }
                // A delimiter may start here once more input arrives.
                Err(ErrMode::Incomplete(_)) => None,
                // Body text runs up to the next delimiter that may start; the
                // rest is held back. Fails only on an empty input.
                Err(_) => safe_text_len_mul(&mut input, &[framed_marker, marker]).ok(),
            };
            let body = self.buffer.drain_prefix(body_len.unwrap_or(0));
            self.push_body(&mut delta, body);
            if body_len.is_none() {
                break;
            }
        }
        delta
    }

    fn push_body(&self, delta: &mut ReasoningDelta, body: DecodedText) {
        if self.current_in_reasoning {
            delta.push_reasoning(body);
        } else {
            delta.push_content(body);
        }
    }
}

/// Determine the reasoning state implied by the last prompt boundary, if any.
pub(crate) fn last_reasoning_boundary(
    prompt_token_ids: &[u32],
    start_token_id: u32,
    end_token_id: u32,
    tokenizer: &dyn Tokenizer,
) -> Option<bool> {
    last_reasoning_boundary_position(prompt_token_ids, start_token_id, end_token_id, tokenizer)
        .map(|(_, in_reasoning)| in_reasoning)
}

fn last_reasoning_boundary_position(
    prompt_token_ids: &[u32],
    start_token_id: u32,
    end_token_id: u32,
    tokenizer: &dyn Tokenizer,
) -> Option<(usize, bool)> {
    for (index, &token_id) in prompt_token_ids.iter().enumerate().rev() {
        if token_id == start_token_id {
            return Some((index, true));
        }
        if token_id == end_token_id {
            return Some((index, false));
        }
        if tokenizer.is_special_id(token_id) {
            return None;
        }
    }

    None
}
