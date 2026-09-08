// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer, Tokenizer};

use super::{ReasoningDelta, ReasoningError, Result};
use crate::utils::partial_prefix_len;

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

        Ok(DelimitedReasoningParser {
            tokenizer: self.tokenizer,
            framed_start_token: format!("{}{}", self.before_start, self.start_token),
            start_token: self.start_token,
            framed_end_token: format!("{}{}", self.before_end, self.end_token),
            end_token: self.end_token,
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
pub(crate) struct DelimitedReasoningParser {
    tokenizer: DynTokenizer,
    start_token: String,
    framed_start_token: String,
    end_token: String,
    start_token_id: u32,
    end_token_id: u32,
    after_start: &'static str,
    after_end: &'static str,
    framed_end_token: String,

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
                (&self.end_token, &self.framed_end_token)
            } else {
                (&self.start_token, &self.framed_start_token)
            };
            if let Some(index) = self.buffer.text.find(marker) {
                let before = &framed_marker[..framed_marker.len() - marker.len()];
                let body_len = if self.buffer.text[..index].ends_with(before) {
                    index - before.len()
                } else {
                    index
                };
                let boundary_len = index + marker.len() - body_len;
                let body = self.buffer.drain_prefix(body_len);
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

            let keep_len = if finishing {
                0
            } else {
                partial_prefix_len(&self.buffer.text, marker)
                    .max(partial_prefix_len(&self.buffer.text, framed_marker))
            };
            let body = self.buffer.drain_prefix(self.buffer.text.len() - keep_len);
            self.push_body(&mut delta, body);
            break;
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
