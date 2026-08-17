// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DynTokenizer, Tokenizer};

use super::{ReasoningDelta, ReasoningError, Result};

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
    current_in_reasoning: bool,
    buffer: String,
    start_token: String,
    end_token: String,
    /// Vocabulary ID of `start_token`, when the tokenizer has one. Families
    /// whose delimiters are not single vocabulary tokens leave this `None` and
    /// match on text alone.
    start_token_id: Option<u32>,
    /// Vocabulary ID of `end_token`, when the tokenizer has one.
    end_token_id: Option<u32>,
    default_in_reasoning: bool,
    /// Whether each section opens with a framing `\n` that is not part of its
    /// text. See [`Self::strip_framing_newlines`].
    strip_framing_newline: bool,
    /// A delimiter was just consumed, so the next `\n` emitted is framing.
    pending_framing_newline: bool,
}

impl DelimitedReasoningParser {
    /// Create one delimited parser state machine.
    ///
    /// `default_in_reasoning` is only used when prompt initialization sees no
    /// reasoning boundary token at all. If the prompt contains either the
    /// start or end delimiter, that prompt boundary always wins.
    pub(crate) fn new(
        tokenizer: DynTokenizer,
        start_token: &'static str,
        end_token: &'static str,
        default_in_reasoning: bool,
    ) -> Result<Self> {
        let start_token_id =
            tokenizer.token_to_id(start_token).ok_or_else(|| ReasoningError::MissingToken {
                token: start_token.to_string(),
            })?;
        let end_token_id =
            tokenizer.token_to_id(end_token).ok_or_else(|| ReasoningError::MissingToken {
                token: end_token.to_string(),
            })?;

        Ok(Self::with_token_ids(
            tokenizer,
            start_token,
            end_token,
            Some(start_token_id),
            Some(end_token_id),
            default_in_reasoning,
        ))
    }

    /// Create one delimited parser state machine that matches on text alone.
    ///
    /// Some families delimit reasoning with strings that are not vocabulary
    /// tokens, so [`Self::new`] cannot resolve their IDs at all. Such a parser
    /// has no prompt token boundary to inspect and therefore always starts from
    /// `default_in_reasoning`.
    pub(crate) fn new_text_only(
        tokenizer: DynTokenizer,
        start_token: &'static str,
        end_token: &'static str,
        default_in_reasoning: bool,
    ) -> Self {
        Self::with_token_ids(
            tokenizer,
            start_token,
            end_token,
            None,
            None,
            default_in_reasoning,
        )
    }

    /// Build the state machine from already-resolved delimiter IDs.
    fn with_token_ids(
        tokenizer: DynTokenizer,
        start_token: &'static str,
        end_token: &'static str,
        start_token_id: Option<u32>,
        end_token_id: Option<u32>,
        default_in_reasoning: bool,
    ) -> Self {
        Self {
            tokenizer,
            current_in_reasoning: default_in_reasoning,
            buffer: String::new(),
            start_token: start_token.to_string(),
            end_token: end_token.to_string(),
            start_token_id,
            end_token_id,
            default_in_reasoning,
            strip_framing_newline: false,
            pending_framing_newline: false,
        }
    }

    /// Drop the single `\n` that frames the start of every section.
    ///
    /// Some families put a newline immediately after each delimiter as protocol
    /// framing rather than as text. Stripping it here rather than in the wrapper
    /// keeps it correct when several sections are parsed out of one push: the
    /// emitted [`ReasoningDelta`] concatenates them, so by the time a wrapper
    /// sees it the per-section boundaries are gone.
    pub(crate) fn strip_framing_newlines(mut self) -> Self {
        self.strip_framing_newline = true;
        self
    }

    /// Initialize the starting state from prompt token IDs.
    ///
    /// Text-only parsers have no delimiter IDs to look for, so they keep
    /// `default_in_reasoning`.
    pub(crate) fn initialize(&mut self, prompt_token_ids: &[u32]) {
        self.pending_framing_newline = false;

        let Some((start_token_id, end_token_id)) = self.start_token_id.zip(self.end_token_id)
        else {
            self.current_in_reasoning = self.default_in_reasoning;
            return;
        };

        self.current_in_reasoning = last_reasoning_boundary(
            prompt_token_ids,
            start_token_id,
            end_token_id,
            self.tokenizer.as_ref(),
        )
        .unwrap_or(self.default_in_reasoning);
    }

    /// Return whether the parser is currently inside a reasoning section.
    pub(crate) fn in_reasoning(&self) -> bool {
        self.current_in_reasoning
    }

    /// Parse one decoded text delta and return its reasoning/content split.
    pub(crate) fn push(&mut self, delta: &str) -> ReasoningDelta {
        self.buffer.push_str(delta);

        let partial_suffix_len = self.partial_suffix_len(&self.buffer);
        let stable_len = self.buffer.len() - partial_suffix_len;
        let pending_suffix = self.buffer.split_off(stable_len);
        let stable_text = std::mem::replace(&mut self.buffer, pending_suffix);

        self.parse_stable_text(&stable_text)
    }

    /// Flush any buffered partial delimiter suffix at end of stream.
    pub(crate) fn finish(&mut self) -> ReasoningDelta {
        let stable_text = std::mem::take(&mut self.buffer);
        self.parse_stable_text(&stable_text)
    }

    /// Parse text that is known not to end with a partial delimiter suffix.
    fn parse_stable_text(&mut self, mut stable: &str) -> ReasoningDelta {
        let mut delta = ReasoningDelta::default();

        while !stable.is_empty() {
            if self.current_in_reasoning {
                if let Some(end_idx) = stable.find(&self.end_token) {
                    let text = self.take_framing_newline(&stable[..end_idx]);
                    delta.push_reasoning(text);
                    stable = &stable[end_idx + self.end_token.len()..];
                    self.current_in_reasoning = false;
                    self.pending_framing_newline = self.strip_framing_newline;
                } else {
                    delta.push_reasoning(self.take_framing_newline(stable));
                    break;
                }
            } else if let Some(start_idx) = stable.find(&self.start_token) {
                let text = self.take_framing_newline(&stable[..start_idx]);
                delta.push_content(text);
                stable = &stable[start_idx + self.start_token.len()..];
                self.current_in_reasoning = true;
                self.pending_framing_newline = self.strip_framing_newline;
            } else {
                delta.push_content(self.take_framing_newline(stable));
                break;
            }
        }

        delta
    }

    /// Drop a pending framing `\n` from the start of one emitted run.
    ///
    /// An empty run leaves the flag set: the delimiter landed at the end of a
    /// push and its framing newline has not arrived yet.
    fn take_framing_newline<'a>(&mut self, text: &'a str) -> &'a str {
        if !self.pending_framing_newline || text.is_empty() {
            return text;
        }

        self.pending_framing_newline = false;
        text.strip_prefix('\n').unwrap_or(text)
    }

    /// Return the longest trailing suffix that could still complete a
    /// delimiter.
    fn partial_suffix_len(&self, text: &str) -> usize {
        let mut best = 0;
        for idx in text.char_indices().map(|(idx, _)| idx).skip(1) {
            let suffix = &text[idx..];
            if self.start_token.starts_with(suffix) && self.start_token != suffix {
                best = best.max(text.len() - idx);
            }
            if self.end_token.starts_with(suffix) && self.end_token != suffix {
                best = best.max(text.len() - idx);
            }
        }

        if self.start_token.starts_with(text) && self.start_token != text {
            best = best.max(text.len());
        }
        if self.end_token.starts_with(text) && self.end_token != text {
            best = best.max(text.len());
        }

        best
    }
}

/// Determine the reasoning state implied by the last prompt boundary, if any.
pub(crate) fn last_reasoning_boundary(
    prompt_token_ids: &[u32],
    start_token_id: u32,
    end_token_id: u32,
    tokenizer: &dyn Tokenizer,
) -> Option<bool> {
    for token_id in prompt_token_ids.iter().rev().copied() {
        if token_id == start_token_id {
            return Some(true);
        }
        if token_id == end_token_id {
            return Some(false);
        }
        if tokenizer.is_special_id(token_id) {
            return None;
        }
    }

    None
}
