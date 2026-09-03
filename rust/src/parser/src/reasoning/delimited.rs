// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer, Tokenizer};

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
    buffer: DecodedText,
    start_token: String,
    end_token: String,
    start_token_id: u32,
    end_token_id: u32,
    default_in_reasoning: bool,
}

impl DelimitedReasoningParser {
    /// Create one delimited parser state machine.
    ///
    /// `default_in_reasoning` is only used when prompt initialization sees no
    /// reasoning boundary token at all. If the prompt contains either the
    /// start or end delimiter, that prompt boundary always wins.
    pub(crate) fn new(
        tokenizer: DynTokenizer,
        start_token: impl Into<String>,
        end_token: impl Into<String>,
        default_in_reasoning: bool,
    ) -> Result<Self> {
        let start_token = start_token.into();
        let end_token = end_token.into();
        let start_token_id =
            tokenizer
                .token_to_id(&start_token)
                .ok_or_else(|| ReasoningError::MissingToken {
                    token: start_token.clone(),
                })?;
        let end_token_id =
            tokenizer.token_to_id(&end_token).ok_or_else(|| ReasoningError::MissingToken {
                token: end_token.clone(),
            })?;

        Ok(Self {
            tokenizer,
            current_in_reasoning: default_in_reasoning,
            buffer: DecodedText::default(),
            start_token,
            end_token,
            start_token_id,
            end_token_id,
            default_in_reasoning,
        })
    }

    /// Initialize the starting state from prompt token IDs.
    pub(crate) fn initialize(&mut self, prompt_token_ids: &[u32]) {
        self.current_in_reasoning = last_reasoning_boundary(
            prompt_token_ids,
            self.start_token_id,
            self.end_token_id,
            self.tokenizer.as_ref(),
        )
        .unwrap_or(self.default_in_reasoning);
    }

    /// Return whether the parser is currently inside a reasoning section.
    pub(crate) fn in_reasoning(&self) -> bool {
        self.current_in_reasoning
    }

    /// Parse one decoded text delta and return its reasoning/content split.
    pub(crate) fn push(&mut self, delta: DecodedText) -> ReasoningDelta {
        self.buffer.append(delta);

        let partial_suffix_len = self.partial_suffix_len(&self.buffer.text);
        let stable_len = self.buffer.text.len() - partial_suffix_len;
        let stable = self.buffer.drain_prefix(stable_len);

        self.parse_stable_text(stable)
    }

    /// Flush any buffered partial delimiter suffix at end of stream.
    pub(crate) fn finish(&mut self) -> ReasoningDelta {
        // `drain_prefix(text.len())` takes trailing zero-width tokens too.
        let stable = self.buffer.drain_prefix(self.buffer.text.len());
        self.parse_stable_text(stable)
    }

    /// Parse text that is known not to end with a partial delimiter suffix.
    ///
    /// Reasoning and content pieces keep the attributions of the tokens that
    /// produced them; delimiter marker spans are drained and dropped, keeping
    /// marker tokens out of any count.
    fn parse_stable_text(&mut self, mut stable: DecodedText) -> ReasoningDelta {
        let mut delta = ReasoningDelta::default();

        while !stable.text.is_empty() {
            if self.current_in_reasoning {
                if let Some(end_idx) = stable.text.find(&self.end_token) {
                    delta.push_reasoning(stable.drain_prefix(end_idx));
                    let _ = stable.drain_prefix(self.end_token.len());
                    self.current_in_reasoning = false;
                } else {
                    delta.push_reasoning(stable);
                    return delta;
                }
            } else if let Some(start_idx) = stable.text.find(&self.start_token) {
                delta.push_content(stable.drain_prefix(start_idx));
                let _ = stable.drain_prefix(self.start_token.len());
                self.current_in_reasoning = true;
            } else {
                delta.push_content(stable);
                return delta;
            }
        }

        // A remainder with empty text may still carry zero-width tokens;
        // attribute them to the current state.
        if !stable.attributions.is_empty() {
            if self.current_in_reasoning {
                delta.push_reasoning(stable);
            } else {
                delta.push_content(stable);
            }
        }

        delta
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
