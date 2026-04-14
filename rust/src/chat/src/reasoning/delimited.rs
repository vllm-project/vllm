use vllm_text::tokenizers::Tokenizer;

use super::{ReasoningDelta, ReasoningError, Result};

/// Shared incremental state machine for tag-delimited reasoning protocols.
///
/// This helper is intentionally not a public parser type. Model-family parser
/// wrappers own one `DelimitedReasoningParser` internally and expose the
/// request-facing [`super::ReasoningParser`] trait.
///
/// The shared state machine stays generic by deriving its initial
/// `current_in_reasoning` state from the rendered prompt instead of hardcoding
/// model-family conventions. That means families with the same delimiters can
/// often reuse this implementation even if one template prefills `<think>` and
/// another expects the model to emit it (like Qwen3 vs Qwen3 Thinking vs Qwen3.5).
pub(crate) struct DelimitedReasoningParser {
    current_in_reasoning: bool,
    buffer: String,
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
        tokenizer: &dyn Tokenizer,
        start_token: &'static str,
        end_token: &'static str,
        default_in_reasoning: bool,
    ) -> Result<Self> {
        let start_token_id =
            tokenizer
                .token_to_id(start_token)
                .ok_or_else(|| ReasoningError::MissingToken {
                    token: start_token.to_string(),
                })?;
        let end_token_id =
            tokenizer
                .token_to_id(end_token)
                .ok_or_else(|| ReasoningError::MissingToken {
                    token: end_token.to_string(),
                })?;

        Ok(Self {
            current_in_reasoning: default_in_reasoning,
            buffer: String::new(),
            start_token: start_token.to_string(),
            end_token: end_token.to_string(),
            start_token_id,
            end_token_id,
            default_in_reasoning,
        })
    }

    /// Initialize the starting state from prompt token IDs.
    pub(crate) fn initialize(&mut self, prompt_token_ids: &[u32]) {
        self.current_in_reasoning =
            last_reasoning_boundary(prompt_token_ids, self.start_token_id, self.end_token_id)
                .unwrap_or(self.default_in_reasoning);
    }

    /// Parse one decoded text delta and return its reasoning/content split.
    pub(crate) fn push(&mut self, delta: &str) -> ReasoningDelta {
        self.buffer.push_str(delta);

        let partial_suffix_len = self.partial_suffix_len(&self.buffer);
        let stable_len = self.buffer.len() - partial_suffix_len;
        let stable_text = self.buffer[..stable_len].to_string();
        let pending_suffix = self.buffer[stable_len..].to_string();
        self.buffer = pending_suffix;

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
                    delta.push_reasoning(&stable[..end_idx]);
                    stable = &stable[end_idx + self.end_token.len()..];
                    self.current_in_reasoning = false;
                } else {
                    delta.push_reasoning(stable);
                    break;
                }
            } else if let Some(start_idx) = stable.find(&self.start_token) {
                delta.push_content(&stable[..start_idx]);
                stable = &stable[start_idx + self.start_token.len()..];
                self.current_in_reasoning = true;
            } else {
                delta.push_content(stable);
                break;
            }
        }

        delta
    }

    /// Return the longest trailing suffix that could still complete a delimiter.
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
fn last_reasoning_boundary(
    prompt_token_ids: &[u32],
    start_token_id: u32,
    end_token_id: u32,
) -> Option<bool> {
    for token_id in prompt_token_ids.iter().rev() {
        if *token_id == start_token_id {
            return Some(true);
        }
        if *token_id == end_token_id {
            return Some(false);
        }
    }

    None
}
