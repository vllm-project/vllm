// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Which side of the prompt to discard when truncation is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TruncationSide {
    /// Keep the last N tokens, discarding the prompt prefix.
    Left,
    /// Keep the first N tokens, discarding the prompt suffix.
    Right,
}

/// Maximum number of prompt tokens to retain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptTruncationLimit {
    /// Retain `max_model_len - 1` tokens, holding one token of context back
    /// so generation always has room for at least one output token.
    ///
    /// Python's `TokenizeParams.max_truncation_tokens`; `max_tokens` is an
    /// upper bound that is clamped after tokenization, not a reservation.
    InputBudget,
    /// Retain at most the given number of prompt tokens.
    Fixed(u64),
}

/// Typed prompt-truncation policy used inside the text frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTruncation {
    /// Maximum number of prompt tokens to retain.
    pub limit: PromptTruncationLimit,
    /// Side from which excess prompt tokens are discarded.
    pub side: TruncationSide,
}

impl PromptTruncation {
    /// Convert the OpenAI-compatible integer representation into a typed policy.
    pub fn from_wire(limit: i64, side: TruncationSide) -> Result<Self> {
        let limit = match limit {
            -1 => PromptTruncationLimit::InputBudget,
            0.. => PromptTruncationLimit::Fixed(limit as u64),
            _ => return Err(Error::InvalidTruncatePromptTokens { value: limit }),
        };
        Ok(Self { limit, side })
    }

    pub(crate) fn apply(self, prompt_token_ids: &mut Vec<u32>, max_model_len: u32) -> Result<()> {
        let max_input_tokens = self.max_input_tokens(max_model_len)?;
        self.truncate(prompt_token_ids, max_input_tokens);
        Ok(())
    }

    /// Truncate a text prompt like Hugging Face tokenizers do, which is what the
    /// Python frontend uses: only the text's own tokens are cut, and the special
    /// tokens the tokenizer added around them (such as BOS) are kept.
    ///
    /// `content_token_ids` encodes the same text without special tokens. It is
    /// only called when the prompt is too long. If its ids are not a contiguous
    /// run of `prompt_token_ids`, or the added tokens alone exceed the limit,
    /// this falls back to [`Self::apply`].
    pub(crate) fn apply_keeping_added_tokens(
        self,
        prompt_token_ids: &mut Vec<u32>,
        content_token_ids: impl FnOnce() -> Result<Vec<u32>>,
        max_model_len: u32,
    ) -> Result<()> {
        let max_input_tokens = self.max_input_tokens(max_model_len)?;
        if prompt_token_ids.len() <= max_input_tokens {
            return Ok(());
        }
        let mut content_token_ids = content_token_ids()?;
        let content_len = content_token_ids.len();
        let num_added = prompt_token_ids.len().saturating_sub(content_len);
        let content_start = (0..=num_added)
            .find(|&start| prompt_token_ids[start..].starts_with(&content_token_ids));
        match content_start {
            Some(start) if num_added <= max_input_tokens => {
                self.truncate(&mut content_token_ids, max_input_tokens - num_added);
                prompt_token_ids.splice(start..start + content_len, content_token_ids);
            }
            _ => self.truncate(prompt_token_ids, max_input_tokens),
        }
        Ok(())
    }

    fn max_input_tokens(self, max_model_len: u32) -> Result<usize> {
        let max_input_tokens = match self.limit {
            PromptTruncationLimit::InputBudget => max_model_len.saturating_sub(1),
            PromptTruncationLimit::Fixed(limit) => {
                if limit > u64::from(max_model_len) {
                    return Err(Error::TruncatePromptTokensExceedsMaxModelLen {
                        value: limit,
                        max_model_len,
                    });
                }
                limit as u32
            }
        };
        Ok(max_input_tokens as usize)
    }

    fn truncate(self, token_ids: &mut Vec<u32>, max_tokens: usize) {
        if token_ids.len() <= max_tokens {
            return;
        }
        match self.side {
            TruncationSide::Left => {
                let start = token_ids.len() - max_tokens;
                token_ids.drain(0..start);
            }
            TruncationSide::Right => token_ids.truncate(max_tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_wire_limits() {
        assert_eq!(
            PromptTruncation::from_wire(-1, TruncationSide::Left).unwrap(),
            PromptTruncation {
                limit: PromptTruncationLimit::InputBudget,
                side: TruncationSide::Left,
            }
        );
        assert_eq!(
            PromptTruncation::from_wire(0, TruncationSide::Right).unwrap(),
            PromptTruncation {
                limit: PromptTruncationLimit::Fixed(0),
                side: TruncationSide::Right,
            }
        );
        assert!(matches!(
            PromptTruncation::from_wire(-2, TruncationSide::Left),
            Err(Error::InvalidTruncatePromptTokens { value: -2 })
        ));
    }

    #[test]
    fn applies_fixed_limit_from_either_side() {
        let mut left = vec![1, 2, 3, 4, 5];
        PromptTruncation {
            limit: PromptTruncationLimit::Fixed(3),
            side: TruncationSide::Left,
        }
        .apply(&mut left, 90)
        .unwrap();
        assert_eq!(left, vec![3, 4, 5]);

        let mut right = vec![1, 2, 3, 4, 5];
        PromptTruncation {
            limit: PromptTruncationLimit::Fixed(3),
            side: TruncationSide::Right,
        }
        .apply(&mut right, 90)
        .unwrap();
        assert_eq!(right, vec![1, 2, 3]);
    }

    #[test]
    fn keeps_added_tokens_when_truncating_from_either_side() {
        // A BOS (100) and an EOS (101) around the text's own tokens 1..=5.
        for (side, expected) in [
            (TruncationSide::Left, vec![100, 4, 5, 101]),
            (TruncationSide::Right, vec![100, 1, 2, 101]),
        ] {
            let mut prompt = vec![100, 1, 2, 3, 4, 5, 101];
            PromptTruncation {
                limit: PromptTruncationLimit::Fixed(4),
                side,
            }
            .apply_keeping_added_tokens(&mut prompt, || Ok(vec![1, 2, 3, 4, 5]), 90)
            .unwrap();
            assert_eq!(prompt, expected);
        }
    }

    #[test]
    fn keeping_added_tokens_falls_back_to_plain_truncation() {
        let left = PromptTruncation {
            limit: PromptTruncationLimit::Fixed(2),
            side: TruncationSide::Left,
        };
        // The text's own ids are not a contiguous run of the prompt.
        let mut prompt = vec![100, 1, 9, 2, 3];
        left.apply_keeping_added_tokens(&mut prompt, || Ok(vec![1, 2, 3]), 90).unwrap();
        assert_eq!(prompt, vec![2, 3]);
        // The added tokens alone exceed the limit.
        let mut prompt = vec![100, 101, 102, 1, 2];
        left.apply_keeping_added_tokens(&mut prompt, || Ok(vec![1, 2]), 90).unwrap();
        assert_eq!(prompt, vec![1, 2]);
    }

    #[test]
    fn input_budget_holds_back_one_token_for_output() {
        let mut prompt = vec![0; 100];
        PromptTruncation {
            limit: PromptTruncationLimit::InputBudget,
            side: TruncationSide::Left,
        }
        .apply(&mut prompt, 70)
        .unwrap();
        assert_eq!(prompt.len(), 69);
    }

    #[test]
    fn fixed_limit_cannot_exceed_max_model_len() {
        let mut prompt = vec![1, 2, 3];
        let error = PromptTruncation {
            limit: PromptTruncationLimit::Fixed(80),
            side: TruncationSide::Left,
        }
        .apply(&mut prompt, 70)
        .unwrap_err();
        assert!(matches!(
            error,
            Error::TruncatePromptTokensExceedsMaxModelLen {
                value: 80,
                max_model_len: 70
            }
        ));

        let mut prompt = vec![1, 2, 3];
        PromptTruncation {
            limit: PromptTruncationLimit::Fixed(70),
            side: TruncationSide::Left,
        }
        .apply(&mut prompt, 70)
        .unwrap();
        assert_eq!(prompt, vec![1, 2, 3]);
    }

    #[test]
    fn zero_limit_clears_prompt() {
        let mut prompt = vec![1, 2, 3];
        PromptTruncation {
            limit: PromptTruncationLimit::Fixed(0),
            side: TruncationSide::Left,
        }
        .apply(&mut prompt, 90)
        .unwrap();
        assert!(prompt.is_empty());
    }

    #[test]
    fn limit_at_or_above_prompt_length_is_a_noop() {
        for limit in [3, 10] {
            let mut prompt = vec![1, 2, 3];
            PromptTruncation {
                limit: PromptTruncationLimit::Fixed(limit),
                side: TruncationSide::Left,
            }
            .apply(&mut prompt, 90)
            .unwrap();
            assert_eq!(prompt, vec![1, 2, 3]);
        }
    }
}
