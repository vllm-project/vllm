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
    /// Use the input budget remaining after reserving output tokens.
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

    pub(crate) fn apply(self, prompt_token_ids: &mut Vec<u32>, input_budget: u32) -> Result<()> {
        let max_input_tokens = match self.limit {
            PromptTruncationLimit::InputBudget => input_budget,
            PromptTruncationLimit::Fixed(limit) => {
                if limit > u64::from(input_budget) {
                    return Err(Error::TruncatePromptTokensExceedsBudget {
                        value: limit,
                        budget: input_budget,
                    });
                }
                limit as u32
            }
        } as usize;

        if prompt_token_ids.len() <= max_input_tokens {
            return Ok(());
        }

        match self.side {
            TruncationSide::Left => {
                let start = prompt_token_ids.len() - max_input_tokens;
                prompt_token_ids.drain(0..start);
            }
            TruncationSide::Right => prompt_token_ids.truncate(max_input_tokens),
        }
        Ok(())
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
    fn input_budget_reserves_output_tokens() {
        let mut prompt = vec![0; 100];
        PromptTruncation {
            limit: PromptTruncationLimit::InputBudget,
            side: TruncationSide::Left,
        }
        .apply(&mut prompt, 70)
        .unwrap();
        assert_eq!(prompt.len(), 70);
    }

    #[test]
    fn fixed_limit_cannot_exceed_input_budget() {
        let mut prompt = vec![1, 2, 3];
        let error = PromptTruncation {
            limit: PromptTruncationLimit::Fixed(80),
            side: TruncationSide::Left,
        }
        .apply(&mut prompt, 70)
        .unwrap_err();
        assert!(matches!(
            error,
            Error::TruncatePromptTokensExceedsBudget {
                value: 80,
                budget: 70
            }
        ));
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
