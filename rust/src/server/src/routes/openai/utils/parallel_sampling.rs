// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Helpers for OpenAI `n > 1` parallel sampling.
//!
//! Like Python vLLM's `ParentRequest`, each choice is submitted as its own
//! child engine request, and the route merges the per-choice outputs back into
//! one response.

use super::types::Usage;
use crate::error::{ApiError, invalid_request};

/// Request ID of one child choice request.
///
/// Uses Python's `{index}_{parent}` scheme when `n > 1`, and keeps the parent
/// ID for `n = 1` so single-choice requests are unaffected.
pub(crate) fn choice_request_id(parent_request_id: &str, index: u32, n: u32) -> String {
    if n > 1 {
        format!("{index}_{parent_request_id}")
    } else {
        parent_request_id.to_string()
    }
}

/// Seed of one child choice request.
///
/// Offset by the choice index, matching Python, so that seeded choices differ
/// from each other while staying reproducible.
pub(crate) fn choice_seed(seed: Option<i64>, index: u32) -> Result<Option<i64>, ApiError> {
    seed.map(|seed| {
        seed.checked_add(i64::from(index)).ok_or_else(|| {
            invalid_request!(
                param = "seed",
                "`seed + n - 1` must fit in a signed 64-bit integer."
            )
        })
    })
    .transpose()
}

/// Fold the usage of one more choice into the usage of a multi-choice response.
///
/// All choices share one prompt, so prompt tokens (and prompt cache details)
/// are counted once while completion and reasoning tokens are summed.
pub(crate) fn merge_choice_usage(usage: Usage, choice_usage: Usage) -> Usage {
    let completion_tokens =
        usage.completion_tokens.unwrap_or(0) + choice_usage.completion_tokens.unwrap_or(0);
    let mut completion_tokens_details = usage.completion_tokens_details;
    completion_tokens_details.reasoning_tokens +=
        choice_usage.completion_tokens_details.reasoning_tokens;
    Usage {
        total_tokens: usage.prompt_tokens + completion_tokens,
        completion_tokens: Some(completion_tokens),
        completion_tokens_details,
        ..usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_request_id_keeps_parent_id_for_single_choice() {
        assert_eq!(choice_request_id("cmpl-1", 0, 1), "cmpl-1");
        assert_eq!(choice_request_id("cmpl-1", 0, 2), "0_cmpl-1");
        assert_eq!(choice_request_id("cmpl-1", 1, 2), "1_cmpl-1");
    }

    #[test]
    fn choice_seed_offsets_by_choice_index() {
        assert_eq!(choice_seed(None, 1).expect("no seed"), None);
        assert_eq!(choice_seed(Some(41), 1).expect("seed fits"), Some(42));
        assert!(choice_seed(Some(i64::MAX), 1).is_err());
    }

    #[test]
    fn merge_choice_usage_counts_prompt_once() {
        let usage = merge_choice_usage(
            Usage::from_counts(5, 3, Some(4), 1),
            Usage::from_counts(5, 2, Some(4), 2),
        );

        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, Some(5));
        assert_eq!(usage.total_tokens, 10);
        assert_eq!(
            usage.prompt_tokens_details.map(|details| details.cached_tokens),
            Some(4)
        );
        assert_eq!(usage.completion_tokens_details.reasoning_tokens, 3);
    }
}
