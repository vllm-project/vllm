// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

pub mod choices;
pub mod logprobs;
pub mod metrics;
pub mod structured_outputs;
pub mod types;
pub mod usage;

use std::num::NonZeroU32;

use vllm_text::{PromptTruncation, TruncationSide};

use self::types::StringOrArray;
use crate::error::{ApiError, bail_invalid_request};

fn max_stop_strings() -> usize {
    std::env::var("VLLM_MAX_STOP_STRINGS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4)
}

/// Reject stop-sequence arrays longer than `VLLM_MAX_STOP_STRINGS` (default
/// 4), matching Python's `StopParam` (which applies `max_length` to the list
/// arm only). Empty stop strings are rejected in `vllm-text` so every route
/// shares one check.
pub(crate) fn validate_stop_strings(stop: Option<&StringOrArray>) -> Result<(), ApiError> {
    validate_stop_strings_with_max(stop, max_stop_strings())
}

fn validate_stop_strings_with_max(
    stop: Option<&StringOrArray>,
    max: usize,
) -> Result<(), ApiError> {
    let Some(StringOrArray::Array(stops)) = stop else {
        return Ok(());
    };
    if stops.len() > max {
        bail_invalid_request!(
            param = "stop",
            "stop strings must contain at most {max} items"
        );
    }
    Ok(())
}

/// Resolve the number of choices to generate, checked like Python's
/// `SamplingParams` checks `n`.
pub(crate) fn resolve_n(n: Option<u32>) -> Result<NonZeroU32, ApiError> {
    let n = n.unwrap_or(1);
    let Some(resolved) = NonZeroU32::new(n) else {
        bail_invalid_request!(param = "n", "n must be at least 1, got {n}.");
    };
    Ok(resolved)
}

/// Reject more choices than the server's `VLLM_MAX_N_SEQUENCES` allows, like
/// Python's `SamplingParams` checks `n`.
pub(crate) fn validate_n_limit(n: u32, max_n_sequences: i64) -> Result<(), ApiError> {
    if i64::from(n) > max_n_sequences {
        bail_invalid_request!(
            param = "n",
            "n must be at most {max_n_sequences}, got {n}. To increase this limit, set the \
             VLLM_MAX_N_SEQUENCES environment variable."
        );
    }
    Ok(())
}

/// Reject an output budget larger than the whole context window, like
/// Python's `TokenizeParams` check for chat and completions. A budget that
/// fits the window is still clamped to the room the prompt leaves when the
/// request is lowered; `/inference/v1/generate` only clamps, as in Python.
pub(crate) fn validate_max_output_tokens(
    param: &'static str,
    max_tokens: Option<u32>,
    max_model_len: u32,
) -> Result<(), ApiError> {
    if let Some(max_tokens) = max_tokens
        && max_tokens > max_model_len
    {
        bail_invalid_request!(
            param = param,
            "{param}={max_tokens} cannot be greater than max_model_len={max_model_len}. \
             Please request fewer output tokens."
        );
    }
    Ok(())
}

pub(crate) fn resolve_generation_prompt_truncation(
    limit: Option<i64>,
    side: Option<TruncationSide>,
) -> vllm_text::Result<Option<PromptTruncation>> {
    limit
        .map(|limit| PromptTruncation::from_wire(limit, side.unwrap_or(TruncationSide::Left)))
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::{validate_stop_strings, validate_stop_strings_with_max};
    use crate::routes::openai::utils::types::StringOrArray;

    #[test]
    fn validate_stop_strings_accepts_four_stop_strings() {
        let stop = StringOrArray::Array(
            ["one", "two", "three", "four"].into_iter().map(str::to_string).collect(),
        );
        validate_stop_strings(Some(&stop)).expect("four stop strings should be accepted");
    }

    #[test]
    fn validate_stop_strings_rejects_more_than_four_stop_strings() {
        let stop = StringOrArray::Array(
            ["one", "two", "three", "four", "five"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        );
        let error = validate_stop_strings(Some(&stop))
            .expect_err("five stop strings should be rejected")
            .to_error_response();
        assert_eq!(error.error.param.as_deref(), Some("stop"));
        assert!(error.error.message.contains("at most 4 items"));
    }

    #[test]
    fn validate_stop_strings_limits_arrays_only() {
        let single = StringOrArray::String("END".to_string());
        validate_stop_strings_with_max(Some(&single), 0)
            .expect("scalar stop string should bypass list max_length, like Python StopParam");

        let array = StringOrArray::Array(vec!["END".to_string()]);
        let error = validate_stop_strings_with_max(Some(&array), 0)
            .expect_err("array stop string should enforce max_length")
            .to_error_response();
        assert_eq!(error.error.param.as_deref(), Some("stop"));
    }
}
