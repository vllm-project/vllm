// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

pub mod logprobs;
pub mod metrics;
pub mod structured_outputs;
pub mod types;
pub mod usage;
pub mod validated_json;

use std::num::NonZeroU32;

use vllm_text::{BeamSearchParams, PromptTruncation, TruncationSide};

use crate::error::{ApiError, bail_invalid_request};

pub(crate) fn validate_generation_prompt_truncation(
    limit: Option<i64>,
    echo: bool,
) -> Result<(), ApiError> {
    if let Some(limit) = limit
        && limit < -1
    {
        bail_invalid_request!(
            param = "truncate_prompt_tokens",
            "truncate_prompt_tokens must be >= -1."
        );
    }
    if echo && limit.is_some() {
        bail_invalid_request!(
            param = "echo",
            "`echo=true` is not supported with `truncate_prompt_tokens`."
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

/// Validate the beam-search fields shared by chat and completions.
pub(crate) fn validate_beam_search(
    use_beam_search: bool,
    n: Option<u32>,
    length_penalty: Option<f64>,
    logprob_token_ids: Option<&[u32]>,
) -> Result<(), ApiError> {
    if !use_beam_search {
        // TODO: support parallel sampling with `n > 1`.
        if n.unwrap_or(1) > 1 {
            bail_invalid_request!(
                param = "n",
                "Only n=1 is supported without use_beam_search."
            );
        }
        // Python's default is 1.0, which SDKs may send explicitly.
        if length_penalty.is_some_and(|penalty| penalty != 1.0) {
            bail_invalid_request!(
                param = "length_penalty",
                "length_penalty is only supported with use_beam_search."
            );
        }
        return Ok(());
    }
    if logprob_token_ids.is_some_and(|ids| !ids.is_empty()) {
        bail_invalid_request!(
            param = "logprob_token_ids",
            "`logprob_token_ids` is not supported with beam search."
        );
    }
    Ok(())
}

/// Beam-search parameters with width `n`, or `None` without `use_beam_search`.
pub(crate) fn resolve_beam_search(
    use_beam_search: bool,
    n: Option<u32>,
    length_penalty: Option<f64>,
) -> Result<Option<BeamSearchParams>, ApiError> {
    if !use_beam_search {
        return Ok(None);
    }
    let Some(beam_width) = NonZeroU32::new(n.unwrap_or(1)) else {
        bail_invalid_request!("beam_width must be at least 1, got 0.");
    };
    Ok(Some(BeamSearchParams {
        beam_width,
        length_penalty: length_penalty.unwrap_or(1.0),
    }))
}
