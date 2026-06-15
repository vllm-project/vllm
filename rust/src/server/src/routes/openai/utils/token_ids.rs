use std::collections::HashMap;

use vllm_text::Prompt;

use crate::error::{ApiError, bail_invalid_request};

/// Reject token-id prompt entries at or above `bound` (the highest in-vocab id is
/// `bound - 1`).
pub(crate) fn validate_prompt_token_ids(prompt: &Prompt, bound: usize) -> Result<(), ApiError> {
    if let Prompt::TokenIds(ids) = prompt
        && let Some(&bad) = ids.iter().find(|&&id| id as usize >= bound)
    {
        bail_invalid_request!(
            param = "prompt",
            "prompt contains out-of-vocab token id {bad}; vocabulary size is {bound}."
        );
    }
    Ok(())
}

/// Reject `allowed_token_ids` entries at or above `bound`.
pub(crate) fn validate_allowed_token_ids(
    allowed_token_ids: Option<&[u32]>,
    bound: usize,
) -> Result<(), ApiError> {
    if let Some(ids) = allowed_token_ids
        && let Some(&bad) = ids.iter().find(|&&id| id as usize >= bound)
    {
        bail_invalid_request!(
            param = "allowed_token_ids",
            "allowed_token_ids contains out-of-vocab token id {bad}; vocabulary size is {bound}."
        );
    }
    Ok(())
}

/// Reject `logit_bias` keys at or above `bound`.
pub(crate) fn validate_logit_bias(
    logit_bias: Option<&HashMap<String, f32>>,
    bound: usize,
) -> Result<(), ApiError> {
    if let Some(bias) = logit_bias {
        for key in bias.keys() {
            if let Ok(id) = key.parse::<u32>()
                && id as usize >= bound
            {
                bail_invalid_request!(
                    param = "logit_bias",
                    "logit_bias contains out-of-vocab token id {id}; vocabulary size is {bound}."
                );
            }
        }
    }
    Ok(())
}
