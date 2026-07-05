use std::result::Result;

use thiserror::Error;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

use crate::SamplingLimits;

#[derive(Debug, Error)]
pub enum TokenIdsError {
    #[error("allowed_token_ids should not be empty")]
    EmptyAllowedTokenIds,
    #[error(
        "token_id(s) {token_ids:?} in {parameter} contain out-of-vocab token ids. \
         Vocabulary size: {vocab_size}"
    )]
    OutOfVocab {
        parameter: &'static str,
        token_ids: Vec<u32>,
        vocab_size: usize,
    },
}

fn validate_param(
    parameter: &'static str,
    token_ids: impl IntoIterator<Item = u32>,
    vocab_size: usize,
) -> Result<(), TokenIdsError> {
    let invalid_token_ids: Vec<_> = token_ids
        .into_iter()
        .filter(|&token_id| token_id as usize >= vocab_size)
        .collect();
    if invalid_token_ids.is_empty() {
        return Ok(());
    }

    Err(TokenIdsError::OutOfVocab {
        parameter,
        token_ids: invalid_token_ids,
        vocab_size,
    })
}

/// Validate that pre-tokenized prompt IDs are within the engine-visible prompt
/// vocabulary range.
pub(crate) fn validate_prompt_token_ids(
    prompt_token_ids: &[u32],
    limits: &SamplingLimits,
) -> Result<(), TokenIdsError> {
    validate_param(
        "prompt",
        prompt_token_ids.iter().copied(),
        limits.prompt_token_vocab_size(),
    )
}

/// Validate that token IDs in text sampling parameters are within their
/// parameter-specific vocabulary ranges.
pub(crate) fn validate_vocab_range(
    params: &EngineCoreSamplingParams,
    limits: &SamplingLimits,
) -> Result<(), TokenIdsError> {
    validate_param(
        "stop_token_ids",
        params.stop_token_ids.iter().copied(),
        limits.model_vocab_size,
    )?;

    if let Some(token_ids) = params.allowed_token_ids.as_deref() {
        if token_ids.is_empty() {
            return Err(TokenIdsError::EmptyAllowedTokenIds);
        }
        validate_param(
            "allowed_token_ids",
            token_ids.iter().copied(),
            limits.tokenizer_vocab_size,
        )?;
    }

    if let Some(logit_bias) = params.logit_bias.as_ref() {
        validate_param(
            "logit_bias",
            logit_bias.keys().copied(),
            limits.model_vocab_size,
        )?;
    }

    if let Some(token_ids) = params.logprob_token_ids.as_deref() {
        validate_param(
            "logprob_token_ids",
            token_ids.iter().copied(),
            limits.model_vocab_size,
        )?;
    }

    if let Some(bad_words_token_ids) = params.bad_words_token_ids.as_deref() {
        validate_param(
            "bad_words",
            bad_words_token_ids.iter().flatten().copied(),
            limits.tokenizer_vocab_size,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_vocab_range_rejects_out_of_vocab_ids() {
        let error = validate_param("logprob_token_ids", [5_u32, 1000, 1001], 1000).unwrap_err();

        assert!(matches!(
            error,
            TokenIdsError::OutOfVocab {
                parameter: "logprob_token_ids",
                token_ids,
                vocab_size: 1000,
            } if token_ids == vec![1000, 1001]
        ));
    }
}
