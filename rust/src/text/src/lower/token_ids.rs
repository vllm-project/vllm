// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeSet;
use std::result::Result;

use thiserror::Error;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

use crate::SamplingLimits;
use crate::request::SamplingParams;

#[derive(Debug, Error)]
pub enum TokenIdsError {
    #[error("allowed_token_ids should not be empty")]
    EmptyAllowedTokenIds,
    #[error(
        "token_id(s) {token_ids:?} in {parameter} are out of vocabulary. \
         Vocabulary size: {vocab_size}"
    )]
    OutOfVocab {
        parameter: &'static str,
        token_ids: Vec<u32>,
        vocab_size: usize,
    },
    #[error("{parameter} has {requested} entries, which exceeds the maximum of {max_allowed}")]
    TooManyEntries {
        parameter: &'static str,
        requested: usize,
        max_allowed: usize,
    },
    #[error("{parameter} entry has length {requested}, which exceeds the maximum of {max_allowed}")]
    EntryTooLong {
        parameter: &'static str,
        requested: usize,
        max_allowed: usize,
    },
    #[error(
        "{parameter} tokenization produced {requested} entries, \
         which exceeds the maximum of {max_allowed}"
    )]
    TooManyTokenizedEntries {
        parameter: &'static str,
        requested: usize,
        max_allowed: usize,
    },
    #[error(
        "{parameter} tokenization produced {requested} total tokens, \
         which exceeds the maximum of {max_allowed}"
    )]
    TooManyTokenizedTokens {
        parameter: &'static str,
        requested: usize,
        max_allowed: usize,
    },
}

fn validate_count(
    parameter: &'static str,
    requested: usize,
    max_allowed: usize,
) -> Result<(), TokenIdsError> {
    if requested <= max_allowed {
        return Ok(());
    }

    Err(TokenIdsError::TooManyEntries {
        parameter,
        requested,
        max_allowed,
    })
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

/// Validate raw sampler-control sizes before any tokenization or engine serialization.
pub(crate) fn validate_sampler_control_sizes(params: &SamplingParams) -> Result<(), TokenIdsError> {
    if let Some(allowed_token_ids) = params.allowed_token_ids.as_deref() {
        validate_count(
            "allowed_token_ids",
            allowed_token_ids.len(),
            SamplingLimits::MAX_ALLOWED_TOKEN_IDS,
        )?;
    }
    if let Some(logit_bias) = params.logit_bias.as_ref() {
        validate_count(
            "logit_bias",
            logit_bias.len(),
            SamplingLimits::MAX_LOGIT_BIAS_TOKENS,
        )?;
    }
    if let Some(bad_words) = params.bad_words.as_deref() {
        validate_count(
            "bad_words",
            bad_words.len(),
            SamplingLimits::MAX_BAD_WORDS_INPUT_COUNT,
        )?;
        for bad_word in bad_words {
            let length = bad_word.chars().count();
            if length > SamplingLimits::MAX_BAD_WORD_INPUT_LENGTH {
                return Err(TokenIdsError::EntryTooLong {
                    parameter: "bad_words",
                    requested: length,
                    max_allowed: SamplingLimits::MAX_BAD_WORD_INPUT_LENGTH,
                });
            }
        }
    }

    Ok(())
}

/// Validate resolved stop-token state after model EOS aliases have been merged in.
pub(crate) fn validate_resolved_stop_token_ids(
    all_stop_token_ids: &BTreeSet<u32>,
) -> Result<(), TokenIdsError> {
    validate_count(
        "stop_token_ids after EOS expansion",
        all_stop_token_ids.len(),
        SamplingLimits::MAX_STOP_TOKEN_IDS,
    )
}

/// Validate tokenized bad-word state against the engine's fixed-size sampler buffers.
pub(crate) fn validate_bad_words_tokenized_shape(
    bad_words_token_ids: &[Vec<u32>],
) -> Result<(), TokenIdsError> {
    if bad_words_token_ids.len() > SamplingLimits::MAX_BAD_WORD_TOKEN_SEQUENCES {
        return Err(TokenIdsError::TooManyTokenizedEntries {
            parameter: "bad_words",
            requested: bad_words_token_ids.len(),
            max_allowed: SamplingLimits::MAX_BAD_WORD_TOKEN_SEQUENCES,
        });
    }

    let total_tokens = bad_words_token_ids.iter().map(Vec::len).sum();
    if total_tokens > SamplingLimits::MAX_BAD_WORD_TOTAL_TOKENS {
        return Err(TokenIdsError::TooManyTokenizedTokens {
            parameter: "bad_words",
            requested: total_tokens,
            max_allowed: SamplingLimits::MAX_BAD_WORD_TOTAL_TOKENS,
        });
    }

    Ok(())
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
