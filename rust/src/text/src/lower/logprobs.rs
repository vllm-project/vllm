//! Python-compatible validation for logprobs sampling params.
//!
//! `-1` is expanded only for bounds checks. The original request values are
//! passed through to engine-core.

use crate::backend::SamplingLimits;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LogprobsError {
    #[error("{parameter} must be non-negative or -1, got {value}")]
    InvalidCount { parameter: &'static str, value: i32 },
    #[error(
        "requested {parameter} of {requested}, which is greater than max allowed: {max_allowed}"
    )]
    TooManyCount {
        parameter: &'static str,
        requested: usize,
        max_allowed: usize,
    },
    #[error(
        "requested logprob_token_ids of length {requested}, \
         which is greater than max allowed: {max_allowed}"
    )]
    TooManyTokenIds {
        requested: usize,
        max_allowed: usize,
    },
    #[error(
        "token_id(s) {token_ids:?} in logprob_token_ids contain out-of-vocab token ids. \
         Vocabulary size: {vocab_size}"
    )]
    InvalidTokenIds {
        token_ids: Vec<u32>,
        vocab_size: usize,
    },
    #[error(
        "when both logprobs and logprob_token_ids are set, logprobs must equal \
         len(logprob_token_ids). Got logprobs={logprobs}, len(logprob_token_ids)={num_token_ids}."
    )]
    TokenIdsMismatch { logprobs: i32, num_token_ids: usize },
}

/// Validate logprobs-related sampling parameters, returning an error if any
/// parameter is out of bounds or if the combination of parameters is invalid.
pub(super) fn validate_logprobs(
    logprobs: Option<i32>,
    prompt_logprobs: Option<i32>,
    logprob_token_ids: Option<&[u32]>,
    sampling_limits: SamplingLimits,
) -> Result<(), LogprobsError> {
    let vocab_size = sampling_limits.logprobs_vocab_size();
    let max_logprobs =
        normalize_logprobs_count(sampling_limits.max_logprobs, vocab_size, "max_logprobs")?;

    validate_logprobs_count(logprobs, max_logprobs, vocab_size, "logprobs")?;
    validate_logprobs_count(prompt_logprobs, max_logprobs, vocab_size, "prompt_logprobs")?;
    validate_logprob_token_ids(logprobs, logprob_token_ids, vocab_size)
}

fn validate_logprobs_count(
    requested: Option<i32>,
    max_logprobs: usize,
    vocab_size: usize,
    parameter: &'static str,
) -> Result<(), LogprobsError> {
    let Some(requested) = requested else {
        return Ok(());
    };

    let requested = normalize_logprobs_count(requested, vocab_size, parameter)?;
    if requested > max_logprobs {
        return Err(LogprobsError::TooManyCount {
            parameter,
            requested,
            max_allowed: max_logprobs,
        });
    }

    Ok(())
}

fn validate_logprob_token_ids(
    logprobs: Option<i32>,
    logprob_token_ids: Option<&[u32]>,
    vocab_size: usize,
) -> Result<(), LogprobsError> {
    let Some(logprob_token_ids) = logprob_token_ids else {
        return Ok(());
    };

    let n = logprob_token_ids.len();
    if n > SamplingLimits::MAX_LOGPROB_TOKEN_IDS {
        return Err(LogprobsError::TooManyTokenIds {
            requested: n,
            max_allowed: SamplingLimits::MAX_LOGPROB_TOKEN_IDS,
        });
    }

    let invalid_token_ids: Vec<_> = logprob_token_ids
        .iter()
        .copied()
        .filter(|&token_id| token_id as usize >= vocab_size)
        .collect();
    if !invalid_token_ids.is_empty() {
        return Err(LogprobsError::InvalidTokenIds {
            token_ids: invalid_token_ids,
            vocab_size,
        });
    }

    if let Some(logprobs) = logprobs
        && logprobs != n as i32
    {
        return Err(LogprobsError::TokenIdsMismatch {
            logprobs,
            num_token_ids: n,
        });
    }

    Ok(())
}

fn normalize_logprobs_count(
    value: i32,
    vocab_size: usize,
    parameter: &'static str,
) -> Result<usize, LogprobsError> {
    match value {
        -1 => Ok(vocab_size),
        value if value < 0 => Err(LogprobsError::InvalidCount { parameter, value }),
        value => Ok(value as usize),
    }
}
