// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Python-compatible validation for logprobs sampling params.
//!
//! `-1` is expanded only for bounds checks. The original request values are
//! passed through to engine-core.

use std::collections::HashSet;

use thiserror::Error;

use crate::backend::SamplingLimits;

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
        "when both logprobs and logprob_token_ids are set, logprobs must equal \
         len(logprob_token_ids). Got logprobs={logprobs}, len(logprob_token_ids)={num_token_ids}."
    )]
    TokenIdsMismatch { logprobs: i32, num_token_ids: usize },
    #[error("prompt_logprob_token_ids must not be empty")]
    EmptyPromptLogprobTokenIds,
    #[error("prompt_logprob_token_ids must not contain duplicates")]
    DuplicatePromptLogprobTokenIds,
    #[error("prompt_logprob_start requires prompt_logprob_token_ids")]
    PromptLogprobStartWithoutTokenIds,
}

/// Validate logprobs count sampling parameters.
pub(super) fn validate_logprobs(
    logprobs: Option<i32>,
    prompt_logprobs: Option<i32>,
    logprob_token_ids: Option<&[u32]>,
    prompt_logprob_token_ids: Option<&[u32]>,
    prompt_logprob_start: Option<u32>,
    sampling_limits: SamplingLimits,
) -> Result<(), LogprobsError> {
    let vocab_size = sampling_limits.model_vocab_size;
    let max_logprobs =
        normalize_logprobs_count(sampling_limits.max_logprobs, vocab_size, "max_logprobs")?;

    validate_logprobs_count(logprobs, max_logprobs, vocab_size, "logprobs")?;
    validate_logprobs_count(prompt_logprobs, max_logprobs, vocab_size, "prompt_logprobs")?;
    validate_prompt_logprob_token_ids(
        prompt_logprob_token_ids,
        prompt_logprob_start,
        max_logprobs,
    )?;
    validate_logprob_token_ids(logprobs, logprob_token_ids)
}

fn validate_prompt_logprob_token_ids(
    token_ids: Option<&[u32]>,
    start: Option<u32>,
    max_logprobs: usize,
) -> Result<(), LogprobsError> {
    let Some(token_ids) = token_ids else {
        return match start {
            Some(_) => Err(LogprobsError::PromptLogprobStartWithoutTokenIds),
            None => Ok(()),
        };
    };
    if token_ids.is_empty() {
        return Err(LogprobsError::EmptyPromptLogprobTokenIds);
    }
    if token_ids.len() > max_logprobs {
        return Err(LogprobsError::TooManyCount {
            parameter: "prompt_logprob_token_ids",
            requested: token_ids.len(),
            max_allowed: max_logprobs,
        });
    }
    if token_ids.iter().collect::<HashSet<_>>().len() != token_ids.len() {
        return Err(LogprobsError::DuplicatePromptLogprobTokenIds);
    }
    Ok(())
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

pub(super) fn validate_logprob_token_ids(
    logprobs: Option<i32>,
    logprob_token_ids: Option<&[u32]>,
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
