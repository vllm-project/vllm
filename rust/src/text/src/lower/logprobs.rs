// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Python-compatible validation for logprobs sampling params.
//!
//! `-1` counts are expanded only for bounds checks and passed to engine-core
//! unchanged; `prompt_logprob_token_ids` rows are padded into an array.

use thiserror::Error;
use vllm_engine_core_client::protocol::tensor::WireNdArray;

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
    #[error(
        "prompt_logprob_token_ids must be a non-empty integer array of shape \
         [num_rows, num_ids]."
    )]
    EmptyPromptLogprobTokenIds,
    #[error(
        "prompt_logprob_token_ids contain out-of-vocab token ids (-1 pads a row). \
         Vocabulary size: {vocab_size}"
    )]
    PromptLogprobTokenIdsOutOfVocab { vocab_size: usize },
    #[error(
        "prompt_logprob_token_ids has {rows} rows, but the prompt has {expected} scored \
         rows (prompt_len - 1 - prompt_logprob_start)."
    )]
    PromptLogprobRowsMismatch { rows: usize, expected: usize },
    #[error("prompt_logprob_start requires prompt_logprobs or prompt_logprob_token_ids.")]
    PromptLogprobStartWithoutPromptLogprobs,
    #[error(
        "prompt_logprob_start must be less than prompt_len - 1 ({max}) so at least \
         one prompt row is scored."
    )]
    PromptLogprobStartOutOfRange { max: u32 },
}

/// Validate logprobs count sampling parameters.
pub(super) fn validate_logprobs(
    logprobs: Option<i32>,
    prompt_logprobs: Option<i32>,
    logprob_token_ids: Option<&[u32]>,
    prompt_logprob_token_ids: Option<&[Vec<i32>]>,
    prompt_logprob_start: Option<u32>,
    sampling_limits: SamplingLimits,
) -> Result<(), LogprobsError> {
    let vocab_size = sampling_limits.model_vocab_size;
    let max_logprobs =
        normalize_logprobs_count(sampling_limits.max_logprobs, vocab_size, "max_logprobs")?;

    validate_logprobs_count(logprobs, max_logprobs, vocab_size, "logprobs")?;
    validate_logprobs_count(prompt_logprobs, max_logprobs, vocab_size, "prompt_logprobs")?;
    validate_logprobs_count(
        prompt_logprob_token_ids.map(|rows| rows.iter().map(Vec::len).max().unwrap_or(0) as i32),
        max_logprobs,
        vocab_size,
        "prompt_logprob_token_ids",
    )?;
    if prompt_logprob_start.is_some()
        && prompt_logprobs.is_none()
        && prompt_logprob_token_ids.is_none()
    {
        return Err(LogprobsError::PromptLogprobStartWithoutPromptLogprobs);
    }
    validate_prompt_logprob_token_ids(prompt_logprob_token_ids)?;
    validate_logprob_token_ids(logprobs, logprob_token_ids)
}

fn validate_prompt_logprob_token_ids(rows: Option<&[Vec<i32>]>) -> Result<(), LogprobsError> {
    match rows {
        Some(rows) if rows.iter().all(Vec::is_empty) => {
            Err(LogprobsError::EmptyPromptLogprobTokenIds)
        }
        _ => Ok(()),
    }
}

/// Like Python, a nonzero start must leave at least one prompt row to score.
pub(super) fn validate_prompt_logprob_start(
    start: Option<u32>,
    prompt_len: u32,
) -> Result<(), LogprobsError> {
    let max = prompt_len.saturating_sub(1);
    match start {
        Some(start) if start > 0 && start >= max => {
            Err(LogprobsError::PromptLogprobStartOutOfRange { max })
        }
        _ => Ok(()),
    }
}

/// Check per-row candidates against the vocabulary and the prompt as Python
/// does, and pad them with `-1` into the engine's `[num_rows, num_ids]` array.
pub(super) fn lower_prompt_logprob_token_ids(
    rows: Option<Vec<Vec<i32>>>,
    start: Option<u32>,
    prompt_len: u32,
    vocab_size: usize,
) -> Result<Option<WireNdArray>, LogprobsError> {
    let Some(rows) = rows else {
        return Ok(None);
    };
    if rows.iter().flatten().any(|&id| id < -1 || i64::from(id) >= vocab_size as i64) {
        return Err(LogprobsError::PromptLogprobTokenIdsOutOfVocab { vocab_size });
    }
    let expected = (prompt_len as usize).saturating_sub(1 + start.unwrap_or(0) as usize);
    if rows.len() != expected {
        return Err(LogprobsError::PromptLogprobRowsMismatch {
            rows: rows.len(),
            expected,
        });
    }
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    let data = rows
        .iter()
        .flat_map(|row| row.iter().copied().chain(std::iter::repeat_n(-1, width - row.len())))
        .collect();
    Ok(Some(
        WireNdArray::from_i32(vec![rows.len(), width], data).expect("padded rows are rectangular"),
    ))
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
