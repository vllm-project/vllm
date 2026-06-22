//! Python-compatible validation for logprobs sampling params.
//!
//! `All` is expanded only for bounds checks. The original request values are
//! passed through to engine-core.

use crate::backend::SamplingLimits;
use thiserror::Error;
use vllm_engine_core_client::protocol::LogprobsCount;

#[derive(Debug, Error)]
pub enum LogprobsError {
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
    TokenIdsMismatch {
        logprobs: LogprobsCount,
        num_token_ids: usize,
    },
}

/// Validate logprobs count sampling parameters.
pub(super) fn validate_logprobs(
    logprobs: Option<LogprobsCount>,
    prompt_logprobs: Option<LogprobsCount>,
    logprob_token_ids: Option<&[u32]>,
    sampling_limits: SamplingLimits,
) -> Result<(), LogprobsError> {
    let vocab_size = sampling_limits.model_vocab_size;
    let max_logprobs = sampling_limits.max_logprobs.expanded(vocab_size);

    validate_logprobs_count(logprobs, max_logprobs, vocab_size, "logprobs")?;
    validate_logprobs_count(prompt_logprobs, max_logprobs, vocab_size, "prompt_logprobs")?;
    validate_logprob_token_ids(logprobs, logprob_token_ids)
}

fn validate_logprobs_count(
    requested: Option<LogprobsCount>,
    max_logprobs: usize,
    vocab_size: usize,
    parameter: &'static str,
) -> Result<(), LogprobsError> {
    let Some(requested) = requested else {
        return Ok(());
    };

    let requested = requested.expanded(vocab_size);
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
    logprobs: Option<LogprobsCount>,
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
        && logprobs != LogprobsCount::Top(n as u32)
    {
        return Err(LogprobsError::TokenIdsMismatch {
            logprobs,
            num_token_ids: n,
        });
    }

    Ok(())
}
