// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashSet;
use std::result::Result;

use thiserror::Error;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

use crate::SamplingLimits;

const MAX_STOP_TOKEN_IDS_ENV: &str = "VLLM_MAX_STOP_TOKEN_IDS";
const DEFAULT_MAX_STOP_TOKEN_IDS: usize = 128;
const MAX_NUM_BAD_WORDS_ENV: &str = "VLLM_MAX_NUM_BAD_WORDS";
const DEFAULT_MAX_NUM_BAD_WORDS: usize = 128;
const MAX_BAD_WORDS_TOTAL_TOKENS_ENV: &str = "VLLM_MAX_BAD_WORDS_TOTAL_TOKENS";
const DEFAULT_MAX_BAD_WORDS_TOTAL_TOKENS: usize = 1024;

#[derive(Debug, Error)]
pub enum TokenIdsError {
    #[error("allowed_token_ids should not be empty")]
    EmptyAllowedTokenIds,
    #[error("bad_words_token_ids must not contain empty sequences")]
    EmptyBadWordSequence,
    #[error("bad_words cannot contain an empty string.")]
    EmptyBadWord,
    #[error("bad_words entries must tokenize to at least one token.")]
    EmptyBadWordTokenization,
    #[error("Too many bad words: {count}. The max number is {max}.")]
    TooManyBadWords {
        parameter: Option<&'static str>,
        count: usize,
        max: usize,
    },
    #[error("Too many bad words after tokenization: {count}. The max number is {max}.")]
    TooManyTokenizedBadWords { count: usize, max: usize },
    #[error("Too many total bad word tokens: {count}. The max is {max}.")]
    TooManyBadWordTokens {
        parameter: Option<&'static str>,
        count: usize,
        max: usize,
    },
    #[error("Too many stop_token_ids: {count}. The max number is {max}.")]
    TooManyStopTokenIds { count: usize, max: usize },
    #[error(
        "token_id(s) {token_ids:?} in {parameter} are out of vocabulary. \
         Vocabulary size: {vocab_size}"
    )]
    OutOfVocab {
        parameter: &'static str,
        token_ids: Vec<u32>,
        vocab_size: usize,
    },
}

impl TokenIdsError {
    /// The canonical request parameter name for this validation error.
    pub fn param(&self) -> Option<&'static str> {
        match self {
            Self::EmptyAllowedTokenIds => Some("allowed_token_ids"),
            Self::EmptyBadWordSequence => Some("bad_words_token_ids"),
            Self::EmptyBadWord
            | Self::EmptyBadWordTokenization
            | Self::TooManyTokenizedBadWords { .. } => Some("bad_words"),
            Self::TooManyBadWords { parameter, .. }
            | Self::TooManyBadWordTokens { parameter, .. } => *parameter,
            Self::TooManyStopTokenIds { .. } => Some("stop_token_ids"),
            Self::OutOfVocab { parameter, .. } => Some(parameter),
        }
    }
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

fn env_limit(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

/// Normalize the explicit stop list before sending it to Python engine-core.
pub(super) fn normalize_stop_token_ids(
    mut stop_token_ids: Vec<u32>,
) -> Result<Vec<u32>, TokenIdsError> {
    let mut seen = HashSet::new();
    stop_token_ids.retain(|token_id| seen.insert(*token_id));
    let count = stop_token_ids.len();
    let max = env_limit(MAX_STOP_TOKEN_IDS_ENV, DEFAULT_MAX_STOP_TOKEN_IDS);
    if count > max {
        return Err(TokenIdsError::TooManyStopTokenIds { count, max });
    }
    Ok(stop_token_ids)
}

/// The per-request cap on bad-word sequences.
pub(super) fn max_num_bad_words() -> usize {
    env_limit(MAX_NUM_BAD_WORDS_ENV, DEFAULT_MAX_NUM_BAD_WORDS)
}

/// Mirrors Model Runner V2's per-request bad-words buffer limits:
/// <https://github.com/vllm-project/vllm/blob/a956a8126e6bd184cb0a6b8f7318fe4717675c2f/vllm/v1/worker/gpu/sample/bad_words.py#L40-L66>
fn validate_bad_words_limits(
    bad_words_token_ids: &[Vec<u32>],
    parameter: Option<&'static str>,
) -> Result<(), TokenIdsError> {
    let count = bad_words_token_ids.len();
    let max = max_num_bad_words();
    if count > max {
        return Err(TokenIdsError::TooManyBadWords {
            parameter,
            count,
            max,
        });
    }

    let count = bad_words_token_ids.iter().map(Vec::len).sum();
    let max = env_limit(
        MAX_BAD_WORDS_TOTAL_TOKENS_ENV,
        DEFAULT_MAX_BAD_WORDS_TOTAL_TOKENS,
    );
    if count > max {
        return Err(TokenIdsError::TooManyBadWordTokens {
            parameter,
            count,
            max,
        });
    }

    Ok(())
}

/// Validate that pre-tokenized prompt IDs are within the engine-visible prompt
/// vocabulary range.
///
/// Only multimodal placeholders may exceed the model vocabulary: the engine
/// replaces them before the embedding lookup. A prompt without multimodal
/// inputs is embedded in full, so tokenizer-only IDs (e.g. Kimi's reserved
/// tokens) are rejected. Python widens the bound for every request to a
/// multimodal model (`BaseRenderer.validate_token_ids`).
pub(crate) fn validate_prompt_token_ids(
    prompt_token_ids: &[u32],
    has_multimodal_inputs: bool,
    limits: &SamplingLimits,
) -> Result<(), TokenIdsError> {
    let vocab_size = if has_multimodal_inputs {
        limits.tokenizer_vocab_size.max(limits.model_vocab_size)
    } else {
        limits.model_vocab_size
    };
    validate_param("prompt", prompt_token_ids.iter().copied(), vocab_size)
}

/// Validate that token IDs in text sampling parameters are within their
/// parameter-specific vocabulary ranges.
///
/// `raw_bad_words_count` is the prefix length of `params.bad_words_token_ids`
/// that came from pre-tokenized `bad_words_token_ids` (the remainder came from
/// tokenizing string `bad_words`).
pub(crate) fn validate_vocab_range(
    params: &EngineCoreSamplingParams,
    limits: &SamplingLimits,
    raw_bad_words_count: usize,
) -> Result<(), TokenIdsError> {
    validate_param(
        "stop_token_ids",
        params.stop_token_ids.iter().chain(&params.stop_token_ids_set).copied(),
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
        let (raw, tokenized) = bad_words_token_ids.split_at(raw_bad_words_count);
        if raw.iter().any(Vec::is_empty) {
            return Err(TokenIdsError::EmptyBadWordSequence);
        }
        let limit_param = match (!raw.is_empty(), !tokenized.is_empty()) {
            (true, false) => Some("bad_words_token_ids"),
            (false, true) => Some("bad_words"),
            _ => None,
        };
        validate_bad_words_limits(bad_words_token_ids, limit_param)?;
        validate_param(
            "bad_words_token_ids",
            raw.iter().flatten().copied(),
            limits.tokenizer_vocab_size,
        )?;
        validate_param(
            "bad_words",
            tokenized.iter().flatten().copied(),
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
