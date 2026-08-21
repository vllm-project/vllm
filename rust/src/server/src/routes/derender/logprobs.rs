// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `token_id:N` placeholder resolution and chat→completion logprobs
//! conversion for the derender endpoints.
//!
//! Ports `_resolve_logprobs`, `_correct_decoded_token` and
//! `_convert_chat_logprobs_to_completion_logprobs` from
//! `vllm/renderers/online_derenderer.py`, plus `resolve_token_id_placeholder`
//! from `vllm/entrypoints/generate/base/serving.py`.

use std::collections::HashMap;

use thiserror_ext::AsReport as _;
use tracing::warn;
use vllm_text::tokenizer::DynTokenizer;

use super::detok::pieces_to_text;
use crate::error::{ApiError, server_error};
use crate::routes::openai::utils::logprobs::text_len;
use crate::routes::openai::utils::types::{
    ChatLogProbs, ChatLogProbsContent, LogProbs, TopLogProb,
};

/// Extract the token ID from a `token_id:N` placeholder string.
fn parse_token_id_placeholder(token: &str) -> Option<u32> {
    token.strip_prefix("token_id:")?.parse().ok()
}

/// Decode a `token_id:N` placeholder back to a token string and UTF-8 bytes.
///
/// Returns `(token, None)` unchanged if token is not a placeholder.
/// This is the inverse of `format_token_id` on the /generate side
/// when `return_tokens_as_token_ids` is active.
fn resolve_token_id_placeholder(
    token: &str,
    tokenizer: &DynTokenizer,
) -> Result<(String, Option<Vec<u8>>), ApiError> {
    let Some(token_id) = parse_token_id_placeholder(token) else {
        return Ok((token.to_string(), None));
    };
    let Some(piece) = tokenizer.id_to_token(token_id) else {
        warn!(
            token_id,
            "resolve_token_id_placeholder: token_id has no vocab entry; substituting empty string"
        );
        return Ok((String::new(), None));
    };
    let token_str = pieces_to_text(tokenizer, &[piece])?;
    Ok((token_str.clone(), Some(token_str.into_bytes())))
}

/// Use preceding tokens as context to fix U+FFFD from byte-fallback.
///
/// Mirrors `LogprobsProcessor._correct_decoded_token` in
/// v1/engine/logprobs.py.
fn correct_decoded_token(
    token_id: u32,
    context_token_ids: &[u32],
    tokenizer: &DynTokenizer,
) -> Result<String, ApiError> {
    let decode = |ids: &[u32]| {
        tokenizer.decode(ids, true).map_err(|error| {
            server_error!("derender logprobs correction failed: {}", error.as_report())
        })
    };
    let max_ctx = context_token_ids.len().min(4);

    for num_ctx in 1..=max_ctx {
        let context = &context_token_ids[context_token_ids.len() - num_ctx..];
        let mut full_ids = context.to_vec();
        full_ids.push(token_id);
        let full_decoded = decode(&full_ids)?;

        if full_decoded.ends_with('\u{FFFD}') {
            continue;
        }

        let mut clean_end = context.len();
        for j in (0..context.len()).rev() {
            if decode(&context[j..=j])?.ends_with('\u{FFFD}') {
                clean_end = j;
            } else {
                break;
            }
        }

        let clean_prefix = if clean_end > 0 {
            decode(&context[..clean_end])?
        } else {
            String::new()
        };

        if let Some(suffix) = full_decoded.strip_prefix(&clean_prefix) {
            return Ok(suffix.to_string());
        }

        let common_len = clean_prefix
            .chars()
            .zip(full_decoded.chars())
            .take_while(|(a, b)| a == b)
            .count();
        return Ok(full_decoded.chars().skip(common_len).collect());
    }

    Ok(String::new())
}

/// Resolve `token_id:N` placeholders in a [`ChatLogProbs`] object.
pub(super) fn resolve_logprobs(
    logprobs: &ChatLogProbs,
    tokenizer: &DynTokenizer,
) -> Result<ChatLogProbs, ApiError> {
    let Some(content) = &logprobs.content else {
        return Ok(logprobs.clone());
    };

    let mut context_token_ids: Vec<u32> = Vec::new();
    let mut resolved_content = Vec::with_capacity(content.len());

    for entry in content {
        let (mut token_str, mut token_bytes) =
            resolve_token_id_placeholder(&entry.token, tokenizer)?;
        let sampled_id = parse_token_id_placeholder(&entry.token);

        if token_str.ends_with('\u{FFFD}')
            && let Some(id) = sampled_id
        {
            token_str = correct_decoded_token(id, &context_token_ids, tokenizer)?;
            token_bytes = Some(token_str.clone().into_bytes());
        }

        let mut resolved_top = Vec::with_capacity(entry.top_logprobs.len());
        for top in &entry.top_logprobs {
            let (mut top_str, mut top_bytes) = resolve_token_id_placeholder(&top.token, tokenizer)?;
            let top_id = parse_token_id_placeholder(&top.token);
            if top_str.ends_with('\u{FFFD}')
                && let Some(id) = top_id
            {
                top_str = correct_decoded_token(id, &context_token_ids, tokenizer)?;
                top_bytes = Some(top_str.clone().into_bytes());
            }
            resolved_top.push(TopLogProb {
                token: top_str,
                logprob: top.logprob,
                bytes: top_bytes,
            });
        }

        resolved_content.push(ChatLogProbsContent {
            token: token_str,
            logprob: entry.logprob,
            bytes: token_bytes,
            top_logprobs: resolved_top,
        });

        if let Some(id) = sampled_id {
            context_token_ids.push(id);
        }
    }

    Ok(ChatLogProbs {
        content: Some(resolved_content),
    })
}

/// Convert [`ChatLogProbs`] (per-token objects) to [`LogProbs`] (parallel
/// flat lists) as required by the /v1/completions response schema.
pub(super) fn chat_logprobs_to_completion(logprobs: &ChatLogProbs) -> LogProbs {
    let mut tokens = Vec::new();
    let mut token_logprobs = Vec::new();
    let mut top_logprobs_list = Vec::new();
    let mut text_offset = Vec::new();

    let mut offset = 0;
    if let Some(content) = &logprobs.content {
        for entry in content {
            text_offset.push(offset);
            tokens.push(entry.token.clone());
            token_logprobs.push(Some(entry.logprob));
            top_logprobs_list.push(if entry.top_logprobs.is_empty() {
                None
            } else {
                Some(
                    entry
                        .top_logprobs
                        .iter()
                        .map(|top| (top.token.clone(), top.logprob))
                        .collect::<HashMap<String, f32>>(),
                )
            });
            offset += text_len(&entry.token);
        }
    }

    LogProbs {
        tokens,
        token_logprobs,
        top_logprobs: top_logprobs_list,
        text_offset,
    }
}
