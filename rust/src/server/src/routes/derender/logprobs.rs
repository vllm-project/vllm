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

use crate::error::{ApiError, server_error};
use crate::routes::inference::generate::GenerateLogprob;
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
///
/// Decodes the ID directly rather than round-tripping through
/// `id_to_token`/`token_to_id`, which is lossy for backends whose byte
/// pieces are not valid UTF-8 on their own.
fn resolve_token_id_placeholder(
    token: &str,
    tokenizer: &DynTokenizer,
) -> Result<(String, Option<Vec<u8>>), ApiError> {
    let Some(token_id) = parse_token_id_placeholder(token) else {
        return Ok((token.to_string(), None));
    };
    if tokenizer.id_to_token(token_id).is_none() {
        warn!(
            token_id,
            "resolve_token_id_placeholder: token_id has no vocab entry; substituting empty string"
        );
        return Ok((String::new(), None));
    }
    let token_str = tokenizer.decode(&[token_id], false).map_err(|error| {
        server_error!("derender logprobs resolution failed: {}", error.as_report())
    })?;
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

/// Resolve one wire token string: `token_id:N` placeholders decode through
/// the tokenizer, `decoded_token` strings pass through, and unknown IDs fall
/// back to the placeholder form.
fn wire_token_str(
    token_id: u32,
    decoded_token: Option<&str>,
    tokenizer: &DynTokenizer,
) -> Result<String, ApiError> {
    match decoded_token {
        Some(token) => Ok(resolve_token_id_placeholder(token, tokenizer)?.0),
        None if tokenizer.id_to_token(token_id).is_some() => {
            tokenizer.decode(&[token_id], false).map_err(|error| {
                server_error!("derender logprobs resolution failed: {}", error.as_report())
            })
        }
        None => Ok(format!("token_id:{token_id}")),
    }
}

/// Convert engine-wire prompt logprobs (one candidate map per prompt
/// position, as carried by `GenerateResponse.prompt_logprobs`) into the
/// completions [`LogProbs`] shape, mirroring the normal path's echoed and
/// prompt-only logprobs.
///
/// `prompt_token_ids` identifies the sampled token at each position; when it
/// is shorter than the position list, the remaining positions fall back to
/// their first candidate.
pub(super) fn prompt_logprobs_to_completion(
    prompt_logprobs: &[Option<HashMap<u32, GenerateLogprob>>],
    prompt_token_ids: &[u32],
    tokenizer: &DynTokenizer,
) -> Result<LogProbs, ApiError> {
    let mut tokens = Vec::with_capacity(prompt_logprobs.len());
    let mut token_logprobs = Vec::with_capacity(prompt_logprobs.len());
    let mut top_logprobs_list = Vec::with_capacity(prompt_logprobs.len());
    let mut text_offset = Vec::with_capacity(prompt_logprobs.len());

    let mut offset = 0;
    for (position, position_logprobs) in prompt_logprobs.iter().enumerate() {
        text_offset.push(offset);
        let sampled_id = prompt_token_ids.get(position).copied();
        match position_logprobs {
            // The first prompt position carries no logprobs.
            None => {
                let token = match sampled_id {
                    Some(id) => wire_token_str(id, None, tokenizer)?,
                    None => String::new(),
                };
                offset += text_len(&token);
                tokens.push(token);
                token_logprobs.push(None);
                top_logprobs_list.push(None);
            }
            Some(candidates) => {
                let sampled = sampled_id
                    .and_then(|id| candidates.get(&id).map(|entry| (id, entry)))
                    .or_else(|| candidates.iter().next().map(|(&id, entry)| (id, entry)));
                let (token, token_logprob) = match sampled {
                    Some((id, entry)) => (
                        wire_token_str(id, entry.decoded_token.as_deref(), tokenizer)?,
                        Some(entry.logprob),
                    ),
                    None => (String::new(), None),
                };
                offset += text_len(&token);
                tokens.push(token);
                token_logprobs.push(token_logprob);
                let mut top = HashMap::with_capacity(candidates.len());
                for (&id, entry) in candidates {
                    top.insert(
                        wire_token_str(id, entry.decoded_token.as_deref(), tokenizer)?,
                        entry.logprob,
                    );
                }
                top_logprobs_list.push((!top.is_empty()).then_some(top));
            }
        }
    }

    Ok(LogProbs {
        tokens,
        token_logprobs,
        top_logprobs: top_logprobs_list,
        text_offset,
    })
}

/// Shift every text offset by `prefix_len`, used when an echoed prompt is
/// prepended to the choice text.
pub(super) fn shift_text_offsets(logprobs: &mut LogProbs, prefix_len: u32) {
    for offset in &mut logprobs.text_offset {
        *offset = offset.saturating_add(prefix_len);
    }
}
