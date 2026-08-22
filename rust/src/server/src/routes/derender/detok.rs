// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Incremental detokenization for the streaming derender endpoints.
//!
//! Ports `vllm/tokenizers/detokenizer_utils.py::detokenize_incrementally`
//! onto the caller-supplied, JSON-serializable [`DerenderStreamState`].
//!
//! Window reconstruction decodes from the token IDs carried alongside the
//! wire pieces (`DerenderStreamState::prev_token_ids`), so byte-fallback
//! pieces whose bytes are not valid UTF-8 on their own still decode exactly
//! like one-shot `decode`. The Rust [`Tokenizer`] trait has no
//! `convert_ids_to_tokens` / `convert_tokens_to_string` pair, and some
//! backends' `id_to_token` is lossy UTF-8 (e.g. tiktoken stores base-token
//! byte pieces with replacement characters), so round-tripping pieces back
//! through `token_to_id` would drop or corrupt split multi-byte characters.
//!
//! States produced by the Python implementation carry only `prev_tokens`
//! pieces; those are mapped back through `token_to_id` as a best-effort
//! fallback.

use thiserror_ext::AsReport as _;
use vllm_text::tokenizer::DynTokenizer;

use super::types::DerenderStreamState;
use crate::error::{ApiError, server_error};

/// Sentinel written into `DerenderStreamState::prev_token_ids` for window
/// slots without a decodable token ID (only reachable via the foreign-state
/// fallback). Always out of vocabulary, so window reconstruction filters it.
const NO_TOKEN_ID: u32 = u32::MAX;

/// One slot in the carried decode window: the wire piece plus its token ID.
///
/// `id` is `None` only for foreign (Python-produced) state whose piece did
/// not map back through `token_to_id`.
struct WindowToken {
    piece: String,
    id: Option<u32>,
}

/// Convert one token id into the raw token piece carried in the decode
/// window.
///
/// Out-of-vocab ids and (when `skip_special_tokens`) special tokens map to
/// `""`, matching Python's `convert_ids_to_tokens` + `_replace_none_with_empty`
/// behavior.
fn token_piece(tokenizer: &DynTokenizer, token_id: u32, skip_special_tokens: bool) -> String {
    if token_id as usize >= tokenizer.vocab_size() {
        return String::new();
    }
    if skip_special_tokens && tokenizer.is_special_id(token_id) {
        return String::new();
    }
    tokenizer.id_to_token(token_id).unwrap_or_default()
}

/// Whether a window token ID participates in decoding: in vocabulary, and
/// not a special token when `skip_special_tokens` is set (matching the `""`
/// piece those tokens produce in [`token_piece`]).
fn is_decodable(tokenizer: &DynTokenizer, token_id: u32, skip_special_tokens: bool) -> bool {
    (token_id as usize) < tokenizer.vocab_size()
        && !(skip_special_tokens && tokenizer.is_special_id(token_id))
}

fn decode_ids(tokenizer: &DynTokenizer, token_ids: &[u32]) -> Result<String, ApiError> {
    tokenizer
        .decode(token_ids, false)
        .map_err(|error| server_error!("derender detokenization failed: {}", error.as_report()))
}

/// Decode the decodable IDs in `window[range]`, approximating Python's
/// `convert_tokens_to_string(pieces)` without a lossy piece round-trip.
fn decode_window(
    tokenizer: &DynTokenizer,
    window: &[WindowToken],
    range: std::ops::Range<usize>,
    skip_special_tokens: bool,
) -> Result<String, ApiError> {
    let ids: Vec<u32> = window[range]
        .iter()
        .filter_map(|token| token.id)
        .filter(|&id| is_decodable(tokenizer, id, skip_special_tokens))
        .collect();
    decode_ids(tokenizer, &ids)
}

/// Slice `text` to the characters after the first `prefix_chars`, mirroring
/// Python's `new_text[len(prefix_text):]` (which slices by code point).
fn drop_prefix_chars(text: &str, prefix_chars: usize) -> String {
    text.chars().skip(prefix_chars).collect()
}

/// Outcome of feeding one new token into the incremental decode window.
struct IncrementalStep {
    /// The window slot appended for this token.
    new_token: WindowToken,
    /// Newly visible text, or `""` when the tail is an unfinished multi-byte
    /// sequence.
    text: String,
    prefix_offset: usize,
    read_offset: usize,
}

/// Port of `detokenize_incrementally` for a single new token id.
///
/// The window is always a (possibly empty) list here, so this only implements
/// the non-first-iteration path from Python.
///
/// The offsets are necessary to defeat cleanup algorithms in the decode which
/// decide to add a space or not depending on the surrounding ids.
fn detokenize_incrementally(
    tokenizer: &DynTokenizer,
    new_token_id: u32,
    window: &[WindowToken],
    prefix_offset: usize,
    read_offset: usize,
    skip_special_tokens: bool,
) -> Result<IncrementalStep, ApiError> {
    let new_piece = token_piece(tokenizer, new_token_id, skip_special_tokens);

    // The prefix text is necessary only to defeat cleanup algorithms in
    // the decode which decide to add a space or not depending on the
    // surrounding ids.
    let prefix_text = decode_window(
        tokenizer,
        window,
        prefix_offset..read_offset,
        skip_special_tokens,
    )?;
    let mut window_ids: Vec<u32> = window[prefix_offset..]
        .iter()
        .filter_map(|token| token.id)
        .filter(|&id| is_decodable(tokenizer, id, skip_special_tokens))
        .collect();
    if is_decodable(tokenizer, new_token_id, skip_special_tokens) {
        window_ids.push(new_token_id);
    }
    let new_text = decode_ids(tokenizer, &window_ids)?;

    let prefix_chars = prefix_text.chars().count();
    let new_token = WindowToken {
        piece: new_piece,
        id: Some(new_token_id),
    };
    // A trailing U+FFFD char means it's a potential unfinished byte sequence
    // from byte fallback tokenization. If it's in the middle, it's probably a
    // real invalid id generated by the model.
    if new_text.chars().count() <= prefix_chars || new_text.ends_with('\u{FFFD}') {
        return Ok(IncrementalStep {
            new_token,
            text: String::new(),
            prefix_offset,
            read_offset,
        });
    }

    Ok(IncrementalStep {
        new_token,
        text: drop_prefix_chars(&new_text, prefix_chars),
        prefix_offset: read_offset,
        read_offset: window.len() + 1,
    })
}

/// Incrementally detokenize `delta_token_ids` from prior stream state.
///
/// Resumes decoding from the offsets carried in `state` rather than
/// replaying token history. `state.prev_tokens` holds the trailing decode
/// window (from `prefix_offset` onward) that `detokenize_incrementally`
/// still needs to reproduce any partially read multi-byte character
/// (tracked by `read_offset`). The delta tokens are fed straight onto it.
///
/// The window is bounded. `detokenize_incrementally` never reads before
/// `prefix_offset`, so after processing we trim `prev_tokens` to that
/// tail and rebase the offsets to it. State transport therefore stays
/// O(window) per chunk instead of re-sending the full token history.
///
/// Returns `(new_text, updated_state)` — the delta text for this chunk and
/// the state to pass to the next call.
pub(super) fn detokenize_delta(
    tokenizer: &DynTokenizer,
    delta_token_ids: &[u32],
    state: &DerenderStreamState,
    skip_special_tokens: bool,
) -> Result<(String, DerenderStreamState), ApiError> {
    // Prefer the carried token IDs. States produced by the Python
    // implementation have only `prev_tokens` pieces; map those back through
    // `token_to_id` as a best-effort fallback (lossy for backends whose
    // `id_to_token` is lossy UTF-8, hence the ID side channel).
    let mut window: Vec<WindowToken> = match &state.prev_token_ids {
        Some(ids) => state
            .prev_tokens
            .iter()
            .zip(ids)
            .map(|(piece, &id)| WindowToken {
                piece: piece.clone(),
                id: (id != NO_TOKEN_ID).then_some(id),
            })
            .collect(),
        None => state
            .prev_tokens
            .iter()
            .map(|piece| WindowToken {
                id: tokenizer.token_to_id(piece),
                piece: piece.clone(),
            })
            .collect(),
    };
    let mut prefix_offset = state.prefix_offset;
    let mut read_offset = state.read_offset;

    let mut text = String::new();
    for &token_id in delta_token_ids {
        let step = detokenize_incrementally(
            tokenizer,
            token_id,
            &window,
            prefix_offset,
            read_offset,
            skip_special_tokens,
        )?;
        window.push(step.new_token);
        text.push_str(&step.text);
        prefix_offset = step.prefix_offset;
        read_offset = step.read_offset;
    }

    // Trim to the tail still readable by detokenize_incrementally
    // (everything before prefix_offset is dead) and rebase the offsets so
    // the carried window stays bounded regardless of generation length.
    let trimmed = window.split_off(prefix_offset);
    let mut updated_state = state.clone();
    updated_state.prev_tokens = trimmed.iter().map(|token| token.piece.clone()).collect();
    updated_state.prev_token_ids =
        Some(trimmed.iter().map(|token| token.id.unwrap_or(NO_TOKEN_ID)).collect());
    updated_state.prefix_offset = 0;
    updated_state.read_offset = read_offset - prefix_offset;
    Ok((text, updated_state))
}
