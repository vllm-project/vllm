// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Port of `xgr.TokenizerInfo.from_huggingface` (Hugging Face fast tokenizer
//! path) and `XgrammarBackend._filter_special_tokens`.
//!
//! The tokenizer itself stays with the caller: these functions take the
//! values Python reads from it.
//!
//! TODO: the tiktoken, SentencePiece, Mistral (`tekken`), and PLaMo 3
//! (`init_xgrammar`) tokenizer paths are not ported; callers build those
//! `TokenizerInfo`s with [`TokenizerInfo::new`] directly.

use rustc_hash::FxHashSet;
use vllm_xgrammar::{TokenizerInfo, VocabType};

use super::Error;

/// What `from_huggingface` reads from a `PreTrainedTokenizerFast`.
#[derive(Clone, Debug, Default)]
pub struct HfTokenizerParts {
    /// `tokenizer.get_vocab().items()`, in dict iteration order.
    pub vocab: Vec<(String, i32)>,
    /// `tokenizer.backend_tokenizer.to_str()`.
    pub backend_str: String,
    /// `tokenizer.eos_token_id`.
    pub eos_token_id: Option<i32>,
    /// `tokenizer.convert_ids_to_tokens(tokenizer.encode("a",
    /// add_special_tokens=False))[0]`, used by
    /// `_detect_add_prefix_space_by_encoding`; `None` if encoding fails or
    /// yields no token.
    pub first_token_of_a: Option<String>,
    /// Ids of `tokenizer.added_tokens_decoder` entries with `special=True`.
    pub special_token_ids: Vec<i32>,
}

/// The tokenizer infos `XgrammarBackend.__post_init__` builds for a
/// Hugging Face fast tokenizer.
pub struct BackendTokenizerInfos {
    /// For JSON and the other non-structural-tag grammars; has no special
    /// tokens in its vocabulary.
    pub filtered: TokenizerInfo,
    /// For structural tags and Lark grammars, which may accept special
    /// tokens.
    pub full_vocab: TokenizerInfo,
}

impl HfTokenizerParts {
    /// `xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)`.
    pub fn tokenizer_info(&self, vocab_size: Option<usize>) -> Result<TokenizerInfo, Error> {
        let max_id = self.vocab.iter().map(|&(_, id)| id).max().ok_or_else(|| Error {
            message: "max() arg is an empty sequence".to_owned(),
        })?;
        // Some tokenizer don't have token id 0 or 1 or 2. So the max_id could be larger than the
        // number of tokens.
        let tokenizer_vocab_size = self.vocab.len().max(max_id as usize + 1);
        // `vocab_size or tokenizer_vocab_size`
        let vocab_size = vocab_size.filter(|&v| v != 0).unwrap_or(tokenizer_vocab_size);

        // maintain tokenizer's indexing
        let mut encoded_vocab = vec![""; vocab_size];
        for (token, id) in &self.vocab {
            if (*id as usize) < vocab_size {
                encoded_vocab[*id as usize] = token;
            }
        }

        // TODO: Python warns when no stop token is known and lets xgrammar
        // auto-detect it.
        let stop_token_ids = self.eos_token_id.map(|id| vec![id]);

        let metadata = TokenizerInfo::detect_metadata_from_hf(&self.backend_str)?;
        let (mut vocab_type, mut add_prefix_space) = parse_metadata(&metadata)?;

        // The serialized backend tokenizer may not describe the vocabulary faithfully:
        // transformers v5 sometimes reconstructs a tokenizer from the SentencePiece model
        // file, dropping or replacing the decoder that the detection above relies on. The
        // vocabulary is decisive evidence for its own encoding, so when the two disagree,
        // trust the vocabulary and re-derive add_prefix_space from the tokenizer's observed
        // encoding behavior.
        if let Some(from_vocab) = detect_vocab_type_from_vocab(&self.vocab)
            && from_vocab != vocab_type
        {
            vocab_type = from_vocab;
            add_prefix_space = self.detect_add_prefix_space_by_encoding(from_vocab);
        }

        Ok(TokenizerInfo::new(
            &encoded_vocab,
            vocab_type,
            Some(vocab_size),
            stop_token_ids.as_deref(),
            add_prefix_space,
        )?)
    }

    /// `XgrammarBackend.__post_init__` for a Hugging Face fast tokenizer:
    /// the full tokenizer info and the one with special tokens filtered out.
    pub fn backend_tokenizer_infos(
        &self,
        vocab_size: usize,
    ) -> Result<BackendTokenizerInfos, Error> {
        let full_vocab = self.tokenizer_info(Some(vocab_size))?;
        let filtered = self.filter_special_tokens(&full_vocab)?;
        Ok(BackendTokenizerInfos {
            filtered,
            full_vocab,
        })
    }

    /// Filter ALL special tokens from tokenizer_info
    fn filter_special_tokens(
        &self,
        tokenizer_info: &TokenizerInfo,
    ) -> Result<TokenizerInfo, Error> {
        let special: FxHashSet<i32> = self.special_token_ids.iter().copied().collect();
        let mut encoded_vocab = vec![""; tokenizer_info.vocab_size()];
        for (token, id) in &self.vocab {
            if (*id as usize) < encoded_vocab.len() && !special.contains(id) {
                encoded_vocab[*id as usize] = token;
            }
        }
        Ok(TokenizerInfo::from_vocab_and_metadata(
            &encoded_vocab,
            &tokenizer_info.dump_metadata()?,
        )?)
    }

    /// Detect whether the tokenizer prepends a space by encoding a one-character text and
    /// inspecting the first produced token. This reflects the tokenizer's actual runtime
    /// behavior, which is what grammar matching must align with.
    fn detect_add_prefix_space_by_encoding(&self, vocab_type: VocabType) -> bool {
        let Some(token) = self.first_token_of_a.as_deref().filter(|t| !t.is_empty()) else {
            return false;
        };
        let prefix = if vocab_type == VocabType::ByteFallback {
            "▁"
        } else {
            "Ġ"
        };
        token.starts_with(prefix)
    }
}

/// Parse `{"vocab_type": int, "add_prefix_space": bool}`.
fn parse_metadata(metadata: &str) -> Result<(VocabType, bool), Error> {
    let invalid = || Error {
        message: format!("invalid tokenizer metadata: {metadata}"),
    };
    let value: serde_json::Value = serde_json::from_str(metadata).map_err(|_| invalid())?;
    let vocab_type = value["vocab_type"]
        .as_i64()
        .and_then(|v| VocabType::from_i32(v as i32))
        .ok_or_else(invalid)?;
    let add_prefix_space = value["add_prefix_space"].as_bool().ok_or_else(invalid)?;
    Ok((vocab_type, add_prefix_space))
}

/// The 256 characters used by the byte-to-unicode conversion of byte-level BPE tokenizers.
fn is_byte_level_char(c: char) -> bool {
    let c = c as u32;
    // Kept bytes map to themselves; the other 68 bytes map to 256..256+68.
    (0x21..=0x7E).contains(&c)
        || (0xA1..=0xAC).contains(&c)
        || (0xAE..=0xFF).contains(&c)
        || (256..256 + 68).contains(&c)
}

/// Detect the vocabulary type from the vocabulary itself. Returns None when the
/// vocabulary clearly matches neither byte encoding.
///
/// A byte-fallback vocabulary contains the 256 byte pieces "<0x00>".."<0xFF>". A byte-level
/// vocabulary spells every token with the 256-character byte-to-unicode alphabet and
/// contains most of that alphabet as single-character tokens. Both are properties of the
/// vocabulary itself, so they stay valid when the serialized backend tokenizer does not
/// describe the vocabulary faithfully.
fn detect_vocab_type_from_vocab(vocab: &[(String, i32)]) -> Option<VocabType> {
    let tokens: FxHashSet<&str> = vocab.iter().map(|(t, _)| t.as_str()).collect();
    let num_byte_pieces =
        (0..256).filter(|b| tokens.contains(format!("<0x{b:02X}>").as_str())).count();
    if num_byte_pieces >= 128 {
        return Some(VocabType::ByteFallback);
    }

    let num_single_chars = tokens
        .iter()
        .filter(|t| {
            let mut chars = t.chars();
            chars.next().is_some_and(is_byte_level_char) && chars.next().is_none()
        })
        .count();
    if num_single_chars < 128 {
        return None;
    }

    // Added tokens (e.g. chat-template markers) may contain characters outside the
    // alphabet, so tolerate a small fraction of such tokens.
    let num_charset_tokens = tokens.iter().filter(|t| t.chars().all(is_byte_level_char)).count();
    if num_charset_tokens as f64 >= 0.99 * tokens.len() as f64 {
        return Some(VocabType::ByteLevel);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_charset_has_256_chars() {
        let n = (0..=0x200u32)
            .filter_map(char::from_u32)
            .filter(|&c| is_byte_level_char(c))
            .count();
        assert_eq!(n, 256);
    }

    #[test]
    fn vocab_type_detection() {
        let byte_fallback: Vec<_> = (0..256).map(|b| (format!("<0x{b:02X}>"), b)).collect();
        assert_eq!(
            detect_vocab_type_from_vocab(&byte_fallback),
            Some(VocabType::ByteFallback)
        );
        let byte_level: Vec<_> = (0x21u32..0x7F)
            .chain(0xA1..0xAD)
            .chain(0xAE..0x100)
            .chain(256..324)
            .map(|c| (char::from_u32(c).unwrap().to_string(), c as i32))
            .collect();
        assert_eq!(
            detect_vocab_type_from_vocab(&byte_level),
            Some(VocabType::ByteLevel)
        );
        let raw: Vec<_> = (0..300).map(|i| (format!("tok{i}"), i)).collect();
        assert_eq!(detect_vocab_type_from_vocab(&raw), None);
    }
}
