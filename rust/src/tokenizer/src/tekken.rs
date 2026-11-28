// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::Path;

use base64::Engine as _;
use rustc_hash::FxHashMap;
use tekken::config::ModelData;
use tekken::{SpecialTokenPolicy, Tekkenizer, TokenInfo, TokenizerError};
use tracing::info;

use crate::{Result, Tokenizer};

/// Mistral Tekken tokenizer from a `tekken.json` file.
pub struct TekkenTokenizer {
    inner: Tekkenizer,
    /// Original bytes for vocabulary pieces that have no standalone UTF-8 form.
    invalid_utf8_pieces: FxHashMap<u32, Box<[u8]>>,
}

impl TekkenTokenizer {
    /// Load a Mistral Tekken tokenizer from a `tekken.json` file.
    pub fn new(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with Mistral Tekken");
        let load_error = |error: TokenizerError| {
            backend!(
                "failed to load tekken tokenizer from {}: {error}",
                path.display()
            )
        };

        let content =
            std::fs::read_to_string(path).map_err(|error| load_error(TokenizerError::Io(error)))?;
        let model_data: ModelData = serde_json::from_str(&content)
            .map_err(|error| load_error(TokenizerError::Json(error)))?;
        let invalid_utf8_pieces = Self::invalid_utf8_pieces(
            &model_data.vocab,
            model_data.config.default_vocab_size,
            model_data.config.default_num_special_tokens,
        )?;
        drop(model_data);
        drop(content);
        let inner = Tekkenizer::from_file(path).map_err(load_error)?;

        Ok(Self {
            inner,
            invalid_utf8_pieces,
        })
    }

    fn invalid_utf8_pieces(
        vocab: &[TokenInfo],
        vocab_size: usize,
        num_special_tokens: usize,
    ) -> Result<FxHashMap<u32, Box<[u8]>>> {
        let num_vocab_tokens = vocab_size
            .checked_sub(num_special_tokens)
            .ok_or_else(|| backend!("tekken special-token count exceeds vocabulary size"))?;
        let special_offset = u32::try_from(num_special_tokens)
            .map_err(|_| backend!("tekken special-token count exceeds u32"))?;
        let mut invalid_utf8_pieces = FxHashMap::default();

        for token in vocab.iter().take(num_vocab_tokens) {
            if token.token_str.is_some() {
                continue;
            }
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&token.token_bytes)
                .map_err(|error| backend!("failed to decode tekken vocabulary: {error}"))?;
            if std::str::from_utf8(&bytes).is_err() {
                let rank = u32::try_from(token.rank)
                    .map_err(|_| backend!("tekken token rank exceeds u32"))?;
                let token_id = special_offset
                    .checked_add(rank)
                    .ok_or_else(|| backend!("tekken token id exceeds u32"))?;
                invalid_utf8_pieces.insert(token_id, bytes.into_boxed_slice());
            }
        }

        Ok(invalid_utf8_pieces)
    }

    fn decode_lossy(&self, token_ids: &[u32], policy: SpecialTokenPolicy) -> Option<String> {
        let mut output = String::new();
        let mut bytes = Vec::with_capacity(token_ids.len().saturating_mul(4));
        let mut used_invalid_utf8_piece = false;

        for &token_id in token_ids {
            if self.inner.is_special_token(token_id) {
                output.push_str(&String::from_utf8_lossy(&bytes));
                bytes.clear();
                let piece = self.inner.id_to_byte_piece(token_id, policy).ok()?;
                output.push_str(std::str::from_utf8(&piece).ok()?);
            } else if let Some(piece) = self.invalid_utf8_pieces.get(&token_id) {
                bytes.extend_from_slice(piece);
                used_invalid_utf8_piece = true;
            } else {
                bytes.extend(self.inner.id_to_byte_piece(token_id, policy).ok()?);
            }
        }
        output.push_str(&String::from_utf8_lossy(&bytes));
        used_invalid_utf8_piece.then_some(output)
    }
}

impl Tokenizer for TekkenTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.inner
            .encode(text, add_special_tokens, false)
            .map_err(|error| backend!("encoding failed: {error}"))
    }

    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, false, false)
            .map_err(|error| backend!("encoding failed: {error}"))
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let policy = if skip_special_tokens {
            SpecialTokenPolicy::Ignore
        } else {
            SpecialTokenPolicy::Keep
        };
        match self.inner.decode(token_ids, policy) {
            Ok(text) => Ok(text),
            Err(error @ TokenizerError::Tokenizers(_)) => self
                .decode_lossy(token_ids, policy)
                .ok_or_else(|| backend!("decoding failed: {error}")),
            Err(error) => Err(backend!("decoding failed: {error}")),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        // tekken-rs exposes `get_control_token` for special tokens. Try that first,
        // then fall back to encoding.
        self.inner.get_control_token(token).ok().or_else(|| {
            let ids = self.inner.encode(token, false, false).ok()?;
            if ids.len() == 1 { Some(ids[0]) } else { None }
        })
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match self.inner.id_to_piece(id) {
            Ok(token) => Some(token),
            Err(TokenizerError::Tokenizers(_)) => self
                .invalid_utf8_pieces
                .get(&id)
                .map(|bytes| String::from_utf8_lossy(bytes).into_owned()),
            Err(_) => None,
        }
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        self.inner.is_special_token(token_id)
    }
}

#[cfg(test)]
mod tests {
    use tekken::SpecialTokenInfo;
    use tekken::config::TokenizerVersion;

    use super::*;

    fn test_tokenizer() -> TekkenTokenizer {
        let vocab: Vec<TokenInfo> = (0_u8..=255)
            .map(|byte| TokenInfo {
                rank: byte as usize,
                token_bytes: base64::engine::general_purpose::STANDARD.encode([byte]),
                token_str: None,
            })
            .collect();
        let special_tokens = vec![SpecialTokenInfo {
            rank: 0,
            token_str: "<control>".to_string(),
            is_control: true,
        }];
        let invalid_utf8_pieces =
            TekkenTokenizer::invalid_utf8_pieces(&vocab, 257, 1).expect("build raw byte pieces");
        let inner = Tekkenizer::new(
            vocab,
            &special_tokens,
            r"(?s).",
            257,
            1,
            TokenizerVersion::V3,
            None,
        )
        .expect("build Tekken tokenizer");
        TekkenTokenizer {
            inner,
            invalid_utf8_pieces,
        }
    }

    #[test]
    fn ordinary_matches_tekkens_empty_special_encoding() {
        let tokenizer = test_tokenizer();
        let text = "user <control> text";
        let control_id = tokenizer.token_to_id("<control>").unwrap();
        let ordinary_ids = tokenizer.encode_ordinary(text).unwrap();

        assert_eq!(control_id, 0);
        assert_eq!(ordinary_ids, tokenizer.encode(text, false).unwrap());
        assert!(!ordinary_ids.contains(&control_id));
        assert_eq!(tokenizer.decode(&ordinary_ids, false).unwrap(), text);
    }

    #[test]
    fn incomplete_utf8_is_held_until_the_character_is_complete() {
        let tokenizer = test_tokenizer();
        let ids = tokenizer.encode("测", false).unwrap();

        assert_eq!(ids.len(), 3);
        assert_eq!(tokenizer.id_to_token(ids[0]).as_deref(), Some("\u{FFFD}"));
        assert_eq!(tokenizer.decode_token(ids[0]).unwrap(), "\u{FFFD}");
        assert_eq!(tokenizer.decode(&ids[..1], false).unwrap(), "\u{FFFD}");
        assert_eq!(tokenizer.decode(&ids[..2], false).unwrap(), "\u{FFFD}");
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), "测");

        let mut decoder = tokenizer.create_decode_stream(&[], false, 0);
        assert_eq!(decoder.push_token(ids[0]).unwrap(), 0);
        assert_eq!(decoder.push_token(ids[1]).unwrap(), 0);
        assert_eq!(decoder.push_token(ids[2]).unwrap(), "测".len());
        assert_eq!(decoder.output(), "测");
    }

    #[test]
    fn lossy_decode_does_not_join_bytes_across_special_tokens() {
        let tokenizer = test_tokenizer();
        let ids = tokenizer.encode("测", false).unwrap();
        let control_id = tokenizer.token_to_id("<control>").unwrap();
        let split = [ids[0], ids[1], control_id, ids[2]];

        assert_eq!(tokenizer.decode(&split, true).unwrap(), "\u{FFFD}\u{FFFD}");
        assert_eq!(
            tokenizer.decode(&split, false).unwrap(),
            "\u{FFFD}<control>\u{FFFD}"
        );
    }
}
