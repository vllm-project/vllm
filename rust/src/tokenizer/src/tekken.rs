// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::Path;

use tekken::Tekkenizer;
use tracing::info;

use crate::{Result, Tokenizer};

/// Mistral Tekken tokenizer from a `tekken.json` file.
pub struct TekkenTokenizer {
    inner: Tekkenizer,
    /// Control tokens, such as `[INST]`, with their IDs.
    control_tokens: Vec<(String, u32)>,
}

impl TekkenTokenizer {
    /// Load a Mistral Tekken tokenizer from a `tekken.json` file.
    pub fn new(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with Mistral Tekken");

        let inner = Tekkenizer::from_file(path).map_err(|error| {
            backend!(
                "failed to load tekken tokenizer from {}: {error}",
                path.display()
            )
        })?;
        Ok(Self::from_tekkenizer(inner))
    }

    fn from_tekkenizer(inner: Tekkenizer) -> Self {
        // Tekken's special tokens are its control tokens; they take the IDs
        // below `num_special_tokens`.
        let control_tokens =
            inner.vocab()[..inner.num_special_tokens()].iter().cloned().zip(0..).collect();
        Self {
            inner,
            control_tokens,
        }
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl TekkenTokenizer {
    /// Build a byte-level tokenizer for tests. `control_tokens` take the IDs
    /// from 0 and each byte follows them.
    pub fn for_test(control_tokens: &[&str]) -> Self {
        use base64::Engine as _;
        use tekken::config::TokenizerVersion;
        use tekken::{SpecialTokenInfo, TokenInfo};

        let vocab = (0_u8..=255)
            .map(|byte| TokenInfo {
                rank: byte as usize,
                token_bytes: base64::engine::general_purpose::STANDARD.encode([byte]),
                token_str: None,
            })
            .collect();
        let special_tokens: Vec<_> = control_tokens
            .iter()
            .enumerate()
            .map(|(rank, token)| SpecialTokenInfo {
                rank,
                token_str: token.to_string(),
                is_control: true,
            })
            .collect();
        let inner = Tekkenizer::new(
            vocab,
            &special_tokens,
            r"(?s).",
            256 + special_tokens.len(),
            special_tokens.len(),
            TokenizerVersion::V3,
            None,
        )
        .expect("build Tekken tokenizer");
        Self::from_tekkenizer(inner)
    }
}

impl Tokenizer for TekkenTokenizer {
    /// Control-token spellings in `text` stay plain text, as in Python's
    /// `MistralTokenizer.encode`.
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
            tekken::SpecialTokenPolicy::Ignore
        } else {
            tekken::SpecialTokenPolicy::Keep
        };
        self.inner
            .decode(token_ids, policy)
            .map_err(|error| backend!("decoding failed: {error}"))
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
        self.inner.id_to_piece(id).ok()
    }

    /// The control tokens, which [`encode`](Self::encode) does not match.
    fn added_vocab(&self) -> &[(String, u32)] {
        &self.control_tokens
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
    use super::*;

    #[test]
    fn ordinary_matches_tekkens_empty_special_encoding() {
        let tokenizer = TekkenTokenizer::for_test(&["<control>"]);
        let text = "user <control> text";
        let control_id = tokenizer.token_to_id("<control>").unwrap();
        let ordinary_ids = tokenizer.encode_ordinary(text).unwrap();

        assert_eq!(control_id, 0);
        assert_eq!(ordinary_ids, tokenizer.encode(text, false).unwrap());
        assert!(!ordinary_ids.contains(&control_id));
        assert_eq!(tokenizer.decode(&ordinary_ids, false).unwrap(), text);
    }
}
