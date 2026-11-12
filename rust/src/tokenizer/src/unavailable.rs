// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use crate::{Result, Tokenizer, TokenizerError};

/// The tokenizer of a server that loads none (`--skip-tokenizer-init`) and
/// serves token IDs only: every text operation fails with
/// [`TokenizerError::Unavailable`], and no token has a string.
pub struct UnavailableTokenizer {
    vocab_size: usize,
}

impl UnavailableTokenizer {
    /// `vocab_size` is the model vocabulary, which bounds token-ID inputs.
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

impl Tokenizer for UnavailableTokenizer {
    fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        Err(TokenizerError::Unavailable)
    }

    fn encode_ordinary(&self, _text: &str) -> Result<Vec<u32>> {
        Err(TokenizerError::Unavailable)
    }

    fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        Err(TokenizerError::Unavailable)
    }

    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }

    fn id_to_token(&self, _id: u32) -> Option<String> {
        None
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
