use std::path::Path;

use tekken::Tekkenizer;
use tracing::info;

use crate::{Result, Tokenizer};

/// Mistral Tekken tokenizer from a `tekken.json` file.
pub struct TekkenTokenizer {
    inner: Tekkenizer,
}

impl TekkenTokenizer {
    /// Load a Mistral Tekken tokenizer from a `tekken.json` file.
    pub fn new(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with Mistral Tekken");

        let inner = Tekkenizer::from_file(path).map_err(|error| {
            tokenizer_error!(
                "failed to load tekken tokenizer from {}: {error}",
                path.display()
            )
        })?;
        Ok(Self { inner })
    }
}

impl Tokenizer for TekkenTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.inner
            .encode(text, add_special_tokens, false)
            .map_err(|error| tokenizer_error!("encoding failed: {error}"))
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let policy = if skip_special_tokens {
            tekken::SpecialTokenPolicy::Ignore
        } else {
            tekken::SpecialTokenPolicy::Keep
        };
        self.inner
            .decode(token_ids, policy)
            .map_err(|error| tokenizer_error!("decoding failed: {error}"))
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

    fn is_special_id(&self, token_id: u32) -> bool {
        self.inner.is_special_token(token_id)
    }
}
