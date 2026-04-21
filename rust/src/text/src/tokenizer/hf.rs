use std::path::Path;
use std::sync::Arc;

use fastokens::Tokenizer as FastokensTokenizer;
use thiserror_ext::AsReport as _;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{info, warn};

use crate::Error;
use crate::error::Result;
use crate::tokenizer::Tokenizer;

enum Backend {
    Hf(Box<HfTokenizer>),
    Fastokens(Box<FastokensTokenizer>),
}

/// Tokenizer from `tokenizer.json` in HuggingFace format.
///
/// This tries to load with `fastokens` first for better performance, then falls back to
/// HuggingFace's `tokenizers` if the former fails (e.g. due to unsupported tokenizer features or
/// file formats).
pub struct HuggingFaceTokenizer {
    backend: Backend,
    special_token_ids: Arc<[u32]>,
}

impl HuggingFaceTokenizer {
    fn from_hf_backend(tokenizer: HfTokenizer) -> Self {
        let special_token_ids = {
            let mut ids: Vec<u32> = tokenizer
                .get_added_tokens_decoder()
                .iter()
                .filter(|(_id, token)| token.special)
                .map(|(id, _token)| *id)
                .collect();
            ids.sort_unstable();
            ids.dedup();
            Arc::from(ids)
        };
        Self {
            backend: Backend::Hf(Box::new(tokenizer)),
            special_token_ids,
        }
    }

    fn from_fastokens_backend(tokenizer: FastokensTokenizer) -> Self {
        let special_token_ids = {
            let mut ids: Vec<u32> = tokenizer
                .added_tokens()
                .into_iter()
                .flat_map(|added_tokens| added_tokens.iter())
                .filter(|token| token.special)
                .map(|token| token.id)
                .collect();
            ids.sort_unstable();
            ids.dedup();
            Arc::from(ids)
        };
        Self {
            backend: Backend::Fastokens(Box::new(tokenizer)),
            special_token_ids,
        }
    }

    /// Load from `tokenizer.json` with `fastokens`.
    pub fn new_fastokens(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with fastokens");
        let t = FastokensTokenizer::from_file(path).map_err(|error| {
            Error::Tokenizer(format!("failed to load tokenizer: {}", error.as_report()))
        })?;
        Ok(Self::from_fastokens_backend(t))
    }

    /// Load from `tokenizer.json` with Hugging Face `tokenizers`.
    pub fn new_hf(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with huggingface tokenizers");
        let t = HfTokenizer::from_file(path).map_err(|error| {
            Error::Tokenizer(format!("failed to load tokenizer: {}", error.as_report()))
        })?;
        Ok(Self::from_hf_backend(t))
    }

    /// Load from `tokenizer.json` via fastokens or HuggingFace tokenizers.
    pub fn new(path: &Path) -> Result<Self> {
        match Self::new_fastokens(path) {
            Ok(tokenizer) => Ok(tokenizer),
            Err(error) => {
                warn!(
                    path = %path.display(),
                    error = %error.as_report(),
                    "failed to load tokenizer with fastokens; falling back to HuggingFace tokenizers"
                );
                Self::new_hf(path)
            }
        }
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        match &self.backend {
            Backend::Hf(t) => {
                let encoding = t.encode(text, add_special_tokens).map_err(|error| {
                    Error::Tokenizer(format!("encoding failed: {}", error.as_report()))
                })?;
                Ok(encoding.get_ids().to_vec())
            }
            Backend::Fastokens(t) => t
                .encode_with_special_tokens(text, add_special_tokens)
                .map_err(|error| {
                    Error::Tokenizer(format!("encoding failed: {}", error.as_report()))
                }),
        }
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match &self.backend {
            Backend::Hf(t) => t.decode(token_ids, skip_special_tokens).map_err(|error| {
                Error::Tokenizer(format!("decoding failed: {}", error.as_report()))
            }),
            Backend::Fastokens(t) => t.decode(token_ids, skip_special_tokens).map_err(|error| {
                Error::Tokenizer(format!("decoding failed: {}", error.as_report()))
            }),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match &self.backend {
            Backend::Hf(t) => t.token_to_id(token),
            Backend::Fastokens(t) => t.token_to_id(token),
        }
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        self.special_token_ids.binary_search(&token_id).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use tokenizers::models::bpe::BPE;
    use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

    use super::{HuggingFaceTokenizer, Tokenizer};

    fn tiny_bpe_tokenizer() -> HfTokenizer {
        let vocab = [
            ("<unk>".to_string(), 0),
            ("h".to_string(), 1),
            ("e".to_string(), 2),
            ("l".to_string(), 3),
            ("o".to_string(), 4),
            ("he".to_string(), 5),
            ("ll".to_string(), 6),
            ("hell".to_string(), 7),
            ("hello".to_string(), 8),
        ];
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
            ("he".to_string(), "ll".to_string()),
            ("hell".to_string(), "o".to_string()),
        ];
        let model = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token("<unk>".to_string())
            .build()
            .expect("build bpe tokenizer");
        HfTokenizer::new(model)
    }

    #[test]
    fn hf_constructor_resolves_added_token_ids() {
        let mut tokenizer = tiny_bpe_tokenizer();
        tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).expect("save tokenizer json");

        let wrapper = HuggingFaceTokenizer::new_hf(&path).expect("load hf wrapper");
        let special_id = wrapper
            .token_to_id("<|im_end|>")
            .expect("resolve added special token id");
        assert!(wrapper.is_special_id(special_id));
    }

    #[test]
    fn new_fastokens_preserves_special_ids_from_fastokens_metadata() {
        let mut tokenizer = tiny_bpe_tokenizer();
        tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).expect("save tokenizer json");

        let wrapper = HuggingFaceTokenizer::new_fastokens(&path)
            .expect("load wrapper with fastokens backend");
        assert!(matches!(wrapper.backend, super::Backend::Fastokens(_)));
        let special_id = wrapper
            .token_to_id("<|im_end|>")
            .expect("resolve added special token id");
        assert!(wrapper.is_special_id(special_id));
    }
}
