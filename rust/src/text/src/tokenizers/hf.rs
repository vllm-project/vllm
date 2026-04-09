use std::path::Path;

use fastokens::Tokenizer as FastokensTokenizer;
use thiserror_ext::AsReport as _;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{info, warn};

use crate::Error;
use crate::error::Result;
use crate::tokenizers::Tokenizer;

/// Tokenizer from `tokenizer.json` in HuggingFace format.
///
/// This tries to load with `fastokens` first for better performance, then falls back to
/// HuggingFace's `tokenizers` if the former fails (e.g. due to unsupported tokenizer features or
/// file formats).
pub enum HuggingFaceTokenizer {
    Hf(Box<HfTokenizer>),
    Fastokens(Box<FastokensTokenizer>),
}

impl HuggingFaceTokenizer {
    /// Load from `tokenizer.json` with `fastokens`.
    pub fn new_fastokens(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with fastokens");
        let t = FastokensTokenizer::from_file(path).map_err(|error| {
            Error::Tokenizer(format!("failed to load tokenizer: {}", error.as_report()))
        })?;
        Ok(Self::Fastokens(Box::new(t)))
    }

    /// Load from `tokenizer.json` with Hugging Face `tokenizers`.
    pub fn new_hf(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with huggingface tokenizers");
        let t = HfTokenizer::from_file(path).map_err(|error| {
            Error::Tokenizer(format!("failed to load tokenizer: {}", error.as_report()))
        })?;
        Ok(Self::Hf(Box::new(t)))
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
        match self {
            Self::Hf(t) => {
                let encoding = t.encode(text, add_special_tokens).map_err(|error| {
                    Error::Tokenizer(format!("encoding failed: {}", error.as_report()))
                })?;
                Ok(encoding.get_ids().to_vec())
            }
            Self::Fastokens(t) => t
                .encode_with_special_tokens(text, add_special_tokens)
                .map_err(|error| {
                    Error::Tokenizer(format!("encoding failed: {}", error.as_report()))
                }),
        }
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self {
            Self::Hf(t) => t.decode(token_ids, skip_special_tokens).map_err(|error| {
                Error::Tokenizer(format!("decoding failed: {}", error.as_report()))
            }),
            Self::Fastokens(t) => t.decode(token_ids, skip_special_tokens).map_err(|error| {
                Error::Tokenizer(format!("decoding failed: {}", error.as_report()))
            }),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::Hf(t) => t.token_to_id(token),
            Self::Fastokens(t) => t.token_to_id(token),
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

    use super::{HuggingFaceTokenizer, Tokenizer};

    #[test]
    fn hf_constructor_resolves_added_token_ids() {
        let model = WordLevel::builder()
            .vocab(
                [("<unk>".to_string(), 0u32), ("hello".to_string(), 1u32)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("<unk>".to_string())
            .build()
            .expect("build wordlevel tokenizer");
        let mut tokenizer = HfTokenizer::new(model);
        tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).expect("save tokenizer json");

        let wrapper = HuggingFaceTokenizer::new_hf(&path).expect("load hf wrapper");
        assert_eq!(wrapper.token_to_id("<|im_end|>"), Some(2));
    }
}
