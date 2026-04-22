use std::path::Path;
use std::sync::Arc;

use fastokens::Tokenizer as FastokensTokenizer;
use fastokens::decoders::Decoder as FastokensDecoder;
use thiserror_ext::AsReport as _;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{info, warn};

use crate::Error;
use crate::error::Result;
use crate::tokenizer::Tokenizer;
use crate::tokenizer::byte_level_decode::decode_byte_level;

enum Backend {
    Hf(Box<HfTokenizer>),
    Fastokens(Box<FastokensTokenizer>),
    /// Fastokens tokenizer whose decoder is pure GPT-2 byte-level, so we can
    /// bypass `Decoder::decode`'s `Vec<String>`/`join("")` assembly.
    FastokensByteLevel(Box<FastokensTokenizer>),
}

/// True if `dec` is effectively a single `ByteLevel` stage — one `ByteLevel`
/// leaf in a tree of `Sequence`s (fastokens represents `Fuse` as an empty
/// `Sequence`, which is a no-op for our purposes).
fn is_byte_level_only(dec: &FastokensDecoder) -> bool {
    fn count_byte_level(dec: &FastokensDecoder) -> usize {
        match dec {
            FastokensDecoder::ByteLevel(_) => 1,
            FastokensDecoder::Sequence(steps) => steps.iter().map(count_byte_level).sum(),
        }
    }
    count_byte_level(dec) == 1
}

fn decode_fastokens_byte_level(
    t: &FastokensTokenizer,
    token_ids: &[u32],
    skip_special_tokens: bool,
) -> Result<String> {
    let tokens: Vec<&str> = token_ids
        .iter()
        .filter(|&&id| !(skip_special_tokens && t.is_special_token(id)))
        .map(|&id| {
            t.id_to_token(id)
                .ok_or_else(|| Error::Tokenizer(format!("decoding failed: unknown token ID: {id}")))
        })
        .collect::<Result<_>>()?;
    Ok(decode_byte_level(tokens))
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
        let byte_level = tokenizer.decoder().is_some_and(is_byte_level_only);
        let backend = if byte_level {
            Backend::FastokensByteLevel(Box::new(tokenizer))
        } else {
            Backend::Fastokens(Box::new(tokenizer))
        };
        Self {
            backend,
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
            Backend::Fastokens(t) | Backend::FastokensByteLevel(t) => t
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
            Backend::FastokensByteLevel(t) => {
                decode_fastokens_byte_level(t, token_ids, skip_special_tokens)
            }
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match &self.backend {
            Backend::Hf(t) => t.token_to_id(token),
            Backend::Fastokens(t) | Backend::FastokensByteLevel(t) => t.token_to_id(token),
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
        assert!(matches!(
            wrapper.backend,
            super::Backend::Fastokens(_) | super::Backend::FastokensByteLevel(_),
        ));
        let special_id = wrapper
            .token_to_id("<|im_end|>")
            .expect("resolve added special token id");
        assert!(wrapper.is_special_id(special_id));
    }

    /// BPE tokenizer that round-trips through fastokens with a genuine
    /// `ByteLevel` decoder; vocab covers both GPT-2 (Ġ U+0120) and non-GPT-2
    /// (｜ U+FF5C) codepoints.
    fn tiny_byte_level_bpe() -> fastokens::Tokenizer {
        let raw = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<|endoftext|>", "single_word": false,
                 "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false,
                              "trim_offsets": true, "use_regex": true},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                        "trim_offsets": true, "use_regex": true},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": {
                    "<|endoftext|>": 0,
                    "H": 1, "e": 2, "l": 3, "o": 4, "w": 5, "r": 6, "d": 7,
                    "Ġ": 8, "!": 9,
                    "｜": 10
                },
                "merges": []
            }
        }"#;
        let value: serde_json::Value = serde_json::from_str(raw).expect("parse tokenizer json");
        fastokens::Tokenizer::from_json(value).expect("build fastokens tokenizer")
    }

    #[test]
    fn byte_level_detected_direct() {
        let t = tiny_byte_level_bpe();
        assert!(super::is_byte_level_only(t.decoder().expect("decoder")));
    }

    #[test]
    fn byte_level_detected_inside_sequence() {
        let raw = r#"{
            "type": "Sequence",
            "decoders": [
                {"type": "ByteLevel", "add_prefix_space": false,
                 "trim_offsets": true, "use_regex": true},
                {"type": "Fuse"}
            ]
        }"#;
        let config: fastokens::DecoderConfig =
            serde_json::from_str(raw).expect("parse decoder config");
        let dec =
            fastokens::decoders::Decoder::from_config(config).expect("build decoder from config");
        assert!(super::is_byte_level_only(&dec));
    }

    /// Fast path must produce byte-identical output to fastokens' own decode.
    #[test]
    fn fast_byte_level_matches_fastokens_decode() {
        let t = tiny_byte_level_bpe();
        let cases: &[&[u32]] = &[
            &[],
            &[1, 2, 3, 3, 4],                   // "Hello"
            &[1, 2, 3, 3, 4, 8, 5, 4, 6, 3, 7], // "Hello world"
            &[0, 1, 2, 3, 3, 4, 0, 9, 0],       // specials interleaved
            &[10, 1, 2, 3, 3, 4, 10],           // ｜Hello｜ (non-GPT2 chars)
        ];
        for ids in cases {
            for &skip in &[false, true] {
                let expected = t.decode(ids, skip).expect("fastokens decode");
                let got =
                    super::decode_fastokens_byte_level(&t, ids, skip).expect("fast-path decode");
                assert_eq!(got, expected, "ids={ids:?} skip={skip}");
            }
        }
    }

    #[test]
    fn fast_byte_level_errors_on_unknown_id() {
        let t = tiny_byte_level_bpe();
        let err = super::decode_fastokens_byte_level(&t, &[999], false)
            .expect_err("unknown id must error");
        assert!(format!("{err:?}").contains("999"));
    }
}
