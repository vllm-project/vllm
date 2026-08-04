// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::Path;
use std::sync::{Arc, LazyLock};

use fastokens::Tokenizer as FastokensTokenizer;
use fastokens::decoders::Decoder as FastokensDecoder;
use thiserror_ext::AsReport as _;
use tokenizers::{
    AddedVocabulary, Model as _, OffsetType, PreTokenizer as _, Tokenizer as HfTokenizer,
};
use tracing::{info, warn};

use crate::byte_level_decode::decode_byte_level;
use crate::hf::added_tokens::load_tokenizer_json_with_extra_tokens;
use crate::{Result, Tokenizer};

mod added_tokens;

static EMPTY_HF_ADDED_VOCABULARY: LazyLock<AddedVocabulary> = LazyLock::new(AddedVocabulary::new);

enum Backend {
    Hf(Box<HfTokenizer>),
    Fastokens(Box<FastokensTokenizer>),
    /// Fastokens tokenizer whose decoder is pure GPT-2 byte-level, so we can
    /// bypass `Decoder::decode`'s `Vec<String>`/`join("")` assembly.
    FastokensByteLevel(Box<FastokensTokenizer>),
}

/// True if `dec` is effectively a single `ByteLevel` stage, optionally wrapped
/// in `Sequence`s or followed by `Fuse`.
fn is_byte_level_only(dec: &FastokensDecoder) -> bool {
    fn count_byte_level(dec: &FastokensDecoder) -> Option<usize> {
        match dec {
            FastokensDecoder::ByteLevel(_) => Some(1),
            FastokensDecoder::Fuse => Some(0),
            FastokensDecoder::Sequence(steps) => {
                steps.iter().try_fold(0, |count, step| Some(count + count_byte_level(step)?))
            }
            _ => None,
        }
    }
    count_byte_level(dec) == Some(1)
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
                .ok_or_else(|| tokenizer_error!("decoding failed: unknown token ID: {id}"))
        })
        .collect::<Result<_>>()?;
    Ok(decode_byte_level(tokens))
}

fn encode_hf_ordinary(tokenizer: &HfTokenizer, text: &str) -> tokenizers::Result<Vec<u32>> {
    let mut pretokenized =
        EMPTY_HF_ADDED_VOCABULARY.extract_and_normalize(tokenizer.get_normalizer(), text);

    if let Some(pre_tokenizer) = tokenizer.get_pre_tokenizer() {
        pre_tokenizer.pre_tokenize(&mut pretokenized)?;
    }
    pretokenized.tokenize(|normalized| tokenizer.get_model().tokenize(normalized.get()))?;
    let encoding = pretokenized.into_encoding(None, 0, OffsetType::Byte)?;
    let encoding = tokenizer.post_process(encoding, None, false)?;
    Ok(encoding.get_ids().to_vec())
}

/// Tokenizer from `tokenizer.json` in HuggingFace format.
///
/// This tries to load with `fastokens` first for better performance, then falls
/// back to HuggingFace's `tokenizers` if the former fails (e.g. due to
/// unsupported tokenizer features or file formats).
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
        let tokenizer_json = load_tokenizer_json_with_extra_tokens(path)?;
        let t = FastokensTokenizer::from_json(tokenizer_json)
            .map_err(|error| tokenizer_error!("failed to load tokenizer: {}", error.as_report()))?;
        Ok(Self::from_fastokens_backend(t))
    }

    /// Load from `tokenizer.json` with Hugging Face `tokenizers`.
    pub fn new_hf(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with huggingface tokenizers");
        let tokenizer_json = load_tokenizer_json_with_extra_tokens(path)?;
        let t = serde_json::from_value::<HfTokenizer>(tokenizer_json)
            .map_err(|error| tokenizer_error!("failed to load tokenizer: {}", error.as_report()))?;
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
                let encoding = t
                    .encode(text, add_special_tokens)
                    .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report()))?;
                Ok(encoding.get_ids().to_vec())
            }
            Backend::Fastokens(t) | Backend::FastokensByteLevel(t) => t
                .encode_with_special_tokens(text, add_special_tokens)
                .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report())),
        }
    }

    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        match &self.backend {
            Backend::Hf(tokenizer) => encode_hf_ordinary(tokenizer, text)
                .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report())),
            Backend::Fastokens(tokenizer) | Backend::FastokensByteLevel(tokenizer) => tokenizer
                .encode_ordinary(text)
                .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report())),
        }
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match &self.backend {
            Backend::Hf(t) => t
                .decode(token_ids, skip_special_tokens)
                .map_err(|error| tokenizer_error!("decoding failed: {}", error.as_report())),
            Backend::Fastokens(t) => t
                .decode(token_ids, skip_special_tokens)
                .map_err(|error| tokenizer_error!("decoding failed: {}", error.as_report())),
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

    fn vocab_size(&self) -> usize {
        match &self.backend {
            Backend::Hf(t) => t.get_vocab_size(true),
            Backend::Fastokens(t) | Backend::FastokensByteLevel(t) => t.vocab_size(),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match &self.backend {
            Backend::Hf(t) => t.id_to_token(id),
            Backend::Fastokens(t) | Backend::FastokensByteLevel(t) => {
                t.id_to_token(id).map(ToOwned::to_owned)
            }
        }
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        self.special_token_ids.binary_search(&token_id).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use serde_json::{Value, json};
    use tempfile::tempdir;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

    use super::{HuggingFaceTokenizer, Tokenizer};

    const REGULAR_TOKEN: &str = "<|regular|>";
    const SPECIAL_TOKEN: &str = "<|special|>";

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

    fn ordinary_test_tokenizer_json(fused: bool, with_added_tokens: bool) -> Value {
        let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        let vocab = alphabet
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token.to_string(), json!(id)))
            .collect::<serde_json::Map<_, _>>();

        let pre_tokenizer = if fused {
            json!({
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": "\\S+|\\s+"},
                        "behavior": "Isolated",
                        "invert": false
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": true,
                        "use_regex": false
                    }
                ]
            })
        } else {
            json!({
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            })
        };
        let added_tokens = with_added_tokens.then(|| {
            json!([
                {
                    "id": 256,
                    "content": REGULAR_TOKEN,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": true,
                    "special": false
                },
                {
                    "id": 257,
                    "content": SPECIAL_TOKEN,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ])
        });

        json!({
            "version": "1.0",
            "truncation": {
                "direction": "Right",
                "max_length": 24,
                "strategy": "LongestFirst",
                "stride": 0
            },
            "padding": null,
            "added_tokens": added_tokens.unwrap_or_else(|| json!([])),
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": pre_tokenizer,
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": []
            }
        })
    }

    fn write_tokenizer_json(dir: &Path, name: &str, value: &Value) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(
            &path,
            serde_json::to_vec(value).expect("serialize tokenizer"),
        )
        .expect("write tokenizer");
        path
    }

    fn assert_ordinary_matches_added_empty(
        constructor: fn(&Path) -> crate::Result<HuggingFaceTokenizer>,
        fused: bool,
    ) {
        let dir = tempdir().expect("create temp dir");
        let added_path = write_tokenizer_json(
            dir.path(),
            "with-added.json",
            &ordinary_test_tokenizer_json(fused, true),
        );
        let empty_path = write_tokenizer_json(
            dir.path(),
            "added-empty.json",
            &ordinary_test_tokenizer_json(fused, false),
        );
        let tokenizer = constructor(&added_path).expect("load tokenizer with added tokens");
        let added_empty = constructor(&empty_path).expect("load tokenizer with empty added tokens");

        assert_eq!(
            tokenizer.encode(REGULAR_TOKEN, false).unwrap(),
            vec![tokenizer.token_to_id(REGULAR_TOKEN).unwrap()]
        );
        assert_eq!(
            tokenizer.encode(SPECIAL_TOKEN, false).unwrap(),
            vec![tokenizer.token_to_id(SPECIAL_TOKEN).unwrap()]
        );

        for text in [
            "",
            "hello",
            "Cafe\u{301}",
            REGULAR_TOKEN,
            SPECIAL_TOKEN,
            "hello <|regular|> Cafe\u{301} <|special|> tail",
        ] {
            assert_eq!(
                tokenizer.encode_ordinary(text).unwrap(),
                added_empty.encode(text, false).unwrap(),
                "fused={fused}, text={text:?}",
            );
        }
        if matches!(&tokenizer.backend, super::Backend::Hf(_)) {
            assert_eq!(
                tokenizer
                    .encode_ordinary("hello <|regular|> Cafe\u{301} <|special|> tail")
                    .unwrap()
                    .len(),
                24,
                "HF post-processing must retain configured truncation",
            );
        }
    }

    #[test]
    fn hf_ordinary_matches_original_encode_with_added_empty() {
        for fused in [false, true] {
            assert_ordinary_matches_added_empty(HuggingFaceTokenizer::new_hf, fused);
        }
    }

    #[test]
    fn fastokens_ordinary_matches_original_encode_with_added_empty() {
        for fused in [false, true] {
            assert_ordinary_matches_added_empty(HuggingFaceTokenizer::new_fastokens, fused);
        }
    }

    #[test]
    fn hf_constructor_resolves_added_token_ids() {
        let mut tokenizer = tiny_bpe_tokenizer();
        tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).expect("save tokenizer json");

        let wrapper = HuggingFaceTokenizer::new_hf(&path).expect("load hf wrapper");
        let special_id = wrapper.token_to_id("<|im_end|>").expect("resolve added special token id");
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
        let special_id = wrapper.token_to_id("<|im_end|>").expect("resolve added special token id");
        assert!(wrapper.is_special_id(special_id));
    }

    #[test]
    fn constructors_merge_extra_added_tokens_from_tokenizer_config() {
        let tokenizer = tiny_bpe_tokenizer();

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).expect("save tokenizer json");
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{
                "added_tokens_decoder": {
                    "9": {
                        "content": "<|image_pad|>",
                        "special": true,
                        "normalized": false
                    }
                }
            }"#,
        )
        .expect("write tokenizer config");

        for wrapper in [
            HuggingFaceTokenizer::new_fastokens(&path).expect("load fastokens wrapper"),
            HuggingFaceTokenizer::new_hf(&path).expect("load hf wrapper"),
        ] {
            assert_eq!(wrapper.token_to_id("<|image_pad|>"), Some(9));
            assert_eq!(wrapper.id_to_token(9).as_deref(), Some("<|image_pad|>"));
            assert!(wrapper.is_special_id(9));
        }
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
