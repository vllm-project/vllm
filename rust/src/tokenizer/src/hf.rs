// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, OnceLock};

use fastokens::Tokenizer as FastokensTokenizer;
use fastokens::decoders::Decoder as FastokensDecoder;
use fastokens::pre_tokenized::{
    PreTokenizedString as FastokensPreTokenizedString, Split as FastokensSplit,
};
use fastokens::{PreTokenizer as FastokensPreTokenizer, Split as FastokensSplitPreTokenizer};
use thiserror_ext::AsReport as _;
use tokenizers::{
    AddedVocabulary, Model as _, OffsetType, PreTokenizer as _, Tokenizer as HfTokenizer,
};
use tracing::{info, warn};

use crate::byte_level_decode::decode_byte_level;
use crate::hf::added_tokens::load_tokenizer_json_with_extra_tokens;
use crate::{EncodedPrompt, Result, Tokenizer};

mod added_tokens;

static EMPTY_HF_ADDED_VOCABULARY: LazyLock<AddedVocabulary> = LazyLock::new(AddedVocabulary::new);

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
        // Match HF and fastokens: model vocabularies may contain undefined
        // tokenizer IDs, which contribute no decoded text.
        .filter_map(|&id| t.id_to_token(id))
        .collect();
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

fn fastokens_fused_split(tokenizer: &FastokensTokenizer) -> Option<&FastokensSplitPreTokenizer> {
    // Keep this predicate aligned with fastokens::Tokenizer::detect_fused_byte_level.
    let FastokensPreTokenizer::Sequence(steps) = tokenizer.pre_tokenizer()? else {
        return None;
    };
    let [
        FastokensPreTokenizer::Split(split),
        FastokensPreTokenizer::ByteLevel(byte_level),
    ] = steps.as_slice()
    else {
        return None;
    };
    byte_level.is_bulk_only().then_some(split)
}

fn fastokens_pre_tokenized_ordinary(
    tokenizer: &FastokensTokenizer,
    text: &str,
) -> FastokensPreTokenizedString {
    // This is fastokens::Tokenizer::build_pre_tokenized with added_tokens = None.
    let normalized = tokenizer
        .normalizer()
        .map_or(Cow::Borrowed(text), |normalizer| normalizer.normalize(text));
    match normalized {
        Cow::Borrowed(_) => FastokensPreTokenizedString::from_text(text),
        Cow::Owned(text) => {
            let len = text.len();
            FastokensPreTokenizedString::new(
                text,
                vec![FastokensSplit {
                    range: 0..len,
                    token_id: None,
                }],
            )
        }
    }
}

fn encode_fastokens_ordinary(
    tokenizer: &FastokensTokenizer,
    text: &str,
) -> std::result::Result<Vec<u32>, fastokens::Error> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let mut pretokenized = fastokens_pre_tokenized_ordinary(tokenizer, text);
    let ids = if let Some(split) = fastokens_fused_split(tokenizer) {
        split.pre_tokenize(&mut pretokenized)?;
        pretokenized
            .tokenize_batched(|buffer, splits, output| {
                tokenizer.model().tokenize_batch_fused(buffer, splits, output)
            })
            .map_err(fastokens::Error::Model)?
    } else {
        if let Some(pre_tokenizer) = tokenizer.pre_tokenizer() {
            pre_tokenizer.pre_tokenize(&mut pretokenized)?;
        }
        pretokenized
            .tokenize(|text, output| tokenizer.model().tokenize_into(text, output))
            .map_err(fastokens::Error::Model)?
    };

    Ok(tokenizer.post_process(ids, false))
}

/// Tokenizer from `tokenizer.json` in HuggingFace format.
///
/// This tries to load with `fastokens` first for better performance, then falls
/// back to HuggingFace's `tokenizers` if the former fails (e.g. due to
/// unsupported tokenizer features or file formats).
pub struct HuggingFaceTokenizer {
    backend: Backend,
    source_path: Option<PathBuf>,
    offsets_tokenizer: OnceLock<Option<Box<HfTokenizer>>>,
    special_token_ids: Arc<[u32]>,
    added_vocab: Box<[(String, u32)]>,
    vocab_size: usize,
}

impl HuggingFaceTokenizer {
    fn from_hf_backend(tokenizer: HfTokenizer) -> Self {
        let added_vocab = {
            let mut vocab: Vec<_> = tokenizer
                .get_added_tokens_decoder()
                .iter()
                .map(|(&id, token)| (token.content.clone(), id))
                .collect();
            vocab.sort_unstable_by_key(|(_, id)| *id);
            vocab.into_boxed_slice()
        };
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
        // HF materializes the merged vocabulary to count it, so cache the result.
        let vocab_size = tokenizer.get_vocab_size(true);
        Self {
            backend: Backend::Hf(Box::new(tokenizer)),
            source_path: None,
            offsets_tokenizer: OnceLock::new(),
            special_token_ids,
            added_vocab,
            vocab_size,
        }
    }

    fn from_fastokens_backend(tokenizer: FastokensTokenizer) -> Self {
        let added_vocab = {
            let mut vocab: Vec<_> = tokenizer
                .added_tokens()
                .into_iter()
                .flat_map(|added_tokens| added_tokens.iter())
                .map(|token| (token.content.to_string(), token.id))
                .collect();
            vocab.sort_unstable_by_key(|(_, id)| *id);
            vocab.into_boxed_slice()
        };
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
        let vocab_size = tokenizer.vocab_size();
        let backend = if byte_level {
            Backend::FastokensByteLevel(Box::new(tokenizer))
        } else {
            Backend::Fastokens(Box::new(tokenizer))
        };
        Self {
            backend,
            source_path: None,
            offsets_tokenizer: OnceLock::new(),
            special_token_ids,
            added_vocab,
            vocab_size,
        }
    }

    /// Load from `tokenizer.json` with `fastokens`.
    pub fn new_fastokens(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with fastokens");
        let tokenizer_json = load_tokenizer_json_with_extra_tokens(path)?;
        let t = FastokensTokenizer::from_json(tokenizer_json)
            .map_err(|error| tokenizer_error!("failed to load tokenizer: {}", error.as_report()))?;
        let mut tokenizer = Self::from_fastokens_backend(t);
        tokenizer.source_path = Some(path.to_path_buf());
        Ok(tokenizer)
    }

    /// Load from `tokenizer.json` with Hugging Face `tokenizers`.
    pub fn new_hf(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with huggingface tokenizers");
        Ok(Self::from_hf_backend(Self::load_hf(path)?))
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

    fn load_hf(path: &Path) -> Result<HfTokenizer> {
        let tokenizer_json = load_tokenizer_json_with_extra_tokens(path)?;
        serde_json::from_value(tokenizer_json)
            .map_err(|error| tokenizer_error!("failed to load tokenizer: {}", error.as_report()))
    }

    fn offsets_tokenizer(&self) -> Option<&HfTokenizer> {
        self.offsets_tokenizer
            .get_or_init(|| {
                let path = self.source_path.as_deref()?;
                match Self::load_hf(path) {
                    Ok(mut tokenizer) => {
                        tokenizer.with_truncation(None).ok()?;
                        tokenizer.with_padding(None);
                        Some(Box::new(tokenizer))
                    }
                    Err(error) => {
                        warn!(error = %error.as_report(), "token offsets are unavailable");
                        None
                    }
                }
            })
            .as_deref()
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

    fn encode_with_offsets(&self, text: &str, add_special_tokens: bool) -> Result<EncodedPrompt> {
        if let Backend::Hf(tokenizer) = &self.backend {
            let encoding = tokenizer
                .encode_char_offsets(text, add_special_tokens)
                .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report()))?;
            return Ok(EncodedPrompt {
                token_ids: encoding.get_ids().to_vec(),
                token_offsets: Some(encoding.get_offsets().to_vec()),
            });
        }

        let token_ids = self.encode(text, add_special_tokens)?;
        let token_offsets = self
            .offsets_tokenizer()
            .and_then(|tokenizer| tokenizer.encode_char_offsets(text, add_special_tokens).ok())
            .filter(|encoding| encoding.get_ids() == token_ids)
            .map(|encoding| encoding.get_offsets().to_vec());
        Ok(EncodedPrompt {
            token_ids,
            token_offsets,
        })
    }

    fn warm_offsets(&self) {
        if !matches!(self.backend, Backend::Hf(_)) {
            self.offsets_tokenizer();
        }
    }

    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        match &self.backend {
            Backend::Hf(tokenizer) => encode_hf_ordinary(tokenizer, text)
                .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report())),
            Backend::Fastokens(tokenizer) | Backend::FastokensByteLevel(tokenizer) => {
                encode_fastokens_ordinary(tokenizer, text)
                    .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report()))
            }
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
        self.vocab_size
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match &self.backend {
            Backend::Hf(t) => t.id_to_token(id),
            Backend::Fastokens(t) | Backend::FastokensByteLevel(t) => {
                t.id_to_token(id).map(ToOwned::to_owned)
            }
        }
    }

    fn added_vocab(&self) -> &[(String, u32)] {
        &self.added_vocab
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
    use crate::{TokenAnchor, TokenAttribution};

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

    #[test]
    fn offsets_match_hf_source_characters_and_preserve_token_ids() {
        let dir = tempdir().unwrap();
        let mut json = ordinary_test_tokenizer_json(false, true);
        let processor = tokenizers::processors::template::TemplateProcessing::builder()
            .try_single(format!("{SPECIAL_TOKEN} $A"))
            .unwrap()
            .special_tokens(vec![(SPECIAL_TOKEN, 257)])
            .build()
            .unwrap();
        json["post_processor"] = serde_json::to_value(processor).unwrap();
        let path = write_tokenizer_json(dir.path(), "tokenizer.json", &json);
        let text = "é🙂Cafe\u{301}<|special|>";
        for tokenizer in [
            HuggingFaceTokenizer::new_hf(&path).unwrap(),
            HuggingFaceTokenizer::new_fastokens(&path).unwrap(),
        ] {
            for add_special_tokens in [false, true] {
                let encoded = tokenizer.encode_with_offsets(text, add_special_tokens).unwrap();
                assert_eq!(
                    encoded.token_ids,
                    tokenizer.encode(text, add_special_tokens).unwrap()
                );
                let mut expected = vec![
                    (0, 1),
                    (0, 1),
                    (1, 2),
                    (1, 2),
                    (1, 2),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (5, 6),
                    (7, 18),
                ];
                if add_special_tokens {
                    expected.insert(0, (0, 0));
                }
                assert_eq!(encoded.token_offsets, Some(expected));
            }
        }
    }

    #[test]
    fn fastokens_offsets_ignore_hf_settings_but_reject_changed_tokenization() {
        let dir = tempdir().unwrap();
        let mut json = ordinary_test_tokenizer_json(false, false);
        json["padding"] = serde_json::json!({
            "strategy": {"Fixed": 30}, "direction": "Right",
            "pad_to_multiple_of": null, "pad_id": 0, "pad_type_id": 0, "pad_token": "!"
        });
        let path = write_tokenizer_json(dir.path(), "tokenizer.json", &json);
        let tokenizer = HuggingFaceTokenizer::new_fastokens(&path).unwrap();
        tokenizer.warm_offsets();
        let x_id = json["model"]["vocab"]["x"].clone();
        let y_id = json["model"]["vocab"]["y"].clone();
        json["model"]["vocab"]["x"] = y_id.clone();
        json["model"]["vocab"]["y"] = x_id.clone();
        write_tokenizer_json(dir.path(), "tokenizer.json", &json);
        let text = "x".repeat(25);
        let encoded = tokenizer.encode_with_offsets(&text, false).unwrap();
        assert_eq!(encoded.token_ids, tokenizer.encode(&text, false).unwrap());
        assert_eq!(encoded.token_ids.len(), 25);
        assert_eq!(
            encoded.token_offsets,
            Some((0..25).map(|i| (i, i + 1)).collect())
        );

        let tokenizer = HuggingFaceTokenizer::new_fastokens(&path).unwrap();
        json["model"]["vocab"]["x"] = x_id;
        json["model"]["vocab"]["y"] = y_id;
        write_tokenizer_json(dir.path(), "tokenizer.json", &json);
        let encoded = tokenizer.encode_with_offsets(&text, false).unwrap();
        assert_eq!(encoded.token_ids, tokenizer.encode(&text, false).unwrap());
        assert!(encoded.token_offsets.is_none());
    }

    #[test]
    fn fastokens_preserves_ids_when_offsets_cannot_load() {
        let dir = tempdir().unwrap();
        let json = ordinary_test_tokenizer_json(false, false);
        let path = write_tokenizer_json(dir.path(), "tokenizer.json", &json);
        let mut tokenizer = HuggingFaceTokenizer::new_fastokens(&path).unwrap();
        tokenizer.source_path = Some(dir.path().join("missing.json"));
        let encoded = tokenizer.encode_with_offsets("hello", false).unwrap();
        assert_eq!(encoded.token_ids, tokenizer.encode("hello", false).unwrap());
        assert!(encoded.token_offsets.is_none());
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

        if let super::Backend::Fastokens(inner) | super::Backend::FastokensByteLevel(inner) =
            &tokenizer.backend
        {
            assert_eq!(super::fastokens_fused_split(inner).is_some(), fused);
        }

        assert_eq!(
            tokenizer.encode(REGULAR_TOKEN, false).unwrap(),
            vec![tokenizer.token_to_id(REGULAR_TOKEN).unwrap()]
        );
        assert_eq!(
            tokenizer.encode(SPECIAL_TOKEN, false).unwrap(),
            vec![tokenizer.token_to_id(SPECIAL_TOKEN).unwrap()]
        );
        assert_eq!(
            tokenizer.added_vocab(),
            vec![
                (REGULAR_TOKEN.to_string(), 256),
                (SPECIAL_TOKEN.to_string(), 257),
            ]
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

    /// Added tokens store raw text, including whitespace that the GPT-2 byte
    /// table never emits, so a byte-level decode must pass it through verbatim,
    /// along with characters the table does emit in the same token (`é`).
    #[test]
    fn added_tokens_with_raw_whitespace_decode_verbatim() {
        const OPEN: &str = "<parameter name=\"";
        const CLOSE: &str = "\n</parameter>";
        const MIXED: &str = "\ncafé";

        let mut value = ordinary_test_tokenizer_json(false, false);
        value["added_tokens"] = json!(
            [OPEN, CLOSE, MIXED]
                .iter()
                .enumerate()
                .map(|(i, content)| {
                    json!({
                        "id": 256 + i,
                        "content": content,
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false,
                        "special": false
                    })
                })
                .collect::<Vec<_>>()
        );
        let dir = tempdir().expect("create temp dir");
        let path = write_tokenizer_json(dir.path(), "tokenizer.json", &value);
        let text = format!("{OPEN}city\">\nParis{CLOSE}{MIXED}");

        let fastokens = HuggingFaceTokenizer::new_fastokens(&path).expect("load fastokens wrapper");
        assert!(matches!(
            fastokens.backend,
            super::Backend::FastokensByteLevel(_)
        ));
        let hf = HuggingFaceTokenizer::new_hf(&path).expect("load hf wrapper");
        for wrapper in [fastokens, hf] {
            let ids = wrapper.encode(&text, false).expect("encode");
            assert!((256..=258).all(|id| ids.contains(&id)), "ids={ids:?}");
            assert_eq!(wrapper.decode(&ids, true).expect("decode"), text);
        }
    }

    #[test]
    fn hf_vocab_size_counts_added_tokens() {
        let mut tokenizer = tiny_bpe_tokenizer();
        tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);
        let expected = tokenizer.get_vocab_size(true);

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).expect("save tokenizer json");

        let wrapper = HuggingFaceTokenizer::new_hf(&path).expect("load hf wrapper");
        assert_eq!(wrapper.vocab_size(), expected);
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
    fn fast_byte_level_skips_undefined_ids() {
        let t = tiny_byte_level_bpe();
        assert_eq!(
            super::decode_fastokens_byte_level(&t, &[1, 2, 999, 3, 3, 4], false)
                .expect("with the id"),
            super::decode_fastokens_byte_level(&t, &[1, 2, 3, 3, 4], false)
                .expect("without the id")
        );
    }

    #[test]
    fn decode_stream_anchors_undefined_ids_zero_width() {
        let wrapper = HuggingFaceTokenizer::from_fastokens_backend(tiny_byte_level_bpe());
        assert!(matches!(
            wrapper.backend,
            super::Backend::FastokensByteLevel(_)
        ));

        let mut stream = wrapper.create_decode_stream(&[], false, 0);
        for id in [999, 1, 999, 2] {
            stream.push_token(id).expect("push token");
        }
        let (_, full) = stream.flush(None).expect("flush");
        assert_eq!(full.text, "He");
        assert_eq!(
            full.attributions.as_slice(),
            [
                TokenAttribution {
                    token_id: 999,
                    anchor: TokenAnchor::ZeroWidth { byte_offset: 0 }
                },
                TokenAttribution {
                    token_id: 1,
                    anchor: TokenAnchor::Visible { byte_offset: 0 }
                },
                TokenAttribution {
                    token_id: 999,
                    anchor: TokenAnchor::ZeroWidth { byte_offset: 1 }
                },
                TokenAttribution {
                    token_id: 2,
                    anchor: TokenAnchor::Visible { byte_offset: 1 }
                },
            ]
        );
    }
}
