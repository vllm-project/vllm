use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use base64::Engine as _;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Deserialize;
use thiserror_ext::AsReport as _;
use tracing::{info, warn};

use super::Tokenizer;
use crate::Error;
use crate::error::Result;

/// Default regex pattern used when loading tiktoken from a BPE file. This is
/// the same `cl100k_base` pattern that HuggingFace transformers uses as its
/// default in `TikTokenConverter`.
const CL100K_BASE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Kimi BPE pattern from `moonshotai/Kimi-K2-Instruct/tokenization_kimi.py`.
const KIMI_PATTERN: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Fallback number of reserved special-token slots to assume when the model's
/// `config.json` is not available (so we cannot read `vocab_size` directly).
///
/// 256 is the value used by Kimi K2 / K2.5 (`tokenization_kimi.py`'s
/// `num_reserved_special_tokens`) and by Llama 3, and it appears to be the most
/// common convention among modern tiktoken-based HF tokenizers. When
/// `config.json` *is* present we honour the model's actual `vocab_size` instead
/// of this fallback — see `Self::new`.
const FALLBACK_NUM_RESERVED_SPECIAL_TOKENS: u32 = 256;
const DISABLE_RIPTOKEN_ENV: &str = "VLLM_RS_DISABLE_RIPTOKEN";

/// Parsed entry from `tokenizer_config.json`'s `added_tokens_decoder`.
#[derive(Debug, Clone, Deserialize)]
struct AddedToken {
    content: String,
    /// HuggingFace `added_tokens_decoder` entries can be marked `"special":
    /// true|false`. Special tokens are dropped from output when `decode` is
    /// called with `skip_special_tokens = true`. Defaults to `false` when
    /// the field is omitted, matching HuggingFace's `AddedToken` default —
    /// so only tokens explicitly marked special are stripped during normal
    /// decode (where `skip_special_tokens` itself defaults to true).
    #[serde(default)]
    special: bool,
}

/// Minimal subset of `tokenizer_config.json` needed by the tiktoken loader.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct TiktokenTokenizerConfig {
    /// Format:
    /// `{ "added_tokens_decoder": { "163584": { "content": "[BOS]", "special":
    /// true }, ... } }`
    #[serde(default)]
    added_tokens_decoder: FxHashMap<u32, AddedToken>,
}

/// Minimal subset of model `config.json` needed by the tiktoken loader.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct TiktokenModelConfig {
    model_type: Option<String>,
    vocab_size: Option<u32>,
    text_config: Option<Box<TiktokenModelConfig>>,
}

impl TiktokenModelConfig {
    /// Read `model_type` from a model `config.json` value, falling back to a
    /// single-level nested `text_config.model_type` for composite (e.g.
    /// multimodal) configs that keep text metadata under a `text_config`
    /// object.
    fn effective_model_type(&self) -> Option<&str> {
        self.model_type
            .as_deref()
            .or_else(|| self.text_config.as_deref()?.effective_model_type())
    }

    /// Read `vocab_size` from a model `config.json` value, falling back to a
    /// single-level nested `text_config.vocab_size` for composite (e.g.
    /// multimodal) configs that keep text metadata under a `text_config`
    /// object — matching the same shape `ModelConfig` parses.
    fn effective_vocab_size(&self) -> Option<u32> {
        self.vocab_size.or_else(|| self.text_config.as_deref()?.effective_vocab_size())
    }
}

/// Tiktoken tokenizer from `tiktoken.model` or `*.tiktoken` BPE files.
pub struct TiktokenTokenizer {
    backend: Backend,
    metadata: TokenMetadata,
}

enum Backend {
    Riptoken(RiptokenBackend),
    TiktokenRs(TiktokenRsBackend),
}

struct RiptokenBackend {
    inner: Box<riptoken::CoreBPE>,
    allowed_special_tokens: Vec<String>,
}

struct TiktokenRsBackend {
    inner: Box<tiktoken_rs::CoreBPE>,
    /// Reverse map for special / added token strings populated from the
    /// reserved range. This lets `token_to_id` answer special-token lookups
    /// directly without round-tripping through `tiktoken-rs`'s encoder,
    /// which can panic for unknown special-looking strings.
    special_token_ids_by_text: FxHashMap<String, u32>,
    /// Set of out-of-vocab token IDs we have already warned about. The
    /// reserved-slot population in the constructor should keep this empty
    /// under normal operation; it only fills up if a model emits ids at or
    /// above `vocab_upper_bound` (e.g. an engine sampling bug). We dedupe
    /// so streaming decode (which calls `decode` repeatedly on the same prefix)
    /// does not spam.
    warned_unknown_ids: Mutex<FxHashSet<u32>>,
}

struct TokenMetadata {
    /// Number of regular BPE tokens. Token ids in `[0, num_base_tokens)` are
    /// BPE tokens that always decode to text; ids in `[num_base_tokens,
    /// vocab_upper_bound)` live in the special-token slots and are subject
    /// to `skip_special_tokens` filtering.
    num_base_tokens: u32,
    /// Exclusive upper bound on token IDs that `inner` is guaranteed to know
    /// how to decode.
    ///
    /// The constructor registers every id in `[num_base_tokens,
    /// vocab_upper_bound)` with the inner `CoreBPE` as a (named or
    /// `<|reserved_token_{id}|>`) special token, and the BPE
    /// encoder densely covers `[0, num_base_tokens)`. So any id below this
    /// bound is in one of the inner `CoreBPE`'s decoder maps and
    /// `_decode_native_and_split` will not panic on it. `decode` filters
    /// out ids at or above this bound to keep that guarantee.
    vocab_upper_bound: u32,
    /// Ids in `[num_base_tokens, vocab_upper_bound)` whose
    /// `added_tokens_decoder` entry was explicitly marked `"special":
    /// false` — i.e. tokens that should still appear in output
    /// even when `skip_special_tokens = true`. For Kimi K2 / K2.5 this
    /// typically holds the tool-call markers and `<think>` / `</think>`.
    /// Reserved-slot placeholders are not in this set (they default to
    /// special and get skipped).
    non_special_added_ids: FxHashSet<u32>,
}

impl TokenMetadata {
    fn filter_special_tokens(&self, token_ids: &[u32]) -> Vec<u32> {
        token_ids
            .iter()
            .copied()
            .filter(|&id| {
                id < self.num_base_tokens
                    || id >= self.vocab_upper_bound
                    || self.non_special_added_ids.contains(&id)
            })
            .collect()
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        token_id >= self.num_base_tokens
            && token_id < self.vocab_upper_bound
            && !self.non_special_added_ids.contains(&token_id)
    }
}

impl RiptokenBackend {
    fn encode(&self, text: &str) -> Vec<u32> {
        // TODO: avoid collecting `allowed_special` every time this method is called.
        let allowed_special: HashSet<&str> =
            self.allowed_special_tokens.iter().map(String::as_str).collect();
        self.inner.encode(text, &allowed_special)
    }

    fn decode(&self, token_ids: &[u32]) -> String {
        let bytes = self.inner.decode_bytes(token_ids);
        // TODO: use `from_utf8_lossy_owned` once it's stabilized.
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.encode_single_token(token.as_bytes())
    }
}

impl TiktokenRsBackend {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode_with_special_tokens(text)
    }

    fn decode(&self, token_ids: &[u32], metadata: &TokenMetadata) -> String {
        let safe_ids: Vec<u32> = token_ids
            .iter()
            .copied()
            .filter(|&id| {
                if id >= metadata.vocab_upper_bound {
                    self.warn_unknown_id(id);
                    return false;
                }
                true
            })
            .collect();
        let bytes: Vec<u8> = self.inner._decode_native_and_split(safe_ids).flatten().collect();
        // TODO: use `from_utf8_lossy_owned` once it's stabilized.
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(&token_id) = self.special_token_ids_by_text.get(token) {
            return Some(token_id);
        }

        // Fall back to ordinary encoding for regular vocabulary items. This
        // deliberately avoids `encode_with_special_tokens`: older `tiktoken-rs`
        // versions can panic if the input text merely *looks* like a special
        // token but is not registered in `special_tokens_encoder`.
        let ids = self.inner.encode_ordinary(token);
        if ids.len() == 1 { Some(ids[0]) } else { None }
    }

    /// Log a warning the first time an unknown token id is seen during decode,
    /// deduped across calls so streaming decode does not spam the log for
    /// the same id.
    fn warn_unknown_id(&self, token_id: u32) {
        let newly_inserted = self
            .warned_unknown_ids
            .lock()
            .map(|mut set| set.insert(token_id))
            .unwrap_or(false);
        if newly_inserted {
            warn!(
                token_id,
                "tiktoken-rs decode encountered token id not in the vocabulary; skipping. \
                 This typically indicates a sparse-vocab model whose `added_tokens_decoder` \
                 does not list every reserved id in the special-token range."
            );
        }
    }
}

impl TiktokenTokenizer {
    /// Load a tiktoken tokenizer from a `.tiktoken` / `tiktoken.model` BPE
    /// file.
    ///
    /// The BPE file format is one `<base64-token-bytes> <rank>` pair per line,
    /// the same format used by OpenAI's tiktoken and by HuggingFace model
    /// repos that ship tiktoken files (e.g. DeepSeek, Kimi K2).
    ///
    /// Special / added tokens are read from `tokenizer_config.json` in the same
    /// directory when present. The `cl100k_base` regex pattern is used as a
    /// reasonable default.
    pub fn new(path: &Path) -> Result<Self> {
        if std::env::var_os(DISABLE_RIPTOKEN_ENV).is_some() {
            return Self::new_tiktoken_rs(path);
        }

        match Self::new_riptoken(path) {
            Ok(tokenizer) => Ok(tokenizer),
            Err(error) => {
                warn!(
                    path = %path.display(),
                    error = %error.as_report(),
                    "failed to load tokenizer with riptoken; falling back to tiktoken-rs"
                );
                Self::new_tiktoken_rs(path)
            }
        }
    }

    /// Load from `tiktoken.model` / `*.tiktoken` with riptoken.
    pub fn new_riptoken(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with riptoken (BPE file)");

        let config = LoadedTiktokenConfig::load(path)?;
        let allowed_special_tokens = config.special_tokens_encoder.keys().cloned().collect();
        let inner = riptoken::CoreBPE::new(
            config.encoder.into_iter().collect(),
            config.special_tokens_encoder.into_iter().collect(),
            config.pattern,
        )
        .map_err(|error| {
            Error::Tokenizer(format!(
                "failed to create riptoken tokenizer from {}: {error}",
                path.display()
            ))
        })?;

        Ok(Self {
            backend: Backend::Riptoken(RiptokenBackend {
                inner: Box::new(inner),
                allowed_special_tokens,
            }),
            metadata: config.metadata,
        })
    }

    /// Load from `tiktoken.model` / `*.tiktoken` with tiktoken-rs.
    pub fn new_tiktoken_rs(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with tiktoken-rs (BPE file)");

        let config = LoadedTiktokenConfig::load(path)?;
        let special_token_ids_by_text = config.special_tokens_encoder.clone();
        let inner = tiktoken_rs::CoreBPE::new(
            config.encoder,
            config.special_tokens_encoder,
            config.pattern,
        )
        .map_err(|error| {
            Error::Tokenizer(format!(
                "failed to create tiktoken-rs tokenizer from {}: {error}",
                path.display()
            ))
        })?;

        Ok(Self {
            backend: Backend::TiktokenRs(TiktokenRsBackend {
                inner: Box::new(inner),
                special_token_ids_by_text,
                warned_unknown_ids: Mutex::new(FxHashSet::default()),
            }),
            metadata: config.metadata,
        })
    }
}

struct LoadedTiktokenConfig {
    encoder: FxHashMap<Vec<u8>, u32>,
    special_tokens_encoder: FxHashMap<String, u32>,
    metadata: TokenMetadata,
    pattern: &'static str,
}

impl LoadedTiktokenConfig {
    fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|error| {
            Error::Tokenizer(format!(
                "failed to read tiktoken file {}: {}",
                path.display(),
                error.as_report()
            ))
        })?;
        let mut encoder: FxHashMap<Vec<u8>, u32> =
            FxHashMap::with_capacity_and_hasher(content.lines().count(), Default::default());
        for line in content.lines() {
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let token_b64 = parts
                .next()
                .ok_or_else(|| Error::Tokenizer("missing token in tiktoken file".to_string()))?;
            let rank_str = parts
                .next()
                .ok_or_else(|| Error::Tokenizer("missing rank in tiktoken file".to_string()))?;
            let token_bytes =
                base64::engine::general_purpose::STANDARD.decode(token_b64).map_err(|error| {
                    Error::Tokenizer(format!("invalid base64 in tiktoken file: {error}"))
                })?;
            let rank: u32 = rank_str.parse().map_err(|error| {
                Error::Tokenizer(format!("invalid rank in tiktoken file: {error}"))
            })?;
            encoder.insert(token_bytes, rank);
        }

        let parent_dir = path.parent();

        // Read added/special tokens (id -> {name, special}) from
        // tokenizer_config.json in the same dir.
        let added_tokens_by_id = parent_dir
            .map(|dir| dir.join("tokenizer_config.json"))
            .filter(|p| p.exists())
            .and_then(|config_path| {
                let content = std::fs::read_to_string(&config_path).ok()?;
                serde_json::from_str(&content).ok()
            })
            .map(|config: TiktokenTokenizerConfig| config.added_tokens_decoder)
            .unwrap_or_default();

        let model_config: Option<TiktokenModelConfig> = parent_dir
            .map(|dir| dir.join("config.json"))
            .filter(|p| p.exists())
            .and_then(|config_path| {
                let content = std::fs::read_to_string(&config_path).ok()?;
                serde_json::from_str(&content).ok()
            });
        let vocab_size_from_config = model_config.as_ref().and_then(|c| c.effective_vocab_size());

        // Build the full special-tokens encoder by populating the reserved
        // range that follows the BPE vocabulary. Unknown reserved slots get
        // Python-compatible placeholder names so sampled ids can still decode.
        //
        // Note: `*.tiktoken` ranks are token ids, and they are not guaranteed
        // to be contiguous. The base-vocab boundary is therefore `max_rank + 1`,
        // not `encoder.len()`.
        let num_base_tokens =
            encoder.values().copied().max().map_or(0, |max_rank| max_rank.saturating_add(1));
        let max_added_id = added_tokens_by_id.keys().copied().max().unwrap_or(0);
        let reserved_end = vocab_size_from_config
            .unwrap_or_else(|| num_base_tokens.saturating_add(FALLBACK_NUM_RESERVED_SPECIAL_TOKENS))
            .max(num_base_tokens)
            .max(max_added_id.saturating_add(1));

        let mut special_tokens_encoder: FxHashMap<String, u32> =
            FxHashMap::with_capacity_and_hasher(
                (reserved_end - num_base_tokens) as usize,
                Default::default(),
            );
        let mut non_special_added_ids: FxHashSet<u32> = FxHashSet::default();
        for id in num_base_tokens..reserved_end {
            let name = match added_tokens_by_id.get(&id) {
                Some(token) => {
                    if !token.special {
                        non_special_added_ids.insert(id);
                    }
                    token.content.clone()
                }
                None => format!("<|reserved_token_{id}|>"),
            };
            special_tokens_encoder.insert(name, id);
        }

        let pattern = model_config.as_ref().map_or(CL100K_BASE_PATTERN, detect_bpe_pattern);

        Ok(Self {
            encoder,
            special_tokens_encoder,
            metadata: TokenMetadata {
                num_base_tokens,
                vocab_upper_bound: reserved_end,
                non_special_added_ids,
            },
            pattern,
        })
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        // Tiktoken does not have a separate add_special_tokens toggle; both
        // backends recognize registered special tokens in the input.
        Ok(match &self.backend {
            Backend::Riptoken(backend) => backend.encode(text),
            Backend::TiktokenRs(backend) => backend.encode(text),
        })
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        // Filter passes:
        //
        // 1. The constructor registers every id in `[num_base_tokens,
        //    vocab_upper_bound)` as a special token (named or `<|reserved_token_{id}|>`
        //    placeholder, matching `tokenization_kimi.py`). The tiktoken-rs backend
        //    additionally drops ids at or above that bound so
        //    `_decode_native_and_split` cannot panic; riptoken's `decode_bytes` already
        //    skips unknown ids.
        //
        // 2. When `skip_special_tokens = true`, ids in `[num_base_tokens,
        //    vocab_upper_bound)` are dropped *unless* they were marked `"special":
        //    false` in `added_tokens_decoder`. This matches HuggingFace's tokenizer
        //    semantics: tool-call markers and `<think>` / `</think>` (which Kimi K2 /
        //    K2.5 declare as non-special) stay in the output, while BOS/EOS/header
        //    tokens and reserved-slot placeholders are stripped.
        //
        // Lossy UTF-8 decoding (instead of strict `String::from_utf8`) is used so
        // partial multi-byte sequences become `\u{FFFD}`, which `DecodeStream`
        // relies on to detect incomplete characters during streaming.
        let ids = if skip_special_tokens {
            &self.metadata.filter_special_tokens(token_ids)
        } else {
            token_ids
        };

        Ok(match &self.backend {
            Backend::Riptoken(backend) => backend.decode(ids),
            Backend::TiktokenRs(backend) => backend.decode(ids, &self.metadata),
        })
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match &self.backend {
            Backend::Riptoken(backend) => backend.token_to_id(token),
            Backend::TiktokenRs(backend) => backend.token_to_id(token),
        }
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        self.metadata.is_special_id(token_id)
    }
}

/// Select the BPE regex pattern for a tiktoken model based on `config.json`.
///
/// Most tiktoken models use the `cl100k_base` regex. Kimi models ship a custom
/// regex in their Python tokenizer implementation; we mirror the explicit
/// `model_type` switch used by Dynamo instead of heuristically parsing Python
/// source files.
fn detect_bpe_pattern(config: &TiktokenModelConfig) -> &'static str {
    let model_type = config.effective_model_type();

    match model_type {
        Some("kimi" | "kimi_k2" | "kimi_k25" | "deepseek_v3") => KIMI_PATTERN,
        _ => CL100K_BASE_PATTERN,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use base64::Engine as _;
    use tempfile::TempDir;

    use super::{
        CL100K_BASE_PATTERN, KIMI_PATTERN, TiktokenModelConfig, TiktokenTokenizer,
        TiktokenTokenizerConfig, detect_bpe_pattern,
    };
    use crate::backend::hf::{ResolvedModelFiles, TokenizerSource};
    use crate::tokenizer::Tokenizer;

    macro_rules! config_json {
        ($($json:tt)+) => {
            serde_json::from_value::<TiktokenModelConfig>(serde_json::json!($($json)+)).unwrap()
        };
    }

    /// Write a minimal `*.tiktoken` BPE file (one token per byte 0..=255) into
    /// `dir` and return its path. The single-byte vocab is enough to
    /// exercise the multi-byte / streaming UTF-8 paths without depending on
    /// any pretrained tokenizer asset.
    fn write_synthetic_bpe_file(dir: &std::path::Path) -> PathBuf {
        let mut content = String::new();
        for byte in 0u8..=255 {
            let b64 = base64::engine::general_purpose::STANDARD.encode([byte]);
            content.push_str(&format!("{b64} {}\n", byte as u32));
        }
        let path = dir.join("test.tiktoken");
        fs::write(&path, content).expect("write tiktoken file");
        path
    }

    /// Write a synthetic `*.tiktoken` file whose base-vocab ranks are
    /// sparse/non-contiguous.
    ///
    /// This reproduces the important edge case for `num_base_tokens`: it must
    /// be derived from `max_rank + 1`, not `encoder.len()`, otherwise
    /// high-rank base tokens get misclassified as reserved/special ids.
    fn write_sparse_rank_bpe_file(dir: &std::path::Path) -> PathBuf {
        let mut content = String::new();
        for byte in 0u8..=255 {
            let b64 = base64::engine::general_purpose::STANDARD.encode([byte]);
            content.push_str(&format!("{b64} {}\n", byte as u32));
        }

        let high_rank_token = base64::engine::general_purpose::STANDARD.encode(b"SPARSE");
        content.push_str(&format!("{high_rank_token} 1000\n"));

        let path = dir.join("sparse-rank.tiktoken");
        fs::write(&path, content).expect("write sparse-rank tiktoken file");
        path
    }

    /// Build a `TiktokenTokenizer` from the synthetic BPE file with no sibling
    /// config files, so the constructor takes the
    /// `FALLBACK_NUM_RESERVED_SPECIAL_TOKENS` (256) path.
    fn explicit_backends(path: &Path) -> Vec<TiktokenTokenizer> {
        vec![
            TiktokenTokenizer::new_riptoken(path).expect("load riptoken backend"),
            TiktokenTokenizer::new_tiktoken_rs(path).expect("load tiktoken-rs backend"),
        ]
    }

    fn tiktoken_backends() -> (Vec<TiktokenTokenizer>, TempDir) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = write_synthetic_bpe_file(dir.path());
        (explicit_backends(&path), dir)
    }

    /// Verify that tiktoken decode uses lossy UTF-8 (producing `\u{FFFD}`)
    /// rather than returning an error for incomplete multi-byte sequences.
    /// This is critical for streaming decode — `DecodeStream` relies on
    /// `\u{FFFD}` to detect incomplete characters.
    #[test]
    fn tiktoken_decode_incomplete_utf8_produces_replacement_char() {
        let (backends, _dir) = tiktoken_backends();

        for backend in backends {
            let ids = backend.encode("你", false).unwrap();
            let full = backend.decode(&ids, false).unwrap();
            assert_eq!(full, "你");

            let text_with_multibyte = "Hello你好World";
            let all_ids = backend.encode(text_with_multibyte, false).unwrap();
            for &id in &all_ids {
                let result = backend.decode(&[id], false);
                assert!(result.is_ok(), "decode of token {id} should not error");
            }
        }
    }

    /// When `config.json` exposes a `vocab_size`, the reserved-token range must
    /// be sized to it rather than to the 256-slot fallback. This is the
    /// general (non-Kimi-specific) path: any tiktoken model whose own
    /// `config.json` says e.g. `vocab_size = 280` should populate
    /// reserved slots for `[num_base_tokens, 280)` and nothing beyond.
    #[test]
    fn tiktoken_reserved_range_uses_vocab_size_from_config_json() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let bpe_path = write_synthetic_bpe_file(dir.path());
        // num_base_tokens = 256, vocab_size = 280 → reserved range = [256, 280) (24
        // slots, smaller than the 256 fallback so we can prove the config value
        // is honoured).
        fs::write(dir.path().join("config.json"), r#"{"vocab_size": 280}"#)
            .expect("write config.json");

        for backend in explicit_backends(&bpe_path) {
            // Inside the configured range: reserved placeholder, round-trips both ways.
            let in_range_id: u32 = 270;
            let placeholder = format!("<|reserved_token_{in_range_id}|>");
            assert_eq!(backend.decode(&[in_range_id], false).unwrap(), placeholder);
            assert_eq!(
                backend.encode(&placeholder, false).unwrap(),
                vec![in_range_id]
            );

            // Outside the configured range: not registered as a reserved slot — falls
            // through to the backend's unknown-id behavior. The point is that we *don't*
            // over-populate beyond what the model actually exposes.
            let out_of_range_id: u32 = 290;
            let out_of_range_placeholder = format!("<|reserved_token_{out_of_range_id}|>");
            assert_eq!(backend.decode(&[out_of_range_id], false).unwrap(), "");
            assert_eq!(backend.token_to_id(&out_of_range_placeholder), None);
        }
    }

    /// Sparse/non-contiguous BPE ranks must still count as base-vocab ids.
    ///
    /// Regression shape:
    /// - base vocabulary contains ids 0..=255 and also a normal BPE token at id
    ///   1000
    /// - if `num_base_tokens` were computed as `encoder.len()` (257), id 1000
    ///   would be misclassified as special/reserved and disappear under
    ///   `skip_special_tokens = true`
    #[test]
    fn tiktoken_sparse_base_ranks_are_not_misclassified_as_special() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let bpe_path = write_sparse_rank_bpe_file(dir.path());
        fs::write(dir.path().join("config.json"), r#"{"vocab_size": 1002}"#)
            .expect("write config.json");

        for backend in explicit_backends(&bpe_path) {
            let sparse_id = backend.token_to_id("SPARSE");
            assert_eq!(sparse_id, Some(1000));
            assert!(!backend.is_special_id(1000));
            assert_eq!(backend.decode(&[1000], false).unwrap(), "SPARSE");
            assert_eq!(backend.decode(&[1000], true).unwrap(), "SPARSE");
        }
    }

    /// `skip_special_tokens` must:
    ///  * keep regular BPE token text unchanged,
    ///  * drop ids whose `added_tokens_decoder` entry says `"special": true`,
    ///  * drop reserved-slot placeholder ids (which default to special),
    ///  * keep ids whose `added_tokens_decoder` entry says `"special": false` —
    ///    this is how Kimi K2 / K2.5 marks tool-call markers and `<think>` /
    ///    `</think>`.
    ///
    /// Synthetic backend has `num_base_tokens = 256`. We write a
    /// `tokenizer_config.json` that names ids 257 (special) and 258
    /// (non-special), and a `config.json` with `vocab_size` covering both.
    /// Id 259 stays a default reserved placeholder (special).
    #[test]
    fn tiktoken_skip_special_tokens_filters_special_but_keeps_non_special_added_tokens() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let bpe_path = write_synthetic_bpe_file(dir.path());
        fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{
                "added_tokens_decoder": {
                    "257": { "content": "<|im_end|>", "special": true },
                    "258": { "content": "<|tool_call_begin|>", "special": false }
                }
            }"#,
        )
        .expect("write tokenizer_config.json");
        fs::write(dir.path().join("config.json"), r#"{"vocab_size": 260}"#)
            .expect("write config.json");

        for backend in explicit_backends(&bpe_path) {
            // Resolve the BPE ids for "Hi" so we can interleave them with special-token
            // ids.
            let h = backend.encode("H", false).unwrap()[0];
            let i = backend.encode("i", false).unwrap()[0];

            let special_id: u32 = 257; // <|im_end|>
            let non_special_id: u32 = 258; // <|tool_call_begin|>
            let reserved_id: u32 = 259; // default <|reserved_token_259|> placeholder

            let ids = vec![h, special_id, i, non_special_id, reserved_id];

            // skip_special_tokens = false: everything is rendered as-is.
            let kept = backend.decode(&ids, false).unwrap();
            assert_eq!(
                kept,
                "H<|im_end|>i<|tool_call_begin|><|reserved_token_259|>"
            );

            // skip_special_tokens = true: special token (257) and reserved placeholder
            // (259) are dropped; the non-special added token (258) survives.
            let stripped = backend.decode(&ids, true).unwrap();
            assert_eq!(stripped, "Hi<|tool_call_begin|>");
        }
    }

    /// `vocab_size` may live under `text_config` for composite (e.g.
    /// multimodal) configs.
    #[test]
    fn tiktoken_reserved_range_reads_text_config_vocab_size() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let bpe_path = write_synthetic_bpe_file(dir.path());
        fs::write(
            dir.path().join("config.json"),
            r#"{"text_config": {"vocab_size": 270}}"#,
        )
        .expect("write config.json");

        for backend in explicit_backends(&bpe_path) {
            let in_range_id: u32 = 260;
            let placeholder = format!("<|reserved_token_{in_range_id}|>");
            assert_eq!(backend.decode(&[in_range_id], false).unwrap(), placeholder);

            // Just outside the nested vocab_size — should not be registered.
            assert_eq!(backend.decode(&[270], false).unwrap(), "");
        }
    }

    #[test]
    fn tiktoken_detects_kimi_pattern_from_model_type() {
        let kimi = config_json!({ "model_type": "kimi_k25" });
        let baseten_kimi = config_json!({ "model_type": "deepseek_v3" });
        let nested_kimi = config_json!({
            "model_type": "composite_wrapper",
            "text_config": { "model_type": "kimi_k2" }
        });
        let generic = config_json!({ "model_type": "gpt2" });
        let nested_generic = config_json!({
            "model_type": "composite_wrapper",
            "text_config": { "model_type": "gpt2" }
        });
        let missing = config_json!({ "text_config": {} });

        assert_eq!(detect_bpe_pattern(&kimi), KIMI_PATTERN);
        assert_eq!(detect_bpe_pattern(&baseten_kimi), KIMI_PATTERN);
        assert_eq!(detect_bpe_pattern(&nested_kimi), CL100K_BASE_PATTERN);
        assert_eq!(detect_bpe_pattern(&generic), CL100K_BASE_PATTERN);
        assert_eq!(detect_bpe_pattern(&nested_generic), CL100K_BASE_PATTERN);
        assert_eq!(detect_bpe_pattern(&missing), CL100K_BASE_PATTERN);
    }

    #[test]
    fn tiktoken_reads_model_type_from_text_config_when_top_level_missing() {
        let nested_only = config_json!({
            "text_config": { "model_type": "kimi_k2" }
        });
        let direct_and_nested = config_json!({
            "model_type": "kimi_k25",
            "text_config": { "model_type": "kimi_k2" }
        });
        let missing = config_json!({
            "text_config": {}
        });

        assert_eq!(nested_only.effective_model_type(), Some("kimi_k2"));
        assert_eq!(direct_and_nested.effective_model_type(), Some("kimi_k25"));
        assert_eq!(missing.effective_model_type(), None);
    }

    #[test]
    fn tiktoken_tokenizer_config_models_added_tokens_decoder() {
        let config: TiktokenTokenizerConfig = serde_json::from_value(serde_json::json!({
            "added_tokens_decoder": {
                "257": { "content": "<think>" },
                "258": { "content": "</think>", "special": true }
            }
        }))
        .unwrap();

        let added_tokens = config.added_tokens_decoder;
        assert_eq!(added_tokens.len(), 2);
        assert_eq!(
            added_tokens.get(&257).map(|t| t.content.as_str()),
            Some("<think>")
        );
        assert_eq!(added_tokens.get(&257).map(|t| t.special), Some(false));
        assert_eq!(
            added_tokens.get(&258).map(|t| (t.content.as_str(), t.special)),
            Some(("</think>", true))
        );
    }

    /// Reserved token ids in `[num_base_tokens, num_base_tokens + 256)` must
    /// decode to their placeholder name (matching `tokenization_kimi.py`'s
    /// `<|reserved_token_{i}|>` format), even when the source
    /// `tokenizer_config.json` does not list them in `added_tokens_decoder`.
    ///
    /// In our synthetic backend `num_base_tokens = 256` (256 single-byte BPE
    /// tokens), so the reserved range is `[256, 512)`. Picking id 300 —
    /// well inside that range and absent from any `added_tokens_decoder` —
    /// should round-trip both ways.
    #[test]
    fn tiktoken_reserved_token_round_trip() {
        let (backends, _dir) = tiktoken_backends();

        for backend in backends {
            let reserved_id: u32 = 300;
            let placeholder = format!("<|reserved_token_{reserved_id}|>");

            let decoded = backend.decode(&[reserved_id], false).unwrap();
            assert_eq!(decoded, placeholder);

            // The placeholder name should also encode back to the same single id, since
            // the constructor registers it as a special token with `CoreBPE`.
            let encoded = backend.encode(&placeholder, false).unwrap();
            assert_eq!(encoded, vec![reserved_id]);

            assert_eq!(backend.token_to_id(&placeholder), Some(reserved_id));
        }
    }

    /// Decoding a token id that is beyond even the reserved range must not
    /// panic — it falls through to the warn-and-skip backstop instead of
    /// crashing the worker thread.
    #[test]
    fn tiktoken_rs_decode_unknown_token_id_does_not_panic() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = write_synthetic_bpe_file(dir.path());
        let backend = TiktokenTokenizer::new_tiktoken_rs(&path).expect("load tiktoken-rs backend");

        // ID well above num_base_tokens (256) + reserved (256) = 512 — guaranteed
        // unknown.
        let unknown_id: u32 = 999_999;
        let result = backend.decode(&[unknown_id], false);
        assert_eq!(result.unwrap(), "");

        // Mixed: known bytes for "Hi" surrounding an unknown id should yield just "Hi".
        let h = backend.encode("H", false).unwrap()[0];
        let i = backend.encode("i", false).unwrap()[0];
        let result = backend.decode(&[h, unknown_id, i], false).unwrap();
        assert_eq!(result, "Hi");
    }

    #[test]
    fn riptoken_decode_unknown_token_id_does_not_panic() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = write_synthetic_bpe_file(dir.path());
        let backend = TiktokenTokenizer::new_riptoken(&path).expect("load riptoken backend");

        let unknown_id: u32 = 999_999;
        assert_eq!(backend.decode(&[unknown_id], false).unwrap(), "");

        let h = backend.encode("H", false).unwrap()[0];
        let i = backend.encode("i", false).unwrap()[0];
        assert_eq!(backend.decode(&[h, unknown_id, i], false).unwrap(), "Hi");
    }

    /// Streaming decode of CJK text through tiktoken should produce the
    /// original text without errors, even though individual tokens may
    /// represent partial UTF-8 byte sequences.
    #[test]
    fn tiktoken_streaming_decode_multibyte() {
        let (backends, _dir) = tiktoken_backends();
        for backend in backends {
            let text = "你好世界"; // 4 CJK characters
            let ids = backend.encode(text, false).unwrap();

            let mut decoder = backend.create_decode_stream(&[], false, 0);
            let mut output = String::new();
            for &id in &ids {
                decoder.push_token(id).unwrap();
                if let Some(chunk) = decoder.next_chunk() {
                    output.push_str(&chunk);
                }
            }
            let (last_chunk, full_text) = decoder.flush(None).unwrap();
            if let Some(chunk) = last_chunk {
                output.push_str(&chunk);
            }

            assert_eq!(output, text);
            assert_eq!(full_text, text);
        }
    }

    /// Mixed ASCII and multi-byte text should stream correctly through
    /// tiktoken.
    #[test]
    fn tiktoken_streaming_decode_mixed_ascii_and_multibyte() {
        let (backends, _dir) = tiktoken_backends();
        for backend in backends {
            let text = "Hello 你好 World 🌍";
            let ids = backend.encode(text, false).unwrap();

            let mut decoder = backend.create_decode_stream(&[], false, 0);
            let mut output = String::new();
            for &id in &ids {
                decoder.push_token(id).unwrap();
                if let Some(chunk) = decoder.next_chunk() {
                    output.push_str(&chunk);
                }
            }
            let (last_chunk, full_text) = decoder.flush(None).unwrap();
            if let Some(chunk) = last_chunk {
                output.push_str(&chunk);
            }

            assert_eq!(output, text);
            assert_eq!(full_text, text);
        }
    }

    #[test]
    fn tiktoken_token_to_id_resolves_added_special_tokens() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let bpe_path = write_synthetic_bpe_file(dir.path());
        fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{
                "added_tokens_decoder": {
                    "257": { "content": "<think>", "special": false },
                    "258": { "content": "</think>", "special": false }
                }
            }"#,
        )
        .expect("write tokenizer_config.json");
        fs::write(dir.path().join("config.json"), r#"{"vocab_size": 259}"#)
            .expect("write config.json");

        for backend in explicit_backends(&bpe_path) {
            assert_eq!(backend.token_to_id("<think>"), Some(257));
            assert_eq!(backend.token_to_id("</think>"), Some(258));
            assert_eq!(
                backend.decode(&[257, 258], true).unwrap(),
                "<think></think>"
            );
        }
    }

    #[test]
    fn riptoken_token_to_id_uses_encode_single_token_path() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let bpe_path = write_synthetic_bpe_file(dir.path());
        fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{
                "added_tokens_decoder": {
                    "257": { "content": "<think>", "special": false }
                }
            }"#,
        )
        .expect("write tokenizer_config.json");
        fs::write(dir.path().join("config.json"), r#"{"vocab_size": 258}"#)
            .expect("write config.json");
        let backend = TiktokenTokenizer::new_riptoken(&bpe_path).expect("load riptoken backend");

        assert_eq!(backend.token_to_id("H"), Some(b'H' as u32));
        assert_eq!(backend.token_to_id("<think>"), Some(257));
    }

    #[test]
    fn tiktoken_rs_token_to_id_handles_unknown_special_like_text_without_panicking() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = write_synthetic_bpe_file(dir.path());
        let backend = TiktokenTokenizer::new_tiktoken_rs(&path).expect("load tiktoken-rs backend");

        assert_eq!(backend.token_to_id("<|definitely_not_registered|>"), None);
    }

    #[test]
    fn riptoken_token_to_id_handles_unknown_special_like_text_without_panicking() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = write_synthetic_bpe_file(dir.path());
        let backend = TiktokenTokenizer::new_riptoken(&path).expect("load riptoken backend");

        assert_eq!(backend.token_to_id("<|definitely_not_registered|>"), None);
    }

    #[tokio::test]
    #[ignore = "requires network access to Hugging Face and downloads the real Kimi K2.5 tokenizer"]
    async fn tiktoken_real_kimi_k25_tokenizer_files_load_and_handle_special_tokens() {
        let files = ResolvedModelFiles::new("moonshotai/Kimi-K2.5")
            .await
            .expect("resolve real Kimi K2.5 model files");

        let tokenizer_path = match &files.tokenizer {
            TokenizerSource::Tiktoken(path) => path.clone(),
            other => panic!("expected tiktoken tokenizer source, got {other:?}"),
        };

        for backend in explicit_backends(&tokenizer_path) {
            let think_id = backend.token_to_id("<think>").expect("resolve <think>");
            let end_think_id = backend.token_to_id("</think>").expect("resolve </think>");
            let tool_section_id = backend
                .token_to_id("<|tool_calls_section_begin|>")
                .expect("resolve tool call section marker");
            let contraction_heavy_text =
                "I'm sure it's fine, but I can't say I'd trust that it's what we'd ship.";
            let contraction_heavy_ids = backend.encode(contraction_heavy_text, false).unwrap();

            assert_eq!(
                (think_id, end_think_id, tool_section_id),
                (163606, 163607, 163595)
            );
            assert_eq!(backend.decode(&[think_id], true).unwrap(), "<think>");
            assert_eq!(backend.decode(&[end_think_id], true).unwrap(), "</think>");
            assert_eq!(
                backend.decode(&[tool_section_id], true).unwrap(),
                "<|tool_calls_section_begin|>"
            );

            // This demonstrates that we're using Kimi's custom BPE pattern.
            // With CL100K this will be 23 tokens instead.
            assert_eq!(
                contraction_heavy_ids,
                vec![
                    17172, 3287, 4643, 8201, 11, 996, 374, 8971, 3637, 20020, 8173, 473, 4643,
                    1573, 56229, 13922, 13,
                ]
            );
            assert_eq!(contraction_heavy_ids.len(), 17);
            assert_eq!(
                backend.decode(&contraction_heavy_ids, false).unwrap(),
                contraction_heavy_text
            );

            // Special-looking text that is not actually registered should fail gracefully.
            assert_eq!(backend.token_to_id("◁think▷"), None);
            assert_eq!(backend.token_to_id("<|definitely_not_registered|>"), None);
        }
    }
}
