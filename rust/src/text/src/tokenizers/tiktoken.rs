use std::path::Path;
use std::sync::Mutex;

use base64::Engine as _;
use rustc_hash::{FxHashMap, FxHashSet};
use thiserror_ext::AsReport as _;
use tiktoken_rs::CoreBPE;
use tracing::{info, warn};

use super::Tokenizer;
use crate::Error;
use crate::error::Result;

/// Default regex pattern used when loading tiktoken from a BPE file. This is the same
/// `cl100k_base` pattern that HuggingFace transformers uses as its default in
/// `TikTokenConverter`.
///
/// The `.tiktoken` file format does not include a regex pattern — each model's pattern is
/// defined in its Python tokenizer source (e.g. `tokenization_kimi.py`). Some models use a
/// different regex (e.g. Kimi K2 adds `\p{Han}` for CJK grouping), which can affect token
/// boundaries but not encode/decode correctness.
const CL100K_BASE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Fallback number of reserved special-token slots to assume when the model's `config.json`
/// is not available (so we cannot read `vocab_size` directly).
///
/// 256 is the value used by Kimi K2 / K2.5 (`tokenization_kimi.py`'s
/// `num_reserved_special_tokens`) and by Llama 3, and it appears to be the most common
/// convention among modern tiktoken-based HF tokenizers. When `config.json` *is* present we
/// honour the model's actual `vocab_size` instead of this fallback — see `Self::new`.
const FALLBACK_NUM_RESERVED_SPECIAL_TOKENS: u32 = 256;

/// Parsed entry from `tokenizer_config.json`'s `added_tokens_decoder`.
struct AddedToken {
    content: String,
    /// HuggingFace `added_tokens_decoder` entries can be marked `"special": true|false`.
    /// Special tokens are dropped from output when `decode` is called with
    /// `skip_special_tokens = true`. Defaults to `false` when the field is omitted, matching
    /// HuggingFace's `AddedToken` default — so only tokens explicitly marked special are
    /// stripped during normal decode (where `skip_special_tokens` itself defaults to true).
    special: bool,
}

/// Tiktoken tokenizer from `tiktoken.model` or `*.tiktoken` BPE files.
pub struct TiktokenTokenizer {
    inner: CoreBPE,
    /// Number of regular BPE tokens. Token ids in `[0, num_base_tokens)` are BPE tokens that
    /// always decode to text; ids in `[num_base_tokens, vocab_upper_bound)` live in the
    /// special-token slots and are subject to `skip_special_tokens` filtering.
    num_base_tokens: u32,
    /// Exclusive upper bound on token IDs that `inner` is guaranteed to know how to decode.
    ///
    /// The constructor registers every id in `[num_base_tokens, vocab_upper_bound)` with the
    /// inner `CoreBPE` as a (named or `<|reserved_token_{id}|>`) special token, and the BPE
    /// encoder densely covers `[0, num_base_tokens)`. So any id below this bound is in one of
    /// the inner `CoreBPE`'s decoder maps and `_decode_native_and_split` will not panic on it.
    /// `decode` filters out ids at or above this bound to keep that guarantee.
    vocab_upper_bound: u32,
    /// Ids in `[num_base_tokens, vocab_upper_bound)` whose `added_tokens_decoder` entry was
    /// explicitly marked `"special": false` — i.e. tokens that should still appear in output
    /// even when `skip_special_tokens = true`. For Kimi K2 / K2.5 this typically holds the
    /// tool-call markers and `<think>` / `</think>`. Reserved-slot placeholders are not in
    /// this set (they default to special and get skipped).
    non_special_added_ids: FxHashSet<u32>,
    /// Set of out-of-vocab token IDs we have already warned about. The reserved-slot population
    /// in the constructor should keep this empty under normal operation; it only fills up if a
    /// model emits ids at or above `vocab_upper_bound` (e.g. an engine sampling bug). We dedupe
    /// so streaming decode (which calls `decode` repeatedly on the same prefix) does not spam.
    warned_unknown_ids: Mutex<FxHashSet<u32>>,
}

impl TiktokenTokenizer {
    /// Load a tiktoken tokenizer from a `.tiktoken` / `tiktoken.model` BPE file.
    ///
    /// The BPE file format is one `<base64-token-bytes> <rank>` pair per line, the same format
    /// used by OpenAI's tiktoken and by HuggingFace model repos that ship tiktoken files (e.g.
    /// DeepSeek, Kimi K2).
    ///
    /// Special / added tokens are read from `tokenizer_config.json` in the same directory when
    /// present. The `cl100k_base` regex pattern is used as a reasonable default.
    pub fn new(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with tiktoken (BPE file)");

        // Parse the BPE file.
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
            let token_bytes = base64::engine::general_purpose::STANDARD
                .decode(token_b64)
                .map_err(|error| {
                    Error::Tokenizer(format!("invalid base64 in tiktoken file: {error}"))
                })?;
            let rank: u32 = rank_str.parse().map_err(|error| {
                Error::Tokenizer(format!("invalid rank in tiktoken file: {error}"))
            })?;
            encoder.insert(token_bytes, rank);
        }

        let parent_dir = path.parent();

        // Read added/special tokens (id → {name, special}) from tokenizer_config.json in the
        // same dir.
        let added_tokens_by_id: FxHashMap<u32, AddedToken> = parent_dir
            .map(|dir| dir.join("tokenizer_config.json"))
            .filter(|p| p.exists())
            .and_then(|config_path| {
                let config_content = std::fs::read_to_string(&config_path).ok()?;
                let config: serde_json::Value = serde_json::from_str(&config_content).ok()?;
                parse_added_tokens_from_config(&config)
            })
            .unwrap_or_default();

        // Read `vocab_size` from the model's config.json (top-level or nested `text_config`)
        // if available.
        let vocab_size_from_config: Option<u32> = parent_dir
            .map(|dir| dir.join("config.json"))
            .filter(|p| p.exists())
            .and_then(|config_path| {
                let content = std::fs::read_to_string(&config_path).ok()?;
                let value: serde_json::Value = serde_json::from_str(&content).ok()?;
                read_vocab_size(&value)
            });

        // Build the full special-tokens encoder by populating the reserved range that follows
        // the BPE vocabulary. The Python reference does this in `tokenization_kimi.py`:
        //
        //   for i in range(num_base_tokens, num_base_tokens + num_reserved_special_tokens):
        //       name = added_tokens_decoder.get(i, f"<|reserved_token_{i}|>")
        //
        // The same idea generalises to any tiktoken-based HF model: any id that the model is
        // allowed to sample but is not listed in `added_tokens_decoder` is a "reserved" slot
        // that should still decode to *something* rather than panic. Without this step, the
        // model could emit a reserved id (e.g. id 163589 for Kimi K2.5) and decoding would
        // panic in `CoreBPE::_decode_native_and_split`.
        //
        // We size the reserved range using whichever upper bound is largest:
        //   1. `vocab_size` from config.json if present (the accurate, per-model answer),
        //   2. otherwise `num_base_tokens + 256` (the Kimi/Llama 3 default convention),
        //   3. extended further to cover any explicit `added_tokens_decoder` id beyond either.
        let num_base_tokens = encoder.len() as u32;
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

        let bpe = CoreBPE::new(encoder, special_tokens_encoder, CL100K_BASE_PATTERN).map_err(
            |error| {
                Error::Tokenizer(format!(
                    "failed to create tiktoken tokenizer from {}: {error}",
                    path.display()
                ))
            },
        )?;

        Ok(Self {
            inner: bpe,
            num_base_tokens,
            vocab_upper_bound: reserved_end,
            non_special_added_ids,
            warned_unknown_ids: Mutex::new(FxHashSet::default()),
        })
    }

    /// Log a warning the first time an unknown token id is seen during decode, deduped across
    /// calls so streaming decode does not spam the log for the same id.
    fn warn_unknown_id(&self, token_id: u32) {
        let newly_inserted = self
            .warned_unknown_ids
            .lock()
            .map(|mut set| set.insert(token_id))
            .unwrap_or(false);
        if newly_inserted {
            warn!(
                token_id,
                "tiktoken decode encountered token id not in the vocabulary; skipping. \
                 This typically indicates a sparse-vocab model whose `added_tokens_decoder` \
                 does not list every reserved id in the special-token range."
            );
        }
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        // tiktoken-rs does not have a separate add_special_tokens toggle;
        // `encode_with_special_tokens` always recognizes special tokens in the input.
        Ok(self.inner.encode_with_special_tokens(text))
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        // Filter passes:
        //
        // 1. Ids at or above `vocab_upper_bound` are dropped (with a warn-once log) — without this,
        //    `_decode_native_and_split` would panic on ids missing from both of `CoreBPE`'s
        //    internal decoder maps. The constructor registers every id in `[num_base_tokens,
        //    vocab_upper_bound)` as a special token (named or `<|reserved_token_{id}|>`
        //    placeholder, matching `tokenization_kimi.py`), so any in-range id is safe and this
        //    branch only fires for genuinely out-of-vocab ids — e.g. an engine sampling bug
        //    emitting an id above the model's stated vocab_size.
        //
        // 2. When `skip_special_tokens = true`, ids in `[num_base_tokens, vocab_upper_bound)` are
        //    dropped *unless* they were marked `"special": false` in `added_tokens_decoder`. This
        //    matches HuggingFace's tokenizer semantics: tool-call markers and `<think>` /
        //    `</think>` (which Kimi K2 / K2.5 declare as non-special) stay in the output, while
        //    BOS/EOS/header tokens and reserved-slot placeholders are stripped.
        //
        // Lossy UTF-8 decoding (instead of strict `String::from_utf8`) is used so partial
        // multi-byte sequences become `\u{FFFD}`, which `DecodeStream` relies on to detect
        // incomplete characters during streaming.
        let safe_ids: Vec<u32> = token_ids
            .iter()
            .copied()
            .filter(|&id| {
                if id >= self.vocab_upper_bound {
                    self.warn_unknown_id(id);
                    return false;
                }
                !skip_special_tokens
                    || id < self.num_base_tokens
                    || self.non_special_added_ids.contains(&id)
            })
            .collect();
        let bytes: Vec<u8> = self
            .inner
            ._decode_native_and_split(safe_ids)
            .flatten()
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        // tiktoken-rs has no direct `token_to_id`; encode the token and return the ID only if
        // it maps to exactly one token.
        let ids = self.inner.encode_with_special_tokens(token);
        if ids.len() == 1 { Some(ids[0]) } else { None }
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        token_id >= self.num_base_tokens
            && token_id < self.vocab_upper_bound
            && !self.non_special_added_ids.contains(&token_id)
    }
}

/// Read `vocab_size` from a model `config.json` value, falling back to a single-level nested
/// `text_config.vocab_size` for composite (e.g. multimodal) configs that keep text metadata
/// under a `text_config` object — matching the same shape `ModelConfig` parses.
fn read_vocab_size(config: &serde_json::Value) -> Option<u32> {
    let direct = config.get("vocab_size").and_then(|v| v.as_u64());
    let nested = config
        .get("text_config")
        .and_then(|tc| tc.get("vocab_size"))
        .and_then(|v| v.as_u64());
    direct.or(nested).and_then(|n| u32::try_from(n).ok())
}

/// Parse `added_tokens_decoder` from `tokenizer_config.json` into an id → `AddedToken` map.
///
/// Format: `{ "added_tokens_decoder": { "163584": { "content": "[BOS]", "special": true }, ... } }`
///
/// The `"special"` flag is honoured by `decode` when `skip_special_tokens = true`. Entries that
/// omit the flag default to `special = false`, matching HuggingFace's `AddedToken` default.
fn parse_added_tokens_from_config(
    config: &serde_json::Value,
) -> Option<FxHashMap<u32, AddedToken>> {
    let added = config
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())?;
    let mut tokens = FxHashMap::default();
    for (id_str, token_info) in added {
        if let (Ok(id), Some(content)) = (
            id_str.parse::<u32>(),
            token_info.get("content").and_then(|v| v.as_str()),
        ) {
            let special = token_info
                .get("special")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            tokens.insert(
                id,
                AddedToken {
                    content: content.to_string(),
                    special,
                },
            );
        }
    }
    Some(tokens)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use base64::Engine as _;
    use tempfile::TempDir;

    use super::TiktokenTokenizer;
    use crate::tokenizers::Tokenizer;

    /// Write a minimal `*.tiktoken` BPE file (one token per byte 0..=255) into `dir` and
    /// return its path. The single-byte vocab is enough to exercise the multi-byte / streaming
    /// UTF-8 paths without depending on any pretrained tokenizer asset.
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

    /// Build a `TiktokenTokenizer` from the synthetic BPE file with no sibling config files,
    /// so the constructor takes the `FALLBACK_NUM_RESERVED_SPECIAL_TOKENS` (256) path.
    fn tiktoken_backend() -> (TiktokenTokenizer, TempDir) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = write_synthetic_bpe_file(dir.path());
        let backend = TiktokenTokenizer::new(&path).expect("load tiktoken backend");
        (backend, dir)
    }

    /// Verify that tiktoken decode uses lossy UTF-8 (producing `\u{FFFD}`) rather than
    /// returning an error for incomplete multi-byte sequences. This is critical for streaming
    /// decode — `DecodeStream` relies on `\u{FFFD}` to detect incomplete characters.
    #[test]
    fn tiktoken_decode_incomplete_utf8_produces_replacement_char() {
        let (backend, _dir) = tiktoken_backend();

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

    /// When `config.json` exposes a `vocab_size`, the reserved-token range must be sized to it
    /// rather than to the 256-slot fallback. This is the general (non-Kimi-specific) path: any
    /// tiktoken model whose own `config.json` says e.g. `vocab_size = 280` should populate
    /// reserved slots for `[num_base_tokens, 280)` and nothing beyond.
    #[test]
    fn tiktoken_reserved_range_uses_vocab_size_from_config_json() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let bpe_path = write_synthetic_bpe_file(dir.path());
        // num_base_tokens = 256, vocab_size = 280 → reserved range = [256, 280) (24 slots,
        // smaller than the 256 fallback so we can prove the config value is honoured).
        fs::write(dir.path().join("config.json"), r#"{"vocab_size": 280}"#)
            .expect("write config.json");
        let backend = TiktokenTokenizer::new(&bpe_path).expect("load tiktoken backend");

        // Inside the configured range: reserved placeholder, round-trips both ways.
        let in_range_id: u32 = 270;
        let placeholder = format!("<|reserved_token_{in_range_id}|>");
        assert_eq!(backend.decode(&[in_range_id], false).unwrap(), placeholder);
        assert_eq!(
            backend.encode(&placeholder, false).unwrap(),
            vec![in_range_id]
        );

        // Outside the configured range: not registered as a reserved slot — falls through to
        // the warn-and-skip backstop. The point is that we *don't* over-populate beyond what
        // the model actually exposes.
        let out_of_range_id: u32 = 290;
        let out_of_range_placeholder = format!("<|reserved_token_{out_of_range_id}|>");
        assert_eq!(backend.decode(&[out_of_range_id], false).unwrap(), "");
        assert_eq!(backend.token_to_id(&out_of_range_placeholder), None);
    }

    /// `skip_special_tokens` must:
    ///  * keep regular BPE token text unchanged,
    ///  * drop ids whose `added_tokens_decoder` entry says `"special": true`,
    ///  * drop reserved-slot placeholder ids (which default to special),
    ///  * keep ids whose `added_tokens_decoder` entry says `"special": false` — this is how Kimi K2
    ///    / K2.5 marks tool-call markers and `<think>` / `</think>`.
    ///
    /// Synthetic backend has `num_base_tokens = 256`. We write a `tokenizer_config.json` that
    /// names ids 257 (special) and 258 (non-special), and a `config.json` with `vocab_size`
    /// covering both. Id 259 stays a default reserved placeholder (special).
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
        let backend = TiktokenTokenizer::new(&bpe_path).expect("load tiktoken backend");

        // Resolve the BPE ids for "Hi" so we can interleave them with special-token ids.
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

        // skip_special_tokens = true: special token (257) and reserved placeholder (259) are
        // dropped; the non-special added token (258) survives.
        let stripped = backend.decode(&ids, true).unwrap();
        assert_eq!(stripped, "Hi<|tool_call_begin|>");
    }

    /// `vocab_size` may live under `text_config` for composite (e.g. multimodal) configs.
    #[test]
    fn tiktoken_reserved_range_reads_text_config_vocab_size() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let bpe_path = write_synthetic_bpe_file(dir.path());
        fs::write(
            dir.path().join("config.json"),
            r#"{"text_config": {"vocab_size": 270}}"#,
        )
        .expect("write config.json");
        let backend = TiktokenTokenizer::new(&bpe_path).expect("load tiktoken backend");

        let in_range_id: u32 = 260;
        let placeholder = format!("<|reserved_token_{in_range_id}|>");
        assert_eq!(backend.decode(&[in_range_id], false).unwrap(), placeholder);

        // Just outside the nested vocab_size — should not be registered.
        assert_eq!(backend.decode(&[270], false).unwrap(), "");
    }

    /// Reserved token ids in `[num_base_tokens, num_base_tokens + 256)` must decode to their
    /// placeholder name (matching `tokenization_kimi.py`'s `<|reserved_token_{i}|>` format),
    /// even when the source `tokenizer_config.json` does not list them in `added_tokens_decoder`.
    ///
    /// In our synthetic backend `num_base_tokens = 256` (256 single-byte BPE tokens), so the
    /// reserved range is `[256, 512)`. Picking id 300 — well inside that range and absent
    /// from any `added_tokens_decoder` — should round-trip both ways.
    #[test]
    fn tiktoken_reserved_token_round_trip() {
        let (backend, _dir) = tiktoken_backend();

        let reserved_id: u32 = 300;
        let placeholder = format!("<|reserved_token_{reserved_id}|>");

        let decoded = backend.decode(&[reserved_id], false).unwrap();
        assert_eq!(decoded, placeholder);

        // The placeholder name should also encode back to the same single id, since the
        // constructor registers it as a special token with `CoreBPE`.
        let encoded = backend.encode(&placeholder, false).unwrap();
        assert_eq!(encoded, vec![reserved_id]);

        assert_eq!(backend.token_to_id(&placeholder), Some(reserved_id));
    }

    /// Decoding a token id that is beyond even the reserved range must not panic — it falls
    /// through to the warn-and-skip backstop instead of crashing the worker thread.
    #[test]
    fn tiktoken_decode_unknown_token_id_does_not_panic() {
        let (backend, _dir) = tiktoken_backend();

        // ID well above num_base_tokens (256) + reserved (256) = 512 — guaranteed unknown.
        let unknown_id: u32 = 999_999;
        let result = backend.decode(&[unknown_id], false);
        assert_eq!(result.unwrap(), "");

        // Mixed: known bytes for "Hi" surrounding an unknown id should yield just "Hi".
        let h = backend.encode("H", false).unwrap()[0];
        let i = backend.encode("i", false).unwrap()[0];
        let result = backend.decode(&[h, unknown_id, i], false).unwrap();
        assert_eq!(result, "Hi");
    }

    /// Streaming decode of CJK text through tiktoken should produce the original text without
    /// errors, even though individual tokens may represent partial UTF-8 byte sequences.
    #[test]
    fn tiktoken_streaming_decode_multibyte() {
        let (backend, _dir) = tiktoken_backend();
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

    /// Mixed ASCII and multi-byte text should stream correctly through tiktoken.
    #[test]
    fn tiktoken_streaming_decode_mixed_ascii_and_multibyte() {
        let (backend, _dir) = tiktoken_backend();
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
