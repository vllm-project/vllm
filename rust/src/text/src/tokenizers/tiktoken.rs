use std::path::Path;

use base64::Engine as _;
use rustc_hash::FxHashMap;
use thiserror_ext::AsReport as _;
use tiktoken_rs::CoreBPE;
use tracing::info;

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

/// Tiktoken tokenizer from `tiktoken.model` or `*.tiktoken` BPE files.
pub struct TiktokenTokenizer {
    inner: CoreBPE,
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

        // Read added/special tokens from tokenizer_config.json in the same directory.
        let special_tokens_encoder: FxHashMap<String, u32> = path
            .parent()
            .map(|dir| dir.join("tokenizer_config.json"))
            .filter(|p| p.exists())
            .and_then(|config_path| {
                let config_content = std::fs::read_to_string(&config_path).ok()?;
                let config: serde_json::Value = serde_json::from_str(&config_content).ok()?;
                parse_added_tokens_from_config(&config)
            })
            .unwrap_or_default();

        let bpe = CoreBPE::new(encoder, special_tokens_encoder, CL100K_BASE_PATTERN).map_err(
            |error| {
                Error::Tokenizer(format!(
                    "failed to create tiktoken tokenizer from {}: {error}",
                    path.display()
                ))
            },
        )?;

        Ok(Self { inner: bpe })
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        // tiktoken-rs does not have a separate add_special_tokens toggle;
        // `encode_with_special_tokens` always recognizes special tokens in the input.
        Ok(self.inner.encode_with_special_tokens(text))
    }

    fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        // Use lossy UTF-8 decoding instead of `CoreBPE::decode()` which does strict
        // `String::from_utf8()`. During streaming, the `DecodeStream` relies on `\u{FFFD}`
        // to detect incomplete multi-byte sequences, but strict decoding returns an error
        // instead. Lossy decoding produces `\u{FFFD}` which the stream buffers correctly.
        //
        // TODO: tiktoken-rs does not natively support `skip_special_tokens`; all tokens are
        // decoded as-is.
        let bytes: Vec<u8> = self
            .inner
            ._decode_native_and_split(token_ids.to_vec())
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
}

/// Parse `added_tokens_decoder` from `tokenizer_config.json` into a special-tokens map for
/// `CoreBPE`.
///
/// Format: `{ "added_tokens_decoder": { "163584": { "content": "[BOS]", "special": true }, ... } }`
fn parse_added_tokens_from_config(
    config: &serde_json::Value,
) -> Option<rustc_hash::FxHashMap<String, u32>> {
    let added = config
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())?;
    let mut tokens = rustc_hash::FxHashMap::default();
    for (id_str, token_info) in added {
        if let (Ok(id), Some(content)) = (
            id_str.parse::<u32>(),
            token_info.get("content").and_then(|v| v.as_str()),
        ) {
            tokens.insert(content.to_string(), id);
        }
    }
    Some(tokens)
}

#[cfg(test)]
mod tests {
    use super::TiktokenTokenizer;
    use crate::tokenizers::Tokenizer;

    fn tiktoken_backend() -> TiktokenTokenizer {
        let bpe = tiktoken_rs::cl100k_base().expect("cl100k_base should load");
        TiktokenTokenizer { inner: bpe }
    }

    /// Verify that tiktoken decode uses lossy UTF-8 (producing `\u{FFFD}`) rather than
    /// returning an error for incomplete multi-byte sequences. This is critical for streaming
    /// decode — `DecodeStream` relies on `\u{FFFD}` to detect incomplete characters.
    #[test]
    fn tiktoken_decode_incomplete_utf8_produces_replacement_char() {
        let backend = tiktoken_backend();

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

    /// Streaming decode of CJK text through tiktoken should produce the original text without
    /// errors, even though individual tokens may represent partial UTF-8 byte sequences.
    #[test]
    fn tiktoken_streaming_decode_multibyte() {
        let backend = tiktoken_backend();
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
        let backend = tiktoken_backend();
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
