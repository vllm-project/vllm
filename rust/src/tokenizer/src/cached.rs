//! Tokenizer caching layer using `llm-tokenizer`'s [`CachedTokenizer`] and
//! L0/L1 cache infrastructure.
//!
//! Since vllm defines its own [`Tokenizer`](crate::Tokenizer) trait and
//! `llm-tokenizer` defines a separate `Encoder + Decoder + Tokenizer` trait
//! hierarchy, this module provides an **adapter** ([`VllmTokenizerAdapter`])
//! that bridges the two. The adapter wraps any `vllm_tokenizer::Tokenizer`
//! and implements `llm_tokenizer::traits::{Encoder, Decoder, Tokenizer}` so
//! that `llm_tokenizer::CachedTokenizer` can wrap it.
//!
//! The public [`LlmCachedTokenizer`] struct then wraps the whole stack and
//! re-implements `vllm_tokenizer::Tokenizer`, making the caching layer
//! transparent to the rest of vllm.
//!
//! # Usage
//!
//! ```ignore
//! use vllm_tokenizer::{LlmCachedTokenizer, LlmCacheConfig};
//!
//! let inner = HuggingFaceTokenizer::new(path)?;
//! let cached = LlmCachedTokenizer::new(inner, LlmCacheConfig::default());
//! let tokens = cached.encode("Hello world", false)?;
//! ```

use std::any::Any;
use std::sync::Arc;

use llm_tokenizer::cache::{CacheConfig as LlmCacheConfigInner, CachedTokenizer};
use llm_tokenizer::traits::{
    Decoder as LlmDecoder, Encoder as LlmEncoder, Encoding as LlmEncoding,
    SpecialTokens as LlmSpecialTokens, TokenIdType, Tokenizer as LlmTokenizer,
};

use crate::incremental::DecodeStream;
use crate::{IncrementalDecoder, Result, Tokenizer};

/// Configuration for the llm-tokenizer cache layer.
#[derive(Debug, Clone)]
pub struct LlmCacheConfig {
    /// Enable L0 (whole-string exact match) cache.
    pub enable_l0: bool,
    /// Maximum number of entries in L0 cache.
    pub l0_max_entries: usize,
    /// Enable L1 (prefix matching) cache.
    pub enable_l1: bool,
    /// Maximum memory for L1 cache in bytes.
    pub l1_max_memory: usize,
}

impl Default for LlmCacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: true,
            l0_max_entries: 10_000,
            enable_l1: false,
            l1_max_memory: 50 * 1024 * 1024,
        }
    }
}

impl From<LlmCacheConfig> for LlmCacheConfigInner {
    fn from(c: LlmCacheConfig) -> Self {
        LlmCacheConfigInner {
            enable_l0: c.enable_l0,
            l0_max_entries: c.l0_max_entries,
            enable_l1: c.enable_l1,
            l1_max_memory: c.l1_max_memory,
        }
    }
}

// ---------------------------------------------------------------------------
// Adapter: vllm Tokenizer → llm-tokenizer traits
// ---------------------------------------------------------------------------

/// Wraps a vllm [`Tokenizer`] and implements `llm_tokenizer`'s
/// `Encoder`, `Decoder`, and `Tokenizer` traits so that
/// `llm_tokenizer::CachedTokenizer` can wrap it.
struct VllmTokenizerAdapter<T: Tokenizer> {
    inner: T,
    /// Placeholder special tokens — vllm's Tokenizer trait does not expose
    /// special-token metadata, so we provide an empty default.
    special_tokens: LlmSpecialTokens,
}

impl<T: Tokenizer> LlmEncoder for VllmTokenizerAdapter<T> {
    fn encode(&self, input: &str, add_special_tokens: bool) -> anyhow::Result<LlmEncoding> {
        let ids = self
            .inner
            .encode(input, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(LlmEncoding::Plain(ids))
    }

    fn encode_batch(
        &self,
        inputs: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<LlmEncoding>> {
        inputs
            .iter()
            .map(|&input| LlmEncoder::encode(self, input, add_special_tokens))
            .collect()
    }
}

impl<T: Tokenizer> LlmDecoder for VllmTokenizerAdapter<T> {
    fn decode(
        &self,
        token_ids: &[TokenIdType],
        skip_special_tokens: bool,
    ) -> anyhow::Result<String> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

impl<T: Tokenizer + 'static> LlmTokenizer for VllmTokenizerAdapter<T> {
    fn vocab_size(&self) -> usize {
        // vllm's Tokenizer trait does not expose vocab_size; return 0 as a
        // safe default (only used by llm-tokenizer's fingerprinting, not the
        // cache hot path).
        0
    }

    fn get_special_tokens(&self) -> &LlmSpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Public wrapper: llm-tokenizer cached stack → vllm Tokenizer
// ---------------------------------------------------------------------------

/// A tokenizer wrapper that uses `llm-tokenizer`'s [`CachedTokenizer`] (with
/// L0 and optional L1 caches) to accelerate repeated `encode` calls.
///
/// Implements vllm's [`Tokenizer`] trait so it can be used as a drop-in
/// replacement wherever `DynTokenizer` is expected.
pub struct LlmCachedTokenizer {
    /// The llm-tokenizer CachedTokenizer wrapping our adapter.
    cached: CachedTokenizer,
    /// Keep a direct reference to the adapter's inner tokenizer for methods
    /// that should not go through the cache (decode, token_to_id, etc.).
    inner: Arc<dyn Tokenizer>,
}

impl LlmCachedTokenizer {
    /// Create a new cached tokenizer wrapping `inner` with the given config.
    pub fn new<T: Tokenizer + 'static>(inner: T, config: LlmCacheConfig) -> Self {
        let inner_arc: Arc<dyn Tokenizer> = Arc::from(inner);

        // We need to give CachedTokenizer an Arc<dyn llm_tokenizer::Tokenizer>.
        // Create the adapter wrapping a clone of the Arc.
        let adapter = VllmTokenizerAdapter {
            inner: inner_arc.clone(),
            special_tokens: LlmSpecialTokens::default(),
        };
        let llm_inner: Arc<dyn LlmTokenizer> = Arc::new(adapter);
        let cached = CachedTokenizer::new(llm_inner, config.into());

        Self {
            cached,
            inner: inner_arc,
        }
    }

    /// Get L0 cache statistics, if the cache is enabled.
    pub fn cache_stats(&self) -> Option<llm_tokenizer::CacheStats> {
        self.cached.cache_stats()
    }

    /// Get L1 cache statistics, if the cache is enabled.
    pub fn l1_cache_stats(&self) -> Option<llm_tokenizer::cache::L1CacheStats> {
        self.cached.l1_cache_stats()
    }

    /// Clear all cached entries.
    pub fn clear_cache(&self) {
        self.cached.clear_cache();
    }
}

impl Tokenizer for LlmCachedTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let start = std::time::Instant::now();

        // Go through llm-tokenizer's CachedTokenizer which handles L0/L1.
        let encoding = LlmEncoder::encode(&self.cached, text, add_special_tokens)
            .map_err(|e| crate::error::TokenizerError(format!("{e}")))?;

        let token_ids = encoding.token_ids().to_vec();

        let elapsed = start.elapsed();
        tracing::debug!(
            elapsed_us = elapsed.as_micros() as u64,
            text_bytes = text.len(),
            tokens = token_ids.len(),
            "llm-tokenizer cached encode"
        );

        Ok(token_ids)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        // Decode is not cached by llm-tokenizer either — pass through directly.
        self.inner.decode(token_ids, skip_special_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        self.inner.is_special_id(token_id)
    }

    fn create_decode_stream(
        &self,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> Box<dyn IncrementalDecoder + '_> {
        Box::new(DecodeStream::new(
            self,
            prompt_token_ids,
            skip_special_tokens,
            min_bytes_to_buffer,
        ))
    }
}

// We need Tokenizer implemented for Arc<dyn Tokenizer> so the adapter works.
impl Tokenizer for Arc<dyn Tokenizer> {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        (**self).encode(text, add_special_tokens)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        (**self).decode(token_ids, skip_special_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        (**self).token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        (**self).id_to_token(id)
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        (**self).is_special_id(token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test tokenizer: each byte becomes a token ID.
    struct ByteTokenizer;

    impl Tokenizer for ByteTokenizer {
        fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
            Ok(text.bytes().map(|b| b as u32).collect())
        }

        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            let bytes: Vec<u8> = token_ids.iter().map(|&id| id as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }
    }

    #[test]
    fn cache_hit_returns_same_result() {
        let cached = LlmCachedTokenizer::new(ByteTokenizer, LlmCacheConfig::default());

        let r1 = cached.encode("hello", false).unwrap();
        let r2 = cached.encode("hello", false).unwrap();
        assert_eq!(r1, r2);

        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn add_special_tokens_flag_separates_entries() {
        let cached = LlmCachedTokenizer::new(ByteTokenizer, LlmCacheConfig::default());

        let _ = cached.encode("test", false).unwrap();
        let _ = cached.encode("test", true).unwrap();

        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn decode_passes_through() {
        let cached = LlmCachedTokenizer::new(ByteTokenizer, LlmCacheConfig::default());
        assert_eq!(cached.decode(&[72, 105], false).unwrap(), "Hi");
    }

    #[test]
    fn concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let cached = Arc::new(LlmCachedTokenizer::new(
            ByteTokenizer,
            LlmCacheConfig::default(),
        ));
        let mut handles = vec![];

        for i in 0..10 {
            let c = Arc::clone(&cached);
            handles.push(thread::spawn(move || {
                let key = format!("k{i}");
                let r1 = c.encode(&key, false).unwrap();
                let r2 = c.encode(&key, false).unwrap();
                assert_eq!(r1, r2);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.entries, 10);
        assert!(stats.hits >= 10);
    }
}
