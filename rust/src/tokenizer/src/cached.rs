//! Tokenizer caching layer inspired by `llm-tokenizer`'s cache architecture.
//!
//! Provides a [`CachedTokenizer`] wrapper around any [`Tokenizer`] implementation
//! to speed up repeated encoding of the same strings (e.g., system prompts).
//!
//! # Architecture
//!
//! - **L0 Cache**: Whole-string exact match using `DashMap` for lock-free
//!   concurrent reads. Uses approximate LRU eviction (sample + evict oldest)
//!   to protect frequently-accessed entries like system prompts.
//!
//! # Usage
//!
//! ```ignore
//! use std::sync::Arc;
//! use vllm_tokenizer::{CachedTokenizer, CacheConfig};
//!
//! let inner = Arc::new(some_tokenizer);
//! let cached = CachedTokenizer::new(inner, CacheConfig::default());
//! let tokens = cached.encode("Hello world", false)?;
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;

use crate::incremental::DecodeStream;
use crate::{IncrementalDecoder, Result, Tokenizer};

/// Number of entries to sample when looking for an eviction candidate.
const EVICTION_SAMPLE_SIZE: usize = 8;

/// Configuration for the tokenizer cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Enable L0 (whole-string) cache.
    pub enable_l0: bool,
    /// Maximum number of entries in L0 cache.
    pub l0_max_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: true,
            l0_max_entries: 10_000,
        }
    }
}

/// A cached encoding entry with access tracking for approximate LRU eviction.
struct CachedEntry {
    token_ids: Arc<Vec<u32>>,
    last_accessed: AtomicU64,
}

/// L0 cache: whole-string exact match using DashMap for lock-free reads.
///
/// Uses two separate maps (one per `add_special_tokens` value) so that
/// lookups can borrow the key as `&str` without allocating.
///
/// Eviction uses approximate LRU: when capacity is reached, sample a few
/// entries and evict the one with the oldest `last_accessed` timestamp.
struct L0Cache {
    map_plain: DashMap<String, CachedEntry>,
    map_special: DashMap<String, CachedEntry>,
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    access_counter: AtomicU64,
}

impl L0Cache {
    fn new(max_entries: usize) -> Self {
        let per_map = max_entries.min(1024) / 2 + 1;
        Self {
            map_plain: DashMap::with_capacity(per_map),
            map_special: DashMap::with_capacity(per_map),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            access_counter: AtomicU64::new(0),
        }
    }

    #[inline]
    fn map_for(&self, add_special_tokens: bool) -> &DashMap<String, CachedEntry> {
        if add_special_tokens {
            &self.map_special
        } else {
            &self.map_plain
        }
    }

    #[inline]
    fn next_timestamp(&self) -> u64 {
        self.access_counter.fetch_add(1, Ordering::Relaxed)
    }

    fn len(&self) -> usize {
        self.map_plain.len() + self.map_special.len()
    }

    /// Look up a cached encoding. Zero-allocation on the lookup path.
    fn get(&self, key: &str, add_special_tokens: bool) -> Option<Arc<Vec<u32>>> {
        match self.map_for(add_special_tokens).get(key) {
            Some(entry) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                let ts = self.next_timestamp();
                entry.value().last_accessed.store(ts, Ordering::Relaxed);
                Some(Arc::clone(&entry.value().token_ids))
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Evict the approximately least-recently-used entry if at capacity.
    fn maybe_evict(&self) {
        if self.len() >= self.max_entries {
            let victim_map = if self.map_plain.len() >= self.map_special.len() {
                &self.map_plain
            } else {
                &self.map_special
            };

            let key_to_remove = {
                let mut oldest_key: Option<String> = None;
                let mut oldest_ts = u64::MAX;

                for (i, entry) in victim_map.iter().enumerate() {
                    let ts = entry.value().last_accessed.load(Ordering::Relaxed);
                    if ts < oldest_ts {
                        oldest_ts = ts;
                        oldest_key = Some(entry.key().clone());
                    }
                    if i + 1 >= EVICTION_SAMPLE_SIZE {
                        break;
                    }
                }
                oldest_key
            };

            if let Some(k) = key_to_remove {
                victim_map.remove(&k);
            }
        }
    }

    fn insert(&self, key: String, add_special_tokens: bool, token_ids: Vec<u32>) {
        self.maybe_evict();
        let ts = self.next_timestamp();
        let entry = CachedEntry {
            token_ids: Arc::new(token_ids),
            last_accessed: AtomicU64::new(ts),
        };
        self.map_for(add_special_tokens).insert(key, entry);
    }

    fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            hits,
            misses,
            entries: self.len(),
            hit_rate: if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            },
        }
    }

    fn clear(&self) {
        self.map_plain.clear();
        self.map_special.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.access_counter.store(0, Ordering::Relaxed);
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub hit_rate: f64,
}

/// A caching wrapper around any [`Tokenizer`] implementation.
///
/// Caches `encode` results using a DashMap-based L0 whole-string exact-match
/// cache with approximate LRU eviction. Decode and other methods pass through
/// to the inner tokenizer unchanged.
pub struct CachedTokenizer<T: Tokenizer> {
    inner: T,
    l0: Option<L0Cache>,
}

impl<T: Tokenizer> CachedTokenizer<T> {
    /// Create a new cached tokenizer wrapping `inner`.
    pub fn new(inner: T, config: CacheConfig) -> Self {
        let l0 = if config.enable_l0 {
            Some(L0Cache::new(config.l0_max_entries))
        } else {
            None
        };
        Self { inner, l0 }
    }

    /// Get L0 cache statistics, if the cache is enabled.
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.l0.as_ref().map(|c| c.stats())
    }

    /// Clear all cached entries.
    pub fn clear_cache(&self) {
        if let Some(l0) = &self.l0 {
            l0.clear();
        }
    }

    /// Get a reference to the inner tokenizer.
    pub fn inner(&self) -> &T {
        &self.inner
    }
}

impl<T: Tokenizer> Tokenizer for CachedTokenizer<T> {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        if let Some(l0) = &self.l0 {
            if let Some(cached) = l0.get(text, add_special_tokens) {
                return Ok((*cached).clone());
            }
        }

        let token_ids = self.inner.encode(text, add_special_tokens)?;

        if let Some(l0) = &self.l0 {
            l0.insert(text.to_owned(), add_special_tokens, token_ids.clone());
        }

        Ok(token_ids)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        // Decode is not cached — it's fast enough and rarely repeated.
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
        // Decode streaming uses the inner tokenizer directly since the
        // DecodeStream relies on the decode() method which is not cached.
        Box::new(DecodeStream::new(
            self,
            prompt_token_ids,
            skip_special_tokens,
            min_bytes_to_buffer,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;

    /// Simple test tokenizer that splits on whitespace and assigns incrementing IDs.
    struct SimpleTokenizer;

    impl Tokenizer for SimpleTokenizer {
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
        let cached = CachedTokenizer::new(SimpleTokenizer, CacheConfig::default());

        let r1 = cached.encode("hello", false).unwrap();
        let r2 = cached.encode("hello", false).unwrap();
        assert_eq!(r1, r2);

        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn add_special_tokens_flag_separates_entries() {
        let cached = CachedTokenizer::new(SimpleTokenizer, CacheConfig::default());

        let _ = cached.encode("test", false).unwrap();
        let _ = cached.encode("test", true).unwrap();

        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.misses, 2); // Two distinct cache entries
    }

    #[test]
    fn eviction_respects_capacity() {
        let config = CacheConfig {
            enable_l0: true,
            l0_max_entries: 2,
        };
        let cached = CachedTokenizer::new(SimpleTokenizer, config);

        let _ = cached.encode("a", false).unwrap();
        let _ = cached.encode("b", false).unwrap();
        let _ = cached.encode("c", false).unwrap();

        let stats = cached.cache_stats().unwrap();
        assert!(stats.entries <= 2);
    }

    #[test]
    fn lru_eviction_keeps_frequently_accessed() {
        let config = CacheConfig {
            enable_l0: true,
            l0_max_entries: 4,
        };
        let cached = CachedTokenizer::new(SimpleTokenizer, config);

        // Insert system prompt + 3 queries
        let _ = cached.encode("system_prompt", false).unwrap();
        let _ = cached.encode("q1", false).unwrap();
        let _ = cached.encode("q2", false).unwrap();
        let _ = cached.encode("q3", false).unwrap();

        // Simulate: each new request accesses system_prompt then inserts a new query
        for i in 4..12 {
            let _ = cached.encode("system_prompt", false).unwrap(); // hit
            let _ = cached.encode(&format!("q{i}"), false).unwrap(); // miss + evict
        }

        // system_prompt should survive due to LRU
        let stats = cached.cache_stats().unwrap();
        assert!(stats.hits >= 8); // system_prompt was hit each iteration
    }

    #[test]
    fn decode_passes_through() {
        let cached = CachedTokenizer::new(SimpleTokenizer, CacheConfig::default());
        let decoded = cached.decode(&[72, 105], false).unwrap();
        assert_eq!(decoded, "Hi");
    }

    #[test]
    fn cache_disabled_still_works() {
        let config = CacheConfig {
            enable_l0: false,
            l0_max_entries: 0,
        };
        let cached = CachedTokenizer::new(SimpleTokenizer, config);

        let r1 = cached.encode("hello", false).unwrap();
        let r2 = cached.encode("hello", false).unwrap();
        assert_eq!(r1, r2);
        assert!(cached.cache_stats().is_none());
    }

    #[test]
    fn concurrent_access() {
        let cached = Arc::new(CachedTokenizer::new(
            SimpleTokenizer,
            CacheConfig::default(),
        ));
        let mut handles = vec![];

        for i in 0..10 {
            let c = Arc::clone(&cached);
            handles.push(thread::spawn(move || {
                let key = format!("key_{i}");
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

    #[test]
    fn clear_cache_works() {
        let cached = CachedTokenizer::new(SimpleTokenizer, CacheConfig::default());
        let _ = cached.encode("test", false).unwrap();

        assert_eq!(cached.cache_stats().unwrap().entries, 1);
        cached.clear_cache();
        assert_eq!(cached.cache_stats().unwrap().entries, 0);
    }
}
