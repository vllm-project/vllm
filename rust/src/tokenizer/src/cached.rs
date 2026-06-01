//! Tokenizer caching layer inspired by `llm-tokenizer`'s cache architecture.
//!
//! Provides a [`CachedTokenizer`] wrapper around any [`Tokenizer`] implementation
//! to speed up repeated encoding of the same strings (e.g., system prompts).
//!
//! # Architecture
//!
//! - **L0 Cache**: Whole-string exact match using `DashMap` with FxHash for
//!   fast, lock-free concurrent reads. Only caches strings up to a configurable
//!   length threshold — long unique prompts skip the cache entirely to avoid
//!   overhead on the dominant miss path.
//!
//! # Performance notes
//!
//! In vllm's serving path, `encode` is typically called on the **full rendered
//! prompt** (system + user + chat template). Since user messages differ per
//! request, hit rate on full prompts is near zero. This cache is most effective
//! when the tokenizer is called on **repeated segments** (e.g., system prompts
//! encoded separately, stop-word encoding, bad-word encoding).
//!
//! To avoid regressing the dominant miss path:
//! - Strings longer than `l0_max_key_bytes` bypass the cache completely (no
//!   hash, no lookup, no allocation).
//! - FxHash is used instead of SipHash for ~3x faster hashing on short keys.
//! - On miss, the token Vec is moved (not cloned) into the cache.
//! - Stats counters are omitted from the hot path; use [`CachedTokenizer::cache_stats`]
//!   for diagnostics only.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use rustc_hash::FxBuildHasher;

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
    /// Maximum key length in bytes. Strings longer than this bypass the cache
    /// entirely — no hash, no lookup, no allocation. This avoids adding
    /// overhead to the dominant miss path (long unique prompts).
    ///
    /// Default: 2048 bytes (~500 tokens of English text).
    pub l0_max_key_bytes: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: true,
            l0_max_entries: 10_000,
            l0_max_key_bytes: 2048,
        }
    }
}

/// A cached encoding entry with insertion timestamp for approximate LRU.
struct CachedEntry {
    /// Cached token IDs, shared via Arc to avoid cloning on hit when possible
    /// in future trait extensions. Currently the trait requires Vec<u32>, so
    /// we call `.to_vec()` on the inner slice.
    token_ids: Arc<[u32]>,
    /// Monotonic timestamp of last access (for LRU eviction).
    last_accessed: AtomicU64,
}

/// L0 cache: whole-string exact match using DashMap with FxHash.
///
/// Two separate maps (one per `add_special_tokens` value) so lookups borrow
/// `&str` without allocating.
struct L0Cache {
    map_plain: DashMap<String, CachedEntry, FxBuildHasher>,
    map_special: DashMap<String, CachedEntry, FxBuildHasher>,
    max_entries: usize,
    max_key_bytes: usize,
    /// Monotonic counter for LRU timestamps.
    access_counter: AtomicU64,
    /// Stats — updated with Relaxed ordering, read only via `stats()`.
    hits: AtomicU64,
    misses: AtomicU64,
    skips: AtomicU64,
}

impl L0Cache {
    fn new(max_entries: usize, max_key_bytes: usize) -> Self {
        let per_map = max_entries.min(1024) / 2 + 1;
        Self {
            map_plain: DashMap::with_capacity_and_hasher(per_map, FxBuildHasher),
            map_special: DashMap::with_capacity_and_hasher(per_map, FxBuildHasher),
            max_entries,
            max_key_bytes,
            access_counter: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            skips: AtomicU64::new(0),
        }
    }

    #[inline]
    fn map_for(&self, add_special_tokens: bool) -> &DashMap<String, CachedEntry, FxBuildHasher> {
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

    /// Returns true if the key is eligible for caching.
    #[inline]
    fn is_cacheable(&self, key: &str) -> bool {
        key.len() <= self.max_key_bytes
    }

    /// Look up a cached encoding. Returns Arc to avoid cloning until necessary.
    #[inline]
    fn get(&self, key: &str, add_special_tokens: bool) -> Option<Arc<[u32]>> {
        let entry = self.map_for(add_special_tokens).get(key)?;
        let ts = self.next_timestamp();
        entry.value().last_accessed.store(ts, Ordering::Relaxed);
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(Arc::clone(&entry.value().token_ids))
    }

    fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    fn record_skip(&self) {
        self.skips.fetch_add(1, Ordering::Relaxed);
    }

    /// Evict the approximately least-recently-used entry if at capacity.
    fn maybe_evict(&self) {
        if self.len() < self.max_entries {
            return;
        }
        let victim_map = if self.map_plain.len() >= self.map_special.len() {
            &self.map_plain
        } else {
            &self.map_special
        };

        // Sample EVICTION_SAMPLE_SIZE entries, evict the oldest.
        // Scope the iterator so shard locks are released before remove().
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

    /// Insert token_ids into the cache. Consumes the Vec (no clone).
    fn insert(&self, key: String, add_special_tokens: bool, token_ids: Vec<u32>) {
        self.maybe_evict();
        let ts = self.next_timestamp();
        let entry = CachedEntry {
            token_ids: Arc::from(token_ids),
            last_accessed: AtomicU64::new(ts),
        };
        self.map_for(add_special_tokens).insert(key, entry);
    }

    fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let skips = self.skips.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            hits,
            misses,
            skips,
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
        self.skips.store(0, Ordering::Relaxed);
        self.access_counter.store(0, Ordering::Relaxed);
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses (key not found but was looked up).
    pub misses: u64,
    /// Number of skipped lookups (key too long, bypassed cache entirely).
    pub skips: u64,
    /// Current number of cached entries.
    pub entries: usize,
    /// Hit rate = hits / (hits + misses). Skips are excluded.
    pub hit_rate: f64,
}

/// A caching wrapper around any [`Tokenizer`] implementation.
///
/// Caches `encode` results using a DashMap-based L0 whole-string exact-match
/// cache with FxHash and approximate LRU eviction. Strings longer than
/// [`CacheConfig::l0_max_key_bytes`] bypass the cache entirely.
///
/// Decode and other methods pass through to the inner tokenizer unchanged.
pub struct CachedTokenizer<T: Tokenizer> {
    inner: T,
    l0: Option<L0Cache>,
}

impl<T: Tokenizer> CachedTokenizer<T> {
    /// Create a new cached tokenizer wrapping `inner`.
    pub fn new(inner: T, config: CacheConfig) -> Self {
        let l0 = if config.enable_l0 {
            Some(L0Cache::new(config.l0_max_entries, config.l0_max_key_bytes))
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
        let Some(l0) = &self.l0 else {
            return self.inner.encode(text, add_special_tokens);
        };

        // Skip cache for long strings — they're almost always unique full
        // prompts. Avoiding hash + lookup + key allocation on this path is
        // critical since it's the dominant case in serving.
        if !l0.is_cacheable(text) {
            l0.record_skip();
            return self.inner.encode(text, add_special_tokens);
        }

        // Cache hit — return a copy of the cached slice.
        if let Some(cached) = l0.get(text, add_special_tokens) {
            return Ok(cached.to_vec());
        }

        // Cache miss — encode, move result into cache, return from cache.
        l0.record_miss();
        let token_ids = self.inner.encode(text, add_special_tokens)?;
        // Clone the key string and move token_ids into the cache.
        // We keep a copy to return to the caller.
        let result = token_ids.clone();
        l0.insert(text.to_owned(), add_special_tokens, token_ids);
        Ok(result)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

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
        let cached = CachedTokenizer::new(ByteTokenizer, CacheConfig::default());

        let r1 = cached.encode("hello", false).unwrap();
        let r2 = cached.encode("hello", false).unwrap();
        assert_eq!(r1, r2);

        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn add_special_tokens_flag_separates_entries() {
        let cached = CachedTokenizer::new(ByteTokenizer, CacheConfig::default());

        let _ = cached.encode("test", false).unwrap();
        let _ = cached.encode("test", true).unwrap();

        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn long_strings_bypass_cache() {
        let config = CacheConfig {
            enable_l0: true,
            l0_max_entries: 100,
            l0_max_key_bytes: 10, // very short threshold
        };
        let cached = CachedTokenizer::new(ByteTokenizer, config);

        // This string is > 10 bytes, should skip cache entirely.
        let long = "this is a long string that exceeds the threshold";
        let r1 = cached.encode(long, false).unwrap();
        let r2 = cached.encode(long, false).unwrap();
        assert_eq!(r1, r2);

        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.skips, 2);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.entries, 0);
    }

    #[test]
    fn short_strings_use_cache() {
        let config = CacheConfig {
            enable_l0: true,
            l0_max_entries: 100,
            l0_max_key_bytes: 100,
        };
        let cached = CachedTokenizer::new(ByteTokenizer, config);

        let _ = cached.encode("hi", false).unwrap();
        let _ = cached.encode("hi", false).unwrap();

        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);
    }

    #[test]
    fn eviction_respects_capacity() {
        let config = CacheConfig {
            enable_l0: true,
            l0_max_entries: 2,
            l0_max_key_bytes: 1024,
        };
        let cached = CachedTokenizer::new(ByteTokenizer, config);

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
            l0_max_key_bytes: 1024,
        };
        let cached = CachedTokenizer::new(ByteTokenizer, config);

        let _ = cached.encode("sys", false).unwrap();
        let _ = cached.encode("q1", false).unwrap();
        let _ = cached.encode("q2", false).unwrap();
        let _ = cached.encode("q3", false).unwrap();

        for i in 4..12 {
            let _ = cached.encode("sys", false).unwrap(); // keep sys hot
            let _ = cached.encode(&format!("q{i}"), false).unwrap();
        }

        let stats = cached.cache_stats().unwrap();
        assert!(stats.hits >= 8);
    }

    #[test]
    fn decode_passes_through() {
        let cached = CachedTokenizer::new(ByteTokenizer, CacheConfig::default());
        assert_eq!(cached.decode(&[72, 105], false).unwrap(), "Hi");
    }

    #[test]
    fn cache_disabled_still_works() {
        let config = CacheConfig {
            enable_l0: false,
            l0_max_entries: 0,
            l0_max_key_bytes: 0,
        };
        let cached = CachedTokenizer::new(ByteTokenizer, config);
        let r1 = cached.encode("hello", false).unwrap();
        let r2 = cached.encode("hello", false).unwrap();
        assert_eq!(r1, r2);
        assert!(cached.cache_stats().is_none());
    }

    #[test]
    fn concurrent_access() {
        let cached = Arc::new(CachedTokenizer::new(ByteTokenizer, CacheConfig::default()));
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

    #[test]
    fn clear_cache_works() {
        let cached = CachedTokenizer::new(ByteTokenizer, CacheConfig::default());
        let _ = cached.encode("test", false).unwrap();
        assert_eq!(cached.cache_stats().unwrap().entries, 1);
        cached.clear_cache();
        assert_eq!(cached.cache_stats().unwrap().entries, 0);
    }
}
