// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Frontend-side wrapper over the SHM multimodal processor cache.
//!
//! Mirrors Python's `ShmObjectStoreSenderCache`
//! (`vllm/multimodal/cache/shm.py`): the shm cache stores preprocessed items;
//! this wrapper additionally shadows each cached hash's `PromptReplacement`
//! (Python's `_p0_cache`) so cache hits skip re-preprocessing entirely.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use llm_multimodal::{PromptReplacement, VisionPreprocessingContext};
use parking_lot::Mutex;
use vllm_engine_core_client::mm_cache::MmProcessorShmCache;
use vllm_engine_core_client::protocol::multimodal::MmKwargsItem;

use super::{PreparedItem, VisionModalitySupport};

/// Vision processors whose preprocessing output depends on the per-request
/// `VisionPreprocessingContext` (today: only Nemotron-H Omni's
/// `token_budget`-driven dynamic resolution). Their cache entries must key on
/// the budget in addition to the content hash.
///
/// TODO: drop this name list once `llm-multimodal` exposes a capability flag
/// on `VisionPreProcessor` for request-context sensitivity.
const BUDGET_SENSITIVE_PROCESSORS: &[&str] = &["nemotron_h_omni"];

/// Cache key for one image item: the content hash, plus the per-request token
/// budget for budget-sensitive processors so requests with different budgets
/// never share processed tensors or prompt replacements.
pub(super) fn image_cache_key<'a>(
    support: &VisionModalitySupport,
    hash: &'a str,
    context: &VisionPreprocessingContext,
) -> Cow<'a, str> {
    if BUDGET_SENSITIVE_PROCESSORS.contains(&support.processor.model_name()) {
        Cow::Owned(format!("{hash}#budget={:?}", context.token_budget))
    } else {
        Cow::Borrowed(hash)
    }
}

/// The P0 (frontend) side of the SHM multimodal processor cache.
///
/// How to update each item:
///
/// - If the item is already in the cache, send its shm address instead of the
///   data to avoid unnecessary IPC, and reuse the shadowed prompt replacement
///   to skip re-preprocessing.
///
/// - If the item is not in the cache, preprocess it, store the data in shared
///   memory, and shadow its prompt replacement.
#[derive(Clone)]
pub(crate) struct MmProcessorCache {
    shm: Arc<MmProcessorShmCache>,
    /// Prompt replacements of cached items, mirroring Python's `_p0_cache`.
    ///
    /// Entries whose shm data was evicted (or cleared by `/reset_mm_cache`,
    /// which only clears the shm side) dangle harmlessly: `lookup` verifies
    /// the shm entry still exists and degrades to a miss otherwise, and
    /// dangling entries are pruned on later puts (Python's
    /// `remove_dangling_items`).
    shadow: Arc<Mutex<HashMap<String, PromptReplacement>>>,
}

/// Outcome of a per-item cache lookup.
pub(super) enum CacheLookup {
    /// Cache hit: the address item plus the shadowed prompt replacement.
    Hit(PreparedItem, PromptReplacement),
    /// Cache miss: preprocess on the miss path. Carries back the request UUID
    /// passed to `lookup`.
    Miss(Option<String>),
}

impl MmProcessorCache {
    pub(crate) fn new(shm: Arc<MmProcessorShmCache>) -> Self {
        Self {
            shm,
            shadow: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Touch the item in shared memory cache to prevent eviction.
    /// Increments writer_flag on sender side.
    pub(super) fn touch(&self, hash: &str) {
        self.shm.touch(hash);
    }

    /// Look up one fetched media item by cache key.
    ///
    /// On a hit, bumps writer_flag and returns the address item plus the
    /// shadowed prompt replacement. On a miss — including a shadowed entry
    /// whose shm data vanished under a concurrent eviction — returns the UUID
    /// back for the preprocessing path.
    ///
    /// `key` is the cache key (see [`image_cache_key`]); `hash` is the raw
    /// content hash recorded on the returned [`PreparedItem`] so the engine
    /// still sees the content-addressed `mm_hash`.
    pub(super) fn lookup(&self, key: &str, hash: &str, uuid: Option<String>) -> CacheLookup {
        let replacement = self.shadow.lock().get(key).cloned();
        let Some(replacement) = replacement else {
            return CacheLookup::Miss(uuid);
        };
        match self.shm.address_item_for_cached(key) {
            Some(data) => CacheLookup::Hit(
                PreparedItem {
                    data,
                    hash: hash.to_owned(),
                    uuid,
                },
                replacement,
            ),
            None => CacheLookup::Miss(uuid),
        }
    }

    /// Put a freshly preprocessed item into shm and shadow its prompt
    /// replacement.
    ///
    /// Oversize, buffer-full, or duplicate-key puts return the item unchanged
    /// for inline sending and shadow nothing (Python's `get_and_update_item`
    /// fallback).
    pub(super) fn put_or_inline_and_track(
        &self,
        hash: &str,
        replacement: &PromptReplacement,
        data: MmKwargsItem,
    ) -> MmKwargsItem {
        let data = self.shm.put_or_inline(hash, data);
        if !self.shm.is_cached(hash) {
            return data;
        }

        let mut shadow = self.shadow.lock();
        // Try to remove dangling items if the shadow is too large (Python:
        // `len(self._p0_cache) >= 2 * len(self._shm_cache.key_index)`, pruned
        // by `remove_dangling_items`).
        if shadow.len() >= 2 * self.shm.cached_len() {
            shadow.retain(|key, _| self.shm.is_cached(key));
        }
        shadow.insert(hash.to_owned(), replacement.clone());
        data
    }
}

/// Put a miss item into the SHM cache when enabled, keeping the inline data
/// otherwise.
pub(super) fn put_or_inline_and_track(
    cache: Option<&MmProcessorCache>,
    hash: &str,
    replacement: &PromptReplacement,
    data: MmKwargsItem,
) -> MmKwargsItem {
    match cache {
        Some(cache) => cache.put_or_inline_and_track(hash, replacement, data),
        None => data,
    }
}

/// Reassemble per-modality replacements and items in request order from
/// cache-hit slots and freshly preprocessed misses.
///
/// `slots` has one entry per media item: `Some` for hits, `None` for misses.
/// `misses` carries one prepared item + replacement per `None` slot, in slot
/// order, so the result stays index-aligned for placeholder expansion.
pub(super) fn merge_slots(
    slots: Vec<Option<(PreparedItem, PromptReplacement)>>,
    misses: Vec<(PreparedItem, PromptReplacement)>,
) -> (Vec<PromptReplacement>, Vec<PreparedItem>) {
    let mut misses = misses.into_iter();
    let mut replacements = Vec::with_capacity(slots.len());
    let mut items = Vec::with_capacity(slots.len());
    for slot in slots {
        let (item, replacement) = match slot {
            Some(hit) => hit,
            None => misses.next().expect("one prepared item per cache-miss slot"),
        };
        items.push(item);
        replacements.push(replacement);
    }
    debug_assert!(misses.next().is_none());
    (replacements, items)
}

#[cfg(test)]
pub(super) mod tests {
    use vllm_engine_core_client::mm_cache::{MmProcessorShmCache, MmShmCacheConfig};
    use vllm_engine_core_client::protocol::multimodal::{MmKwargValue, MmKwargsItem};

    use super::*;

    /// 64x64 solid-color PNG.
    const TEST_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAeklEQVR4nO3PUQkAIBTAwBfNaEYzmiH8OITBAtxm7fN1wwUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWPHYBHsYBafyS08sAAAAASUVORK5CYII=";

    fn test_png_bytes() -> Vec<u8> {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(TEST_PNG_BASE64)
            .expect("test PNG should decode")
    }

    pub(crate) fn test_shm_cache() -> Arc<MmProcessorShmCache> {
        let cache = MmProcessorShmCache::create(MmShmCacheConfig {
            shm_name: MmShmCacheConfig::unique_shm_name(),
            data_buffer_size: 16 * 1024 * 1024,
            max_object_size: 4 * 1024 * 1024,
        })
        .expect("shm cache should be created");
        cache.set_n_readers(1);
        Arc::new(cache)
    }

    pub(crate) fn is_address_item(item: &MmKwargsItem) -> bool {
        item.len() == 2 && item.contains_key("address") && item.contains_key("monotonic_id")
    }

    /// The `(address, monotonic_id)` pair of an address item.
    pub(crate) fn address_of(item: &MmKwargsItem) -> (i64, i64) {
        let get = |key: &str| match item[key].data {
            Some(MmKwargValue::Int(value)) => value,
            _ => panic!("address item `{key}` should be an int"),
        };
        (get("address"), get("monotonic_id"))
    }

    fn test_replacement() -> PromptReplacement {
        PromptReplacement::repeated(llm_multimodal::Modality::Image, "<|image_pad|>", 151_655, 2)
    }

    fn test_item(value: i64) -> MmKwargsItem {
        MmKwargsItem::from([(
            "pixel_values".to_string(),
            vllm_engine_core_client::protocol::multimodal::MmFieldElem {
                data: Some(MmKwargValue::Int(value)),
                field: vllm_engine_core_client::protocol::multimodal::MmField::Batched(
                    vllm_engine_core_client::protocol::multimodal::MmBatchedField {
                        keep_on_cpu: false,
                    },
                ),
            },
        )])
    }

    #[test]
    fn cache_is_inert_until_n_readers_is_set() {
        let shm = MmProcessorShmCache::create(MmShmCacheConfig {
            shm_name: MmShmCacheConfig::unique_shm_name(),
            data_buffer_size: 1024 * 1024,
            max_object_size: 256 * 1024,
        })
        .expect("shm cache should be created");
        let cache = MmProcessorCache::new(Arc::new(shm));

        cache.touch("hash-a");
        let item = test_item(7);
        let sent = cache.put_or_inline_and_track("hash-a", &test_replacement(), item.clone());
        assert_eq!(sent, item, "before set_n_readers the item goes inline");
        assert!(
            matches!(
                cache.lookup("hash-a", "hash-a", None),
                CacheLookup::Miss(None)
            ),
            "before set_n_readers nothing is cached"
        );
    }

    #[test]
    fn hit_reuses_shadowed_replacement_and_address() {
        let cache = MmProcessorCache::new(test_shm_cache());

        let item = test_item(11);
        let sent = cache.put_or_inline_and_track("hash-b", &test_replacement(), item);
        assert!(
            is_address_item(&sent),
            "a fresh put yields the address item"
        );

        let CacheLookup::Hit(hit, replacement) =
            cache.lookup("hash-b", "hash-b", Some("uuid-1".into()))
        else {
            panic!("hash-b should be cached");
        };
        assert_eq!(replacement.tokens, test_replacement().tokens);
        assert_eq!(hit.hash, "hash-b");
        assert_eq!(hit.uuid.as_deref(), Some("uuid-1"));
        assert_eq!(address_of(&hit.data), address_of(&sent));
    }

    #[test]
    fn lookup_misses_for_unknown_hash_and_returns_uuid() {
        let cache = MmProcessorCache::new(test_shm_cache());

        assert!(matches!(
            cache.lookup("hash-c", "hash-c", Some("uuid-2".into())),
            CacheLookup::Miss(Some(uuid)) if uuid == "uuid-2"
        ));
    }

    #[test]
    fn merge_slots_restores_request_order() {
        let miss = |value: i64| PreparedItem {
            data: test_item(value),
            hash: format!("hash-{value}"),
            uuid: None,
        };
        let slots = vec![
            Some((miss(1), test_replacement())),
            None,
            Some((miss(3), test_replacement())),
            None,
        ];
        let misses = vec![(miss(2), test_replacement()), (miss(4), test_replacement())];

        let (replacements, items) = merge_slots(slots, misses);

        let hashes: Vec<&str> = items.iter().map(|item| item.hash.as_str()).collect();
        assert_eq!(hashes, ["hash-1", "hash-2", "hash-3", "hash-4"]);
        assert_eq!(replacements.len(), 4);
    }

    /// End-to-end over `prepare_multimodal`: the first prepare preprocesses
    /// and puts into shm; the second prepare of identical content must skip
    /// preprocessing — a re-put of the same hash would hit the duplicate-key
    /// inline fallback, so an address item with the *same* address proves the
    /// hit path ran.
    #[tokio::test]
    async fn image_cache_hit_skips_preprocessing_and_reuses_replacement() {
        use llm_multimodal::MediaContentPart;
        use vllm_engine_core_client::protocol::dtype::ModelDtype;

        use super::super::tests::{QWEN3_IMAGE_PAD_ID, qwen3_vl_info};

        let cache = test_shm_cache();
        let info = qwen3_vl_info().with_mm_processor_cache(Some(cache));
        let media_part = || MediaContentPart::ImageData {
            data: test_png_bytes(),
            mime_type: Some("image/png".to_string()),
            uuid: None,
            detail: None,
        };

        let mut tokens_first = vec![1, QWEN3_IMAGE_PAD_ID, 2];
        let first = info
            .prepare_multimodal(
                vec![media_part()],
                &mut tokens_first,
                ModelDtype::Float32,
                None,
            )
            .await
            .expect("first image prepare should succeed");
        assert_eq!(first.len(), 1);
        let data_first = first[0].data.as_ref().expect("feature data");
        assert!(
            is_address_item(data_first),
            "first prepare should put the item into shm and send the address item"
        );

        let mut tokens_second = vec![1, QWEN3_IMAGE_PAD_ID, 2];
        let second = info
            .prepare_multimodal(
                vec![media_part()],
                &mut tokens_second,
                ModelDtype::Float32,
                None,
            )
            .await
            .expect("second image prepare should succeed");
        assert_eq!(second.len(), 1);
        let data_second = second[0].data.as_ref().expect("feature data");
        assert!(
            is_address_item(data_second),
            "second prepare should hit the cache and send the address item"
        );
        assert_eq!(
            address_of(data_second),
            address_of(data_first),
            "a cache hit reuses the same shm slot instead of re-putting"
        );

        // The shadowed prompt replacement reproduces the expansion exactly.
        assert_eq!(tokens_second, tokens_first);
        let (first, second) = (&first[0], &second[0]);
        assert_eq!(first.mm_hash, second.mm_hash);
        assert_eq!(first.identifier, second.identifier);
        assert_eq!(first.mm_position.offset, second.mm_position.offset);
        assert_eq!(first.mm_position.length, second.mm_position.length);
    }
}
