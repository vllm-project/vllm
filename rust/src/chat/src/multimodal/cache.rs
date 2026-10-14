// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Frontend-side wrapper over the SHM multimodal processor cache.
//!
//! Mirrors Python's `ShmObjectStoreSenderCache`
//! (`vllm/multimodal/cache/shm.py`): the shm cache stores preprocessed items;
//! this wrapper additionally shadows each cached hash's `PromptReplacement`
//! (Python's `_p0_cache`) so cache hits skip re-preprocessing entirely.

use std::collections::HashMap;
use std::sync::Arc;

use llm_multimodal::PromptReplacement;
use parking_lot::Mutex;
use vllm_engine_core_client::mm_cache::MmProcessorShmCache;
use vllm_engine_core_client::protocol::multimodal::MmKwargsItem;

use super::{PreparedItem, VisionModalitySupport};

/// Vision processors whose preprocessing output depends on the per-request
/// `VisionPreprocessingContext` (today: only Nemotron-H Omni's
/// `token_budget`-driven dynamic resolution). Their per-request budget varies,
/// so cache hits would be rare anyway: such processors bypass the processor
/// cache entirely and always preprocess inline.
///
/// TODO: drop this name list once `llm-multimodal` exposes a capability flag
/// on `VisionPreProcessor` for request-context sensitivity.
pub(super) const BUDGET_SENSITIVE_PROCESSORS: &[&str] = &["nemotron_h_omni"];

/// Whether a vision processor's output depends on the per-request
/// preprocessing context, in which case its items bypass the SHM cache.
pub(super) fn is_budget_sensitive(support: &VisionModalitySupport) -> bool {
    BUDGET_SENSITIVE_PROCESSORS.contains(&support.processor.model_name())
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
    ///
    /// The item stays protected until it is looked up (`get_cached`) or
    /// `release_touches` is called at the end of the request's merge.
    pub(super) fn touch(&self, hash: &str) {
        self.shm.touch(hash);
    }

    /// Stop protecting touched items that were not looked up.
    pub(super) fn release_touches(&self) {
        self.shm.release_touches();
    }

    /// Guard releasing this cache's touch protections on drop, mirroring
    /// Python's `finally: release_sender_touches()` after each request's
    /// merge: early `?` returns release the protections too.
    pub(super) fn touches_guard(&self) -> TouchesGuard<'_> {
        TouchesGuard(self)
    }

    /// Look up one fetched media item by cache key.
    ///
    /// The cache key is the feature-facing `mm_hash` itself: the client UUID
    /// when supplied, otherwise the content hash.
    ///
    /// On a hit, bumps writer_flag and returns the address item plus the
    /// shadowed prompt replacement. On a miss — including a shadowed entry
    /// whose shm data vanished under a concurrent eviction — returns the UUID
    /// back for the preprocessing path.
    pub(super) fn lookup(&self, key: &str, uuid: Option<String>) -> CacheLookup {
        let replacement = self.shadow.lock().get(key).cloned();
        let Some(replacement) = replacement else {
            return CacheLookup::Miss(uuid);
        };
        match self.shm.address_item_for_cached(key) {
            Some(data) => CacheLookup::Hit(
                PreparedItem {
                    data,
                    hash: key.to_owned(),
                    uuid,
                },
                replacement,
            ),
            None => CacheLookup::Miss(uuid),
        }
    }

    /// Look up a UUID-keyed item before its media is fetched.
    ///
    /// UUID hits never pass through the post-fetch pre-touch, so this touches
    /// the key itself: with the new touch semantics the item is protected from
    /// eviction until the `get_cached` inside `address_item_for_cached` (or
    /// the request-end `release_touches`), so a hit cannot be evicted before
    /// the engine reads it.
    ///
    /// On a hit the returned [`PreparedItem`] carries the UUID as its `hash`
    /// (Python parity: `hashes.append(uuid_item)`).
    pub(super) fn lookup_uuid(
        &self,
        key: &str,
        uuid: &str,
    ) -> Option<(PreparedItem, PromptReplacement)> {
        let replacement = self.shadow.lock().get(key).cloned()?;
        self.touch(key);
        let data = self.shm.address_item_for_cached(key)?;
        Some((
            PreparedItem {
                data,
                hash: uuid.to_owned(),
                uuid: Some(uuid.to_owned()),
            },
            replacement,
        ))
    }

    /// Put a freshly preprocessed item into shm and shadow its prompt
    /// replacement.
    ///
    /// `key` is the cache key, equal to the feature-facing `mm_hash` the
    /// issued handle is signed over.
    ///
    /// Oversize, buffer-full, or duplicate-key puts return the item unchanged
    /// for inline sending and shadow nothing (Python's `get_and_update_item`
    /// fallback).
    pub(super) fn put_or_inline_and_track(
        &self,
        key: &str,
        replacement: &PromptReplacement,
        data: MmKwargsItem,
    ) -> MmKwargsItem {
        let data = self.shm.put_or_inline(key, data);
        if !self.shm.is_cached(key) {
            return data;
        }

        let mut shadow = self.shadow.lock();
        // Try to remove dangling items if the shadow is too large (Python:
        // `len(self._p0_cache) >= 2 * len(self._shm_cache.key_index)`, pruned
        // by `remove_dangling_items`).
        if shadow.len() >= 2 * self.shm.cached_len() {
            shadow.retain(|key, _| self.shm.is_cached(key));
        }
        shadow.insert(key.to_owned(), replacement.clone());
        data
    }
}

/// RAII guard releasing a cache's touch protections on drop; see
/// [`MmProcessorCache::touches_guard`].
pub(super) struct TouchesGuard<'a>(&'a MmProcessorCache);

impl Drop for TouchesGuard<'_> {
    fn drop(&mut self) {
        self.0.release_touches();
    }
}

/// Put a miss item into the SHM cache when enabled, keeping the inline data
/// otherwise.
pub(super) fn put_or_inline_and_track(
    cache: Option<&MmProcessorCache>,
    key: &str,
    replacement: &PromptReplacement,
    data: MmKwargsItem,
) -> MmKwargsItem {
    match cache {
        Some(cache) => cache.put_or_inline_and_track(key, replacement, data),
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
pub(crate) mod tests {
    use vllm_engine_core_client::mm_cache::{MmProcessorShmCache, MmShmCacheConfig};
    use vllm_engine_core_client::protocol::multimodal::{MmKwargValue, MmKwargsItem};

    use super::*;

    /// 64x64 solid-color PNG.
    pub(crate) const TEST_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAeklEQVR4nO3PUQkAIBTAwBfNaEYzmiH8OITBAtxm7fN1wwUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWPHYBHsYBafyS08sAAAAASUVORK5CYII=";

    pub(crate) fn test_png_bytes() -> Vec<u8> {
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
        item.len() == 3
            && item.contains_key("address")
            && item.contains_key("monotonic_id")
            && item.contains_key("signature")
    }

    /// The `(address, monotonic_id)` pair of an address item.
    pub(crate) fn address_of(item: &MmKwargsItem) -> (i64, i64) {
        let get = |key: &str| match item[key].data {
            Some(MmKwargValue::Int(value)) => value,
            _ => panic!("address item `{key}` should be an int"),
        };
        (get("address"), get("monotonic_id"))
    }

    /// The 32-byte handle signature of an address item.
    pub(crate) fn signature_of(item: &MmKwargsItem) -> Vec<i64> {
        match item["signature"].data.as_ref() {
            Some(MmKwargValue::List(bytes)) => bytes
                .iter()
                .map(|value| match value {
                    MmKwargValue::Int(byte) => *byte,
                    _ => panic!("signature bytes should be ints"),
                })
                .collect(),
            _ => panic!("address item `signature` should be a list"),
        }
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
            matches!(cache.lookup("hash-a", None), CacheLookup::Miss(None)),
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

        let CacheLookup::Hit(hit, replacement) = cache.lookup("hash-b", Some("uuid-1".into()))
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
            cache.lookup("hash-c", Some("uuid-2".into())),
            CacheLookup::Miss(Some(uuid)) if uuid == "uuid-2"
        ));
    }

    /// Address items carry the 3-field signed handle `(address, monotonic_id,
    /// signature)`; a hit reuses the exact handle issued at put time.
    #[test]
    fn address_items_carry_a_signed_handle() {
        let cache = MmProcessorCache::new(test_shm_cache());

        let sent = cache.put_or_inline_and_track("hash-s", &test_replacement(), test_item(3));
        assert!(is_address_item(&sent));
        let signature = signature_of(&sent);
        assert_eq!(signature.len(), 32, "HMAC-SHA256 signature is 32 bytes");
        assert!(signature.iter().all(|byte| (0..=255).contains(byte)));

        let CacheLookup::Hit(hit, _) = cache.lookup("hash-s", None) else {
            panic!("hash-s should be cached");
        };
        assert!(is_address_item(&hit.data));
        assert_eq!(signature_of(&hit.data), signature);
    }

    /// With the new touch semantics, a touch protects the item from eviction
    /// until the request-end guard drops (Python's `finally:
    /// release_sender_touches()`) or the item is looked up.
    #[test]
    fn touches_protect_until_guard_drop_or_lookup() {
        let shm = test_shm_cache();
        let cache = MmProcessorCache::new(Arc::clone(&shm));
        let sent = cache.put_or_inline_and_track("hash-t", &test_replacement(), test_item(5));
        assert!(is_address_item(&sent));

        let guard = cache.touches_guard();
        cache.touch("hash-t");
        assert_eq!(shm.touched_len(), 1);
        drop(guard);
        assert_eq!(shm.touched_len(), 0);

        cache.touch("hash-t");
        assert_eq!(shm.touched_len(), 1);
        assert!(matches!(cache.lookup("hash-t", None), CacheLookup::Hit(..)));
        assert_eq!(shm.touched_len(), 0);
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
        let info = qwen3_vl_info().with_mm_processor_cache(Some(Arc::clone(&cache)));
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
                true,
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
                true,
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

        // Both prepares released their touch protections on completion.
        assert_eq!(cache.touched_len(), 0);
    }

    /// A UUID-carrying item is keyed by its UUID (Python's `get_mm_hashes`):
    /// a second request with the same UUID and EMPTY content must succeed
    /// without any fetch, decode, or preprocessing — an empty URL would fail
    /// in the connector if a fetch were attempted.
    #[tokio::test]
    async fn uuid_hit_skips_fetch_decode_and_preprocessing() {
        use llm_multimodal::MediaContentPart;
        use vllm_engine_core_client::protocol::dtype::ModelDtype;

        use super::super::tests::{QWEN3_IMAGE_PAD_ID, qwen3_vl_info};

        let cache = test_shm_cache();
        let info = qwen3_vl_info().with_mm_processor_cache(Some(cache));

        let mut tokens_first = vec![1, QWEN3_IMAGE_PAD_ID, 2];
        let first = info
            .prepare_multimodal(
                vec![MediaContentPart::ImageData {
                    data: test_png_bytes(),
                    mime_type: Some("image/png".to_string()),
                    uuid: Some("img-uuid-1".to_string()),
                    detail: None,
                }],
                &mut tokens_first,
                ModelDtype::Float32,
                None,
                true,
            )
            .await
            .expect("first image prepare should succeed");
        assert_eq!(first.len(), 1);
        // A UUID-keyed item reports the UUID as `mm_hash` and `identifier`.
        assert_eq!(first[0].mm_hash.as_deref(), Some("img-uuid-1"));
        assert_eq!(first[0].identifier, "img-uuid-1");
        let data_first = first[0].data.as_ref().expect("feature data");
        assert!(is_address_item(data_first));

        let mut tokens_second = vec![1, QWEN3_IMAGE_PAD_ID, 2];
        let second = info
            .prepare_multimodal(
                vec![MediaContentPart::ImageUrl {
                    url: String::new(),
                    detail: None,
                    uuid: Some("img-uuid-1".to_string()),
                }],
                &mut tokens_second,
                ModelDtype::Float32,
                None,
                true,
            )
            .await
            .expect("a UUID cache reference should resolve without fetching");
        assert_eq!(second.len(), 1);
        let data_second = second[0].data.as_ref().expect("feature data");
        assert!(is_address_item(data_second));
        assert_eq!(address_of(data_second), address_of(data_first));

        assert_eq!(tokens_second, tokens_first);
        let (first, second) = (&first[0], &second[0]);
        assert_eq!(first.mm_hash, second.mm_hash);
        assert_eq!(first.identifier, second.identifier);
        assert_eq!(first.mm_position.offset, second.mm_position.offset);
        assert_eq!(first.mm_position.length, second.mm_position.length);
    }

    /// uuid + content with a cache hit skips the fetch too: the UUID contract
    /// (same UUID ⟹ same content) makes this safe, so an unreachable URL must
    /// still succeed.
    #[tokio::test]
    async fn uuid_hit_with_unreachable_url_skips_fetch() {
        use llm_multimodal::MediaContentPart;
        use vllm_engine_core_client::protocol::dtype::ModelDtype;

        use super::super::tests::{QWEN3_IMAGE_PAD_ID, qwen3_vl_info};

        let cache = test_shm_cache();
        let info = qwen3_vl_info().with_mm_processor_cache(Some(cache));

        let mut tokens_first = vec![1, QWEN3_IMAGE_PAD_ID, 2];
        let first = info
            .prepare_multimodal(
                vec![MediaContentPart::ImageData {
                    data: test_png_bytes(),
                    mime_type: Some("image/png".to_string()),
                    uuid: Some("img-uuid-2".to_string()),
                    detail: None,
                }],
                &mut tokens_first,
                ModelDtype::Float32,
                None,
                true,
            )
            .await
            .expect("first image prepare should succeed");

        let mut tokens_second = vec![1, QWEN3_IMAGE_PAD_ID, 2];
        let second = info
            .prepare_multimodal(
                vec![MediaContentPart::ImageUrl {
                    url: "https://127.0.0.1:1/unreachable.png".to_string(),
                    detail: None,
                    uuid: Some("img-uuid-2".to_string()),
                }],
                &mut tokens_second,
                ModelDtype::Float32,
                None,
                true,
            )
            .await
            .expect("a UUID cache hit should skip fetching unreachable content");
        assert_eq!(
            address_of(second[0].data.as_ref().unwrap()),
            address_of(first[0].data.as_ref().unwrap())
        );
        assert_eq!(tokens_second, tokens_first);
    }

    /// Python's `_get_cache_missing_items` message, mapped to a 400-class
    /// request error.
    #[tokio::test]
    async fn uuid_reference_cache_miss_is_a_request_error() {
        use llm_multimodal::MediaContentPart;
        use vllm_engine_core_client::protocol::dtype::ModelDtype;

        use super::super::tests::{QWEN3_IMAGE_PAD_ID, qwen3_vl_info};

        let shm = test_shm_cache();
        let info = qwen3_vl_info().with_mm_processor_cache(Some(Arc::clone(&shm)));
        let error = info
            .prepare_multimodal(
                vec![MediaContentPart::ImageUrl {
                    url: String::new(),
                    detail: None,
                    uuid: Some("missing-uuid".to_string()),
                }],
                &mut vec![1, QWEN3_IMAGE_PAD_ID, 2],
                ModelDtype::Float32,
                None,
                true,
            )
            .await
            .expect_err("a UUID reference without cached data must fail");
        assert!(matches!(&error, crate::Error::MmUuidReference { message }
                if message == "Cache miss for image at index 0 but data is not provided."));
        assert!(error.is_request_validation_error());
        // Even on the error path, the request's touches are released.
        assert_eq!(shm.touched_len(), 0);
    }

    /// A UUID-only reference cannot be fetched, so it needs the shm cache.
    #[tokio::test]
    async fn uuid_reference_without_content_requires_the_shm_cache() {
        use llm_multimodal::MediaContentPart;
        use vllm_engine_core_client::protocol::dtype::ModelDtype;

        use super::super::tests::{QWEN3_IMAGE_PAD_ID, qwen3_vl_info};

        let info = qwen3_vl_info();
        let error = info
            .prepare_multimodal(
                vec![MediaContentPart::ImageData {
                    data: Vec::new(),
                    mime_type: Some("image/png".to_string()),
                    uuid: Some("img-uuid-3".to_string()),
                    detail: None,
                }],
                &mut vec![1, QWEN3_IMAGE_PAD_ID, 2],
                ModelDtype::Float32,
                None,
                true,
            )
            .await
            .expect_err("a UUID reference without the cache must fail");
        assert!(matches!(&error, crate::Error::MmUuidReference { message }
                if message.contains("carries a UUID but no content")));
        assert!(error.is_request_validation_error());
    }

    /// Budget-sensitive image processors bypass the cache entirely: repeated
    /// prepares of the same image — even with a client UUID and under
    /// different token budgets — always preprocess inline, and no shm entry
    /// is ever created.
    #[tokio::test]
    async fn budget_sensitive_image_processors_bypass_the_cache() {
        use llm_multimodal::MediaContentPart;
        use vllm_engine_core_client::protocol::dtype::ModelDtype;

        use super::super::tests::{NEMOTRON_IMAGE_ID, nemotron_h_omni_info};

        let shm = test_shm_cache();
        let info = nemotron_h_omni_info().with_mm_processor_cache(Some(Arc::clone(&shm)));
        let media_part = || MediaContentPart::ImageData {
            data: test_png_bytes(),
            mime_type: Some("image/png".to_string()),
            uuid: Some("nem-uuid".to_string()),
            detail: None,
        };
        let prompt = || {
            let mut prompt = vec![11; 100];
            prompt.push(NEMOTRON_IMAGE_ID);
            prompt
        };

        for max_model_len in [Some(512), Some(600)] {
            let mut tokens = prompt();
            let features = info
                .prepare_multimodal(
                    vec![media_part()],
                    &mut tokens,
                    ModelDtype::Float32,
                    max_model_len,
                    true,
                )
                .await
                .expect("budget-sensitive image prepare should succeed");
            assert_eq!(features.len(), 1);
            let data = features[0].data.as_ref().expect("feature data");
            assert!(
                !is_address_item(data),
                "budget-sensitive image processors always send the item inline"
            );
        }
        assert_eq!(shm.cached_len(), 0, "no shm entry is ever created");
        assert_eq!(shm.touched_len(), 0, "nothing is pre-touched");
    }

    /// A UUID-only cache reference needs the cache, which budget-sensitive
    /// image processors bypass: the request fails with a clear 400-class
    /// error rather than a cache miss.
    #[tokio::test]
    async fn budget_sensitive_images_reject_uuid_cache_references() {
        use llm_multimodal::MediaContentPart;
        use vllm_engine_core_client::protocol::dtype::ModelDtype;

        use super::super::tests::{NEMOTRON_IMAGE_ID, nemotron_h_omni_info};

        let info = nemotron_h_omni_info().with_mm_processor_cache(Some(test_shm_cache()));
        let error = info
            .prepare_multimodal(
                vec![MediaContentPart::ImageUrl {
                    url: String::new(),
                    detail: None,
                    uuid: Some("nem-uuid".to_string()),
                }],
                &mut vec![11, NEMOTRON_IMAGE_ID],
                ModelDtype::Float32,
                Some(512),
                true,
            )
            .await
            .expect_err("budget-sensitive models reject UUID cache references");
        assert!(matches!(&error, crate::Error::MmUuidReference { message }
                if message.contains("cache references are not supported for this model")));
        assert!(error.is_request_validation_error());
    }

    #[test]
    fn mm_cache_for_bypasses_only_budget_sensitive_image_processors() {
        use llm_multimodal::Modality;

        use super::super::tests::{nemotron_h_omni_info, qwen3_vl_info};

        let qwen = qwen3_vl_info().with_mm_processor_cache(Some(test_shm_cache()));
        assert!(qwen.mm_cache_for(Modality::Image).is_some());
        assert!(qwen.mm_cache_for(Modality::Video).is_some());

        let nemotron = nemotron_h_omni_info().with_mm_processor_cache(Some(test_shm_cache()));
        assert!(nemotron.mm_cache_for(Modality::Image).is_none());

        assert!(qwen3_vl_info().mm_cache_for(Modality::Image).is_none());
    }
}
