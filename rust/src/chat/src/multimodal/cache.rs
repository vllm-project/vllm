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

use itertools::izip;
use llm_multimodal::PromptReplacement;
use parking_lot::Mutex;
use vllm_engine_core_client::protocol::multimodal::MmKwargsItem;
use vllm_mm_cache::MmProcessorShmCache;

use super::{PreparedItem, VisionModalitySupport};
use crate::error::Result;

/// Vision processors whose preprocessing output depends on the per-request
/// `VisionPreprocessingContext` (today: only Nemotron-H Omni's
/// `token_budget`-driven dynamic resolution). The cache key does not include
/// the request budget, so reusing output across budgets would be incorrect:
/// such processors bypass the processor cache entirely and preprocess inline.
///
/// TODO: drop this name list once `llm-multimodal` exposes a capability flag
/// on `VisionPreProcessor` for request-context sensitivity.
const BUDGET_SENSITIVE_PROCESSORS: &[&str] = &["nemotron_omni"];

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
pub(super) struct MmProcessorCache {
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

impl MmProcessorCache {
    pub(super) fn new(shm: Arc<MmProcessorShmCache>) -> Self {
        Self {
            shm,
            shadow: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Look up one fetched media item by its engine-facing `mm_hash`.
    ///
    /// On a hit, bumps writer_flag and returns the address item plus the
    /// shadowed prompt replacement. On a miss — including a shadowed entry
    /// whose shm data vanished under a concurrent eviction — returns `None`.
    pub(super) fn lookup(&self, key: &str) -> Option<(PreparedItem, PromptReplacement)> {
        let replacement = self.shadow.lock().get(key).cloned()?;
        let data = self.shm.address_item_for_cached(key)?;
        Some((
            PreparedItem {
                data,
                mm_hash: key.to_owned(),
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

    /// Look up fetched items by their final `mm_hash` so only misses reach
    /// `compute`; computed misses are put into shm and shadowed.
    pub(super) async fn prepare_cached<M, F>(
        &self,
        sources: Vec<(M, String)>,
        compute: F,
    ) -> Result<(Vec<PromptReplacement>, Vec<PreparedItem>)>
    where
        F: AsyncFnOnce(Vec<(M, String)>) -> Result<(Vec<PromptReplacement>, Vec<PreparedItem>)>,
    {
        let mut slots: Vec<Option<(PreparedItem, PromptReplacement)>> =
            Vec::with_capacity(sources.len());
        let mut pending = Vec::new();
        for (media, key) in sources {
            match self.lookup(&key) {
                Some(hit) => slots.push(Some(hit)),
                None => {
                    slots.push(None);
                    pending.push((media, key));
                }
            }
        }

        let mut misses = Vec::with_capacity(pending.len());
        if !pending.is_empty() {
            let (replacements, items) = compute(pending).await?;
            for (mut item, replacement) in izip!(items, replacements) {
                item.data = self.put_or_inline_and_track(&item.mm_hash, &replacement, item.data);
                misses.push((item, replacement));
            }
        }
        Ok(merge_slots(slots, misses))
    }
}

/// Reassemble per-modality replacements and items in request order from
/// cache-hit slots and freshly preprocessed misses.
///
/// `slots` has one entry per media item: `Some` for hits, `None` for misses.
/// `misses` carries one prepared item + replacement per `None` slot, in slot
/// order, so the result stays index-aligned for placeholder expansion.
fn merge_slots(
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

/// 64x64 solid-color PNG.
#[cfg(test)]
pub(super) const TEST_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAeklEQVR4nO3PUQkAIBTAwBfNaEYzmiH8OITBAtxm7fN1wwUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWPHYBHsYBafyS08sAAAAASUVORK5CYII=";

#[cfg(test)]
pub(super) fn test_shm_cache() -> Arc<MmProcessorShmCache> {
    let cache = MmProcessorShmCache::create(vllm_mm_cache::MmCacheConfig {
        shm_name: vllm_mm_cache::MmCacheConfig::unique_shm_name(),
        data_buffer_size: 16 * 1024 * 1024,
        max_object_size: 4 * 1024 * 1024,
    })
    .expect("shm cache should be created");
    cache.set_n_readers(1);
    Arc::new(cache)
}

#[cfg(test)]
pub(super) fn is_address_item(item: &MmKwargsItem) -> bool {
    item.len() == 3
        && item.contains_key("address")
        && item.contains_key("monotonic_id")
        && item.contains_key("signature")
}

#[cfg(test)]
pub(super) mod tests {
    use vllm_engine_core_client::protocol::multimodal::{MmKwargValue, MmKwargsItem};

    use super::*;

    fn address_of(item: &MmKwargsItem) -> (i64, i64) {
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
    fn hit_reuses_shadowed_replacement_and_address() {
        let cache = MmProcessorCache::new(test_shm_cache());
        let sent = cache.put_or_inline_and_track("uuid:uuid-1", &test_replacement(), test_item(11));
        assert!(is_address_item(&sent));

        let Some((hit, replacement)) = cache.lookup("uuid:uuid-1") else {
            panic!("uuid-1 should be cached");
        };
        assert_eq!(replacement.tokens, test_replacement().tokens);
        assert_eq!(hit.mm_hash, "uuid:uuid-1");
        assert!(is_address_item(&hit.data));
        assert_eq!(address_of(&hit.data), address_of(&sent));
    }

    #[test]
    fn merge_slots_restores_request_order() {
        let item = |value: i64| PreparedItem {
            data: test_item(value),
            mm_hash: format!("hash-{value}"),
        };
        let slots = vec![
            Some((item(1), test_replacement())),
            None,
            Some((item(3), test_replacement())),
            None,
        ];
        let misses = vec![(item(2), test_replacement()), (item(4), test_replacement())];

        let (replacements, items) = merge_slots(slots, misses);

        let hashes: Vec<&str> = items.iter().map(|item| item.mm_hash.as_str()).collect();
        assert_eq!(hashes, ["hash-1", "hash-2", "hash-3", "hash-4"]);
        assert_eq!(replacements.len(), 4);
    }

    /// Both prepares fetch and decode the image. The second one reuses the
    /// cached address and shadowed replacement instead of re-preprocessing or
    /// re-putting it, which would produce inline data for the duplicate key.
    #[tokio::test]
    async fn image_cache_hit_reuses_address_and_replacement_after_fetch() {
        use llm_multimodal::MediaContentPart;
        use vllm_engine_core_client::protocol::dtype::ModelDtype;

        use super::super::tests::{QWEN3_IMAGE_PAD_ID, qwen3_vl_info};

        let shm = test_shm_cache();
        let info = qwen3_vl_info().with_mm_processor_cache(Some(shm.clone()));
        let prepare = async |uuid| {
            let media_part = MediaContentPart::ImageUrl {
                max_long_side_pixel: None,
                url: format!("data:image/png;base64,{TEST_PNG_BASE64}"),
                detail: None,
                uuid,
            };
            let mut tokens = vec![1, QWEN3_IMAGE_PAD_ID, 2];
            let [feature] = info
                .prepare_multimodal(vec![media_part], &mut tokens, ModelDtype::Float32, None)
                .await
                .unwrap()
                .try_into()
                .unwrap();
            assert!(is_address_item(feature.data.as_ref().unwrap()));
            assert!(
                shm.is_cached(feature.mm_hash.as_deref().unwrap()),
                "SHM key must match feature mm_hash"
            );
            (tokens, feature)
        };

        let by_content = prepare(None).await;
        let content_key = by_content.1.mm_hash.as_deref().unwrap();
        let uuid = content_key.strip_prefix("content:").unwrap().to_owned();
        let by_uuid = prepare(Some(uuid.clone())).await;
        assert_eq!(
            by_uuid.1.mm_hash.as_deref().unwrap(),
            format!("uuid:{uuid}")
        );
        assert_ne!(
            address_of(by_content.1.data.as_ref().unwrap()),
            address_of(by_uuid.1.data.as_ref().unwrap())
        );

        assert_eq!(prepare(None).await, by_content);
        assert_eq!(prepare(Some(uuid)).await, by_uuid);
    }
}
