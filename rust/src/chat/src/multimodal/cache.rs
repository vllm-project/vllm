// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Prepare multimodal items using the SHM processor cache.

use std::collections::{HashMap, hash_map::Entry};
use std::num::NonZeroUsize;
use std::sync::Arc;

use itertools::izip;
use llm_multimodal::{Modality, PromptReplacement};
#[cfg(test)]
use vllm_engine_core_client::protocol::multimodal::MmKwargsItem;

use super::{
    MmProcessorCache, MultimodalModelInfo, PreparedItem, PreparedMedia, ResolvedPlaceholder,
};
use crate::error::Result;

impl MultimodalModelInfo {
    /// Nemotron-H Omni's image resolution depends on the request token budget,
    /// which the cache key does not include.
    // TODO: use a processor capability flag once llm-multimodal exposes one.
    fn mm_cache_for(&self, modality: Modality) -> Option<&MmProcessorCache> {
        let cache = self.mm_processor_cache.as_deref()?;
        if modality == Modality::Image
            && self.image.as_ref().is_some_and(|image| {
                matches!(image.spec.raw.name(), "nemotron_omni" | "nemotron_vl")
            })
        {
            return None;
        }
        Some(cache)
    }

    /// Prepare one modality in request order, computing only cache misses.
    pub(super) async fn prepare_media<Source>(
        &self,
        modality: Modality,
        placeholder: &ResolvedPlaceholder,
        sources: Vec<(Source, String)>,
        compute: impl AsyncFnOnce(
            Vec<(Source, String)>,
        ) -> Result<(Vec<PromptReplacement>, Vec<PreparedItem>)>,
    ) -> Result<PreparedMedia> {
        let Some(cache) = self.mm_cache_for(modality) else {
            let (replacements, items) = compute(sources).await?;
            return Ok(PreparedMedia::from_preprocessed(
                modality,
                placeholder.clone(),
                replacements,
                items,
            ));
        };
        let count = sources.len();
        let mut slots = vec![None; count];
        let mut pending = Vec::new();
        let mut positions: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut content_misses = HashMap::<String, usize>::new();
        for (index, (media, key)) in sources.into_iter().enumerate() {
            if let Some((data, replacement)) = cache.get_cached(&key) {
                slots[index] = Some((replacement, PreparedItem { data, mm_hash: key }));
                continue;
            }
            // A client UUID can name different media within the same cold batch.
            if count > 1 && key.starts_with("content:") {
                match content_misses.entry(key.clone()) {
                    Entry::Occupied(entry) => {
                        let (_, duplicates) = &mut positions[*entry.get()];
                        duplicates.push(index);
                        continue;
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(pending.len());
                    }
                }
            }
            pending.push((media, key));
            positions.push((index, Vec::new()));
        }

        if !pending.is_empty() {
            let (replacements, items) = compute(pending).await?;
            for ((first, duplicates), replacement, mut item) in
                izip!(positions, replacements, items)
            {
                let occurrences = NonZeroUsize::MIN.saturating_add(duplicates.len());
                let (data, replacement) = cache.put_or_inline(
                    &item.mm_hash,
                    (item.data, Arc::new(replacement)),
                    occurrences,
                );
                item.data = data;
                for index in duplicates {
                    slots[index] = Some((Arc::clone(&replacement), item.clone()));
                }
                slots[first] = Some((replacement, item));
            }
        }
        let (replacements, items) =
            slots.into_iter().map(|slot| slot.expect("one result per media item")).unzip();
        Ok(PreparedMedia {
            modality,
            placeholder: placeholder.clone(),
            replacements,
            items,
        })
    }
}

/// 64x64 solid-color PNG.
#[cfg(test)]
pub(super) const TEST_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAeklEQVR4nO3PUQkAIBTAwBfNaEYzmiH8OITBAtxm7fN1wwUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWPHYBHsYBafyS08sAAAAASUVORK5CYII=";

#[cfg(test)]
pub(super) fn test_shm_cache() -> Arc<MmProcessorCache> {
    let cache = MmProcessorCache::create(vllm_mm_cache::MmCacheConfig {
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
    fn duplicate_put_preserves_cached_replacement_and_address() {
        let cache = test_shm_cache();
        let (sent, replacement) = cache.put_or_inline(
            "uuid:uuid-1",
            (test_item(11), Arc::new(test_replacement())),
            NonZeroUsize::MIN,
        );
        assert!(is_address_item(&sent));

        let mut other_replacement = test_replacement();
        other_replacement.tokens.push(151_655);
        let other_replacement = Arc::new(other_replacement);
        let (inline, inline_replacement) = cache.put_or_inline(
            "uuid:uuid-1",
            (test_item(22), Arc::clone(&other_replacement)),
            NonZeroUsize::MIN,
        );
        assert_eq!(inline, test_item(22));
        assert!(Arc::ptr_eq(&inline_replacement, &other_replacement));

        let Some((hit, cached_replacement)) = cache.get_cached("uuid:uuid-1") else {
            panic!("uuid-1 should be cached");
        };
        assert!(Arc::ptr_eq(&cached_replacement, &replacement));
        assert_eq!(hit, sent);
    }

    #[tokio::test]
    async fn cold_content_duplicates_share_one_result_in_request_order() {
        let cache = test_shm_cache();
        let (hot, _) = cache.put_or_inline(
            "content:hot",
            (test_item(0), Arc::new(test_replacement())),
            NonZeroUsize::MIN,
        );
        let info = super::super::tests::qwen3_vl_info().with_mm_processor_cache(Some(cache));
        let placeholder = &info.image.as_ref().unwrap().placeholder;
        let keys = ["content:a", "uuid:b", "content:hot", "content:a", "uuid:b"];
        let sources = keys.iter().enumerate().map(|(i, key)| (i, key.to_string())).collect();
        let PreparedMedia {
            replacements,
            items,
            ..
        } = info
            .prepare_media(Modality::Image, placeholder, sources, async |misses| {
                assert_eq!(
                    misses.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
                    [0, 1, 4]
                );
                Ok(misses
                    .into_iter()
                    .map(|(i, mm_hash)| {
                        (
                            test_replacement(),
                            PreparedItem {
                                data: test_item(i as i64),
                                mm_hash,
                            },
                        )
                    })
                    .unzip())
            })
            .await
            .unwrap();

        assert_eq!(
            items.iter().map(|item| item.mm_hash.as_str()).collect::<Vec<_>>(),
            keys
        );
        assert_eq!(items[2].data, hot);
        assert!(is_address_item(&items[0].data));
        assert_eq!(items[0].data, items[3].data);
        assert!(Arc::ptr_eq(&replacements[0], &replacements[3]));
        assert!(is_address_item(&items[1].data));
        assert_eq!(items[4].data, test_item(4));
    }

    #[tokio::test]
    async fn oversize_repeated_images_match_uncached_features_and_prompt() {
        use llm_multimodal::MediaContentPart;
        use vllm_engine_core_client::protocol::dtype::ModelDtype;

        use super::super::tests::{QWEN3_IMAGE_PAD_ID, qwen3_vl_info};

        let shm = Arc::new(
            MmProcessorCache::create(vllm_mm_cache::MmCacheConfig {
                shm_name: vllm_mm_cache::MmCacheConfig::unique_shm_name(),
                data_buffer_size: 4096,
                max_object_size: 1,
            })
            .unwrap(),
        );
        shm.set_n_readers(1);
        let prepare = async |cache| {
            let media = MediaContentPart::ImageUrl {
                max_long_side_pixel: None,
                url: format!("data:image/png;base64,{TEST_PNG_BASE64}").into(),
                detail: None,
                uuid: None,
            };
            let mut tokens = vec![1, QWEN3_IMAGE_PAD_ID, 2, QWEN3_IMAGE_PAD_ID, 3];
            let features = qwen3_vl_info()
                .with_mm_processor_cache(cache)
                .prepare_multimodal(
                    vec![media.clone(), media],
                    &mut tokens,
                    ModelDtype::Float32,
                    None,
                )
                .await
                .unwrap();
            (tokens, features)
        };
        assert_eq!(prepare(Some(shm)).await, prepare(None).await);
    }

    /// Both prepares fetch the image. The second one reuses the
    /// cached address and replacement instead of re-preprocessing or
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
                url: format!("data:image/png;base64,{TEST_PNG_BASE64}").into(),
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
