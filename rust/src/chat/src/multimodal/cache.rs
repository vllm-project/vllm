// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Process-local cache for multimodal processor outputs.

use std::collections::{BTreeMap, HashMap};
use std::mem::size_of;

use bytes::Bytes;
use llm_multimodal::{Modality, PromptReplacement};
use parking_lot::Mutex;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::{MmFieldElem, MmKwargValue, MmKwargsItem};
use vllm_engine_core_client::protocol::tensor::{WireArrayData, WireTensor};

#[derive(Clone)]
pub(super) struct CachedProcessorOutput {
    pub(super) data: MmKwargsItem,
    pub(super) replacement: PromptReplacement,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    modality: Modality,
    hash: String,
    dtype: &'static str,
    variant: u8,
}

impl CacheKey {
    fn new(modality: Modality, hash: &str, dtype: ModelDtype, variant: u8) -> Self {
        Self {
            modality,
            hash: hash.to_string(),
            dtype: dtype.as_str(),
            variant,
        }
    }
}

struct CacheEntry {
    output: CachedProcessorOutput,
    weight: usize,
    stamp: u64,
}

struct WeightedLru {
    capacity: usize,
    weight: usize,
    clock: u64,
    entries: HashMap<CacheKey, CacheEntry>,
    order: BTreeMap<u64, CacheKey>,
    hits: u64,
    misses: u64,
}

impl WeightedLru {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            weight: 0,
            clock: 0,
            entries: HashMap::new(),
            order: BTreeMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    fn next_stamp(&mut self) -> u64 {
        if self.clock == u64::MAX {
            let old_order = std::mem::take(&mut self.order);
            for (stamp, key) in old_order.into_values().enumerate() {
                let stamp = stamp as u64;
                self.entries
                    .get_mut(&key)
                    .expect("LRU order and entries must stay aligned")
                    .stamp = stamp;
                self.order.insert(stamp, key);
            }
            self.clock = self.entries.len() as u64;
        }
        let stamp = self.clock;
        self.clock += 1;
        stamp
    }

    fn get(&mut self, key: &CacheKey) -> Option<CachedProcessorOutput> {
        if !self.entries.contains_key(key) {
            self.misses += 1;
            return None;
        }

        let stamp = self.next_stamp();
        let entry = self.entries.get_mut(key).expect("cache entry was checked above");
        self.order.remove(&entry.stamp);
        entry.stamp = stamp;
        self.order.insert(stamp, key.clone());
        self.hits += 1;
        Some(entry.output.clone())
    }

    fn insert(&mut self, key: CacheKey, output: CachedProcessorOutput, weight: usize) {
        if self.capacity == 0 || weight > self.capacity {
            return;
        }

        if let Some(old) = self.entries.remove(&key) {
            self.order.remove(&old.stamp);
            self.weight -= old.weight;
        }

        while self.weight.saturating_add(weight) > self.capacity {
            let Some((_, oldest_key)) = self.order.pop_first() else {
                break;
            };
            let oldest = self
                .entries
                .remove(&oldest_key)
                .expect("LRU order and entries must stay aligned");
            self.weight -= oldest.weight;
        }

        let stamp = self.next_stamp();
        self.weight += weight;
        self.order.insert(stamp, key.clone());
        self.entries.insert(
            key,
            CacheEntry {
                output,
                weight,
                stamp,
            },
        );
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.weight = 0;
    }
}

struct CacheState {
    generation: u64,
    lru: WeightedLru,
}

pub(super) struct ProcessorCache {
    state: Mutex<CacheState>,
}

impl ProcessorCache {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(CacheState {
                generation: 0,
                lru: WeightedLru::new(capacity),
            }),
        }
    }

    pub(super) fn generation(&self) -> u64 {
        self.state.lock().generation
    }

    pub(super) fn get(
        &self,
        modality: Modality,
        hash: &str,
        dtype: ModelDtype,
        variant: u8,
    ) -> Option<CachedProcessorOutput> {
        self.state.lock().lru.get(&CacheKey::new(modality, hash, dtype, variant))
    }

    pub(super) fn insert(
        &self,
        generation: u64,
        modality: Modality,
        hash: &str,
        dtype: ModelDtype,
        variant: u8,
        data: &MmKwargsItem,
        replacement: &PromptReplacement,
    ) {
        let capacity = {
            let state = self.state.lock();
            if state.generation != generation || state.lru.capacity == 0 {
                return;
            }
            state.lru.capacity
        };

        let Some(weight) = item_weight(data) else {
            return;
        };
        if weight > capacity {
            return;
        }

        let Some(data) = detached_item(data) else {
            return;
        };
        let output = CachedProcessorOutput {
            data,
            replacement: replacement.clone(),
        };
        let key = CacheKey::new(modality, hash, dtype, variant);

        let mut state = self.state.lock();
        if state.generation == generation {
            state.lru.insert(key, output, weight);
        }
    }

    pub(super) fn clear(&self) {
        let mut state = self.state.lock();
        state.generation = state.generation.wrapping_add(1);
        state.lru.clear();
    }

    #[cfg(test)]
    pub(super) fn snapshot(&self) -> CacheSnapshot {
        let state = self.state.lock();
        CacheSnapshot {
            generation: state.generation,
            len: state.lru.entries.len(),
            weight: state.lru.weight,
            hits: state.lru.hits,
            misses: state.lru.misses,
        }
    }
}

fn item_weight(item: &MmKwargsItem) -> Option<usize> {
    let mut weight = 0usize;
    for elem in item.values() {
        if let Some(value) = &elem.data {
            weight = weight.saturating_add(value_weight(value)?);
        }
    }
    Some(weight.max(1))
}

fn value_weight(value: &MmKwargValue) -> Option<usize> {
    match value {
        MmKwargValue::Tensor(tensor) => match &tensor.data {
            WireArrayData::RawView(bytes) => Some(bytes.len()),
            WireArrayData::AuxIndex(_) => None,
        },
        MmKwargValue::Int(_) => Some(size_of::<i64>()),
        MmKwargValue::Float(_) => Some(size_of::<f64>()),
        MmKwargValue::List(values) => values.iter().try_fold(0usize, |weight, value| {
            Some(weight.saturating_add(value_weight(value)?))
        }),
    }
}

fn detached_item(item: &MmKwargsItem) -> Option<MmKwargsItem> {
    let mut detached = MmKwargsItem::new();
    for (key, elem) in item {
        let data = match &elem.data {
            Some(value) => Some(detached_value(value)?),
            None => None,
        };
        detached.insert(
            key.clone(),
            MmFieldElem {
                data,
                field: elem.field.clone(),
            },
        );
    }
    Some(detached)
}

fn detached_value(value: &MmKwargValue) -> Option<MmKwargValue> {
    match value {
        MmKwargValue::Tensor(tensor) => {
            let WireArrayData::RawView(bytes) = &tensor.data else {
                return None;
            };
            let tensor = WireTensor {
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
                data: WireArrayData::RawView(Bytes::copy_from_slice(bytes)),
            };
            Some(MmKwargValue::Tensor(tensor))
        }
        MmKwargValue::Int(value) => Some(MmKwargValue::Int(*value)),
        MmKwargValue::Float(value) => Some(MmKwargValue::Float(*value)),
        MmKwargValue::List(values) => {
            let mut detached = Vec::with_capacity(values.len());
            for value in values {
                detached.push(detached_value(value)?);
            }
            Some(MmKwargValue::List(detached))
        }
    }
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(super) struct CacheSnapshot {
    pub(super) generation: u64,
    pub(super) len: usize,
    pub(super) weight: usize,
    pub(super) hits: u64,
    pub(super) misses: u64,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_multimodal::PromptReplacement;
    use vllm_engine_core_client::protocol::multimodal::{MmBatchedField, MmField};

    use super::*;

    fn output(bytes: &'static [u8]) -> (MmKwargsItem, PromptReplacement) {
        let mut data = MmKwargsItem::new();
        data.insert(
            "pixel_values".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Tensor(WireTensor::from_raw_bytes(
                    "uint8",
                    vec![bytes.len()],
                    Bytes::from_static(bytes),
                ))),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
            },
        );
        (
            data,
            PromptReplacement::repeated(Modality::Image, "<image>", 7, 2),
        )
    }

    fn insert(cache: &ProcessorCache, hash: &str, bytes: &'static [u8]) {
        let (data, replacement) = output(bytes);
        cache.insert(
            cache.generation(),
            Modality::Image,
            hash,
            ModelDtype::Float16,
            0,
            &data,
            &replacement,
        );
    }

    #[test]
    fn hit_refreshes_weighted_lru_order() {
        let cache = ProcessorCache::new(8);
        insert(&cache, "a", b"aaaa");
        insert(&cache, "b", b"bbbb");

        assert!(cache.get(Modality::Image, "a", ModelDtype::Float16, 0).is_some());
        insert(&cache, "c", b"cccc");

        assert!(cache.get(Modality::Image, "a", ModelDtype::Float16, 0).is_some());
        assert!(cache.get(Modality::Image, "b", ModelDtype::Float16, 0).is_none());
        assert!(cache.get(Modality::Image, "c", ModelDtype::Float16, 0).is_some());
        assert_eq!(cache.snapshot().weight, 8);
    }

    #[test]
    fn oversized_and_disabled_entries_are_not_admitted() {
        let cache = ProcessorCache::new(3);
        insert(&cache, "large", b"four");
        assert_eq!(cache.snapshot().len, 0);

        let disabled = ProcessorCache::new(0);
        insert(&disabled, "small", b"x");
        assert_eq!(disabled.snapshot().len, 0);
    }

    #[test]
    fn keys_separate_modality_dtype_and_variant() {
        let cache = ProcessorCache::new(16);
        insert(&cache, "same", b"data");

        assert!(cache.get(Modality::Image, "same", ModelDtype::Float16, 0).is_some());
        assert!(cache.get(Modality::Video, "same", ModelDtype::Float16, 0).is_none());
        assert!(cache.get(Modality::Image, "same", ModelDtype::Float32, 0).is_none());
        assert!(cache.get(Modality::Image, "same", ModelDtype::Float16, 1).is_none());
    }

    #[test]
    fn clear_rejects_in_flight_insert_from_old_generation() {
        let cache = ProcessorCache::new(16);
        let old_generation = cache.generation();
        let (data, replacement) = output(b"data");

        cache.clear();
        cache.insert(
            old_generation,
            Modality::Image,
            "stale",
            ModelDtype::Float16,
            0,
            &data,
            &replacement,
        );

        assert_eq!(
            cache.snapshot(),
            CacheSnapshot {
                generation: 1,
                len: 0,
                weight: 0,
                hits: 0,
                misses: 0,
            }
        );
    }

    #[test]
    fn cached_tensor_owns_only_its_exact_bytes() {
        let backing = Bytes::from_static(b"0123456789");
        let slice = backing.slice(2..6);
        let source_ptr = slice.as_ptr();
        let mut data = MmKwargsItem::new();
        data.insert(
            "pixel_values".to_string(),
            MmFieldElem {
                data: Some(MmKwargValue::Tensor(WireTensor::from_raw_bytes(
                    "uint8",
                    vec![4],
                    slice,
                ))),
                field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
            },
        );
        let replacement = PromptReplacement::repeated(Modality::Image, "<image>", 7, 2);
        let cache = ProcessorCache::new(8);
        cache.insert(
            0,
            Modality::Image,
            "image",
            ModelDtype::Float16,
            0,
            &data,
            &replacement,
        );

        let cached = cache.get(Modality::Image, "image", ModelDtype::Float16, 0).unwrap();
        let MmKwargValue::Tensor(tensor) = cached.data["pixel_values"].data.as_ref().unwrap()
        else {
            panic!("expected tensor");
        };
        let WireArrayData::RawView(bytes) = &tensor.data else {
            panic!("expected raw bytes");
        };
        assert_eq!(bytes.as_ref(), b"2345");
        assert_ne!(bytes.as_ptr(), source_ptr);
        assert_eq!(cache.snapshot().weight, 4);
    }

    #[test]
    fn concurrent_access_keeps_weight_bounded() {
        let cache = Arc::new(ProcessorCache::new(32));
        let threads = (0..8)
            .map(|thread| {
                let cache = Arc::clone(&cache);
                std::thread::spawn(move || {
                    for item in 0..100 {
                        let hash = format!("{thread}-{item}");
                        let (data, replacement) = output(b"data");
                        cache.insert(
                            cache.generation(),
                            Modality::Image,
                            &hash,
                            ModelDtype::Float16,
                            0,
                            &data,
                            &replacement,
                        );
                        let _ = cache.get(Modality::Image, &hash, ModelDtype::Float16, 0);
                    }
                })
            })
            .collect::<Vec<_>>();

        for thread in threads {
            thread.join().unwrap();
        }
        let snapshot = cache.snapshot();
        assert!(snapshot.weight <= 32);
        assert!(snapshot.len <= 8);
    }
}
