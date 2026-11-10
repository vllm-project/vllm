// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Frontend shadow of one engine's LRU multimodal receiver cache.
//!
//! Python `MultiModalReceiverCache` keeps processed items keyed by `mm_hash`,
//! so an item it already holds can be sent with `data = None`. The shadow
//! replays the receiver's updates for every request sent to that engine, in
//! send order, and so knows which items the engine holds without asking it.
//! Drift (for example after an engine-side reset) is reported back through
//! `EngineCoreOutput::mm_cache_miss_hashes` and fixed by [`MmSenderCache::invalidate`].

use std::collections::{BTreeMap, HashMap};

use crate::protocol::multimodal::{MmFeatureSpec, MmKwargValue, MmKwargsItem};
use crate::protocol::tensor::ShapeExt as _;

/// `sys.getsizeof(None)` on CPython 3.12.
const PY_NONE_SIZE: u64 = 16;
/// `sys.getsizeof(<float>)` on CPython 3.12.
const PY_FLOAT_SIZE: u64 = 24;
/// `sys.getsizeof(<int>)` on CPython 3.12 is this header plus 4 bytes per
/// 30-bit digit, with at least one digit.
const PY_INT_HEADER_SIZE: u64 = 24;

pub(crate) struct MmSenderCache {
    capacity: u64,
    used: u64,
    next_tick: u64,
    entries: HashMap<String, Entry>,
    /// Recency order: the smallest tick is the least recently used key.
    order: BTreeMap<u64, String>,
}

struct Entry {
    size: u64,
    tick: u64,
}

impl MmSenderCache {
    pub(crate) fn new(capacity: u64) -> Self {
        Self {
            capacity,
            used: 0,
            next_tick: 0,
            entries: HashMap::new(),
            order: BTreeMap::new(),
        }
    }

    /// Drop the data of items the engine already holds, updating the shadow
    /// exactly as `BaseMultiModalReceiverCache.get_and_update_features` will
    /// update the engine: touch every key first, then hit or insert in order.
    pub(crate) fn strip_cached(&mut self, features: &mut [MmFeatureSpec]) {
        for feature in features.iter() {
            self.touch(cache_key(feature));
        }
        for feature in features {
            let key = cache_key(feature).to_owned();
            if self.entries.contains_key(&key) {
                self.touch(&key);
                feature.data = None;
            } else if let Some(data) = &feature.data {
                self.insert_if_fits(key, item_size(data));
            }
        }
    }

    /// Forget keys the engine reported as missing.
    pub(crate) fn invalidate<'a>(&mut self, keys: impl IntoIterator<Item = &'a String>) {
        for key in keys {
            if let Some(entry) = self.entries.remove(key) {
                self.order.remove(&entry.tick);
                self.used -= entry.size;
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.used = 0;
    }

    fn touch(&mut self, key: &str) {
        let tick = self.next_tick;
        if let Some(entry) = self.entries.get_mut(key) {
            self.order.remove(&entry.tick);
            entry.tick = tick;
            self.order.insert(tick, key.to_owned());
            self.next_tick += 1;
        }
    }

    /// Mirror `cachetools.LRUCache.__setitem__` for a new key: items larger
    /// than the whole cache are skipped, otherwise the least recently used
    /// entries are evicted until the item fits.
    fn insert_if_fits(&mut self, key: String, size: u64) {
        if size > self.capacity {
            return;
        }
        while self.used + size > self.capacity {
            let (_, evicted) = self.order.pop_first().expect("used > 0 implies an entry");
            let entry = self.entries.remove(&evicted).expect("ordered keys are cached");
            self.used -= entry.size;
        }
        let tick = self.next_tick;
        self.next_tick += 1;
        self.order.insert(tick, key.clone());
        self.entries.insert(key, Entry { size, tick });
        self.used += size;
    }
}

/// The receiver cache key: `feature.mm_hash or feature.identifier`.
fn cache_key(feature: &MmFeatureSpec) -> &str {
    feature
        .mm_hash
        .as_deref()
        .filter(|hash| !hash.is_empty())
        .unwrap_or(&feature.identifier)
}

/// `MultiModalCache.get_item_size` of the decoded `MultiModalKwargsItem`.
fn item_size(item: &MmKwargsItem) -> u64 {
    item.values()
        .map(|elem| elem.data.as_ref().map_or(PY_NONE_SIZE, value_size))
        .sum()
}

fn value_size(value: &MmKwargValue) -> u64 {
    match value {
        MmKwargValue::Tensor(tensor) => {
            let numel = tensor.shape.checked_numel().expect("validated tensor shape");
            (numel * tensor.dtype.element_size()) as u64
        }
        MmKwargValue::Int(value) => {
            let bits = u64::from(64 - value.unsigned_abs().leading_zeros());
            PY_INT_HEADER_SIZE + 4 * bits.div_ceil(30).max(1)
        }
        MmKwargValue::Float(_) => PY_FLOAT_SIZE,
        MmKwargValue::List(values) => values.iter().map(value_size).sum(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::protocol::dtype::TensorDtype;
    use crate::protocol::multimodal::{
        MmBatchedField, MmField, MmFieldElem, MmModality, PlaceholderRange,
    };
    use crate::protocol::tensor::WireTensor;

    fn feature(hash: &str, data: Option<MmKwargsItem>) -> MmFeatureSpec {
        MmFeatureSpec {
            data,
            modality: MmModality::Image,
            identifier: format!("id-{hash}"),
            mm_position: PlaceholderRange {
                offset: 0,
                length: 1,
                is_embed: None,
            },
            mm_hash: Some(hash.to_owned()),
        }
    }

    fn item(values: Vec<(&str, MmKwargValue)>) -> MmKwargsItem {
        values
            .into_iter()
            .map(|(name, value)| {
                let elem = MmFieldElem {
                    data: Some(value),
                    field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
                };
                (name.to_owned(), elem)
            })
            .collect::<BTreeMap<_, _>>()
    }

    /// An item whose Python size is exactly `bytes` (a 1-D uint8 tensor).
    fn sized_item(bytes: usize) -> MmKwargsItem {
        let tensor = WireTensor::from_raw(TensorDtype::U8, vec![bytes], vec![0; bytes]);
        item(vec![("pixel_values", MmKwargValue::Tensor(tensor))])
    }

    fn sent(features: &[MmFeatureSpec]) -> Vec<bool> {
        features.iter().map(|feature| feature.data.is_some()).collect()
    }

    #[test]
    fn item_size_matches_python_getsizeof() {
        let tensor = WireTensor::from_bool(vec![3, 4], vec![false; 12]).unwrap();
        let ints = [
            0,
            1,
            -1,
            (1 << 30) - 1,
            1 << 30,
            (1 << 60) - 1,
            1 << 60,
            i64::MIN,
        ];
        let values = vec![
            ("tensor", MmKwargValue::Tensor(tensor)),
            ("float", MmKwargValue::Float(1.5)),
            (
                "ints",
                MmKwargValue::List(ints.into_iter().map(MmKwargValue::Int).collect()),
            ),
        ];
        // Measured with CPython 3.12: ints are 28, 28, 28, 28, 32, 32, 36, 36.
        assert_eq!(
            item_size(&item(values)),
            12 + 24 + (28 * 4 + 32 * 2 + 36 * 2)
        );

        let mut with_none = sized_item(5);
        with_none.insert(
            "missing".to_owned(),
            MmFieldElem {
                data: None,
                field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
            },
        );
        assert_eq!(item_size(&with_none), 5 + PY_NONE_SIZE);
    }

    #[test]
    fn repeated_items_are_sent_once() {
        let mut cache = MmSenderCache::new(100);
        let mut first = vec![
            feature("a", Some(sized_item(10))),
            feature("a", Some(sized_item(10))),
        ];
        cache.strip_cached(&mut first);
        assert_eq!(sent(&first), [true, false]);

        let mut second = vec![feature("a", Some(sized_item(10)))];
        cache.strip_cached(&mut second);
        assert_eq!(sent(&second), [false]);
    }

    #[test]
    fn eviction_follows_receiver_recency() {
        let mut cache = MmSenderCache::new(30);
        for hash in ["a", "b", "c"] {
            cache.strip_cached(&mut [feature(hash, Some(sized_item(10)))]);
        }
        // Using `a` makes `b` the least recently used entry.
        cache.strip_cached(&mut [feature("a", Some(sized_item(10)))]);
        cache.strip_cached(&mut [feature("d", Some(sized_item(10)))]);

        let held = ["a", "b", "c", "d"].map(|hash| cache.entries.contains_key(hash));
        assert_eq!(held, [true, false, true, true]);
    }

    #[test]
    fn touch_before_insert_protects_items_hit_in_the_same_request() {
        let mut cache = MmSenderCache::new(20);
        cache.strip_cached(&mut [feature("a", Some(sized_item(10)))]);
        cache.strip_cached(&mut [feature("b", Some(sized_item(10)))]);

        // `a` is touched before `c` is inserted, so `b` is evicted instead.
        let mut request = vec![
            feature("c", Some(sized_item(10))),
            feature("a", Some(sized_item(10))),
        ];
        cache.strip_cached(&mut request);
        assert_eq!(sent(&request), [true, false]);
        assert!(!cache.entries.contains_key("b"));
    }

    #[test]
    fn items_larger_than_the_cache_are_always_sent() {
        let mut cache = MmSenderCache::new(10);
        for _ in 0..2 {
            let mut request = vec![feature("big", Some(sized_item(11)))];
            cache.strip_cached(&mut request);
            assert_eq!(sent(&request), [true]);
        }
    }

    #[test]
    fn invalidated_and_cleared_items_are_resent() {
        let mut cache = MmSenderCache::new(100);
        cache.strip_cached(&mut [
            feature("a", Some(sized_item(10))),
            feature("b", Some(sized_item(10))),
        ]);

        cache.invalidate(&["a".to_owned()]);
        let mut request = vec![
            feature("a", Some(sized_item(10))),
            feature("b", Some(sized_item(10))),
        ];
        cache.strip_cached(&mut request);
        assert_eq!(sent(&request), [true, false]);
        assert_eq!(cache.used, 20);

        cache.clear();
        let mut request = vec![feature("b", Some(sized_item(10)))];
        cache.strip_cached(&mut request);
        assert_eq!(sent(&request), [true]);
    }

    #[test]
    fn cache_key_falls_back_to_identifier() {
        let mut cache = MmSenderCache::new(100);
        let mut without_hash = feature("a", Some(sized_item(10)));
        without_hash.mm_hash = None;
        cache.strip_cached(std::slice::from_mut(&mut without_hash));
        assert!(cache.entries.contains_key("id-a"));
    }
}
