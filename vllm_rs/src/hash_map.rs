use pyo3::prelude::*;
use rustc_hash::FxHashMap;

use crate::block::KVCacheBlock;

/// Single/multi-block entry — single block is the common case and avoids the
/// overhead of the inner HashMap.
enum Entry {
    One(Py<KVCacheBlock>),
    Many(FxHashMap<i32, Py<KVCacheBlock>>),
}

#[pyclass(module = "vllm_rs")]
pub struct BlockHashToBlockMap {
    inner: FxHashMap<Vec<u8>, Entry>,
}

impl BlockHashToBlockMap {
    pub fn inner(&self) -> &FxHashMap<Vec<u8>, Entry> {
        &self.inner
    }
}

#[pymethods]
impl BlockHashToBlockMap {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: FxHashMap::default(),
        }
    }

    /// Get any block for the given hash key, or None if not cached.
    #[pyo3(name = "get_one_block")]
    pub fn get_one_block<'py>(
        &self,
        py: Python<'py>,
        key: Vec<u8>,
    ) -> Option<Bound<'py, KVCacheBlock>> {
        match self.inner.get(&key)? {
            Entry::One(b) => Some(b.bind(py).clone()),
            Entry::Many(map) => map
                .values()
                .next()
                .map(|b| b.bind(py).clone()),
        }
    }

    pub fn insert(
        &mut self,
        py: Python<'_>,
        key: Vec<u8>,
        block: Py<KVCacheBlock>,
    ) -> PyResult<()> {
        let block_id = block.borrow(py).block_id;
        match self.inner.remove(&key) {
            None => {
                self.inner.insert(key, Entry::One(block));
            }
            Some(Entry::One(existing)) => {
                let existing_id = existing.borrow(py).block_id;
                let mut map: FxHashMap<i32, Py<KVCacheBlock>> = FxHashMap::default();
                map.insert(existing_id, existing);
                map.insert(block_id, block);
                self.inner.insert(key, Entry::Many(map));
            }
            Some(Entry::Many(mut map)) => {
                map.insert(block_id, block);
                self.inner.insert(key, Entry::Many(map));
            }
        }
        Ok(())
    }

    pub fn pop<'py>(
        &mut self,
        py: Python<'py>,
        key: Vec<u8>,
        block_id: i32,
    ) -> Option<Bound<'py, KVCacheBlock>> {
        let entry = self.inner.remove(&key)?;
        match entry {
            Entry::One(existing) => {
                let existing_id = existing.borrow(py).block_id;
                if existing_id == block_id {
                    return Some(existing.into_bound(py));
                }
                // Mismatched: restore (matches Python behaviour).
                self.inner.insert(key, Entry::One(existing));
                None
            }
            Entry::Many(mut map) => {
                let popped = map.remove(&block_id);
                if !map.is_empty() {
                    self.inner.insert(key, Entry::Many(map));
                }
                popped.map(|p| p.into_bound(py))
            }
        }
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Whether a key exists at all, regardless of one/many.
    pub fn __contains__(&self, key: Vec<u8>) -> bool {
        self.inner.contains_key(&key)
    }
}
