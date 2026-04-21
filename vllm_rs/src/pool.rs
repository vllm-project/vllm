use pyo3::exceptions::{PyAssertionError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList};

use crate::block::{FreeKVCacheBlockQueue, KVCacheBlock};
use crate::hash_map::BlockHashToBlockMap;

/// Rust version of `vllm.v1.core.block_pool.BlockPool`.
///
/// Implements the hot methods:
///   - get_new_blocks
///   - free_blocks
///   - touch
///   - get_cached_block  (single-group fast path; multi-group falls back
///     to repeated lookups, matches Python)
///   - get_num_free_blocks
///   - get_usage
///   - reset_prefix_cache
///
/// Out of scope (kept in Python):
///   - cache_full_blocks          (touches Request.block_hashes + MM logic)
///   - evict_blocks               (called only on connector disconnect)
///   - take_events                (emits KV cache dataclasses)
///   - metrics_collector hooks    (Python callback interface)
#[pyclass(module = "vllm_rs")]
pub struct BlockPool {
    #[pyo3(get)]
    pub num_gpu_blocks: u32,
    #[pyo3(get)]
    pub enable_caching: bool,
    #[pyo3(get)]
    pub hash_block_size: u32,
    blocks: Vec<Py<KVCacheBlock>>,
    #[pyo3(get)]
    free_block_queue: Py<FreeKVCacheBlockQueue>,
    #[pyo3(get)]
    cached_block_hash_to_block: Py<BlockHashToBlockMap>,
    #[pyo3(get)]
    null_block: Py<KVCacheBlock>,
}

impl BlockPool {
    fn get_block_at(&self, py: Python<'_>, block_id: i32) -> Py<KVCacheBlock> {
        self.blocks[block_id as usize].clone_ref(py)
    }

    /// Inner helper: mirror of Python's `_maybe_evict_cached_block`.
    /// Returns whether a hash was evicted.
    fn maybe_evict_cached_block(
        &self,
        py: Python<'_>,
        block: &Bound<'_, KVCacheBlock>,
    ) -> PyResult<bool> {
        let hash_bytes: Option<Vec<u8>> = {
            let b = block.borrow();
            b.block_hash_ref().map(|s| s.to_vec())
        };
        let hash_bytes = match hash_bytes {
            None => return Ok(false),
            Some(h) => h,
        };
        let block_id = block.borrow().block_id;
        let popped = {
            let mut m = self.cached_block_hash_to_block.borrow_mut(py);
            m.pop(py, hash_bytes, block_id)
        };
        if popped.is_none() {
            return Ok(false);
        }
        // Reset the hash on the block.
        block.borrow_mut().reset_hash();
        Ok(true)
    }
}

#[pymethods]
impl BlockPool {
    #[new]
    #[pyo3(signature = (num_gpu_blocks, enable_caching, hash_block_size, enable_kv_cache_events=false))]
    pub fn new(
        py: Python<'_>,
        num_gpu_blocks: u32,
        enable_caching: bool,
        hash_block_size: u32,
        enable_kv_cache_events: bool,
    ) -> PyResult<Self> {
        let _ = enable_kv_cache_events; // events stay in Python
        if num_gpu_blocks == 0 {
            return Err(PyValueError::new_err("num_gpu_blocks must be > 0"));
        }
        let mut blocks: Vec<Py<KVCacheBlock>> = Vec::with_capacity(num_gpu_blocks as usize);
        for i in 0..num_gpu_blocks {
            blocks.push(Py::new(py, KVCacheBlock::new(i as i32, 0, None, false))?);
        }
        let queue_blocks: Vec<Py<KVCacheBlock>> =
            blocks.iter().map(|b| b.clone_ref(py)).collect();
        let queue = Py::new(py, FreeKVCacheBlockQueue::new(py, queue_blocks)?)?;
        // Pop the first block to be the null block (block_id 0), matching the
        // Python constructor.
        let null_block = {
            let mut q = queue.borrow_mut(py);
            q.popleft(py)?.unbind()
        };
        null_block.borrow_mut(py).is_null = true;
        let map = Py::new(py, BlockHashToBlockMap::new())?;
        Ok(Self {
            num_gpu_blocks,
            enable_caching,
            hash_block_size,
            blocks,
            free_block_queue: queue,
            cached_block_hash_to_block: map,
            null_block,
        })
    }

    pub fn get_num_free_blocks(&self, py: Python<'_>) -> usize {
        self.free_block_queue.borrow(py).rust_num_free()
    }

    pub fn get_usage(&self, py: Python<'_>) -> f64 {
        let total = (self.num_gpu_blocks - 1) as usize;
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.get_num_free_blocks(py) as f64 / total as f64)
    }

    /// Look up `block_hash` under each group id; return None if any group misses.
    /// Matches Python's `BlockPool.get_cached_block`.
    pub fn get_cached_block<'py>(
        &self,
        py: Python<'py>,
        block_hash: Vec<u8>,
        kv_cache_group_ids: Vec<u32>,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let out = PyList::empty_bound(py);
        let map = self.cached_block_hash_to_block.borrow(py);
        for gid in kv_cache_group_ids {
            // Compose key: block_hash + big-endian u32 group_id. Same as
            // make_block_hash_with_group_id in vllm.v1.core.kv_cache_utils.
            let mut key = Vec::with_capacity(block_hash.len() + 4);
            key.extend_from_slice(&block_hash);
            key.extend_from_slice(&gid.to_be_bytes());
            match map.get_one_block(py, key) {
                Some(b) => out.append(b)?,
                None => return Ok(None),
            }
        }
        Ok(Some(out))
    }

    /// Pop `num_blocks` from the free queue and increment their ref_cnt by 1.
    /// If caching is enabled, also evicts each from the hash cache.
    pub fn get_new_blocks<'py>(
        &mut self,
        py: Python<'py>,
        num_blocks: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        if num_blocks > self.get_num_free_blocks(py) {
            return Err(PyValueError::new_err(format!(
                "Cannot get {} free blocks from the pool",
                num_blocks
            )));
        }
        let ret = {
            let mut q = self.free_block_queue.borrow_mut(py);
            q.popleft_n(py, num_blocks)?
        };
        if self.enable_caching {
            for item in ret.iter() {
                let block: Bound<'_, KVCacheBlock> = item.downcast_into::<KVCacheBlock>()?;
                self.maybe_evict_cached_block(py, &block)?;
                let mut b = block.borrow_mut();
                if b.ref_cnt != 0 {
                    return Err(PyAssertionError::new_err(
                        "block.ref_cnt != 0 on get_new_blocks",
                    ));
                }
                b.ref_cnt += 1;
            }
        } else {
            for item in ret.iter() {
                let block: Bound<'_, KVCacheBlock> = item.downcast_into::<KVCacheBlock>()?;
                let mut b = block.borrow_mut();
                if b.ref_cnt != 0 {
                    return Err(PyAssertionError::new_err(
                        "block.ref_cnt != 0 on get_new_blocks",
                    ));
                }
                b.ref_cnt += 1;
            }
        }
        Ok(ret)
    }

    /// Touch raises ref_cnt by 1 on each block; if ref_cnt was 0 (in free
    /// queue) and the block is not null, remove it from the queue.
    pub fn touch(&mut self, py: Python<'_>, blocks: Bound<'_, PyAny>) -> PyResult<()> {
        let it = PyIterator::from_bound_object(&blocks)?;
        for item in it {
            let item = item?;
            let block: Bound<'_, KVCacheBlock> = item.downcast_into::<KVCacheBlock>()?;
            let (was_free, is_null) = {
                let b = block.borrow();
                (b.ref_cnt == 0, b.is_null)
            };
            if was_free && !is_null {
                let mut q = self.free_block_queue.borrow_mut(py);
                q.remove(py, block.clone())?;
            }
            block.borrow_mut().ref_cnt += 1;
        }
        Ok(())
    }

    /// Decrement ref_cnt for each block in iteration order; append freed
    /// (ref_cnt==0, not null) back to the free queue.
    pub fn free_blocks(&mut self, py: Python<'_>, ordered_blocks: Bound<'_, PyAny>) -> PyResult<()> {
        // Materialize once.
        let it = PyIterator::from_bound_object(&ordered_blocks)?;
        let mut materialized: Vec<Py<KVCacheBlock>> = Vec::new();
        for item in it {
            let b: Bound<'_, KVCacheBlock> = item?.downcast_into::<KVCacheBlock>()?;
            materialized.push(b.unbind());
        }
        // First pass: dec ref_cnt.
        for b in &materialized {
            b.borrow_mut(py).ref_cnt -= 1;
        }
        // Second pass: collect freed-for-real blocks.
        let mut to_append: Vec<Py<KVCacheBlock>> = Vec::new();
        for b in materialized {
            let ok = {
                let br = b.borrow(py);
                br.ref_cnt == 0 && !br.is_null
            };
            if ok {
                to_append.push(b);
            }
        }
        if !to_append.is_empty() {
            let mut q = self.free_block_queue.borrow_mut(py);
            q.append_n(py, to_append)?;
        }
        Ok(())
    }

    pub fn reset_prefix_cache(&mut self, py: Python<'_>) -> bool {
        let num_used = self.num_gpu_blocks as usize - self.get_num_free_blocks(py);
        if num_used != 1 {
            // Null block stays as used; anything above means non-freed real blocks.
            return false;
        }
        {
            let mut m = self.cached_block_hash_to_block.borrow_mut(py);
            m.clear();
        }
        for b in &self.blocks {
            b.borrow_mut(py).reset_hash();
        }
        true
    }

    /// Return the underlying block at index (exposed mostly for parity tests).
    pub fn get_block(&self, py: Python<'_>, block_id: i32) -> Option<Py<KVCacheBlock>> {
        if block_id < 0 || block_id as u32 >= self.num_gpu_blocks {
            return None;
        }
        Some(self.get_block_at(py, block_id))
    }

    /// Fast-path cache-full-blocks: stamp each non-null block's hash and
    /// register it in `cached_block_hash_to_block`. Skips events (caller
    /// keeps event emission in Python) and the block_size != hash_block_size
    /// branch. Python wrapper falls back to its own implementation for those.
    ///
    /// Arguments:
    ///   blocks: list[KVCacheBlock] of length >= num_full
    ///   block_hashes: list[bytes] of length >= (num_full - num_cached)
    ///     — pre-sliced to start at num_cached to match vLLM's layout
    ///   num_cached: number of already-cached blocks
    ///   num_full: total number of full blocks to cover
    ///   kv_cache_group_id: u32 group id
    pub fn cache_full_blocks_fast(
        &mut self,
        py: Python<'_>,
        blocks: Bound<'_, PyList>,
        block_hashes: Bound<'_, PyList>,
        num_cached: usize,
        num_full: usize,
        kv_cache_group_id: u32,
    ) -> PyResult<()> {
        use pyo3::types::PyBytes;
        if num_cached >= num_full {
            return Ok(());
        }
        let count = num_full - num_cached;
        if block_hashes.len() < count {
            return Err(PyAssertionError::new_err(
                "block_hashes slice is shorter than num_full - num_cached",
            ));
        }
        let group_bytes = kv_cache_group_id.to_be_bytes();
        for i in 0..count {
            let blk_obj = blocks.get_item(num_cached + i)?;
            let blk: Bound<'_, KVCacheBlock> = blk_obj.downcast_into::<KVCacheBlock>()?;
            let (is_null, has_hash) = {
                let b = blk.borrow();
                (b.is_null, b.block_hash_ref().is_some())
            };
            if is_null {
                continue;
            }
            if has_hash {
                return Err(PyAssertionError::new_err(
                    "cache_full_blocks: block already has a hash",
                ));
            }
            let raw_hash_obj = block_hashes.get_item(i)?;
            let raw_bytes: &[u8] = raw_hash_obj.downcast::<PyBytes>()?.as_bytes();
            let mut combined = Vec::with_capacity(raw_bytes.len() + 4);
            combined.extend_from_slice(raw_bytes);
            combined.extend_from_slice(&group_bytes);
            blk.borrow_mut().set_hash_internal(combined.clone());
            let mut map = self.cached_block_hash_to_block.borrow_mut(py);
            map.insert(py, combined, blk.as_unbound().clone_ref(py))?;
        }
        Ok(())
    }
}
