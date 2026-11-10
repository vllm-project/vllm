// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Python views of the Rust KV cache state: `KVCacheBlocks`, blocks, and
//! the block pool / coordinator / single-type manager facades that the
//! scheduler and KV connectors read.
//!
//! Views hold a reference to their KV cache manager ([`Owner`]) and read the
//! live state on access, as the Python objects they stand in for do.

use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyInt, PyList, PySet, PyTuple};
use vllm_engine_core::kv_cache::{
    BlockHashWithGroupId, BlockId, BlockPool, HOST_BLOCK_FLAG, KvCacheBlocks, KvCacheManager,
};

use super::events::events_to_py;
use super::manager::KVCacheManager;
use super::request::convert_block_hashes;

/// The KV cache manager inside a Rust scheduler, as reachable from Python.
///
/// The scheduler owns the manager; Python code reaches it only while the
/// scheduler is not using it: between scheduler calls, and inside a KV
/// connector hook that was handed the manager. The binding layer publishes
/// the manager's address for exactly those windows ([`Self::set`],
/// [`Self::scoped`]); views fail with `RuntimeError` outside of them.
pub struct SchedulerKv {
    manager: AtomicPtr<KvCacheManager>,
    /// The scheduler-side Python spec of each group.
    specs: Vec<Py<PyAny>>,
    kv_cache_config: Py<PyAny>,
}

impl SchedulerKv {
    pub fn new(specs: Vec<Py<PyAny>>, kv_cache_config: Py<PyAny>) -> Self {
        Self {
            manager: AtomicPtr::new(ptr::null_mut()),
            specs,
            kv_cache_config,
        }
    }

    /// Publish `manager` (or nothing, with null) until the next call.
    ///
    /// The caller guarantees that the manager outlives the publication and
    /// that Rust code does not use it meanwhile.
    pub fn set(&self, manager: *mut KvCacheManager) {
        self.manager.store(manager, Ordering::Release);
    }

    /// Publish `manager` while `f` runs, e.g. a connector hook that received
    /// it. Restores the previous publication afterwards, also on unwind.
    pub fn scoped<R>(&self, manager: &mut KvCacheManager, f: impl FnOnce() -> R) -> R {
        struct Restore<'a>(&'a AtomicPtr<KvCacheManager>, *mut KvCacheManager);
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                self.0.store(self.1, Ordering::Release);
            }
        }
        let prev = self.manager.swap(manager, Ordering::AcqRel);
        let _restore = Restore(&self.manager, prev);
        f()
    }

    fn get(&self) -> PyResult<*mut KvCacheManager> {
        let manager = self.manager.load(Ordering::Acquire);
        if manager.is_null() {
            return Err(PyRuntimeError::new_err(
                "the scheduler's KV cache manager is not accessible here: it is \
                 in use by the scheduler, or the scheduler is gone",
            ));
        }
        Ok(manager)
    }
}

/// The KV cache manager a view reads.
pub enum Owner {
    /// A Python-facing [`KVCacheManager`].
    Manager(Py<KVCacheManager>),
    /// The manager inside a Rust scheduler.
    Scheduler(Arc<SchedulerKv>),
}

impl Owner {
    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            Self::Manager(m) => Self::Manager(m.clone_ref(py)),
            Self::Scheduler(s) => Self::Scheduler(s.clone()),
        }
    }

    pub fn is(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Manager(a), Self::Manager(b)) => a.is(b),
            (Self::Scheduler(a), Self::Scheduler(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Run `f` on the manager. `f` must not run Python code.
    pub fn with<R>(&self, py: Python<'_>, f: impl FnOnce(&KvCacheManager) -> R) -> PyResult<R> {
        match self {
            Self::Manager(m) => Ok(f(&m.bind(py).try_borrow()?.inner)),
            // SAFETY: see `SchedulerKv`: the manager is published only while
            // nothing else uses it, and `f` runs no Python code that could
            // reach it again.
            Self::Scheduler(s) => Ok(f(unsafe { &*s.get()? })),
        }
    }

    /// Run `f` on the manager, mutably. `f` must not run Python code.
    pub fn with_mut<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut KvCacheManager) -> R,
    ) -> PyResult<R> {
        match self {
            Self::Manager(m) => Ok(f(&mut m.bind(py).try_borrow_mut()?.inner)),
            // SAFETY: as in `with`.
            Self::Scheduler(s) => Ok(f(unsafe { &mut *s.get()? })),
        }
    }

    fn spec(&self, py: Python<'_>, group_id: usize) -> PyResult<Py<PyAny>> {
        match self {
            Self::Manager(m) => Ok(m.bind(py).try_borrow()?.specs[group_id].clone_ref(py)),
            Self::Scheduler(s) => Ok(s.specs[group_id].clone_ref(py)),
        }
    }

    fn kv_cache_config(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Manager(m) => Ok(m.bind(py).try_borrow()?.kv_cache_config.clone_ref(py)),
            Self::Scheduler(s) => Ok(s.kv_cache_config.clone_ref(py)),
        }
    }
}

/// The device pool, or the HiSparse host pool if `host`.
fn pool_of(m: &KvCacheManager, host: bool) -> &BlockPool {
    match m.host_block_pool() {
        Some(pool) if host => pool,
        _ => m.block_pool(),
    }
}

/// `ids` with [`HOST_BLOCK_FLAG`] set if they are blocks of group
/// `group_id` and that group is the HiSparse host group.
fn tag_group(m: &KvCacheManager, group_id: usize, ids: &[BlockId]) -> Vec<BlockId> {
    if m.coordinator.host_group_id() == Some(group_id) {
        ids.iter().map(|b| b | HOST_BLOCK_FLAG).collect()
    } else {
        ids.to_vec()
    }
}

/// `KVCacheBlock`: a view of one block of the pool.
#[pyclass(frozen, module = "vllm._rust_core", name = "KVCacheBlock")]
pub struct Block {
    owner: Owner,
    #[pyo3(get)]
    block_id: BlockId,
    /// Whether this is a block of the HiSparse host pool.
    host: bool,
}

impl Block {
    fn with<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&vllm_engine_core::kv_cache::KvCacheBlock) -> R,
    ) -> PyResult<R> {
        self.owner.with(py, |m| f(pool_of(m, self.host).block(self.block_id)))
    }

    /// The block id, with [`HOST_BLOCK_FLAG`] for a host block.
    fn tagged_id(&self) -> BlockId {
        if self.host {
            self.block_id | HOST_BLOCK_FLAG
        } else {
            self.block_id
        }
    }
}

fn packed_hash<'py>(
    py: Python<'py>,
    hash: Option<&BlockHashWithGroupId>,
) -> Option<Bound<'py, PyBytes>> {
    hash.map(|h| PyBytes::new(py, &h.to_packed_bytes()))
}

#[pymethods]
impl Block {
    #[getter]
    fn ref_cnt(&self, py: Python<'_>) -> PyResult<i32> {
        self.with(py, |b| b.ref_cnt)
    }

    /// The packed `BlockHashWithGroupId` bytes, or None.
    #[getter]
    fn block_hash<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let hash = self.with(py, |b| b.block_hash.clone())?;
        Ok(packed_hash(py, hash.as_ref()))
    }

    #[getter]
    fn block_hash_num_tokens(&self, py: Python<'_>) -> PyResult<Option<usize>> {
        self.with(py, |b| b.block_hash_num_tokens)
    }

    #[getter]
    fn is_null(&self, py: Python<'_>) -> PyResult<bool> {
        self.with(py, |b| b.is_null)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<Block>()
            .is_ok_and(|o| o.get().tagged_id() == self.tagged_id() && o.get().owner.is(&self.owner))
    }

    fn __hash__(&self) -> u64 {
        u64::from(self.block_id)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let (ref_cnt, hash, num_tokens) = self.with(py, |b| {
            (b.ref_cnt, b.block_hash.clone(), b.block_hash_num_tokens)
        })?;
        let hash = packed_hash(py, hash.as_ref()).map_or_else(
            || "None".to_owned(),
            |h| h.repr().map_or_else(|_| "?".into(), |r| r.to_string()),
        );
        let num_tokens = num_tokens.map_or_else(|| "None".to_owned(), |n| n.to_string());
        Ok(format!(
            "KVCacheBlock(block_id={}, ref_cnt={ref_cnt}, block_hash={hash}, \
             block_hash_num_tokens={num_tokens})",
            self.block_id
        ))
    }
}

fn block_view(owner: &Owner, py: Python<'_>, id: BlockId) -> Block {
    Block {
        owner: owner.clone_ref(py),
        block_id: id & !HOST_BLOCK_FLAG,
        host: id & HOST_BLOCK_FLAG != 0,
    }
}

/// Views of `ids` (host blocks with [`HOST_BLOCK_FLAG`]), as a Python list.
pub fn block_list<'py>(
    py: Python<'py>,
    owner: &Owner,
    ids: &[BlockId],
) -> PyResult<Bound<'py, PyList>> {
    let views = ids
        .iter()
        .map(|&id| Bound::new(py, block_view(owner, py, id)))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, views)
}

/// Views of the blocks `ids` of group `group_id`.
fn group_block_list<'py>(
    py: Python<'py>,
    owner: &Owner,
    group_id: usize,
    ids: &[BlockId],
) -> PyResult<Bound<'py, PyList>> {
    let ids = owner.with(py, |m| tag_group(m, group_id, ids))?;
    block_list(py, owner, &ids)
}

/// The block id of a block view (with [`HOST_BLOCK_FLAG`] for a host block),
/// a Python `KVCacheBlock` or an int.
pub fn block_id_of(obj: &Bound<'_, PyAny>) -> PyResult<BlockId> {
    if let Ok(block) = obj.cast::<Block>() {
        return Ok(block.get().tagged_id());
    }
    if obj.is_instance_of::<PyInt>() {
        return obj.extract();
    }
    obj.getattr(pyo3::intern!(obj.py(), "block_id"))?.extract()
}

pub fn block_ids_of(blocks: &Bound<'_, PyAny>) -> PyResult<Vec<BlockId>> {
    blocks.try_iter()?.map(|b| block_id_of(&b?)).collect()
}

/// Per-group block ids of a `KVCacheBlocks` (Rust or Python) or of a tuple
/// of per-group block sequences.
pub fn grouped_block_ids(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<BlockId>>> {
    if let Ok(blocks) = obj.cast::<KVCacheBlocksView>() {
        return Ok(blocks.get().blocks.blocks.clone());
    }
    let groups = if obj.hasattr(pyo3::intern!(obj.py(), "blocks"))? {
        obj.getattr(pyo3::intern!(obj.py(), "blocks"))?
    } else {
        obj.clone()
    };
    // A group's blocks all come from one pool: drop the host flag.
    groups
        .try_iter()?
        .map(|g| Ok(block_ids_of(&g?)?.into_iter().map(|b| b & !HOST_BLOCK_FLAG).collect()))
        .collect()
}

pub fn group_tuple<'py>(
    py: Python<'py>,
    groups: impl IntoIterator<Item = impl AsRef<[BlockId]>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let lists = groups
        .into_iter()
        .map(|g| PyList::new(py, g.as_ref()))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, lists)
}

/// `KVCacheBlocks`: the allocation result of the KV cache manager.
#[pyclass(frozen, module = "vllm._rust_core", name = "KVCacheBlocks")]
pub struct KVCacheBlocksView {
    /// None when every group is empty: the manager shares one such instance,
    /// which must not reference the manager back.
    owner: Option<Owner>,
    pub blocks: KvCacheBlocks,
}

impl KVCacheBlocksView {
    pub fn new(owner: Option<Owner>, blocks: KvCacheBlocks) -> Self {
        let owner = owner.filter(|_| !blocks.is_empty());
        Self { owner, blocks }
    }

    fn owner(&self) -> PyResult<&Owner> {
        self.owner
            .as_ref()
            .ok_or_else(|| PyTypeError::new_err("KVCacheBlocks has no KV cache manager"))
    }
}

#[pymethods]
impl KVCacheBlocksView {
    /// `blocks[i][j]` refers to the i-th kv_cache_group and the j-th block of
    /// tokens. Empty groups are `()` as in Python's shared empty instance.
    #[getter]
    fn blocks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let groups = self
            .blocks
            .blocks
            .iter()
            .enumerate()
            .map(|(group_id, g)| {
                if self.blocks.is_empty() {
                    Ok(PyTuple::empty(py).into_any())
                } else {
                    group_block_list(py, self.owner()?, group_id, g).map(Bound::into_any)
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, groups)
    }

    /// Converts the KVCacheBlocks instance to block_ids.
    #[pyo3(signature = (allow_none=false, *, group_ids=None))]
    fn get_block_ids<'py>(
        &self,
        py: Python<'py>,
        allow_none: bool,
        group_ids: Option<Vec<usize>>,
    ) -> PyResult<Option<Bound<'py, PyTuple>>> {
        let groups: Vec<&Vec<BlockId>> = match &group_ids {
            None => self.blocks.blocks.iter().collect(),
            Some(ids) => ids
                .iter()
                .map(|&g| {
                    self.blocks
                        .blocks
                        .get(g)
                        .ok_or_else(|| PyIndexError::new_err("group id out of range"))
                })
                .collect::<PyResult<_>>()?,
        };
        if allow_none && groups.iter().all(|g| g.is_empty()) {
            return Ok(None);
        }
        group_tuple(py, groups).map(Some)
    }

    /// Get block_ids of unhashed blocks from KVCacheBlocks instance.
    fn get_unhashed_block_ids(&self, py: Python<'_>) -> PyResult<Vec<BlockId>> {
        if self.blocks.is_empty() {
            assert!(self.blocks.blocks.len() == 1, "Only one group is supported");
            return Ok(Vec::new());
        }
        self.owner()?.with(py, |m| self.blocks.get_unhashed_block_ids(m.group_pool(0)))
    }

    /// Get block_ids of unhashed blocks from KVCacheBlocks instance.
    fn get_unhashed_block_ids_all_groups(&self, py: Python<'_>) -> PyResult<Vec<Vec<BlockId>>> {
        if self.blocks.is_empty() {
            return Ok(self.blocks.new_empty().blocks);
        }
        self.owner()?.with(py, |m| {
            self.blocks.get_unhashed_block_ids_all_groups(|g| m.group_pool(g))
        })
    }

    /// Creates a new KVCacheBlocks instance with no blocks.
    fn new_empty(&self) -> Self {
        Self::new(None, self.blocks.new_empty())
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let owner = match (&self.owner, other.cast::<KVCacheBlocksView>()) {
            (Some(owner), _) => Some(owner.clone_ref(py)),
            (None, Ok(other)) => other.get().owner.as_ref().map(|o| o.clone_ref(py)),
            (None, Err(_)) => None,
        };
        let other = KvCacheBlocks::new(grouped_block_ids(other)?);
        Ok(Self::new(owner, self.blocks.add(&other)))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        grouped_block_ids(other).is_ok_and(|other| other == self.blocks.blocks)
    }

    fn __repr__(&self) -> String {
        format!("KVCacheBlocks(blocks={:?})", self.blocks.blocks)
    }
}

/// `block_pool.blocks`: indexable views of all blocks.
#[pyclass(frozen, module = "vllm._rust_core", name = "BlockList", sequence)]
pub struct BlockList {
    owner: Owner,
    host: bool,
}

#[pymethods]
impl BlockList {
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.owner.with(py, |m| pool_of(m, self.host).num_gpu_blocks)
    }

    fn __getitem__(&self, py: Python<'_>, idx: isize) -> PyResult<Block> {
        let len = self.__len__(py)? as isize;
        let idx = if idx < 0 { idx + len } else { idx };
        if !(0..len).contains(&idx) {
            return Err(PyIndexError::new_err("block index out of range"));
        }
        Ok(Block {
            owner: self.owner.clone_ref(py),
            block_id: idx as BlockId,
            host: self.host,
        })
    }
}

/// `BlockPool` facade: the device pool, or the HiSparse host pool.
#[pyclass(frozen, module = "vllm._rust_core", name = "BlockPool")]
pub struct BlockPoolView {
    owner: Owner,
    host: bool,
}

impl BlockPoolView {
    pub fn new(owner: Owner) -> Self {
        Self { owner, host: false }
    }

    /// The HiSparse host pool.
    pub fn host(owner: Owner) -> Self {
        Self { owner, host: true }
    }

    fn with<R>(&self, py: Python<'_>, f: impl FnOnce(&BlockPool) -> R) -> PyResult<R> {
        self.owner.with(py, |m| f(pool_of(m, self.host)))
    }

    fn with_mut<R>(&self, py: Python<'_>, f: impl FnOnce(&mut BlockPool) -> R) -> PyResult<R> {
        self.owner.with_mut(py, |m| m.with_pool_mut(self.host, f))
    }

    /// Ids of this pool: ints are its block ids.
    fn own_ids(&self, blocks: &Bound<'_, PyAny>) -> PyResult<Vec<BlockId>> {
        blocks
            .try_iter()?
            .map(|b| {
                let b = b?;
                let id = block_id_of(&b)?;
                Ok(if self.host && b.cast::<Block>().is_err() {
                    id | HOST_BLOCK_FLAG
                } else {
                    id
                })
            })
            .collect()
    }

    /// `own_ids` of this pool's blocks, without the host flag.
    fn raw_ids(&self, blocks: &Bound<'_, PyAny>) -> PyResult<Vec<BlockId>> {
        Ok(self.own_ids(blocks)?.into_iter().map(|b| b & !HOST_BLOCK_FLAG).collect())
    }

    fn tag(&self, ids: &[BlockId]) -> Vec<BlockId> {
        if self.host {
            ids.iter().map(|b| b | HOST_BLOCK_FLAG).collect()
        } else {
            ids.to_vec()
        }
    }
}

#[pymethods]
impl BlockPoolView {
    /// Views of the same pool are equal (Python hands out the pool itself).
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<Self>()
            .is_ok_and(|o| o.get().host == self.host && o.get().owner.is(&self.owner))
    }

    #[getter]
    fn num_gpu_blocks(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |p| p.num_gpu_blocks)
    }

    #[getter]
    fn enable_caching(&self, py: Python<'_>) -> PyResult<bool> {
        self.with(py, |p| p.enable_caching)
    }

    #[getter]
    fn hash_block_size(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |p| p.hash_block_size)
    }

    #[getter]
    fn num_hash_lookahead_tokens(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |p| p.num_hash_lookahead_tokens)
    }

    #[getter]
    fn enable_kv_cache_events(&self, py: Python<'_>) -> PyResult<bool> {
        self.with(py, |p| p.enable_kv_cache_events)
    }

    #[getter]
    fn medium(&self, py: Python<'_>) -> PyResult<String> {
        self.with(py, |p| p.medium.clone())
    }

    #[getter]
    fn null_block(&self, py: Python<'_>) -> PyResult<Block> {
        let block_id = self.with(py, |p| p.null_block)?;
        Ok(Block {
            owner: self.owner.clone_ref(py),
            block_id,
            host: self.host,
        })
    }

    #[getter]
    fn blocks(&self, py: Python<'_>) -> BlockList {
        BlockList {
            owner: self.owner.clone_ref(py),
            host: self.host,
        }
    }

    #[getter]
    fn metrics_collector(&self) -> Option<Py<PyAny>> {
        None
    }

    /// `cached_block_hashes_by_block`: a snapshot `dict[int, set[bytes]]` of
    /// each block's secondary (partial) hash keys.
    #[getter]
    fn cached_block_hashes_by_block<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let entries = self.with(py, |p| {
            p.cached_block_hashes_by_block()
                .map(|(block, hashes)| {
                    (
                        block,
                        hashes.iter().map(|h| h.to_packed_bytes()).collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        })?;
        let dict = PyDict::new(py);
        for (block, hashes) in entries {
            let set = PySet::empty(py)?;
            for hash in hashes {
                set.add(PyBytes::new(py, &hash))?;
            }
            dict.set_item(block, set)?;
        }
        Ok(dict)
    }

    #[getter]
    fn free_block_queue(&self, py: Python<'_>) -> FreeBlockQueue {
        FreeBlockQueue {
            owner: self.owner.clone_ref(py),
            host: self.host,
        }
    }

    fn get_num_free_blocks(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, BlockPool::get_num_free_blocks)
    }

    fn get_usage(&self, py: Python<'_>) -> PyResult<f64> {
        self.with(py, BlockPool::get_usage)
    }

    /// Get the cached block by the block hash for each group in
    /// `kv_cache_group_ids`, or None if cache miss for any group.
    fn get_cached_block<'py>(
        &self,
        py: Python<'py>,
        block_hash: &[u8],
        kv_cache_group_ids: Vec<usize>,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let hash = vllm_engine_core::hash::BlockHash::from_slice(block_hash);
        let blocks = self.with(py, |p| p.get_cached_block(&hash, &kv_cache_group_ids))?;
        blocks.map(|b| block_list(py, &self.owner, &self.tag(&b))).transpose()
    }

    /// Get new blocks from the free block pool.
    fn get_new_blocks<'py>(
        &self,
        py: Python<'py>,
        num_blocks: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let blocks = self.with_mut(py, |p| {
            (num_blocks <= p.get_num_free_blocks()).then(|| p.get_new_blocks(num_blocks))
        })?;
        let Some(blocks) = blocks else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot get {num_blocks} free blocks from the pool"
            )));
        };
        block_list(py, &self.owner, &self.tag(&blocks))
    }

    fn is_block_writable(&self, py: Python<'_>, block: &Bound<'_, PyAny>) -> PyResult<bool> {
        let block = block_id_of(block)? & !HOST_BLOCK_FLAG;
        self.with(py, |p| p.is_block_writable(block))
    }

    fn touch(&self, py: Python<'_>, blocks: &Bound<'_, PyAny>) -> PyResult<()> {
        let ids = self.raw_ids(blocks)?;
        self.with_mut(py, |p| p.touch(&ids))
    }

    /// Free blocks; blocks of the other pool (HiSparse) go back to it.
    #[pyo3(signature = (ordered_blocks, evict_first=false))]
    fn free_blocks(
        &self,
        py: Python<'_>,
        ordered_blocks: &Bound<'_, PyAny>,
        evict_first: bool,
    ) -> PyResult<()> {
        let ids = self.own_ids(ordered_blocks)?;
        self.owner.with_mut(py, |m| m.free_blocks_from(ids, evict_first, self.host))
    }

    fn evict_blocks(&self, py: Python<'_>, block_ids: &Bound<'_, PyAny>) -> PyResult<()> {
        let ids = self.raw_ids(block_ids)?;
        self.with_mut(py, |p| p.evict_blocks(ids))
    }

    fn reset_prefix_cache(&self, py: Python<'_>) -> PyResult<bool> {
        self.with_mut(py, BlockPool::reset_prefix_cache)
    }

    /// Take the KV cache events. The HiSparse host pool publishes into the
    /// device pool's queue, so both take the same events.
    fn take_events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let events = self.owner.with_mut(py, |m| m.block_pool_mut().take_events())?;
        events_to_py(py, events)
    }
}

/// `BlockPool.free_block_queue` facade.
#[pyclass(frozen, module = "vllm._rust_core", name = "FreeKVCacheBlockQueue")]
pub struct FreeBlockQueue {
    owner: Owner,
    host: bool,
}

#[pymethods]
impl FreeBlockQueue {
    #[getter]
    fn num_free_blocks(&self, py: Python<'_>) -> PyResult<usize> {
        self.owner.with(py, |m| pool_of(m, self.host).get_num_free_blocks())
    }

    /// The free blocks, in eviction order.
    fn get_all_free_blocks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let ids = self.owner.with(py, |m| {
            let ids = pool_of(m, self.host).free_block_queue.get_all_free_blocks();
            if self.host {
                ids.into_iter().map(|b| b | HOST_BLOCK_FLAG).collect()
            } else {
                ids
            }
        })?;
        block_list(py, &self.owner, &ids)
    }
}

/// `SingleTypeKVCacheManager` facade.
#[pyclass(frozen, module = "vllm._rust_core", name = "SingleTypeKVCacheManager")]
pub struct SingleTypeManagerView {
    owner: Owner,
    #[pyo3(get)]
    kv_cache_group_id: usize,
}

impl SingleTypeManagerView {
    pub(crate) fn new(owner: Owner, kv_cache_group_id: usize) -> Self {
        Self {
            owner,
            kv_cache_group_id,
        }
    }

    fn with<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&vllm_engine_core::kv_cache::SingleTypeKvCacheManager) -> R,
    ) -> PyResult<R> {
        self.owner.with(py, |m| {
            f(&m.coordinator.single_type_managers[self.kv_cache_group_id])
        })
    }
}

#[pymethods]
impl SingleTypeManagerView {
    #[getter]
    fn block_size(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |m| m.block_size)
    }

    /// The Python spec of the group.
    #[getter]
    fn kv_cache_spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.owner.spec(py, self.kv_cache_group_id)
    }

    #[getter]
    fn has_positionally_stable_blocks(&self, py: Python<'_>) -> PyResult<bool> {
        self.with(py, |m| m.has_positionally_stable_blocks())
    }

    #[getter]
    fn use_eagle(&self, py: Python<'_>) -> PyResult<bool> {
        self.with(py, |m| m.use_eagle)
    }

    #[getter]
    fn block_pool(&self, py: Python<'_>) -> PyResult<BlockPoolView> {
        let host = self.owner.with(py, |m| {
            m.coordinator.host_group_id() == Some(self.kv_cache_group_id)
        })?;
        Ok(BlockPoolView {
            owner: self.owner.clone_ref(py),
            host,
        })
    }

    #[getter]
    fn max_admission_blocks_per_request(&self, py: Python<'_>) -> PyResult<Option<usize>> {
        self.with(py, |m| m.max_admission_blocks_per_request)
    }

    /// `HiSparseHotManager.require_hot`.
    fn require_hot(&self, py: Python<'_>, request_id: &str) -> PyResult<()> {
        self.owner.with_mut(py, |m| {
            m.coordinator.single_type_managers[self.kv_cache_group_id].require_hot(request_id);
        })
    }

    /// `HiSparseHotManager.hot_required`: a snapshot.
    #[getter]
    fn hot_required<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let ids: Vec<String> =
            self.with(py, |m| m.hot_required_ids().map(str::to_owned).collect())?;
        PySet::new(py, ids)
    }

    #[getter]
    fn blocks_per_request(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |m| m.blocks_per_request())
    }

    #[pyo3(signature = (
        request_id,
        num_tokens,
        new_computed_blocks,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=false,
        prefill_end=0,
    ))]
    fn get_num_blocks_to_allocate(
        &self,
        py: Python<'_>,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &Bound<'_, PyAny>,
        total_computed_tokens: usize,
        num_local_computed_tokens: usize,
        num_tokens_main_model: usize,
        apply_admission_cap: bool,
        prefill_end: usize,
    ) -> PyResult<i64> {
        let new_computed_blocks: Vec<BlockId> = block_ids_of(new_computed_blocks)?
            .into_iter()
            .map(|b| b & !HOST_BLOCK_FLAG)
            .collect();
        self.owner.with_mut(py, |m| {
            m.coordinator.sync_hisparse();
            m.coordinator.group_get_num_blocks_to_allocate(
                self.kv_cache_group_id,
                request_id,
                num_tokens,
                &new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap,
                prefill_end,
            )
        })
    }

    fn allocate_new_blocks<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
        num_tokens: usize,
        num_tokens_main_model: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let blocks = self.owner.with_mut(py, |m| {
            m.coordinator.sync_hisparse();
            m.coordinator.group_allocate_new_blocks(
                self.kv_cache_group_id,
                request_id,
                num_tokens,
                num_tokens_main_model,
            )
        })?;
        group_block_list(py, &self.owner, self.kv_cache_group_id, &blocks)
    }

    /// `HiSparseHotManager.has_hot`.
    fn has_hot(&self, py: Python<'_>, request_id: &str) -> PyResult<bool> {
        self.with(py, |m| m.has_hot(request_id))
    }

    /// `HiSparseResidentManager.get_resident_page`.
    fn get_resident_page(
        &self,
        py: Python<'_>,
        request_id: &str,
        block_idx: usize,
    ) -> PyResult<Option<Block>> {
        let block = self.with(py, |m| m.get_resident_page(request_id, block_idx))?;
        Ok(block.map(|id| block_view(&self.owner, py, id)))
    }

    /// `HiSparseResidentManager.get_residency_row`.
    #[pyo3(signature = (request_id, start=0))]
    fn get_residency_row<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
        start: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let ids = self.with(py, |m| m.get_residency_row(request_id, start))?;
        block_list(py, &self.owner, &ids)
    }

    /// `req_to_blocks`: a live read-only view of the block tables.
    #[getter]
    fn req_to_blocks(&self, py: Python<'_>) -> ReqToBlocks {
        ReqToBlocks {
            owner: self.owner.clone_ref(py),
            kv_cache_group_id: self.kv_cache_group_id,
        }
    }

    /// `num_cached_block`: a snapshot of the number of cached blocks of each
    /// running request.
    #[getter]
    fn num_cached_block<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let counts: Vec<(String, usize)> = self.with(py, |m| {
            m.req_ids()
                .filter_map(|id| m.num_cached_blocks(id).map(|n| (id.to_owned(), n)))
                .collect()
        })?;
        let out = pyo3::types::PyDict::new(py);
        for (request_id, n) in counts {
            out.set_item(request_id, n)?;
        }
        Ok(out)
    }

    /// The request's block table (empty if unknown).
    fn get_blocks<'py>(&self, py: Python<'py>, request_id: &str) -> PyResult<Bound<'py, PyList>> {
        let ids = self.with(py, |m| {
            m.req_blocks(request_id).unwrap_or_default().to_vec()
        })?;
        group_block_list(py, &self.owner, self.kv_cache_group_id, &ids)
    }
}

/// `KVCacheCoordinator` facade.
#[pyclass(frozen, module = "vllm._rust_core", name = "KVCacheCoordinator")]
pub struct CoordinatorView {
    owner: Owner,
}

impl CoordinatorView {
    pub fn new(owner: Owner) -> Self {
        Self { owner }
    }
}

fn hit_blocks<'py>(
    py: Python<'py>,
    owner: &Owner,
    blocks: &[Vec<BlockId>],
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        blocks
            .iter()
            .enumerate()
            .map(|(group_id, g)| group_block_list(py, owner, group_id, g))
            .collect::<PyResult<Vec<_>>>()?,
    )
}

#[pymethods]
impl CoordinatorView {
    #[getter]
    fn single_type_managers(&self, py: Python<'_>) -> PyResult<Vec<SingleTypeManagerView>> {
        let n = self.owner.with(py, |m| m.coordinator.single_type_managers.len())?;
        Ok((0..n)
            .map(|kv_cache_group_id| SingleTypeManagerView {
                owner: self.owner.clone_ref(py),
                kv_cache_group_id,
            })
            .collect())
    }

    #[getter]
    fn block_pool(&self, py: Python<'_>) -> BlockPoolView {
        BlockPoolView::new(self.owner.clone_ref(py))
    }

    #[getter]
    fn kv_cache_config(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.owner.kv_cache_config(py)
    }

    #[getter]
    fn enable_partial_hash_hits(&self, py: Python<'_>) -> PyResult<bool> {
        self.owner.with(py, |m| m.coordinator.enable_partial_hash_hits)
    }

    #[getter]
    fn full_attention_group_id(&self, py: Python<'_>) -> PyResult<Option<usize>> {
        self.owner.with(py, |m| m.coordinator.full_attention_group_id)
    }

    #[getter]
    fn num_reprefillable_tokens(&self, py: Python<'_>) -> PyResult<usize> {
        self.owner.with(py, |m| m.coordinator.num_reprefillable_tokens)
    }

    #[getter]
    fn scheduler_block_size(&self, py: Python<'_>) -> PyResult<usize> {
        self.owner.with(py, |m| m.coordinator.scheduler_block_size)
    }

    #[getter]
    fn hash_block_size(&self, py: Python<'_>) -> PyResult<usize> {
        self.owner.with(py, |m| m.coordinator.hash_block_size)
    }

    #[getter]
    fn retention_interval(&self, py: Python<'_>) -> PyResult<Option<usize>> {
        self.owner.with(py, |m| m.coordinator.retention_interval)
    }

    #[getter]
    fn eagle_group_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let ids: Vec<usize> = self.owner.with(py, |m| {
            m.coordinator.eagle_group_ids.iter().copied().collect()
        })?;
        PySet::new(py, ids)
    }

    #[getter]
    fn drop_eagle_group_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let ids: Vec<usize> = self.owner.with(py, |m| {
            m.coordinator.drop_eagle_group_ids.iter().copied().collect()
        })?;
        PySet::new(py, ids)
    }

    fn find_longest_cache_hit<'py>(
        &self,
        py: Python<'py>,
        block_hashes: &Bound<'py, PyAny>,
        max_cache_hit_length: usize,
    ) -> PyResult<(Bound<'py, PyTuple>, usize, usize)> {
        let hashes = convert_block_hashes(block_hashes)?;
        let (blocks, hit, uncached) = self.owner.with(py, |m| {
            m.coordinator.find_longest_cache_hit(&hashes, max_cache_hit_length)
        })?;
        Ok((hit_blocks(py, &self.owner, &blocks)?, hit, uncached))
    }

    fn find_longest_cache_hit_per_group<'py>(
        &self,
        py: Python<'py>,
        block_hashes: &Bound<'py, PyAny>,
        max_cache_hit_length: usize,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
        let hashes = convert_block_hashes(block_hashes)?;
        let result = self.owner.with(py, |m| {
            (m.coordinator.kind == vllm_engine_core::kv_cache::CoordinatorKind::Hybrid).then(|| {
                m.coordinator.find_longest_cache_hit_per_group(&hashes, max_cache_hit_length)
            })
        })?;
        let Some((blocks, hits)) = result else {
            return Err(PyTypeError::new_err(
                "find_longest_cache_hit_per_group is only defined for hybrid coordinators",
            ));
        };
        Ok((
            hit_blocks(py, &self.owner, &blocks)?,
            PyTuple::new(py, hits)?,
        ))
    }

    #[pyo3(signature = (
        request_id,
        num_tokens,
        new_computed_blocks,
        num_encoder_tokens,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=false,
        prefill_end=0,
    ))]
    fn get_num_blocks_to_allocate(
        &self,
        py: Python<'_>,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &Bound<'_, PyAny>,
        num_encoder_tokens: usize,
        total_computed_tokens: usize,
        num_local_computed_tokens: usize,
        num_tokens_main_model: usize,
        apply_admission_cap: bool,
        prefill_end: usize,
    ) -> PyResult<i64> {
        let new_computed_blocks = grouped_block_ids(new_computed_blocks)?;
        self.owner.with_mut(py, |m| {
            m.coordinator.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                &new_computed_blocks,
                num_encoder_tokens,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap,
                prefill_end,
            )
        })
    }

    fn get_blocks<'py>(&self, py: Python<'py>, request_id: &str) -> PyResult<Bound<'py, PyTuple>> {
        let blocks = self.owner.with(py, |m| m.coordinator.get_blocks(request_id))?;
        hit_blocks(py, &self.owner, &blocks)
    }
}

/// `SingleTypeKVCacheManager.req_to_blocks`: read-only mapping from request
/// id to the request's block views.
#[pyclass(frozen, module = "vllm._rust_core", name = "ReqToBlocks", mapping)]
pub struct ReqToBlocks {
    owner: Owner,
    kv_cache_group_id: usize,
}

impl ReqToBlocks {
    fn with<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&vllm_engine_core::kv_cache::SingleTypeKvCacheManager) -> R,
    ) -> PyResult<R> {
        self.owner.with(py, |m| {
            f(&m.coordinator.single_type_managers[self.kv_cache_group_id])
        })
    }

    fn lookup<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let ids = self.with(py, |m| m.req_blocks(request_id).map(<[BlockId]>::to_vec))?;
        ids.map(|ids| group_block_list(py, &self.owner, self.kv_cache_group_id, &ids))
            .transpose()
    }
}

#[pymethods]
impl ReqToBlocks {
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |m| m.req_ids().count())
    }

    fn keys(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.with(py, |m| m.req_ids().map(str::to_owned).collect())
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(PyList::new(py, self.keys(py)?)?.into_any().try_iter()?.into_any())
    }

    fn __contains__(&self, py: Python<'_>, request_id: &str) -> PyResult<bool> {
        self.with(py, |m| m.req_blocks(request_id).is_some())
    }

    fn __getitem__<'py>(&self, py: Python<'py>, request_id: &str) -> PyResult<Bound<'py, PyList>> {
        self.lookup(py, request_id)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(request_id.to_owned()))
    }

    #[pyo3(signature = (request_id, default=None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(self.lookup(py, request_id)?.map(Bound::into_any).or(default))
    }
}

/// `KVCacheManager` facade of a Rust scheduler's manager: what KV connectors
/// (`bind_kv_cache_manager`) and callers inspecting the scheduler read.
#[pyclass(frozen, module = "vllm._rust_core", name = "SchedulerKVCacheManager")]
pub struct SchedulerKvManagerView {
    owner: Owner,
}

impl SchedulerKvManagerView {
    pub fn new(owner: Arc<SchedulerKv>) -> Self {
        Self {
            owner: Owner::Scheduler(owner),
        }
    }
}

#[pymethods]
impl SchedulerKvManagerView {
    #[getter]
    fn block_pool(&self, py: Python<'_>) -> BlockPoolView {
        BlockPoolView::new(self.owner.clone_ref(py))
    }

    #[getter]
    fn coordinator(&self, py: Python<'_>) -> CoordinatorView {
        CoordinatorView::new(self.owner.clone_ref(py))
    }

    #[getter]
    fn kv_cache_config(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.owner.kv_cache_config(py)
    }

    #[getter]
    fn enable_caching(&self, py: Python<'_>) -> PyResult<bool> {
        self.owner.with(py, |m| m.enable_caching)
    }

    #[getter]
    fn num_kv_cache_groups(&self, py: Python<'_>) -> PyResult<usize> {
        self.owner.with(py, |m| m.num_kv_cache_groups)
    }

    #[getter]
    fn max_model_len(&self, py: Python<'_>) -> PyResult<usize> {
        self.owner.with(py, |m| m.max_model_len)
    }

    /// Get the KV cache usage (between 0.0 and 1.0).
    #[getter]
    fn usage(&self, py: Python<'_>) -> PyResult<f64> {
        self.owner.with(py, |m| m.usage())
    }

    /// Pre-constructed KVCacheBlocks with no blocks.
    #[getter]
    fn empty_kv_cache_blocks(&self, py: Python<'_>) -> PyResult<KVCacheBlocksView> {
        let blocks = self.owner.with(py, |m| m.empty_kv_cache_blocks())?;
        Ok(KVCacheBlocksView::new(None, blocks))
    }

    /// Get the blocks of a request.
    fn get_blocks(&self, py: Python<'_>, request_id: &str) -> PyResult<KVCacheBlocksView> {
        let blocks = self.owner.with(py, |m| m.get_blocks(request_id))?;
        Ok(KVCacheBlocksView::new(
            Some(self.owner.clone_ref(py)),
            blocks,
        ))
    }

    /// Get the block ids of a request.
    fn get_block_ids<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let tables = self.owner.with(py, |m| {
            m.get_block_tables(request_id)
                .into_iter()
                .map(<[BlockId]>::to_vec)
                .collect::<Vec<_>>()
        })?;
        group_tuple(py, tables)
    }

    /// Get block ids covering the request's computed tokens.
    fn get_block_ids_for_computed_tokens<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let tables = self.owner.with(py, |m| {
            m.get_block_ids_for_computed_tokens(request_id, num_computed_tokens)
        })?;
        group_tuple(py, tables)
    }

    /// `get_hisparse_coordinator(kv_cache_manager, import_step_tokens=...)`.
    #[pyo3(signature = (import_step_tokens=None))]
    fn hisparse_coordinator(
        &self,
        py: Python<'_>,
        import_step_tokens: Option<usize>,
    ) -> PyResult<super::hisparse::HiSparseCoordinatorView> {
        super::hisparse::HiSparseCoordinatorView::bind(
            py,
            self.owner.clone_ref(py),
            import_step_tokens,
        )
    }
}
