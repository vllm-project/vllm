// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Bindings for the Mooncake store connector hot paths
//! (`vllm_engine_core_connectors::mooncake_store`).
//!
//! The Python objects stay the source of truth: the Rust objects are built
//! from a constructed `MooncakeStoreCoordinator` / `MooncakeStoreWorker` and
//! raise `NotImplementedError` for layouts this port does not model, in which
//! case the caller keeps the Python path.
//!
//! Calls keep the GIL: they take well under a millisecond, while re-acquiring
//! it from a background thread (the lookup server, the transfer threads) can
//! wait a whole switch interval behind the busy worker main thread.
//!
//! [`scheduler`] holds the native scheduler half of the connector.

pub mod scheduler;

use std::collections::BTreeSet;

use pyo3::exceptions::{PyAssertionError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyList, PyTuple};
use rustc_hash::FxHashSet;
use thiserror_ext::AsReport;
use vllm_engine_core::hash::BlockHash;
use vllm_engine_core::kv_cache::{KvCacheGroupRole, KvCacheSpec, ManagerType, SpecGroup};
use vllm_engine_core_connectors::mooncake_store::{
    ExternalCachedBlockPool, LookupCandidates, LookupGroup, MooncakeStoreError, StoreCoordinator,
    StoreLookup,
};

use crate::kv_cache::config::{convert_spec, to_py_err as kv_cache_py_err};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MooncakeStoreCoordinator>()?;
    m.add_class::<MooncakeStoreLookup>()?;
    m.add_class::<MooncakeLookupCandidates>()?;
    scheduler::register(m)?;
    Ok(())
}

fn to_py_err(err: MooncakeStoreError) -> PyErr {
    match err {
        MooncakeStoreError::NoTailKey { .. } => PyAssertionError::new_err(err.to_report_string()),
        MooncakeStoreError::InvalidArgument { .. } => PyValueError::new_err(err.to_report_string()),
    }
}

const COORDINATOR_MODULE: &str =
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.coordinator";

/// The Python manager classes of the ported [`ManagerType`]s.
const MANAGER_CLASSES: [(&str, ManagerType); 8] = [
    ("FullAttentionManager", ManagerType::FullAttention),
    ("RSWAManager", ManagerType::Rswa),
    ("SlidingWindowManager", ManagerType::SlidingWindow),
    ("CircularBufferManager", ManagerType::CircularBuffer),
    (
        "ChunkedLocalAttentionManager",
        ManagerType::ChunkedLocalAttention,
    ),
    ("MambaManager", ManagerType::Mamba),
    ("CrossAttentionManager", ManagerType::CrossAttention),
    ("SinkFullAttentionManager", ManagerType::SinkFullAttention),
];

static MANAGER_TYPES: PyOnceLock<Vec<(Py<PyAny>, ManagerType)>> = PyOnceLock::new();

/// The [`ManagerType`] of a Python manager class, matched by identity so that
/// custom (platform-registered) managers are rejected.
fn manager_type(manager_cls: &Bound<'_, PyAny>) -> PyResult<ManagerType> {
    let py = manager_cls.py();
    let types = MANAGER_TYPES.get_or_try_init(py, || {
        let module = PyModule::import(py, "vllm.v1.core.single_type_kv_cache_manager")?;
        MANAGER_CLASSES
            .iter()
            .map(|&(name, ty)| Ok((module.getattr(name)?.unbind(), ty)))
            .collect::<PyResult<Vec<_>>>()
    })?;
    types
        .iter()
        .find(|(cls, _)| manager_cls.is(cls.bind(py)))
        .map(|&(_, ty)| ty)
        .ok_or_else(|| {
            PyNotImplementedError::new_err(format!("KV cache manager {manager_cls} is not ported"))
        })
}

/// A spec and its registered manager, checked against the port's own
/// spec-to-manager mapping.
fn spec_and_manager(
    spec: &Bound<'_, PyAny>,
    manager_cls: &Bound<'_, PyAny>,
) -> PyResult<(KvCacheSpec, ManagerType)> {
    let converted = convert_spec(spec, 0)?;
    let manager = manager_type(manager_cls)?;
    let expected =
        ManagerType::for_group(&converted, KvCacheGroupRole::Default).map_err(kv_cache_py_err)?;
    if manager != expected {
        return Err(PyNotImplementedError::new_err(format!(
            "{spec} is registered with {manager_cls}"
        )));
    }
    Ok((converted, manager))
}

/// Read the resolved state of a Python `MooncakeStoreCoordinator`.
fn coordinator_from_py(coord: &Bound<'_, PyAny>) -> PyResult<StoreCoordinator> {
    let py = coord.py();
    let module = PyModule::import(py, COORDINATOR_MODULE)?;
    let coverage_spec = module.getattr("_coverage_spec")?;
    let registry =
        PyModule::import(py, "vllm.v1.kv_cache_spec_registry")?.getattr("KVCacheSpecRegistry")?;

    let mut group_specs = Vec::new();
    for group in coord.getattr("kv_cache_groups")?.try_iter()? {
        let spec = coverage_spec.call1((group?.getattr("kv_cache_spec")?,))?;
        let manager_cls = registry.call_method1("get_manager_class", (&spec,))?;
        group_specs.push(spec_and_manager(&spec, &manager_cls)?);
    }
    let mut attention_groups = Vec::new();
    for group in coord.getattr("attention_groups")?.try_iter()? {
        let group = group?;
        let (spec, manager_type) =
            spec_and_manager(&group.getattr("spec")?, &group.getattr("manager_cls")?)?;
        attention_groups.push(SpecGroup {
            spec,
            group_ids: group.getattr("group_ids")?.extract()?,
            manager_type,
            use_eagle: group.getattr("use_eagle")?.extract()?,
        });
    }
    let eagle_group_ids: BTreeSet<usize> = coord.getattr("eagle_group_ids")?.extract()?;
    Ok(StoreCoordinator::new(
        group_specs,
        attention_groups,
        eagle_group_ids,
        coord.getattr("hash_block_size")?.extract()?,
        coord.getattr("lcm_block_size")?.extract()?,
        coord.getattr("enable_partial_hash_hits")?.extract()?,
    ))
}

/// `bytes(obj)` for a bytes-like object (the limited API has no buffer
/// protocol).
fn to_bytes<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    match obj.cast::<PyBytes>() {
        Ok(b) => Ok(b.clone()),
        Err(_) => Ok(obj.py().get_type::<PyBytes>().call1((obj,))?.cast_into()?),
    }
}

/// Block hashes from a `BlobBlockHashes` (read from its buffer in one go) or
/// any sequence of bytes-like hashes.
fn extract_block_hashes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<BlockHash>> {
    if let (Ok(blob), Ok(hash_len)) = (obj.getattr("_blob"), obj.getattr("_hash_len")) {
        let hash_len: usize = hash_len.extract()?;
        if hash_len == 0 {
            return Ok(Vec::new());
        }
        // `BlobBlockHashes` ignores a trailing partial hash.
        return Ok(to_bytes(&blob)?
            .as_bytes()
            .chunks_exact(hash_len)
            .map(BlockHash::from_slice)
            .collect());
    }
    obj.try_iter()?
        .map(|item| Ok(BlockHash::from_slice(to_bytes(&item?)?.as_bytes())))
        .collect()
}

fn masks_to_py<'py>(py: Python<'py>, masks: Vec<Vec<bool>>) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        masks
            .into_iter()
            .map(|mask| PyList::new(py, mask))
            .collect::<PyResult<Vec<_>>>()?,
    )
}

/// Rust mirror of a Python `MooncakeStoreCoordinator`.
#[pyclass(frozen, module = "vllm._rust_core")]
pub struct MooncakeStoreCoordinator {
    inner: StoreCoordinator,
}

#[pymethods]
impl MooncakeStoreCoordinator {
    #[new]
    fn new(coord: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: coordinator_from_py(coord)?,
        })
    }

    fn align_lookup_length(&self, length: usize) -> usize {
        self.inner.align_lookup_length(length)
    }

    /// `find_longest_cache_hit` over an `ExternalCachedBlockPool` whose exists
    /// set is `exists` (`None`: every block is present).
    #[pyo3(signature = (block_hashes, max_length, exists, *, apply_eagle = true))]
    fn find_longest_cache_hit<'py>(
        &self,
        py: Python<'py>,
        block_hashes: &Bound<'py, PyAny>,
        max_length: usize,
        exists: Option<&Bound<'py, PyAny>>,
        apply_eagle: bool,
    ) -> PyResult<(Bound<'py, PyTuple>, usize)> {
        let block_hashes = extract_block_hashes(block_hashes)?;
        let exists = exists
            .map(|pairs| {
                pairs
                    .try_iter()?
                    .map(|pair| pair?.extract::<(usize, Vec<u8>)>())
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let exists = exists
            .as_ref()
            .map(|pairs| pairs.iter().map(|(g, h)| (*g, h.as_slice())).collect::<FxHashSet<_>>());
        let pool = ExternalCachedBlockPool::new(self.inner.hash_block_size(), exists);
        let (masks, hit_length) =
            self.inner.find_longest_cache_hit(&block_hashes, max_length, &pool, apply_eagle);
        Ok((masks_to_py(py, masks)?, hit_length))
    }

    fn load_mask<'py>(
        &self,
        py: Python<'py>,
        block_hashes: &Bound<'py, PyAny>,
        token_len: usize,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let block_hashes = extract_block_hashes(block_hashes)?;
        let masks = self.inner.load_mask(&block_hashes, token_len);
        masks_to_py(py, masks)
    }

    fn lookup_mask(&self, aligned_token_len: usize) -> Vec<Option<Vec<bool>>> {
        self.inner.lookup_mask(aligned_token_len)
    }
}

/// The keys one lookup probes, with the request's block hashes.
#[pyclass(frozen, module = "vllm._rust_core")]
pub struct MooncakeLookupCandidates {
    block_hashes: Vec<BlockHash>,
    candidates: LookupCandidates,
}

#[pymethods]
impl MooncakeLookupCandidates {
    /// Candidate keys expanded across rank namespaces, in probe order.
    #[getter]
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, &self.candidates.keys)
    }
}

/// The worker-side prefix lookup of a Python `MooncakeStoreWorker`
/// (`lookup` without the selective lease), split around the existence RPC.
#[pyclass(frozen, module = "vllm._rust_core")]
pub struct MooncakeStoreLookup {
    inner: StoreLookup,
}

#[pymethods]
impl MooncakeStoreLookup {
    /// Build from a worker's `coord`, `token_dbs`, `_kv_cache_groups` and
    /// `_lookup_key_prefixes`.
    #[new]
    fn new(worker: &Bound<'_, PyAny>) -> PyResult<Self> {
        let coord = coordinator_from_py(&worker.getattr("coord")?)?;
        let token_dbs = worker.getattr("token_dbs")?;
        let kv_cache_groups = worker.getattr("_kv_cache_groups")?;
        let key_prefixes = worker.getattr("_lookup_key_prefixes")?;
        let mut groups = Vec::new();
        for (g_idx, db) in token_dbs.try_iter()?.enumerate() {
            let spec = kv_cache_groups.get_item(g_idx)?.getattr("kv_cache_spec")?;
            groups.push(LookupGroup {
                prefix_cacheable: spec.getattr("prefix_cacheable")?.extract()?,
                spec_block_size: spec.getattr("block_size")?.extract()?,
                chunk_size: db?.getattr("chunk_size")?.extract()?,
                key_prefixes: key_prefixes.get_item(g_idx)?.extract()?,
            });
        }
        let hash_block_size: usize = worker.getattr("hash_block_size")?.extract()?;
        if hash_block_size != coord.hash_block_size() {
            return Err(PyNotImplementedError::new_err(
                "worker and coordinator hash block sizes differ",
            ));
        }
        Ok(Self {
            inner: StoreLookup::new(coord, groups).map_err(to_py_err)?,
        })
    }

    /// The keys to probe for a `num_tokens` request, or `None` for a miss
    /// that needs no probe.
    fn candidates(
        &self,
        num_tokens: usize,
        block_hashes: &Bound<'_, PyAny>,
    ) -> PyResult<Option<MooncakeLookupCandidates>> {
        let block_hashes = extract_block_hashes(block_hashes)?;
        let candidates = self.inner.candidates(num_tokens, &block_hashes);
        Ok(candidates.map(|candidates| MooncakeLookupCandidates {
            block_hashes,
            candidates,
        }))
    }

    /// The hit length and `(group_id, num_tokens)` tail-key boundaries for
    /// the existence results of `candidates.keys`.
    fn resolve(
        &self,
        candidates: &MooncakeLookupCandidates,
        exists: Vec<i64>,
    ) -> PyResult<(usize, Vec<(usize, usize)>)> {
        let result = self
            .inner
            .resolve(&candidates.block_hashes, &candidates.candidates, &exists)
            .map_err(to_py_err)?;
        Ok((
            result.hit_length,
            result
                .tail_key_boundaries
                .into_iter()
                .map(|b| (b.group_id, b.num_tokens))
                .collect(),
        ))
    }
}
