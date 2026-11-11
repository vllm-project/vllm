// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The native scheduler half of the `MooncakeStoreConnector`
//! (`vllm_engine_core_connectors::mooncake_store_scheduler`) as the Rust
//! scheduler's [`KvConnector`], with what it needs from Python:
//!
//! - its configuration, read from a constructed (scheduler-role) Python
//!   `MooncakeStoreConnector` ([`NativeMooncakeStoreConnector::new`], which
//!   raises `NotImplementedError` for what the port does not support, so the
//!   caller keeps the Python connector);
//! - the prefix lookup, an RPC to worker rank 0 that stays on the Python
//!   `LookupKeyClient` ([`PyLookupClient`]): it keeps per-request lookup
//!   state, timeouts, a circuit breaker and a ZMQ `REQ_CORRELATE` socket, and
//!   runs once per request rather than per step;
//! - the Python `MooncakeStoreConnectorMetadata` the (Python) worker half
//!   reads, built from the Rust metadata ([`MetaConverter`]) with the same
//!   object sharing, so that it pickles to the same bytes.

use std::sync::Arc;

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyList, PySet, PyString, PyTuple};
use rustc_hash::FxHashMap;
use thiserror_ext::AsReport;
use vllm_engine_core::hash::BlockHash;
use vllm_engine_core::kv_cache::{BlockId, BoundaryStateOffload, KvCacheBlocks, KvCacheManager};
use vllm_engine_core::sched::output::{GroupedBlockIds, Opaque, SchedulerOutput};
use vllm_engine_core::sched::request::Request;
use vllm_engine_core::sched::scheduler::{
    KvConnector, KvConnectorBlockState, KvConnectorOutput, RequestFinished,
};
use vllm_engine_core_connectors::mooncake_store::{LookupResult, TailKeyBoundary};
use vllm_engine_core_connectors::mooncake_store_scheduler::{
    BoundaryPut, KvRole, LoadSpec, LookupClient, MooncakeStoreConnectorMetadata,
    MooncakeStoreScheduler, MooncakeStoreSchedulerConfig, MooncakeStoreWorkerMetadata,
    ReqBlockHashes, ReqMeta, SavePinPolicy, SetOp, StoreGeometry,
};

const MOONCAKE_STORE: &str = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store";

/// The Python classes the metadata is built from.
struct DataTypes {
    req_meta: Py<PyAny>,
    load_spec: Py<PyAny>,
    boundary_put: Py<PyAny>,
    tail_key_boundary: Py<PyAny>,
    metadata: Py<PyAny>,
    worker_metadata: Py<PyAny>,
}

static DATA_TYPES: PyOnceLock<DataTypes> = PyOnceLock::new();

fn data_types(py: Python<'_>) -> PyResult<&DataTypes> {
    DATA_TYPES.get_or_try_init(py, || {
        let data = PyModule::import(py, format!("{MOONCAKE_STORE}.data"))?;
        let get = |name: &str| data.getattr(name).map(Bound::unbind);
        Ok(DataTypes {
            req_meta: get("ReqMeta")?,
            load_spec: get("LoadSpec")?,
            boundary_put: get("BoundaryPut")?,
            tail_key_boundary: get("TailKeyBoundary")?,
            metadata: get("MooncakeStoreConnectorMetadata")?,
            worker_metadata: get("MooncakeStoreWorkerMetadata")?,
        })
    })
}

/// What the native connector needs from the binding layer.
pub trait BindingContext: Send + Sync {
    /// The Python request the binding holds for `request_id`, if any, with
    /// its `block_hashes` brought up to date with `request` when given.
    /// Metadata entries reference the request's own id object and
    /// `block_hashes` list, as in Python.
    fn py_request<'py>(
        &self,
        _py: Python<'py>,
        _request_id: &str,
        _request: Option<&Request>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(None)
    }

    /// Abort the current scheduler call with a Python exception.
    fn raise(&self, err: PyErr) -> ! {
        panic!("{}", err.to_report_string())
    }
}

/// A [`BindingContext`] without Python requests: the metadata gets fresh id
/// strings and hash lists, shared within one metadata as Python shares them.
pub struct NoPyRequests;

impl BindingContext for NoPyRequests {}

/// Read the native connector's configuration from a constructed Python
/// `MooncakeStoreConnector` (scheduler role), whose `MooncakeStoreScheduler`
/// has resolved it. Raises `NotImplementedError` for what the native port
/// does not support.
pub fn config_from_py(
    connector: &Bound<'_, PyAny>,
    save_pin_policy: SavePinPolicy,
) -> PyResult<MooncakeStoreSchedulerConfig> {
    let py = connector.py();
    let unsupported = |what: &str| {
        Err(PyNotImplementedError::new_err(format!(
            "The native MooncakeStoreConnector scheduler does not support {what}"
        )))
    };
    let cls = PyModule::import(py, format!("{MOONCAKE_STORE}.connector"))?
        .getattr("MooncakeStoreConnector")?;
    if !connector.get_type().is(&cls) {
        // A MultiConnector, or a subclass that may override hooks.
        return unsupported(&format!("{}", connector.get_type()));
    }
    let cs = connector.getattr("connector_scheduler")?;
    if cs.is_none() {
        return Err(PyValueError::new_err("not a scheduler-role connector"));
    }
    if cs.getattr("enable_kv_events")?.is_truthy()? {
        // TODO: `RequestTracker.token_ids`, lookahead `token_ids` and the
        // `MooncakeStoreKVEvents` aggregation of `take_events`.
        return unsupported("KV cache events");
    }
    // The state is carried over only at construction.
    let len = |name: &str| -> PyResult<usize> { cs.getattr(name)?.len() };
    if cs.getattr("_next_store_job_id")?.extract::<u64>()? != 0
        || len("_pinned_saves")? != 0
        || len("_request_trackers")? != 0
        || len("_unfinished_requests")? != 0
        || len("load_specs")? != 0
    {
        return Err(PyValueError::new_err("the connector has already been used"));
    }

    let kv_role: String = cs.getattr("kv_role")?.extract()?;
    let Some(kv_role) = KvRole::from_name(&kv_role) else {
        return unsupported(&format!("kv_role {kv_role:?}"));
    };
    let flag = |name: &str| -> PyResult<bool> { cs.getattr(name)?.is_truthy() };
    let ids = |obj: Bound<'_, PyAny>| -> PyResult<Vec<usize>> {
        let mut ids = obj.try_iter()?.map(|g| g?.extract()).collect::<PyResult<Vec<usize>>>()?;
        ids.sort_unstable();
        Ok(ids)
    };
    let boundary_state_group_ids = ids(cs.getattr("_boundary_state_group_ids")?)?;
    let coord = cs.getattr("_store_coord")?;
    let store = if coord.is_none() {
        None
    } else {
        let mamba_group_ids = ids(coord.getattr("mamba_group_ids")?)?;
        let mut eagle_proof_margin_by_group: Vec<(usize, usize)> = coord
            .getattr("eagle_proof_margin_by_group")?
            .call_method0("items")?
            .try_iter()?
            .map(|item| item?.extract())
            .collect::<PyResult<_>>()?;
        eagle_proof_margin_by_group.sort_unstable();
        Some(StoreGeometry {
            group_block_sizes: coord
                .getattr("kv_cache_groups")?
                .try_iter()?
                .map(|g| g?.getattr("kv_cache_spec")?.getattr("block_size")?.extract())
                .collect::<PyResult<_>>()?,
            mamba_group_ids,
            hash_block_size: coord.getattr("hash_block_size")?.extract()?,
            lcm_block_size: coord.getattr("lcm_block_size")?.extract()?,
            use_eagle: coord.getattr("use_eagle")?.is_truthy()?,
            num_hash_lookahead_tokens: coord.getattr("num_hash_lookahead_tokens")?.extract()?,
            eagle_proof_margin_by_group,
            enable_partial_hash_hits: coord.getattr("enable_partial_hash_hits")?.is_truthy()?,
        })
    };
    let config = MooncakeStoreSchedulerConfig {
        kv_role,
        load_async: flag("load_async")?,
        lookup_async: flag("lookup_async")?,
        enable_lookup: flag("enable_lookup")?,
        save_decode_cache: flag("save_decode_cache")?,
        store_group_ids: cs
            .getattr("_store_group_ids")?
            .try_iter()?
            .map(|g| g?.extract())
            .collect::<PyResult<_>>()?,
        block_size: cs.getattr("_block_size")?.extract()?,
        hash_block_size: cs.getattr("_hash_block_size")?.extract()?,
        boundary_state_group_ids,
        store,
        num_workers: cs.getattr("_num_workers")?.extract()?,
        save_pin_policy,
    };

    // The port derives these itself; they must agree with Python's.
    let capacity_only =
        config.kv_role == KvRole::Consumer && !config.enable_lookup && !config.save_decode_cache;
    let partial_hits = config.store.as_ref().is_some_and(|s| s.enable_partial_hash_hits);
    let group_map: Vec<(usize, usize)> = cs
        .getattr("_store_group_id_by_kv_cache_group_id")?
        .call_method0("items")?
        .try_iter()?
        .map(|item| item?.extract())
        .collect::<PyResult<_>>()?;
    let expected_map: Vec<(usize, usize)> =
        config.store_group_ids.iter().enumerate().map(|(s, &g)| (g, s)).collect();
    if flag("_capacity_only")? != capacity_only
        || capacity_only != config.store.is_none()
        || flag("enable_partial_hash_hits")? != partial_hits
        || group_map != expected_map
        || config
            .store
            .as_ref()
            .is_some_and(|s| s.hash_block_size != config.hash_block_size)
    {
        return unsupported("a connector state that differs from MooncakeStoreScheduler's");
    }
    Ok(config)
}

/// `request.block_hashes` of the binding's Python request, or a new list of
/// the Rust request's hashes.
fn py_block_hashes<'py>(
    py: Python<'py>,
    py_request: Option<&Bound<'py, PyAny>>,
    hashes: &[BlockHash],
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(obj) = py_request {
        return obj.getattr(intern!(py, "block_hashes"));
    }
    Ok(PyList::new(py, hashes.iter().map(|h| PyBytes::new(py, h.as_bytes())))?.into_any())
}

/// The Python `LookupKeyClient` as the native connector's [`LookupClient`].
pub struct PyLookupClient {
    /// The Python `MooncakeStoreScheduler`, whose `client` and `reset_store`
    /// are used.
    scheduler: Py<PyAny>,
    context: Arc<dyn BindingContext>,
}

impl PyLookupClient {
    fn call<R>(&self, f: impl FnOnce(Python<'_>, &Bound<'_, PyAny>) -> PyResult<R>) -> R {
        Python::attach(|py| {
            let client = self.scheduler.bind(py).getattr(intern!(py, "client"));
            client
                .and_then(|client| f(py, &client))
                .unwrap_or_else(|err| self.context.raise(err))
        })
    }

    fn request_args<'py>(
        &self,
        py: Python<'py>,
        request: &Request,
    ) -> PyResult<(Bound<'py, PyAny>, usize, Bound<'py, PyAny>)> {
        let obj = self.context.py_request(py, &request.request_id, Some(request))?;
        let request_id = match &obj {
            Some(obj) => obj.getattr(intern!(py, "request_id"))?,
            None => PyString::new(py, &request.request_id).into_any(),
        };
        let hashes = py_block_hashes(py, obj.as_ref(), &request.block_hashes)?;
        Ok((request_id, request.num_tokens(), hashes))
    }
}

impl LookupClient for PyLookupClient {
    fn prefetch(&mut self, request: &Request) -> bool {
        self.call(|py, client| {
            let args = self.request_args(py, request)?;
            client.call_method1(intern!(py, "prefetch"), args)?.is_truthy()
        })
    }

    fn lookup(&mut self, request: &Request, non_block: bool) -> Option<LookupResult> {
        self.call(|py, client| {
            let (request_id, num_tokens, hashes) = self.request_args(py, request)?;
            let result = client.call_method1(
                intern!(py, "lookup"),
                (request_id, num_tokens, hashes, non_block),
            )?;
            if result.is_none() {
                return Ok(None);
            }
            let tail_key_boundaries = result
                .getattr(intern!(py, "tail_key_boundaries"))?
                .try_iter()?
                .map(|b| {
                    let b = b?;
                    Ok(TailKeyBoundary {
                        group_id: b.getattr(intern!(py, "group_id"))?.extract()?,
                        num_tokens: b.getattr(intern!(py, "num_tokens"))?.extract()?,
                    })
                })
                .collect::<PyResult<_>>()?;
            Ok(Some(LookupResult {
                hit_length: result.getattr(intern!(py, "hit_length"))?.extract()?,
                tail_key_boundaries,
            }))
        })
    }

    fn discard(&mut self, request_id: &str) {
        self.call(|py, client| {
            client.call_method1(intern!(py, "discard"), (request_id,))?;
            Ok(())
        })
    }

    fn reset(&mut self) -> bool {
        // `reset_store` logs the outcome and turns RPC errors into False.
        Python::attach(|py| {
            self.scheduler
                .bind(py)
                .call_method0(intern!(py, "reset_store"))
                .and_then(|ok| ok.is_truthy())
                .unwrap_or_else(|err| self.context.raise(err))
        })
    }
}

/// Builds the Python `MooncakeStoreConnectorMetadata` of each step from the
/// Rust one.
///
/// Python shares objects the pickled metadata then references once: every
/// metadata holds the scheduler's live `_unfinished_request_ids` set, an
/// entry holds its request's id object and live `block_hashes` list, entries
/// may share a tracker's block-id tuple, and a finish-time copy of the hashes
/// shares their objects. The converter keeps one persistent set that it
/// changes as Python changed its own (same operations, so the same iteration
/// order), and within one metadata shares what Python shares.
pub struct MetaConverter {
    unfinished_request_ids: Py<PySet>,
    /// The id objects in `unfinished_request_ids` of requests the binding
    /// holds no Python request for.
    interned_ids: FxHashMap<String, Py<PyString>>,
}

/// The Python objects one metadata shares, by request id.
struct Shared<'a, 'py, 'r> {
    py: Python<'py>,
    requests: &'a dyn Fn(&str) -> Option<&'r Request>,
    context: &'a dyn BindingContext,
    interned_ids: &'a FxHashMap<String, Py<PyString>>,
    py_requests: FxHashMap<String, Option<Bound<'py, PyAny>>>,
    ids: FxHashMap<String, Bound<'py, PyString>>,
    hashes: FxHashMap<String, Bound<'py, PyAny>>,
    block_ids: FxHashMap<*const GroupedBlockIds, Bound<'py, PyTuple>>,
}

impl<'py> Shared<'_, 'py, '_> {
    fn py_request(&mut self, id: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Some(obj) = self.py_requests.get(id) {
            return Ok(obj.clone());
        }
        let obj = self.context.py_request(self.py, id, (self.requests)(id))?;
        self.py_requests.insert(id.to_owned(), obj.clone());
        Ok(obj)
    }

    /// The request's id object: the Python request's own, else the one in
    /// the unfinished set, else a new one.
    fn id_obj(&mut self, id: &str) -> PyResult<Bound<'py, PyString>> {
        if let Some(obj) = self.ids.get(id) {
            return Ok(obj.clone());
        }
        let py = self.py;
        let obj = match self.py_request(id)? {
            Some(req) => req.getattr(intern!(py, "request_id"))?.cast_into::<PyString>()?,
            None => match self.interned_ids.get(id) {
                Some(obj) => obj.bind(py).clone(),
                None => PyString::new(py, id),
            },
        };
        self.ids.insert(id.to_owned(), obj.clone());
        Ok(obj)
    }

    fn block_hashes(&mut self, m: &ReqMeta) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let base = match self.hashes.get(&m.req_id) {
            Some(list) => list.clone(),
            None => {
                let list = match &m.block_hashes {
                    ReqBlockHashes::Request { len } => {
                        let obj = self.py_request(&m.req_id)?;
                        let hashes = match (&obj, (self.requests)(&m.req_id)) {
                            (Some(_), _) => &[][..],
                            (None, Some(request)) => &request.block_hashes[..*len],
                            (None, None) => {
                                return Err(PyValueError::new_err(format!(
                                    "no block hashes for request {}",
                                    m.req_id
                                )));
                            }
                        };
                        py_block_hashes(py, obj.as_ref(), hashes)?
                    }
                    ReqBlockHashes::Finished { hashes, .. } => py_block_hashes(py, None, hashes)?,
                };
                self.hashes.insert(m.req_id.clone(), list.clone());
                list
            }
        };
        if let ReqBlockHashes::Finished { copy: true, .. } = m.block_hashes {
            // `list(request.block_hashes)`: the same hash objects.
            return Ok(
                PyList::new(py, base.try_iter()?.collect::<PyResult<Vec<_>>>()?)?.into_any(),
            );
        }
        Ok(base)
    }

    fn block_ids(&mut self, block_ids: &Arc<GroupedBlockIds>) -> PyResult<Bound<'py, PyTuple>> {
        let key = Arc::as_ptr(block_ids);
        if let Some(tuple) = self.block_ids.get(&key) {
            return Ok(tuple.clone());
        }
        let py = self.py;
        let tuple = PyTuple::new(
            py,
            block_ids.iter().map(|g| PyList::new(py, g)).collect::<PyResult<Vec<_>>>()?,
        )?;
        self.block_ids.insert(key, tuple.clone());
        Ok(tuple)
    }
}

impl MetaConverter {
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            unfinished_request_ids: PySet::empty(py)?.unbind(),
            interned_ids: FxHashMap::default(),
        })
    }

    /// The Python metadata of `meta`.
    ///
    /// `requests` resolves the scheduler's Rust requests, for entries of
    /// requests `context` has no Python request for. `preempted_req_ids` is
    /// the Python `SchedulerOutput`'s set, which Python's metadata shares
    /// when it is non-empty (`scheduler_output.preempted_req_ids or set()`).
    pub fn convert<'py, 'r>(
        &mut self,
        py: Python<'py>,
        meta: &MooncakeStoreConnectorMetadata,
        requests: &dyn Fn(&str) -> Option<&'r Request>,
        context: &dyn BindingContext,
        preempted_req_ids: Option<&Bound<'py, PySet>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let types = data_types(py)?;
        let unfinished = self.unfinished_request_ids.bind(py).clone();
        // Replay the set operations first: the entries share their ids.
        let mut interned_changes = Vec::new();
        {
            let mut shared = Shared {
                py,
                requests,
                context,
                interned_ids: &self.interned_ids,
                py_requests: FxHashMap::default(),
                ids: FxHashMap::default(),
                hashes: FxHashMap::default(),
                block_ids: FxHashMap::default(),
            };
            for op in &meta.unfinished_request_id_ops {
                match op {
                    SetOp::Add(id) => {
                        let obj = shared.id_obj(id)?;
                        if shared.py_request(id)?.is_none() {
                            interned_changes.push((id.clone(), Some(obj.clone().unbind())));
                        }
                        unfinished.add(obj)?;
                    }
                    SetOp::Discard(id) => {
                        unfinished.discard(id)?;
                        interned_changes.push((id.clone(), None));
                    }
                }
            }
        }
        for (id, obj) in interned_changes {
            match obj {
                Some(obj) => self.interned_ids.insert(id, obj),
                None => self.interned_ids.remove(&id),
            };
        }

        let mut shared = Shared {
            py,
            requests,
            context,
            interned_ids: &self.interned_ids,
            py_requests: FxHashMap::default(),
            ids: FxHashMap::default(),
            hashes: FxHashMap::default(),
            block_ids: FxHashMap::default(),
        };
        let py_metas = PyList::empty(py);
        for m in &meta.requests {
            let req_id = shared.id_obj(&m.req_id)?;
            let block_hashes = shared.block_hashes(m)?;
            let block_ids = shared.block_ids(&m.block_ids)?;
            py_metas.append(req_meta_to_py(
                py,
                types,
                m,
                req_id,
                block_ids,
                block_hashes,
            )?)?;
        }

        let preempted = match preempted_req_ids {
            Some(set) if !set.is_empty() => set.clone(),
            _ => {
                let set = PySet::empty(py)?;
                for id in &meta.preempted_req_ids {
                    set.add(shared.id_obj(id)?)?;
                }
                set
            }
        };
        let obj = types.metadata.bind(py).call1((unfinished, preempted))?;
        obj.setattr(intern!(py, "requests"), py_metas)?;
        Ok(obj)
    }
}

fn req_meta_to_py<'py>(
    py: Python<'py>,
    types: &DataTypes,
    m: &ReqMeta,
    req_id: Bound<'py, PyString>,
    block_ids: Bound<'py, PyTuple>,
    block_hashes: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let load_spec = m.load_spec.as_ref().map(|s| load_spec_to_py(py, types, s)).transpose()?;
    let boundary_puts = m
        .boundary_puts
        .as_ref()
        .map(|puts| -> PyResult<_> {
            let boundary_put = types.boundary_put.bind(py);
            PyList::new(
                py,
                puts.iter()
                    .map(
                        |&BoundaryPut {
                             group_id,
                             block_id,
                             num_tokens,
                         }| {
                            boundary_put.call1((group_id, block_id, num_tokens))
                        },
                    )
                    .collect::<PyResult<Vec<_>>>()?,
            )
        })
        .transpose()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "req_id"), req_id)?;
    kwargs.set_item(intern!(py, "token_len_chunk"), m.token_len_chunk)?;
    kwargs.set_item(intern!(py, "block_ids"), block_ids)?;
    kwargs.set_item(intern!(py, "block_hashes"), block_hashes)?;
    kwargs.set_item(intern!(py, "can_save"), m.can_save)?;
    kwargs.set_item(intern!(py, "load_spec"), load_spec)?;
    kwargs.set_item(intern!(py, "token_ids_start"), m.token_ids_start)?;
    kwargs.set_item(intern!(py, "num_prompt_tokens"), m.num_prompt_tokens)?;
    kwargs.set_item(intern!(py, "prefill_end_tokens"), m.prefill_end_tokens)?;
    kwargs.set_item(intern!(py, "store_job_id"), m.store_job_id)?;
    kwargs.set_item(intern!(py, "boundary_puts"), boundary_puts)?;
    kwargs.set_item(intern!(py, "completed_token_len"), m.completed_token_len)?;
    kwargs.set_item(intern!(py, "publish_partial_tail"), m.publish_partial_tail)?;
    types.req_meta.bind(py).call((), Some(&kwargs))
}

fn load_spec_to_py<'py>(
    py: Python<'py>,
    types: &DataTypes,
    s: &LoadSpec,
) -> PyResult<Bound<'py, PyAny>> {
    let tail_key_boundary = types.tail_key_boundary.bind(py);
    let tails = PyTuple::new(
        py,
        s.tail_key_boundaries
            .iter()
            .map(|b| tail_key_boundary.call1((b.group_id, b.num_tokens)))
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "vllm_cached_tokens"), s.vllm_cached_tokens)?;
    kwargs.set_item(intern!(py, "kvpool_cached_tokens"), s.kvpool_cached_tokens)?;
    kwargs.set_item(intern!(py, "can_load"), s.can_load)?;
    kwargs.set_item(intern!(py, "token_len"), s.token_len)?;
    kwargs.set_item(intern!(py, "tail_key_boundaries"), tails)?;
    types.load_spec.bind(py).call((), Some(&kwargs))
}

/// The `MooncakeStoreWorkerMetadata` of a Python `KVConnectorOutput`, `None`
/// for other worker metadata (which Python ignores).
pub fn worker_meta_from_py(
    output: &Bound<'_, PyAny>,
) -> PyResult<Option<MooncakeStoreWorkerMetadata>> {
    let py = output.py();
    let meta = output.getattr(intern!(py, "kv_connector_worker_meta"))?;
    if !meta.is_instance(data_types(py)?.worker_metadata.bind(py))? {
        return Ok(None);
    }
    Ok(Some(MooncakeStoreWorkerMetadata {
        completed_saves: meta
            .getattr(intern!(py, "completed_saves"))?
            .call_method0(intern!(py, "items"))?
            .try_iter()?
            .map(|item| item?.extract())
            .collect::<PyResult<_>>()?,
        saved_offsets: Vec::new(),
    }))
}

/// The native `MooncakeStoreConnector` scheduler half, for the Rust
/// scheduler binding: hooks run in Rust, the metadata is handed out as the
/// Python object the worker reads, and only lookups call Python.
pub struct NativeMooncakeStoreConnector {
    inner: MooncakeStoreScheduler,
    converter: MetaConverter,
    context: Arc<dyn BindingContext>,
}

impl NativeMooncakeStoreConnector {
    /// Build from a constructed scheduler-role Python `MooncakeStoreConnector`
    /// (whose lookup client this keeps using). `NotImplementedError` when the
    /// port does not support its configuration; keep the Python connector
    /// then.
    pub fn new(
        connector: &Bound<'_, PyAny>,
        context: Arc<dyn BindingContext>,
        save_pin_policy: SavePinPolicy,
    ) -> PyResult<Self> {
        let py = connector.py();
        let config = config_from_py(connector, save_pin_policy)?;
        let client = PyLookupClient {
            scheduler: connector.getattr("connector_scheduler")?.unbind(),
            context: context.clone(),
        };
        Ok(Self {
            inner: MooncakeStoreScheduler::new(config, Box::new(client)),
            converter: MetaConverter::new(py)?,
            context,
        })
    }
}

impl KvConnector for NativeMooncakeStoreConnector {
    fn supports_divergent_local_hybrid_hits(&self) -> bool {
        self.inner.supports_divergent_local_hybrid_hits()
    }

    fn requires_kv_delivery(&self) -> bool {
        self.inner.requires_kv_delivery()
    }

    fn on_new_request(&mut self, request: &Request, kv_cache_manager: &mut KvCacheManager) {
        self.inner.on_new_request(request, kv_cache_manager)
    }

    fn may_load_async(&mut self, request: &Request) -> bool {
        self.inner.may_load_async(request)
    }

    fn get_num_new_matched_tokens(
        &mut self,
        request: &mut Request,
        num_computed_tokens: usize,
        kv_cache_manager: &mut KvCacheManager,
    ) -> (Option<usize>, bool) {
        self.inner
            .get_num_new_matched_tokens(request, num_computed_tokens, kv_cache_manager)
    }

    fn get_loaded_kv_cache_group_ids(&mut self, request: &Request) -> Vec<usize> {
        self.inner.get_loaded_kv_cache_group_ids(request)
    }

    fn update_state_after_alloc(
        &mut self,
        request: &Request,
        blocks: &KvCacheBlocks,
        kv_cache_manager: &mut KvCacheManager,
        num_external_tokens: usize,
    ) {
        self.inner
            .update_state_after_alloc(request, blocks, kv_cache_manager, num_external_tokens)
    }

    fn get_external_cache_hit_sources(
        &mut self,
        request: &Request,
        num_external_tokens: usize,
    ) -> Option<Vec<(String, usize)>> {
        self.inner.get_external_cache_hit_sources(request, num_external_tokens)
    }

    fn has_pending_block_frees(&mut self, kv_cache_manager: &mut KvCacheManager) -> bool {
        self.inner.has_pending_block_frees(kv_cache_manager)
    }

    /// The Python `MooncakeStoreConnectorMetadata`. Its `preempted_req_ids`
    /// is a set of its own: the binding may replace it with the Python
    /// `SchedulerOutput`'s set when that one is non-empty, which Python
    /// shares.
    fn build_connector_meta(
        &mut self,
        scheduler_output: &SchedulerOutput,
        block_state: &mut KvConnectorBlockState<'_>,
    ) -> Opaque {
        let meta = self.inner.build_meta(scheduler_output, block_state);
        let requests = block_state.requests;
        let context = &*self.context;
        Python::attach(|py| {
            let obj = self
                .converter
                .convert(py, &meta, &|id| requests.get(id), context, None)
                .unwrap_or_else(|err| context.raise(err));
            Opaque::new(obj.unbind())
        })
    }

    fn register_finished_partial_tail(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        partial_tail_offloads: &[BoundaryStateOffload],
        kv_cache_manager: &mut KvCacheManager,
    ) -> bool {
        self.inner.register_finished_partial_tail(
            request,
            block_ids,
            partial_tail_offloads,
            kv_cache_manager,
        )
    }

    fn request_finished(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        kv_cache_manager: &mut KvCacheManager,
    ) -> RequestFinished {
        self.inner.request_finished(request, block_ids, kv_cache_manager)
    }

    /// Reads the worker metadata of the Python `KVConnectorOutput`.
    fn update_connector_output(
        &mut self,
        kv_connector_output: &KvConnectorOutput,
        kv_cache_manager: &mut KvCacheManager,
    ) {
        let Some(output) = &kv_connector_output.output else {
            return;
        };
        let Some(obj) = output.0.downcast_ref::<Py<PyAny>>() else {
            // Already the Rust worker metadata.
            return self.inner.update_connector_output(kv_connector_output, kv_cache_manager);
        };
        let meta = Python::attach(|py| worker_meta_from_py(obj.bind(py)))
            .unwrap_or_else(|err| self.context.raise(err));
        if let Some(meta) = meta {
            self.inner.update_worker_meta(&meta, kv_cache_manager.block_pool_mut());
        }
    }

    fn get_kv_connector_stats(&mut self, kv_cache_manager: &mut KvCacheManager) -> Option<Opaque> {
        self.inner.get_kv_connector_stats(kv_cache_manager)
    }

    fn take_events(&mut self, kv_cache_manager: &mut KvCacheManager) -> Option<Opaque> {
        self.inner.take_events(kv_cache_manager)
    }

    fn has_pending_push_work(&mut self, kv_cache_manager: &mut KvCacheManager) -> bool {
        self.inner.has_pending_push_work(kv_cache_manager)
    }

    fn reset_cache(&mut self, kv_cache_manager: &mut KvCacheManager) -> Option<bool> {
        self.inner.reset_cache(kv_cache_manager)
    }
}

/// Why the native scheduler half cannot replace `connector` (a constructed
/// scheduler-role `MooncakeStoreConnector`), or `None` when it can.
#[pyfunction]
pub fn mooncake_store_native_scheduler_unsupported(
    connector: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    let py = connector.py();
    match NativeMooncakeStoreConnector::new(
        connector,
        Arc::new(NoPyRequests),
        SavePinPolicy::WholeTable,
    ) {
        Ok(_) => Ok(None),
        Err(err) if err.is_instance_of::<PyNotImplementedError>(py) => {
            Ok(Some(err.value(py).str()?.extract()?))
        }
        Err(err) => Err(err),
    }
}

/// A Rust `MooncakeStoreConnectorMetadata` (testing aid): parsed from the
/// canonical form `gen_mooncake_traces.py` records.
#[pyclass(frozen, module = "vllm._rust_core")]
pub struct RustMooncakeStoreMetadata {
    inner: MooncakeStoreConnectorMetadata,
}

fn get<'py>(d: &Bound<'py, PyAny>, k: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(k)
}

fn parse_canonical(
    canonical: &Bound<'_, PyDict>,
    finished_hashes: &Bound<'_, PyDict>,
) -> PyResult<MooncakeStoreConnectorMetadata> {
    let mut requests: Vec<ReqMeta> = Vec::new();
    for r in canonical.as_any().get_item("requests")?.try_iter()? {
        let r = r?;
        let alias: Option<usize> = get(&r, "alias")?.extract()?;
        let block_ids = match alias {
            Some(j) => requests[j].block_ids.clone(),
            None => Arc::new(get(&r, "block_ids")?.extract::<GroupedBlockIds>()?),
        };
        let hashes = get(&r, "hashes")?;
        let block_hashes = match hashes.get_item(0)?.extract::<String>()?.as_str() {
            "live" => ReqBlockHashes::Request {
                len: hashes.get_item(1)?.extract()?,
            },
            _ => {
                let req_id: String = get(&r, "req_id")?.extract()?;
                let list = finished_hashes
                    .get_item(&req_id)?
                    .ok_or_else(|| PyValueError::new_err(format!("no hashes of {req_id}")))?;
                let snapshot: Vec<BlockHash> = list
                    .try_iter()?
                    .map(|h| Ok(BlockHash::from_slice(h?.cast_into::<PyBytes>()?.as_bytes())))
                    .collect::<PyResult<_>>()?;
                ReqBlockHashes::Finished {
                    hashes: snapshot.into(),
                    copy: hashes.get_item(1)?.extract()?,
                }
            }
        };
        let load_spec = get(&r, "load_spec")?;
        let load_spec = if load_spec.is_none() {
            None
        } else {
            let item = |k: usize| load_spec.get_item(k);
            let (vllm_cached_tokens, kvpool_cached_tokens, can_load, token_len) = (
                item(0)?.extract::<usize>()?,
                item(1)?.extract::<usize>()?,
                item(2)?.extract::<bool>()?,
                item(3)?.extract::<usize>()?,
            );
            let tails: Vec<Vec<usize>> = item(4)?.extract()?;
            Some(LoadSpec {
                vllm_cached_tokens,
                kvpool_cached_tokens,
                can_load,
                token_len,
                tail_key_boundaries: tails
                    .into_iter()
                    .map(|t| TailKeyBoundary {
                        group_id: t[0],
                        num_tokens: t[1],
                    })
                    .collect(),
            })
        };
        let boundary_puts: Option<Vec<Vec<usize>>> = get(&r, "boundary_puts")?.extract()?;
        requests.push(ReqMeta {
            req_id: get(&r, "req_id")?.extract()?,
            token_len_chunk: get(&r, "token_len_chunk")?.extract()?,
            block_ids,
            block_hashes,
            can_save: get(&r, "can_save")?.extract()?,
            load_spec,
            token_ids_start: get(&r, "token_ids_start")?.extract()?,
            num_prompt_tokens: get(&r, "num_prompt_tokens")?.extract()?,
            prefill_end_tokens: get(&r, "prefill_end_tokens")?.extract()?,
            store_job_id: get(&r, "store_job_id")?.extract()?,
            boundary_puts: boundary_puts.map(|puts| {
                puts.into_iter()
                    .map(|p| BoundaryPut {
                        group_id: p[0],
                        block_id: p[1] as BlockId,
                        num_tokens: p[2],
                    })
                    .collect()
            }),
            completed_token_len: get(&r, "completed_token_len")?.extract()?,
            publish_partial_tail: get(&r, "publish_partial_tail")?.extract()?,
        });
    }
    let ops = canonical
        .as_any()
        .get_item("unfinished_ops")?
        .extract::<Vec<Vec<String>>>()?
        .into_iter()
        .map(|op| match op[0].as_str() {
            "add" => Ok(SetOp::Add(op[1].clone())),
            "discard" => Ok(SetOp::Discard(op[1].clone())),
            other => Err(PyValueError::new_err(format!("unknown op {other}"))),
        })
        .collect::<PyResult<_>>()?;
    let preempted: Vec<String> = canonical.as_any().get_item("preempted")?.extract()?;
    Ok(MooncakeStoreConnectorMetadata {
        requests,
        unfinished_request_id_ops: ops,
        preempted_req_ids: preempted.into_iter().collect(),
    })
}

#[pymethods]
impl RustMooncakeStoreMetadata {
    /// Parse the canonical metadata; `finished_hashes` maps the requests of
    /// `["finished", ...]` entries to their block hashes.
    #[new]
    fn new(canonical: &Bound<'_, PyDict>, finished_hashes: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            inner: parse_canonical(canonical, finished_hashes)?,
        })
    }
}

/// A [`BindingContext`] whose Python requests come from a callable
/// (`request_id -> Request | None`).
struct CallableContext(Py<PyAny>);

impl BindingContext for CallableContext {
    fn py_request<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
        _request: Option<&Request>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let obj = self.0.bind(py).call1((request_id,))?;
        Ok((!obj.is_none()).then_some(obj))
    }
}

/// [`MetaConverter`] for Python (testing aid): one per metadata stream.
#[pyclass(module = "vllm._rust_core", name = "MooncakeStoreMetaConverter")]
pub struct PyMetaConverter {
    inner: MetaConverter,
}

#[pymethods]
impl PyMetaConverter {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            inner: MetaConverter::new(py)?,
        })
    }

    /// The Python metadata of `meta`; `py_requests(request_id)` returns the
    /// Python request of a live request.
    #[pyo3(signature = (meta, py_requests, preempted_req_ids = None))]
    fn convert<'py>(
        &mut self,
        py: Python<'py>,
        meta: &RustMooncakeStoreMetadata,
        py_requests: Py<PyAny>,
        preempted_req_ids: Option<Bound<'py, PySet>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let context = CallableContext(py_requests);
        self.inner.convert(
            py,
            &meta.inner,
            &|_| None,
            &context,
            preempted_req_ids.as_ref(),
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        mooncake_store_native_scheduler_unsupported,
        m
    )?)?;
    m.add_class::<RustMooncakeStoreMetadata>()?;
    m.add_class::<PyMetaConverter>()?;
    Ok(())
}
