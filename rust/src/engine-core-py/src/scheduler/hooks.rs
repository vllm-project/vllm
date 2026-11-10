// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The Rust scheduler's collaborator traits over the Python objects: the
//! scheduler-side KV connector and the `StructuredOutputManager`.
//!
//! Each call brings the Python request up to date first ([`sync_request`]),
//! since Python collaborators keep and read the request objects. A Python
//! exception aborts the scheduler call ([`Shared::raise`]).
//!
//! [`PyEcConnector`] and [`PyAuxOutputConnector`] wrap the EC connector and
//! the aux output connector.
//!
//! The native Mooncake store connector replaces the Python one's hooks;
//! [`SchedulerContext`] gives it the binding's Python requests.

use std::sync::{Arc, Mutex};

use indexmap::IndexSet;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};
use vllm_engine_core::kv_cache::{BlockId, BoundaryStateOffload, KvCacheBlocks, KvCacheManager};
use vllm_engine_core::sched::output::{Opaque, SchedulerOutput};
use vllm_engine_core::sched::request::Request;
use vllm_engine_core::sched::scheduler::{
    AuxOutputConnector, EcConnector, GrammarStatus, KvConnector, KvConnectorBlockState,
    KvConnectorOutput, KvTransferParams, RequestFinished, Requests, StructuredOutput,
};

use super::output::{opaque, opaque_obj, scheduler_output_to_py};
use super::request::{Shared, sync_block_hashes, sync_request};
use super::types::types;
use crate::kv_cache::views::{KVCacheBlocksView, Owner, SchedulerKvManagerView};
use crate::mooncake::scheduler::BindingContext;

/// Run `f` with the synced Python request of `request`.
fn with_request<R>(
    shared: &Shared,
    request: &Request,
    f: impl FnOnce(Python<'_>, &Bound<'_, PyAny>) -> PyResult<R>,
) -> R {
    Python::attach(|py| {
        let obj = shared.py_request(py, &request.request_id);
        let obj = obj.bind(py);
        let result = sync_request(obj, request).and_then(|()| f(py, obj));
        shared.check(result)
    })
}

fn block_ids_tuple<'py>(
    py: Python<'py>,
    block_ids: &[Vec<BlockId>],
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        block_ids.iter().map(|g| PyList::new(py, g)).collect::<PyResult<Vec<_>>>()?,
    )
}

/// `KVConnectorBase_V1` (scheduler role) as the Rust scheduler's
/// [`KvConnector`].
pub struct PyKvConnector {
    /// The connector object; the binding may replace it (tests do).
    connector: Arc<Mutex<Py<PyAny>>>,
    shared: Arc<Shared>,
    supports_divergent_local_hybrid_hits: bool,
    requires_kv_delivery: bool,
    kv_cache_manager: Py<SchedulerKvManagerView>,
}

impl PyKvConnector {
    pub fn new(
        connector: Arc<Mutex<Py<PyAny>>>,
        shared: Arc<Shared>,
        kv_cache_manager: Py<SchedulerKvManagerView>,
    ) -> PyResult<Self> {
        let (supports_divergent_local_hybrid_hits, requires_kv_delivery) =
            Python::attach(|py| -> PyResult<_> {
                let c = connector.lock().unwrap().clone_ref(py).into_bound(py);
                Ok((
                    c.getattr("supports_divergent_local_hybrid_hits")?.extract()?,
                    c.getattr("requires_kv_delivery")?.extract()?,
                ))
            })?;
        Ok(Self {
            supports_divergent_local_hybrid_hits,
            requires_kv_delivery,
            connector,
            shared,
            kv_cache_manager,
        })
    }

    /// A connector over `connector` that reports the given settings instead
    /// of reading them from it (for a connector swapped in for another, e.g.
    /// a mock).
    pub fn with_settings(
        connector: Arc<Mutex<Py<PyAny>>>,
        shared: Arc<Shared>,
        kv_cache_manager: Py<SchedulerKvManagerView>,
        supports_divergent_local_hybrid_hits: bool,
        requires_kv_delivery: bool,
    ) -> Self {
        Self {
            connector,
            shared,
            supports_divergent_local_hybrid_hits,
            requires_kv_delivery,
            kv_cache_manager,
        }
    }

    fn obj<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.connector.lock().unwrap().clone_ref(py).into_bound(py)
    }

    fn call<R>(&self, f: impl FnOnce(Python<'_>, &Bound<'_, PyAny>) -> PyResult<R>) -> R {
        Python::attach(|py| self.shared.check(f(py, &self.obj(py))))
    }

    /// Run `f` with the KV cache manager reachable from Python.
    fn scoped<R>(&self, kv_cache_manager: &mut KvCacheManager, f: impl FnOnce() -> R) -> R {
        self.shared.kv.scoped(kv_cache_manager, f)
    }
}

impl KvConnector for PyKvConnector {
    fn supports_divergent_local_hybrid_hits(&self) -> bool {
        self.supports_divergent_local_hybrid_hits
    }

    fn requires_kv_delivery(&self) -> bool {
        self.requires_kv_delivery
    }

    /// The binding calls the Python hook before it builds the Rust request
    /// (`RustSchedulerCore.add_request`): connectors may rewrite the request,
    /// e.g. truncate a P-side prompt.
    fn on_new_request(&mut self, _request: &Request, _kv_cache_manager: &mut KvCacheManager) {}

    fn may_load_async(&mut self, request: &Request) -> bool {
        with_request(&self.shared, request, |py, req| {
            self.obj(py).call_method1(intern!(py, "may_load_async"), (req,))?.is_truthy()
        })
    }

    fn get_num_new_matched_tokens(
        &mut self,
        request: &mut Request,
        num_computed_tokens: usize,
        kv_cache_manager: &mut KvCacheManager,
    ) -> (Option<usize>, bool) {
        let (result, shared_prefix_boundary) = self.scoped(kv_cache_manager, || {
            with_request(&self.shared, request, |py, req| {
                let result = self
                    .obj(py)
                    .call_method1(
                        intern!(py, "get_num_new_matched_tokens"),
                        (req, num_computed_tokens),
                    )?
                    .extract()?;
                let boundary = req.getattr(intern!(py, "shared_prefix_boundary"))?.extract()?;
                Ok((result, boundary))
            })
        });
        request.shared_prefix_boundary = shared_prefix_boundary;
        result
    }

    fn get_loaded_kv_cache_group_ids(&mut self, request: &Request) -> Vec<usize> {
        with_request(&self.shared, request, |py, req| {
            self.obj(py)
                .call_method1(intern!(py, "get_loaded_kv_cache_group_ids"), (req,))?
                .try_iter()?
                .map(|g| g?.extract())
                .collect()
        })
    }

    fn update_state_after_alloc(
        &mut self,
        request: &Request,
        blocks: &KvCacheBlocks,
        kv_cache_manager: &mut KvCacheManager,
        num_external_tokens: usize,
    ) {
        let shared = &self.shared;
        self.shared.kv.scoped(kv_cache_manager, || {
            with_request(shared, request, |py, req| {
                let blocks = KVCacheBlocksView::new(
                    Some(Owner::Scheduler(shared.kv.clone())),
                    blocks.clone(),
                );
                self.obj(py).call_method1(
                    intern!(py, "update_state_after_alloc"),
                    (req, blocks, num_external_tokens),
                )?;
                Ok(())
            })
        })
    }

    fn get_external_cache_hit_sources(
        &mut self,
        request: &Request,
        num_external_tokens: usize,
    ) -> Option<Vec<(String, usize)>> {
        with_request(&self.shared, request, |py, req| {
            let sources = self.obj(py).call_method1(
                intern!(py, "get_external_cache_hit_sources"),
                (req, num_external_tokens),
            )?;
            if sources.is_none() {
                return Ok(None);
            }
            sources
                .cast::<PyDict>()?
                .iter()
                .map(|(source, n)| Ok((source.str()?.extract()?, n.extract()?)))
                .collect::<PyResult<_>>()
                .map(Some)
        })
    }

    fn has_pending_block_frees(&mut self, kv_cache_manager: &mut KvCacheManager) -> bool {
        self.scoped(kv_cache_manager, || {
            self.call(|py, c| c.call_method0(intern!(py, "has_pending_block_frees"))?.is_truthy())
        })
    }

    fn build_connector_meta(
        &mut self,
        scheduler_output: &SchedulerOutput,
        block_state: &mut KvConnectorBlockState<'_>,
    ) -> Opaque {
        let shared = self.shared.clone();
        let requests = block_state.requests;
        let meta = Python::attach(|py| {
            let result = (|| -> PyResult<Bound<'_, PyAny>> {
                let py_so = pending_scheduler_output(
                    py,
                    &shared,
                    scheduler_output,
                    requests,
                    &block_state.req_ids,
                )?;
                let types = types(py)?;
                let offloads = PyDict::new(py);
                for (req_id, entries) in &block_state.boundary_state_offloads {
                    offloads.set_item(req_id, PyList::new(py, entries)?)?;
                }
                let kwargs = PyDict::new(py);
                kwargs.set_item("req_ids", PySet::new(py, &block_state.req_ids)?)?;
                kwargs.set_item(
                    "resolve_block_ids",
                    self.kv_cache_manager.bind(py).getattr("get_block_ids")?,
                )?;
                kwargs.set_item("boundary_state_offloads", offloads)?;
                let state = types.kv_connector_block_state.bind(py).call((), Some(&kwargs))?;
                py_so.setattr(intern!(py, "kv_connector_block_state"), state)?;
                let meta = shared.kv.scoped(block_state.kv_cache_manager, || {
                    self.obj(py).call_method1(intern!(py, "build_connector_meta"), (&py_so,))
                })?;
                // Connector-only block state must not be dispatched to workers.
                py_so.setattr(intern!(py, "kv_connector_block_state"), py.None())?;
                Ok(meta)
            })();
            shared.check(result).unbind()
        });
        Opaque::new(meta)
    }

    fn register_finished_partial_tail(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        partial_tail_offloads: &[BoundaryStateOffload],
        kv_cache_manager: &mut KvCacheManager,
    ) -> bool {
        self.shared.kv.scoped(kv_cache_manager, || {
            with_request(&self.shared, request, |py, req| {
                self.obj(py)
                    .call_method1(
                        intern!(py, "register_finished_partial_tail"),
                        (
                            req,
                            block_ids_tuple(py, block_ids)?,
                            PyList::new(py, partial_tail_offloads)?,
                        ),
                    )?
                    .is_truthy()
            })
        })
    }

    fn request_finished(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        kv_cache_manager: &mut KvCacheManager,
    ) -> RequestFinished {
        self.shared.kv.scoped(kv_cache_manager, || {
            with_request(&self.shared, request, |py, req| {
                let connector = self.obj(py);
                let supports_hma =
                    PyModule::import(py, "vllm.distributed.kv_transfer.kv_connector.v1")?
                        .getattr("SupportsHMA")?;
                let result = if connector.is_instance(&supports_hma)? {
                    connector.call_method1(
                        intern!(py, "request_finished_all_groups"),
                        (req, block_ids_tuple(py, block_ids)?),
                    )?
                } else {
                    // NOTE(Kuntai): We should deprecate this code path after we
                    // enforce all connectors to support HMA.
                    assert_eq!(block_ids.len(), 1);
                    connector.call_method1(
                        intern!(py, "request_finished"),
                        (req, PyList::new(py, &block_ids[0])?),
                    )?
                };
                let (delay_free_blocks, params): (bool, Bound<'_, PyAny>) = result.extract()?;
                let kv_transfer_params = if params.is_none() {
                    None
                } else {
                    let do_remote_prefill = params
                        .call_method1(intern!(py, "get"), ("do_remote_prefill",))?
                        .is_truthy()?;
                    Some(KvTransferParams {
                        params: opaque(params),
                        do_remote_prefill,
                        remote_prefill_cached_tokens: None,
                    })
                };
                Ok(RequestFinished {
                    delay_free_blocks,
                    kv_transfer_params,
                })
            })
        })
    }

    fn update_connector_output(
        &mut self,
        kv_connector_output: &KvConnectorOutput,
        kv_cache_manager: &mut KvCacheManager,
    ) {
        let output = kv_connector_output.output.as_ref().expect("the Python KVConnectorOutput");
        self.scoped(kv_cache_manager, || {
            self.call(|py, c| {
                c.call_method1(
                    intern!(py, "update_connector_output"),
                    (opaque_obj(py, output),),
                )?;
                Ok(())
            })
        })
    }

    fn get_kv_connector_stats(&mut self, kv_cache_manager: &mut KvCacheManager) -> Option<Opaque> {
        self.scoped(kv_cache_manager, || self.kv_connector_stats())
    }

    fn take_events(&mut self, kv_cache_manager: &mut KvCacheManager) -> Option<Opaque> {
        self.scoped(kv_cache_manager, || self.take_connector_events())
    }

    fn has_pending_push_work(&mut self, kv_cache_manager: &mut KvCacheManager) -> bool {
        self.scoped(kv_cache_manager, || {
            self.call(|py, c| c.call_method0(intern!(py, "has_pending_push_work"))?.is_truthy())
        })
    }

    fn reset_cache(&mut self, kv_cache_manager: &mut KvCacheManager) -> Option<bool> {
        self.scoped(kv_cache_manager, || {
            self.call(|py, c| c.call_method0(intern!(py, "reset_cache"))?.extract())
        })
    }
}

impl PyKvConnector {
    fn kv_connector_stats(&self) -> Option<Opaque> {
        self.call(|py, c| {
            let stats = c.call_method0(intern!(py, "get_kv_connector_stats"))?;
            if stats.is_none() || stats.call_method0(intern!(py, "is_empty"))?.is_truthy()? {
                return Ok(None);
            }
            Ok(Some(opaque(stats)))
        })
    }

    fn take_connector_events(&self) -> Option<Opaque> {
        self.call(|py, c| {
            let events = c.call_method0(intern!(py, "take_events"))?;
            if events.is_none() {
                return Ok(None);
            }
            let events = PyList::new(py, events.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
            Ok((!events.is_empty()).then(|| opaque(events.into_any())))
        })
    }
}

/// `StructuredOutputManager` as the Rust scheduler's [`StructuredOutput`].
pub struct PyStructuredOutput {
    manager: Py<PyAny>,
    shared: Arc<Shared>,
}

impl PyStructuredOutput {
    pub fn new(manager: Py<PyAny>, shared: Arc<Shared>) -> Self {
        Self { manager, shared }
    }
}

impl StructuredOutput for PyStructuredOutput {
    fn grammar_status(&mut self, request: &Request) -> GrammarStatus {
        Python::attach(|py| {
            let result = (|| -> PyResult<GrammarStatus> {
                let obj = self.shared.py_request(py, &request.request_id);
                let so_request = obj.bind(py).getattr(intern!(py, "structured_output_request"))?;
                if so_request.is_none() {
                    return Ok(GrammarStatus::Pending);
                }
                let grammar = so_request.getattr(intern!(py, "grammar"))?;
                if grammar.is_none() {
                    return Ok(GrammarStatus::Pending);
                }
                if grammar.is_instance_of::<pyo3::exceptions::PyException>() {
                    return Ok(GrammarStatus::Failed {
                        message: grammar.str()?.extract()?,
                    });
                }
                Ok(GrammarStatus::Ready)
            })();
            self.shared.check(result)
        })
    }

    fn accept_tokens(&mut self, request: &Request, new_token_ids: &[i32]) -> bool {
        let manager = &self.manager;
        with_request(&self.shared, request, |py, req| {
            manager
                .bind(py)
                .call_method1(
                    intern!(py, "accept_tokens"),
                    (req, PyList::new(py, new_token_ids)?),
                )?
                .is_truthy()
        })
    }

    fn trim_spec_tokens(&mut self, request: &Request, spec_token_ids: Vec<i32>) -> Vec<i32> {
        let manager = &self.manager;
        with_request(&self.shared, request, |py, req| {
            manager
                .bind(py)
                .call_method1(
                    intern!(py, "trim_spec_tokens"),
                    (req, PyList::new(py, spec_token_ids)?),
                )?
                .extract()
        })
    }
}

/// The binding as the native Mooncake store connector's [`BindingContext`].
pub struct SchedulerContext(pub Arc<Shared>);

impl BindingContext for SchedulerContext {
    fn py_request<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
        request: Option<&Request>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let obj = match self.0.table.lock().unwrap().get(request_id) {
            Some(entry) => entry.obj.clone_ref(py).into_bound(py),
            None => return Ok(None),
        };
        if let Some(request) = request {
            sync_block_hashes(&obj, request)?;
        }
        Ok(Some(obj))
    }

    fn raise(&self, err: PyErr) -> ! {
        self.0.raise(err)
    }
}

/// The Python `SchedulerOutput` of this step that connectors' metadata
/// builders read, built once (`RustSchedulerCore.schedule` returns it): the
/// requests it covers (and `also_sync`, which a connector reads too) are
/// brought up to date first.
fn pending_scheduler_output<'py>(
    py: Python<'py>,
    shared: &Shared,
    scheduler_output: &SchedulerOutput,
    requests: Requests<'_>,
    also_sync: &IndexSet<String>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(py_so) = shared.pending_output.lock().unwrap().as_ref() {
        return Ok(py_so.bind(py).clone());
    }
    let table = shared.table.lock().unwrap();
    // The connector reads the requests it was handed earlier.
    let ids: IndexSet<&str> = scheduler_output
        .num_scheduled_tokens
        .keys()
        .chain(also_sync)
        .map(String::as_str)
        .collect();
    for id in ids {
        if let (Some(request), Some(entry)) = (requests.get(id), table.get(id)) {
            sync_request(entry.obj.bind(py), request)?;
        }
    }
    let py_so = scheduler_output_to_py(py, scheduler_output, &table, shared.uses_mrope, |id| {
        requests.get(id)
    })?;
    drop(table);
    *shared.pending_output.lock().unwrap() = Some(py_so.clone().unbind());
    Ok(py_so)
}

/// The scheduler-role `ECConnectorBase` as the Rust scheduler's
/// [`EcConnector`].
pub struct PyEcConnector {
    connector: Py<PyAny>,
    shared: Arc<Shared>,
}

impl PyEcConnector {
    pub fn new(connector: Py<PyAny>, shared: Arc<Shared>) -> Self {
        Self { connector, shared }
    }

    fn call<R>(&self, f: impl FnOnce(Python<'_>, &Bound<'_, PyAny>) -> PyResult<R>) -> R {
        Python::attach(|py| self.shared.check(f(py, self.connector.bind(py))))
    }
}

impl EcConnector for PyEcConnector {
    fn ensure_cache_available(&mut self, request: &Request, num_computed_tokens: usize) -> bool {
        with_request(&self.shared, request, |py, req| {
            self.connector
                .bind(py)
                .call_method1(
                    intern!(py, "ensure_cache_available"),
                    (req, num_computed_tokens),
                )?
                .is_truthy()
        })
    }

    fn has_cache_item(&mut self, identifier: &str) -> bool {
        self.call(|py, c| c.call_method1(intern!(py, "has_cache_item"), (identifier,))?.is_truthy())
    }

    fn update_state_after_alloc(&mut self, request: &Request, index: usize) {
        with_request(&self.shared, request, |py, req| {
            self.connector
                .bind(py)
                .call_method1(intern!(py, "update_state_after_alloc"), (req, index))?;
            Ok(())
        })
    }

    fn update_state_after_free(&mut self, request: &Request, index: usize) {
        with_request(&self.shared, request, |py, req| {
            self.connector
                .bind(py)
                .call_method1(intern!(py, "update_state_after_free"), (req, index))?;
            Ok(())
        })
    }

    fn build_connector_meta(
        &mut self,
        scheduler_output: &SchedulerOutput,
        requests: Requests<'_>,
    ) -> Opaque {
        let shared = &self.shared;
        let meta = self.call(|py, c| {
            let py_so =
                pending_scheduler_output(py, shared, scheduler_output, requests, &IndexSet::new())?;
            c.call_method1(intern!(py, "build_connector_meta"), (py_so,)).map(Bound::unbind)
        });
        Opaque::new(meta)
    }

    fn take_unavailable_requests(&mut self) -> IndexSet<String> {
        self.call(|py, c| {
            c.call_method0(intern!(py, "take_unavailable_requests"))?
                .try_iter()?
                .map(|id| id?.extract())
                .collect()
        })
    }

    fn update_connector_output(&mut self, ec_connector_output: &Opaque) {
        self.call(|py, c| {
            c.call_method1(
                intern!(py, "update_connector_output"),
                (opaque_obj(py, ec_connector_output),),
            )?;
            Ok(())
        })
    }

    fn get_ec_connector_stats(&mut self) -> Option<Opaque> {
        self.call(|py, c| {
            let stats = c.call_method0(intern!(py, "get_ec_connector_stats"))?;
            if stats.is_none() || stats.call_method0(intern!(py, "is_empty"))?.is_truthy()? {
                return Ok(None);
            }
            Ok(Some(opaque(stats)))
        })
    }

    fn request_finished(&mut self, request: &Request) -> (bool, Option<Opaque>) {
        with_request(&self.shared, request, |py, req| {
            let (delay, params): (bool, Bound<'_, PyAny>) = self
                .connector
                .bind(py)
                .call_method1(intern!(py, "request_finished"), (req,))?
                .extract()?;
            Ok((delay, (!params.is_none()).then(|| opaque(params))))
        })
    }

    fn has_pending_push_work(&mut self) -> bool {
        self.call(|py, c| c.call_method0(intern!(py, "has_pending_push_work"))?.is_truthy())
    }
}

/// `AuxOutputSchedulerConnector` as the Rust scheduler's
/// [`AuxOutputConnector`]. It reads the binding's `KVCacheManager` facade.
pub struct PyAuxOutputConnector {
    connector: Py<PyAny>,
    shared: Arc<Shared>,
}

impl PyAuxOutputConnector {
    pub fn new(connector: Py<PyAny>, shared: Arc<Shared>) -> Self {
        Self { connector, shared }
    }
}

impl AuxOutputConnector for PyAuxOutputConnector {
    fn max_cache_hit(&mut self, request: &Request, kv_cache_manager: &mut KvCacheManager) -> usize {
        self.shared.kv.scoped(kv_cache_manager, || {
            with_request(&self.shared, request, |py, req| {
                self.connector
                    .bind(py)
                    .call_method1(intern!(py, "max_cache_hit"), (req,))?
                    .extract()
            })
        })
    }

    fn cap_external_hit(
        &mut self,
        request: &Request,
        num_local: usize,
        num_external: usize,
        kv_cache_manager: &mut KvCacheManager,
    ) -> usize {
        self.shared.kv.scoped(kv_cache_manager, || {
            with_request(&self.shared, request, |py, req| {
                self.connector
                    .bind(py)
                    .call_method1(
                        intern!(py, "cap_external_hit"),
                        (req, num_local, num_external),
                    )?
                    .extract()
            })
        })
    }

    fn request_loading(&mut self, request: &Request, num_tokens: usize) {
        with_request(&self.shared, request, |py, req| {
            self.connector
                .bind(py)
                .call_method1(intern!(py, "request_loading"), (req, num_tokens))?;
            Ok(())
        })
    }

    /// The connector reads `requests[request_id]` of the scheduled requests
    /// and of those it sent hashes for; every held request is brought up to
    /// date.
    // TODO: sync only the requests the connector tracks.
    fn build_connector_meta(
        &mut self,
        scheduler_output: &SchedulerOutput,
        requests: Requests<'_>,
        kv_cache_manager: &mut KvCacheManager,
    ) -> Opaque {
        let shared = &self.shared;
        let meta = shared.kv.scoped(kv_cache_manager, || {
            Python::attach(|py| {
                let result = (|| -> PyResult<Py<PyAny>> {
                    let py_so = pending_scheduler_output(
                        py,
                        shared,
                        scheduler_output,
                        requests,
                        &IndexSet::new(),
                    )?;
                    let py_requests = PyDict::new(py);
                    {
                        let table = shared.table.lock().unwrap();
                        for (id, entry) in table.iter() {
                            let Some(request) = requests.get(id) else {
                                continue;
                            };
                            let obj = entry.obj.bind(py);
                            sync_request(obj, request)?;
                            py_requests.set_item(entry.id.bind(py), obj)?;
                        }
                    }
                    Ok(self
                        .connector
                        .bind(py)
                        .call_method1(intern!(py, "build_connector_meta"), (py_so, py_requests))?
                        .unbind())
                })();
                shared.check(result)
            })
        });
        Opaque::new(meta)
    }

    fn request_finished(&mut self, request: &Request, kv_cache_manager: &mut KvCacheManager) {
        self.shared.kv.scoped(kv_cache_manager, || {
            with_request(&self.shared, request, |py, req| {
                self.connector.bind(py).call_method1(intern!(py, "request_finished"), (req,))?;
                Ok(())
            })
        })
    }

    fn take_output(&mut self, request: &Request, output: Option<&Opaque>) -> Option<Opaque> {
        with_request(&self.shared, request, |py, req| {
            let output = match output {
                Some(output) => opaque_obj(py, output),
                None => py.None().into_bound(py),
            };
            let rows = self
                .connector
                .bind(py)
                .call_method1(intern!(py, "take_output"), (req, output))?;
            Ok((!rows.is_none()).then(|| opaque(rows)))
        })
    }

    fn reset(&mut self) {
        Python::attach(|py| {
            let result = self.connector.bind(py).call_method0(intern!(py, "reset")).map(drop);
            self.shared.check(result)
        })
    }
}
