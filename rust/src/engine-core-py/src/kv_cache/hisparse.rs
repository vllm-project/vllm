// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `HiSparseCoordinator`: drop-in replacement of
//! `vllm.v1.hisparse.coordinator.HiSparseCoordinator` backed by the HiSparse
//! coordinator of a Rust KV cache manager, as `get_hisparse_coordinator`
//! returns it.

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PySet, PyTuple, PyType};
use vllm_engine_core::kv_cache::{BlockId, HiSparseHandle, KvCacheManager, RequestState};

use super::views::{BlockPoolView, Owner, SingleTypeManagerView, block_ids_of, block_list};

static TYPES: PyOnceLock<Vec<Py<PyType>>> = PyOnceLock::new();

/// The `vllm.v1.hisparse.types` dataclasses and `KVCacheBlockCopy`.
fn py_type<'py>(py: Python<'py>, idx: usize) -> PyResult<Bound<'py, PyType>> {
    let types = TYPES.get_or_try_init(py, || {
        let types = PyModule::import(py, "vllm.v1.hisparse.types")?;
        let utils = PyModule::import(py, "vllm.v1.core.kv_cache_utils")?;
        [
            types.getattr("SparseKVPageTransfer")?,
            types.getattr("SparseKVRowMirror")?,
            types.getattr("SparseKVResidencyUpdate")?,
            types.getattr("SparseKVOffloadCommand")?,
            utils.getattr("KVCacheBlockCopy")?,
        ]
        .into_iter()
        .map(|t| Ok(t.cast_into::<PyType>()?.unbind()))
        .collect::<PyResult<Vec<_>>>()
    })?;
    Ok(types[idx].bind(py).clone())
}

const PAGE_TRANSFER: usize = 0;
const ROW_MIRROR: usize = 1;
const RESIDENCY_UPDATE: usize = 2;
const OFFLOAD_COMMAND: usize = 3;
const BLOCK_COPY: usize = 4;

fn kwargs<'py>(
    py: Python<'py>,
    items: &[(&str, Bound<'py, PyAny>)],
) -> PyResult<Bound<'py, PyDict>> {
    let kwargs = PyDict::new(py);
    for (key, value) in items {
        kwargs.set_item(key, value)?;
    }
    Ok(kwargs)
}

/// `(request_id, a, b)` triples of a Python iterable.
fn extract_triples(requests: &Bound<'_, PyAny>) -> PyResult<Vec<(String, usize, usize)>> {
    requests.try_iter()?.map(|r| r?.extract()).collect()
}

/// `HiSparseCoordinator` facade.
///
/// Deviation: `get_hisparse_coordinator` returns a new facade per call, all
/// equal (`==`) for one manager, where Python returns the same object.
/// Instances have a `__dict__` so that callers (tests) can patch methods.
#[pyclass(frozen, dict, module = "vllm._rust_core", name = "HiSparseCoordinator")]
pub struct HiSparseCoordinatorView {
    owner: Owner,
}

impl HiSparseCoordinatorView {
    /// `get_hisparse_coordinator`: the first call fixes `import_step_tokens`.
    pub fn bind(py: Python<'_>, owner: Owner, import_step_tokens: Option<usize>) -> PyResult<Self> {
        let bound = owner.with_mut(py, |m| {
            m.hisparse_mut().map(|mut h| h.bind(import_step_tokens)).is_some()
        })?;
        if !bound {
            return Err(PyValueError::new_err(
                "No HiSparse cache group is configured.",
            ));
        }
        Ok(Self { owner })
    }

    /// Run `f` on the HiSparse coordinator. `f` must not run Python code.
    fn with<R>(&self, py: Python<'_>, f: impl FnOnce(&mut HiSparseHandle<'_>) -> R) -> PyResult<R> {
        self.owner.with_mut(py, |m| {
            f(&mut m.hisparse_mut().expect("HiSparse is configured"))
        })
    }

    fn with_manager<R>(&self, py: Python<'_>, f: impl FnOnce(&KvCacheManager) -> R) -> PyResult<R> {
        self.owner.with(py, f)
    }

    fn group_views(&self, group_ids: Vec<usize>) -> Vec<SingleTypeManagerView> {
        group_ids
            .into_iter()
            .map(|g| SingleTypeManagerView::new(self.owner.clone(), g))
            .collect()
    }
}

#[pymethods]
impl HiSparseCoordinatorView {
    #[getter]
    fn import_step_tokens(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |h| h.coordinator().import_step_tokens)
    }

    #[getter]
    fn num_reprefillable_tokens(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |h| h.coordinator().num_reprefillable_tokens)
    }

    #[getter]
    fn host_group_id(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |h| h.coordinator().host_group_id)
    }

    #[getter]
    fn max_spill_pages(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |h| h.coordinator().max_spill_pages)
    }

    #[setter]
    fn set_max_spill_pages(&self, py: Python<'_>, max_spill_pages: usize) -> PyResult<()> {
        self.with(py, |h| {
            h.coordinator_mut().max_spill_pages = max_spill_pages
        })
    }

    #[getter]
    fn transition_watermark(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |h| h.coordinator().transition_watermark)
    }

    /// Host pages admitted imports still need for their sequence.
    #[getter]
    fn reserved_import_pages(&self, py: Python<'_>) -> PyResult<i64> {
        self.with(py, |h| h.coordinator().reserved_import_pages())
    }

    #[getter]
    fn resident_managers(&self, py: Python<'_>) -> PyResult<Vec<SingleTypeManagerView>> {
        let ids = self.with(py, |h| h.coordinator().resident_group_ids.clone())?;
        Ok(self.group_views(ids))
    }

    #[getter]
    fn hot_managers(&self, py: Python<'_>) -> PyResult<Vec<SingleTypeManagerView>> {
        let ids = self.with(py, |h| h.coordinator().hot_group_ids.clone())?;
        Ok(self.group_views(ids))
    }

    #[getter]
    fn host_manager(&self, py: Python<'_>) -> PyResult<SingleTypeManagerView> {
        let id = self.with(py, |h| h.coordinator().host_group_id)?;
        Ok(SingleTypeManagerView::new(self.owner.clone(), id))
    }

    /// The shared GPU pool of the resident groups, if any.
    #[getter]
    fn gpu_pool(&self, py: Python<'_>) -> PyResult<Option<BlockPoolView>> {
        let has_residents = self.with(py, |h| !h.coordinator().resident_group_ids.is_empty())?;
        Ok(has_residents.then(|| BlockPoolView::new(self.owner.clone())))
    }

    fn get_host_block_pool(&self) -> BlockPoolView {
        BlockPoolView::host(self.owner.clone())
    }

    /// `pending_spills`: the ids of the in-flight spills (values are not
    /// exposed).
    #[getter]
    fn pending_spills<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let ids: Vec<u64> = self.with(py, |h| h.coordinator().pending_spill_ids().collect())?;
        let out = PyDict::new(py);
        for id in ids {
            out.set_item(id, py.None())?;
        }
        Ok(out)
    }

    #[getter]
    fn spills_to_send<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let transfers = self.with(py, |h| h.coordinator().spills_to_send().to_vec())?;
        page_transfers(py, &transfers)
    }

    /// `copies`: published host block hash -> GPU copies of that page.
    #[getter]
    fn copies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let copies: Vec<(Vec<u8>, Vec<BlockId>)> = self.with(py, |h| {
            h.coordinator()
                .copies()
                .iter()
                .map(|(hash, blocks)| (hash.to_packed_bytes(), blocks.clone()))
                .collect()
        })?;
        let out = PyDict::new(py);
        for (hash, blocks) in copies {
            let blocks = block_list(py, &self.owner, &blocks)?;
            out.set_item(
                pyo3::types::PyBytes::new(py, &hash),
                PyTuple::new(py, blocks)?,
            )?;
        }
        Ok(out)
    }

    /// `request_states`: read-only snapshots of the per-request residency
    /// bookkeeping.
    #[getter]
    fn request_states(&self) -> RequestStates {
        RequestStates {
            owner: self.owner.clone(),
        }
    }

    fn advance_scheduled(&self, py: Python<'_>, requests: &Bound<'_, PyAny>) -> PyResult<()> {
        let requests: Vec<(String, usize)> =
            requests.try_iter()?.map(|r| r?.extract()).collect::<PyResult<_>>()?;
        self.with(py, |h| {
            h.advance_scheduled(requests.iter().map(|(r, n)| (r.as_str(), *n)));
        })
    }

    fn update_residency(&self, py: Python<'_>, request_id: &str) -> PyResult<()> {
        self.with(py, |h| h.update_residency(request_id))
    }

    fn plan_prefix_materialization(
        &self,
        py: Python<'_>,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> PyResult<()> {
        self.with(py, |h| {
            h.plan_prefix_materialization(request_id, num_computed_tokens)
        })
    }

    /// Package residency decisions for the worker connector.
    fn build_offload_command<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(transfers) = self.with(py, |h| h.build_offload_command())? else {
            return Ok(None);
        };
        let transfers = page_transfers(py, &transfers)?;
        let kwargs = kwargs(py, &[("page_transfers", transfers.into_any())])?;
        py_type(py, OFFLOAD_COMMAND)?.call((), Some(&kwargs)).map(Some)
    }

    /// Drain host copy-on-write work, retaining both endpoints.
    fn take_host_block_copies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let copies = self.with(py, |h| h.take_host_block_copies())?;
        let cls = py_type(py, BLOCK_COPY)?;
        let copies = copies
            .iter()
            .map(|c| {
                let kwargs = kwargs(
                    py,
                    &[
                        ("src_block_id", c.src_block_id.into_pyobject(py)?.into_any()),
                        ("dst_block_id", c.dst_block_id.into_pyobject(py)?.into_any()),
                    ],
                )?;
                cls.call((), Some(&kwargs))
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, copies)
    }

    /// Release the endpoints of host copies the worker reports having run.
    fn release_completed_host_copies(
        &self,
        py: Python<'_>,
        dst_block_ids: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let ids = block_ids_of(dst_block_ids)?;
        self.with(py, |h| h.release_completed_host_copies(ids))
    }

    /// Map scheduled rows to scheduler-owned resident and host blocks.
    fn build_row_mirrors<'py>(
        &self,
        py: Python<'py>,
        requests: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let requests = extract_triples(requests)?;
        let mirrors = self.with(py, |h| {
            h.build_row_mirrors(requests.iter().map(|(r, a, b)| (r.as_str(), *a, *b)))
        })?;
        let cls = py_type(py, ROW_MIRROR)?;
        let mirrors = mirrors
            .iter()
            .map(|m| {
                let kwargs = kwargs(
                    py,
                    &[
                        (
                            "source_starts",
                            PyTuple::new(py, &m.source_starts)?.into_any(),
                        ),
                        (
                            "destination_start",
                            m.destination_start.into_pyobject(py)?.into_any(),
                        ),
                        ("num_rows", m.num_rows.into_pyobject(py)?.into_any()),
                    ],
                )?;
                cls.call((), Some(&kwargs))
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, mirrors)
    }

    /// Return whether every scheduled request can read only resident KV.
    fn all_context_pages_resident(
        &self,
        py: Python<'_>,
        requests: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let requests = extract_triples(requests)?;
        self.with(py, |h| {
            h.all_context_pages_resident(requests.iter().map(|(r, a, b)| (r.as_str(), *a, *b)))
        })
    }

    /// Resident block ids the worker lacks for the scheduled requests.
    fn take_residency_updates<'py>(
        &self,
        py: Python<'py>,
        request_ids: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let request_ids: Vec<String> =
            request_ids.try_iter()?.map(|r| r?.extract()).collect::<PyResult<_>>()?;
        let updates = self.with(py, |h| {
            h.take_residency_updates(request_ids.iter().map(String::as_str))
        })?;
        let cls = py_type(py, RESIDENCY_UPDATE)?;
        let out = PyDict::new(py);
        for (request_id, update) in updates {
            let block_ids = update
                .block_ids
                .iter()
                .map(|g| PyList::new(py, g))
                .collect::<PyResult<Vec<_>>>()?;
            let kwargs = kwargs(
                py,
                &[
                    ("pages", PyList::new(py, &update.pages)?.into_any()),
                    ("block_ids", PyTuple::new(py, block_ids)?.into_any()),
                ],
            )?;
            out.set_item(request_id, cls.call((), Some(&kwargs))?)?;
        }
        Ok(out)
    }

    #[pyo3(signature = (request_id, *, failed))]
    fn finish_host_import(&self, py: Python<'_>, request_id: &str, failed: bool) -> PyResult<()> {
        self.with(py, |h| h.finish_host_import(request_id, failed))
    }

    /// Publish externally populated host pages after connector completion.
    fn complete_host_import(
        &self,
        py: Python<'_>,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> PyResult<()> {
        self.with(py, |h| {
            h.complete_host_import(request_id, num_computed_tokens)
        })
    }

    fn update_spills(
        &self,
        py: Python<'_>,
        enqueued_counts: &Bound<'_, PyAny>,
        completed_counts: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let counts = |m: &Bound<'_, PyAny>| -> PyResult<Vec<(u64, usize)>> {
            m.call_method0("items")?.try_iter()?.map(|kv| kv?.extract()).collect()
        };
        let (enqueued, completed) = (counts(enqueued_counts)?, counts(completed_counts)?);
        self.with(py, |h| h.update_spills(enqueued, completed))
    }

    /// Whether an in-flight spill will free GPU or host blocks on completion.
    fn has_pending_reclamation(&self, py: Python<'_>) -> PyResult<bool> {
        self.with(py, |h| h.has_pending_reclamation())
    }

    fn has_pending_work(&self, py: Python<'_>) -> PyResult<bool> {
        self.with(py, |h| h.has_pending_work())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<Self>().is_ok_and(|o| o.get().owner.is(&self.owner))
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<u64> {
        self.with_manager(py, |m| std::ptr::from_ref(m) as u64)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let host_group_id = self.with_manager(py, |m| m.coordinator.host_group_id())?;
        Ok(format!(
            "HiSparseCoordinator(host_group_id={host_group_id:?})"
        ))
    }
}

fn page_transfers<'py>(
    py: Python<'py>,
    transfers: &[vllm_engine_core::kv_cache::SparseKvPageTransfer],
) -> PyResult<Bound<'py, PyList>> {
    let cls = py_type(py, PAGE_TRANSFER)?;
    let transfers = transfers
        .iter()
        .map(|t| {
            let kwargs = kwargs(
                py,
                &[
                    ("transfer_id", t.transfer_id.into_pyobject(py)?.into_any()),
                    (
                        "host_block_id",
                        t.host_block_id.into_pyobject(py)?.into_any(),
                    ),
                    (
                        "resident_block_ids",
                        PyTuple::new(py, &t.resident_block_ids)?.into_any(),
                    ),
                    (
                        "restore",
                        pyo3::types::PyBool::new(py, t.restore).to_owned().into_any(),
                    ),
                ],
            )?;
            cls.call((), Some(&kwargs))
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, transfers)
}

/// `HiSparseCoordinator.request_states`: read-only mapping from request id to
/// a snapshot (`types.SimpleNamespace`) of the request's residency state.
#[pyclass(
    frozen,
    module = "vllm._rust_core",
    name = "HiSparseRequestStates",
    mapping
)]
pub struct RequestStates {
    owner: Owner,
}

/// The fields of a [`RequestState`] snapshot.
struct StateSnapshot {
    valid_pages: Vec<usize>,
    ready_prefix_pages: usize,
    pending_pages: Vec<(usize, u64)>,
    has_publication: bool,
    pinned_clean: Vec<usize>,
    unpinned_pages: Vec<usize>,
    pages_to_adopt: usize,
    synced_pages: usize,
    stale_pages: Vec<usize>,
}

impl StateSnapshot {
    fn new(state: &RequestState) -> Self {
        Self {
            valid_pages: state.valid_pages.iter().copied().collect(),
            ready_prefix_pages: state.ready_prefix_pages,
            pending_pages: state.pending_pages.iter().map(|(&p, &s)| (p, s)).collect(),
            has_publication: state.has_publication(),
            pinned_clean: state.pinned_clean.iter().copied().collect(),
            unpinned_pages: state.unpinned_pages.iter().copied().collect(),
            pages_to_adopt: state.pages_to_adopt,
            synced_pages: state.synced_pages,
            stale_pages: state.stale_pages.iter().copied().collect(),
        }
    }

    fn into_py(self, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let pending_pages = PyDict::new(py);
        for (page, spill) in self.pending_pages {
            pending_pages.set_item(page, spill)?;
        }
        let kwargs = kwargs(
            py,
            &[
                ("valid_pages", PySet::new(py, &self.valid_pages)?.into_any()),
                (
                    "ready_prefix_pages",
                    self.ready_prefix_pages.into_pyobject(py)?.into_any(),
                ),
                ("pending_pages", pending_pages.into_any()),
                (
                    "publication",
                    if self.has_publication {
                        py.Ellipsis()
                    } else {
                        py.None()
                    }
                    .into_bound(py),
                ),
                (
                    "pinned_clean",
                    PySet::new(py, &self.pinned_clean)?.into_any(),
                ),
                (
                    "unpinned_pages",
                    PySet::new(py, &self.unpinned_pages)?.into_any(),
                ),
                (
                    "pages_to_adopt",
                    self.pages_to_adopt.into_pyobject(py)?.into_any(),
                ),
                (
                    "synced_pages",
                    self.synced_pages.into_pyobject(py)?.into_any(),
                ),
                ("stale_pages", PySet::new(py, &self.stale_pages)?.into_any()),
            ],
        )?;
        PyModule::import(py, "types")?
            .getattr("SimpleNamespace")?
            .call((), Some(&kwargs))
    }
}

impl RequestStates {
    fn lookup<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let snapshot = self.owner.with(py, |m| {
            m.hisparse().and_then(|h| h.request_state(request_id)).map(StateSnapshot::new)
        })?;
        snapshot.map(|s| s.into_py(py)).transpose()
    }
}

#[pymethods]
impl RequestStates {
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.owner.with(py, |m| {
            m.hisparse().map_or(0, |h| h.request_state_ids().count())
        })
    }

    fn keys(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.owner.with(py, |m| {
            m.hisparse()
                .map(|h| h.request_state_ids().map(str::to_owned).collect())
                .unwrap_or_default()
        })
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(PyList::new(py, self.keys(py)?)?.into_any().try_iter()?.into_any())
    }

    fn __contains__(&self, py: Python<'_>, request_id: &str) -> PyResult<bool> {
        self.owner.with(py, |m| {
            m.hisparse().is_some_and(|h| h.request_state(request_id).is_some())
        })
    }

    fn __getitem__<'py>(&self, py: Python<'py>, request_id: &str) -> PyResult<Bound<'py, PyAny>> {
        self.lookup(py, request_id)?
            .ok_or_else(|| PyKeyError::new_err(request_id.to_owned()))
    }

    #[pyo3(signature = (request_id, default=None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(self.lookup(py, request_id)?.or(default))
    }
}
