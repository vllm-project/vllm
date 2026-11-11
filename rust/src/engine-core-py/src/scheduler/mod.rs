// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `RustSchedulerCore`: the Rust scheduler
//! (`vllm_engine_core::sched::scheduler::Scheduler`) behind
//! `vllm.v1.core.sched.rust_scheduler.RustScheduler`.
//!
//! The Python wrapper constructs the Python collaborators (KV connector,
//! structured output manager, KV event publisher) exactly as `Scheduler` does
//! and hands them over with the resolved configuration. This class converts
//! requests and outputs, and implements the scheduler's collaborator traits
//! over the Python objects ([`hooks`]).
//!
//! Where the wrapper selected one, a native collaborator replaces the Python
//! hooks: the NIXL or Mooncake store connector's scheduler half
//! ([`ConnectorImpl`]).

mod config;
mod hooks;
mod native_input;
mod native_output;
mod output;
mod request;
mod types;

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use indexmap::IndexSet;
use pyo3::exceptions::{
    PyAssertionError, PyKeyError, PyNotImplementedError, PyRuntimeError, PyValueError,
};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyModule, PyString};
use rustc_hash::FxHashSet;
use thiserror_ext::AsReport;
use vllm_engine_core::connectors::mooncake_store_scheduler::SavePinPolicy;
use vllm_engine_core::sched::request::{Request, RequestStatus};
use vllm_engine_core::sched::scheduler::{
    AuxOutputConnector, EcConnector, ErrorRequestIds, IdSetOrder, KvConnector, PauseState,
    Scheduler, SchedulerError, StructuredOutput, UpdateOutput,
};
use vllm_engine_core_client::protocol::output::{EngineCoreOutputs, encode_engine_core_outputs};

use self::hooks::{
    PyAuxOutputConnector, PyEcConnector, PyKvConnector, PyStructuredOutput, SchedulerContext,
};
use self::output::{
    ExtraStats, ModelOutput, counts_of, draft_token_ids_from_py, finish_scheduler_output,
    model_output_from_py, opaque_obj, scheduler_output_from_py, scheduler_output_to_py,
    scheduler_stats_to_py, write_back_drafts,
};
use self::request::{HookError, PyReq, Shared, new_request, sync_request};
use self::types::types;
use crate::kv_cache::config::convert_config;
use crate::kv_cache::views::{SchedulerKv, SchedulerKvManagerView};
use crate::mooncake::scheduler::NativeMooncakeStoreConnector;
use crate::nixl::RustNixlConnectorScheduler;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustSchedulerCore>()?;
    m.add_class::<native_output::EngineOutputWriter>()?;
    m.add_class::<native_input::EngineInputReader>()?;
    m.add_class::<native_input::NativeAddRequest>()?;
    m.add_class::<SchedulerKvManagerView>()?;
    Ok(())
}

/// `VLLM_MSGPACK_ZERO_COPY_THRESHOLD`'s default, for outputs returned encoded.
const DEFAULT_AUX_FRAME_THRESHOLD: usize = 256;

/// Scheduler outputs kept for `update_from_output` beyond this many are
/// dropped: the engine processes outputs in order and keeps at most
/// `max_concurrent_batches` in flight.
const MAX_INFLIGHT_OUTPUTS: usize = 64;

/// The scheduler, owned by a [`RustSchedulerCore`].
struct Inner(Box<Scheduler>);

// SAFETY: the scheduler's collaborators are this module's hooks, whose state
// is Python objects and `Shared` (both `Send + Sync`), and the default clock.
// PyO3 serializes access to the pyclass (under the GIL, with its borrow
// flag), so the scheduler is never used from two threads at once.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl std::ops::Deref for Inner {
    type Target = Scheduler;

    fn deref(&self) -> &Scheduler {
        &self.0
    }
}

impl std::ops::DerefMut for Inner {
    fn deref_mut(&mut self) -> &mut Scheduler {
        &mut self.0
    }
}

/// The implementation of the KV connector's scheduler half the scheduler
/// calls. The Python connector object stays the engine's
/// (`get_kv_connector`): native halves keep using it for what runs outside
/// the scheduling loop.
enum ConnectorImpl {
    /// The Python connector's hooks ([`PyKvConnector`]).
    Python,
    /// The native NIXL scheduler half; the Python connector keeps the
    /// handshake side channel.
    Nixl(Py<RustNixlConnectorScheduler>),
    /// The native Mooncake store scheduler half; the Python connector keeps
    /// the lookup client.
    MooncakeStore,
}

impl ConnectorImpl {
    fn name(&self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::Nixl(_) => "nixl",
            Self::MooncakeStore => "mooncake_store",
        }
    }
}

#[pyclass(module = "vllm._rust_core")]
pub struct RustSchedulerCore {
    inner: Inner,
    shared: Arc<Shared>,
    kv_cache_manager: Py<SchedulerKvManagerView>,
    kv_event_publisher: Py<PyAny>,
    structured_output_manager: Py<PyAny>,
    enable_kv_cache_events: bool,
    /// The Python KV connector, shared with the hooks when they call it.
    connector: Option<Arc<Mutex<Py<PyAny>>>>,
    connector_impl: ConnectorImpl,
    /// The scheduler outputs handed out by `schedule`, each with the Rust
    /// output it was built from, until `update_from_output` consumes it.
    inflight: VecDeque<(Py<PyAny>, vllm_engine_core::sched::output::SchedulerOutput)>,
    /// The sessions with streaming-input updates queued: their Python
    /// requests' `streaming_queue` mirrors the Rust one.
    queued_sessions: FxHashSet<String>,
    /// Where `update_from_output` sends the outputs, in the engine.
    output_writer: Option<Py<native_output::EngineOutputWriter>>,
}

impl RustSchedulerCore {
    /// The part of `update_from_output` shared by both output paths: update
    /// the scheduler, publish the KV cache events, and aggregate the
    /// Python-side stats.
    fn process_model_output<'py>(
        &mut self,
        py: Python<'py>,
        scheduler_output: &Bound<'py, PyAny>,
        model_runner_output: &Bound<'py, PyAny>,
        perf_stats: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(UpdateOutput, ModelOutput<'py>, ExtraStats<'py>)> {
        let so = match self.take_inflight(scheduler_output) {
            Some(mut so) => {
                // Set by `update_draft_token_ids_in_output`, or by a caller.
                let invalid = scheduler_output.getattr(intern!(py, "num_invalid_spec_tokens"))?;
                if !invalid.is_none() {
                    so.num_invalid_spec_tokens = Some(counts_of(&invalid)?);
                }
                so
            }
            None => scheduler_output_from_py(scheduler_output)?,
        };
        let mo = model_output_from_py(model_runner_output)?;
        let update = self.run(|s| s.update_from_output(&so, &mo.rust))?;
        self.apply_session_updates(py)?;

        // Worker-side KV connector stats, aggregated with the scheduler's.
        let mut kv_connector_stats = None;
        if !mo.kv_connector_output.is_none() {
            let stats = mo.kv_connector_output.getattr(intern!(py, "kv_connector_stats"))?;
            if !stats.is_none() {
                kv_connector_stats = Some(stats);
            }
        }
        if let Some(sched_stats) = &update.scheduler_kv_connector_stats {
            let sched_stats = opaque_obj(py, sched_stats);
            kv_connector_stats = Some(match kv_connector_stats {
                Some(stats) => stats.call_method1(intern!(py, "aggregate"), (sched_stats,))?,
                None => sched_stats,
            });
        }

        // Publish the KV cache events of the manager and the connector.
        let mut events = None;
        if !update.kv_cache_events.is_empty() {
            events = Some(crate::kv_cache::events::events_to_py(
                py,
                update.kv_cache_events.clone(),
            )?);
        }
        if let Some(connector_events) = &update.connector_kv_cache_events {
            let connector_events = opaque_obj(py, connector_events);
            match &events {
                Some(events) => {
                    events.call_method1(intern!(py, "extend"), (connector_events,))?;
                }
                None => events = Some(connector_events.cast_into::<PyList>()?),
            }
        }
        if let Some(events) = events.filter(|e| !e.is_empty()) {
            let types = types(py)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("ts", types.time.bind(py).call_method0(intern!(py, "time"))?)?;
            kwargs.set_item("events", events)?;
            let batch = types.kv_event_batch.bind(py).call((), Some(&kwargs))?;
            self.kv_event_publisher
                .bind(py)
                .call_method1(intern!(py, "publish"), (batch,))?;
        }

        // Worker-side EC connector stats, aggregated with the scheduler's.
        let mut ec_connector_stats = None;
        if mo.ec_connector_output.is_truthy()? {
            let stats = mo.ec_connector_output.getattr(intern!(py, "ec_connector_stats"))?;
            if !stats.is_none() {
                ec_connector_stats = Some(stats);
            }
        }
        if let Some(sched_stats) = &update.scheduler_ec_connector_stats {
            let sched_stats = opaque_obj(py, sched_stats);
            ec_connector_stats = Some(match ec_connector_stats {
                Some(stats) => stats.call_method1(intern!(py, "aggregate"), (sched_stats,))?,
                None => sched_stats,
            });
        }

        let extra = ExtraStats {
            kv_connector_stats,
            cudagraph_stats: Some(mo.cudagraph_stats.clone()).filter(|s| !s.is_none()),
            perf_stats: perf_stats.filter(|s| !s.is_none()),
            ec_connector_stats,
        };
        Ok((update, mo, extra))
    }

    /// `add_request` of a request the native input path admitted without a
    /// Python `Request` (`NativeAddRequest`).
    fn add_native_request(
        &mut self,
        py: Python<'_>,
        native: &Bound<'_, native_input::NativeAddRequest>,
    ) -> PyResult<()> {
        if self.connector.is_some() {
            // Its hooks take Python requests; the native input path is off then.
            return Err(PyRuntimeError::new_err(
                "a request without a Python Request needs a scheduler without a KV connector",
            ));
        }
        let new = (native.borrow().take_new_request())
            .ok_or_else(|| PyValueError::new_err("the request was already added"))?;
        let request_id = new.request_id.clone();
        if self.inner.has_request(&request_id) {
            return Err(PyValueError::new_err(format!(
                "request {request_id} is already in the scheduler"
            )));
        }
        let id = native.getattr(intern!(py, "request_id"))?.cast_into::<PyString>()?;
        self.shared.table.lock().unwrap().insert(
            request_id.clone(),
            PyReq {
                obj: native.clone().into_any().unbind(),
                native: true,
                id: id.unbind(),
                trace_headers: None,
            },
        );
        let result = self
            .run(|_| Request::new(new))
            .and_then(|r| r.map_err(|e| PyValueError::new_err(e.to_report_string())))
            .and_then(|rust_request| self.run(|s| s.add_request(rust_request)));
        if result.is_err() && !self.inner.has_request(&request_id) {
            self.shared.table.lock().unwrap().shift_remove(&request_id);
        }
        self.drain_removed(py)?;
        result
    }

    /// Run `f` on the scheduler: the KV cache manager is unreachable from
    /// Python meanwhile (except from the hooks that receive it), and a
    /// Python exception raised by a hook, or a panic, becomes the call's
    /// exception.
    fn run<R>(&mut self, f: impl FnOnce(&mut Scheduler) -> R) -> PyResult<R> {
        self.shared.kv.set(std::ptr::null_mut());
        let result = catch_unwind(AssertUnwindSafe(|| f(&mut self.inner)));
        self.shared.kv.set(self.inner.kv_cache_manager_mut());
        result.map_err(|payload| self.panic_error(payload))
    }

    /// The exception of a panic: the Python exception a hook raised, else
    /// an `AssertionError` with the panic message.
    fn panic_error(&self, payload: Box<dyn std::any::Any + Send>) -> PyErr {
        if payload.is::<HookError>()
            && let Some(err) = self.shared.take_error()
        {
            return err;
        }
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_default();
        PyAssertionError::new_err(message)
    }

    /// Release the Python objects of the requests the scheduler dropped,
    /// bringing them to their final state first: callers may still hold them.
    fn drain_removed(&mut self, py: Python<'_>) -> PyResult<()> {
        let removed = self.inner.take_removed_requests();
        if removed.is_empty() {
            return Ok(());
        }
        let mut table = self.shared.table.lock().unwrap();
        let mut removed_ids = FxHashSet::default();
        for request in &removed {
            if let Some(entry) = table.get(request.request_id.as_str())
                && !entry.native
            {
                sync_request(entry.obj.bind(py), request)?;
            }
            removed_ids.insert(request.request_id.as_str());
        }
        if removed_ids.len() == 1 {
            table.shift_remove(removed[0].request_id.as_str());
        } else {
            table.retain(|id, _| !removed_ids.contains(id.as_str()));
        }
        Ok(())
    }

    /// Mirror the streaming-input updates the scheduler applied
    /// (`_update_request_as_session`) on the sessions' Python requests.
    fn apply_session_updates(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.queued_sessions.is_empty() {
            return Ok(());
        }
        let inner = &self.inner;
        let shared = &self.shared;
        let mut result = Ok(());
        self.queued_sessions.retain(|id| {
            let Some(request) = inner.get_request(id) else {
                return false;
            };
            let queued = request.streaming_queue.as_ref().map_or(0, VecDeque::len);
            let session = shared.py_request(py, id).into_bound(py);
            let step = || -> PyResult<bool> {
                let updates = session.getattr(intern!(py, "streaming_queue"))?;
                while updates.len()? > queued {
                    let update = updates.call_method0(intern!(py, "popleft"))?;
                    if !update.is_none() {
                        apply_session_update(&session, &update, request)?;
                    }
                }
                Ok(queued > 0)
            };
            step().unwrap_or_else(|err| {
                result = Err(err);
                true
            })
        });
        result
    }

    /// `add_request` of a request id the scheduler holds: the next input
    /// chunk of a streaming-input session (or its finished sentinel), which
    /// the Rust scheduler applies or queues; the session keeps its Python
    /// request.
    fn add_session_update(
        &mut self,
        py: Python<'_>,
        request: &Bound<'_, PyAny>,
        request_id: String,
    ) -> PyResult<()> {
        let waiting = self
            .inner
            .get_request(&request_id)
            .is_some_and(|r| r.status == RequestStatus::WaitingForStreamingReq);
        let new = new_request(&self.shared, request)?;
        let max_tokens: usize = request.getattr(intern!(py, "max_tokens"))?.extract()?;
        let mut update = self
            .run(|_| Request::new(new))?
            .map_err(|e| PyValueError::new_err(e.to_report_string()))?;
        update.max_tokens = max_tokens;
        self.run(|s| s.add_request(update))?;
        let update = types(py)?.streaming_update_from_request.bind(py).call1((request,))?;
        let session = self.shared.py_request(py, &request_id).into_bound(py);
        if !waiting {
            // Queue next input chunk (or finished sentinel).
            session
                .getattr(intern!(py, "streaming_queue"))?
                .call_method1(intern!(py, "append"), (update,))?;
            self.queued_sessions.insert(request_id);
        } else if let (false, Some(request)) =
            (update.is_none(), self.inner.get_request(&request_id))
        {
            apply_session_update(&session, &update, request)?;
        }
        self.drain_removed(py)
    }

    /// The synced Python request of `request_id`, if the scheduler holds it.
    fn synced_request<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(request) = self.inner.get_request(request_id) else {
            return Ok(None);
        };
        let (obj, native) = self.shared.py_request_entry(py, request_id);
        let obj = obj.into_bound(py);
        if !native {
            sync_request(&obj, request)?;
        }
        Ok(Some(obj))
    }

    /// The Rust output `schedule` built `py_so` from, if it is in flight.
    fn take_inflight(
        &mut self,
        py_so: &Bound<'_, PyAny>,
    ) -> Option<vllm_engine_core::sched::output::SchedulerOutput> {
        let pos = self.inflight.iter().position(|(obj, _)| obj.is(py_so))?;
        self.inflight.remove(pos).map(|(_, so)| so)
    }
}

impl Drop for RustSchedulerCore {
    fn drop(&mut self) {
        // Views handed to Python (e.g. the connector's block pool) outlive
        // the scheduler; they must not reach its manager any more.
        self.shared.kv.set(std::ptr::null_mut());
    }
}

fn scheduler_error(err: SchedulerError) -> PyErr {
    match err {
        SchedulerError::Unsupported { .. } => {
            PyNotImplementedError::new_err(err.to_report_string())
        }
        SchedulerError::InvalidConfig { .. } => PyValueError::new_err(err.to_report_string()),
        SchedulerError::KvCache(err) => crate::kv_cache::config::to_py_err(err),
    }
}

/// Finish error requests in Python's order: `error_order` (see
/// `rust_scheduler._error_request_order`) rebuilds Python's set of their ids
/// from the ids of each source, in the order they were found.
fn python_error_order(error_order: Py<PyAny>) -> IdSetOrder {
    Box::new(move |ids: &ErrorRequestIds| {
        Python::attach(|py| {
            let list = |ids: &IndexSet<String>| PyList::new(py, ids).expect("a list of str");
            let f = &ids.failed_kv_loads;
            let order = error_order.bind(py).call1((
                list(&ids.grammar),
                list(&ids.encoder_cache_mismatch),
                list(&f.recving),
                list(&f.invalid_async),
                list(&f.invalid_sync),
                list(&ids.ec_unavailable),
            ));
            let order = order.and_then(|o| o.extract::<Vec<String>>());
            order.expect("rust_scheduler._error_request_order").into_iter().collect()
        })
    })
}

/// `_update_request_as_session` on the session's Python request, from the
/// update's `StreamingUpdate`, once the Rust scheduler applied the update to
/// `request`: the payloads the binding hands out from the Python request
/// (multi-modal features, sampling params) and everything [`sync_request`]
/// brings up to date.
fn apply_session_update(
    session: &Bound<'_, PyAny>,
    update: &Bound<'_, PyAny>,
    request: &Request,
) -> PyResult<()> {
    let py = session.py();
    let mm_features = update.getattr(intern!(py, "mm_features"))?;
    if mm_features.is_truthy()? {
        let num_new_tokens = match update.getattr(intern!(py, "prompt_token_ids"))? {
            tokens if tokens.is_none() => 0,
            tokens => tokens.len()?,
        };
        let base = request.num_prompt_tokens - num_new_tokens;
        let replace = PyModule::import(py, "dataclasses")?.getattr(intern!(py, "replace"))?;
        for feature in mm_features.try_iter()? {
            let feature = feature?;
            let position = feature.getattr(intern!(py, "mm_position"))?;
            let offset: usize = position.getattr(intern!(py, "offset"))?.extract()?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("offset", offset + base)?;
            feature.setattr(
                intern!(py, "mm_position"),
                replace.call((position,), Some(&kwargs))?,
            )?;
        }
        session
            .getattr(intern!(py, "mm_features"))?
            .call_method1(intern!(py, "extend"), (mm_features,))?;
    }
    session.setattr(
        intern!(py, "arrival_time"),
        update.getattr(intern!(py, "arrival_time"))?,
    )?;
    session.setattr(
        intern!(py, "sampling_params"),
        update.getattr(intern!(py, "sampling_params"))?,
    )?;
    sync_request(session, request)
}

fn status_of(status: &Bound<'_, PyAny>) -> PyResult<RequestStatus> {
    use RequestStatus::*;
    let value: u8 = status.getattr("value")?.extract()?;
    [
        Waiting,
        WaitingForStructuredOutputGrammar,
        WaitingForRemoteKvs,
        WaitingForStreamingReq,
        Running,
        Preempted,
        FinishedStopped,
        FinishedLengthCapped,
        FinishedAborted,
        FinishedIgnored,
        FinishedError,
        FinishedRepetition,
    ]
    .into_iter()
    .find(|s| *s as u8 == value)
    .ok_or_else(|| PyValueError::new_err(format!("unknown request status {value}")))
}

#[pymethods]
impl RustSchedulerCore {
    /// `config` holds the values `Scheduler.__init__` derives from
    /// `VllmConfig` (see `rust_scheduler.py`); `connector` is the
    /// scheduler-side KV connector, if any. `native_connector` selects a
    /// native scheduler half for it: a `RustNixlConnectorScheduler` built
    /// from it, or `"mooncake_store"`.
    #[new]
    #[pyo3(signature = (config, kv_cache_config, structured_output_manager, kv_event_publisher, error_order, connector=None, native_connector=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        config: &Bound<'_, PyDict>,
        kv_cache_config: &Bound<'_, PyAny>,
        structured_output_manager: Py<PyAny>,
        kv_event_publisher: Py<PyAny>,
        error_order: Py<PyAny>,
        connector: Option<Bound<'_, PyAny>>,
        native_connector: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let rust_config = config::scheduler_config(config)?;
        let (kv_config, specs) = convert_config(kv_cache_config)?;
        let kv = Arc::new(SchedulerKv::new(
            specs.into_iter().map(Bound::unbind).collect(),
            kv_cache_config.clone().unbind(),
        ));
        let shared = Arc::new(Shared::new(kv.clone(), rust_config.uses_mrope));
        let kv_cache_manager = Py::new(py, SchedulerKvManagerView::new(kv))?;
        let connector = connector.filter(|c| !c.is_none());
        let native_connector = native_connector.filter(|c| !c.is_none());
        let mut connector_impl = ConnectorImpl::Python;
        let kv_connector: Option<Box<dyn KvConnector>> = match (&connector, native_connector) {
            (None, Some(_)) => {
                return Err(PyValueError::new_err(
                    "a native connector needs the Python connector",
                ));
            }
            (None, None) => None,
            (Some(c), None) => Some(Box::new(PyKvConnector::new(
                Arc::new(Mutex::new(c.clone().unbind())),
                shared.clone(),
                kv_cache_manager.clone_ref(py),
            )?)),
            (Some(c), Some(native)) => {
                if let Ok(nixl) = native.cast::<RustNixlConnectorScheduler>() {
                    connector_impl = ConnectorImpl::Nixl(nixl.clone().unbind());
                    Some(Box::new(nixl.get().kv_connector()))
                } else if native.extract::<&str>()? == "mooncake_store" {
                    connector_impl = ConnectorImpl::MooncakeStore;
                    Some(Box::new(NativeMooncakeStoreConnector::new(
                        c,
                        Arc::new(SchedulerContext(shared.clone())),
                        SavePinPolicy::WholeTable,
                    )?))
                } else {
                    return Err(PyValueError::new_err(format!(
                        "unknown native connector {native}"
                    )));
                }
            }
        };
        let connector = connector.map(|c| Arc::new(Mutex::new(c.unbind())));
        let structured_output: Box<dyn StructuredOutput> = Box::new(PyStructuredOutput::new(
            structured_output_manager.clone_ref(py),
            shared.clone(),
        ));
        let enable_kv_cache_events = rust_config.enable_kv_cache_events;
        // Construction asserts the config like Python does.
        let inner = catch_unwind(AssertUnwindSafe(|| {
            Scheduler::new(rust_config, kv_config, kv_connector, structured_output)
        }))
        .map_err(|panic| {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_default();
            PyAssertionError::new_err(message)
        })?
        .map_err(scheduler_error)?;
        let mut inner = inner;
        inner.set_id_set_order(python_error_order(error_order));
        let mut this = Self {
            inner: Inner(Box::new(inner)),
            shared,
            kv_cache_manager,
            kv_event_publisher,
            structured_output_manager,
            enable_kv_cache_events,
            connector,
            connector_impl,
            inflight: VecDeque::new(),
            queued_sessions: FxHashSet::default(),
            output_writer: None,
        };
        let manager = this.inner.kv_cache_manager_mut() as *mut _;
        this.shared.kv.set(manager);
        Ok(this)
    }

    /// Raise `NotImplementedError` if the KV cache layout is not ported.
    #[staticmethod]
    fn check_kv_cache_config(kv_cache_config: &Bound<'_, PyAny>) -> PyResult<()> {
        convert_config(kv_cache_config).map(drop)
    }

    /// Replace the KV connector object the scheduler calls (tests swap in
    /// mocks; None removes it). A connector must have been configured. A
    /// native scheduler half is replaced by the new object's hooks, which
    /// report the settings the scheduler read from the connector it
    /// replaces.
    fn set_connector(&mut self, py: Python<'_>, connector: Py<PyAny>) -> PyResult<()> {
        let Some(current) = &self.connector else {
            return Err(PyValueError::new_err("the scheduler has no KV connector"));
        };
        if connector.is_none(py) {
            self.connector = None;
            self.connector_impl = ConnectorImpl::Python;
            self.inner.set_connector(None);
            return Ok(());
        }
        if let ConnectorImpl::Python = self.connector_impl {
            *current.lock().unwrap() = connector;
            return Ok(());
        }
        let native = self.inner.connector().expect("a native connector");
        let settings = (
            native.supports_divergent_local_hybrid_hits(),
            native.requires_kv_delivery(),
        );
        let connector = Arc::new(Mutex::new(connector));
        self.inner.set_connector(Some(Box::new(PyKvConnector::with_settings(
            connector.clone(),
            self.shared.clone(),
            self.kv_cache_manager.clone_ref(py),
            settings.0,
            settings.1,
        ))));
        self.connector = Some(connector);
        self.connector_impl = ConnectorImpl::Python;
        Ok(())
    }

    /// Set (or, with None, remove) the scheduler-role EC connector
    /// (`ec_transfer_config`).
    fn set_ec_connector(&mut self, connector: Option<Py<PyAny>>) {
        let shared = self.shared.clone();
        self.inner.set_ec_connector(
            connector.map(|c| -> Box<dyn EcConnector> { Box::new(PyEcConnector::new(c, shared)) }),
        );
    }

    /// Set (or, with None, remove) the `AuxOutputSchedulerConnector`
    /// (`aux_output_config.enabled`), built over `kv_cache_manager`.
    fn set_aux_output_connector(&mut self, connector: Option<Py<PyAny>>) {
        let shared = self.shared.clone();
        self.inner
            .set_aux_output_connector(connector.map(|c| -> Box<dyn AuxOutputConnector> {
                Box::new(PyAuxOutputConnector::new(c, shared))
            }));
    }

    /// The implementation of the KV connector's scheduler half:
    /// `"python"`, `"nixl"` or `"mooncake_store"`.
    #[getter]
    fn connector_impl(&self) -> &'static str {
        self.connector_impl.name()
    }

    /// The native NIXL scheduler half, if the scheduler calls it.
    #[getter]
    fn native_nixl_connector(&self, py: Python<'_>) -> Option<Py<RustNixlConnectorScheduler>> {
        match &self.connector_impl {
            ConnectorImpl::Nixl(nixl) => Some(nixl.clone_ref(py)),
            _ => None,
        }
    }

    /// The `KVCacheManager` facade connectors are bound to.
    #[getter]
    fn kv_cache_manager(&self, py: Python<'_>) -> Py<SchedulerKvManagerView> {
        self.kv_cache_manager.clone_ref(py)
    }

    fn schedule<'py>(
        &mut self,
        py: Python<'py>,
        throttle_prefills: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Set if the connector's `build_connector_meta` hook built the output.
        self.shared.pending_output.lock().unwrap().take();
        let so = self.run(|s| s.schedule(throttle_prefills))?;
        let pending = self.shared.pending_output.lock().unwrap().take();
        let py_so = match pending {
            Some(py_so) => py_so.into_bound(py),
            None => {
                let table = self.shared.table.lock().unwrap();
                let inner = &self.inner;
                scheduler_output_to_py(py, &so, &table, self.shared.uses_mrope, |id| {
                    inner.get_request(id)
                })?
            }
        };
        finish_scheduler_output(&py_so, &so)?;
        if let (ConnectorImpl::MooncakeStore, Some(meta)) =
            (&self.connector_impl, &so.kv_connector_metadata)
            && so.preempted_req_ids.as_ref().is_some_and(|ids| !ids.is_empty())
        {
            // Python's metadata shares the output's set
            // (`scheduler_output.preempted_req_ids or set()`).
            opaque_obj(py, meta).setattr(
                intern!(py, "preempted_req_ids"),
                py_so.getattr(intern!(py, "preempted_req_ids"))?,
            )?;
        }
        if self.inflight.len() >= MAX_INFLIGHT_OUTPUTS {
            self.inflight.pop_front();
        }
        self.inflight.push_back((py_so.clone().unbind(), so));
        self.drain_removed(py)?;
        Ok(py_so)
    }

    /// Send the outputs through the engine's native output path instead of
    /// returning them (see `update_from_output`).
    fn set_output_writer(&mut self, writer: Py<native_output::EngineOutputWriter>) {
        self.output_writer = Some(writer);
    }

    /// Update the scheduler with a step's model output. The outputs, in the
    /// Rust frontend's wire types with `weight_version` set, go to the output
    /// writer if one is set; else they are returned encoded, as
    /// `[(client_index, frames)]`, for the caller to decode as the frontend
    /// would.
    #[pyo3(signature = (scheduler_output, model_runner_output, perf_stats=None, weight_version=None))]
    fn update_from_output<'py>(
        &mut self,
        py: Python<'py>,
        scheduler_output: &Bound<'py, PyAny>,
        model_runner_output: &Bound<'py, PyAny>,
        perf_stats: Option<Bound<'py, PyAny>>,
        weight_version: Option<String>,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let (update, mo, extra) =
            self.process_model_output(py, scheduler_output, model_runner_output, perf_stats)?;
        let batches = {
            let table = self.shared.table.lock().unwrap();
            native_output::native_outputs(
                py,
                &update,
                &table,
                &mo,
                extra,
                weight_version.as_deref(),
            )?
        };
        self.drain_removed(py)?;
        if let Some(writer) = &self.output_writer {
            let writer = &writer.get().inner;
            for (client_index, batch) in batches {
                writer.send_batch(client_index, batch);
            }
            return Ok(None);
        }
        let encoded = PyList::empty(py);
        for (client_index, batch) in batches {
            let frames = encode_engine_core_outputs(
                EngineCoreOutputs::RequestBatch(batch),
                DEFAULT_AUX_FRAME_THRESHOLD,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let frames = PyList::new(py, frames.iter().map(|f| PyBytes::new(py, f)))?;
            encoded.append((client_index, frames))?;
        }
        Ok(Some(encoded))
    }

    fn get_grammar_bitmask<'py>(
        &mut self,
        py: Python<'py>,
        scheduler_output: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !scheduler_output
            .getattr(intern!(py, "has_structured_output_requests"))?
            .is_truthy()?
        {
            return Ok(py.None().into_bound(py));
        }
        // Collect list of scheduled request ids that use structured output.
        // The corresponding rows of the bitmask will be in this order.
        let ids = PyList::empty(py);
        let requests = PyDict::new(py);
        let num_scheduled_tokens = scheduler_output.getattr(intern!(py, "num_scheduled_tokens"))?;
        for req_id in num_scheduled_tokens.try_iter()? {
            let req_id = req_id?;
            let id = req_id.cast::<PyString>()?.to_str()?;
            let Some(request) = self.inner.get_request(id) else {
                continue;
            };
            if !request.use_structured_output() || request.is_prefill_chunk {
                continue;
            }
            ids.append(&req_id)?;
            let obj = self.shared.py_request(py, id).into_bound(py);
            sync_request(&obj, request)?;
            requests.set_item(&req_id, obj)?;
        }
        if ids.is_empty() {
            return Ok(py.None().into_bound(py));
        }
        let bitmask = self.structured_output_manager.bind(py).call_method1(
            intern!(py, "grammar_bitmask"),
            (
                requests,
                &ids,
                scheduler_output.getattr(intern!(py, "scheduled_spec_decode_tokens"))?,
            ),
        )?;
        types(py)?.grammar_output.bind(py).call1((ids, bitmask))
    }

    fn update_draft_token_ids(&mut self, draft_token_ids: &Bound<'_, PyAny>) -> PyResult<()> {
        let drafts = draft_token_ids_from_py(draft_token_ids)?;
        self.run(|s| s.update_draft_token_ids(drafts))
    }

    fn update_draft_token_ids_in_output(
        &mut self,
        draft_token_ids: &Bound<'_, PyAny>,
        scheduler_output: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let drafts = draft_token_ids_from_py(draft_token_ids)?;
        let req_ids = drafts.req_ids.clone();
        let pos = self.inflight.iter().position(|(obj, _)| obj.is(scheduler_output));
        let mut so = match pos {
            Some(pos) => std::mem::take(&mut self.inflight[pos].1),
            None => scheduler_output_from_py(scheduler_output)?,
        };
        let result = self.run(|s| s.update_draft_token_ids_in_output(drafts, &mut so));
        write_back_drafts(scheduler_output, &so, &req_ids)?;
        if let Some(pos) = pos {
            self.inflight[pos].1 = so;
        }
        result
    }

    fn add_request(&mut self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(native) = request.cast::<native_input::NativeAddRequest>() {
            return self.add_native_request(py, native);
        }
        let id = request.getattr(intern!(py, "request_id"))?.cast_into::<PyString>()?;
        let request_id = id.to_str()?.to_owned();
        if self.inner.has_request(&request_id) {
            return self.add_session_update(py, request, request_id);
        }
        if self.enable_kv_cache_events && !request.getattr(intern!(py, "prompt_embeds"))?.is_none()
        {
            return Err(PyNotImplementedError::new_err(
                "Prompt embeds with KV cache events are not supported by the Rust scheduler",
            ));
        }
        // Before the Rust request is built: the connector may rewrite it.
        match (&self.connector_impl, &self.connector) {
            (ConnectorImpl::Python, Some(connector)) => {
                let connector = connector.lock().unwrap().clone_ref(py);
                connector.bind(py).call_method1(intern!(py, "on_new_request"), (request,))?;
            }
            (ConnectorImpl::Nixl(nixl), _) => nixl.get().add_request(request)?,
            // The native half's `on_new_request` (from `Scheduler::add_request`)
            // is all of it.
            (ConnectorImpl::MooncakeStore, _) | (ConnectorImpl::Python, None) => {}
        }
        let new = new_request(&self.shared, request)?;
        let trace_headers = request.getattr(intern!(py, "trace_headers"))?;
        self.shared.table.lock().unwrap().insert(
            request_id.clone(),
            PyReq {
                obj: request.clone().unbind(),
                native: false,
                id: id.unbind(),
                trace_headers: (!trace_headers.is_none()).then(|| trace_headers.unbind()),
            },
        );
        // The scheduler takes the request in the state it is given, as
        // `Scheduler.add_request` does.
        let status = status_of(&request.getattr(intern!(py, "status"))?)?;
        let num_computed_tokens: usize =
            request.getattr(intern!(py, "num_computed_tokens"))?.extract()?;
        let max_tokens: usize = request.getattr(intern!(py, "max_tokens"))?.extract()?;
        if request.getattr(intern!(py, "resumable"))?.is_truthy()? {
            request.setattr(
                intern!(py, "streaming_queue"),
                types(py)?.deque.bind(py).call0()?,
            )?;
        }
        let result = self
            .run(|_| Request::new(new))
            .and_then(|r| r.map_err(|e| PyValueError::new_err(e.to_report_string())))
            .and_then(|mut rust_request| {
                rust_request.status = status;
                rust_request.num_computed_tokens = num_computed_tokens;
                rust_request.max_tokens = max_tokens;
                self.run(|s| s.add_request(rust_request))
            });
        if result.is_err() && !self.inner.has_request(&request_id) {
            self.shared.table.lock().unwrap().shift_remove(&request_id);
        }
        self.drain_removed(py)?;
        result
    }

    /// `request_ids` is None (finish all) or the ids in the order Python
    /// iterates them; returns the finished Python requests.
    fn finish_requests<'py>(
        &mut self,
        py: Python<'py>,
        request_ids: Option<Vec<String>>,
        finished_status: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let status = status_of(finished_status)?;
        if !status.is_finished() {
            return Err(PyAssertionError::new_err(
                "finished_status must be finished",
            ));
        }
        // The finished requests are returned as Python requests; the Rust
        // ones may be gone afterwards, so sync them now.
        let candidates: Vec<String> = match &request_ids {
            Some(ids) => ids.clone(),
            None => self.shared.table.lock().unwrap().keys().cloned().collect(),
        };
        for id in &candidates {
            if let Some(request) = self.inner.get_request(id)
                && !request.is_finished()
            {
                let (obj, native) = self.shared.py_request_entry(py, id);
                if !native {
                    sync_request(obj.bind(py), request)?;
                }
            }
        }
        let objs: Vec<Py<PyAny>> = {
            let table = self.shared.table.lock().unwrap();
            candidates
                .iter()
                .filter_map(|id| table.get(id).map(|r| r.obj.clone_ref(py)))
                .collect()
        };
        let finished = self.run(|s| s.finish_requests(request_ids.as_deref(), status))?;
        let finished: FxHashSet<&str> = finished.iter().map(String::as_str).collect();
        let out = PyList::empty(py);
        for (id, obj) in candidates.iter().zip(&objs) {
            if finished.contains(id.as_str()) {
                obj.bind(py).setattr(intern!(py, "status"), finished_status)?;
                out.append(obj)?;
            }
        }
        self.drain_removed(py)?;
        Ok(out)
    }

    fn get_num_unfinished_requests(&self) -> usize {
        self.inner.get_num_unfinished_requests()
    }

    fn has_unfinished_requests(&self) -> bool {
        self.inner.has_unfinished_requests()
    }

    fn has_finished_requests(&self) -> bool {
        self.inner.has_finished_requests()
    }

    fn has_requests(&mut self) -> PyResult<bool> {
        self.run(|s| s.has_requests())
    }

    fn has_request(&self, request_id: &str) -> bool {
        self.inner.has_request(request_id)
    }

    fn get_request_counts(&self) -> (usize, usize) {
        self.inner.get_request_counts()
    }

    fn get_kv_cache_usage(&self) -> f64 {
        self.inner.get_kv_cache_usage()
    }

    #[getter]
    fn pause_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = match self.inner.pause_state() {
            PauseState::Unpaused => 0,
            PauseState::PausedNew => 1,
            PauseState::PausedAll => 2,
        };
        Ok(types(py)?.pause_state[value].bind(py).clone())
    }

    fn set_pause_state(&mut self, pause_state: &Bound<'_, PyAny>) -> PyResult<()> {
        let value: u8 = pause_state.extract()?;
        let state = match value {
            0 => PauseState::Unpaused,
            1 => PauseState::PausedNew,
            2 => PauseState::PausedAll,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown pause state {value}"
                )));
            }
        };
        self.inner.set_pause_state(state);
        Ok(())
    }

    #[pyo3(signature = (reset_running_requests=false, reset_connector=false))]
    fn reset_prefix_cache(
        &mut self,
        py: Python<'_>,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> PyResult<bool> {
        let result = self.run(|s| s.reset_prefix_cache(reset_running_requests, reset_connector));
        self.drain_removed(py)?;
        // Python raises `RuntimeError` where the Rust scheduler asserts.
        result.map_err(|err| {
            if err.is_instance_of::<PyAssertionError>(py) {
                PyRuntimeError::new_err(err.value(py).to_string())
            } else {
                err
            }
        })
    }

    fn reset_encoder_cache(&mut self) {
        self.inner.reset_encoder_cache();
    }

    fn make_stats<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let stats = self.run(|s| s.make_stats(None))?;
        stats.map(|s| scheduler_stats_to_py(py, &s, ExtraStats::default())).transpose()
    }

    /// The ids of the requests the scheduler holds, in arrival order.
    fn request_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let table = self.shared.table.lock().unwrap();
        PyList::new(py, table.values().map(|r| r.id.bind(py)))
    }

    fn num_requests(&self) -> usize {
        self.shared.table.lock().unwrap().len()
    }

    /// The Python request of `request_id`, brought up to date, or None.
    fn get_request<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.synced_request(py, request_id)
    }

    /// Bring every Python request up to date (for tests that read the
    /// request objects they added).
    fn sync_all_requests(&self, py: Python<'_>) -> PyResult<()> {
        let table = self.shared.table.lock().unwrap();
        for (id, entry) in table.iter() {
            if let Some(request) = self.inner.get_request(id)
                && !entry.native
            {
                sync_request(entry.obj.bind(py), request)?;
            }
        }
        Ok(())
    }

    /// The running requests, brought up to date.
    fn running<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for id in self.inner.running_request_ids() {
            let obj =
                self.synced_request(py, id)?.ok_or_else(|| PyKeyError::new_err(id.to_owned()))?;
            out.append(obj)?;
        }
        Ok(out)
    }
}
