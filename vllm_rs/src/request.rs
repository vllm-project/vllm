//! Rust port of `vllm.v1.request.Request` + `RequestStatus`.
//!
//! The scheduler's per-step loops read/write ~15 attributes per request. With
//! Request as a #[pyclass], those reads/writes become direct memory access
//! from Rust (vs PyO3 getattr round-trips in the earlier preamble-only port).
//!
//! SamplingParams / PoolingParams / mm_features / structured_output_request /
//! lora_request / trace_headers stay as opaque `Py<PyAny>` handles — they're
//! Python dataclasses used from Python-only code paths.
//!
//! `output_token_ids` / `all_token_ids` are stored as Rust `Vec<i64>` and
//! exposed to Python via a view class `TokenIdsView` that defers conversion
//! (so common operations like `out[-1]` don't copy the whole list).

use pyo3::exceptions::{PyAssertionError, PyIndexError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PySlice, PyTuple};
use std::cell::RefCell;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------- RequestStatus (IntEnum-equivalent) ----------

#[pyclass(eq, eq_int, module = "vllm_rs")]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RequestStatus {
    WAITING = 1,
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = 2,
    WAITING_FOR_REMOTE_KVS = 3,
    WAITING_FOR_STREAMING_REQ = 4,
    RUNNING = 5,
    PREEMPTED = 6,
    // Anything > PREEMPTED is finished.
    FINISHED_STOPPED = 7,
    FINISHED_LENGTH_CAPPED = 8,
    FINISHED_ABORTED = 9,
    FINISHED_IGNORED = 10,
    FINISHED_ERROR = 11,
    FINISHED_REPETITION = 12,
}

#[pymethods]
impl RequestStatus {
    fn __int__(&self) -> i32 {
        *self as i32
    }

    fn __repr__(&self) -> &'static str {
        match self {
            RequestStatus::WAITING => "RequestStatus.WAITING",
            RequestStatus::WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR => {
                "RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"
            }
            RequestStatus::WAITING_FOR_REMOTE_KVS => "RequestStatus.WAITING_FOR_REMOTE_KVS",
            RequestStatus::WAITING_FOR_STREAMING_REQ => "RequestStatus.WAITING_FOR_STREAMING_REQ",
            RequestStatus::RUNNING => "RequestStatus.RUNNING",
            RequestStatus::PREEMPTED => "RequestStatus.PREEMPTED",
            RequestStatus::FINISHED_STOPPED => "RequestStatus.FINISHED_STOPPED",
            RequestStatus::FINISHED_LENGTH_CAPPED => "RequestStatus.FINISHED_LENGTH_CAPPED",
            RequestStatus::FINISHED_ABORTED => "RequestStatus.FINISHED_ABORTED",
            RequestStatus::FINISHED_IGNORED => "RequestStatus.FINISHED_IGNORED",
            RequestStatus::FINISHED_ERROR => "RequestStatus.FINISHED_ERROR",
            RequestStatus::FINISHED_REPETITION => "RequestStatus.FINISHED_REPETITION",
        }
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.__repr__().trim_start_matches("RequestStatus.")
    }

    fn __str__(&self) -> &'static str {
        self.name()
    }

    #[staticmethod]
    #[pyo3(name = "is_finished")]
    fn py_is_finished(status: RequestStatus) -> bool {
        status as i32 > RequestStatus::PREEMPTED as i32
    }
}

// ---------- TokenIdsView: read-only list proxy ----------

/// Holds a strong ref to its parent Request so we can read the underlying
/// `Vec<i64>` on __getitem__ / __len__ without copying the whole list.
#[pyclass(module = "vllm_rs")]
pub struct TokenIdsView {
    parent: Py<Request>,
    // which buffer: 0 = output_token_ids_vec, 1 = all_token_ids_vec,
    //                2 = prompt_token_ids_vec, 3 = spec_token_ids_vec
    which: u8,
}

impl TokenIdsView {
    fn with_slice<R>(&self, py: Python<'_>, f: impl FnOnce(&[i64]) -> R) -> R {
        let bound = self.parent.bind(py).borrow();
        let slice = match self.which {
            0 => bound.output_token_ids_vec.borrow_ref(),
            1 => bound.all_token_ids_vec.borrow_ref(),
            2 => bound.prompt_token_ids_or_default(),
            3 => bound.spec_token_ids_vec.borrow_ref(),
            _ => &[][..],
        };
        f(slice)
    }
}

#[pymethods]
impl TokenIdsView {
    fn __len__(&self, py: Python<'_>) -> usize {
        self.with_slice(py, |s| s.len())
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(i) = key.extract::<isize>() {
            return self.with_slice(py, |s| {
                let n = s.len() as isize;
                let idx = if i < 0 { i + n } else { i };
                if idx < 0 || idx >= n {
                    Err(PyIndexError::new_err("TokenIdsView index out of range"))
                } else {
                    Ok(s[idx as usize].into_py(py))
                }
            });
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (start, stop, step, n) = self.with_slice(py, |s| {
                let ind = slice.indices(s.len() as isize).unwrap();
                (ind.start, ind.stop, ind.step, s.len() as isize)
            });
            let _ = n;
            let res: Vec<i64> = self.with_slice(py, |s| {
                if step == 1 {
                    s[start as usize..stop as usize].to_vec()
                } else {
                    let mut out = Vec::new();
                    let mut i = start;
                    while (step > 0 && i < stop) || (step < 0 && i > stop) {
                        if i >= 0 && (i as usize) < s.len() {
                            out.push(s[i as usize]);
                        }
                        i += step;
                    }
                    out
                }
            });
            return Ok(PyList::new_bound(py, res).into_py(py));
        }
        Err(PyTypeError::new_err("TokenIdsView index must be int or slice"))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> TokenIdsViewIter {
        TokenIdsViewIter {
            parent: slf.parent.clone_ref(slf.py()),
            which: slf.which,
            pos: 0,
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        self.with_slice(py, |s| format!("TokenIdsView({:?})", s))
    }
}

#[pyclass(module = "vllm_rs")]
pub struct TokenIdsViewIter {
    parent: Py<Request>,
    which: u8,
    pos: usize,
}

#[pymethods]
impl TokenIdsViewIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<i64> {
        let bound = self.parent.bind(py).borrow();
        let slice = match self.which {
            0 => bound.output_token_ids_vec.borrow_ref(),
            1 => bound.all_token_ids_vec.borrow_ref(),
            2 => bound.prompt_token_ids_or_default(),
            3 => bound.spec_token_ids_vec.borrow_ref(),
            _ => &[][..],
        };
        if self.pos >= slice.len() {
            return None;
        }
        let v = slice[self.pos];
        self.pos += 1;
        Some(v)
    }
}

// Helper newtype so field-access inside PyRef is cleaner.
pub struct VecI64(pub Vec<i64>);
impl VecI64 {
    fn borrow_ref(&self) -> &[i64] {
        &self.0
    }
}

// ---------- Request pyclass ----------

#[pyclass(module = "vllm_rs")]
pub struct Request {
    #[pyo3(get, set)]
    pub request_id: String,
    #[pyo3(get, set)]
    pub client_index: usize,
    #[pyo3(get, set)]
    pub priority: i64,
    #[pyo3(get, set)]
    pub arrival_time: f64,

    // Python param objects kept opaque.
    sampling_params: Option<Py<PyAny>>,
    pooling_params: Option<Py<PyAny>>,
    lora_request: Option<Py<PyAny>>,
    structured_output_request: Option<Py<PyAny>>,
    trace_headers: Option<Py<PyAny>>,
    mm_features: Vec<Py<PyAny>>,
    prompt_embeds: Option<Py<PyAny>>,

    #[pyo3(get, set)]
    pub status: RequestStatus,
    #[pyo3(get, set)]
    pub max_tokens: i64,
    #[pyo3(get, set)]
    pub num_prompt_tokens: usize,
    #[pyo3(get, set)]
    pub num_cached_tokens: i64,
    #[pyo3(get, set)]
    pub num_computed_tokens: i64,
    #[pyo3(get, set)]
    pub num_output_placeholders: i64,
    #[pyo3(get, set)]
    pub num_external_computed_tokens: i64,
    #[pyo3(get, set)]
    pub num_nans_in_logits: i64,
    #[pyo3(get, set)]
    pub num_preemptions: i64,
    #[pyo3(get, set)]
    pub discard_latest_async_tokens: bool,
    #[pyo3(get, set)]
    pub is_prefill_chunk: bool,
    #[pyo3(get, set)]
    pub resumable: bool,
    #[pyo3(get, set)]
    pub skip_reading_prefix_cache: bool,

    pub output_token_ids_vec: VecI64,
    pub all_token_ids_vec: VecI64,
    prompt_token_ids_vec: Option<Vec<i64>>,
    pub spec_token_ids_vec: VecI64,
    pub block_hashes_vec: RefCell<Vec<Vec<u8>>>,
    block_hasher: Option<Py<PyAny>>,

    events_vec: Vec<Py<PyAny>>,

    #[pyo3(get, set)]
    pub cache_salt: Option<String>,
    stop_reason_py: Option<Py<PyAny>>,
    kv_transfer_params_py: Option<Py<PyAny>>,
    streaming_queue: Option<Py<PyAny>>,
    prompt_embeds_per_block_hashes: Py<PyDict>,
}

impl Request {
    pub fn prompt_token_ids_or_default(&self) -> &[i64] {
        self.prompt_token_ids_vec.as_deref().unwrap_or(&[])
    }

    /// Rust-visible duplicates of the #[pymethods] getters — they're
    /// private at the module level because pymethods hides them behind
    /// PyO3's generated descriptor functions.
    pub fn rust_is_finished(&self) -> bool {
        self.status as i32 > RequestStatus::PREEMPTED as i32
    }

    pub fn rust_sampling_params(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.sampling_params.as_ref().map(|p| p.clone_ref(py))
    }

    pub fn rust_stop_reason(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.stop_reason_py.as_ref().map(|p| p.clone_ref(py))
    }

    pub fn rust_take_events<'py>(&mut self, py: Python<'py>) -> Option<Bound<'py, PyList>> {
        if self.events_vec.is_empty() {
            return None;
        }
        let evs = std::mem::take(&mut self.events_vec);
        Some(PyList::new_bound(py, evs.into_iter().map(|e| e.into_bound(py))))
    }
}

#[pymethods]
impl Request {
    #[new]
    #[pyo3(signature = (
        request_id,
        prompt_token_ids,
        sampling_params,
        pooling_params,
        client_index=0,
        arrival_time=None,
        prompt_embeds=None,
        mm_features=None,
        lora_request=None,
        cache_salt=None,
        priority=0,
        trace_headers=None,
        block_hasher=None,
        resumable=false,
        reasoning_ended=None,
        structured_output_request=None,
    ))]
    pub fn new(
        py: Python<'_>,
        request_id: String,
        prompt_token_ids: Option<Vec<i64>>,
        sampling_params: Option<Py<PyAny>>,
        pooling_params: Option<Py<PyAny>>,
        client_index: usize,
        arrival_time: Option<f64>,
        prompt_embeds: Option<Py<PyAny>>,
        mm_features: Option<Vec<Py<PyAny>>>,
        lora_request: Option<Py<PyAny>>,
        cache_salt: Option<String>,
        priority: i64,
        trace_headers: Option<Py<PyAny>>,
        block_hasher: Option<Py<PyAny>>,
        resumable: bool,
        reasoning_ended: Option<bool>,
        structured_output_request: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        // Resolve max_tokens from sampling_params / pooling_params.
        let max_tokens: i64 = if let Some(pp) = pooling_params.as_ref() {
            let _ = pp;
            1
        } else if let Some(sp) = sampling_params.as_ref() {
            let v: Option<i64> = sp.bind(py).getattr(intern!(py, "max_tokens"))?.extract()?;
            match v {
                Some(v) => v,
                None => {
                    return Err(PyAssertionError::new_err(
                        "sampling_params.max_tokens is None",
                    ));
                }
            }
        } else {
            return Err(PyValueError::new_err(
                "sampling_params and pooling_params can't both be unset",
            ));
        };

        // Initial status: if structured_output_request is set, start waiting
        // for its grammar; else WAITING.
        let status = if structured_output_request.is_some() {
            RequestStatus::WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        } else {
            RequestStatus::WAITING
        };
        // Optionally stamp reasoning_ended onto structured_output_request
        if let (Some(sor), Some(re)) = (structured_output_request.as_ref(), reasoning_ended) {
            sor.bind(py).setattr(intern!(py, "reasoning_ended"), re)?;
        }

        let arrival = arrival_time.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0)
        });

        let num_prompt_tokens = prompt_token_ids
            .as_ref()
            .map(|v| v.len())
            .unwrap_or_else(|| {
                // Attempt prompt_embeds.shape[0] if no ids
                if let Some(pe) = prompt_embeds.as_ref() {
                    pe.bind(py)
                        .getattr(intern!(py, "shape"))
                        .ok()
                        .and_then(|s| s.get_item(0).ok()?.extract::<usize>().ok())
                        .unwrap_or(0)
                } else {
                    0
                }
            });

        let all_tokens: Vec<i64> = match &prompt_token_ids {
            Some(v) => v.clone(),
            None => vec![0; num_prompt_tokens],
        };

        // kv_transfer_params from sampling_params.extra_args if present
        let kv_transfer_params_py: Option<Py<PyAny>> = if let Some(sp) = sampling_params.as_ref() {
            let extra = sp.bind(py).getattr(intern!(py, "extra_args")).ok();
            if let Some(extra) = extra {
                if !extra.is_none() {
                    let d: Option<Bound<'_, PyDict>> = extra.downcast_into::<PyDict>().ok();
                    if let Some(d) = d {
                        d.get_item("kv_transfer_params")
                            .ok()
                            .flatten()
                            .map(|v| v.unbind())
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let out = Request {
            request_id,
            client_index,
            priority,
            arrival_time: arrival,
            sampling_params,
            pooling_params,
            lora_request,
            structured_output_request,
            trace_headers,
            mm_features: mm_features.unwrap_or_default(),
            prompt_embeds,
            status,
            max_tokens,
            num_prompt_tokens,
            num_cached_tokens: -1,
            num_computed_tokens: 0,
            num_output_placeholders: 0,
            num_external_computed_tokens: 0,
            num_nans_in_logits: 0,
            num_preemptions: 0,
            discard_latest_async_tokens: false,
            is_prefill_chunk: false,
            resumable,
            skip_reading_prefix_cache: false,
            output_token_ids_vec: VecI64(Vec::new()),
            all_token_ids_vec: VecI64(all_tokens),
            prompt_token_ids_vec: prompt_token_ids,
            spec_token_ids_vec: VecI64(Vec::new()),
            block_hashes_vec: RefCell::new(Vec::new()),
            block_hasher,
            events_vec: Vec::new(),
            cache_salt,
            stop_reason_py: None,
            kv_transfer_params_py,
            streaming_queue: None,
            prompt_embeds_per_block_hashes: PyDict::new_bound(py).unbind(),
        };
        Ok(out)
    }

    // ---- param-object getters (opaque Py<PyAny>) ----

    #[getter]
    fn sampling_params(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.sampling_params.as_ref().map(|p| p.clone_ref(py))
    }
    #[setter]
    fn set_sampling_params(&mut self, v: Option<Py<PyAny>>) {
        self.sampling_params = v;
    }

    #[getter]
    fn pooling_params(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.pooling_params.as_ref().map(|p| p.clone_ref(py))
    }
    #[setter]
    fn set_pooling_params(&mut self, v: Option<Py<PyAny>>) {
        self.pooling_params = v;
    }

    #[getter]
    fn lora_request(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.lora_request.as_ref().map(|p| p.clone_ref(py))
    }
    #[setter]
    fn set_lora_request(&mut self, v: Option<Py<PyAny>>) {
        self.lora_request = v;
    }

    #[getter]
    fn structured_output_request(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.structured_output_request.as_ref().map(|p| p.clone_ref(py))
    }
    #[setter]
    fn set_structured_output_request(&mut self, v: Option<Py<PyAny>>) {
        self.structured_output_request = v;
    }

    #[getter]
    fn trace_headers(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.trace_headers.as_ref().map(|p| p.clone_ref(py))
    }
    #[setter]
    fn set_trace_headers(&mut self, v: Option<Py<PyAny>>) {
        self.trace_headers = v;
    }

    #[getter]
    fn mm_features<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::new_bound(py, self.mm_features.iter().map(|p| p.bind(py).clone()))
    }
    #[setter]
    fn set_mm_features(&mut self, py: Python<'_>, v: Bound<'_, PyList>) -> PyResult<()> {
        let _ = py;
        self.mm_features = v.iter().map(|b| b.unbind()).collect();
        Ok(())
    }

    #[getter]
    fn prompt_embeds(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.prompt_embeds.as_ref().map(|p| p.clone_ref(py))
    }

    #[getter]
    fn kv_transfer_params(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.kv_transfer_params_py.as_ref().map(|p| p.clone_ref(py))
    }
    #[setter]
    fn set_kv_transfer_params(&mut self, v: Option<Py<PyAny>>) {
        self.kv_transfer_params_py = v;
    }

    #[getter]
    fn stop_reason(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.stop_reason_py.as_ref().map(|p| p.clone_ref(py))
    }
    #[setter]
    fn set_stop_reason(&mut self, v: Option<Py<PyAny>>) {
        self.stop_reason_py = v;
    }

    // ---- token-ids views ----

    #[getter]
    fn prompt_token_ids<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyList>> {
        self.prompt_token_ids_vec
            .as_ref()
            .map(|v| PyList::new_bound(py, v))
    }

    /// Returns a TokenIdsView bound to this Request (cheap — no copy).
    #[getter]
    fn output_token_ids<'py>(slf: Py<Self>, py: Python<'py>) -> PyResult<Bound<'py, TokenIdsView>> {
        Py::new(
            py,
            TokenIdsView {
                parent: slf,
                which: 0,
            },
        )
        .map(|p| p.into_bound(py))
    }

    #[getter]
    fn all_token_ids<'py>(slf: Py<Self>, py: Python<'py>) -> PyResult<Bound<'py, TokenIdsView>> {
        Py::new(
            py,
            TokenIdsView {
                parent: slf,
                which: 1,
            },
        )
        .map(|p| p.into_bound(py))
    }

    #[getter]
    fn spec_token_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::new_bound(py, &self.spec_token_ids_vec.0)
    }

    #[setter]
    fn set_spec_token_ids(&mut self, v: Vec<i64>) {
        self.spec_token_ids_vec.0 = v;
    }

    // ---- computed properties ----

    #[getter]
    fn num_tokens(&self) -> usize {
        self.all_token_ids_vec.0.len()
    }

    #[getter]
    fn num_tokens_with_spec(&self) -> usize {
        self.all_token_ids_vec.0.len() + self.spec_token_ids_vec.0.len()
    }

    #[getter]
    fn num_output_tokens(&self) -> usize {
        self.output_token_ids_vec.0.len()
    }

    #[getter]
    fn num_encoder_inputs(&self) -> usize {
        self.mm_features.len()
    }

    #[getter]
    fn has_encoder_inputs(&self) -> bool {
        !self.mm_features.is_empty()
    }

    #[getter]
    fn use_structured_output(&self) -> bool {
        self.structured_output_request.is_some()
    }

    // ---- block hashes ----

    #[getter]
    fn block_hashes<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let list = PyList::empty_bound(py);
        for h in self.block_hashes_vec.borrow().iter() {
            list.append(PyBytes::new_bound(py, h)).unwrap();
        }
        list
    }

    #[pyo3(name = "update_block_hashes")]
    fn py_update_block_hashes(slf: Py<Self>, py: Python<'_>) -> PyResult<()> {
        let hasher = {
            let this = slf.borrow(py);
            this.block_hasher.as_ref().map(|h| h.clone_ref(py))
        };
        if let Some(hasher) = hasher {
            let out = hasher.call1(py, (slf.clone_ref(py),))?;
            let list: Bound<'_, PyList> = out.downcast_bound::<PyList>(py)?.clone();
            let this = slf.borrow(py);
            let mut bh = this.block_hashes_vec.borrow_mut();
            for item in list.iter() {
                let bytes: Vec<u8> = item.extract()?;
                bh.push(bytes);
            }
        }
        Ok(())
    }

    // ---- events ----

    #[getter]
    fn events<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::new_bound(py, self.events_vec.iter().map(|e| e.bind(py).clone()))
    }

    fn record_event(&mut self, py: Python<'_>, event_type: Bound<'_, PyAny>, timestamp: Option<f64>) -> PyResult<()> {
        let engine_mod = PyModule::import_bound(py, "vllm.v1.engine")?;
        let ece = engine_mod.getattr("EngineCoreEvent")?;
        let ev = ece.call_method1("new_event", (event_type, timestamp))?;
        self.events_vec.push(ev.unbind());
        Ok(())
    }

    fn take_events<'py>(&mut self, py: Python<'py>) -> Option<Bound<'py, PyList>> {
        if self.events_vec.is_empty() {
            return None;
        }
        let evs = std::mem::take(&mut self.events_vec);
        Some(PyList::new_bound(py, evs.into_iter().map(|e| e.into_bound(py))))
    }

    // ---- is_finished / finish reason ----

    fn is_finished(&self) -> bool {
        self.status as i32 > RequestStatus::PREEMPTED as i32
    }

    fn get_finished_reason<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        // Delegate to the Python `RequestStatus.get_finished_reason` (via
        // _FINISHED_REASON_MAP) — that logic lives in vllm.v1.request.
        let request_mod = PyModule::import_bound(py, "vllm.v1.request")?;
        let rs_cls = request_mod.getattr("RequestStatus")?;
        let py_status = rs_cls.call1((self.status as i32,))?;
        let reason = rs_cls.call_method1("get_finished_reason", (py_status,))?;
        Ok(reason.unbind())
    }

    // ---- mutating helpers ----

    /// append_output_token_ids: accepts int or list[int]. Updates
    /// _output_token_ids, _all_token_ids, then calls update_block_hashes.
    #[pyo3(name = "append_output_token_ids")]
    fn py_append_output_token_ids(
        slf: Py<Self>,
        py: Python<'_>,
        token_ids: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // Extract tokens first (outside borrow).
        let tokens: Vec<i64> = if let Ok(single) = token_ids.extract::<i64>() {
            vec![single]
        } else {
            token_ids.extract()?
        };
        let has_hasher = {
            let mut this = slf.borrow_mut(py);
            this.output_token_ids_vec.0.extend_from_slice(&tokens);
            this.all_token_ids_vec.0.extend_from_slice(&tokens);
            this.block_hasher.is_some()
        };
        if has_hasher {
            // Invoke the hasher — requires releasing the borrow.
            let hasher = slf.borrow(py).block_hasher.as_ref().unwrap().clone_ref(py);
            let out = hasher.call1(py, (slf.clone_ref(py),))?;
            let list: Bound<'_, PyList> = out.downcast_bound::<PyList>(py)?.clone();
            let this = slf.borrow(py);
            let mut bh = this.block_hashes_vec.borrow_mut();
            for item in list.iter() {
                let bytes: Vec<u8> = item.extract()?;
                bh.push(bytes);
            }
        }
        Ok(())
    }

    fn get_skip_reading_prefix_cache(&self, py: Python<'_>) -> PyResult<bool> {
        if let Some(sp) = self.sampling_params.as_ref() {
            let v = sp.bind(py).getattr(intern!(py, "skip_reading_prefix_cache"))?;
            if !v.is_none() {
                return v.extract();
            }
        }
        if let Some(pp) = self.pooling_params.as_ref() {
            let v = pp.bind(py).getattr(intern!(py, "skip_reading_prefix_cache"))?;
            if !v.is_none() {
                return v.extract();
            }
        }
        Ok(false)
    }

    fn get_num_encoder_embeds(&self, py: Python<'_>, input_id: usize) -> PyResult<usize> {
        let feat = self
            .mm_features
            .get(input_id)
            .ok_or_else(|| PyIndexError::new_err("input_id out of range"))?;
        let pos = feat.bind(py).getattr(intern!(py, "mm_position"))?;
        let n = pos.call_method0(intern!(py, "get_num_embeds"))?;
        n.extract()
    }

    // ---- ordering (used in priority scheduling) ----

    fn __lt__(&self, other: PyRef<'_, Request>) -> bool {
        if self.priority != other.priority {
            return self.priority < other.priority;
        }
        if self.arrival_time != other.arrival_time {
            return self.arrival_time < other.arrival_time;
        }
        if self.request_id != other.request_id {
            return self.request_id < other.request_id;
        }
        // Fall back to pointer identity to break ties.
        (self as *const _ as usize) < (&*other as *const _ as usize)
    }

    fn __repr__(&self) -> String {
        format!(
            "Request(request_id={:?}, status={:?}, num_tokens={}, num_output={})",
            self.request_id,
            self.status,
            self.all_token_ids_vec.0.len(),
            self.output_token_ids_vec.0.len(),
        )
    }

    // ---- pickle support ----

    fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        // Snapshot just the scalar fields + token buffers. Python-held
        // opaque fields (sampling_params etc.) are forwarded as-is.
        PyTuple::new_bound(
            py,
            &[
                self.request_id.clone().into_py(py),
                self.client_index.into_py(py),
                self.priority.into_py(py),
                self.arrival_time.into_py(py),
                self.sampling_params
                    .as_ref()
                    .map(|p| p.clone_ref(py).into_py(py))
                    .unwrap_or_else(|| py.None()),
                self.pooling_params
                    .as_ref()
                    .map(|p| p.clone_ref(py).into_py(py))
                    .unwrap_or_else(|| py.None()),
                (self.status as i32).into_py(py),
                self.max_tokens.into_py(py),
                self.num_prompt_tokens.into_py(py),
                self.num_cached_tokens.into_py(py),
                self.num_computed_tokens.into_py(py),
                self.num_output_placeholders.into_py(py),
                self.num_external_computed_tokens.into_py(py),
                self.num_nans_in_logits.into_py(py),
                self.num_preemptions.into_py(py),
                self.all_token_ids_vec.0.clone().into_py(py),
                self.output_token_ids_vec.0.clone().into_py(py),
                self.spec_token_ids_vec.0.clone().into_py(py),
                self.resumable.into_py(py),
                self.skip_reading_prefix_cache.into_py(py),
                self.is_prefill_chunk.into_py(py),
                self.discard_latest_async_tokens.into_py(py),
                self.cache_salt.clone().into_py(py),
            ],
        )
    }
}
