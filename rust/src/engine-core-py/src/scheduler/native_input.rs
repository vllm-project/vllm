// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The native input path: the input socket thread of `vllm-engine-core-io`
//! decodes add requests off the GIL, and admits the ones that meet no Python
//! collaborator without a Python `Request` (the fast path). Every other
//! message goes to `EngineCoreProc._process_input_message` with its original
//! frames, as the Python input thread handles it (the slow path).
//!
//! Each request is classified once, when it arrives: a fast-path request
//! never gets a Python `Request`, so whatever would need one must send the
//! request down the slow path. A fast-path request is a [`NativeAddRequest`],
//! which exposes to Python only what the engine and the scheduler binding
//! read of a request it did not build.

use std::sync::{Arc, Mutex};

use bytes::Bytes;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyList, PyString, PyTuple};
use vllm_engine_core::hash::{BlockHash, HashAlgo, RequestBlockHasher, RequestExtraKeys, Value};
use vllm_engine_core::sched::request::{BlockHasher, KvCacheReportMode, NewRequest, Request};
use vllm_engine_core_client::protocol::encode_msgpack;
use vllm_engine_core_client::protocol::request::{
    EngineCoreRequest, EngineCoreRequestType, decode_engine_core_request,
};
use vllm_engine_core_io::input::{InputConfig, InputReader};

use super::request::sampling_params_of;

/// How the engine hashes prompt blocks (`EngineCore.prefix_hash_config`), for
/// the fast path, which hashes in Rust only.
#[derive(Clone)]
struct PrefixHashing {
    hasher: Arc<RequestBlockHasher>,
}

struct NativeBlockHasher {
    hasher: Arc<RequestBlockHasher>,
    extra_keys: RequestExtraKeys,
}

impl BlockHasher for NativeBlockHasher {
    fn hash_new_blocks(&self, request: &Request) -> Vec<BlockHash> {
        self.hasher.new_block_hashes(
            request.all_token_ids(),
            request.block_hashes.len(),
            request.block_hashes.last(),
            &self.extra_keys,
        )
    }
}

/// Why a request takes the slow path, or `None` for the fast path: it needs
/// a Python `Request` or Python preprocessing (`preprocess_add_request`).
fn slow_path_reason(request: &EngineCoreRequest) -> Option<&'static str> {
    let Some(sp) = &request.sampling_params else {
        return Some("pooling");
    };
    if sp.structured_outputs.is_some() {
        // The structured output manager compiles the grammar of the Python
        // request.
        return Some("structured outputs");
    }
    if sp.extra_args.as_ref().is_some_and(|args| !args.is_empty()) {
        // KV and EC transfer params and the KV cache report mode.
        return Some("extra args");
    }
    if sp.prompt_logprob_token_ids.is_some() {
        return Some("prompt logprob token ids");
    }
    if request.mm_features.as_ref().is_some_and(|f| !f.is_empty()) {
        // The multimodal receiver cache, and hashing with multimodal keys.
        return Some("multimodal inputs");
    }
    if request.prompt_embeds.is_some() || request.prompt_is_token_ids.is_some() {
        return Some("prompt embeds");
    }
    // TODO: a LoRA request needs only its `LoRARequest` payload.
    if request.lora_request.is_some() {
        return Some("LoRA");
    }
    if request.resumable || request.session_id.is_some() {
        // Session updates go through the Python request.
        return Some("sessions");
    }
    if request.kv_hints.is_some() || request.trace_headers.is_some() {
        return Some("kv hints or trace headers");
    }
    if request.prompt_token_ids.is_none() {
        return Some("no prompt token ids");
    }
    None
}

static TOKEN_BUFFER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// A `TokenBuffer` of `tokens`.
pub(super) fn token_buffer<'py>(py: Python<'py>, tokens: &[i32]) -> PyResult<Bound<'py, PyAny>> {
    let buffer = TOKEN_BUFFER.import(py, "vllm.v1.token_buffer", "TokenBuffer")?.call0()?;
    let bytes = PyBytes::new(py, bytemuck_i32(tokens));
    buffer.call_method1(intern!(py, "frombytes"), (bytes,))?;
    Ok(buffer)
}

fn bytemuck_i32(tokens: &[i32]) -> &[u8] {
    // SAFETY: an `i32` slice is valid as bytes of four times its length.
    unsafe { std::slice::from_raw_parts(tokens.as_ptr().cast(), std::mem::size_of_val(tokens)) }
}

static WAITING: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// A request the native input path admitted without a Python `Request`.
///
/// It stands for the request where the engine and the scheduler binding hold
/// one: in the input queue, the scheduler's request table, and the requests
/// `finish_requests` returns. Python reads only these attributes of it.
#[pyclass(module = "vllm._rust_core")]
pub struct NativeAddRequest {
    #[pyo3(get)]
    request_id: Py<PyString>,
    #[pyo3(get)]
    client_index: usize,
    #[pyo3(get)]
    abort_immediately: bool,
    #[pyo3(get)]
    sampling_params: Py<PyAny>,
    #[pyo3(get)]
    prompt_token_ids: Py<PyAny>,
    /// Set by `finish_requests`.
    #[pyo3(get, set)]
    status: Py<PyAny>,
    /// Taken by the scheduler when it adds the request.
    new_request: Mutex<Option<NewRequest>>,
}

#[pymethods]
impl NativeAddRequest {
    #[getter]
    fn pooling_params(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[getter]
    fn kv_transfer_params(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[getter]
    fn ec_transfer_params(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[getter]
    fn prompt_embeds(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[getter]
    fn prompt_is_token_ids(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[getter]
    fn lora_request(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[getter]
    fn mm_features<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::empty(py)
    }
}

impl NativeAddRequest {
    /// The Rust request's arguments; `None` once taken.
    pub(super) fn take_new_request(&self) -> Option<NewRequest> {
        self.new_request.lock().unwrap().take()
    }

    /// Build the fast-path request of `request`, with the Python payloads
    /// the scheduler sends to the workers.
    fn new(
        py: Python<'_>,
        request: EngineCoreRequest,
        hashing: Option<&PrefixHashing>,
        decode_sampling_params: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let dto = request.sampling_params.expect("a fast-path request has sampling params");
        let encoded = encode_msgpack(&dto).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let sp = decode_sampling_params.call1((PyBytes::new(py, &encoded),))?;
        let sampling_params = sampling_params_of(&sp, false, KvCacheReportMode::Incremental)?;
        let prompt: Vec<i32> = (request.prompt_token_ids.as_ref())
            .and_then(|ids| ids.to_u32_vec())
            .ok_or_else(|| PyValueError::new_err("prompt token ids must be 32-bit integers"))?
            .into_iter()
            .map(|t| t as i32)
            .collect();
        let cache_salt = request.cache_salt.filter(|salt| !salt.is_empty());
        let block_hasher = hashing.map(|hashing| -> Arc<dyn BlockHasher> {
            Arc::new(NativeBlockHasher {
                hasher: hashing.hasher.clone(),
                extra_keys: RequestExtraKeys {
                    lora: None,
                    cache_salt: cache_salt.clone().map(Value::str),
                },
            })
        });
        let prompt_token_ids = token_buffer(py, &prompt)?.unbind();
        let new_request = NewRequest {
            request_id: request.request_id.clone(),
            prompt_token_ids: Some(prompt),
            num_prompt_embeds: None,
            sampling_params: Some(sampling_params),
            pooling_params: None,
            client_index: request.client_index as usize,
            arrival_time: Some(request.arrival_time),
            prompt_is_token_ids: None,
            mm_features: Vec::new(),
            lora_int_id: None,
            lora_name: None,
            cache_salt,
            priority: i64::from(request.priority),
            block_hasher,
            resumable: false,
            session_id: None,
            abort_immediately: request.abort_immediately,
            // TODO: honor the frontend's prompt block hashes; the engine
            // recomputes them for now, and announces no `prefix_hash_config`.
            block_hashes: Vec::new(),
            do_remote_prefill: false,
        };
        let waiting = WAITING
            .get_or_try_init(py, || -> PyResult<_> {
                Ok(py
                    .import("vllm.v1.request")?
                    .getattr("RequestStatus")?
                    .getattr("WAITING")?
                    .unbind())
            })?
            .clone_ref(py);
        Ok(Self {
            request_id: PyString::new(py, &request.request_id).unbind(),
            client_index: request.client_index as usize,
            abort_immediately: request.abort_immediately,
            sampling_params: sp.unbind(),
            prompt_token_ids,
            status: waiting,
            new_request: Mutex::new(Some(new_request)),
        })
    }
}

/// The input socket thread of the native input path.
#[pyclass(frozen, module = "vllm._rust_core")]
pub struct EngineInputReader {
    inner: InputReader,
}

#[pymethods]
impl EngineInputReader {
    /// Connect the input sockets, send them `ready_payload`, join the DP
    /// coordinator if any, then call `ready_event.set()` and start handling
    /// messages.
    ///
    /// `decode_sampling_params(bytes)` builds the Python `SamplingParams` of
    /// a fast-path request as `preprocess_add_request` leaves it.
    /// `prefix_hash_config` is the engine's (`EngineCore.prefix_hash_config`),
    /// with a `*_cbor` algorithm.
    #[new]
    #[pyo3(signature = (
        engine, input_addresses, coordinator_address, identity, ready_payload, ready_event,
        decode_sampling_params, prefix_hash_config=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        engine: Py<PyAny>,
        input_addresses: Vec<String>,
        coordinator_address: Option<String>,
        identity: Vec<u8>,
        ready_payload: Vec<u8>,
        ready_event: Py<PyAny>,
        decode_sampling_params: Py<PyAny>,
        prefix_hash_config: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let hashing = prefix_hash_config.map(|c| prefix_hashing(&c)).transpose()?;
        let input_queue = engine.bind(py).getattr(intern!(py, "input_queue"))?.unbind();
        let add_type = py
            .import("vllm.v1.engine")?
            .getattr("EngineCoreRequestType")?
            .getattr("ADD")?
            .unbind();
        let config = InputConfig {
            input_addresses,
            coordinator_address,
            identity: Bytes::from(identity),
            ready_payload: Bytes::from(ready_payload),
        };
        let on_ready = Box::new(move || {
            Python::attach(|py| {
                if let Err(err) = ready_event.bind(py).call_method0(intern!(py, "set")) {
                    err.print(py);
                }
            });
        });
        let on_message = Box::new(move |frames: Vec<Bytes>| -> Result<(), String> {
            // Decoded off the GIL.
            let fast = (frames[0].as_ref() == [EngineCoreRequestType::Add as u8])
                .then(|| decode_engine_core_request(&frames[1..]).ok())
                .flatten()
                .filter(|request| slow_path_reason(request).is_none());
            Python::attach(|py| {
                let result = match fast {
                    Some(request) => {
                        let wave = request.current_wave;
                        NativeAddRequest::new(
                            py,
                            request,
                            hashing.as_ref(),
                            decode_sampling_params.bind(py),
                        )
                        .and_then(|request| Py::new(py, request))
                        .and_then(|request| {
                            let item = PyTuple::new(
                                py,
                                [
                                    add_type.clone_ref(py).into_bound(py),
                                    (request, wave).into_pyobject(py)?.into_any(),
                                ],
                            )?;
                            input_queue.bind(py).call_method1(intern!(py, "put_nowait"), (item,))
                        })
                    }
                    None => PyList::new(py, frames[1..].iter().map(|f| PyBytes::new(py, f)))
                        .and_then(|data| {
                            engine.bind(py).call_method1(
                                intern!(py, "_process_input_message"),
                                (PyBytes::new(py, &frames[0]), data),
                            )
                        }),
                };
                result.map(drop).map_err(|err| {
                    err.print(py);
                    err.to_string()
                })
            })
        });
        let on_error = Box::new(|error: String| {
            Python::attach(|py| {
                let _ = py
                    .import("vllm.logger")
                    .and_then(|m| m.call_method1("init_logger", ("vllm.v1.engine.core",)))
                    .and_then(|logger| {
                        logger.call_method1(
                            "error",
                            ("The engine core input thread failed: %s", error),
                        )
                    });
            });
        });
        let inner = py
            .detach(|| InputReader::start(config, on_ready, on_message, on_error))
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self { inner })
    }

    /// `threading.Thread.is_alive`.
    fn is_alive(&self) -> bool {
        self.inner.is_alive()
    }
}

/// The Rust hasher of a `PrefixHashConfig`.
fn prefix_hashing(config: &Bound<'_, PyAny>) -> PyResult<PrefixHashing> {
    let py = config.py();
    let name: String = config.getattr(intern!(py, "hash_algo"))?.extract()?;
    let algo = HashAlgo::from_name(&name)
        .filter(|_| name.ends_with("_cbor"))
        .ok_or_else(|| PyValueError::new_err(format!("{name} is not a *_cbor algorithm")))?;
    let none_hash = config.getattr(intern!(py, "none_hash"))?;
    let none_hash = BlockHash::from_slice(none_hash.cast::<PyBytes>()?.as_bytes());
    Ok(PrefixHashing {
        hasher: Arc::new(RequestBlockHasher::new(
            algo,
            none_hash,
            config.getattr(intern!(py, "hash_block_size"))?.extract()?,
            config.getattr(intern!(py, "num_lookahead_tokens"))?.extract()?,
        )),
    })
}
