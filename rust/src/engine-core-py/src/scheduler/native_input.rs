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
use vllm_engine_core_io::input::{InputConfig, InputReader};
use vllm_engine_core_protocol::request::{
    EngineCoreRequest, EngineCoreRequestType, RawRequestPayloads, decode_engine_core_request,
    raw_request_payloads,
};

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
///
/// `kv_transfer_params`, the only extra arg the fast path takes, are taken
/// with `accept_kv_transfer_params`: when the engine's KV connector is a
/// native one that reads them off the Rust request.
fn slow_path_reason(
    request: &EngineCoreRequest,
    accept_kv_transfer_params: bool,
) -> Option<&'static str> {
    let Some(sp) = &request.sampling_params else {
        return Some("pooling");
    };
    if sp.structured_outputs.is_some() {
        // The structured output manager compiles the grammar of the Python
        // request.
        return Some("structured outputs");
    }
    if let Some(args) = &sp.extra_args
        && args.keys().any(|key| !(accept_kv_transfer_params && key == KV_TRANSFER_PARAMS))
    {
        // EC transfer params, the KV cache report mode, and KV transfer
        // params for a connector that reads them off the Python request.
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

const KV_TRANSFER_PARAMS: &str = "kv_transfer_params";

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
    /// Set by the engine to abort a request it rejects while paused, so that
    /// the KV connector frees what it holds for it.
    #[pyo3(get)]
    abort_immediately: bool,
    #[pyo3(get)]
    sampling_params: Py<PyAny>,
    #[pyo3(get)]
    prompt_token_ids: Py<PyAny>,
    #[pyo3(get)]
    lora_request: Py<PyAny>,
    /// `sampling_params.extra_args["kv_transfer_params"]`, as the Python
    /// `Request` reads them.
    #[pyo3(get)]
    kv_transfer_params: Py<PyAny>,
    /// Set by `finish_requests`.
    #[pyo3(get, set)]
    status: Py<PyAny>,
    /// Taken by the scheduler when it adds the request, with the
    /// `kv_transfer_params` the native KV connector reads.
    new_request: Mutex<Option<(NewRequest, Option<serde_json::Value>)>>,
}

#[pymethods]
impl NativeAddRequest {
    #[getter]
    fn pooling_params(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[setter]
    fn set_abort_immediately(&mut self, value: bool) {
        self.abort_immediately = value;
        if let Some((new, _)) = self.new_request.get_mut().unwrap() {
            new.abort_immediately = value;
        }
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
    fn mm_features<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::empty(py)
    }
}

impl NativeAddRequest {
    /// The Rust request's arguments; `None` once taken.
    pub(super) fn take_new_request(&self) -> Option<(NewRequest, Option<serde_json::Value>)> {
        self.new_request.lock().unwrap().take()
    }

    /// Replace the prompt token ids Python reads, once the Rust request's
    /// were truncated.
    pub(super) fn set_prompt_token_ids(&mut self, prompt_token_ids: Py<PyAny>) {
        self.prompt_token_ids = prompt_token_ids;
    }

    /// Build the fast-path request of `request`, with the Python payloads
    /// the scheduler sends to the workers.
    fn new(
        py: Python<'_>,
        request: EngineCoreRequest,
        raw: RawRequestPayloads,
        hashing: Option<&PrefixHashing>,
        decoders: &PayloadDecoders,
    ) -> PyResult<Self> {
        let sp = decoders.decode(
            py,
            &decoders.sampling_params,
            raw.sampling_params.as_ref().expect("a fast-path request has sampling params"),
        )?;
        let sampling_params = sampling_params_of(&sp, false, KvCacheReportMode::Incremental)?;
        // As `Request.__init__` reads them.
        let extra_args = sp.getattr(intern!(py, "extra_args"))?;
        let kv_transfer_params = if extra_args.is_none() {
            py.None().into_bound(py)
        } else {
            extra_args.call_method1(intern!(py, "get"), (KV_TRANSFER_PARAMS,))?
        };
        let wire_kv_transfer_params = (request.sampling_params.as_ref())
            .and_then(|sp| sp.extra_args.as_ref())
            .and_then(|args| args.get(KV_TRANSFER_PARAMS))
            .cloned();
        let prompt: Vec<i32> = (request.prompt_token_ids.as_ref())
            .and_then(|ids| ids.to_u32_vec())
            .ok_or_else(|| PyValueError::new_err("prompt token ids must be 32-bit integers"))?
            .into_iter()
            .map(|t| t as i32)
            .collect();
        let lora_request = match &raw.lora_request {
            Some(lora) => decoders.decode(py, &decoders.lora_request, lora)?,
            None => py.None().into_bound(py),
        };
        let lora = request.lora_request.as_ref();
        let cache_salt = request.cache_salt.filter(|salt| !salt.is_empty());
        let block_hasher = hashing.map(|hashing| -> Arc<dyn BlockHasher> {
            Arc::new(NativeBlockHasher {
                hasher: hashing.hasher.clone(),
                extra_keys: RequestExtraKeys {
                    lora: lora.map(|l| (Value::str(&l.lora_name), Value::str(&l.lora_path))),
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
            lora_int_id: lora.map(|l| l.lora_int_id as i64),
            lora_name: lora.map(|l| l.lora_name.clone()),
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
            lora_request: lora_request.unbind(),
            kv_transfer_params: kv_transfer_params.unbind(),
            status: waiting,
            new_request: Mutex::new(Some((new_request, wire_kv_transfer_params))),
        })
    }
}

/// Build the Python payloads of a fast-path request from their msgpack bytes
/// as the frontend sent them, so that what the Rust types leave out survives.
struct PayloadDecoders {
    /// `SamplingParams`, as `preprocess_add_request` leaves them.
    sampling_params: Py<PyAny>,
    /// `LoRARequest`.
    lora_request: Py<PyAny>,
}

impl PayloadDecoders {
    fn decode<'py>(
        &self,
        py: Python<'py>,
        decoder: &Py<PyAny>,
        payload: &Bytes,
    ) -> PyResult<Bound<'py, PyAny>> {
        decoder.bind(py).call1((PyBytes::new(py, payload),))
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
    /// `decode_sampling_params(bytes)` and `decode_lora_request(bytes)` build
    /// the Python `SamplingParams` (as `preprocess_add_request` leaves them)
    /// and `LoRARequest` of a fast-path request.
    /// `prefix_hash_config` is the engine's (`EngineCore.prefix_hash_config`),
    /// with a `*_cbor` algorithm. `accept_kv_transfer_params` when the
    /// scheduler's KV connector is a native one, which reads them off the
    /// Rust request.
    #[new]
    #[pyo3(signature = (
        engine, input_addresses, coordinator_address, identity, ready_payload, ready_event,
        decode_sampling_params, decode_lora_request, prefix_hash_config=None,
        accept_kv_transfer_params=false,
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
        decode_lora_request: Py<PyAny>,
        prefix_hash_config: Option<Bound<'_, PyAny>>,
        accept_kv_transfer_params: bool,
    ) -> PyResult<Self> {
        let decoders = PayloadDecoders {
            sampling_params: decode_sampling_params,
            lora_request: decode_lora_request,
        };
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
                .then(|| {
                    let request = decode_engine_core_request(&frames[1..]).ok()?;
                    let raw = raw_request_payloads(&frames[1]).ok()?;
                    Some((request, raw))
                })
                .flatten()
                .filter(|(request, _)| {
                    slow_path_reason(request, accept_kv_transfer_params).is_none()
                });
            Python::attach(|py| {
                let result = match fast {
                    Some((request, raw)) => {
                        let wave = request.current_wave;
                        NativeAddRequest::new(py, request, raw, hashing.as_ref(), &decoders)
                            .and_then(|request| Py::new(py, request))
                            .and_then(|request| {
                                let item = PyTuple::new(
                                    py,
                                    [
                                        add_type.clone_ref(py).into_bound(py),
                                        (request, wave).into_pyobject(py)?.into_any(),
                                    ],
                                )?;
                                input_queue
                                    .bind(py)
                                    .call_method1(intern!(py, "put_nowait"), (item,))
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
