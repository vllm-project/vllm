// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Bridging the engine's Python `Request`s and the Rust scheduler's.
//!
//! The Rust [`Request`] is the source of truth while the scheduler holds a
//! request. The Python object stays with the binding (for the payloads the
//! scheduler passes along, and for Python collaborators) and is brought up to
//! date ([`sync_request`]) only right before Python code reads it, so that
//! the plain decode path touches no Python request objects.

use std::hash::BuildHasherDefault;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyList, PyModule, PyString, PyType};
use rustc_hash::FxHasher;
use vllm_engine_core::hash::{BlockHash, HashAlgo, RequestBlockHasher, RequestExtraKeys};
use vllm_engine_core::sched::multimodal::{
    MultiModalData, MultiModalFeatureSpec, PlaceholderRange,
};
use vllm_engine_core::sched::request::{
    BlockHasher, KvCacheReportMode, NewRequest, PoolingParams, RepetitionDetectionParams, Request,
    SamplingParams, StopReason,
};

use super::output::{prefill_stats_to_py, spec_decode_metrics_to_py};
use super::types::types;
use crate::hash::{
    HasherParams, RustRequestBlockHasher, current_none_hash, parse_algo, request_extra_keys,
};
use crate::kv_cache::views::SchedulerKv;

/// The binding's side of a request.
pub struct PyReq {
    /// The engine's `Request`.
    pub obj: Py<PyAny>,
    /// `request.request_id`, the object used as the key in Python outputs.
    pub id: Py<PyString>,
    pub trace_headers: Option<Py<PyAny>>,
}

/// The requests the scheduler holds, in arrival order.
pub type RequestTable = IndexMap<String, PyReq, BuildHasherDefault<FxHasher>>;

/// A Python block hasher and how it hashes in Rust (`None`: it can't).
type CachedHasher = (Py<PyAny>, Option<Arc<HasherParams>>);

/// Payload of the panic that carries a Python exception out of a hook; see
/// [`Shared::raise`].
pub struct HookError;

/// State shared by the binding and the hooks the Rust scheduler calls.
pub struct Shared {
    pub table: Mutex<RequestTable>,
    pub kv: Arc<SchedulerKv>,
    pub uses_mrope: bool,
    /// The Python exception a hook raised, re-raised by the binding.
    hook_error: Mutex<Option<PyErr>>,
    /// The `SchedulerOutput` built for the connector during `schedule`.
    pub pending_output: Mutex<Option<Py<PyAny>>>,
    /// The last Python block hasher seen, and how it hashes in Rust.
    hasher_cache: Mutex<Option<CachedHasher>>,
}

impl Shared {
    pub fn new(kv: Arc<SchedulerKv>, uses_mrope: bool) -> Self {
        Self {
            table: Mutex::new(RequestTable::default()),
            kv,
            uses_mrope,
            hook_error: Mutex::new(None),
            pending_output: Mutex::new(None),
            hasher_cache: Mutex::new(None),
        }
    }

    /// Abort the scheduler call with `err`: hooks return plain values, so a
    /// Python exception unwinds to the binding, which re-raises it
    /// ([`Self::take_error`]).
    pub fn raise(&self, err: PyErr) -> ! {
        *self.hook_error.lock().unwrap() = Some(err);
        std::panic::resume_unwind(Box::new(HookError))
    }

    pub fn check<T>(&self, result: PyResult<T>) -> T {
        result.unwrap_or_else(|err| self.raise(err))
    }

    pub fn take_error(&self) -> Option<PyErr> {
        self.hook_error.lock().unwrap().take()
    }

    /// The Python request of `request_id`.
    pub fn py_request(&self, py: Python<'_>, request_id: &str) -> Py<PyAny> {
        self.table.lock().unwrap()[request_id].obj.clone_ref(py)
    }
}

static ARRAY_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Token ids of a `TokenBuffer` (an int32 `array.array`) or a sequence.
pub fn token_ids_of(tokens: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    let py = tokens.py();
    let array_type = ARRAY_TYPE.import(py, "array", "array")?;
    if tokens.is_instance(array_type)?
        && tokens.getattr(intern!(py, "typecode"))?.eq("i")?
        && tokens.getattr(intern!(py, "itemsize"))?.extract::<usize>()? == size_of::<i32>()
    {
        let (address, len): (usize, usize) =
            tokens.call_method0(intern!(py, "buffer_info"))?.extract()?;
        if len == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: `buffer_info` is the address and length of the array's int32
        // storage; no Python code runs while it is copied.
        return Ok(unsafe { std::slice::from_raw_parts(address as *const i32, len) }.to_vec());
    }
    tokens.extract()
}

fn sampling_params(request: &Bound<'_, PyAny>, sp: &Bound<'_, PyAny>) -> PyResult<SamplingParams> {
    let py = sp.py();
    let is_set =
        |name: &Bound<'_, PyString>| -> PyResult<bool> { Ok(!sp.getattr(name)?.is_none()) };
    let repetition_detection = sp.getattr(intern!(py, "repetition_detection"))?;
    let repetition_detection = if repetition_detection.is_none() {
        None
    } else {
        // Python treats non-positive values like zero.
        let field = |name: &str| -> PyResult<usize> {
            Ok(repetition_detection.getattr(name)?.extract::<i64>()?.max(0) as usize)
        };
        Some(RepetitionDetectionParams {
            max_pattern_size: field("max_pattern_size")?,
            min_pattern_size: field("min_pattern_size")?,
            min_count: field("min_count")?,
        })
    };
    let stop_token_ids = sp.getattr(intern!(py, "stop_token_ids"))?;
    Ok(SamplingParams {
        max_tokens: sp.getattr(intern!(py, "max_tokens"))?.extract()?,
        min_tokens: sp.getattr(intern!(py, "min_tokens"))?.extract()?,
        eos_token_id: sp.getattr(intern!(py, "eos_token_id"))?.extract()?,
        stop_token_ids: if stop_token_ids.is_none() {
            Vec::new()
        } else {
            stop_token_ids.extract()?
        },
        repetition_detection,
        skip_reading_prefix_cache: sp
            .getattr(intern!(py, "skip_reading_prefix_cache"))?
            .is_truthy()?,
        has_prompt_logprobs: is_set(intern!(py, "prompt_logprobs"))?,
        has_prompt_logprob_token_ids: is_set(intern!(py, "prompt_logprob_token_ids"))?,
        prompt_logprob_start: sp.getattr(intern!(py, "prompt_logprob_start"))?.extract()?,
        num_logprobs: sp
            .getattr(intern!(py, "num_logprobs"))?
            .extract::<Option<i64>>()?
            .map(|n| n.max(0) as usize),
        use_structured_output: !request
            .getattr(intern!(py, "structured_output_request"))?
            .is_none(),
        kv_cache_report_mode: kv_cache_report_mode(request)?,
        watermarking: sp.getattr(intern!(py, "watermarking"))?.is_truthy()?,
        temperature: sp.getattr(intern!(py, "temperature"))?.extract()?,
        has_trace_decode_token_ids: is_set(intern!(py, "trace_decode_token_ids"))?,
    })
}

fn kv_cache_report_mode(request: &Bound<'_, PyAny>) -> PyResult<KvCacheReportMode> {
    let py = request.py();
    let mode = request.getattr(intern!(py, "kv_cache_report_mode"));
    Ok(match mode {
        Ok(mode) if mode.eq("full")? => KvCacheReportMode::Full,
        _ => KvCacheReportMode::Incremental,
    })
}

fn mm_feature(feature: &Bound<'_, PyAny>) -> PyResult<MultiModalFeatureSpec> {
    let py = feature.py();
    let pos = feature.getattr(intern!(py, "mm_position"))?;
    let is_embed = pos.getattr(intern!(py, "is_embed"))?;
    let is_embed: Option<Vec<bool>> = if is_embed.is_none() {
        None
    } else {
        Some(is_embed.call_method0(intern!(py, "tolist"))?.extract()?)
    };
    let data = feature.getattr(intern!(py, "data"))?;
    let data = if data.is_none() {
        MultiModalData::Missing
    } else if data.contains("address")? {
        MultiModalData::ShmAddress
    } else {
        MultiModalData::Payload
    };
    Ok(MultiModalFeatureSpec {
        modality: feature.getattr(intern!(py, "modality"))?.extract()?,
        identifier: feature.getattr(intern!(py, "identifier"))?.extract()?,
        mm_position: PlaceholderRange::new(
            pos.getattr(intern!(py, "offset"))?.extract()?,
            pos.getattr(intern!(py, "length"))?.extract()?,
            is_embed.as_deref(),
        ),
        mm_hash: feature.getattr(intern!(py, "mm_hash"))?.extract()?,
        data,
    })
}

/// Convert an engine `Request` (as `add_request` receives it) into the
/// arguments of the Rust request.
pub fn new_request(shared: &Arc<Shared>, request: &Bound<'_, PyAny>) -> PyResult<NewRequest> {
    let py = request.py();
    let sp = request.getattr(intern!(py, "sampling_params"))?;
    let pp = request.getattr(intern!(py, "pooling_params"))?;
    let prompt_token_ids = request.getattr(intern!(py, "prompt_token_ids"))?;
    let prompt_embeds = request.getattr(intern!(py, "prompt_embeds"))?;
    let lora_request = request.getattr(intern!(py, "lora_request"))?;
    let lora_request = lora_request.is_truthy()?.then_some(lora_request);
    let kv_transfer_params = request.getattr(intern!(py, "kv_transfer_params"))?;
    let do_remote_prefill = kv_transfer_params.is_truthy()?
        && kv_transfer_params
            .call_method1(intern!(py, "get"), ("do_remote_prefill",))?
            .is_truthy()?;
    let mm_features = request
        .getattr(intern!(py, "mm_features"))?
        .try_iter()?
        .map(|f| mm_feature(&f?))
        .collect::<PyResult<Vec<_>>>()?;
    let block_hasher = request.getattr(intern!(py, "_block_hasher"))?;
    let block_hasher: Option<Arc<dyn BlockHasher>> = if block_hasher.is_none() {
        None
    } else {
        Some(Arc::new(SchedBlockHasher::new(
            shared,
            request,
            &block_hasher,
        )?))
    };
    Ok(NewRequest {
        request_id: request.getattr(intern!(py, "request_id"))?.extract()?,
        prompt_token_ids: if prompt_token_ids.is_none() {
            None
        } else {
            Some(token_ids_of(&prompt_token_ids)?)
        },
        num_prompt_embeds: if prompt_embeds.is_none() {
            None
        } else {
            Some(prompt_embeds.getattr(intern!(py, "shape"))?.get_item(0)?.extract()?)
        },
        sampling_params: if sp.is_none() {
            None
        } else {
            Some(sampling_params(request, &sp)?)
        },
        pooling_params: if pp.is_none() {
            None
        } else {
            Some(PoolingParams {
                skip_reading_prefix_cache: pp
                    .getattr(intern!(py, "skip_reading_prefix_cache"))?
                    .is_truthy()?,
            })
        },
        client_index: request.getattr(intern!(py, "client_index"))?.extract()?,
        arrival_time: Some(request.getattr(intern!(py, "arrival_time"))?.extract()?),
        prompt_is_token_ids: request.getattr(intern!(py, "prompt_is_token_ids"))?.extract()?,
        mm_features,
        lora_int_id: lora_request
            .as_ref()
            .map(|l| l.getattr(intern!(py, "lora_int_id"))?.extract())
            .transpose()?,
        lora_name: lora_request
            .as_ref()
            .map(|l| l.getattr(intern!(py, "lora_name"))?.extract())
            .transpose()?,
        cache_salt: request.getattr(intern!(py, "cache_salt"))?.extract()?,
        priority: request.getattr(intern!(py, "priority"))?.extract()?,
        block_hasher,
        resumable: request.getattr(intern!(py, "resumable"))?.extract()?,
        session_id: request.getattr(intern!(py, "session_id"))?.extract()?,
        abort_immediately: request.getattr(intern!(py, "abort_immediately"))?.extract()?,
        block_hashes: request
            .getattr(intern!(py, "block_hashes"))?
            .try_iter()?
            .map(|h| Ok(BlockHash::from_slice(h?.cast::<PyBytes>()?.as_bytes())))
            .collect::<PyResult<_>>()?,
        do_remote_prefill,
    })
}

/// Bring the token ids of the Python request up to the Rust request's.
///
/// Output tokens are append-only while a request is scheduled, except that a
/// streaming-input update (`_update_request_as_session`) moves the kept
/// output into the prompt: then all of them are brought up to date.
pub fn sync_tokens(obj: &Bound<'_, PyAny>, request: &Request) -> PyResult<()> {
    let py = obj.py();
    let output = obj.getattr(intern!(py, "_output_token_ids"))?;
    let output = output.cast::<PyList>()?;
    let rust_output = request.output_token_ids();
    let synced = output.len();
    if synced == rust_output.len() && !request.resumable {
        return Ok(());
    }
    if synced > rust_output.len()
        || (request.resumable
            && obj.getattr(intern!(py, "num_prompt_tokens"))?.extract::<usize>()?
                != request.num_prompt_tokens)
    {
        return sync_session_tokens(obj, request);
    }
    if synced == rust_output.len() {
        return Ok(());
    }
    let new = PyList::new(py, &rust_output[synced..])?;
    output.call_method1(intern!(py, "extend"), (&new,))?;
    obj.getattr(intern!(py, "_all_token_ids"))?
        .call_method1(intern!(py, "extend"), (&new,))?;
    Ok(())
}

/// The tokens of a streaming-input session after an update: the prompt
/// (lists changed in place, as Python does) and the remaining output.
fn sync_session_tokens(obj: &Bound<'_, PyAny>, request: &Request) -> PyResult<()> {
    let py = obj.py();
    if !request.resumable {
        return Err(PyNotImplementedError::new_err(
            "the request's output tokens were truncated",
        ));
    }
    let replace = |name: &Bound<'_, PyString>, tokens: &[i32]| -> PyResult<()> {
        let list = obj.getattr(name)?;
        list.del_item(pyo3::types::PySlice::full(py))?;
        list.call_method1(intern!(py, "extend"), (PyList::new(py, tokens)?,))?;
        Ok(())
    };
    if let Some(prompt) = &request.prompt_token_ids {
        replace(intern!(py, "prompt_token_ids"), prompt)?;
    }
    replace(intern!(py, "_all_token_ids"), request.all_token_ids())?;
    replace(intern!(py, "_output_token_ids"), request.output_token_ids())?;
    obj.setattr(intern!(py, "num_prompt_tokens"), request.num_prompt_tokens)
}

/// Bring `request.block_hashes` of the Python request up to the Rust one's.
pub fn sync_block_hashes(obj: &Bound<'_, PyAny>, request: &Request) -> PyResult<()> {
    let py = obj.py();
    let hashes = obj.getattr(intern!(py, "block_hashes"))?;
    let hashes = hashes.cast::<PyList>()?;
    let rust_hashes = &request.block_hashes;
    let mut synced = hashes.len();
    // Hashes chain over their prefix: if the last synced one still matches,
    // so does every one before it.
    let stale = synced > rust_hashes.len()
        || (synced > 0
            && hashes.get_item(synced - 1)?.cast::<PyBytes>()?.as_bytes()
                != rust_hashes[synced - 1].as_bytes());
    if stale {
        hashes.del_slice(0, usize::MAX)?;
        synced = 0;
    }
    for hash in &rust_hashes[synced..] {
        hashes.append(PyBytes::new(py, hash.as_bytes()))?;
    }
    Ok(())
}

/// Bring the Python request up to date with the Rust request: everything the
/// scheduler changes that Python collaborators read.
///
/// Not mirrored: `events`, which the scheduler reports through its outputs
/// instead.
pub fn sync_request(obj: &Bound<'_, PyAny>, request: &Request) -> PyResult<()> {
    let py = obj.py();
    let types = types(py)?;
    let set = |name: &Bound<'_, PyString>, value: Bound<'_, PyAny>| obj.setattr(name, value);
    macro_rules! set_value {
        ($name:literal, $value:expr) => {
            set(intern!(py, $name), $value.into_bound_py_any(py)?)?
        };
    }
    set(intern!(py, "status"), types.status(py, request.status))?;
    set_value!("num_computed_tokens", request.num_computed_tokens);
    set_value!("num_output_placeholders", request.num_output_placeholders);
    set_value!("num_stale_output_tokens", request.num_stale_output_tokens);
    set_value!("drop_stale_output", request.drop_stale_output);
    set_value!("num_in_flight_tokens", request.num_in_flight_tokens);
    set_value!(
        "next_decode_eligible_step",
        request.next_decode_eligible_step
    );
    set_value!("last_sched_seq", request.last_sched_seq);
    set_value!("is_prefill_chunk", request.is_prefill_chunk);
    set_value!("shared_prefix_boundary", request.shared_prefix_boundary);
    set_value!("replay_start", request.replay_start);
    set_value!("num_nans_in_logits", request.num_nans_in_logits);
    set_value!("num_preemptions", request.num_preemptions);
    set_value!("max_tokens", request.max_tokens);
    set_value!("prefix_cache_hit_limit", request.prefix_cache_hit_limit);
    set_value!("resumable", request.resumable);
    set(
        intern!(py, "spec_token_ids"),
        PyList::new(py, &request.spec_token_ids)?.into_any(),
    )?;
    let stop_reason = match &request.stop_reason {
        None => py.None().into_bound(py),
        Some(StopReason::TokenId(t)) => t.into_pyobject(py)?.into_any(),
        Some(StopReason::Str(s)) => s.into_pyobject(py)?.into_any(),
    };
    set(intern!(py, "stop_reason"), stop_reason)?;
    let prefill_stats = match &request.prefill_stats {
        None => py.None().into_bound(py),
        Some(stats) => prefill_stats_to_py(py, types, stats)?,
    };
    set(intern!(py, "prefill_stats"), prefill_stats)?;
    let spec_decode_metrics = match &request.spec_decode_metrics {
        None => py.None().into_bound(py),
        Some(metrics) => spec_decode_metrics_to_py(py, types, metrics)?,
    };
    set(intern!(py, "spec_decode_metrics"), spec_decode_metrics)?;
    sync_tokens(obj, request)?;
    sync_block_hashes(obj, request)
}

/// Computes the block hashes of a Rust request like its Python request's
/// `block_hasher` does: in Rust when the hasher is known
/// (`get_request_block_hasher` or `RustRequestBlockHasher`) and the new
/// blocks need no multi-modal or prompt-embeds keys, else by calling the
/// Python hasher on the synced Python request.
struct SchedBlockHasher {
    native: Option<(Arc<HasherParams>, RequestExtraKeys)>,
    /// The Python hasher (or a `RustRequestBlockHasher`'s fallback).
    fallback: Option<Py<PyAny>>,
    request: Py<PyAny>,
    shared: Arc<Shared>,
}

static HASHING: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

/// The parameters of a `get_request_block_hasher` closure, read from its
/// cells, or `None` if `hasher` is something else.
fn closure_params(hasher: &Bound<'_, PyAny>) -> PyResult<Option<HasherParams>> {
    let py = hasher.py();
    let Ok(qualname) = hasher.getattr(intern!(py, "__qualname__")) else {
        return Ok(None);
    };
    if !qualname.eq("get_request_block_hasher.<locals>.request_block_hasher")? {
        return Ok(None);
    }
    let names: Vec<String> = hasher.getattr("__code__")?.getattr("co_freevars")?.extract()?;
    let cells = hasher.getattr("__closure__")?;
    let cell = |name: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        names
            .iter()
            .position(|n| n == name)
            .map(|i| cells.get_item(i)?.getattr("cell_contents"))
            .transpose()
    };
    let (Some(hash_block_size), Some(hash_fn)) =
        (cell("hash_block_size")?, cell("caching_hash_fn")?)
    else {
        return Ok(None);
    };
    let num_hash_lookahead_tokens = match cell("num_hash_lookahead_tokens")? {
        Some(n) => n.extract()?,
        None => 0,
    };
    let hashing = HASHING
        .get_or_try_init(py, || {
            PyModule::import(py, "vllm.utils.hashing").map(Bound::unbind)
        })?
        .bind(py);
    let mut algo: Option<HashAlgo> = None;
    for name in ["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"] {
        if hash_fn.is(&hashing.getattr(name)?) {
            // Without the xxhash package the Python hasher raises.
            if name.starts_with("xxhash") && hashing.getattr("_xxhash")?.is_none() {
                return Ok(None);
            }
            algo = Some(parse_algo(name)?);
        }
    }
    let Some(algo) = algo else {
        return Ok(None);
    };
    Ok(Some(HasherParams {
        algo,
        hash_block_size: hash_block_size.extract()?,
        num_hash_lookahead_tokens,
        none_hash: None,
    }))
}

impl SchedBlockHasher {
    fn new(
        shared: &Arc<Shared>,
        request: &Bound<'_, PyAny>,
        hasher: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let py = request.py();
        let (params, fallback) = if let Ok(rust) = hasher.cast::<RustRequestBlockHasher>() {
            let rust = rust.get();
            (
                Some(Arc::new(rust.params())),
                rust.fallback().map(|f| f.clone_ref(py)),
            )
        } else {
            let mut cache = shared.hasher_cache.lock().unwrap();
            let params = match &*cache {
                Some((cached, params)) if cached.is(hasher) => params.clone(),
                _ => {
                    let params = closure_params(hasher)?.map(Arc::new);
                    *cache = Some((hasher.clone().unbind(), params.clone()));
                    params
                }
            };
            (params, Some(hasher.clone().unbind()))
        };
        let native = match params {
            Some(params) => request_extra_keys(request)?.map(|keys| (params, keys)),
            None => None,
        };
        Ok(Self {
            native,
            fallback,
            request: request.clone().unbind(),
            shared: shared.clone(),
        })
    }

    /// Hash in Rust, or `None` if the Python hasher must run.
    fn hash_native(&self, request: &Request) -> Option<Vec<BlockHash>> {
        let (params, extra_keys) = self.native.as_ref()?;
        let bs = params.hash_block_size;
        let num_existing = request.block_hashes.len();
        let start = num_existing * bs;
        if start + bs + params.num_hash_lookahead_tokens > request.num_tokens() {
            // Early stop when there no new full blocks created.
            return Some(Vec::new());
        }
        // Multi-modal keys are pickled by object identity: leave them to
        // Python. Inputs ending before the first new block add no keys.
        if request
            .mm_features
            .last()
            .is_some_and(|f| f.mm_position.offset + f.mm_position.length > start)
        {
            return None;
        }
        let none_hash = match (&params.none_hash, num_existing) {
            (Some(h), _) => h.clone(),
            (None, 0) => {
                let none_hash = Python::attach(current_none_hash);
                self.shared.check(none_hash)?
            }
            (None, _) => BlockHash::from_slice(&[]),
        };
        let hasher =
            RequestBlockHasher::new(params.algo, none_hash, bs, params.num_hash_lookahead_tokens);
        Some(hasher.new_block_hashes(
            request.all_token_ids(),
            num_existing,
            request.block_hashes.last(),
            extra_keys,
        ))
    }

    fn hash_in_python(&self, request: &Request) -> Vec<BlockHash> {
        Python::attach(|py| {
            let result = (|| -> PyResult<Vec<BlockHash>> {
                let Some(fallback) = &self.fallback else {
                    return Err(PyNotImplementedError::new_err(
                        "request needs block-hash extra keys but no fallback hasher was given",
                    ));
                };
                let obj = self.request.bind(py);
                sync_tokens(obj, request)?;
                sync_block_hashes(obj, request)?;
                fallback
                    .bind(py)
                    .call1((obj,))?
                    .try_iter()?
                    .map(|h| Ok(BlockHash::from_slice(h?.cast::<PyBytes>()?.as_bytes())))
                    .collect()
            })();
            self.shared.check(result)
        })
    }
}

impl BlockHasher for SchedBlockHasher {
    fn hash_new_blocks(&self, request: &Request) -> Vec<BlockHash> {
        match self.hash_native(request) {
            Some(hashes) => hashes,
            None => self.hash_in_python(request),
        }
    }
}
