// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! [`KvRequest`] over a Python `vllm.v1.request.Request`.

use std::cell::OnceCell;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyList, PyModule, PyString, PyTuple, PyType};
use rustc_hash::FxHashMap;
use vllm_engine_core::hash::BlockHash;
use vllm_engine_core::kv_cache::{EventExtraKey, EventExtraKeys, KvRequest};

static ARRAY_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static REQUEST_STATUS: PyOnceLock<(Py<PyAny>, Py<PyAny>)> = PyOnceLock::new();
static KV_CACHE_UTILS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

/// Converted `request.block_hashes` of the live requests, kept in sync
/// incrementally so that long prompts are converted once.
#[derive(Default)]
pub struct BlockHashCache {
    by_request: FxHashMap<String, Vec<BlockHash>>,
}

fn hash_bytes<'a>(item: &'a Bound<'_, PyAny>) -> PyResult<&'a [u8]> {
    Ok(item
        .cast::<PyBytes>()
        .map_err(|_| PyTypeError::new_err("block hashes must be bytes"))?
        .as_bytes())
}

impl BlockHashCache {
    /// Sync and return the converted hashes of `request_id` with the Python
    /// list `block_hashes`.
    ///
    /// The list is append-only except where the scheduler truncates it
    /// (preemption, streaming input) and possibly re-extends it with other
    /// hashes. Hashes chain over their prefix, so the converted prefix is
    /// valid up to the last position whose hash still matches.
    pub fn sync(
        &mut self,
        request_id: &str,
        block_hashes: &Bound<'_, PyAny>,
    ) -> PyResult<&[BlockHash]> {
        if !self.by_request.contains_key(request_id) {
            self.by_request.insert(request_id.to_owned(), Vec::new());
        }
        let cached = self.by_request.get_mut(request_id).expect("just inserted");
        let list = block_hashes.cast::<PyList>();
        let get = |i: usize| -> PyResult<Bound<'_, PyAny>> {
            match list {
                Ok(list) => list.get_item(i),
                Err(_) => block_hashes.get_item(i),
            }
        };
        let n = block_hashes.len()?;
        let common = n.min(cached.len());
        cached.truncate(common);
        if common > 0 && hash_bytes(&get(common - 1)?)? != cached[common - 1].as_bytes() {
            // Replaced: find the first mismatch (matches form a prefix).
            let (mut lo, mut hi) = (0, common - 1);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if hash_bytes(&get(mid)?)? == cached[mid].as_bytes() {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            cached.truncate(lo);
        }
        cached.reserve(n - cached.len());
        for i in cached.len()..n {
            cached.push(BlockHash::from_slice(hash_bytes(&get(i)?)?));
        }
        Ok(cached)
    }

    pub fn remove(&mut self, request_id: &str) {
        self.by_request.remove(request_id);
    }
}

/// Convert a Python sequence of block hashes (no caching).
pub fn convert_block_hashes(block_hashes: &Bound<'_, PyAny>) -> PyResult<Vec<BlockHash>> {
    block_hashes
        .try_iter()?
        .map(|item| Ok(BlockHash::from_slice(hash_bytes(&item?)?)))
        .collect()
}

/// A Python request seen through [`KvRequest`].
///
/// Scalar fields are read on first use. The KV cache manager never runs
/// Python code that mutates the request, so values stay valid for the call.
pub struct PyRequest<'a, 'py> {
    obj: &'a Bound<'py, PyAny>,
    request_id: &'a str,
    block_hashes: &'a [BlockHash],
    num_tokens: OnceCell<usize>,
    num_prompt_tokens: OnceCell<usize>,
    num_computed_tokens: OnceCell<usize>,
    num_in_flight_tokens: OnceCell<usize>,
    shared_prefix_boundary: OnceCell<usize>,
    mm_ranges: OnceCell<Vec<(usize, usize)>>,
    has_extra_keys: OnceCell<bool>,
}

/// Read a Python attribute; the trait accessors are infallible, so a request
/// without the attribute is a programming error.
fn attr<'py, T: for<'a> FromPyObject<'a, 'py>>(
    obj: &Bound<'py, PyAny>,
    name: &Bound<'py, PyString>,
) -> T {
    obj.getattr(name)
        .and_then(|v| v.extract::<T>().map_err(Into::into))
        .unwrap_or_else(|e| panic!("cannot read request.{name}: {e}"))
}

pub fn request_id<'py>(request: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyString>> {
    Ok(request.getattr(intern!(request.py(), "request_id"))?.cast_into::<PyString>()?)
}

impl<'a, 'py> PyRequest<'a, 'py> {
    pub fn new(
        obj: &'a Bound<'py, PyAny>,
        request_id: &'a str,
        block_hashes: &'a [BlockHash],
    ) -> Self {
        Self {
            obj,
            request_id,
            block_hashes,
            num_tokens: OnceCell::new(),
            num_prompt_tokens: OnceCell::new(),
            num_computed_tokens: OnceCell::new(),
            num_in_flight_tokens: OnceCell::new(),
            shared_prefix_boundary: OnceCell::new(),
            mm_ranges: OnceCell::new(),
            has_extra_keys: OnceCell::new(),
        }
    }

    fn py(&self) -> Python<'py> {
        self.obj.py()
    }

    fn get(&self, cell: &OnceCell<usize>, name: &Bound<'py, PyString>) -> usize {
        *cell.get_or_init(|| attr(self.obj, name))
    }

    /// `request.lora_request`, if truthy.
    fn lora_request(&self) -> Option<Bound<'py, PyAny>> {
        let lora = self
            .obj
            .getattr(intern!(self.py(), "lora_request"))
            .expect("request.lora_request");
        lora.is_truthy().expect("bool(request.lora_request)").then_some(lora)
    }

    /// Whether `generate_block_hash_extra_keys` can return anything.
    fn has_extra_keys(&self) -> bool {
        *self.has_extra_keys.get_or_init(|| {
            let py = self.py();
            let truthy = |name: &Bound<'py, PyString>| {
                self.obj.getattr(name).and_then(|v| v.is_truthy()).unwrap_or(false)
            };
            truthy(intern!(py, "lora_request"))
                || truthy(intern!(py, "mm_features"))
                || truthy(intern!(py, "cache_salt"))
                || !self
                    .obj
                    .getattr(intern!(py, "prompt_embeds"))
                    .map(|v| v.is_none())
                    .unwrap_or(true)
        })
    }

    fn mm_ranges(&self) -> &[(usize, usize)] {
        self.mm_ranges.get_or_init(|| {
            let py = self.py();
            let features =
                self.obj.getattr(intern!(py, "mm_features")).expect("request.mm_features");
            if !features.is_truthy().unwrap_or(false) {
                return Vec::new();
            }
            features
                .try_iter()
                .expect("iter(request.mm_features)")
                .map(|f| {
                    let pos =
                        f.and_then(|f| f.getattr(intern!(py, "mm_position"))).expect("mm_position");
                    (
                        attr(&pos, intern!(py, "offset")),
                        attr(&pos, intern!(py, "length")),
                    )
                })
                .collect()
        })
    }
}

fn convert_event_key(key: &Bound<'_, PyAny>) -> PyResult<EventExtraKey> {
    if let Ok(s) = key.cast::<PyString>() {
        Ok(EventExtraKey::Str(s.to_str()?.to_owned()))
    } else if let Ok(b) = key.cast::<PyBytes>() {
        Ok(EventExtraKey::Bytes(b.as_bytes().to_vec()))
    } else if let Ok(t) = key.cast::<PyTuple>()
        && t.len() == 2
    {
        Ok(EventExtraKey::Mm {
            identifier: t.get_item(0)?.extract()?,
            offset: t.get_item(1)?.extract()?,
        })
    } else {
        Err(PyTypeError::new_err(format!(
            "unsupported event extra key: {key}"
        )))
    }
}

impl KvRequest for PyRequest<'_, '_> {
    fn request_id(&self) -> &str {
        self.request_id
    }

    fn block_hashes(&self) -> &[BlockHash] {
        self.block_hashes
    }

    fn num_tokens(&self) -> usize {
        self.get(&self.num_tokens, intern!(self.py(), "num_tokens"))
    }

    fn num_prompt_tokens(&self) -> usize {
        self.get(
            &self.num_prompt_tokens,
            intern!(self.py(), "num_prompt_tokens"),
        )
    }

    fn num_computed_tokens(&self) -> usize {
        self.get(
            &self.num_computed_tokens,
            intern!(self.py(), "num_computed_tokens"),
        )
    }

    fn num_in_flight_tokens(&self) -> usize {
        self.get(
            &self.num_in_flight_tokens,
            intern!(self.py(), "num_in_flight_tokens"),
        )
    }

    fn num_preemptions(&self) -> usize {
        attr(self.obj, intern!(self.py(), "num_preemptions"))
    }

    fn shared_prefix_boundary(&self) -> usize {
        self.get(
            &self.shared_prefix_boundary,
            intern!(self.py(), "shared_prefix_boundary"),
        )
    }

    fn prefix_cache_hit_limit(&self) -> Option<usize> {
        attr(self.obj, intern!(self.py(), "prefix_cache_hit_limit"))
    }

    fn is_waiting_or_preempted(&self) -> bool {
        let py = self.py();
        let (waiting, preempted) = REQUEST_STATUS
            .get_or_try_init(py, || -> PyResult<_> {
                let status = PyModule::import(py, "vllm.v1.request")?.getattr("RequestStatus")?;
                Ok((
                    status.getattr("WAITING")?.unbind(),
                    status.getattr("PREEMPTED")?.unbind(),
                ))
            })
            .expect("vllm.v1.request.RequestStatus");
        let status = self.obj.getattr(intern!(py, "status")).expect("request.status");
        // Enum members are singletons.
        status.is(waiting.bind(py)) || status.is(preempted.bind(py))
    }

    fn kv_cache_report_mode_full(&self) -> bool {
        self.obj
            .getattr(intern!(self.py(), "kv_cache_report_mode"))
            .and_then(|mode| mode.eq("full"))
            .unwrap_or(false)
    }

    fn token_ids(&self, start: usize, end: usize) -> Vec<i32> {
        let py = self.py();
        let tokens =
            self.obj.getattr(intern!(py, "_all_token_ids")).expect("request._all_token_ids");
        let array_type = ARRAY_TYPE.import(py, "array", "array").expect("array.array");
        if tokens.is_instance(array_type).unwrap_or(false)
            && tokens.getattr("typecode").and_then(|t| t.eq("i")).unwrap_or(false)
            && tokens.getattr("itemsize").and_then(|s| s.extract::<usize>()).ok() == Some(4)
        {
            let (address, len): (usize, usize) = tokens
                .call_method0("buffer_info")
                .and_then(|b| b.extract())
                .expect("buffer_info");
            let end = end.min(len);
            let start = start.min(end);
            if start == end {
                return Vec::new();
            }
            // SAFETY: `buffer_info` is the address and length of the array's
            // int32 storage; no Python code runs while it is copied.
            let all = unsafe { std::slice::from_raw_parts(address as *const i32, len) };
            return all[start..end].to_vec();
        }
        let all: Vec<i32> = tokens.extract().expect("request._all_token_ids as int list");
        let end = end.min(all.len());
        all[start.min(end)..end].to_vec()
    }

    fn lora_id(&self) -> Option<i64> {
        self.lora_request().map(|lora| attr(&lora, intern!(self.py(), "adapter_id")))
    }

    fn lora_name(&self) -> Option<String> {
        self.lora_request().map(|lora| attr(&lora, intern!(self.py(), "name")))
    }

    fn session_id(&self) -> Option<String> {
        self.obj
            .getattr(intern!(self.py(), "session_id"))
            .ok()
            .and_then(|s| s.extract().ok())
    }

    fn event_extra_keys(
        &self,
        start: usize,
        end: usize,
        start_mm_idx: isize,
    ) -> (Option<EventExtraKeys>, isize) {
        if !self.has_extra_keys() {
            return (None, start_mm_idx);
        }
        let py = self.py();
        let result = (|| -> PyResult<_> {
            let utils = KV_CACHE_UTILS
                .get_or_try_init(py, || {
                    PyModule::import(py, "vllm.v1.core.kv_cache_utils").map(Bound::unbind)
                })?
                .bind(py);
            let (keys, next_mm_idx): (Bound<'_, PyAny>, isize) = utils
                .getattr("generate_block_hash_extra_keys")?
                .call1((self.obj, start, end, start_mm_idx))?
                .extract()?;
            if keys.is_none() {
                return Ok((None, next_mm_idx));
            }
            let event_keys = utils
                .getattr("to_event_extra_keys")?
                .call1((PyList::new(py, [keys])?,))?
                .get_item(0)?;
            if event_keys.is_none() {
                return Ok((None, next_mm_idx));
            }
            let converted = event_keys
                .try_iter()?
                .map(|key| convert_event_key(&key?))
                .collect::<PyResult<Vec<_>>>()?;
            Ok((Some(converted), next_mm_idx))
        })();
        result.unwrap_or_else(|e| panic!("cannot generate KV event extra keys: {e}"))
    }

    fn num_mm_features(&self) -> usize {
        self.mm_ranges().len()
    }

    fn mm_feature_range(&self, idx: usize) -> (usize, usize) {
        self.mm_ranges()[idx]
    }
}
