// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Bindings for prefix-caching block hashing (`vllm_engine_core::hash`).

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyBytes, PyInt, PyList, PyString, PyTuple, PyType};
use vllm_engine_core::hash::{
    BlockHash, ExtraKeys, HashAlgo, RequestBlockHasher, RequestExtraKeys, Value,
};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_value, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_value, m)?)?;
    m.add_function(wrap_pyfunction!(hash_block_tokens, m)?)?;
    m.add_class::<RustRequestBlockHasher>()?;
    Ok(())
}

pub(crate) fn parse_algo(algo_name: &str) -> PyResult<HashAlgo> {
    HashAlgo::from_name(algo_name)
        .ok_or_else(|| PyValueError::new_err(format!("Unsupported hash function: {algo_name}")))
}

/// Convert a Python object into a [`Value`], recording the identity of
/// memoizable objects so that pickle's memo references are reproduced.
fn to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        Ok(Value::None)
    } else if obj.is_exact_instance_of::<PyBool>() {
        Ok(Value::Bool(obj.is_truthy()?))
    } else if obj.is_exact_instance_of::<PyInt>() {
        Ok(Value::Int(obj.extract()?))
    } else {
        let inner = if let Ok(b) = obj.cast_exact::<PyBytes>() {
            Value::Bytes(b.as_bytes().to_vec())
        } else if let Ok(s) = obj.cast_exact::<PyString>() {
            Value::Str(s.to_str()?.to_owned())
        } else if let Ok(t) = obj.cast_exact::<PyTuple>() {
            Value::Tuple(t.iter().map(|item| to_value(&item)).collect::<PyResult<_>>()?)
        } else {
            return Err(PyTypeError::new_err(format!(
                "unsupported type for block hashing: {}",
                obj.get_type().name()?
            )));
        };
        Ok(Value::Ident(obj.as_ptr() as usize, Box::new(inner)))
    }
}

/// Hash `obj` like the `vllm.utils.hashing` function named `algo_name`.
///
/// Supports `None`, `bool`, `int` (within 128 bits), `bytes`, `str` (without
/// lone surrogates) and nested `tuple`s of them; subclasses are rejected.
#[pyfunction]
fn hash_value<'py>(
    py: Python<'py>,
    algo_name: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let algo = parse_algo(algo_name)?;
    let value = to_value(obj)?;
    Ok(PyBytes::new(py, algo.hash_value(&value).as_bytes()))
}

/// The bytes that `hash_value` digests: the pickle or canonical CBOR
/// serialization of `obj`.
#[pyfunction]
fn serialize_value<'py>(
    py: Python<'py>,
    algo_name: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let algo = parse_algo(algo_name)?;
    let value = to_value(obj)?;
    Ok(PyBytes::new(py, &algo.serialize_value(&value)))
}

/// `kv_cache_utils.hash_block_tokens` with the hash function named
/// `algo_name` and `NONE_HASH` given as `none_hash`.
///
/// `extra_keys` must not contain `parent_block_hash` itself, whose identity
/// isn't tracked.
#[pyfunction]
#[pyo3(signature = (algo_name, parent_block_hash, curr_block_token_ids, extra_keys=None, *, none_hash))]
fn hash_block_tokens<'py>(
    py: Python<'py>,
    algo_name: &str,
    parent_block_hash: Option<&[u8]>,
    curr_block_token_ids: Vec<i32>,
    extra_keys: Option<&Bound<'py, PyAny>>,
    none_hash: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let algo = parse_algo(algo_name)?;
    let parent = parent_block_hash.map(BlockHash::from_slice);
    let extra_keys = extra_keys.map(to_value).transpose()?;
    let hash = vllm_engine_core::hash::hash_block_tokens(
        algo,
        parent.as_ref(),
        &BlockHash::from_slice(none_hash),
        &curr_block_token_ids,
        ExtraKeys::Plain(extra_keys.as_ref()),
    );
    Ok(PyBytes::new(py, hash.as_bytes()))
}

/// Drop-in replacement for the closure returned by
/// `kv_cache_utils.get_request_block_hasher`.
///
/// Requests whose new blocks need multi-modal or prompt-embeds extra keys, or
/// whose state can't be reproduced exactly, are delegated to `fallback`.
#[pyclass(frozen, module = "vllm._rust_core")]
pub(crate) struct RustRequestBlockHasher {
    algo: HashAlgo,
    hash_block_size: usize,
    num_hash_lookahead_tokens: usize,
    /// `None` reads `kv_cache_utils.NONE_HASH` on each call, since it may be
    /// (re-)initialized after the hasher is created.
    none_hash: Option<BlockHash>,
    fallback: Option<Py<PyAny>>,
}

static ARRAY_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static KV_CACHE_UTILS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

/// How a block hasher hashes, for callers that hash in Rust directly.
pub(crate) struct HasherParams {
    pub algo: HashAlgo,
    pub hash_block_size: usize,
    pub num_hash_lookahead_tokens: usize,
    /// `None` reads `kv_cache_utils.NONE_HASH` when needed.
    pub none_hash: Option<BlockHash>,
}

/// `kv_cache_utils.NONE_HASH`, or `None` if it is not initialized.
pub(crate) fn current_none_hash(py: Python<'_>) -> PyResult<Option<BlockHash>> {
    let module = KV_CACHE_UTILS.get_or_try_init(py, || {
        PyModule::import(py, "vllm.v1.core.kv_cache_utils").map(Bound::unbind)
    })?;
    let Ok(none_hash) = module.bind(py).getattr("NONE_HASH") else {
        return Ok(None);
    };
    Ok(none_hash
        .cast_exact::<PyBytes>()
        .ok()
        .map(|b| BlockHash::from_slice(b.as_bytes())))
}

/// The request-level extra keys of `request` that the Rust hasher supports
/// (LoRA and cache salt), or `None` if its blocks must be hashed in Python:
/// prompt embeds, or keys that can't be pickled exactly. Multi-modal keys are
/// checked per block by the caller.
pub(crate) fn request_extra_keys(request: &Bound<'_, PyAny>) -> PyResult<Option<RequestExtraKeys>> {
    if !request.getattr("prompt_embeds")?.is_none() {
        return Ok(None);
    }
    let mut extra_keys = RequestExtraKeys::default();
    let lora_request = request.getattr("lora_request")?;
    if lora_request.is_truthy()? {
        let name = key_str(&lora_request.getattr("lora_name")?)?;
        let path = key_str(&lora_request.getattr("lora_path")?)?;
        let (Some(name), Some(path)) = (name, path) else {
            return Ok(None);
        };
        extra_keys.lora = Some((name, path));
    }
    let salt = request.getattr("cache_salt")?;
    if salt.is_truthy()? {
        let Some(salt) = key_str(&salt)? else {
            return Ok(None);
        };
        extra_keys.cache_salt = Some(salt);
    }
    Ok(Some(extra_keys))
}

/// What a call reads from the request for the Rust path.
struct Snapshot {
    num_existing: usize,
    last_hash: Option<BlockHash>,
    extra_keys: RequestExtraKeys,
}

#[pymethods]
impl RustRequestBlockHasher {
    #[new]
    #[pyo3(signature = (hash_block_size, algo_name, none_hash=None, num_hash_lookahead_tokens=0, fallback=None))]
    fn new(
        hash_block_size: usize,
        algo_name: &str,
        none_hash: Option<&[u8]>,
        num_hash_lookahead_tokens: usize,
        fallback: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if hash_block_size == 0 {
            return Err(PyValueError::new_err("hash_block_size must be positive"));
        }
        Ok(Self {
            algo: parse_algo(algo_name)?,
            hash_block_size,
            num_hash_lookahead_tokens,
            none_hash: none_hash.map(BlockHash::from_slice),
            fallback,
        })
    }

    fn __call__<'py>(&self, request: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
        let py = request.py();
        let block_hashes = request.getattr("block_hashes")?;
        let num_existing = block_hashes.len()?;
        let start_token_idx = num_existing * self.hash_block_size;
        let num_tokens: usize = request.getattr("num_tokens")?.extract()?;
        if start_token_idx + self.hash_block_size + self.num_hash_lookahead_tokens > num_tokens {
            // Early stop when there no new full blocks created.
            return Ok(PyList::empty(py));
        }
        if let Some(hashes) = self.hash_new_blocks(request, &block_hashes, num_tokens)? {
            let hashes = hashes.iter().map(|h| PyBytes::new(py, h.as_bytes()));
            return PyList::new(py, hashes);
        }
        match &self.fallback {
            Some(fallback) => Ok(fallback.bind(py).call1((request,))?.cast_into()?),
            None => Err(PyValueError::new_err(
                "request needs block-hash extra keys but no fallback hasher was given",
            )),
        }
    }
}

impl RustRequestBlockHasher {
    pub(crate) fn params(&self) -> HasherParams {
        HasherParams {
            algo: self.algo,
            hash_block_size: self.hash_block_size,
            num_hash_lookahead_tokens: self.num_hash_lookahead_tokens,
            none_hash: self.none_hash.clone(),
        }
    }

    pub(crate) fn fallback(&self) -> Option<&Py<PyAny>> {
        self.fallback.as_ref()
    }

    /// The new block hashes, or `None` if the request must be delegated.
    fn hash_new_blocks(
        &self,
        request: &Bound<'_, PyAny>,
        block_hashes: &Bound<'_, PyAny>,
        num_tokens: usize,
    ) -> PyResult<Option<Vec<BlockHash>>> {
        let Some(snapshot) = self.snapshot(request, block_hashes)? else {
            return Ok(None);
        };
        let needs_none_hash = snapshot.last_hash.as_ref().is_none_or(|h| h.0.is_empty());
        let none_hash = match (&self.none_hash, needs_none_hash) {
            (Some(h), _) => h.clone(),
            (None, false) => BlockHash::from_slice(&[]),
            (None, true) => {
                let py = request.py();
                let module = KV_CACHE_UTILS.get_or_try_init(py, || {
                    PyModule::import(py, "vllm.v1.core.kv_cache_utils").map(Bound::unbind)
                })?;
                // Not initialized yet: let the fallback raise the same error.
                let Ok(none_hash) = module.bind(py).getattr("NONE_HASH") else {
                    return Ok(None);
                };
                match none_hash.cast_exact::<PyBytes>() {
                    Ok(b) => BlockHash::from_slice(b.as_bytes()),
                    Err(_) => return Ok(None),
                }
            }
        };
        let hasher = RequestBlockHasher::new(
            self.algo,
            none_hash,
            self.hash_block_size,
            self.num_hash_lookahead_tokens,
        );
        let hash = |tokens: &[i32]| {
            hasher.new_block_hashes(
                tokens,
                snapshot.num_existing,
                snapshot.last_hash.as_ref(),
                &snapshot.extra_keys,
            )
        };

        let all_token_ids = request.getattr("_all_token_ids")?;
        let array_type = ARRAY_TYPE.import(request.py(), "array", "array")?;
        if all_token_ids.is_instance(array_type)?
            && all_token_ids.getattr("typecode")?.extract::<String>()? == "i"
            && all_token_ids.getattr("itemsize")?.extract::<usize>()? == size_of::<i32>()
        {
            let (address, len): (usize, usize) =
                all_token_ids.call_method0("buffer_info")?.extract()?;
            if len < num_tokens {
                return Ok(None);
            }
            let tokens = if num_tokens == 0 {
                &[][..]
            } else {
                // SAFETY: `buffer_info` is the address and length of the
                // array's int32 storage. We hold the GIL and run no Python
                // code until hashing is done, so it is neither resized nor
                // freed meanwhile.
                unsafe { std::slice::from_raw_parts(address as *const i32, num_tokens) }
            };
            return Ok(Some(hash(tokens)));
        }
        // Not an int32 array: read the token ids as a sequence; ids outside
        // the int32 range go to the fallback.
        let Ok(tokens) = all_token_ids.extract::<Vec<i32>>() else {
            return Ok(None);
        };
        if tokens.len() < num_tokens {
            return Ok(None);
        }
        Ok(Some(hash(&tokens[..num_tokens])))
    }

    /// Read the existing hashes and the extra keys of the new blocks,
    /// mirroring `generate_block_hash_extra_keys`.
    fn snapshot(
        &self,
        request: &Bound<'_, PyAny>,
        block_hashes: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Snapshot>> {
        let num_existing = block_hashes.len()?;
        let start_token_idx = num_existing * self.hash_block_size;

        if !request.getattr("prompt_embeds")?.is_none() {
            return Ok(None);
        }
        let mut extra_keys = RequestExtraKeys::default();
        let lora_request = request.getattr("lora_request")?;
        if lora_request.is_truthy()? {
            let name = key_str(&lora_request.getattr("lora_name")?)?;
            let path = key_str(&lora_request.getattr("lora_path")?)?;
            let (Some(name), Some(path)) = (name, path) else {
                return Ok(None);
            };
            extra_keys.lora = Some((name, path));
        }
        // `_gen_mm_extra_hash_keys` yields nothing once the last input ends
        // at or before the first new block.
        let mm_features = request.getattr("mm_features")?;
        if mm_features.is_truthy()? {
            let last_pos = mm_features.get_item(-1)?.getattr("mm_position")?;
            let offset: usize = last_pos.getattr("offset")?.extract()?;
            let length: usize = last_pos.getattr("length")?.extract()?;
            if offset + length > start_token_idx {
                return Ok(None);
            }
        }
        // The salt only keys the first block.
        if start_token_idx == 0 {
            let salt = request.getattr("cache_salt")?;
            if salt.is_truthy()? {
                let Some(salt) = key_str(&salt)? else {
                    return Ok(None);
                };
                extra_keys.cache_salt = Some(salt);
            }
        }

        let last_hash = if num_existing == 0 {
            None
        } else {
            match block_hashes.get_item(num_existing - 1)?.cast_exact::<PyBytes>() {
                Ok(b) => Some(BlockHash::from_slice(b.as_bytes())),
                Err(_) => return Ok(None),
            }
        };
        Ok(Some(Snapshot {
            num_existing,
            last_hash,
            extra_keys,
        }))
    }
}

/// A string in [`RequestExtraKeys`], with its identity, or `None` if it
/// can't be pickled exactly: a `str` subclass, lone surrogates, or possibly
/// the very object of the `"lora"` or `"cache_salt"` key literal.
fn key_str(obj: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    let Ok(s) = obj.cast_exact::<PyString>() else {
        return Ok(None);
    };
    Ok(match s.to_str() {
        Ok(s) if s != "lora" && s != "cache_salt" => {
            Some(Value::Ident(obj.as_ptr() as usize, Box::new(Value::str(s))))
        }
        _ => None,
    })
}
