// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Conversion of the Python `KVCacheConfig` into the Rust scheduler-side config.

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyType};
use vllm_engine_core::kv_cache::{
    KvCacheConfig, KvCacheError, KvCacheGroupRole, KvCacheGroupSpec, KvCacheSpec, KvCacheSpecType,
    MambaCacheMode,
};

/// Map a [`KvCacheError`] to the Python exception the Python manager raises
/// for it (`NotImplementedError` for configurations this port lacks).
pub fn to_py_err(err: KvCacheError) -> PyErr {
    match err {
        KvCacheError::Unsupported { message } => PyNotImplementedError::new_err(message),
        KvCacheError::InvalidConfig { message } | KvCacheError::InvalidArgument { message } => {
            PyValueError::new_err(message)
        }
    }
}

fn unsupported(message: impl Into<String>) -> PyErr {
    PyNotImplementedError::new_err(message.into())
}

/// The spec classes of `vllm.v1.kv_cache_interface` the port knows, matched by
/// exact type as the manager registry does.
#[derive(Clone, Copy)]
enum SpecClass {
    FullAttention,
    MlaAttention,
    HiddenStateCache,
    Rswa,
    SinkFullAttention,
    SlidingWindow,
    SlidingWindowMla,
    ChunkedLocalAttention,
    CircularBuffer,
    Mamba,
    CrossAttention,
    EncoderOnlyAttention,
    HiSparseHot,
    HiSparseResident,
}

const SPEC_CLASSES: [(&str, SpecClass); 14] = [
    ("FullAttentionSpec", SpecClass::FullAttention),
    ("MLAAttentionSpec", SpecClass::MlaAttention),
    ("HiddenStateCacheSpec", SpecClass::HiddenStateCache),
    ("RSWASpec", SpecClass::Rswa),
    ("SinkFullAttentionSpec", SpecClass::SinkFullAttention),
    ("SlidingWindowSpec", SpecClass::SlidingWindow),
    ("SlidingWindowMLASpec", SpecClass::SlidingWindowMla),
    (
        "ChunkedLocalAttentionSpec",
        SpecClass::ChunkedLocalAttention,
    ),
    ("CircularBufferSpec", SpecClass::CircularBuffer),
    ("MambaSpec", SpecClass::Mamba),
    ("CrossAttentionSpec", SpecClass::CrossAttention),
    ("EncoderOnlyAttentionSpec", SpecClass::EncoderOnlyAttention),
    ("HiSparseHotSpec", SpecClass::HiSparseHot),
    ("HiSparseResidentSpec", SpecClass::HiSparseResident),
];

static SPEC_TYPES: PyOnceLock<Vec<(Py<PyType>, SpecClass)>> = PyOnceLock::new();
static UNIFORM_TYPE_SPECS: PyOnceLock<Py<PyType>> = PyOnceLock::new();

fn spec_class(spec: &Bound<'_, PyAny>) -> PyResult<SpecClass> {
    let py = spec.py();
    let types = SPEC_TYPES.get_or_try_init(py, || {
        let module = PyModule::import(py, "vllm.v1.kv_cache_interface")?;
        SPEC_CLASSES
            .iter()
            .map(|&(name, class)| {
                Ok((module.getattr(name)?.cast_into::<PyType>()?.unbind(), class))
            })
            .collect::<PyResult<Vec<_>>>()
    })?;
    let ty = spec.get_type();
    types
        .iter()
        .find(|(t, _)| ty.is(t.bind(py)))
        .map(|&(_, class)| class)
        .ok_or_else(|| {
            unsupported(format!(
                "KV cache spec {} is not supported",
                ty.name().map_or_else(|_| "?".into(), |n| n.to_string())
            ))
        })
}

/// `generate_scheduler_kv_cache_config`: a `UniformTypeKVCacheSpecs` group is
/// represented by its first layer spec.
pub fn scheduler_spec<'py>(spec: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = spec.py();
    let uniform =
        UNIFORM_TYPE_SPECS.import(py, "vllm.v1.kv_cache_interface", "UniformTypeKVCacheSpecs")?;
    if !spec.is_instance(uniform)? {
        return Ok(spec);
    }
    let specs = spec.getattr("kv_cache_specs")?.cast_into::<PyDict>()?;
    specs
        .values()
        .iter()
        .next()
        .ok_or_else(|| PyValueError::new_err("empty UniformTypeKVCacheSpecs"))
}

pub(crate) fn convert_spec(spec: &Bound<'_, PyAny>, fingerprint: u64) -> PyResult<KvCacheSpec> {
    let get_usize = |name: &str| -> PyResult<usize> { spec.getattr(name)?.extract() };
    let spec_type = match spec_class(spec)? {
        SpecClass::FullAttention => KvCacheSpecType::FullAttention,
        SpecClass::MlaAttention => KvCacheSpecType::MlaAttention,
        SpecClass::HiddenStateCache => KvCacheSpecType::HiddenStateCache,
        SpecClass::Rswa => KvCacheSpecType::Rswa {
            rswa_window: get_usize("rswa_window")?,
        },
        SpecClass::SinkFullAttention => KvCacheSpecType::SinkFullAttention {
            sink_len: spec.getattr("sink_len")?.extract()?,
        },
        SpecClass::SlidingWindow => KvCacheSpecType::SlidingWindow {
            sliding_window: get_usize("sliding_window")?,
            extra_retained_tokens: get_usize("extra_retained_tokens")?,
        },
        SpecClass::SlidingWindowMla => KvCacheSpecType::SlidingWindowMla {
            sliding_window: get_usize("sliding_window")?,
            extra_retained_tokens: get_usize("extra_retained_tokens")?,
            bounded_replay: spec.getattr("bounded_replay")?.extract()?,
        },
        SpecClass::ChunkedLocalAttention => KvCacheSpecType::ChunkedLocalAttention {
            attention_chunk_size: get_usize("attention_chunk_size")?,
        },
        SpecClass::CircularBuffer => KvCacheSpecType::CircularBuffer,
        SpecClass::Mamba => {
            let mode: String = spec.getattr("mamba_cache_mode")?.extract()?;
            KvCacheSpecType::Mamba {
                mamba_cache_mode: match mode.as_str() {
                    "none" => MambaCacheMode::None,
                    "all" => MambaCacheMode::All,
                    "align" => MambaCacheMode::Align,
                    _ => return Err(unsupported(format!("mamba_cache_mode {mode:?}"))),
                },
                num_speculative_blocks: get_usize("num_speculative_blocks")?,
                num_prefill_checkpoint_blocks: get_usize("num_prefill_checkpoint_blocks")?,
                prefill_checkpoint_alignment: spec
                    .getattr("prefill_checkpoint_alignment")?
                    .extract()?,
            }
        }
        SpecClass::CrossAttention => KvCacheSpecType::CrossAttention,
        SpecClass::EncoderOnlyAttention => KvCacheSpecType::EncoderOnlyAttention,
        SpecClass::HiSparseHot => KvCacheSpecType::HiSparseHot {
            blocks_per_request: get_usize("blocks_per_request")?,
        },
        SpecClass::HiSparseResident => KvCacheSpecType::HiSparseResident,
    };
    let converted = KvCacheSpec {
        block_size: get_usize("block_size")?,
        dcp_sharded: spec.getattr("dcp_sharded")?.extract()?,
        spec_type,
        fingerprint,
    };
    // The port derives these from the spec type; a subclass overriding them
    // would diverge silently.
    let prefix_cacheable: bool = spec.getattr("prefix_cacheable")?.extract()?;
    let prefix_replay_tokens: usize = get_usize("prefix_replay_tokens")?;
    if prefix_cacheable != converted.prefix_cacheable()
        || prefix_replay_tokens != converted.prefix_replay_tokens()
    {
        return Err(unsupported(format!(
            "KV cache spec {spec} has unexpected prefix caching properties"
        )));
    }
    Ok(converted)
}

/// Convert the scheduler's `KVCacheConfig`.
///
/// Returns the Rust config and the (flattened) Python spec of each group.
pub fn convert_config<'py>(
    config: &Bound<'py, PyAny>,
) -> PyResult<(KvCacheConfig, Vec<Bound<'py, PyAny>>)> {
    let groups = config.getattr("kv_cache_groups")?;
    let mut py_specs: Vec<Bound<'py, PyAny>> = Vec::new();
    let mut rust_groups = Vec::new();
    for group in groups.try_iter()? {
        let group = group?;
        let spec = scheduler_spec(group.getattr("kv_cache_spec")?)?;
        // Python batches groups by dataclass equality of their specs; the
        // fingerprint is the index of the first equal spec.
        let mut fingerprint = py_specs.len() as u64;
        for (i, other) in py_specs.iter().enumerate() {
            if other.eq(&spec)? {
                fingerprint = i as u64;
                break;
            }
        }
        let kv_cache_spec = convert_spec(&spec, fingerprint)?;
        let role: String = group.getattr("role")?.getattr("value")?.extract()?;
        let role = match role.as_str() {
            "default" => KvCacheGroupRole::Default,
            "hisparse_source" => KvCacheGroupRole::HiSparseSource,
            "hisparse_indexer" => KvCacheGroupRole::HiSparseIndexer,
            _ => return Err(unsupported(format!("KV cache group role {role:?}"))),
        };
        rust_groups.push(KvCacheGroupSpec {
            kv_cache_spec,
            is_eagle_group: group.getattr("is_eagle_group")?.extract()?,
            host_resident: group.getattr("host_resident")?.extract()?,
            enable_kv_transfer: group.getattr("enable_kv_transfer")?.extract()?,
            role,
        });
        py_specs.push(spec);
    }
    let mut rust_config = KvCacheConfig::new(config.getattr("num_blocks")?.extract()?, rust_groups);
    rust_config.prefix_cache_retention_interval =
        config.getattr("prefix_cache_retention_interval")?.extract()?;
    rust_config.needs_kv_cache_zeroing = config.getattr("needs_kv_cache_zeroing")?.extract()?;
    rust_config.hisparse_host_num_blocks = config.getattr("hisparse_host_num_blocks")?.extract()?;
    Ok((rust_config, py_specs))
}
