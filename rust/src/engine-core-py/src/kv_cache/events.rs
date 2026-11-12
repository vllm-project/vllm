// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Conversion of KV cache events and stats into their Python classes.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use vllm_engine_core::kv_cache::{EventExtraKey, ExternalBlockHash, KvCacheEvent};

/// `vllm.distributed.kv_events.{BlockStored, BlockRemoved, AllBlocksCleared}`.
static EVENT_CLASSES: PyOnceLock<[Py<PyAny>; 3]> = PyOnceLock::new();

fn external_hash<'py>(py: Python<'py>, hash: &ExternalBlockHash) -> PyResult<Bound<'py, PyAny>> {
    Ok(match hash {
        ExternalBlockHash::Bytes(h) => PyBytes::new(py, h.as_bytes()).into_any(),
        ExternalBlockHash::Int(h) => h.into_pyobject(py)?.into_any(),
    })
}

fn external_hashes<'py>(
    py: Python<'py>,
    hashes: &[ExternalBlockHash],
) -> PyResult<Bound<'py, PyList>> {
    PyList::new(
        py,
        hashes.iter().map(|h| external_hash(py, h)).collect::<PyResult<Vec<_>>>()?,
    )
}

fn extra_key<'py>(py: Python<'py>, key: &EventExtraKey) -> PyResult<Bound<'py, PyAny>> {
    Ok(match key {
        EventExtraKey::Str(s) => s.into_pyobject(py)?.into_any(),
        EventExtraKey::Bytes(b) => PyBytes::new(py, b).into_any(),
        EventExtraKey::Mm { identifier, offset } => {
            (identifier, offset).into_pyobject(py)?.into_any()
        }
    })
}

/// Build the Python events, in order.
pub fn events_to_py<'py>(
    py: Python<'py>,
    events: Vec<KvCacheEvent>,
) -> PyResult<Bound<'py, PyList>> {
    let [stored, removed, cleared] = EVENT_CLASSES.get_or_try_init(py, || -> PyResult<_> {
        let module = PyModule::import(py, "vllm.distributed.kv_events")?;
        Ok([
            module.getattr("BlockStored")?.unbind(),
            module.getattr("BlockRemoved")?.unbind(),
            module.getattr("AllBlocksCleared")?.unbind(),
        ])
    })?;
    let mut out = Vec::with_capacity(events.len());
    for event in events {
        let kwargs = PyDict::new(py);
        let obj = match event {
            KvCacheEvent::BlockStored(e) => {
                kwargs.set_item("block_hashes", external_hashes(py, &e.block_hashes)?)?;
                kwargs.set_item(
                    "parent_block_hash",
                    e.parent_block_hash.as_ref().map(|h| external_hash(py, h)).transpose()?,
                )?;
                kwargs.set_item("token_ids", PyList::new(py, &e.token_ids)?)?;
                kwargs.set_item("block_size", e.block_size)?;
                kwargs.set_item("lora_id", e.lora_id)?;
                kwargs.set_item("medium", e.medium)?;
                kwargs.set_item("lora_name", e.lora_name)?;
                let extra_keys = e
                    .extra_keys
                    .map(|blocks| {
                        let items = blocks
                            .iter()
                            .map(|keys| {
                                keys.as_ref()
                                    .map(|keys| {
                                        PyTuple::new(
                                            py,
                                            keys.iter()
                                                .map(|k| extra_key(py, k))
                                                .collect::<PyResult<Vec<_>>>()?,
                                        )
                                    })
                                    .transpose()
                            })
                            .collect::<PyResult<Vec<_>>>()?;
                        PyList::new(py, items)
                    })
                    .transpose()?;
                kwargs.set_item("extra_keys", extra_keys)?;
                kwargs.set_item("group_idx", e.group_idx)?;
                kwargs.set_item("kv_cache_spec_kind", e.kv_cache_spec_kind)?;
                kwargs.set_item(
                    "kv_cache_spec_sliding_window",
                    e.kv_cache_spec_sliding_window,
                )?;
                kwargs.set_item("session_id", e.session_id)?;
                kwargs.set_item("lookahead_token_ids", e.lookahead_token_ids)?;
                stored.bind(py).call((), Some(&kwargs))?
            }
            KvCacheEvent::BlockRemoved(e) => {
                kwargs.set_item("block_hashes", external_hashes(py, &e.block_hashes)?)?;
                kwargs.set_item("medium", e.medium)?;
                kwargs.set_item("group_idx", e.group_idx)?;
                removed.bind(py).call((), Some(&kwargs))?
            }
            KvCacheEvent::AllBlocksCleared(_) => cleared.bind(py).call0()?,
        };
        out.push(obj);
    }
    PyList::new(py, out)
}
