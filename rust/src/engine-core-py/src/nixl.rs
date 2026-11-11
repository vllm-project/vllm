// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Bindings for the native NIXL scheduler half
//! (`vllm_engine_core_connectors::nixl`).
//!
//! [`RustNixlConnectorScheduler`] is built from a constructed Python
//! `NixlPullConnector` / `NixlPushConnector` (scheduler role), whose resolved
//! settings it copies; it raises `NotImplementedError` for connectors it
//! does not port, in which case the caller keeps the Python connector. The
//! Python connector stays alive for the handshake side channel and
//! shutdown.
//!
//! How the Rust scheduler (`RustSchedulerCore`, given one by
//! `rust_scheduler._native_connector`) uses it:
//! - [`RustNixlConnectorScheduler::kv_connector`] is the scheduler's
//!   connector instead of a `PyKvConnector`;
//! - `add_request` calls [`RustNixlConnectorScheduler::add_request`] with the
//!   Python request in place of the Python `on_new_request`, before the Rust
//!   request is built (it may truncate a P-side prompt);
//! - outputs: [`metadata_to_py`] for `kv_connector_metadata` and
//!   [`finished_params_to_py`] for `KvTransferParams::params`, both
//!   carrying the Rust values in their `Opaque`;
//! - the Python connector scheduler's `set_xfer_handshake_metadata` is
//!   wrapped to forward `_registration_generation` with
//!   [`RustNixlConnectorScheduler::set_registration_generation`].
//!
//! The native half keeps each request's `kv_transfer_params` flags
//! (`do_remote_prefill`, `_remote_blocks_processed`, push `remote_block_ids`)
//! to itself: unlike Python's, it does not update them in the Python
//! request's dict, which only the connector reads.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyAssertionError, PyNotImplementedError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PySet, PySlice, PyString, PyTuple};
use thiserror_ext::AsReport;
use vllm_engine_core::kv_cache::BlockId;
use vllm_engine_core::sched::output::Opaque;
use vllm_engine_core::sched::request::{Request, RequestStatus};
use vllm_engine_core_connectors::nixl::{
    AllocatedBlocks, BlockIds, NixlConnectorMetadata, NixlConnectorScheduler, NixlFinishedParams,
    NixlKvConnector, NixlKvTransferParams, NixlRequestView, NixlSchedulerConfig, PyNumber,
    RemoteBlockIds, ReqMeta, ScheduledReq, Seq, SeqKind, TransferMode,
};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustNixlConnectorScheduler>()?;
    Ok(())
}

const NIXL_MODULE: &str = "vllm.distributed.kv_transfer.kv_connector.v1.nixl";

/// The Python classes this module builds or accepts.
struct NixlTypes {
    pull_connector: Py<PyAny>,
    push_connector: Py<PyAny>,
    pull_scheduler: Py<PyAny>,
    push_scheduler: Py<PyAny>,
    metadata: Py<PyAny>,
    req_meta: Py<PyAny>,
    remote_meta: Py<PyAny>,
    heartbeat_info: Py<PyAny>,
}

static TYPES: PyOnceLock<NixlTypes> = PyOnceLock::new();

fn types(py: Python<'_>) -> PyResult<&NixlTypes> {
    TYPES.get_or_try_init(py, || {
        let connector = PyModule::import(py, format!("{NIXL_MODULE}.connector"))?;
        let metadata = PyModule::import(py, format!("{NIXL_MODULE}.metadata"))?;
        let get = |m: &Bound<'_, PyModule>, name: &str| -> PyResult<Py<PyAny>> {
            Ok(m.getattr(name)?.unbind())
        };
        Ok(NixlTypes {
            pull_connector: get(&connector, "NixlPullConnector")?,
            push_connector: get(&connector, "NixlPushConnector")?,
            pull_scheduler: get(
                &PyModule::import(py, format!("{NIXL_MODULE}.pull_scheduler"))?,
                "NixlPullConnectorScheduler",
            )?,
            push_scheduler: get(
                &PyModule::import(py, format!("{NIXL_MODULE}.push_scheduler"))?,
                "NixlPushConnectorScheduler",
            )?,
            metadata: get(&metadata, "NixlConnectorMetadata")?,
            req_meta: get(&metadata, "ReqMeta")?,
            remote_meta: get(&metadata, "RemoteMeta")?,
            heartbeat_info: get(&metadata, "HeartbeatInfo")?,
        })
    })
}

// ---------------------------------------------------------------------------
// kv_transfer_params
// ---------------------------------------------------------------------------

fn invalid(key: &str, what: &str, value: &Bound<'_, PyAny>) -> PyErr {
    PyValueError::new_err(format!(
        "invalid kv_transfer_params: {key} must be {what}, got {}",
        value.repr().map_or_else(|_| "?".to_owned(), |r| r.to_string())
    ))
}

/// A Python `int` (not a `bool`).
fn extract_int(key: &str, value: &Bound<'_, PyAny>) -> PyResult<i64> {
    if value.is_instance_of::<PyInt>() && !value.is_instance_of::<PyBool>() {
        value.extract().map_err(|_| invalid(key, "a 64-bit int", value))
    } else {
        Err(invalid(key, "an int", value))
    }
}

fn extract_str(key: &str, value: &Bound<'_, PyAny>) -> PyResult<String> {
    value.cast::<PyString>().map_err(|_| invalid(key, "a str", value))?.extract()
}

/// A Python `list` or `tuple`, with its items.
fn extract_seq<'py, T>(
    key: &str,
    value: &Bound<'py, PyAny>,
    item: impl Fn(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<Seq<T>> {
    let (kind, items) = if let Ok(list) = value.cast::<PyList>() {
        (
            SeqKind::List,
            list.iter().map(|x| item(&x)).collect::<PyResult<_>>()?,
        )
    } else if let Ok(tuple) = value.cast::<PyTuple>() {
        (
            SeqKind::Tuple,
            tuple.iter().map(|x| item(&x)).collect::<PyResult<_>>()?,
        )
    } else {
        return Err(invalid(key, "a list or tuple", value));
    };
    Ok(Seq::new(kind, items))
}

fn extract_remote_block_ids(value: &Bound<'_, PyAny>) -> PyResult<RemoteBlockIds> {
    const KEY: &str = "remote_block_ids";
    let first = value.get_item(0).ok();
    if first.is_some_and(|x| x.is_instance_of::<PyList>() || x.is_instance_of::<PyTuple>()) {
        let groups = extract_seq(KEY, value, |g| extract_seq(KEY, g, |b| extract_int(KEY, b)))?;
        Ok(RemoteBlockIds::Grouped(groups))
    } else {
        Ok(RemoteBlockIds::Flat(extract_seq(KEY, value, |b| {
            extract_int(KEY, b)
        })?))
    }
}

/// Parse `request.kv_transfer_params`: `None` for absent or empty params.
///
/// Raises `ValueError` for values the Python connector would forward to the
/// worker with a type it does not expect (see [`NixlKvTransferParams`]).
pub fn parse_kv_transfer_params(
    params: &Bound<'_, PyAny>,
) -> PyResult<Option<NixlKvTransferParams>> {
    if params.is_none() {
        return Ok(None);
    }
    let dict = params
        .cast::<PyDict>()
        .map_err(|_| invalid("kv_transfer_params", "a dict", params))?;
    if dict.is_empty() {
        return Ok(None);
    }
    // Present and not None.
    let value = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        Ok(dict.get_item(key)?.filter(|v| !v.is_none()))
    };
    // Keys read by presence or with a non-None default: None is rejected.
    let required = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        match dict.get_item(key)? {
            Some(v) if v.is_none() => Err(invalid(key, "set when present", &v)),
            v => Ok(v),
        }
    };
    let flag =
        |key: &str| -> PyResult<bool> { dict.get_item(key)?.map_or(Ok(false), |v| v.is_truthy()) };
    let int = |v: Option<Bound<'_, PyAny>>, key: &str| v.map(|v| extract_int(key, &v)).transpose();
    let string =
        |v: Option<Bound<'_, PyAny>>, key: &str| v.map(|v| extract_str(key, &v)).transpose();
    Ok(Some(NixlKvTransferParams {
        do_remote_prefill: flag("do_remote_prefill")?,
        do_remote_decode: flag("do_remote_decode")?,
        remote_block_ids: required("remote_block_ids")?
            .map(|v| extract_remote_block_ids(&v).map(Arc::new))
            .transpose()?,
        remote_engine_id: string(required("remote_engine_id")?, "remote_engine_id")?.map(Arc::from),
        remote_request_id: string(required("remote_request_id")?, "remote_request_id")?,
        remote_host: string(required("remote_host")?, "remote_host")?.map(Arc::from),
        remote_port: int(required("remote_port")?, "remote_port")?,
        tp_size: int(required("tp_size")?, "tp_size")?,
        dcp_size: int(required("dcp_size")?, "dcp_size")?,
        pp_size: int(required("pp_size")?, "pp_size")?,
        remote_block_size: int(value("remote_block_size")?, "remote_block_size")?,
        remote_num_tokens: int(value("remote_num_tokens")?, "remote_num_tokens")?,
        remote_blocks_expiry_time: value("remote_blocks_expiry_time")?
            .map(|v| {
                if v.is_exact_instance_of::<PyFloat>() {
                    Ok(PyNumber::Float(v.extract()?))
                } else {
                    extract_int("remote_blocks_expiry_time", &v).map(PyNumber::Int)
                }
            })
            .transpose()?,
        remote_registration_generation: int(
            required("remote_registration_generation")?,
            "remote_registration_generation",
        )?,
        p_side_truncated: flag("_p_side_truncated")?,
        remote_blocks_processed: flag("_remote_blocks_processed")?,
    }))
}

fn block_ids_to_py<'py>(py: Python<'py>, block_ids: &BlockIds) -> PyResult<Bound<'py, PyAny>> {
    let groups = block_ids.items.iter().map(|g| PyList::new(py, g));
    seq_to_py(py, block_ids.kind, groups.collect::<PyResult<Vec<_>>>()?)
}

fn seq_to_py<'py, T: IntoPyObject<'py>>(
    py: Python<'py>,
    kind: SeqKind,
    items: Vec<T>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(match kind {
        SeqKind::List => PyList::new(py, items)?.into_any(),
        SeqKind::Tuple => PyTuple::new(py, items)?.into_any(),
    })
}

fn remote_block_ids_to_py<'py>(
    py: Python<'py>,
    block_ids: &RemoteBlockIds,
) -> PyResult<Bound<'py, PyAny>> {
    match block_ids {
        RemoteBlockIds::Flat(s) => seq_to_py(py, s.kind, s.items.clone()),
        RemoteBlockIds::Grouped(s) => {
            let groups = s
                .items
                .iter()
                .map(|g| seq_to_py(py, g.kind, g.items.clone()))
                .collect::<PyResult<Vec<_>>>()?;
            seq_to_py(py, s.kind, groups)
        }
    }
}

fn number_to_py<'py>(py: Python<'py>, n: Option<PyNumber>) -> PyResult<Bound<'py, PyAny>> {
    Ok(match n {
        None => py.None().into_bound(py),
        Some(PyNumber::Int(i)) => i.into_pyobject(py)?.into_any(),
        Some(PyNumber::Float(f)) => f.into_pyobject(py)?.into_any(),
    })
}

/// The `kv_transfer_params` dict `request_finished` returns.
pub fn finished_params_to_py<'py>(
    py: Python<'py>,
    p: &NixlFinishedParams,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(intern!(py, "do_remote_prefill"), p.do_remote_prefill)?;
    d.set_item(intern!(py, "do_remote_decode"), p.do_remote_decode)?;
    let groups = p.remote_block_ids.iter().map(|g| PyList::new(py, g));
    d.set_item(
        intern!(py, "remote_block_ids"),
        PyTuple::new(py, groups.collect::<PyResult<Vec<_>>>()?)?,
    )?;
    d.set_item(intern!(py, "remote_engine_id"), &*p.remote_engine_id)?;
    d.set_item(intern!(py, "remote_request_id"), &p.remote_request_id)?;
    d.set_item(intern!(py, "remote_host"), &*p.remote_host)?;
    d.set_item(intern!(py, "remote_port"), p.remote_port)?;
    d.set_item(intern!(py, "tp_size"), p.tp_size)?;
    d.set_item(intern!(py, "dcp_size"), p.dcp_size)?;
    d.set_item(intern!(py, "pp_size"), p.pp_size)?;
    d.set_item(intern!(py, "remote_num_tokens"), p.remote_num_tokens)?;
    if p.mode == TransferMode::Pull {
        d.set_item(
            intern!(py, "remote_blocks_expiry_time"),
            p.remote_blocks_expiry_time,
        )?;
        d.set_item(
            intern!(py, "remote_registration_generation"),
            p.remote_registration_generation,
        )?;
    }
    d.set_item(intern!(py, "transfer_mode"), p.mode.as_str())?;
    Ok(d)
}

/// [`finished_params_to_py`] of a `KvTransferParams::params` built by the
/// native connector, `None` for other connectors' params.
pub fn finished_params_opaque_to_py<'py>(
    py: Python<'py>,
    params: &Opaque,
) -> Option<PyResult<Bound<'py, PyDict>>> {
    let params = params.0.downcast_ref::<NixlFinishedParams>()?;
    Some(finished_params_to_py(py, params))
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Python objects already built for shared block id lists: Python builds a
/// push registration and its recv `ReqMeta` from one object, which a pickle
/// of the metadata references twice.
type Shared<'py> = HashMap<*const BlockIds, Bound<'py, PyAny>>;

fn shared_block_ids_to_py<'py>(
    py: Python<'py>,
    block_ids: &Arc<BlockIds>,
    shared: &mut Shared<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(obj) = shared.get(&Arc::as_ptr(block_ids)) {
        return Ok(obj.clone());
    }
    let obj = block_ids_to_py(py, block_ids)?;
    shared.insert(Arc::as_ptr(block_ids), obj.clone());
    Ok(obj)
}

fn req_meta_to_py<'py>(
    py: Python<'py>,
    types: &NixlTypes,
    m: &ReqMeta,
    shared: &mut Shared<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let local_block_ids = shared_block_ids_to_py(py, &m.local_block_ids, shared)?;
    let remote = match &m.remote {
        None => py.None().into_bound(py),
        Some(r) => {
            let kwargs = PyDict::new(py);
            kwargs.set_item("block_ids", remote_block_ids_to_py(py, &r.block_ids)?)?;
            kwargs.set_item("host", &*r.host)?;
            kwargs.set_item("port", r.port)?;
            kwargs.set_item("engine_id", &*r.engine_id)?;
            kwargs.set_item("request_id", &r.request_id)?;
            kwargs.set_item(
                "blocks_expiry_time",
                number_to_py(py, r.blocks_expiry_time)?,
            )?;
            kwargs.set_item("num_tokens", r.num_tokens)?;
            kwargs.set_item("registration_generation", r.registration_generation)?;
            types.remote_meta.bind(py).call((), Some(&kwargs))?
        }
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item("local_block_ids", &local_block_ids)?;
    // The same object, as `_add_new_req` passes it.
    kwargs.set_item("local_physical_block_ids", &local_block_ids)?;
    kwargs.set_item("tp_size", m.tp_size)?;
    kwargs.set_item("dcp_size", m.dcp_size)?;
    kwargs.set_item(
        "local_num_computed_blocks",
        PyTuple::new(py, &m.local_num_computed_blocks)?,
    )?;
    kwargs.set_item("remote", remote)?;
    kwargs.set_item("remote_block_size", m.remote_block_size)?;
    kwargs.set_item("pp_size", m.pp_size)?;
    kwargs.set_item("awaiting_kvs", m.awaiting_kvs)?;
    types.req_meta.bind(py).call((), Some(&kwargs))
}

/// The Python `NixlConnectorMetadata` for the worker half.
pub fn metadata_to_py<'py>(
    py: Python<'py>,
    meta: &NixlConnectorMetadata,
) -> PyResult<Bound<'py, PyAny>> {
    let types = types(py)?;
    let mut shared = Shared::new();
    let obj = types.metadata.bind(py).call0()?;

    let reqs = |reqs: &indexmap::IndexMap<String, ReqMeta>,
                shared: &mut Shared<'py>|
     -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (id, m) in reqs {
            d.set_item(id, req_meta_to_py(py, types, m, shared)?)?;
        }
        Ok(d)
    };
    obj.setattr(
        intern!(py, "reqs_to_recv"),
        reqs(&meta.reqs_to_recv, &mut shared)?,
    )?;
    obj.setattr(
        intern!(py, "reqs_to_save"),
        reqs(&meta.reqs_to_save, &mut shared)?,
    )?;
    let send = PyDict::new(py);
    for (id, deadline) in &meta.reqs_to_send {
        send.set_item(id, deadline)?;
    }
    obj.setattr(intern!(py, "reqs_to_send"), send)?;
    obj.setattr(intern!(py, "scheduler_clock"), meta.scheduler_clock)?;
    obj.setattr(
        intern!(py, "reqs_in_batch"),
        PySet::new(py, &meta.reqs_in_batch)?,
    )?;
    obj.setattr(
        intern!(py, "reqs_not_processed"),
        PySet::new(py, &meta.reqs_not_processed)?,
    )?;

    let heartbeats = PyDict::new(py);
    for (engine_id, info) in &meta.heartbeat_by_engine {
        let kwargs = PyDict::new(py);
        kwargs.set_item("req_ids", PySet::new(py, &info.req_ids)?)?;
        kwargs.set_item("host", &*info.host)?;
        kwargs.set_item("port", info.port)?;
        kwargs.set_item("tp_size", info.tp_size)?;
        kwargs.set_item("dcp_size", info.dcp_size)?;
        kwargs.set_item("pp_size", info.pp_size)?;
        heartbeats.set_item(
            &**engine_id,
            types.heartbeat_info.bind(py).call((), Some(&kwargs))?,
        )?;
    }
    obj.setattr(intern!(py, "heartbeat_by_engine"), heartbeats)?;

    let registrations = PyDict::new(py);
    for (id, r) in &meta.push_registrations {
        let d = PyDict::new(py);
        d.set_item("request_id", &r.request_id)?;
        d.set_item("decode_engine_id", &*r.decode_engine_id)?;
        d.set_item("decode_host", &*r.decode_host)?;
        d.set_item("decode_port", r.decode_port)?;
        d.set_item("decode_tp_size", r.decode_tp_size)?;
        d.set_item(
            "local_block_ids",
            shared_block_ids_to_py(py, &r.local_block_ids, &mut shared)?,
        )?;
        d.set_item("remote_engine_id", &*r.remote_engine_id)?;
        d.set_item("remote_host", &*r.remote_host)?;
        d.set_item("remote_port", r.remote_port)?;
        d.set_item("remote_tp_size", r.remote_tp_size)?;
        d.set_item("remote_pp_size", r.remote_pp_size)?;
        registrations.set_item(id, d)?;
    }
    obj.setattr(intern!(py, "push_registrations"), registrations)?;

    let finished = PyDict::new(py);
    for (id, block_ids) in &meta.push_finished_blocks {
        finished.set_item(id, block_ids_to_py(py, block_ids)?)?;
    }
    obj.setattr(intern!(py, "push_finished_blocks"), finished)?;
    Ok(obj)
}

/// [`metadata_to_py`] of a `kv_connector_metadata` built by the native
/// connector, `None` for other connectors' metadata.
pub fn metadata_opaque_to_py<'py>(
    py: Python<'py>,
    meta: &Opaque,
) -> Option<PyResult<Bound<'py, PyAny>>> {
    let meta = meta.0.downcast_ref::<NixlConnectorMetadata>()?;
    Some(metadata_to_py(py, meta))
}

// ---------------------------------------------------------------------------
// The connector
// ---------------------------------------------------------------------------

fn unsupported(what: impl std::fmt::Display) -> PyErr {
    PyNotImplementedError::new_err(format!("Native NIXL scheduler: {what} is not supported"))
}

fn config_from_py(
    py: Python<'_>,
    connector: &Bound<'_, PyAny>,
) -> PyResult<(NixlSchedulerConfig, usize, usize)> {
    let types = types(py)?;
    let connector_type = connector.get_type();
    let mode = if connector_type.is(types.pull_connector.bind(py)) {
        TransferMode::Pull
    } else if connector_type.is(types.push_connector.bind(py)) {
        TransferMode::Push
    } else {
        return Err(unsupported(format!("connector class {connector_type}")));
    };
    let s = connector.getattr("connector_scheduler")?;
    if s.is_none() {
        return Err(PyValueError::new_err(
            "expected a scheduler-role NIXL connector",
        ));
    }
    let expected = match mode {
        TransferMode::Pull => &types.pull_scheduler,
        TransferMode::Push => &types.push_scheduler,
    };
    if !s.get_type().is(expected.bind(py)) {
        return Err(unsupported(format!(
            "connector scheduler class {}",
            s.get_type()
        )));
    }
    let vllm_config = s.getattr("vllm_config")?;
    let parallel = vllm_config.getattr("parallel_config")?;
    let kv_cache_config = s.getattr("kv_cache_config")?;
    let kv_transfer_config = vllm_config.getattr("kv_transfer_config")?;
    let ssm_spec_blocks: Vec<Option<usize>> = s.getattr("_ssm_spec_blocks")?.extract()?;
    let config = NixlSchedulerConfig {
        mode,
        engine_id: Arc::from(s.getattr("engine_id")?.extract::<String>()?),
        side_channel_host: Arc::from(s.getattr("side_channel_host")?.extract::<String>()?),
        side_channel_port: s.getattr("side_channel_port")?.extract()?,
        transfer_tp_size: s.getattr("transfer_tp_size")?.extract()?,
        tensor_parallel_size: parallel.getattr("tensor_parallel_size")?.extract()?,
        decode_context_parallel_size: parallel
            .getattr("decode_context_parallel_size")?
            .extract()?,
        pipeline_parallel_size: parallel.getattr("pipeline_parallel_size")?.extract()?,
        kv_lease_duration: s.getattr("_kv_lease_duration")?.extract()?,
        heartbeat_interval: s.getattr("_heartbeat_interval")?.extract()?,
        use_host_buffer: s.getattr("use_host_buffer")?.is_truthy()?,
        is_hma_required: s.getattr("_is_hma_required")?.is_truthy()?,
        num_kv_cache_groups: kv_cache_config.getattr("kv_cache_groups")?.len()?,
        transfer_group_ids: kv_cache_config.getattr("transfer_group_ids")?.extract()?,
        blocks_per_sw: s.getattr("blocks_per_sw")?.extract()?,
        ssm_spec_blocks,
        has_mamba: s.getattr("_has_mamba")?.is_truthy()?,
        bounded_replay: s.getattr("_bounded_replay")?.is_truthy()?,
        num_prefill_lookahead_tokens: vllm_config
            .getattr("num_prefill_lookahead_tokens")?
            .extract()?,
        kv_recompute_threshold: s.getattr("kv_recompute_threshold")?.extract()?,
        bidirectional_kv_xfer: s.getattr("is_bidirectional_kv_xfer_enabled")?.is_truthy()?,
        decoder_kv_blocks_ttl: s.getattr("decoder_kv_blocks_ttl")?.extract()?,
        is_kv_producer: kv_transfer_config.getattr("is_kv_producer")?.is_truthy()?,
        push_registration_timeout: match mode {
            TransferMode::Pull => 0.0,
            TransferMode::Push => s.getattr("_push_registration_timeout")?.extract()?,
        },
    };
    let hash_block_size = kv_cache_config.getattr("hash_block_size")?;
    let hash_block_size = if hash_block_size.is_none() {
        0
    } else {
        hash_block_size.extract()?
    };
    let num_hash_lookahead_tokens = vllm_config.getattr("num_hash_lookahead_tokens")?.extract()?;
    Ok((config, hash_block_size, num_hash_lookahead_tokens))
}

/// Blocks given as Python values (`update_state_after_alloc` from Python).
struct GivenBlocks {
    unhashed: Vec<Vec<BlockId>>,
    num_computed: Vec<usize>,
}

impl AllocatedBlocks for GivenBlocks {
    fn unhashed_block_ids_all_groups(&self) -> Vec<Vec<BlockId>> {
        self.unhashed.clone()
    }

    fn num_computed_blocks(&self) -> Vec<usize> {
        self.num_computed.clone()
    }
}

fn status_from_value(value: u8) -> PyResult<RequestStatus> {
    use RequestStatus::*;
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

/// The native NIXL scheduler half. The hook methods besides `add_request`
/// take plain values and exist for Python-driven tests; the Rust scheduler
/// drives [`Self::kv_connector`] directly.
#[pyclass(frozen, module = "vllm._rust_core")]
pub struct RustNixlConnectorScheduler {
    connector: NixlKvConnector,
    hash_block_size: usize,
    num_hash_lookahead_tokens: usize,
}

impl RustNixlConnectorScheduler {
    /// The connector for `Scheduler::new`.
    pub fn kv_connector(&self) -> NixlKvConnector {
        self.connector.clone()
    }

    /// [`Self::add_request`] for a request without a Python `Request`, with
    /// the `kv_transfer_params` it carried on the wire
    /// ([`NixlKvTransferParams::from_wire`]): the P-side truncation applies to
    /// the Rust request.
    pub fn add_native_request(
        &self,
        request: &mut Request,
        params: Option<NixlKvTransferParams>,
    ) -> PyResult<()> {
        self.connector
            .add_wire_request(
                request,
                params,
                self.hash_block_size,
                self.num_hash_lookahead_tokens,
            )
            .map_err(|e| PyValueError::new_err(e.to_report_string()))
    }
}

#[pymethods]
impl RustNixlConnectorScheduler {
    /// Build from a scheduler-role `NixlPullConnector` / `NixlPushConnector`.
    #[new]
    fn new(py: Python<'_>, connector: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (config, hash_block_size, num_hash_lookahead_tokens) = config_from_py(py, connector)?;
        let generation: u64 = connector
            .getattr("connector_scheduler")?
            .getattr("_registration_generation")?
            .extract()?;
        let mut scheduler = NixlConnectorScheduler::new(config)
            .map_err(|e| PyNotImplementedError::new_err(e.to_report_string()))?;
        scheduler.set_registration_generation(generation);
        Ok(Self {
            connector: NixlKvConnector::new(scheduler),
            hash_block_size,
            num_hash_lookahead_tokens,
        })
    }

    /// The `on_new_request` work that precedes building the Rust request:
    /// parse and register `request.kv_transfer_params`, and truncate a
    /// P-side prompt (`_truncate_request_for_prefill`) on the Python request.
    /// The heartbeat tracking runs in the Rust `on_new_request` hook.
    pub fn add_request(&self, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = request.py();
        let py_params = request.getattr(intern!(py, "kv_transfer_params"))?;
        let mut params = parse_kv_transfer_params(&py_params)?;
        let request_id: String = request.getattr(intern!(py, "request_id"))?.extract()?;
        let num_prompt_tokens: usize =
            request.getattr(intern!(py, "num_prompt_tokens"))?.extract()?;
        let skip_reading_external_kv =
            request.getattr(intern!(py, "skip_reading_external_kv"))?.is_truthy()?;
        let truncation = self.connector.lock().prefill_truncation(
            params.as_ref(),
            num_prompt_tokens,
            skip_reading_external_kv,
        );
        if let (Some(backoff), Some(params)) = (truncation, params.as_mut())
            && self.truncate_prompt(request, backoff)?
        {
            py_params.set_item(intern!(py, "_p_side_truncated"), true)?;
            params.p_side_truncated = true;
        }
        self.connector
            .lock()
            .add_request(&request_id, params)
            .map_err(|e| PyValueError::new_err(e.to_report_string()))
    }

    /// Forward the Python connector's `_registration_generation`.
    fn set_registration_generation(&self, generation: u64) {
        self.connector.lock().set_registration_generation(generation);
    }

    /// Fix the clock at `now` (tests).
    fn set_fixed_clock(&self, now: f64) {
        self.connector.lock().set_clock(Box::new(move || now));
    }

    fn on_new_request(&self, request_id: &str) {
        self.connector.lock().on_new_request(request_id);
    }

    fn may_load_async(&self, request_id: &str) -> bool {
        self.connector.lock().may_load_async(request_id)
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_prompt_token_ids: usize,
        num_prompt_tokens: usize,
        num_computed_tokens: usize,
    ) -> (usize, bool) {
        let view = NixlRequestView {
            request_id,
            num_prompt_token_ids,
            num_prompt_tokens,
            num_computed_tokens,
            status: RequestStatus::Waiting,
        };
        self.connector.lock().get_num_new_matched_tokens(&view, num_computed_tokens)
    }

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        unhashed_block_ids: Vec<Vec<BlockId>>,
        num_computed_blocks: Vec<usize>,
        num_external_tokens: usize,
    ) {
        let blocks = GivenBlocks {
            unhashed: unhashed_block_ids,
            num_computed: num_computed_blocks,
        };
        self.connector
            .lock()
            .update_state_after_alloc(request_id, &blocks, num_external_tokens);
    }

    /// `scheduled`: `(req_id, new_block_ids | None, num_scheduled_tokens,
    /// num_computed_tokens, num_prompt_tokens)` in `yield_req_data` order.
    #[allow(clippy::type_complexity)]
    fn build_connector_meta<'py>(
        &self,
        py: Python<'py>,
        scheduled: Vec<(String, Option<Vec<Vec<BlockId>>>, usize, usize, usize)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let meta = self.connector.lock().build_connector_meta(scheduled.iter().map(
            |(
                req_id,
                new_block_ids,
                num_scheduled_tokens,
                num_computed_tokens,
                num_prompt_tokens,
            )| {
                ScheduledReq {
                    req_id,
                    new_block_ids: new_block_ids.as_deref(),
                    num_scheduled_tokens: *num_scheduled_tokens,
                    num_computed_tokens: *num_computed_tokens,
                    num_prompt_tokens: *num_prompt_tokens,
                }
            },
        ));
        metadata_to_py(py, &meta)
    }

    fn request_finished<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
        status: u8,
        num_computed_tokens: usize,
        block_ids: Vec<Vec<BlockId>>,
    ) -> PyResult<(bool, Option<Bound<'py, PyDict>>)> {
        let view = NixlRequestView {
            request_id,
            num_prompt_token_ids: 0,
            num_prompt_tokens: 0,
            num_computed_tokens,
            status: status_from_value(status)?,
        };
        let (delay, params) = self.connector.lock().request_finished(&view, &block_ids);
        Ok((
            delay,
            params.map(|p| finished_params_to_py(py, &p)).transpose()?,
        ))
    }

    #[pyo3(signature = (finished_sending=None, finished_recving=None))]
    fn update_connector_output(
        &self,
        finished_sending: Option<Vec<String>>,
        finished_recving: Option<Vec<String>>,
    ) {
        self.connector
            .lock()
            .update_connector_output(finished_sending.as_deref(), finished_recving.as_deref());
    }

    fn has_pending_push_work(&self) -> bool {
        self.connector.lock().has_pending_push_work()
    }
}

impl RustNixlConnectorScheduler {
    /// Drop the trailing `backoff` prompt tokens of the Python request as
    /// `_truncate_request_for_prefill` does; `false` when it has neither
    /// prompt token ids nor prompt embeds.
    fn truncate_prompt(&self, request: &Bound<'_, PyAny>, backoff: usize) -> PyResult<bool> {
        let py = request.py();
        let tail = PySlice::new(py, -(backoff as isize), isize::MAX, 1);
        let prompt_token_ids = request.getattr(intern!(py, "prompt_token_ids"))?;
        let prompt_embeds = request.getattr(intern!(py, "prompt_embeds"))?;
        if !prompt_token_ids.is_none() {
            prompt_token_ids.del_item(&tail)?;
        } else if !prompt_embeds.is_none() {
            let head = PySlice::new(py, 0, -(backoff as isize), 1);
            request.setattr(intern!(py, "prompt_embeds"), prompt_embeds.get_item(head)?)?;
        } else {
            return Ok(false);
        }
        request.getattr(intern!(py, "_all_token_ids"))?.del_item(&tail)?;
        let num_prompt_tokens: usize =
            request.getattr(intern!(py, "num_prompt_tokens"))?.extract()?;
        request.setattr(
            intern!(py, "num_prompt_tokens"),
            num_prompt_tokens - backoff,
        )?;
        request.setattr(intern!(py, "max_tokens"), 1)?;
        if self.hash_block_size == 0 {
            return Err(PyAssertionError::new_err("hash_block_size is not resolved"));
        }
        request.call_method1(
            intern!(py, "truncate_block_hashes"),
            (self.hash_block_size, self.num_hash_lookahead_tokens),
        )?;
        Ok(true)
    }
}
