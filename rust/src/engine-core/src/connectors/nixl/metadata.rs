// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `NixlConnectorMetadata` and the structs it holds
//! (`nixl/metadata.py`): what the scheduler hands the NIXL workers each step.
//!
//! Each struct holds exactly the data of its Python counterpart; the binding
//! layer builds the Python objects (same classes, field values, types and
//! order) for the unchanged Python worker.

use std::sync::Arc;

use indexmap::{IndexMap, IndexSet};

use super::params::{BlockIds, PyNumber, RemoteBlockIds};

/// `RemoteMeta`.
#[derive(Clone, Debug, PartialEq)]
pub struct RemoteMeta {
    pub block_ids: Arc<RemoteBlockIds>,
    pub host: Arc<str>,
    pub port: i64,
    pub engine_id: Arc<str>,
    pub request_id: String,
    pub blocks_expiry_time: Option<PyNumber>,
    pub num_tokens: Option<i64>,
    pub registration_generation: i64,
}

/// `ReqMeta`. `local_physical_block_ids` is the same object as
/// `local_block_ids`; `region_blocks_to_zero` and `covered_sub_blocks` are
/// worker-only and always `None` here.
#[derive(Clone, Debug, PartialEq)]
pub struct ReqMeta {
    /// Shared with a push registration of the same step, which Python builds
    /// from the same object.
    pub local_block_ids: Arc<BlockIds>,
    pub tp_size: i64,
    pub dcp_size: i64,
    /// A tuple.
    pub local_num_computed_blocks: Vec<usize>,
    pub remote: Option<RemoteMeta>,
    pub remote_block_size: Option<i64>,
    pub pp_size: i64,
    pub awaiting_kvs: bool,
}

/// `HeartbeatInfo`.
#[derive(Clone, Debug, PartialEq)]
pub struct HeartbeatInfo {
    /// A Python `set`.
    pub req_ids: IndexSet<String>,
    pub host: Arc<str>,
    pub port: i64,
    pub tp_size: i64,
    pub dcp_size: i64,
    pub pp_size: i64,
}

/// A push-mode D-side registration (`push_registrations` values, a dict with
/// these keys in this order).
#[derive(Clone, Debug, PartialEq)]
pub struct PushRegistration {
    pub request_id: String,
    pub decode_engine_id: Arc<str>,
    pub decode_host: Arc<str>,
    pub decode_port: i64,
    pub decode_tp_size: i64,
    pub local_block_ids: Arc<BlockIds>,
    pub remote_engine_id: Arc<str>,
    pub remote_host: Arc<str>,
    pub remote_port: i64,
    pub remote_tp_size: i64,
    pub remote_pp_size: i64,
}

/// `NixlConnectorMetadata`. Python dicts and sets keep insertion order,
/// mirrored with `IndexMap` / `IndexSet`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NixlConnectorMetadata {
    pub reqs_to_recv: IndexMap<String, ReqMeta>,
    pub reqs_to_save: IndexMap<String, ReqMeta>,
    pub reqs_to_send: IndexMap<String, f64>,
    pub scheduler_clock: f64,
    pub reqs_in_batch: IndexSet<String>,
    pub reqs_not_processed: IndexSet<String>,
    pub heartbeat_by_engine: IndexMap<Arc<str>, HeartbeatInfo>,
    pub push_registrations: IndexMap<String, PushRegistration>,
    pub push_finished_blocks: IndexMap<String, BlockIds>,
}
