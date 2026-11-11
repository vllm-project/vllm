// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `kv_transfer_params` as the NIXL scheduler reads and returns them.
//!
//! The request's params dict is parsed once when the request is added
//! ([`NixlKvTransferParams`]); the params returned by `request_finished` are
//! built here ([`NixlFinishedParams`]) and converted to the Python dict by
//! the binding layer, key order included.

use std::sync::Arc;

use crate::kv_cache::BlockId;

/// The Python sequence type of a block id list. Pickles of a `list` and a
/// `tuple` differ, and the worker receives what the scheduler built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqKind {
    List,
    Tuple,
}

/// A Python `list` or `tuple`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seq<T> {
    pub kind: SeqKind,
    pub items: Vec<T>,
}

impl<T> Seq<T> {
    pub fn new(kind: SeqKind, items: Vec<T>) -> Self {
        Self { kind, items }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// A `BlockIds` value the scheduler builds: per-group lists of block ids in a
/// list or a tuple.
pub type BlockIds = Seq<Vec<BlockId>>;

/// `remote_block_ids` as the request carried it: per-group lists from a
/// hybrid-aware producer, or a flat list of block ids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoteBlockIds {
    Flat(Seq<i64>),
    Grouped(Seq<Seq<i64>>),
}

impl RemoteBlockIds {
    /// The empty tuple the push scheduler seeds (`params["remote_block_ids"] = ()`).
    pub fn empty_tuple() -> Self {
        Self::Flat(Seq::new(SeqKind::Tuple, Vec::new()))
    }

    /// Python truthiness.
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Flat(s) => !s.is_empty(),
            Self::Grouped(s) => !s.is_empty(),
        }
    }
}

/// A Python `int` or `float`, kept as given.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PyNumber {
    Int(i64),
    Float(f64),
}

/// The keys of a request's `kv_transfer_params` the NIXL scheduler reads.
///
/// Optional keys are `None` when absent. A key whose value is `None` reads
/// the same as an absent one for the optional keys the scheduler reads with
/// `.get()`; the parser rejects `None` for the keys Python reads by presence
/// (`remote_engine_id`, `remote_request_id`, `remote_host`, `remote_port`,
/// `remote_block_ids`) or with a non-`None` default (`tp_size`, `dcp_size`,
/// `pp_size`, `remote_registration_generation`), where a `None` value would
/// travel on to the worker. Unknown keys are ignored: the scheduler never
/// reads them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NixlKvTransferParams {
    pub do_remote_prefill: bool,
    pub do_remote_decode: bool,
    pub remote_block_ids: Option<Arc<RemoteBlockIds>>,
    pub remote_engine_id: Option<Arc<str>>,
    pub remote_request_id: Option<String>,
    pub remote_host: Option<Arc<str>>,
    pub remote_port: Option<i64>,
    pub tp_size: Option<i64>,
    pub dcp_size: Option<i64>,
    pub pp_size: Option<i64>,
    pub remote_block_size: Option<i64>,
    pub remote_num_tokens: Option<i64>,
    pub remote_blocks_expiry_time: Option<PyNumber>,
    pub remote_registration_generation: Option<i64>,
    /// `_p_side_truncated`: the P-side prompt truncation already ran.
    pub p_side_truncated: bool,
    /// `_remote_blocks_processed`: the (single) KV transfer was set up.
    pub remote_blocks_processed: bool,
}

impl NixlKvTransferParams {
    /// Whether `remote_block_ids` is present and truthy.
    pub fn has_remote_block_ids(&self) -> bool {
        self.remote_block_ids.as_deref().is_some_and(RemoteBlockIds::is_truthy)
    }

    /// Whether `remote_engine_id`, `remote_request_id`, `remote_host` and
    /// `remote_port` are all present.
    pub fn has_remote_address(&self) -> bool {
        self.remote_engine_id.is_some()
            && self.remote_request_id.is_some()
            && self.remote_host.is_some()
            && self.remote_port.is_some()
    }
}

impl NixlKvTransferParams {
    /// Parse `kv_transfer_params` as an add request carries them on the wire
    /// (`sampling_params.extra_args["kv_transfer_params"]`), with the result
    /// of parsing the dict Python decodes from it: arrays are lists, and the
    /// same keys are rejected for the same values. `None` for absent or empty
    /// params.
    pub fn from_wire(params: &serde_json::Value) -> Result<Option<Self>, String> {
        use serde_json::Value;

        fn invalid(key: &str, what: &str, value: &Value) -> String {
            format!("invalid kv_transfer_params: {key} must be {what}, got {value}")
        }
        fn int(key: &str, value: &Value) -> Result<i64, String> {
            value.as_i64().ok_or_else(|| invalid(key, "a 64-bit int", value))
        }
        fn list<'a>(key: &str, value: &'a Value) -> Result<&'a Vec<Value>, String> {
            value.as_array().ok_or_else(|| invalid(key, "a list or tuple", value))
        }
        fn ints(key: &str, value: &Value) -> Result<Seq<i64>, String> {
            let items = list(key, value)?.iter().map(|v| int(key, v));
            Ok(Seq::new(SeqKind::List, items.collect::<Result<_, _>>()?))
        }
        /// Python truthiness.
        fn truthy(value: &Value) -> bool {
            match value {
                Value::Null => false,
                Value::Bool(b) => *b,
                Value::Number(n) => n.as_f64().is_some_and(|n| n != 0.0),
                Value::String(s) => !s.is_empty(),
                Value::Array(a) => !a.is_empty(),
                Value::Object(o) => !o.is_empty(),
            }
        }

        let dict = match params {
            Value::Null => return Ok(None),
            Value::Object(dict) if dict.is_empty() => return Ok(None),
            Value::Object(dict) => dict,
            other => return Err(invalid("kv_transfer_params", "a dict", other)),
        };
        // Present and not None.
        let value = |key: &str| dict.get(key).filter(|v| !v.is_null());
        // Keys read by presence or with a non-None default: None is rejected.
        let required = |key: &str| -> Result<Option<&Value>, String> {
            match dict.get(key) {
                Some(v) if v.is_null() => Err(invalid(key, "set when present", v)),
                v => Ok(v),
            }
        };
        let flag = |key: &str| dict.get(key).is_some_and(truthy);
        let opt_int = |v: Option<&Value>, key: &str| v.map(|v| int(key, v)).transpose();
        let opt_str = |v: Option<&Value>, key: &str| {
            v.map(|v| v.as_str().map(str::to_owned).ok_or_else(|| invalid(key, "a str", v)))
                .transpose()
        };
        let remote_block_ids = required("remote_block_ids")?
            .map(|v| -> Result<_, String> {
                const KEY: &str = "remote_block_ids";
                let items = list(KEY, v)?;
                Ok(Arc::new(if items.first().is_some_and(Value::is_array) {
                    let groups = items.iter().map(|g| ints(KEY, g));
                    RemoteBlockIds::Grouped(Seq::new(
                        SeqKind::List,
                        groups.collect::<Result<_, _>>()?,
                    ))
                } else {
                    RemoteBlockIds::Flat(ints(KEY, v)?)
                }))
            })
            .transpose()?;
        Ok(Some(Self {
            do_remote_prefill: flag("do_remote_prefill"),
            do_remote_decode: flag("do_remote_decode"),
            remote_block_ids,
            remote_engine_id: opt_str(required("remote_engine_id")?, "remote_engine_id")?
                .map(Arc::from),
            remote_request_id: opt_str(required("remote_request_id")?, "remote_request_id")?,
            remote_host: opt_str(required("remote_host")?, "remote_host")?.map(Arc::from),
            remote_port: opt_int(required("remote_port")?, "remote_port")?,
            tp_size: opt_int(required("tp_size")?, "tp_size")?,
            dcp_size: opt_int(required("dcp_size")?, "dcp_size")?,
            pp_size: opt_int(required("pp_size")?, "pp_size")?,
            remote_block_size: opt_int(value("remote_block_size"), "remote_block_size")?,
            remote_num_tokens: opt_int(value("remote_num_tokens"), "remote_num_tokens")?,
            remote_blocks_expiry_time: value("remote_blocks_expiry_time")
                .map(|v| match v.as_i64() {
                    Some(i) => Ok(PyNumber::Int(i)),
                    None => v
                        .as_f64()
                        .filter(|_| v.is_f64())
                        .map(PyNumber::Float)
                        .ok_or_else(|| invalid("remote_blocks_expiry_time", "an int", v)),
                })
                .transpose()?,
            remote_registration_generation: opt_int(
                required("remote_registration_generation")?,
                "remote_registration_generation",
            )?,
            p_side_truncated: flag("_p_side_truncated"),
            remote_blocks_processed: flag("_remote_blocks_processed"),
        }))
    }
}

/// `NixlBaseConnectorScheduler._TRANSFER_MODE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferMode {
    Pull,
    Push,
}

impl TransferMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pull => "pull",
            Self::Push => "push",
        }
    }
}

/// The `kv_transfer_params` dict `request_finished` returns, in key order:
///
/// `do_remote_prefill`, `do_remote_decode`, `remote_block_ids` (a tuple of
/// lists), `remote_engine_id`, `remote_request_id`, `remote_host`,
/// `remote_port`, `tp_size`, `dcp_size`, `pp_size`, `remote_num_tokens`,
/// then for pull only `remote_blocks_expiry_time` and
/// `remote_registration_generation`, and last `transfer_mode`.
#[derive(Clone, Debug, PartialEq)]
pub struct NixlFinishedParams {
    pub mode: TransferMode,
    pub do_remote_prefill: bool,
    pub do_remote_decode: bool,
    pub remote_block_ids: Vec<Vec<BlockId>>,
    pub remote_engine_id: Arc<str>,
    pub remote_request_id: String,
    pub remote_host: Arc<str>,
    pub remote_port: i64,
    pub tp_size: i64,
    pub dcp_size: i64,
    pub pp_size: i64,
    pub remote_num_tokens: usize,
    /// Pull only.
    pub remote_blocks_expiry_time: Option<f64>,
    /// Pull only.
    pub remote_registration_generation: u64,
}
