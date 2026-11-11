// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Conversion between the Rust scheduler's inputs and outputs and the Python
//! objects the engine exchanges with the scheduler: `SchedulerOutput`,
//! `ModelRunnerOutput`, `EngineCoreOutputs` and `SchedulerStats`.

use indexmap::IndexMap;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyString, PyTuple};
use rustc_hash::FxHashMap;
use vllm_engine_core::kv_cache::PrefixCacheStats;
use vllm_engine_core::sched::output::{EncoderInputUpdate, Opaque, SchedulerOutput};
use vllm_engine_core::sched::request::{PrefillStats, Request, StopReason};
use vllm_engine_core::sched::scheduler::{
    DraftTokenIds, EngineCoreOutput, KvConnectorOutput, KvTransferParams, ModelRunnerOutput,
    UpdateOutput,
};
use vllm_engine_core::sched::stats::{
    ConnectorPrefixCacheStats, RequestSpecDecodeMetrics, SchedulerStats, SpecDecodingStats,
};

use super::request::{RequestTable, sync_tokens};
use super::types::{PyTypes, types};

/// The Python object an [`Opaque`] carries.
pub fn opaque_obj<'py>(py: Python<'py>, opaque: &Opaque) -> Bound<'py, PyAny> {
    opaque
        .0
        .downcast_ref::<Py<PyAny>>()
        .expect("opaque values of the binding are Python objects")
        .bind(py)
        .clone()
}

/// The Python value of an [`Opaque`] a hook returned: the Python object of
/// a Python hook, or the Python form of a native connector's value.
pub fn opaque_to_py<'py>(py: Python<'py>, opaque: &Opaque) -> PyResult<Bound<'py, PyAny>> {
    if let Some(obj) = opaque.0.downcast_ref::<Py<PyAny>>() {
        return Ok(obj.bind(py).clone());
    }
    if let Some(meta) = crate::nixl::metadata_opaque_to_py(py, opaque) {
        return meta;
    }
    if let Some(params) = crate::nixl::finished_params_opaque_to_py(py, opaque) {
        return params.map(Bound::into_any);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "unexpected native connector value",
    ))
}

pub fn opaque(obj: Bound<'_, PyAny>) -> Opaque {
    Opaque::new(obj.unbind())
}

/// The id object of a request: the Python request's own while it is held.
fn id_obj<'py>(py: Python<'py>, table: &RequestTable, id: &str) -> Bound<'py, PyString> {
    match table.get(id) {
        Some(req) => req.id.bind(py).clone(),
        None => PyString::new(py, id),
    }
}

fn group_tuple<'py>(py: Python<'py>, groups: &[Vec<u32>]) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        groups.iter().map(|g| PyList::new(py, g)).collect::<PyResult<Vec<_>>>()?,
    )
}

fn str_set<'py>(
    py: Python<'py>,
    ids: impl IntoIterator<Item = impl AsRef<str>>,
) -> PyResult<Bound<'py, PySet>> {
    let set = PySet::empty(py)?;
    for id in ids {
        set.add(id.as_ref())?;
    }
    Ok(set)
}

/// Build the Python `SchedulerOutput` of `so`.
///
/// `lookup` resolves the Rust requests, whose output tokens a resumed
/// request's `prefill_token_ids` include. Fields `schedule` sets after the
/// connector metadata is built are set by [`finish_scheduler_output`].
pub fn scheduler_output_to_py<'a, 'py>(
    py: Python<'py>,
    so: &SchedulerOutput,
    table: &RequestTable,
    uses_mrope: bool,
    lookup: impl Fn(&str) -> Option<&'a Request>,
) -> PyResult<Bound<'py, PyAny>> {
    let types = types(py)?;

    let mut new_reqs = Vec::with_capacity(so.scheduled_new_reqs.len());
    for nr in &so.scheduled_new_reqs {
        let entry = &table[nr.req_id.as_str()];
        let req = entry.obj.bind(py);
        let prompt = req.getattr(intern!(py, "prompt_token_ids"))?;
        let prompt_is_token_ids = req.getattr(intern!(py, "prompt_is_token_ids"))?;
        // `request.prefill_token_ids`: the prompt object itself while it holds
        // all the tokens.
        let prefill = if !prompt.is_none()
            && prompt_is_token_ids.is_none()
            && prompt.len()? == nr.prefill_token_ids.len()
        {
            prompt.clone()
        } else {
            let request = lookup(&nr.req_id).expect("a scheduled request exists");
            sync_tokens(req, request)?;
            req.getattr(intern!(py, "_all_token_ids"))?
        };
        let mut mm_features = req.getattr(intern!(py, "mm_features"))?;
        if nr.num_computed_tokens > 0 && mm_features.is_truthy()? {
            mm_features = types.strip_covered_mm_data.bind(py).call1((
                mm_features,
                nr.num_computed_tokens,
                uses_mrope,
            ))?;
        }
        let args = PyTuple::new(
            py,
            [
                entry.id.bind(py).clone().into_any(),
                prompt,
                mm_features,
                req.getattr(intern!(py, "sampling_params"))?,
                req.getattr(intern!(py, "pooling_params"))?,
                group_tuple(py, &nr.block_ids)?.into_any(),
                nr.num_computed_tokens.into_pyobject(py)?.into_any(),
                req.getattr(intern!(py, "lora_request"))?,
                prefill,
                req.getattr(intern!(py, "prompt_embeds"))?,
                prompt_is_token_ids,
                nr.replay_start.into_pyobject(py)?.into_any(),
            ],
        )?;
        new_reqs.push(types.new_request_data.bind(py).call1(args)?);
    }

    let cached = &so.scheduled_cached_reqs;
    let cached_req_ids = PyList::new(py, cached.req_ids.iter().map(|id| id_obj(py, table, id)))?;
    let new_token_ids = PyList::new(
        py,
        cached
            .new_token_ids
            .iter()
            .map(|t| PyList::new(py, t))
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    let new_block_ids = PyList::new(
        py,
        cached
            .new_block_ids
            .iter()
            .map(|b| match b {
                None => Ok(py.None().into_bound(py)),
                Some(groups) => group_tuple(py, groups).map(Bound::into_any),
            })
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    let cached_reqs = types.cached_request_data.bind(py).call1((
        cached_req_ids,
        new_token_ids,
        new_block_ids,
        PyList::new(py, &cached.num_computed_tokens)?,
        PyList::new(py, &cached.num_output_tokens)?,
    ))?;

    let num_scheduled_tokens = PyDict::new(py);
    for (id, &n) in &so.num_scheduled_tokens {
        num_scheduled_tokens.set_item(id_obj(py, table, id), n)?;
    }
    let spec_decode_tokens = PyDict::new(py);
    for (id, tokens) in &so.scheduled_spec_decode_tokens {
        spec_decode_tokens.set_item(id_obj(py, table, id), PyList::new(py, tokens)?)?;
    }
    let encoder_inputs = PyDict::new(py);
    for (id, inputs) in &so.scheduled_encoder_inputs {
        encoder_inputs.set_item(id_obj(py, table, id), PyList::new(py, inputs)?)?;
    }
    let encoder_input_updates = PyDict::new(py);
    for (id, updates) in &so.encoder_input_updates {
        let per_input = PyDict::new(py);
        for (&input_id, update) in updates {
            let value = match update {
                EncoderInputUpdate::Release => py.None().into_bound(py),
                EncoderInputUpdate::Restore => table[id.as_str()]
                    .obj
                    .bind(py)
                    .getattr(intern!(py, "mm_features"))?
                    .get_item(input_id)?
                    .getattr(intern!(py, "data"))?,
            };
            per_input.set_item(input_id, value)?;
        }
        encoder_input_updates.set_item(id_obj(py, table, id), per_input)?;
    }
    let iteration_details = match &so.iteration_details {
        None => py.None().into_bound(py),
        Some(d) => types.iteration_details.bind(py).call1((
            d.num_prefill_requests,
            d.num_prefill_tokens,
            d.num_decode_requests,
            d.num_decode_tokens,
            d.num_encoder_inputs,
            d.num_encoder_output_tokens,
        ))?,
    };
    let block_copies = match &so.kv_cache_block_copies {
        None => py.None().into_bound(py),
        Some(copies) => {
            let cls = types.kv_cache_block_copy.bind(py);
            PyList::new(
                py,
                copies
                    .iter()
                    .map(|c| cls.call1((c.src_block_id, c.dst_block_id)))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any()
        }
    };

    let kwargs = PyDict::new(py);
    kwargs.set_item(
        intern!(py, "scheduled_new_reqs"),
        PyList::new(py, new_reqs)?,
    )?;
    kwargs.set_item(intern!(py, "scheduled_cached_reqs"), cached_reqs)?;
    kwargs.set_item(intern!(py, "num_scheduled_tokens"), num_scheduled_tokens)?;
    kwargs.set_item(
        intern!(py, "total_num_scheduled_tokens"),
        so.total_num_scheduled_tokens,
    )?;
    kwargs.set_item(
        intern!(py, "scheduled_spec_decode_tokens"),
        spec_decode_tokens,
    )?;
    kwargs.set_item(intern!(py, "scheduled_encoder_inputs"), encoder_inputs)?;
    kwargs.set_item(intern!(py, "iteration_details"), iteration_details)?;
    kwargs.set_item(
        intern!(py, "preempted_req_ids"),
        so.preempted_req_ids.as_ref().map(|ids| str_set(py, ids)).transpose()?,
    )?;
    kwargs.set_item(
        intern!(py, "finished_req_ids"),
        str_set(py, &so.finished_req_ids)?,
    )?;
    kwargs.set_item(
        intern!(py, "free_encoder_mm_hashes"),
        PyList::new(py, &so.free_encoder_mm_hashes)?,
    )?;
    kwargs.set_item(intern!(py, "encoder_input_updates"), encoder_input_updates)?;
    kwargs.set_item(
        intern!(py, "new_block_ids_to_zero"),
        so.new_block_ids_to_zero.as_ref().map(|ids| PyList::new(py, ids)).transpose()?,
    )?;
    kwargs.set_item(intern!(py, "has_sync_kv_loads"), so.has_sync_kv_loads)?;
    kwargs.set_item(intern!(py, "kv_cache_block_copies"), block_copies)?;
    kwargs.set_item(
        intern!(py, "num_spec_tokens_to_schedule"),
        so.num_spec_tokens_to_schedule,
    )?;
    kwargs.set_item(intern!(py, "skip_drafts"), so.skip_drafts)?;
    types.scheduler_output.bind(py).call((), Some(&kwargs))
}

/// Set the fields of `py_so` that `schedule` fills in after the connector
/// metadata was built.
pub fn finish_scheduler_output(py_so: &Bound<'_, PyAny>, so: &SchedulerOutput) -> PyResult<()> {
    let py = py_so.py();
    py_so.setattr(
        intern!(py, "has_structured_output_requests"),
        so.has_structured_output_requests,
    )?;
    py_so.setattr(
        intern!(py, "has_structured_output_spec_decode"),
        so.has_structured_output_spec_decode,
    )?;
    py_so.setattr(
        intern!(py, "pending_structured_output_tokens"),
        so.pending_structured_output_tokens,
    )?;
    if let Some(meta) = &so.kv_connector_metadata {
        py_so.setattr(
            intern!(py, "kv_connector_metadata"),
            opaque_to_py(py, meta)?,
        )?;
    }
    if let Some(meta) = &so.aux_output_connector_metadata {
        py_so.setattr(
            intern!(py, "aux_output_connector_metadata"),
            opaque_obj(py, meta),
        )?;
    }
    if let Some(meta) = &so.ec_connector_metadata {
        py_so.setattr(intern!(py, "ec_connector_metadata"), opaque_obj(py, meta))?;
    }
    Ok(())
}

/// A `dict[str, int]`, in its order.
pub fn counts_of(dict: &Bound<'_, PyAny>) -> PyResult<IndexMap<String, usize>> {
    dict.cast::<PyDict>()?
        .iter()
        .map(|(k, v)| Ok((k.extract()?, v.extract()?)))
        .collect()
}

fn spec_tokens_of(dict: &Bound<'_, PyAny>) -> PyResult<IndexMap<String, Vec<i32>>> {
    dict.cast::<PyDict>()?
        .iter()
        .map(|(k, v)| Ok((k.extract()?, v.extract()?)))
        .collect()
}

/// The fields of a Python `SchedulerOutput` that `update_from_output` and
/// `update_draft_token_ids_in_output` read, for outputs the scheduler did not
/// produce itself.
pub fn scheduler_output_from_py(py_so: &Bound<'_, PyAny>) -> PyResult<SchedulerOutput> {
    let py = py_so.py();
    let num_invalid = py_so.getattr(intern!(py, "num_invalid_spec_tokens"))?;
    Ok(SchedulerOutput {
        num_scheduled_tokens: counts_of(&py_so.getattr(intern!(py, "num_scheduled_tokens"))?)?,
        total_num_scheduled_tokens: py_so
            .getattr(intern!(py, "total_num_scheduled_tokens"))?
            .extract()?,
        scheduled_spec_decode_tokens: spec_tokens_of(
            &py_so.getattr(intern!(py, "scheduled_spec_decode_tokens"))?,
        )?,
        num_invalid_spec_tokens: if num_invalid.is_none() {
            None
        } else {
            Some(counts_of(&num_invalid)?)
        },
        has_structured_output_requests: py_so
            .getattr(intern!(py, "has_structured_output_requests"))?
            .extract()?,
        ..Default::default()
    })
}

/// Write the draft updates of `update_draft_token_ids_in_output` back to the
/// Python `SchedulerOutput`.
pub fn write_back_drafts(
    py_so: &Bound<'_, PyAny>,
    so: &SchedulerOutput,
    req_ids: &[String],
) -> PyResult<()> {
    let py = py_so.py();
    let spec_tokens = py_so.getattr(intern!(py, "scheduled_spec_decode_tokens"))?;
    let spec_tokens = spec_tokens.cast::<PyDict>()?;
    for req_id in req_ids {
        if let Some(tokens) = so.scheduled_spec_decode_tokens.get(req_id)
            && spec_tokens.contains(req_id)?
        {
            spec_tokens.set_item(req_id, PyList::new(py, tokens)?)?;
        }
    }
    let invalid = PyDict::new(py);
    for (req_id, &n) in so.num_invalid_spec_tokens.iter().flatten() {
        invalid.set_item(req_id, n)?;
    }
    py_so.setattr(intern!(py, "num_invalid_spec_tokens"), invalid)
}

pub fn draft_token_ids_from_py(draft: &Bound<'_, PyAny>) -> PyResult<DraftTokenIds> {
    let py = draft.py();
    Ok(DraftTokenIds {
        req_ids: draft.getattr(intern!(py, "req_ids"))?.extract()?,
        draft_token_ids: draft.getattr(intern!(py, "draft_token_ids"))?.extract()?,
    })
}

fn kv_connector_output_from_py(output: &Bound<'_, PyAny>) -> PyResult<KvConnectorOutput> {
    let py = output.py();
    let ids = |name: &Bound<'_, PyString>| -> PyResult<Option<Vec<String>>> {
        let set = output.getattr(name)?;
        if set.is_none() {
            return Ok(None);
        }
        set.try_iter()?.map(|id| id?.extract()).collect::<PyResult<_>>().map(Some)
    };
    Ok(KvConnectorOutput {
        finished_sending: ids(intern!(py, "finished_sending"))?,
        finished_recving: ids(intern!(py, "finished_recving"))?,
        invalid_block_ids: output
            .getattr(intern!(py, "invalid_block_ids"))?
            .try_iter()?
            .map(|b| b?.extract())
            .collect::<PyResult<_>>()?,
        failed_recving: ids(intern!(py, "failed_recving"))?.unwrap_or_default(),
        output: Some(opaque(output.clone())),
    })
}

/// A Python `ModelRunnerOutput`, with the fields the Rust scheduler reads
/// converted and the payloads the outputs are sliced from kept.
pub struct ModelOutput<'py> {
    pub rust: ModelRunnerOutput,
    pub logprobs: Bound<'py, PyAny>,
    pub sampling_masks: Bound<'py, PyAny>,
    pooler_output: Bound<'py, PyAny>,
    pub prompt_logprobs: Option<Bound<'py, PyDict>>,
    pub prompt_token_id_logprobs: Option<Bound<'py, PyDict>>,
    pub kv_connector_output: Bound<'py, PyAny>,
    pub ec_connector_output: Bound<'py, PyAny>,
    pub cudagraph_stats: Bound<'py, PyAny>,
}

/// A dict, or `None` when it is empty (the common case: nothing to look up).
fn non_empty_dict<'py>(obj: Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyDict>>> {
    let dict = obj.cast_into::<PyDict>()?;
    Ok((!dict.is_empty()).then_some(dict))
}

pub fn model_output_from_py<'py>(output: &Bound<'py, PyAny>) -> PyResult<ModelOutput<'py>> {
    let py = output.py();
    let get = |name: &Bound<'py, PyString>| output.getattr(name);
    let sampled = get(intern!(py, "sampled_token_ids"))?;
    let sampled_token_ids = match sampled.cast::<PyList>() {
        Ok(list) => list.iter().map(|t| t.extract()).collect::<PyResult<_>>()?,
        Err(_) => sampled.extract()?,
    };
    let logprobs = get(intern!(py, "logprobs"))?;
    let sampling_masks = get(intern!(py, "sampling_masks"))?;
    let pooler_output = get(intern!(py, "pooler_output"))?;
    let pooler_flags = if pooler_output.is_none() {
        Vec::new()
    } else {
        pooler_output.try_iter()?.map(|p| Ok(!p?.is_none())).collect::<PyResult<_>>()?
    };
    let num_nans = get(intern!(py, "num_nans_in_logits"))?;
    let num_nans_in_logits = if num_nans.is_none() {
        None
    } else {
        Some(num_nans.extract::<FxHashMap<String, usize>>()?)
    };
    let kv_connector_output = get(intern!(py, "kv_connector_output"))?;
    let ec_connector_output = get(intern!(py, "ec_connector_output"))?;
    let aux_output = get(intern!(py, "aux_output_connector_output"))?;
    Ok(ModelOutput {
        rust: ModelRunnerOutput {
            req_ids: get(intern!(py, "req_ids"))?.extract()?,
            sampled_token_ids,
            has_logprobs: !logprobs.is_none(),
            has_sampling_masks: !sampling_masks.is_none(),
            pooler_output: pooler_flags,
            num_nans_in_logits,
            kv_connector_output: if kv_connector_output.is_none() {
                None
            } else {
                Some(kv_connector_output_from_py(&kv_connector_output)?)
            },
            ec_connector_output: ec_connector_output
                .is_truthy()?
                .then(|| opaque(ec_connector_output.clone())),
            aux_output_connector_output: (!aux_output.is_none()).then(|| opaque(aux_output)),
        },
        logprobs,
        sampling_masks,
        pooler_output,
        prompt_logprobs: non_empty_dict(get(intern!(py, "prompt_logprobs_dict"))?)?,
        prompt_token_id_logprobs: non_empty_dict(get(intern!(
            py,
            "prompt_token_id_logprobs_dict"
        ))?)?,
        kv_connector_output,
        ec_connector_output,
        cudagraph_stats: get(intern!(py, "cudagraph_stats"))?,
    })
}

pub(super) fn kv_transfer_params_to_py<'py>(
    py: Python<'py>,
    params: &KvTransferParams,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = opaque_to_py(py, &params.params)?;
    if let Some(n) = params.remote_prefill_cached_tokens {
        dict.set_item(intern!(py, "remote_prefill_cached_tokens"), n)?;
    }
    Ok(dict)
}

pub(super) fn prefill_stats_to_py<'py>(
    py: Python<'py>,
    types: &PyTypes,
    s: &PrefillStats,
) -> PyResult<Bound<'py, PyAny>> {
    types.prefill_stats.bind(py).call1((
        s.num_prompt_tokens,
        s.num_computed_tokens,
        s.num_cached_tokens,
        s.num_local_cached_tokens,
        s.num_external_cached_tokens,
        s.num_cache_creation_tokens,
    ))
}

pub(super) fn spec_decode_metrics_to_py<'py>(
    py: Python<'py>,
    types: &PyTypes,
    m: &RequestSpecDecodeMetrics,
) -> PyResult<Bound<'py, PyAny>> {
    types.request_spec_decode_metrics.bind(py).call1((
        m.num_spec_tokens,
        PyList::new(py, &m.histogram)?,
        m.num_draft_tokens,
        PyList::new(py, &m.per_step_accepted)?,
        PyList::new(py, &m.per_step_drafted)?,
    ))
}

/// Position of `EngineCoreOutput` fields, for positional construction.
mod field {
    pub const NEW_LOGPROBS: usize = 2;
    pub const NEW_PROMPT_LOGPROBS_TENSORS: usize = 3;
    pub const POOLING_OUTPUT: usize = 4;
    pub const FINISH_REASON: usize = 5;
    pub const STOP_REASON: usize = 6;
    pub const EVENTS: usize = 7;
    pub const KV_TRANSFER_PARAMS: usize = 8;
    pub const EC_TRANSFER_PARAMS: usize = 9;
    pub const TRACE_HEADERS: usize = 10;
    pub const PREFILL_STATS: usize = 11;
    pub const ROUTED_EXPERTS: usize = 12;
    pub const NUM_NANS_IN_LOGITS: usize = 13;
    pub const NEW_SAMPLING_MASK: usize = 15;
    pub const SPEC_DECODE_METRICS: usize = 16;
    pub const PROMPT_TOKEN_ID_LOGPROBS: usize = 17;
    pub const ERROR: usize = 18;
    pub const WATERMARKING: usize = 19;
    pub const COUNT: usize = 20;
}

fn engine_core_output_to_py<'py>(
    py: Python<'py>,
    types: &PyTypes,
    table: &RequestTable,
    mo: &ModelOutput<'py>,
    o: &EngineCoreOutput,
) -> PyResult<Bound<'py, PyAny>> {
    let entry = table.get(o.request_id.as_str());
    let request_id = match entry {
        Some(e) => e.id.bind(py).clone(),
        None => PyString::new(py, &o.request_id),
    };
    let new_token_ids = PyList::new(py, &o.new_token_ids)?;
    let mut args: [Option<Bound<'py, PyAny>>; field::COUNT] = Default::default();
    args[0] = Some(request_id.into_any());
    args[1] = Some(new_token_ids.into_any());
    let n = o.new_token_ids.len();
    if let Some(i) = o.req_index {
        if o.new_logprobs {
            args[field::NEW_LOGPROBS] =
                Some(mo.logprobs.call_method1(intern!(py, "slice_request"), (i, n))?);
        }
        if let Some(dict) = &mo.prompt_logprobs {
            args[field::NEW_PROMPT_LOGPROBS_TENSORS] =
                dict.get_item(&o.request_id)?.filter(|t| !t.is_none());
        }
        if let Some(dict) = &mo.prompt_token_id_logprobs {
            args[field::PROMPT_TOKEN_ID_LOGPROBS] = dict.get_item(&o.request_id)?;
        }
        if o.pooling_output {
            args[field::POOLING_OUTPUT] = Some(mo.pooler_output.get_item(i)?);
        }
        if o.new_sampling_mask {
            args[field::NEW_SAMPLING_MASK] =
                Some(mo.sampling_masks.call_method1(intern!(py, "slice_request"), (i, n))?);
        }
    }
    if let Some(reason) = o.finish_reason {
        args[field::FINISH_REASON] = Some(types.finish_reason[reason as usize].bind(py).clone());
    }
    args[field::STOP_REASON] = match &o.stop_reason {
        None => None,
        Some(StopReason::TokenId(t)) => Some(t.into_pyobject(py)?.into_any()),
        Some(StopReason::Str(s)) => Some(s.into_pyobject(py)?.into_any()),
    };
    if let Some(events) = &o.events {
        let cls = types.engine_core_event.bind(py);
        let events = events
            .iter()
            .map(|e| {
                let event_type =
                    types.event_type[e.r#type as usize].as_ref().expect("known event type");
                cls.call1((event_type.bind(py), e.timestamp))
            })
            .collect::<PyResult<Vec<_>>>()?;
        args[field::EVENTS] = Some(PyList::new(py, events)?.into_any());
    }
    if let Some(params) = &o.kv_transfer_params {
        args[field::KV_TRANSFER_PARAMS] = Some(kv_transfer_params_to_py(py, params)?);
    }
    if let Some(params) = &o.ec_transfer_params {
        args[field::EC_TRANSFER_PARAMS] = Some(opaque_obj(py, params));
    }
    if let Some(rows) = &o.routed_experts {
        args[field::ROUTED_EXPERTS] = Some(opaque_obj(py, rows));
    }
    args[field::TRACE_HEADERS] =
        entry.and_then(|e| e.trace_headers.as_ref()).map(|h| h.bind(py).clone());
    if let Some(stats) = &o.prefill_stats {
        args[field::PREFILL_STATS] = Some(prefill_stats_to_py(py, types, stats)?);
    }
    if o.num_nans_in_logits != 0 {
        args[field::NUM_NANS_IN_LOGITS] = Some(o.num_nans_in_logits.into_pyobject(py)?.into_any());
    }
    if let Some(metrics) = &o.spec_decode_metrics {
        args[field::SPEC_DECODE_METRICS] = Some(spec_decode_metrics_to_py(py, types, metrics)?);
    }
    if let Some(error) = &o.error {
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "code"), &error.code)?;
        kwargs.set_item(intern!(py, "message"), &error.message)?;
        args[field::ERROR] = Some(types.request_error.bind(py).call((), Some(&kwargs))?);
    }
    if let Some(watermarking) = o.watermarking {
        args[field::WATERMARKING] =
            Some(pyo3::types::PyBool::new(py, watermarking).to_owned().into_any());
    }
    // Pass the fields up to the last one set; the rest keep their defaults.
    let len = args.iter().rposition(Option::is_some).map_or(0, |i| i + 1);
    let none = py.None().into_bound(py);
    let zero = 0i64.into_pyobject(py)?.into_any();
    let args = PyTuple::new(
        py,
        args[..len].iter().enumerate().map(|(i, a)| match a {
            Some(a) => a,
            None if i == field::NUM_NANS_IN_LOGITS => &zero,
            None => &none,
        }),
    )?;
    types.engine_core_output.bind(py).call1(args)
}

fn prefix_cache_stats_to_py<'py>(
    py: Python<'py>,
    types: &PyTypes,
    s: &PrefixCacheStats,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "reset"), s.reset)?;
    kwargs.set_item(intern!(py, "requests"), s.requests)?;
    kwargs.set_item(intern!(py, "queries"), s.queries)?;
    kwargs.set_item(intern!(py, "hits"), s.hits)?;
    kwargs.set_item(intern!(py, "preempted_requests"), s.preempted_requests)?;
    kwargs.set_item(intern!(py, "preempted_queries"), s.preempted_queries)?;
    kwargs.set_item(intern!(py, "preempted_hits"), s.preempted_hits)?;
    types.prefix_cache_stats.bind(py).call((), Some(&kwargs))
}

fn connector_prefix_cache_stats_to_py<'py>(
    py: Python<'py>,
    types: &PyTypes,
    s: &ConnectorPrefixCacheStats,
) -> PyResult<Bound<'py, PyAny>> {
    let stats = prefix_cache_stats_to_py(py, types, &s.stats)?;
    let hits_by_source = PyDict::new(py);
    for (source, &n) in &s.hits_by_source {
        hits_by_source.set_item(types.cache_hit_source.bind(py).call1((source,))?, n)?;
    }
    stats.setattr(intern!(py, "hits_by_source"), hits_by_source)?;
    Ok(stats)
}

fn spec_decoding_stats_to_py<'py>(
    py: Python<'py>,
    types: &PyTypes,
    s: &SpecDecodingStats,
) -> PyResult<Bound<'py, PyAny>> {
    types.spec_decoding_stats.bind(py).call1((
        s.num_spec_tokens,
        s.num_drafts,
        s.num_draft_tokens,
        s.num_accepted_tokens,
        PyList::new(py, &s.num_accepted_tokens_per_pos)?,
        PyList::new(py, &s.num_draft_tokens_per_pos)?,
    ))
}

/// What `make_stats` adds to the Rust stats from Python objects.
#[derive(Default)]
pub struct ExtraStats<'py> {
    /// The aggregated `KVConnectorStats`.
    pub kv_connector_stats: Option<Bound<'py, PyAny>>,
    pub cudagraph_stats: Option<Bound<'py, PyAny>>,
    pub perf_stats: Option<Bound<'py, PyAny>>,
    /// The aggregated `ECConnectorStats`.
    pub ec_connector_stats: Option<Bound<'py, PyAny>>,
}

pub fn scheduler_stats_to_py<'py>(
    py: Python<'py>,
    stats: &SchedulerStats,
    extra: ExtraStats<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let types = types(py)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "num_running_reqs"), stats.num_running_reqs)?;
    kwargs.set_item(intern!(py, "num_waiting_reqs"), stats.num_waiting_reqs)?;
    kwargs.set_item(
        intern!(py, "num_skipped_waiting_reqs"),
        stats.num_skipped_waiting_reqs,
    )?;
    let by_stage = PyDict::new(py);
    for (stage, n) in stats.num_kv_fetch_reqs_by_stage.iter() {
        by_stage.set_item(stage, n)?;
    }
    kwargs.set_item(intern!(py, "num_kv_fetch_reqs_by_stage"), by_stage)?;
    kwargs.set_item(intern!(py, "kv_cache_usage"), stats.kv_cache_usage)?;
    kwargs.set_item(
        intern!(py, "prefix_cache_stats"),
        prefix_cache_stats_to_py(py, types, &stats.prefix_cache_stats)?,
    )?;
    kwargs.set_item(
        intern!(py, "connector_prefix_cache_stats"),
        stats
            .connector_prefix_cache_stats
            .as_ref()
            .map(|s| connector_prefix_cache_stats_to_py(py, types, s))
            .transpose()?,
    )?;
    let eviction_cls = types.kv_cache_eviction_event.bind(py);
    let eviction_events = stats
        .kv_cache_eviction_events
        .iter()
        .map(|e| {
            eviction_cls.call1((
                e.lifetime_seconds,
                e.idle_seconds,
                PyTuple::new(py, &e.reuse_gaps_seconds)?,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    kwargs.set_item(
        intern!(py, "kv_cache_eviction_events"),
        PyList::new(py, eviction_events)?,
    )?;
    kwargs.set_item(
        intern!(py, "spec_decoding_stats"),
        stats
            .spec_decoding_stats
            .as_ref()
            .map(|s| spec_decoding_stats_to_py(py, types, s))
            .transpose()?,
    )?;
    let connector_stats = extra
        .kv_connector_stats
        .map(|s| s.call_method0(intern!(py, "to_dict")))
        .transpose()?;
    kwargs.set_item(intern!(py, "kv_connector_stats"), connector_stats)?;
    kwargs.set_item(intern!(py, "cudagraph_stats"), extra.cudagraph_stats)?;
    kwargs.set_item(intern!(py, "perf_stats"), extra.perf_stats)?;
    let ec_connector_stats = match extra.ec_connector_stats {
        Some(stats) if stats.is_truthy()? => Some(stats.getattr(intern!(py, "data"))?),
        _ => None,
    };
    kwargs.set_item(intern!(py, "ec_connector_stats"), ec_connector_stats)?;
    types.scheduler_stats.bind(py).call((), Some(&kwargs))
}

/// Build `dict[int, EngineCoreOutputs]` from the scheduler's update.
pub fn engine_core_outputs_to_py<'py>(
    py: Python<'py>,
    update: &UpdateOutput,
    table: &RequestTable,
    mo: &ModelOutput<'py>,
    mut extra: ExtraStats<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let types = types(py)?;
    let out = PyDict::new(py);
    for (&client_index, eco) in &update.engine_core_outputs {
        let outputs = PyList::new(
            py,
            eco.outputs
                .iter()
                .map(|o| engine_core_output_to_py(py, types, table, mo, o))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "outputs"), outputs)?;
        if let Some(finished) = &eco.finished_requests {
            kwargs.set_item(intern!(py, "finished_requests"), str_set(py, finished)?)?;
        }
        if let Some(stats) = &eco.scheduler_stats {
            kwargs.set_item(
                intern!(py, "scheduler_stats"),
                scheduler_stats_to_py(py, stats, std::mem::take(&mut extra))?,
            )?;
        }
        out.set_item(
            client_index,
            types.engine_core_outputs.bind(py).call((), Some(&kwargs))?,
        )?;
    }
    Ok(out)
}
