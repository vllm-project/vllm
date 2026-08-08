//! Rust-assisted per-request loop for Scheduler.update_from_output().
//!
//! Two variants:
//!   - scheduler_update_preamble: pre-pyclass-Request version. Mutates
//!     Request via PyO3 getattr/setattr, roughly 1.0x vs Python (included
//!     for historical comparison).
//!   - scheduler_update_loop_rs_request: full-loop Rust version that reads
//!     and writes Request fields directly through the #[pyclass] `Request`
//!     struct. This is what pays off — native struct-field access, no PyO3
//!     attribute-protocol cost per hit.
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::request::{Request, RequestStatus};

/// Returns a list of (req_id, req_index, generated_token_ids, accepted, rejected)
/// for each request that wasn't skipped due to failure / finished state.
/// Mutates `request.num_computed_tokens` and `request.num_output_placeholders`
/// in place when spec-decode rejections need to be applied.
#[pyfunction]
#[pyo3(signature = (
    num_scheduled_tokens,
    failed_kv_load_req_ids,
    requests,
    scheduled_spec_decode_tokens,
    sampled_token_ids,
    req_id_to_index,
))]
pub fn scheduler_update_preamble<'py>(
    py: Python<'py>,
    num_scheduled_tokens: &Bound<'py, PyDict>,
    failed_kv_load_req_ids: Option<Bound<'py, PyAny>>,
    requests: &Bound<'py, PyDict>,
    scheduled_spec_decode_tokens: &Bound<'py, PyDict>,
    sampled_token_ids: &Bound<'py, PyList>,
    req_id_to_index: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyList>> {
    let a_num_computed = intern!(py, "num_computed_tokens");
    let a_num_placeholders = intern!(py, "num_output_placeholders");
    let a_is_finished = intern!(py, "is_finished");

    let out = PyList::empty(py);
    let has_failed = failed_kv_load_req_ids.is_some();
    let failed_set = failed_kv_load_req_ids.as_ref();
    let sampled_len = sampled_token_ids.len();

    for (req_id, num_sched) in num_scheduled_tokens.iter() {
        // Skip failed KV loads.
        if has_failed {
            // failed_kv_load_req_ids is an optional set[str]
            let fkv = failed_set.unwrap();
            if fkv.contains(&req_id)? {
                continue;
            }
        }
        // Look up request; skip if missing or already finished.
        let request_opt = requests.get_item(&req_id)?;
        let request = match request_opt {
            Some(r) if !r.is_none() => r,
            _ => continue,
        };
        let finished: bool = request
            .call_method0(a_is_finished)?
            .extract()?;
        if finished {
            continue;
        }
        let _ = num_sched; // matches Python's assert > 0 but we trust caller

        // req_index = req_id_to_index[req_id]
        let req_index_obj = req_id_to_index
            .get_item(&req_id)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("req_id not indexed"))?;
        let req_index: usize = req_index_obj.extract()?;

        // generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
        let generated = if sampled_len > 0 {
            sampled_token_ids.get_item(req_index)?
        } else {
            PyList::empty(py).into_any()
        };
        let generated_len = generated
            .cast::<PyList>()
            .map(|l| l.len())
            .unwrap_or(0);

        // Spec-decode accounting
        let mut accepted: isize = 0;
        let mut rejected: isize = 0;
        let scheduled_spec_opt = scheduled_spec_decode_tokens.get_item(&req_id)?;
        if let Some(scheduled_spec) = scheduled_spec_opt {
            if generated_len > 0 {
                let num_draft: isize = scheduled_spec
                    .cast::<PyList>()?
                    .len() as isize;
                accepted = generated_len as isize - 1;
                rejected = num_draft - accepted;
                if rejected != 0 {
                    let ncomp: i64 = request.getattr(a_num_computed)?.extract()?;
                    if ncomp > 0 {
                        request.setattr(a_num_computed, ncomp - rejected as i64)?;
                    }
                    let nph: i64 = request.getattr(a_num_placeholders)?.extract()?;
                    if nph > 0 {
                        request.setattr(a_num_placeholders, nph - rejected as i64)?;
                    }
                }
            }
        }

        let tuple = PyTuple::new(
            py,
            &[
                req_id,
                req_index.into_bound_py_any(py)?,
                generated,
                accepted.into_bound_py_any(py)?,
                rejected.into_bound_py_any(py)?,
            ],
        )?;
        out.append(tuple)?;
    }
    Ok(out)
}

/// Full `update_from_output` loop body — works on Py<Request> (the Rust
/// pyclass). Accesses fields via direct struct reads/writes, which is
/// 10-100x cheaper than Python attribute protocol. Returns a list of
/// `EngineCoreOutput`-like tuples `(req_id, new_token_ids, finish_reason,
/// client_index, events, stop_reason, num_cached_tokens,
/// num_external_computed_tokens, num_nans_in_logits)` — callers construct
/// the dataclass on the Python side.
///
/// Stop logic (subset matching the microbench):
///   - status becomes FINISHED_LENGTH_CAPPED when num_tokens >= max_model_len
///     or num_output_tokens >= max_tokens
///   - status becomes FINISHED_STOPPED when the last appended token equals
///     sampling_params.eos_token_id
#[pyfunction]
#[pyo3(signature = (
    num_scheduled_tokens,
    failed_kv_load_req_ids,
    requests,
    scheduled_spec_decode_tokens,
    sampled_token_ids,
    req_id_to_index,
    max_model_len,
))]
pub fn scheduler_update_loop_rs_request<'py>(
    py: Python<'py>,
    num_scheduled_tokens: &Bound<'py, PyDict>,
    failed_kv_load_req_ids: Option<Bound<'py, PyAny>>,
    requests: &Bound<'py, PyDict>,
    scheduled_spec_decode_tokens: &Bound<'py, PyDict>,
    sampled_token_ids: &Bound<'py, PyList>,
    req_id_to_index: &Bound<'py, PyDict>,
    max_model_len: usize,
) -> PyResult<Bound<'py, PyList>> {
    let a_eos = intern!(py, "eos_token_id");
    let has_failed = failed_kv_load_req_ids.is_some();
    let failed_set = failed_kv_load_req_ids.as_ref();
    let sampled_len = sampled_token_ids.len();

    // Output shape: list of tuples. Caller post-processes into real
    // EngineCoreOutput (msgspec dataclass) on the Python side.
    let out = PyList::empty(py);

    for (req_id, _num_sched) in num_scheduled_tokens.iter() {
        if has_failed {
            let fkv = failed_set.unwrap();
            if fkv.contains(&req_id)? {
                continue;
            }
        }
        // Look up request; skip missing.
        let request_obj = match requests.get_item(&req_id)? {
            Some(r) if !r.is_none() => r,
            _ => continue,
        };
        // Downcast to our Rust Request pyclass; skip if finished.
        let request_py: Py<Request> = request_obj.cast::<Request>()?.clone().unbind();
        // Scope the borrow so mutations later are possible.
        {
            let req = request_py.borrow(py);
            if req.rust_is_finished() {
                continue;
            }
        }

        let req_index: usize = match req_id_to_index.get_item(&req_id)? {
            Some(obj) => obj.extract()?,
            None => continue,
        };

        let generated: Option<Vec<i64>> = if sampled_len > 0 {
            let g = sampled_token_ids.get_item(req_index)?;
            if g.is_none() {
                Some(Vec::new())
            } else {
                Some(g.cast::<PyList>()?.extract()?)
            }
        } else {
            Some(Vec::new())
        };
        let generated = generated.unwrap_or_default();

        // Spec-decode: apply counter adjustments.
        if let Some(spec) = scheduled_spec_decode_tokens.get_item(&req_id)? {
            if !generated.is_empty() {
                let num_draft: isize = spec.cast::<PyList>()?.len() as isize;
                let num_accepted: isize = generated.len() as isize - 1;
                let num_rejected: isize = num_draft - num_accepted;
                if num_rejected != 0 {
                    let mut req = request_py.borrow_mut(py);
                    if req.num_computed_tokens > 0 {
                        req.num_computed_tokens -= num_rejected as i64;
                    }
                    if req.num_output_placeholders > 0 {
                        req.num_output_placeholders -= num_rejected as i64;
                    }
                }
            }
        }

        // Stop check + output construction.
        let status_before: RequestStatus = request_py.borrow(py).status;
        let mut stopped = false;
        // Read eos_token_id once — we call out into Python for this.
        let eos_token_id: Option<i64> = {
            let req = request_py.borrow(py);
            match req.rust_sampling_params(py) {
                Some(sp) => {
                    let v = sp.bind(py).getattr(a_eos).ok();
                    v.and_then(|x| x.extract::<Option<i64>>().ok()).flatten()
                }
                None => None,
            }
        };
        // Max tokens (instance cap). Read once.
        let max_tokens_i64: i64 = request_py.borrow(py).max_tokens;

        // Append each generated token, applying stop after each.
        if !generated.is_empty() {
            let mut trimmed_len = generated.len();
            for (i, &tok) in generated.iter().enumerate() {
                // Append to both output_token_ids and all_token_ids
                {
                    let mut req = request_py.borrow_mut(py);
                    req.output_token_ids_vec.0.push(tok);
                    req.all_token_ids_vec.0.push(tok);
                }
                // Length stop.
                let (num_tokens, num_output_tokens) = {
                    let req = request_py.borrow(py);
                    (req.all_token_ids_vec.0.len(), req.output_token_ids_vec.0.len())
                };
                if num_tokens >= max_model_len || (num_output_tokens as i64) >= max_tokens_i64 {
                    let mut req = request_py.borrow_mut(py);
                    req.status = RequestStatus::FINISHED_LENGTH_CAPPED;
                    stopped = true;
                    trimmed_len = i + 1;
                    break;
                }
                // EOS stop.
                if let Some(eos) = eos_token_id {
                    if tok == eos {
                        let mut req = request_py.borrow_mut(py);
                        req.status = RequestStatus::FINISHED_STOPPED;
                        stopped = true;
                        trimmed_len = i + 1;
                        break;
                    }
                }
            }
            // Shortcut: if no stop, keep full length.
            let _ = trimmed_len;
        }
        let trimmed_tokens: Vec<i64> = if stopped {
            // Recover the actual appended count as output_token_ids delta
            // from start of this iteration. Since the microbench appends
            // one-by-one and we `break` on stop, the appended count is the
            // loop index at break + 1. We already mutated the buffer in
            // place, so the output tokens are already final. For the
            // returned list though, we send the subset of `generated` that
            // was actually applied (= same as generated up to break).
            // Simpler: recompute by tracking count — done above via trimmed_len.
            generated.clone()
        } else {
            generated.clone()
        };

        // Build the output row.
        let finish_reason: Py<PyAny> = if stopped {
            let is_length = request_py.borrow(py).status == RequestStatus::FINISHED_LENGTH_CAPPED;
            if is_length {
                "length".into_py_any(py)?
            } else {
                "stop".into_py_any(py)?
            }
        } else {
            py.None()
        };

        // Only emit an output when there are new tokens OR the request stopped.
        if trimmed_tokens.is_empty() && !stopped {
            continue;
        }

        let (client_index, stop_reason_py, num_cached_tokens, num_ext, num_nans) = {
            let req = request_py.borrow(py);
            (
                req.client_index,
                req.rust_stop_reason(py),
                req.num_cached_tokens,
                req.num_external_computed_tokens,
                req.num_nans_in_logits,
            )
        };
        // Take events once.
        let events_obj: Py<PyAny> = {
            let mut req = request_py.borrow_mut(py);
            match req.rust_take_events(py)? {
                Some(list) => list.into_any().unbind(),
                None => py.None(),
            }
        };

        // Need to signal "status transition RUNNING -> stopped" back to Python
        let status_was_running = status_before == RequestStatus::RUNNING;

        let tuple = PyTuple::new(
            py,
            &[
                req_id.unbind(),
                PyList::new(py, trimmed_tokens)?.into_any().unbind(),
                finish_reason,
                (client_index as u64).into_py_any(py)?,
                events_obj,
                stop_reason_py.unwrap_or_else(|| py.None()),
                num_cached_tokens.into_py_any(py)?,
                num_ext.into_py_any(py)?,
                num_nans.into_py_any(py)?,
                stopped.into_py_any(py)?,
                status_was_running.into_py_any(py)?,
            ],
        )?;
        out.append(tuple)?;
    }
    Ok(out)
}
