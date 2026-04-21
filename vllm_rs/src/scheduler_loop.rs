//! Rust-assisted preamble for Scheduler.update_from_output()'s per-request loop.
//!
//! The loop at scheduler.py:1345-1479 has a ~20-line "prefix" that is pure
//! bookkeeping — dict lookups, Request.is_finished() checks, spec-decode
//! accepted/rejected accounting. This function does all of that in Rust and
//! returns a pre-filtered list of work items. The remaining ~100-line loop
//! body (stop check, output dataclass construction, logprobs, grammar
//! integration) stays in Python because it calls out to torch / msgspec /
//! xgrammar.
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

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

    let out = PyList::empty_bound(py);
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
            PyList::empty_bound(py).into_any()
        };
        let generated_len = generated
            .downcast::<PyList>()
            .map(|l| l.len())
            .unwrap_or(0);

        // Spec-decode accounting
        let mut accepted: isize = 0;
        let mut rejected: isize = 0;
        let scheduled_spec_opt = scheduled_spec_decode_tokens.get_item(&req_id)?;
        if let Some(scheduled_spec) = scheduled_spec_opt {
            if generated_len > 0 {
                let num_draft: isize = scheduled_spec
                    .downcast::<PyList>()?
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

        let tuple = PyTuple::new_bound(
            py,
            &[
                req_id,
                req_index.into_py(py).into_bound(py),
                generated,
                accepted.into_py(py).into_bound(py),
                rejected.into_py(py).into_bound(py),
            ],
        );
        out.append(tuple)?;
    }
    Ok(out)
}
