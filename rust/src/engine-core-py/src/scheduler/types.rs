// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The Python classes and enum members the scheduler binding builds.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyModule;

pub struct PyTypes {
    /// `RequestStatus` members by value.
    pub request_status: Vec<Option<Py<PyAny>>>,
    /// `PauseState` members by value.
    pub pause_state: Vec<Py<PyAny>>,
    pub scheduler_output: Py<PyAny>,
    pub new_request_data: Py<PyAny>,
    pub cached_request_data: Py<PyAny>,
    pub iteration_details: Py<PyAny>,
    pub kv_connector_block_state: Py<PyAny>,
    pub kv_cache_block_copy: Py<PyAny>,
    pub grammar_output: Py<PyAny>,
    pub prefill_stats: Py<PyAny>,
    pub scheduler_stats: Py<PyAny>,
    pub prefix_cache_stats: Py<PyAny>,
    pub spec_decoding_stats: Py<PyAny>,
    pub request_spec_decode_metrics: Py<PyAny>,
    pub kv_cache_eviction_event: Py<PyAny>,
    pub cache_hit_source: Py<PyAny>,
    pub kv_event_batch: Py<PyAny>,
    pub strip_covered_mm_data: Py<PyAny>,
    pub time: Py<PyAny>,
    /// `collections.deque`.
    pub deque: Py<PyAny>,
    /// `StreamingUpdate.from_request`.
    pub streaming_update_from_request: Py<PyAny>,
}

static TYPES: PyOnceLock<PyTypes> = PyOnceLock::new();

fn members(cls: &Bound<'_, PyAny>, names: &[&str]) -> PyResult<Vec<Option<Py<PyAny>>>> {
    let mut out: Vec<Option<Py<PyAny>>> = Vec::new();
    for name in names {
        let member = cls.getattr(*name)?;
        let value: usize = member.getattr("value")?.extract()?;
        if out.len() <= value {
            out.resize_with(value + 1, || None);
        }
        out[value] = Some(member.unbind());
    }
    Ok(out)
}

fn dense(members: Vec<Option<Py<PyAny>>>) -> Vec<Py<PyAny>> {
    members.into_iter().map(|m| m.expect("enum values are dense")).collect()
}

pub fn types(py: Python<'_>) -> PyResult<&'static PyTypes> {
    TYPES.get_or_try_init(py, || {
        let attr = |module: &str, name: &str| -> PyResult<Py<PyAny>> {
            Ok(PyModule::import(py, module)?.getattr(name)?.unbind())
        };
        let output = "vllm.v1.core.sched.output";
        let stats = "vllm.v1.metrics.stats";
        let request_status = members(
            attr("vllm.v1.request", "RequestStatus")?.bind(py),
            &[
                "WAITING",
                "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR",
                "WAITING_FOR_REMOTE_KVS",
                "WAITING_FOR_STREAMING_REQ",
                "RUNNING",
                "PREEMPTED",
                "FINISHED_STOPPED",
                "FINISHED_LENGTH_CAPPED",
                "FINISHED_ABORTED",
                "FINISHED_IGNORED",
                "FINISHED_ERROR",
                "FINISHED_REPETITION",
            ],
        )?;
        let pause_state = dense(members(
            attr("vllm.v1.core.sched.interface", "PauseState")?.bind(py),
            &["UNPAUSED", "PAUSED_NEW", "PAUSED_ALL"],
        )?);
        Ok(PyTypes {
            request_status,
            pause_state,
            scheduler_output: attr(output, "SchedulerOutput")?,
            new_request_data: attr(output, "NewRequestData")?,
            cached_request_data: attr(output, "CachedRequestData")?,
            iteration_details: attr(output, "IterationDetails")?,
            kv_connector_block_state: attr(output, "KVConnectorBlockState")?,
            kv_cache_block_copy: attr("vllm.v1.core.kv_cache_utils", "KVCacheBlockCopy")?,
            grammar_output: attr(output, "GrammarOutput")?,
            prefill_stats: attr(stats, "PrefillStats")?,
            scheduler_stats: attr(stats, "SchedulerStats")?,
            prefix_cache_stats: attr(stats, "PrefixCacheStats")?,
            spec_decoding_stats: attr("vllm.v1.spec_decode.metrics", "SpecDecodingStats")?,
            request_spec_decode_metrics: attr(stats, "RequestSpecDecodeMetrics")?,
            kv_cache_eviction_event: attr(stats, "KVCacheEvictionEvent")?,
            cache_hit_source: attr("vllm.v1.metrics.cache_hit_source", "CacheHitSource")?,
            kv_event_batch: attr("vllm.distributed.kv_events", "KVEventBatch")?,
            strip_covered_mm_data: attr("vllm.multimodal.utils", "strip_covered_mm_data")?,
            time: PyModule::import(py, "time")?.into_any().unbind(),
            deque: attr("collections", "deque")?,
            streaming_update_from_request: attr("vllm.v1.request", "StreamingUpdate")?
                .bind(py)
                .getattr("from_request")?
                .unbind(),
        })
    })
}

impl PyTypes {
    pub fn status<'py>(
        &self,
        py: Python<'py>,
        status: vllm_engine_core::sched::request::RequestStatus,
    ) -> Bound<'py, PyAny> {
        self.request_status[status as usize]
            .as_ref()
            .expect("all statuses are known")
            .bind(py)
            .clone()
    }
}
