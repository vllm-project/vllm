// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `KVCacheManager`: drop-in replacement of
//! `vllm.v1.core.kv_cache_manager.KVCacheManager` backed by
//! [`vllm_engine_core::kv_cache::KvCacheManager`].

use pyo3::exceptions::{PyAssertionError, PyNotImplementedError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple, PyType};
use vllm_engine_core::kv_cache::{
    AllocateSlotsArgs, BlockId, KvCacheBlocks, KvCacheManager, KvCacheManagerConfig,
};

use super::config::{convert_config, to_py_err};
use super::events::events_to_py;
use super::request::{BlockHashCache, PyRequest, request_id};
use super::views::{
    BlockPoolView, CoordinatorView, KVCacheBlocksView, Owner, block_ids_of, block_list,
    group_tuple, grouped_block_ids,
};

/// Instances have a `__dict__` so that callers (tests) can patch methods.
#[pyclass(module = "vllm._rust_core", dict)]
pub struct KVCacheManager {
    pub(super) inner: KvCacheManager,
    hashes: BlockHashCache,
    pub(super) kv_cache_config: Py<PyAny>,
    /// The scheduler-side Python spec of each group.
    pub(super) specs: Vec<Py<PyAny>>,
    #[pyo3(get)]
    kv_cache_event_metadata: Py<PyTuple>,
    /// The Python `PrefixCacheStats` being filled, if `log_stats`. Kept in
    /// Python since callers hold on to it and read it live.
    #[pyo3(get)]
    prefix_cache_stats: Option<Py<PyAny>>,
    /// The shared all-empty `KVCacheBlocks` (Python's `empty_kv_cache_blocks`).
    empty_blocks: Py<KVCacheBlocksView>,
}

static PREFIX_CACHE_STATS: PyOnceLock<Py<PyType>> = PyOnceLock::new();

fn new_prefix_cache_stats(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let cls = PREFIX_CACHE_STATS.import(py, "vllm.v1.metrics.stats", "PrefixCacheStats")?;
    Ok(cls.call0()?.unbind())
}

impl KVCacheManager {
    /// Wrap `blocks`, sharing one instance for the all-empty result as
    /// Python does (which also saves an allocation per decode step).
    fn blocks_view(
        slf: &Bound<'_, Self>,
        blocks: KvCacheBlocks,
    ) -> PyResult<Py<KVCacheBlocksView>> {
        let py = slf.py();
        if blocks.is_empty() {
            return Ok(slf.borrow().empty_blocks.clone_ref(py));
        }
        Py::new(
            py,
            KVCacheBlocksView::new(Some(Owner::Manager(slf.clone().unbind())), blocks),
        )
    }
}

/// Sync the request's block hashes and run `f` with the bridged request.
fn with_request<'py, R>(
    slf: &Bound<'py, KVCacheManager>,
    request: &Bound<'py, PyAny>,
    f: impl FnOnce(&mut KvCacheManager, &PyRequest<'_, 'py>) -> R,
) -> PyResult<R> {
    let request_id = request_id(request)?;
    let request_id = request_id.to_str()?;
    let block_hashes = request.getattr(pyo3::intern!(slf.py(), "block_hashes"))?;
    let mut this = slf.borrow_mut();
    let this = &mut *this;
    let hashes = this.hashes.sync(request_id, &block_hashes)?;
    let req = PyRequest::new(request, request_id, hashes);
    Ok(f(&mut this.inner, &req))
}

#[pymethods]
impl KVCacheManager {
    #[new]
    #[pyo3(signature = (
        kv_cache_config,
        max_model_len,
        scheduler_block_size,
        hash_block_size,
        max_in_flight_tokens=None,
        enable_caching=true,
        use_eagle=false,
        num_hash_lookahead_tokens=0,
        num_prefill_lookahead=0,
        log_stats=false,
        enable_kv_cache_events=false,
        dcp_world_size=1,
        pcp_world_size=1,
        metrics_collector=None,
        watermark=0.0,
        enable_mamba_shared_prefix_checkpoint=false,
    ))]
    fn new(
        kv_cache_config: &Bound<'_, PyAny>,
        max_model_len: usize,
        scheduler_block_size: usize,
        hash_block_size: usize,
        max_in_flight_tokens: Option<usize>,
        enable_caching: bool,
        use_eagle: bool,
        num_hash_lookahead_tokens: usize,
        num_prefill_lookahead: usize,
        log_stats: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: usize,
        pcp_world_size: usize,
        metrics_collector: Option<&Bound<'_, PyAny>>,
        watermark: f64,
        enable_mamba_shared_prefix_checkpoint: bool,
    ) -> PyResult<Self> {
        let py = kv_cache_config.py();
        // TODO: feed block lifecycle metrics to the Python collector.
        if metrics_collector.is_some_and(|c| !c.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "KV cache metrics collection is not supported",
            ));
        }
        let (config, specs) = convert_config(kv_cache_config)?;
        let kv_events_use_int_block_hashes: bool = PyModule::import(py, "vllm.envs")?
            .getattr("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES")?
            .extract()?;
        let manager_config = KvCacheManagerConfig {
            max_model_len,
            scheduler_block_size,
            hash_block_size,
            max_in_flight_tokens,
            enable_caching,
            use_eagle,
            num_hash_lookahead_tokens,
            num_prefill_lookahead,
            log_stats,
            enable_kv_cache_events,
            kv_events_use_int_block_hashes,
            dcp_world_size,
            pcp_world_size,
            metrics_collector: None,
            watermark,
            enable_mamba_shared_prefix_checkpoint,
        };
        // Construction asserts the config like Python does; report them as the
        // `AssertionError` Python raises.
        let inner = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            KvCacheManager::new(config, manager_config)
        }))
        .map_err(|panic| {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_default();
            PyAssertionError::new_err(message)
        })?
        .map_err(to_py_err)?;
        let metadata = inner
            .kv_cache_config
            .kv_cache_groups
            .iter()
            .map(|g| (g.kv_cache_spec.kind_str(), g.kv_cache_spec.sliding_window()))
            .collect::<Vec<_>>();
        let empty_blocks = inner.empty_kv_cache_blocks();
        Ok(Self {
            inner,
            hashes: BlockHashCache::default(),
            kv_cache_config: kv_cache_config.clone().unbind(),
            specs: specs.into_iter().map(Bound::unbind).collect(),
            kv_cache_event_metadata: PyTuple::new(py, metadata)?.unbind(),
            prefix_cache_stats: log_stats.then(|| new_prefix_cache_stats(py)).transpose()?,
            empty_blocks: Py::new(py, KVCacheBlocksView::new(None, empty_blocks))?,
        })
    }

    // ---- Attributes ----

    #[getter]
    fn max_model_len(&self) -> usize {
        self.inner.max_model_len
    }

    #[getter]
    fn enable_caching(&self) -> bool {
        self.inner.enable_caching
    }

    #[getter]
    fn enable_kv_cache_events(&self) -> bool {
        self.inner.enable_kv_cache_events
    }

    #[getter]
    fn use_eagle(&self) -> bool {
        self.inner.use_eagle
    }

    #[getter]
    fn log_stats(&self) -> bool {
        self.inner.log_stats
    }

    #[getter]
    fn metrics_collector(&self) -> Option<Py<PyAny>> {
        None
    }

    #[getter]
    fn mamba_shared_prefix_checkpoint(&self) -> bool {
        self.inner.mamba_shared_prefix_checkpoint
    }

    #[getter]
    fn num_kv_cache_groups(&self) -> usize {
        self.inner.num_kv_cache_groups
    }

    #[getter]
    fn retained_hit_group_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.retained_hit_group_ids)
    }

    #[getter]
    fn kv_cache_config(&self, py: Python<'_>) -> Py<PyAny> {
        self.kv_cache_config.clone_ref(py)
    }

    #[getter]
    fn watermark_blocks(&self) -> usize {
        self.inner.watermark_blocks
    }

    #[getter]
    fn coordinator(slf: &Bound<'_, Self>) -> CoordinatorView {
        CoordinatorView::new(Owner::Manager(slf.clone().unbind()))
    }

    #[getter]
    fn block_pool(slf: &Bound<'_, Self>) -> BlockPoolView {
        BlockPoolView::new(Owner::Manager(slf.clone().unbind()))
    }

    /// Pre-constructed KVCacheBlocks with no blocks.
    #[getter]
    fn empty_kv_cache_blocks(&self, py: Python<'_>) -> Py<KVCacheBlocksView> {
        self.empty_blocks.clone_ref(py)
    }

    /// Get the KV cache usage (between 0.0 and 1.0).
    #[getter]
    fn usage(&self) -> f64 {
        self.inner.usage()
    }

    // ---- Methods ----

    /// Get (and reset) the prefix cache stats, or None if logging is disabled.
    fn make_prefix_cache_stats(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if !self.inner.log_stats {
            return Ok(None);
        }
        Ok(self.prefix_cache_stats.replace(new_prefix_cache_stats(py)?))
    }

    /// Whether a local prefix cache lookup may be run for this request.
    fn prefix_cache_lookup_enabled(&self, request: &Bound<'_, PyAny>) -> PyResult<bool> {
        let limit: Option<usize> = request.getattr("prefix_cache_hit_limit")?.extract()?;
        Ok(self.inner.enable_caching && limit != Some(0))
    }

    /// How many of a request's tokens are already in the prefix cache.
    fn get_num_cached_tokens(slf: &Bound<'_, Self>, request: &Bound<'_, PyAny>) -> PyResult<usize> {
        with_request(slf, request, |m, req| m.get_num_cached_tokens(req))
    }

    fn record_prefix_cache_stats(
        &self,
        request: &Bound<'_, PyAny>,
        num_hits: usize,
    ) -> PyResult<()> {
        // Don't count a request that skipped the cache lookup.
        let Some(stats) = &self.prefix_cache_stats else {
            return Ok(());
        };
        if !self.prefix_cache_lookup_enabled(request)? {
            return Ok(());
        }
        let py = request.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("num_tokens", request.getattr("num_tokens")?)?;
        kwargs.set_item("num_hits", num_hits)?;
        let num_preemptions: usize = request.getattr("num_preemptions")?.extract()?;
        kwargs.set_item("preempted", num_preemptions > 0)?;
        stats.bind(py).call_method("record", (), Some(&kwargs))?;
        Ok(())
    }

    /// Get the computed (cached) blocks for the request.
    #[pyo3(signature = (request, max_cache_hit_length=None))]
    fn get_computed_blocks(
        slf: &Bound<'_, Self>,
        request: &Bound<'_, PyAny>,
        max_cache_hit_length: Option<usize>,
    ) -> PyResult<(Py<KVCacheBlocksView>, usize, usize)> {
        let (blocks, n, boundary) = with_request(slf, request, |m, req| {
            m.get_computed_blocks(req, max_cache_hit_length)
        })?;
        Ok((Self::blocks_view(slf, blocks)?, n, boundary))
    }

    /// Local prefix-cache lookup for a request scheduled with a KV connector.
    #[pyo3(signature = (request, max_cache_hit_length=None))]
    fn get_computed_blocks_for_connector(
        slf: &Bound<'_, Self>,
        request: &Bound<'_, PyAny>,
        max_cache_hit_length: Option<usize>,
    ) -> PyResult<(Py<KVCacheBlocksView>, usize, usize, bool)> {
        let (blocks, n, boundary, diverged) = with_request(slf, request, |m, req| {
            m.get_computed_blocks_for_connector(req, max_cache_hit_length)
        })?;
        Ok((Self::blocks_view(slf, blocks)?, n, boundary, diverged))
    }

    /// Add slots for a request with new tokens to append.
    #[pyo3(signature = (
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        num_external_computed_tokens=0,
        delay_cache_blocks=false,
        num_encoder_tokens=0,
        full_sequence_must_fit=false,
        reserved_blocks=0,
        has_scheduled_reqs=true,
        skip_zeroing_group_ids=Vec::new(),
    ))]
    fn allocate_slots(
        slf: &Bound<'_, Self>,
        request: &Bound<'_, PyAny>,
        num_new_tokens: usize,
        num_new_computed_tokens: usize,
        new_computed_blocks: Option<&Bound<'_, PyAny>>,
        num_lookahead_tokens: usize,
        num_external_computed_tokens: usize,
        delay_cache_blocks: bool,
        num_encoder_tokens: usize,
        full_sequence_must_fit: bool,
        reserved_blocks: usize,
        has_scheduled_reqs: bool,
        skip_zeroing_group_ids: Vec<usize>,
    ) -> PyResult<Option<Py<KVCacheBlocksView>>> {
        let owned;
        let new_computed_blocks = match new_computed_blocks {
            None => None,
            Some(obj) if obj.is_none() => None,
            Some(obj) => match obj.cast::<KVCacheBlocksView>() {
                Ok(view) => Some(&view.get().blocks),
                Err(_) => {
                    owned = KvCacheBlocks::new(grouped_block_ids(obj)?);
                    Some(&owned)
                }
            },
        };
        let args = AllocateSlotsArgs {
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs: Some(has_scheduled_reqs),
            skip_zeroing_group_ids: &skip_zeroing_group_ids,
        };
        let out = with_request(slf, request, |m, req| {
            m.allocate_slots(req, num_new_tokens, args)
        })?
        .map_err(to_py_err)?;
        out.map(|blocks| Self::blocks_view(slf, blocks)).transpose()
    }

    /// Free the blocks allocated for the request.
    fn free(&mut self, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request_id = request_id(request)?;
        let request_id = request_id.to_str()?;
        let finished = is_finished(request)?;
        self.hashes.remove(request_id);
        self.inner.free(request_id, finished);
        Ok(())
    }

    #[pyo3(signature = (request_id, processed_computed_tokens, num_prompt_tokens=None))]
    fn remove_skipped_blocks(
        &mut self,
        request_id: &str,
        processed_computed_tokens: usize,
        num_prompt_tokens: Option<usize>,
    ) {
        self.inner
            .remove_skipped_blocks(request_id, processed_computed_tokens, num_prompt_tokens);
    }

    /// Pop the request's bookkeeping and return its blocks without
    /// returning them to the block pool.
    fn pop_blocks_for_free<'py>(
        slf: &Bound<'py, Self>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let request_id = request_id(request)?;
        let request_id = request_id.to_str()?;
        let finished = is_finished(request)?;
        let blocks = {
            let mut this = slf.borrow_mut();
            this.hashes.remove(request_id);
            this.inner.pop_blocks_for_free(request_id, finished)
        };
        block_list(slf.py(), &Owner::Manager(slf.clone().unbind()), &blocks)
    }

    /// Evict blocks from the prefix cache by their block IDs.
    fn evict_blocks(&mut self, block_ids: &Bound<'_, PyAny>) -> PyResult<()> {
        let ids = block_ids_of(block_ids)?;
        self.inner.evict_blocks(ids);
        Ok(())
    }

    fn reset_prefix_cache(&mut self, py: Python<'_>) -> PyResult<bool> {
        if !self.inner.reset_prefix_cache() {
            return Ok(false);
        }
        if let Some(stats) = &self.prefix_cache_stats {
            stats.bind(py).setattr("reset", true)?;
        }
        Ok(true)
    }

    /// Take the KV cache events from the block pool.
    fn take_events<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        events_to_py(py, self.inner.take_events())
    }

    /// Get the blocks of a request.
    fn get_blocks(slf: &Bound<'_, Self>, request_id: &str) -> PyResult<Py<KVCacheBlocksView>> {
        let blocks = slf.borrow().inner.get_blocks(request_id);
        Self::blocks_view(slf, blocks)
    }

    /// Get the block ids of a request.
    fn get_block_ids<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
    ) -> PyResult<Bound<'py, PyTuple>> {
        group_tuple(py, self.inner.get_block_tables(request_id))
    }

    /// Get block ids covering the request's computed tokens.
    fn get_block_ids_for_computed_tokens<'py>(
        &self,
        py: Python<'py>,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> PyResult<Bound<'py, PyTuple>> {
        group_tuple(
            py,
            self.inner.get_block_ids_for_computed_tokens(request_id, num_computed_tokens),
        )
    }

    /// Estimate the number of tokens cached by the request.
    fn estimate_cached_tokens(&self, request: &Bound<'_, PyAny>) -> PyResult<usize> {
        let request_id = request_id(request)?;
        let req = PyRequest::new(request, request_id.to_str()?, &[]);
        Ok(self.inner.estimate_cached_tokens(&req))
    }

    /// Cache the blocks for the request, if enabled.
    fn cache_blocks(
        slf: &Bound<'_, Self>,
        request: &Bound<'_, PyAny>,
        num_computed_tokens: usize,
    ) -> PyResult<()> {
        with_request(slf, request, |m, req| {
            m.cache_blocks(req, num_computed_tokens)
        })
    }

    fn create_kv_cache_blocks(
        slf: &Bound<'_, Self>,
        blocks: &Bound<'_, PyAny>,
    ) -> PyResult<Py<KVCacheBlocksView>> {
        let groups = grouped_block_ids(blocks)?;
        let blocks = slf.borrow().inner.create_kv_cache_blocks(groups);
        Self::blocks_view(slf, blocks)
    }

    /// Return a lookup-result view truncated at an aligned token endpoint.
    fn truncate_computed_blocks(
        slf: &Bound<'_, Self>,
        blocks: &Bound<'_, PyAny>,
        num_computed_tokens: usize,
    ) -> PyResult<Py<KVCacheBlocksView>> {
        let blocks = KvCacheBlocks::new(grouped_block_ids(blocks)?);
        let truncated = slf.borrow().inner.truncate_computed_blocks(&blocks, num_computed_tokens);
        Self::blocks_view(slf, truncated)
    }

    /// Drain and return new attention block IDs for zeroing.
    fn take_new_block_ids(&mut self) -> Vec<BlockId> {
        self.inner.take_new_block_ids()
    }

    fn record_blocks_for_zeroing(&mut self, request_id: &str, start_token: usize) {
        self.inner.record_blocks_for_zeroing(request_id, start_token);
    }

    /// Drain pending copies and return their retained endpoints.
    fn take_kv_cache_block_copies<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let py = slf.py();
        let (copies, retained) = slf.borrow_mut().inner.take_kv_cache_block_copies();
        let copy_cls =
            PyModule::import(py, "vllm.v1.core.kv_cache_utils")?.getattr("KVCacheBlockCopy")?;
        let copies = copies
            .iter()
            .map(|c| {
                let kwargs = PyDict::new(py);
                kwargs.set_item("src_block_id", c.src_block_id)?;
                kwargs.set_item("dst_block_id", c.dst_block_id)?;
                copy_cls.call((), Some(&kwargs))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok((
            PyList::new(py, copies)?,
            block_list(py, &Owner::Manager(slf.clone().unbind()), &retained)?,
        ))
    }

    /// Drain this step's boundary-state hand-offs for a KV connector.
    fn take_boundary_state_offloads<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (req_id, entries) in self.inner.take_boundary_state_offloads() {
            out.set_item(req_id, entries)?;
        }
        Ok(out)
    }

    /// Consume safe producer partial tails when a request finishes.
    fn finalize_partial_tail_offloads(
        &mut self,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<(usize, BlockId, usize)>> {
        let request_id = request_id(request)?;
        let req = PyRequest::new(request, request_id.to_str()?, &[]);
        Ok(self.inner.finalize_partial_tail_offloads(&req))
    }

    /// Notify the coordinator that a new step is starting.
    fn new_step_starts(&mut self) {
        self.inner.new_step_starts();
    }

    /// `get_hisparse_coordinator(kv_cache_manager, import_step_tokens=...)`.
    #[pyo3(signature = (import_step_tokens=None))]
    fn hisparse_coordinator(
        slf: &Bound<'_, Self>,
        import_step_tokens: Option<usize>,
    ) -> PyResult<super::hisparse::HiSparseCoordinatorView> {
        super::hisparse::HiSparseCoordinatorView::bind(
            slf.py(),
            Owner::Manager(slf.clone().unbind()),
            import_step_tokens,
        )
    }
}

/// `request.is_finished()`.
fn is_finished(request: &Bound<'_, PyAny>) -> PyResult<bool> {
    request.call_method0(pyo3::intern!(request.py(), "is_finished"))?.extract()
}
