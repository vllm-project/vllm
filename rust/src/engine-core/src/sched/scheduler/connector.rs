// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! KV connector support: the scheduler-side connector interface and the
//! "KV Connector Related Methods" of `Scheduler`.

use indexmap::{IndexMap, IndexSet};
use rustc_hash::FxHashSet;

use super::RequestStore;
use super::Scheduler;
use crate::kv_cache::{BlockId, BoundaryStateOffload, KvCacheBlocks, KvCacheManager};
use crate::sched::output::{Opaque, SchedulerOutput};
use crate::sched::py_int_set::PyIntSet;
use crate::sched::request::{Request, RequestStatus};
use crate::sched::stats::KvFetchStage;

/// The `kv_transfer_params` a connector returns from `request_finished`.
#[derive(Clone, Debug, PartialEq)]
pub struct KvTransferParams {
    /// The params dict.
    pub params: Opaque,
    /// `params.get("do_remote_prefill")` is truthy.
    pub do_remote_prefill: bool,
    /// To set as `params["remote_prefill_cached_tokens"]`: P-side cache hits,
    /// so D can report them in prompt_tokens_details instead of its own
    /// (~100%) hit rate from the KV transfer.
    pub remote_prefill_cached_tokens: Option<usize>,
}

/// What a connector's `request_finished` returns.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RequestFinished {
    /// Whether the connector frees the request's blocks asynchronously: they
    /// stay allocated until the request shows up in `finished_sending` /
    /// `finished_recving`.
    pub delay_free_blocks: bool,
    pub kv_transfer_params: Option<KvTransferParams>,
}

/// `KVConnectorOutput` as the scheduler reads it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KvConnectorOutput {
    pub finished_sending: Option<Vec<String>>,
    pub finished_recving: Option<Vec<String>>,
    /// Blocks that failed to load, in the Python set's iteration order.
    pub invalid_block_ids: Vec<BlockId>,
    pub failed_recving: Vec<String>,
    /// The Python object, for `update_connector_output`.
    pub output: Option<Opaque>,
}

/// `KVConnectorBlockState`: connector-only block state handed to
/// `build_connector_meta`, never dispatched to workers.
pub struct KvConnectorBlockState<'a> {
    /// Requests that can become connector jobs this step.
    pub req_ids: IndexSet<String>,
    /// Resolves block ids (`kv_cache_manager.get_block_ids`); see
    /// [`KvConnector`] for why it is mutable.
    pub kv_cache_manager: &'a mut KvCacheManager,
    pub boundary_state_offloads: IndexMap<String, Vec<BoundaryStateOffload>>,
    /// The scheduler's requests (`Scheduler.requests`), e.g. for the binding
    /// layer to hand the connector up-to-date Python requests.
    pub requests: Requests<'a>,
}

/// Read-only access to the scheduler's requests by id.
#[derive(Clone, Copy)]
pub struct Requests<'a>(pub(super) &'a RequestStore);

impl<'a> Requests<'a> {
    /// `requests.get(request_id)`.
    pub fn get(&self, request_id: &str) -> Option<&'a Request> {
        self.0.handle(request_id).map(|h| &self.0[h])
    }
}

/// The scheduler-side `KVConnectorBase_V1` methods the scheduler calls.
///
/// Implementations wrapping a Python connector sync the Python request from
/// the [`Request`] passed in before calling it.
///
/// Python connectors hold the scheduler's KV cache manager
/// (`bind_kv_cache_manager`) and may change the block pool from their hooks,
/// e.g. pin blocks of in-flight saves with `block_pool.touch` and release
/// them with `block_pool.free_blocks`. Hooks that run such code receive the
/// manager mutably.
///
/// `bind_kv_cache_manager` and `shutdown` are the binding layer's: the
/// scheduler never calls them.
pub trait KvConnector {
    /// Whether the connector restores lagging recurrent state when the local
    /// full-attention group reaches a deeper boundary.
    fn supports_divergent_local_hybrid_hits(&self) -> bool;

    /// Whether a request preempted while its hand-off is still pending is
    /// recomputed rather than allowed to finish and hand off blocks that the
    /// preemption already freed.
    fn requires_kv_delivery(&self) -> bool;

    /// Called when a new request is added to the scheduler.
    fn on_new_request(&mut self, request: &Request, _kv_cache_manager: &mut KvCacheManager);

    /// Whether `get_num_new_matched_tokens` may return `load_kv_async=True`
    /// for this request.
    fn may_load_async(&mut self, request: &Request) -> bool;

    /// The number of new tokens that can be loaded from the external KV
    /// cache beyond `num_computed_tokens` (`None` when not yet known), and
    /// whether they are loaded asynchronously.
    ///
    /// The connector may raise the request's `shared_prefix_boundary` to pin
    /// a shared-prefix junction its lookup found.
    fn get_num_new_matched_tokens(
        &mut self,
        request: &mut Request,
        num_computed_tokens: usize,
        _kv_cache_manager: &mut KvCacheManager,
    ) -> (Option<usize>, bool);

    /// The KV cache groups the connector's load for `request` restores.
    fn get_loaded_kv_cache_group_ids(&mut self, request: &Request) -> Vec<usize>;

    /// Update connector state after block allocation.
    fn update_state_after_alloc(
        &mut self,
        request: &Request,
        blocks: &KvCacheBlocks,
        kv_cache_manager: &mut KvCacheManager,
        num_external_tokens: usize,
    );

    /// `hits` split by the cache tier that supplied them, as
    /// `(CacheHitSource value, hits)` pairs in dict order.
    fn get_external_cache_hit_sources(
        &mut self,
        request: &Request,
        num_external_tokens: usize,
    ) -> Option<Vec<(String, usize)>>;

    fn has_pending_block_frees(&mut self, kv_cache_manager: &mut KvCacheManager) -> bool;

    /// Build the connector metadata for this step.
    fn build_connector_meta(
        &mut self,
        scheduler_output: &SchedulerOutput,
        block_state: &mut KvConnectorBlockState<'_>,
    ) -> Opaque;

    /// Register the boundary states of a finished producer request
    /// (`finalize_partial_tail_offloads`). Returns whether the request's
    /// blocks must stay allocated until the connector releases them.
    fn register_finished_partial_tail(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        partial_tail_offloads: &[BoundaryStateOffload],
        kv_cache_manager: &mut KvCacheManager,
    ) -> bool;

    /// `request_finished_all_groups` for connectors supporting HMA,
    /// `request_finished` with the single group's block ids otherwise.
    fn request_finished(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        kv_cache_manager: &mut KvCacheManager,
    ) -> RequestFinished;

    fn update_connector_output(
        &mut self,
        kv_connector_output: &KvConnectorOutput,
        kv_cache_manager: &mut KvCacheManager,
    );

    /// Scheduler-side connector stats, `None` when absent or empty.
    fn get_kv_connector_stats(&mut self, kv_cache_manager: &mut KvCacheManager) -> Option<Opaque>;

    /// KV cache events collected by the connector, `None` when empty.
    fn take_events(&mut self, _kv_cache_manager: &mut KvCacheManager) -> Option<Opaque>;

    fn has_pending_push_work(&mut self, kv_cache_manager: &mut KvCacheManager) -> bool;

    /// `None` when the connector does not support resetting.
    fn reset_cache(&mut self, _kv_cache_manager: &mut KvCacheManager) -> Option<bool>;
}

impl Scheduler {
    /// Invoke the KV connector request_finished() method if applicable.
    ///
    /// Returns optional kv transfer parameters to be included with the
    /// request outputs.
    pub(super) fn connector_finished(&mut self, h: usize) -> (bool, Option<KvTransferParams>) {
        let Some(connector) = self.connector.as_mut() else {
            return (false, None);
        };
        let request = &self.requests[h];

        let mut finished_partial_tails = Vec::new();
        if self.kv_transfer.is_some_and(|c| c.is_kv_producer) {
            finished_partial_tails = self.kv_cache_manager.finalize_partial_tail_offloads(request);
        }

        // Free any out-of-window prefix blocks before we hand the block table to
        // the connector, on the processed-token basis (see `allocate_slots`).
        self.kv_cache_manager.remove_skipped_blocks(
            &request.request_id,
            request.num_computed_tokens.saturating_sub(request.num_in_flight_tokens),
            Some(request.num_prompt_tokens),
        );

        let block_ids = self
            .kv_cache_manager
            .get_block_ids_for_computed_tokens(&request.request_id, request.num_computed_tokens);
        let mut partial_tail_delay = false;
        if !finished_partial_tails.is_empty() {
            partial_tail_delay = connector.register_finished_partial_tail(
                request,
                &block_ids,
                &finished_partial_tails,
                &mut self.kv_cache_manager,
            );
        }

        // NOTE(Kuntai): connectors without HMA support see a single group;
        // the implementation picks `request_finished` or
        // `request_finished_all_groups`.
        let finished = connector.request_finished(request, &block_ids, &mut self.kv_cache_manager);
        (
            finished.delay_free_blocks || partial_tail_delay,
            finished.kv_transfer_params,
        )
    }

    /// Blocks `request` still needs to hold its full sequence and be promoted.
    pub(super) fn request_remaining_blocks(&mut self, h: usize) -> i64 {
        let request = &self.requests[h];
        let full_num_tokens = request.num_tokens().min(self.max_model_len);
        let mut num_tokens = full_num_tokens;
        if self.num_spec_tokens > 0 {
            num_tokens += 1 + self.num_spec_tokens + self.num_lookahead_tokens;
        }
        self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            &request.request_id,
            num_tokens,
            &[],
            0,
            request.num_computed_tokens,
            request.num_computed_tokens,
            full_num_tokens,
            true,
            request.num_prompt_tokens.max(request.num_tokens().saturating_sub(1)),
        )
    }

    pub(super) fn set_kv_fetch_stage(&mut self, h: usize, stage: Option<KvFetchStage>) {
        if let Some(kv_fetch_stages) = &mut self.kv_fetch_stages {
            if let Some(prev_stage) = kv_fetch_stages.remove(&h) {
                *self.kv_fetch_counts.get_mut(prev_stage) -= 1;
            }
            if let Some(stage) = stage {
                kv_fetch_stages.insert(h, stage);
                *self.kv_fetch_counts.get_mut(stage) += 1;
            }
        }
    }

    /// Num blocks in-flight prefills still need to finish (their reservation).
    pub(super) fn inflight_prefill_reserved_blocks(&mut self) -> i64 {
        let inflight: Vec<usize> = self.inflight_prefills.iter().copied().collect();
        self.parked_reserved_blocks
            + inflight.into_iter().map(|h| self.request_remaining_blocks(h)).sum::<i64>()
    }

    /// Cache a parked async KV load's reservation. Set when parked and refreshed
    /// at promotion, the only points where the load's held blocks change.
    pub(super) fn reserve_parked(&mut self, h: usize) {
        if self.kv_cache_manager.kv_cache_config.hisparse_host_num_blocks.is_some() {
            // HiSparse's depends on free host blocks, so it is summed on demand.
            self.inflight_prefills.insert(h);
            return;
        }
        let blocks = self.request_remaining_blocks(h);
        let cached = self.parked_reservations.get(&h).copied().unwrap_or(0);
        self.parked_reserved_blocks += blocks - cached;
        self.parked_reservations.insert(h, blocks);
    }

    pub(super) fn unreserve_parked(&mut self, h: usize) {
        self.parked_reserved_blocks -= self.parked_reservations.remove(&h).unwrap_or(0);
    }

    /// Whether the KV connector's load for `request` restores every DSV41
    /// SWA bounded-replay group, as a P/D transfer of the request's own blocks
    /// does; a prefix-cache store holds no window.
    pub(super) fn load_restores_replay_window(&mut self, h: usize) -> bool {
        let connector = self.connector.as_mut().expect("KV connector is configured");
        if self.prefix_replay_group_ids.is_empty() {
            return false;
        }
        let loaded = connector.get_loaded_kv_cache_group_ids(&self.requests[h]);
        self.prefix_replay_group_ids.iter().all(|g| loaded.contains(g))
    }

    /// Record where a prefix hit's replay starts and return the number of
    /// hit tokens to recompute; the caller rewinds the computed count by it.
    ///
    /// The replayed tokens are the hit's last window (see
    /// `Request.replay_start`): the worker rebuilds their sliding-window KV
    /// and leaves their cached KV alone. A hit whose KV load carries the
    /// window replays nothing.
    pub(super) fn mark_prefix_replay(
        &mut self,
        h: usize,
        num_hit_tokens: usize,
        window_loaded: bool,
    ) -> usize {
        if self.prefix_replay_tokens == 0 {
            return 0;
        }
        let request = &mut self.requests[h];
        if window_loaded {
            request.replay_start = 0;
            return 0;
        }
        let num_replay_tokens = self.prefix_replay_tokens.min(num_hit_tokens);
        request.replay_start = num_hit_tokens - num_replay_tokens;
        num_replay_tokens
    }

    /// KV Connector: update request state after async recv is finished.
    ///
    /// When the kv transfer is ready, we cache the blocks
    /// and the request state will be moved back to WAITING from
    /// WAITING_FOR_REMOTE_KV.
    fn update_waiting_for_remote_kv(&mut self, h: usize) {
        assert!(self.connector.is_some());
        let request_id = self.requests[h].request_id.clone();

        let load_failed = self.failed_recving_kv_req_ids.contains(&request_id);
        if load_failed {
            // Request had KV load failures; num_computed_tokens was already
            // updated in _update_requests_with_invalid_blocks
            let request = &self.requests[h];
            if request.num_computed_tokens > 0 {
                // Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens);
                if self.needs_kv_cache_zeroing {
                    // The failed load left the blocks beyond the valid
                    // prefix unwritten and their zeroing was skipped; zero
                    // them before they are recomputed locally.
                    self.kv_cache_manager
                        .record_blocks_for_zeroing(&request_id, request.num_computed_tokens);
                }
            } else {
                // No valid computed tokens, release allocated blocks.
                // There may be a local cache hit on retry.
                // (Freed blocks are re-recorded for zeroing when
                // reallocated, so the skipped blocks need no handling.)
                let finished = self.requests[h].is_finished();
                self.kv_cache_manager.free(&request_id, finished);
            }

            self.failed_recving_kv_req_ids.remove(&request_id);
        } else {
            // Now that the blocks are ready, actually cache them.
            // This will cache the blocks iff caching is enabled.
            let request = &self.requests[h];
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens);
        }

        // DSV41 SWA bounded replay recomputes the tail of a hit loaded without
        // its window, or of a failed load, which may have lost it; the replay
        // covers the last token. Otherwise a full prompt hit re-computes that
        // token so the next one can be sampled.
        let window_loaded = !load_failed && self.load_restores_replay_window(h);
        let num_computed_tokens = self.requests[h].num_computed_tokens;
        let num_replay_tokens = self.mark_prefix_replay(h, num_computed_tokens, window_loaded);
        let request = &mut self.requests[h];
        if num_replay_tokens > 0 {
            request.num_computed_tokens -= num_replay_tokens;
        } else if request.num_computed_tokens == request.num_tokens() {
            request.num_computed_tokens = request.num_tokens() - 1;
        }
        assert!(self.finished_recving_kv_req_ids.remove(&request_id));
    }

    /// Returns True if scheduling of this request should proceed, False if
    /// it should be skipped for this step.
    pub(super) fn handle_blocked_waiting_request(&mut self, h: usize) -> bool {
        match self.requests[h].status {
            RequestStatus::WaitingForRemoteKvs => {
                // Check whether async loading requests are finished.
                if !self.finished_recving_kv_req_ids.contains(&self.requests[h].request_id) {
                    return false;
                }
                // Promote request to waiting or preempted.
                self.update_waiting_for_remote_kv(h);
                self.reserve_parked(h);
                let request = &mut self.requests[h];
                request.status = if request.num_preemptions > 0 {
                    RequestStatus::Preempted
                } else {
                    RequestStatus::Waiting
                };
            }
            RequestStatus::WaitingForStructuredOutputGrammar => {
                // Check whether grammar compilation has finished.
                match self.structured_output.grammar_status(&self.requests[h]) {
                    GrammarStatus::Pending => return false,
                    GrammarStatus::Failed { message } => {
                        let request_id = self.requests[h].request_id.clone();
                        self.grammar_compile_error_reqs.insert(
                            request_id,
                            RequestError {
                                code: "validation_error".to_owned(),
                                message: format!("Invalid structured outputs: {message}"),
                            },
                        );
                        return false;
                    }
                    // Grammar is ready, promote to waiting state.
                    GrammarStatus::Ready => self.requests[h].status = RequestStatus::Waiting,
                }
            }
            RequestStatus::WaitingForStreamingReq => {
                // Streaming input request waiting for more input.
                assert!(self.requests[h].streaming_queue.as_ref().is_none_or(|q| q.is_empty()));
                return false;
            }
            _ => {}
        }
        true
    }

    /// KV Connector: update the scheduler state based on the output.
    ///
    /// The Worker side connectors add finished_recving and
    /// finished_sending reqs to the output.
    /// * if finished_sending: free the blocks
    /// * if finished_recving: add to state so we can
    ///   schedule the request during the next step.
    pub(super) fn update_from_kv_xfer_finished(&mut self, kv_connector_output: &KvConnectorOutput) {
        if let Some(connector) = self.connector.as_mut() {
            connector.update_connector_output(kv_connector_output, &mut self.kv_cache_manager);
        }

        // KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving.iter().flatten() {
            let h = self.requests.handle(req_id).expect("finished recving request exists");
            if self.requests[h].status == RequestStatus::WaitingForRemoteKvs {
                self.finished_recving_kv_req_ids.insert(req_id.clone());
                self.set_kv_fetch_stage(h, Some(KvFetchStage::CompletedWaiting));
            } else {
                assert!(self.requests[h].is_finished());
                self.aborted_kv_loads.remove(req_id);
                self.free_blocks(h);
            }
        }
        for req_id in kv_connector_output.finished_sending.iter().flatten() {
            let h = self.requests.handle(req_id).expect("finished sending request exists");
            self.free_blocks(h);
        }
    }

    /// Identify and update requests affected by invalid KV cache blocks.
    ///
    /// This method scans the given requests, detects those with invalid blocks
    /// and adjusts their `num_computed_tokens` to the longest valid prefix.
    /// For observability, it also accumulates the total number of tokens that
    /// will need to be recomputed across all affected requests.
    ///
    /// Args:
    ///     requests: The set of requests to scan for invalid blocks.
    ///     invalid_block_ids: IDs of invalid blocks.
    ///     num_scheduled_tokens: req_id -> number of scheduled tokens.
    ///     evict_blocks: Whether to collect blocks for eviction (False for
    ///         async requests which aren't cached yet).
    ///
    /// Returns:
    ///     tuple:
    ///         - affected_req_ids (set[str]): IDs of requests impacted by
    ///         invalid blocks.
    ///         - total_affected_tokens (int): Total number of tokens that must
    ///         be recomputed across all affected requests.
    ///         - blocks_to_evict (set[int]): Block IDs to evict from cache,
    ///         including invalid blocks and downstream dependent blocks.
    fn update_requests_with_invalid_blocks(
        &mut self,
        requests: &[usize],
        invalid_block_ids: &FxHashSet<BlockId>,
        num_scheduled_tokens: &IndexMap<String, usize>,
        evict_blocks: bool,
    ) -> (IndexSet<String>, usize, PyIntSet) {
        let mut affected_req_ids = IndexSet::new();
        let mut total_affected_tokens = 0;
        // TODO(rust-engine-core): a spec-defined order (see `py_int_set`).
        let mut blocks_to_evict = PyIntSet::default();
        // If a block is invalid and shared by multiple requests in the batch,
        // these requests must be rescheduled, but only the first will recompute
        // it. This set tracks blocks already marked for recomputation.
        let mut marked_invalid_block_ids: FxHashSet<BlockId> = FxHashSet::default();
        for &h in requests {
            let mut is_affected = false;
            let mut marked_invalid_block = false;
            let req_id = self.requests[h].request_id.clone();
            // One block-id list per KV cache group; single-group layouts are
            // enforced by _handle_invalid_blocks.
            let req_block_ids = self.kv_cache_manager.get_block_ids(&req_id).swap_remove(0);
            // We iterate only over blocks that may contain externally computed
            // tokens
            let request = &mut self.requests[h];
            let req_num_computed_tokens = request.num_computed_tokens as i64
                - num_scheduled_tokens.get(&req_id).copied().unwrap_or(0) as i64;

            let req_num_computed_blocks = (req_num_computed_tokens + self.block_size as i64 - 1)
                .div_euclid(self.block_size as i64);
            for (idx, &block_id) in
                req_block_ids.iter().enumerate().take(req_num_computed_blocks.max(0) as usize)
            {
                if !invalid_block_ids.contains(&block_id) {
                    continue;
                }

                is_affected = true;

                if marked_invalid_block_ids.contains(&block_id) {
                    // This invalid block is shared with a previous request
                    // and was already marked for recomputation.
                    // This means this request can still consider this block
                    // as computed when rescheduled.
                    // Currently this only applies to sync loading; Async
                    // loading does not yet support block sharing
                    continue;
                }

                marked_invalid_block_ids.insert(block_id);

                if marked_invalid_block {
                    // This request has already marked an invalid block for
                    // recomputation and updated its num_computed_tokens.
                    continue;
                }

                marked_invalid_block = true;
                // Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size;
                let num_affected_tokens =
                    req_num_computed_tokens - request.num_computed_tokens as i64;
                total_affected_tokens += num_affected_tokens;

                // collect invalid block and all downstream dependent blocks
                if evict_blocks {
                    for &b in &req_block_ids[idx..] {
                        blocks_to_evict.add(b as usize);
                    }
                }
            }

            if is_affected {
                if !marked_invalid_block {
                    // All invalid blocks of this request are shared with
                    // previous requests and will be recomputed by them.
                    // Revert to considering only cached tokens as computed.
                    // Currently this only applies to sync loading; Async
                    // loading does not yet support block sharing
                    total_affected_tokens +=
                        request.num_computed_tokens as i64 - req_num_computed_tokens;
                    request.num_computed_tokens = req_num_computed_tokens as usize;
                }

                affected_req_ids.insert(req_id);
            }
        }

        (
            affected_req_ids,
            total_affected_tokens.max(0) as usize,
            blocks_to_evict,
        )
    }

    /// Fail closed for layouts whose block IDs are not globally unique.
    pub(super) fn handle_failed_recving(&mut self, failed_req_ids: &[String]) -> IndexSet<String> {
        let mut affected_req_ids = IndexSet::new();
        for req_id in failed_req_ids {
            let Some(h) = self.requests.handle(req_id) else {
                continue;
            };
            if self.requests[h].status == RequestStatus::WaitingForRemoteKvs {
                affected_req_ids.insert(req_id.clone());
                if self.recompute_kv_load_failures {
                    self.requests[h].num_computed_tokens = 0;
                    self.failed_recving_kv_req_ids.insert(req_id.clone());
                }
            }
        }
        if self.recompute_kv_load_failures {
            IndexSet::new()
        } else {
            affected_req_ids
        }
    }

    /// Handle requests affected by invalid KV cache blocks.
    ///
    /// Returns:
    ///     Set of affected request IDs to skip in update_from_output main loop,
    ///     as the async and the sync ones (their union in Python).
    // Kept apart for `IdSetOrder`, which reproduces Python's union.
    pub(super) fn handle_invalid_blocks(
        &mut self,
        invalid_block_ids: &[BlockId],
        num_scheduled_tokens: &IndexMap<String, usize>,
    ) -> (IndexSet<String>, IndexSet<String>) {
        // Block IDs are only unique within a group, so a flat set of
        // invalid block IDs cannot be mapped back to requests.
        assert!(
            self.kv_cache_manager.num_kv_cache_groups <= 1,
            "A KV connector reported block-level load failures \
             (invalid_block_ids) which is not supported for models \
             with multiple KV cache groups. \
             Connectors must report failed requests via \
             KVConnectorTransferResults.failed_recving instead."
        );
        let invalid_block_ids: FxHashSet<BlockId> = invalid_block_ids.iter().copied().collect();

        let should_fail = !self.recompute_kv_load_failures;

        // handle async KV loads (not cached yet, evict_blocks=False)
        let async_load_reqs: Vec<usize> = self
            .kv_holding_waiting
            .iter(&self.requests)
            .filter(|&h| self.requests[h].status == RequestStatus::WaitingForRemoteKvs)
            .collect();
        let (async_failed_req_ids, num_failed_tokens, _) = self
            .update_requests_with_invalid_blocks(
                &async_load_reqs,
                &invalid_block_ids,
                num_scheduled_tokens,
                false,
            );

        let mut total_failed_requests = async_failed_req_ids.len();
        let mut _total_failed_tokens = num_failed_tokens;

        // handle sync loads (may be cached, collect blocks for eviction)
        let running = self.running.clone();
        let (sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict) = self
            .update_requests_with_invalid_blocks(
                &running,
                &invalid_block_ids,
                num_scheduled_tokens,
                true,
            );

        total_failed_requests += sync_failed_req_ids.len();
        _total_failed_tokens += num_failed_tokens;

        if total_failed_requests == 0 {
            return Default::default();
        }

        // evict invalid blocks and downstream dependent blocks from cache
        // only when not using recompute policy (where blocks will be recomputed
        // and reused by other requests sharing them)
        if !sync_blocks_to_evict.is_empty() && !self.recompute_kv_load_failures {
            self.kv_cache_manager
                .evict_blocks(sync_blocks_to_evict.iter().map(|b| b as BlockId));
        }

        if should_fail {
            // TODO: Python logs the failed requests at error level.
            return (async_failed_req_ids, sync_failed_req_ids);
        }

        // Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids.extend(async_failed_req_ids);
        // Return sync affected IDs to skip in update_from_output main loop
        (IndexSet::new(), sync_failed_req_ids)
    }

    /// Whether a new async KV load stays within --max-num-inflight-kv-loads.
    pub(super) fn admits_inflight_kv_load(&self) -> bool {
        match self.max_num_inflight_kv_loads {
            None => true,
            Some(limit) => {
                self.kv_fetch_counts.in_progress
                    + self.kv_fetch_counts.completed_waiting
                    + self.aborted_kv_loads.len()
                    < limit
            }
        }
    }

    /// Start deferred async loads whose fence step has completed.
    pub(super) fn drain_deferred_loads(&mut self) {
        while let Some((_, &(fence, num_tokens, queries, hits))) = self.deferred_loads.first() {
            if fence > self.processed_step_seq {
                break;
            }
            let (request_id, _) = self.deferred_loads.shift_remove_index(0).unwrap();
            let connector = self.connector.as_mut().expect("KV connector is configured");
            let h = self.requests.handle(&request_id).expect("deferred load request exists");
            let request = &self.requests[h];
            let blocks = self.kv_cache_manager.get_blocks(&request_id);
            connector.update_state_after_alloc(
                request,
                &blocks,
                &mut self.kv_cache_manager,
                num_tokens,
            );
            if let Some(stats) = &mut self.connector_prefix_cache_stats
                && queries != 0
            {
                let preempted = request.num_preemptions > 0;
                let hits_by_source = if hits > 0 && !preempted {
                    connector.get_external_cache_hit_sources(request, hits)
                } else {
                    None
                };
                stats.record(queries, hits, preempted, hits_by_source);
            }
        }
    }

    pub fn reset_connector_cache(&mut self) -> bool {
        let Some(connector) = self.connector.as_mut() else {
            // No connector attached -> nothing to reset, treat as success so
            // callers that unconditionally request a connector reset (e.g. as
            // part of a cache-clearing cascade after a weight update) don't
            // see reset_prefix_cache() flip to False purely because they
            // didn't configure a connector.
            return true;
        };

        if connector.reset_cache(&mut self.kv_cache_manager) == Some(false) {
            return false;
        }

        if self.log_stats {
            self.connector_prefix_cache_stats
                .as_mut()
                .expect("connector prefix cache stats exist when log_stats is set")
                .stats
                .reset = true;
        }

        true
    }
}

use super::structured_output::GrammarStatus;
use super::update::RequestError;
