// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Native scheduler half of the `MooncakeStoreConnector`
//! (`mooncake_store/scheduler.py` `MooncakeStoreScheduler` and the
//! scheduler-role methods of `connector.py`), as a [`KvConnector`].
//!
//! Every per-step hook runs in Rust: the metadata is built as
//! [`MooncakeStoreConnectorMetadata`] (the binding layer turns it into the
//! Python objects the Python worker half reads), and the blocks of in-flight
//! store jobs are pinned in and released from the scheduler's block pool
//! directly. Only the prefix lookup leaves Rust: it is an RPC to worker rank 0
//! behind [`LookupClient`], issued per request rather than per step.
//!
//! The Python implementation is the behavioral reference: for the same
//! inputs this produces the same metadata and the same block pool operations.
//!
//! Not supported (the binding rejects them at construction): KV cache
//! events, and composition inside a `MultiConnector`.

mod meta;
#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::sync::Arc;

use indexmap::IndexMap;
use rustc_hash::{FxHashMap, FxHashSet};

pub use self::meta::{
    BoundaryPut, LoadSpec, MooncakeStoreConnectorMetadata, MooncakeStoreWorkerMetadata,
    ReqBlockHashes, ReqMeta, SetOp, StoreGeometry,
};
use self::meta::{FromTracker, RequestTracker, from_request_tracker, partial_tail_non_mamba_puts};
use crate::connectors::mooncake_store::LookupResult;
use crate::kv_cache::{
    BlockId, BlockPool, BoundaryStateOffload, KvCacheBlocks, KvCacheManager, NULL_BLOCK_ID,
};
use crate::sched::output::{GroupedBlockIds, Opaque, SchedulerOutput};
use crate::sched::request::Request;
use crate::sched::scheduler::{
    KvConnector, KvConnectorBlockState, KvConnectorOutput, RequestFinished, Requests,
};
use crate::sched::stats::CACHE_HIT_SOURCE_EXTERNAL_UNSPECIFIED;

/// `kv_transfer_config.kv_role`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvRole {
    Producer,
    Consumer,
    Both,
}

impl KvRole {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "kv_producer" => Some(Self::Producer),
            "kv_consumer" => Some(Self::Consumer),
            "kv_both" => Some(Self::Both),
            _ => None,
        }
    }
}

/// Which blocks a store job pins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SavePinPolicy {
    /// Python's protocol: every save job pins the request's whole (non-Mamba)
    /// store block table, since each rank may retry from an earlier
    /// successful save offset. O(blocks) per save step, O(n²) over a chunked
    /// prefill.
    #[default]
    WholeTable,
    /// Opt-in, changes behavior: skip the blocks wholly below the request's
    /// confirmed save offset, the highest
    /// [`MooncakeStoreWorkerMetadata::saved_offsets`] reported for a finished
    /// job of the request. Requires worker ranks that report the offset and
    /// never retry below it; the current Python worker reports none, so this
    /// pins the whole table until it does.
    SkipConfirmed,
}

/// The `MooncakeStoreScheduler` configuration, resolved by the Python
/// constructor (`MooncakeStoreScheduler.__init__`).
#[derive(Clone, Debug)]
pub struct MooncakeStoreSchedulerConfig {
    pub kv_role: KvRole,
    pub load_async: bool,
    pub lookup_async: bool,
    /// Skips lookup CPU cost on instances that never load KV from the store.
    pub enable_lookup: bool,
    pub save_decode_cache: bool,
    /// `kv_cache_config.prefix_cacheable_group_ids`: the scheduler's KV cache
    /// groups the store holds, in store group order.
    pub store_group_ids: Vec<usize>,
    /// The engine's scheduler block size.
    pub block_size: usize,
    pub hash_block_size: usize,
    /// Store groups with a `MambaSpec` (`_boundary_state_group_ids`).
    pub boundary_state_group_ids: Vec<usize>,
    /// The store coordinator; `None` in capacity-only mode, where the
    /// connector contributes store capacity but transfers no KV.
    pub store: Option<StoreGeometry>,
    /// `parallel_config.world_size`: the ranks that report each store job.
    pub num_workers: usize,
    pub save_pin_policy: SavePinPolicy,
}

/// The `LookupKeyClient` of the scheduler: the prefix-lookup RPC to worker
/// rank 0 (and its admin commands). It keeps per-request lookup state
/// (prefetches, async lookups) until the result is consumed or discarded.
pub trait LookupClient: Send {
    /// Start a lookup for a newly added request; a later `lookup` reuses it.
    /// Returns whether a lookup was started.
    fn prefetch(&mut self, request: &Request) -> bool;

    /// The lookup result for `request`. With `non_block`, `None` until the
    /// result is ready, which is then cached until expiry or `discard`;
    /// otherwise the lookup state is dropped once answered.
    fn lookup(&mut self, request: &Request, non_block: bool) -> Option<LookupResult>;

    /// Drop any cached/in-flight lookup for `request_id` (e.g. on abort).
    fn discard(&mut self, request_id: &str);

    /// Trigger a global `remove_all(force=True)` on the Mooncake master.
    /// True on ACK from worker, False on NACK or RPC error.
    fn reset(&mut self) -> bool;
}

/// A store job: its referenced blocks and the ranks yet to report it done.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PinnedSave {
    pub block_ids: Vec<BlockId>,
    pub remaining: i64,
    /// The request and its confirmed save offset once every rank finished
    /// this job ([`SavePinPolicy::SkipConfirmed`] only).
    req_id: Option<String>,
}

/// `MooncakeStoreScheduler`: scheduler-side component for
/// MooncakeStoreConnector.
pub struct MooncakeStoreScheduler {
    config: MooncakeStoreSchedulerConfig,
    client: Box<dyn LookupClient>,
    /// Maps scheduler group IDs to store group indices. Groups outside the
    /// store projection, such as the QSA ring, are absent and skipped.
    store_group_id_by_kv_cache_group_id: FxHashMap<usize, usize>,
    /// Mirrors MooncakeStoreConnector._capacity_only.
    capacity_only: bool,
    enable_partial_hash_hits: bool,

    next_store_job_id: u64,
    /// store_job_id -> (referenced block ids, ranks yet to report completion)
    pinned_saves: FxHashMap<u64, PinnedSave>,

    // Per-request state
    /// to be loaded
    load_specs: FxHashMap<String, LoadSpec>,
    /// Save progress starts when a request is scheduled for compute.
    request_trackers: FxHashMap<String, RequestTracker>,
    /// `_unfinished_requests`: the block ids of the request's external load
    /// (empty without one); the request itself is looked up by id.
    unfinished_requests: IndexMap<String, Arc<GroupedBlockIds>>,
    unfinished_request_ids: FxHashSet<String>,
    /// Changes to `unfinished_request_ids` the next metadata reports.
    unfinished_request_id_ops: Vec<SetOp>,
    finished_partial_tail_metas: IndexMap<String, ReqMeta>,
    finished_lookahead_metas: IndexMap<String, ReqMeta>,

    /// Requests the client may hold lookup state for. The client drops the
    /// state of a synchronous lookup itself, so only prefetches and
    /// non-blocking lookups add to it; discarding any other request is a
    /// no-op and is skipped.
    lookups_in_client: FxHashSet<String>,
    /// [`SavePinPolicy::SkipConfirmed`]: per request, the save offset every
    /// rank has confirmed.
    confirmed_save_offsets: FxHashMap<String, usize>,
}

impl MooncakeStoreScheduler {
    pub fn new(config: MooncakeStoreSchedulerConfig, client: Box<dyn LookupClient>) -> Self {
        // Mirrors MooncakeStoreConnector._capacity_only.
        let capacity_only = config.kv_role == KvRole::Consumer
            && !config.enable_lookup
            && !config.save_decode_cache;
        assert_eq!(
            capacity_only,
            config.store.is_none(),
            "the store coordinator exists unless capacity-only"
        );
        let enable_partial_hash_hits =
            config.store.as_ref().is_some_and(|s| s.enable_partial_hash_hits);
        let store_group_id_by_kv_cache_group_id = config
            .store_group_ids
            .iter()
            .enumerate()
            .map(|(store_group_id, &group_id)| (group_id, store_group_id))
            .collect();
        Self {
            config,
            client,
            store_group_id_by_kv_cache_group_id,
            capacity_only,
            enable_partial_hash_hits,
            next_store_job_id: 0,
            pinned_saves: FxHashMap::default(),
            load_specs: FxHashMap::default(),
            request_trackers: FxHashMap::default(),
            unfinished_requests: IndexMap::new(),
            unfinished_request_ids: FxHashSet::default(),
            unfinished_request_id_ops: Vec::new(),
            finished_partial_tail_metas: IndexMap::new(),
            finished_lookahead_metas: IndexMap::new(),
            lookups_in_client: FxHashSet::default(),
            confirmed_save_offsets: FxHashMap::default(),
        }
    }

    pub fn config(&self) -> &MooncakeStoreSchedulerConfig {
        &self.config
    }

    /// The in-flight store jobs, for tests and diagnostics.
    pub fn pinned_saves(&self) -> impl Iterator<Item = (u64, &PinnedSave)> {
        self.pinned_saves.iter().map(|(&job, pinned)| (job, pinned))
    }

    fn is_consumer(&self) -> bool {
        self.config.kv_role == KvRole::Consumer
    }

    fn lookup_align(&self) -> usize {
        // Fine-grained hits may land on a hash boundary inside a block; without
        // partial hits, prefixes shorter than one physical block are skipped.
        if self.enable_partial_hash_hits {
            self.config.hash_block_size
        } else {
            self.config.block_size
        }
    }

    /// `client.discard`, skipped when the client holds nothing for the request.
    fn discard_lookup(&mut self, request_id: &str) {
        if self.lookups_in_client.remove(request_id) {
            self.client.discard(request_id);
        }
    }

    fn store_group_block_ids(&self, block_ids: &[Vec<BlockId>]) -> GroupedBlockIds {
        self.config.store_group_ids.iter().map(|&g| block_ids[g].clone()).collect()
    }

    fn tracker_args(
        &self,
        request: &Request,
        load_spec: Option<LoadSpec>,
        skip_save: bool,
        save_partial_tail: bool,
    ) -> FromTracker {
        FromTracker {
            block_size: self.config.block_size,
            hash_block_size: self.config.hash_block_size,
            load_spec,
            skip_save,
            block_hashes: ReqBlockHashes::Request {
                len: request.block_hashes.len(),
            },
            save_partial_tail,
            num_prompt_tokens: Some(request.num_prompt_tokens),
        }
    }

    /// Start an async lookup as soon as the request is added, so it can
    /// complete while the request waits to be scheduled.
    fn start_lookup_prefetch(&mut self, request: &Request) {
        if !self.config.enable_lookup
            || !self.config.lookup_async
            || request.skip_reading_external_kv()
            || request.num_tokens() < self.lookup_align()
            || request.block_hashes.is_empty()
        {
            return;
        }
        if self.client.prefetch(request) {
            self.lookups_in_client.insert(request.request_id.clone());
        }
    }

    /// Build connector metadata for this scheduler step.
    pub fn build_meta(
        &mut self,
        scheduler_output: &SchedulerOutput,
        block_state: &mut KvConnectorBlockState<'_>,
    ) -> MooncakeStoreConnectorMetadata {
        let requests = block_state.requests;
        let is_consumer = self.is_consumer();
        let can_process_cached = !is_consumer || self.config.save_decode_cache;

        for finished_req_id in &scheduler_output.finished_req_ids {
            self.load_specs.remove(finished_req_id);
            self.request_trackers.remove(finished_req_id);
            self.unfinished_requests.shift_remove(finished_req_id);
            if self.unfinished_request_ids.remove(finished_req_id) {
                self.unfinished_request_id_ops.push(SetOp::Discard(finished_req_id.clone()));
            }
            self.confirmed_save_offsets.remove(finished_req_id);
        }

        let preempted_ids = scheduler_output.preempted_req_ids.clone().unwrap_or_default();
        for req_id in &preempted_ids {
            self.discard_lookup(req_id);
            self.load_specs.remove(req_id);
            self.request_trackers.remove(req_id);
            self.unfinished_requests.shift_remove(req_id);
            // A resumed request saves from its start again.
            self.confirmed_save_offsets.remove(req_id);
        }

        let mut meta = MooncakeStoreConnectorMetadata {
            requests: Vec::new(),
            unfinished_request_id_ops: std::mem::take(&mut self.unfinished_request_id_ops),
            preempted_req_ids: preempted_ids,
        };

        // Handle new requests
        for request in &scheduler_output.scheduled_new_reqs {
            let load_spec = self.load_specs.remove(&request.req_id);
            let num_tokens_to_compute = request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[&request.req_id];
            assert!(self.unfinished_requests.contains_key(&request.req_id));
            let request_real = requests.get(&request.req_id).expect("scheduled request exists");

            let unfolded_block_ids = self.store_group_block_ids(&request.block_ids);
            let mut request_tracker = RequestTracker::new(
                request.req_id.clone(),
                num_tokens_to_compute,
                Arc::new(unfolded_block_ids),
                request.prefill_token_ids.len(),
            );
            // A consumer may write decode KV without becoming a prefill
            // producer. Loads are still carried by the same metadata.
            let args = self.tracker_args(
                request_real,
                load_spec,
                is_consumer,
                self.enable_partial_hash_hits,
            );
            let req_meta = from_request_tracker(&mut request_tracker, args);
            self.request_trackers.insert(request.req_id.clone(), request_tracker);
            if let Some(req_meta) = req_meta {
                meta.requests.push(req_meta);
            }
        }

        // Handle cached (running) requests
        let cached_reqs = &scheduler_output.scheduled_cached_reqs;
        if can_process_cached {
            for (i, req_id) in cached_reqs.req_ids.iter().enumerate() {
                let new_block_ids = cached_reqs.new_block_ids[i]
                    .as_ref()
                    .map(|ids| self.store_group_block_ids(ids));

                // Decode/chunked request
                let request_tracker = self
                    .request_trackers
                    .get_mut(req_id)
                    .unwrap_or_else(|| panic!("no tracker for request {req_id}"));
                let num_computed_token = cached_reqs.num_computed_tokens[i];
                // Use the tracker's snapshot of the prefill range so resumed
                // requests keep saving past the original prompt boundary.
                let prefill_end = request_tracker.prefill_end_tokens;
                let is_decode = num_computed_token >= prefill_end;
                if is_decode && !self.config.save_decode_cache {
                    continue;
                }

                let num_new_tokens = scheduler_output.num_scheduled_tokens[req_id];
                assert!(
                    self.unfinished_requests.contains_key(req_id),
                    "Request {req_id} is not in _unfinished_requests"
                );
                let unfinished_req = requests.get(req_id).expect("scheduled request exists");
                let num_current_tokens = request_tracker.token_len;
                // len(all_token_ids[num_current_tokens : num_current_tokens + num_new_tokens])
                let num_new_token_ids = unfinished_req
                    .num_tokens()
                    .min(num_current_tokens + num_new_tokens)
                    .saturating_sub(num_current_tokens);
                request_tracker.token_len += num_new_token_ids;
                // A block is usually allocated before the step that fills
                // it, so reaching a save boundary does not imply that this
                // step has new block ids.
                if let Some(new_block_ids) = new_block_ids {
                    request_tracker.update(new_block_ids);
                }
                if is_consumer && !is_decode {
                    continue;
                }

                let args = FromTracker {
                    block_size: self.config.block_size,
                    hash_block_size: self.config.hash_block_size,
                    load_spec: None,
                    skip_save: false,
                    block_hashes: ReqBlockHashes::Request {
                        len: unfinished_req.block_hashes.len(),
                    },
                    save_partial_tail: self.enable_partial_hash_hits,
                    num_prompt_tokens: Some(unfinished_req.num_prompt_tokens),
                };
                if let Some(req_meta) = from_request_tracker(request_tracker, args) {
                    meta.requests.push(req_meta);
                }
            }
        }

        // Handle requests with pending load specs not yet scheduled
        let scheduled: FxHashSet<&str> = scheduler_output
            .scheduled_new_reqs
            .iter()
            .map(|r| r.req_id.as_str())
            .chain(cached_reqs.req_ids.iter().map(String::as_str))
            .collect();
        let pending: Vec<usize> = (0..self.unfinished_requests.len())
            .filter(|&i| {
                let request_id = self.unfinished_requests.get_index(i).unwrap().0;
                !scheduled.contains(request_id.as_str()) && self.load_specs.contains_key(request_id)
            })
            .collect();
        for i in pending {
            let (request_id, block_ids) = self.unfinished_requests.get_index(i).unwrap();
            let load_spec = self.load_specs.remove(request_id);
            // A load spec may have been proposed by this connector's
            // lookup but rejected by MultiConnector in favor of another
            // connector. Only the chosen connector may issue the pending
            // load; the normal store path gets its blocks later from
            // SchedulerOutput once the request is actually scheduled.
            let Some(load_spec) = load_spec.filter(|s| s.can_load) else {
                continue;
            };
            let unfinished_req = requests.get(request_id).expect("loading request exists");
            let num_tokens_to_compute = load_spec.kvpool_cached_tokens;
            let mut request_tracker = RequestTracker::new(
                request_id.clone(),
                num_tokens_to_compute,
                block_ids.clone(),
                0,
            );
            let args = self.tracker_args(unfinished_req, Some(load_spec), false, false);
            if let Some(req_meta) = from_request_tracker(&mut request_tracker, args) {
                meta.requests.push(req_meta);
            }
        }

        if !block_state.boundary_state_offloads.is_empty() && can_process_cached {
            self.handle_boundary_state_offloads(
                &block_state.boundary_state_offloads,
                requests,
                &mut meta,
            );
        }

        self.apply_current_save_block_ids(&mut meta, block_state);

        // Finish-time handoffs arrive after the producing step's metadata was
        // built. Their exact blocks were pinned when they were registered, so
        // they remain valid after request cleanup and need no current snapshot.
        meta.requests.extend(self.finished_partial_tail_metas.drain(..).map(|(_, m)| m));
        meta.requests.extend(self.finished_lookahead_metas.drain(..).map(|(_, m)| m));

        if !self.capacity_only {
            let pool = block_state.kv_cache_manager.block_pool_mut();
            self.reference_save_blocks(&mut meta.requests, pool);
        }
        meta
    }

    /// Replace append-only mirrors with the core's current block tables.
    fn apply_current_save_block_ids(
        &self,
        meta: &mut MooncakeStoreConnectorMetadata,
        block_state: &mut KvConnectorBlockState<'_>,
    ) {
        for req_meta in meta.requests.iter_mut().filter(|m| m.can_save) {
            assert!(
                block_state.req_ids.contains(&req_meta.req_id),
                "Missing current block table for store request {}",
                req_meta.req_id
            );
            let mut block_ids = block_state.kv_cache_manager.get_block_ids(&req_meta.req_id);
            req_meta.block_ids = Arc::new(
                self.config
                    .store_group_ids
                    .iter()
                    .map(|&g| std::mem::take(&mut block_ids[g]))
                    .collect(),
            );
        }
    }

    /// Take a GPU block reference for every store job this step emits.
    ///
    /// The worker DMAs out of these blocks after the step that scheduled them,
    /// so a reference keeps them out of the free queue even once the request
    /// itself is freed, until every rank reports the job done.
    fn reference_save_blocks(&mut self, requests: &mut [ReqMeta], pool: &mut BlockPool) {
        let coord = self.config.store.as_ref().expect("store coordinator");
        let mut seen = FxHashSet::default();
        for req_meta in requests.iter_mut() {
            if !req_meta.can_save {
                continue;
            }
            if let Some(store_job_id) = req_meta.store_job_id {
                assert!(self.pinned_saves.contains_key(&store_job_id));
                continue;
            }
            let tail = partial_tail_non_mamba_puts(coord, req_meta);
            if !tail.is_empty() {
                let mut puts = tail;
                puts.extend(req_meta.boundary_puts.take().unwrap_or_default());
                req_meta.boundary_puts = Some(puts);
            }
            let store_job_id = self.next_store_job_id;
            req_meta.store_job_id = Some(store_job_id);
            self.next_store_job_id += 1;
            let mut block_ids: Vec<BlockId> =
                req_meta.boundary_puts.iter().flatten().map(|put| put.block_id).collect();
            assert!(
                !block_ids.contains(&NULL_BLOCK_ID),
                "A null block cannot back a boundary put"
            );
            if req_meta.token_len_chunk > 0 {
                // Each rank may retry from an earlier successful save offset.
                let confirmed = match self.config.save_pin_policy {
                    SavePinPolicy::WholeTable => 0,
                    SavePinPolicy::SkipConfirmed => {
                        self.confirmed_save_offsets.get(&req_meta.req_id).copied().unwrap_or(0)
                    }
                };
                for (group_id, group) in req_meta.block_ids.iter().enumerate() {
                    if self.config.boundary_state_group_ids.contains(&group_id) {
                        continue;
                    }
                    let skip = confirmed / coord.group_block_sizes[group_id];
                    block_ids
                        .extend(group.iter().skip(skip).copied().filter(|&b| b != NULL_BLOCK_ID));
                }
            }
            // An aligned boundary block may also be present in the request's
            // block table. Take and release exactly one reference per block.
            seen.clear();
            block_ids.retain(|&b| seen.insert(b));
            if block_ids.is_empty() {
                continue;
            }
            pool.touch(&block_ids);
            let req_id = (self.config.save_pin_policy == SavePinPolicy::SkipConfirmed)
                .then(|| req_meta.req_id.clone());
            self.pinned_saves.insert(
                store_job_id,
                PinnedSave {
                    block_ids,
                    remaining: self.config.num_workers as i64,
                    req_id,
                },
            );
        }
    }

    /// Pin and queue KV newly eligible for saving after the final hash update.
    fn register_finished_lookahead_save(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        pool: &mut BlockPool,
    ) {
        if self.is_consumer() && !self.config.save_decode_cache {
            return;
        }
        let coord = self.config.store.as_ref().expect("store coordinator");
        if coord.num_hash_lookahead_tokens == 0 {
            return;
        }
        let Some(tracker) = self.request_trackers.get_mut(&request.request_id) else {
            return;
        };
        if block_ids.iter().all(Vec::is_empty) {
            return;
        }
        let save_limit = if self.config.save_decode_cache {
            request.num_tokens()
        } else {
            tracker.prefill_end_tokens
        };
        let in_flight_tokens = request.num_in_flight_tokens - request.num_stale_output_tokens;
        let completed_tokens = request.num_computed_tokens - in_flight_tokens;
        tracker.token_len = completed_tokens.min(save_limit);
        let args = FromTracker {
            block_size: self.config.block_size,
            hash_block_size: self.config.hash_block_size,
            load_spec: None,
            skip_save: false,
            block_hashes: ReqBlockHashes::Finished {
                hashes: Arc::from(request.block_hashes.as_slice()),
                copy: false,
            },
            save_partial_tail: false,
            num_prompt_tokens: Some(request.num_prompt_tokens),
        };
        let Some(mut req_meta) = from_request_tracker(tracker, args) else {
            return;
        };
        assert!(!self.finished_lookahead_metas.contains_key(&request.request_id));
        req_meta.block_ids = Arc::new(self.store_group_block_ids(block_ids));
        self.reference_save_blocks(std::slice::from_mut(&mut req_meta), pool);
        if req_meta.store_job_id.is_some_and(|job| self.pinned_saves.contains_key(&job)) {
            self.finished_lookahead_metas.insert(request.request_id.clone(), req_meta);
        }
    }

    /// Queue and pin a finish-time tail for the next connector step.
    fn register_partial_tail(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        partial_tail_offloads: &[BoundaryStateOffload],
        pool: &mut BlockPool,
    ) -> bool {
        if self.is_consumer() || partial_tail_offloads.is_empty() {
            return false;
        }
        let Some(tracker) = self.request_trackers.get(&request.request_id) else {
            return false;
        };
        if block_ids.iter().all(Vec::is_empty) {
            return false;
        }
        let boundaries: BTreeSet<usize> =
            partial_tail_offloads.iter().map(|&(_, _, b)| b).collect();
        assert!(
            boundaries.len() == 1,
            "Partial-tail offloads for one request must share a boundary"
        );
        let boundary_tokens = *boundaries.first().unwrap();
        if boundary_tokens > tracker.prefill_end_tokens {
            return false;
        }

        let mut mamba_puts = Vec::new();
        for &(group_id, block_id, boundary) in partial_tail_offloads {
            let store_group_id = self.store_group_id_by_kv_cache_group_id.get(&group_id);
            if !store_group_id.is_some_and(|g| self.config.boundary_state_group_ids.contains(g)) {
                return false;
            }
            if block_id == NULL_BLOCK_ID {
                return false;
            }
            mamba_puts.push(BoundaryPut {
                group_id: *store_group_id.unwrap(),
                block_id,
                num_tokens: boundary,
            });
        }
        let prefill_end_tokens = tracker.prefill_end_tokens;
        let mut req_meta = ReqMeta::offload_only(
            request.request_id.clone(),
            Arc::new(self.store_group_block_ids(block_ids)),
            ReqBlockHashes::Finished {
                hashes: Arc::from(request.block_hashes.as_slice()),
                copy: true,
            },
            request.num_prompt_tokens,
            prefill_end_tokens,
            mamba_puts.clone(),
            request.num_computed_tokens,
        );
        let coord = self.config.store.as_ref().expect("store coordinator");
        let mut puts = partial_tail_non_mamba_puts(coord, &req_meta);
        puts.extend(mamba_puts);
        let mut seen = FxHashSet::default();
        let pinned_block_ids: Vec<BlockId> =
            puts.iter().map(|put| put.block_id).filter(|&b| seen.insert(b)).collect();
        req_meta.boundary_puts = Some(puts);

        assert!(!self.finished_partial_tail_metas.contains_key(&request.request_id));
        let store_job_id = self.next_store_job_id;
        self.next_store_job_id += 1;
        pool.touch(&pinned_block_ids);
        self.pinned_saves.insert(
            store_job_id,
            PinnedSave {
                block_ids: pinned_block_ids,
                remaining: self.config.num_workers as i64,
                req_id: None,
            },
        );

        req_meta.store_job_id = Some(store_job_id);
        self.finished_partial_tail_metas.insert(request.request_id.clone(), req_meta);
        self.request_trackers
            .get_mut(&request.request_id)
            .expect("tracker exists")
            .has_pending_offload = true;
        // The store job owns exact block refs, so request cleanup need not wait.
        false
    }

    /// Attach exact boundary-state blocks to store jobs for this step.
    ///
    /// Flushed in the step they arrive: the CoW copy is enqueued before the
    /// connector event records, so this step's event fences the exact block.
    /// Entries ride the request's save meta when present, else an
    /// offload-only ReqMeta (`token_len_chunk=0` skips the normal save,
    /// `can_save=True` takes the normal enqueue and store-job pinning path).
    fn handle_boundary_state_offloads(
        &mut self,
        offloads: &IndexMap<String, Vec<BoundaryStateOffload>>,
        requests: Requests<'_>,
        meta: &mut MooncakeStoreConnectorMetadata,
    ) {
        // `{m.req_id: m for m in meta.requests if m.can_save}`: the last wins.
        let save_metas: FxHashMap<String, usize> = meta
            .requests
            .iter()
            .enumerate()
            .filter(|(_, m)| m.can_save)
            .map(|(i, m)| (m.req_id.clone(), i))
            .collect();
        for (req_id, entries) in offloads {
            let Some(tracker) = self.request_trackers.get_mut(req_id) else {
                // Request finished/preempted within this step; its blocks are
                // going away, so the offload is conservatively dropped.
                continue;
            };
            if !self.unfinished_requests.contains_key(req_id) {
                continue;
            }
            let mut accepted = Vec::new();
            for &(group_id, block_id, boundary_tokens) in entries {
                let is_decode_boundary = boundary_tokens > tracker.prefill_end_tokens;
                if is_decode_boundary && !self.config.save_decode_cache {
                    continue;
                }
                if !is_decode_boundary && self.config.kv_role == KvRole::Consumer {
                    continue;
                }
                if block_id == NULL_BLOCK_ID {
                    continue;
                }
                let Some(&store_group_id) = self.store_group_id_by_kv_cache_group_id.get(&group_id)
                else {
                    continue;
                };
                if !self.config.boundary_state_group_ids.contains(&store_group_id) {
                    continue;
                }
                accepted.push(BoundaryPut {
                    group_id: store_group_id,
                    block_id,
                    num_tokens: boundary_tokens,
                });
            }
            if accepted.is_empty() {
                continue;
            }
            tracker.has_pending_offload = true;
            if let Some(&i) = save_metas.get(req_id) {
                meta.requests[i].boundary_puts = Some(accepted);
                continue;
            }
            let request = requests.get(req_id).expect("unfinished request exists");
            meta.requests.push(ReqMeta::offload_only(
                req_id.clone(),
                tracker.allocated_block_ids.clone(),
                ReqBlockHashes::Request {
                    len: request.block_hashes.len(),
                },
                request.num_prompt_tokens,
                tracker.prefill_end_tokens,
                accepted,
                tracker.token_len,
            ));
        }
    }

    /// Drop the block references of store jobs every rank has finished.
    pub fn update_worker_meta(&mut self, meta: &MooncakeStoreWorkerMetadata, pool: &mut BlockPool) {
        let saved_offsets: FxHashMap<u64, usize> = meta.saved_offsets.iter().copied().collect();
        for &(store_job_id, count) in &meta.completed_saves {
            let Some(pinned) = self.pinned_saves.get_mut(&store_job_id) else {
                // The job referenced no blocks, so nothing was recorded for it.
                continue;
            };
            pinned.remaining -= count as i64;
            if pinned.remaining > 0 {
                continue;
            }
            assert!(
                pinned.remaining == 0,
                "store job {store_job_id} reported by too many ranks"
            );
            let pinned = self.pinned_saves.remove(&store_job_id).unwrap();
            if let (Some(req_id), Some(&offset)) = (pinned.req_id, saved_offsets.get(&store_job_id))
            {
                let confirmed = self.confirmed_save_offsets.entry(req_id).or_default();
                *confirmed = (*confirmed).max(offset);
            }
            // Tail-first, as elsewhere, so the shared prefix is evicted last.
            pool.free_blocks(pinned.block_ids.into_iter().rev(), false);
        }
    }
}

impl KvConnector for MooncakeStoreScheduler {
    fn supports_divergent_local_hybrid_hits(&self) -> bool {
        false
    }

    fn requires_kv_delivery(&self) -> bool {
        // The base default: the producer role.
        matches!(self.config.kv_role, KvRole::Producer | KvRole::Both)
    }

    fn on_new_request(&mut self, request: &Request, _kv_cache_manager: &mut KvCacheManager) {
        self.start_lookup_prefetch(request);
    }

    fn may_load_async(&mut self, request: &Request) -> bool {
        request.do_remote_prefill
    }

    /// Check for external KV cache hit.
    ///
    /// Returns `(None, False)` when an async lookup is still in flight,
    /// signaling the scheduler to retry this request on a later step.
    fn get_num_new_matched_tokens(
        &mut self,
        request: &mut Request,
        num_computed_tokens: usize,
        _kv_cache_manager: &mut KvCacheManager,
    ) -> (Option<usize>, bool) {
        if !self.config.enable_lookup {
            return (Some(0), false);
        }

        if request.num_tokens() < self.lookup_align() {
            return (Some(0), false);
        }

        let non_block = self.config.lookup_async;
        let lookup_result = self.client.lookup(request, non_block);
        if non_block {
            self.lookups_in_client.insert(request.request_id.clone());
        }
        let Some(lookup_result) = lookup_result else {
            // Lookup not ready yet; scheduler will retry on a later step.
            return (None, false);
        };
        let num_external_hit_tokens = lookup_result.hit_length;

        let need_to_allocate = num_external_hit_tokens.saturating_sub(num_computed_tokens);
        if need_to_allocate == 0 {
            return (Some(0), false);
        }

        self.load_specs.insert(
            request.request_id.clone(),
            LoadSpec {
                vllm_cached_tokens: num_computed_tokens,
                kvpool_cached_tokens: num_external_hit_tokens,
                can_load: false,
                token_len: 0,
                tail_key_boundaries: lookup_result.tail_key_boundaries,
            },
        );

        (Some(need_to_allocate), self.config.load_async)
    }

    fn get_loaded_kv_cache_group_ids(&mut self, _request: &Request) -> Vec<usize> {
        self.config.store_group_ids.clone()
    }

    /// Update state after block allocation.
    fn update_state_after_alloc(
        &mut self,
        request: &Request,
        blocks: &KvCacheBlocks,
        _kv_cache_manager: &mut KvCacheManager,
        num_external_tokens: usize,
    ) {
        // The lookup is consumed; a reschedule (e.g. after a failed load) must
        // not reuse its hit.
        self.discard_lookup(&request.request_id);
        let local_block_ids = if num_external_tokens > 0 {
            blocks
                .get_block_ids(false, Some(&self.config.store_group_ids))
                .expect("allow_none is false")
        } else {
            Vec::new()
        };

        let request_id = &request.request_id;
        self.unfinished_requests.insert(request_id.clone(), Arc::new(local_block_ids));
        if self.unfinished_request_ids.insert(request_id.clone()) {
            self.unfinished_request_id_ops.push(SetOp::Add(request_id.clone()));
        }

        let Some(load_spec) = self.load_specs.get_mut(request_id) else {
            return;
        };

        if num_external_tokens == 0 {
            load_spec.can_load = false;
            return;
        }

        assert!(
            num_external_tokens == load_spec.kvpool_cached_tokens - load_spec.vllm_cached_tokens,
            "Mismatch in number of tokens: {num_external_tokens} vs {} - {} for request {request_id}",
            load_spec.kvpool_cached_tokens,
            load_spec.vllm_cached_tokens,
        );

        load_spec.can_load = true;
    }

    fn get_external_cache_hit_sources(
        &mut self,
        _request: &Request,
        num_external_tokens: usize,
    ) -> Option<Vec<(String, usize)>> {
        // The base default: all tokens under `_cache_hit_source`.
        if num_external_tokens == 0 {
            return Some(Vec::new());
        }
        Some(vec![(
            CACHE_HIT_SOURCE_EXTERNAL_UNSPECIFIED.to_owned(),
            num_external_tokens,
        )])
    }

    fn has_pending_block_frees(&mut self, _kv_cache_manager: &mut KvCacheManager) -> bool {
        false
    }

    fn build_connector_meta(
        &mut self,
        scheduler_output: &SchedulerOutput,
        block_state: &mut KvConnectorBlockState<'_>,
    ) -> Opaque {
        Opaque::new(self.build_meta(scheduler_output, block_state))
    }

    fn register_finished_partial_tail(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        partial_tail_offloads: &[BoundaryStateOffload],
        kv_cache_manager: &mut KvCacheManager,
    ) -> bool {
        self.register_partial_tail(
            request,
            block_ids,
            partial_tail_offloads,
            kv_cache_manager.block_pool_mut(),
        )
    }

    fn request_finished(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        kv_cache_manager: &mut KvCacheManager,
    ) -> RequestFinished {
        self.discard_lookup(&request.request_id);
        // An in-flight store job holds its own reference on the blocks it reads,
        // so a finishing request never has to defer freeing them.
        self.register_finished_lookahead_save(
            request,
            block_ids,
            kv_cache_manager.block_pool_mut(),
        );
        RequestFinished::default()
    }

    /// Reads `kv_connector_worker_meta` from `output`, which carries a
    /// [`MooncakeStoreWorkerMetadata`] (the binding layer converts the
    /// Python one); anything else is ignored, as Python ignores other
    /// worker metadata types.
    fn update_connector_output(
        &mut self,
        kv_connector_output: &KvConnectorOutput,
        kv_cache_manager: &mut KvCacheManager,
    ) {
        let Some(meta) = kv_connector_output
            .output
            .as_ref()
            .and_then(|o| o.0.downcast_ref::<MooncakeStoreWorkerMetadata>())
        else {
            return;
        };
        self.update_worker_meta(meta, kv_cache_manager.block_pool_mut());
    }

    fn get_kv_connector_stats(&mut self, _kv_cache_manager: &mut KvCacheManager) -> Option<Opaque> {
        // Stats are collected on the worker side only.
        None
    }

    fn take_events(&mut self, _kv_cache_manager: &mut KvCacheManager) -> Option<Opaque> {
        // KV cache events are not supported (rejected at construction).
        None
    }

    /// Keep the engine stepping while any store job still holds block refs.
    ///
    /// Completions only reach the scheduler as worker metadata on a step, so an
    /// engine that quiesced with jobs in flight would leave those references
    /// held indefinitely. Nothing else keeps it alive now that a finishing
    /// request no longer defers its own free.
    fn has_pending_push_work(&mut self, _kv_cache_manager: &mut KvCacheManager) -> bool {
        !self.pinned_saves.is_empty()
    }

    /// Reset the external Mooncake store on prefix-cache reset.
    fn reset_cache(&mut self, _kv_cache_manager: &mut KvCacheManager) -> Option<bool> {
        // Clear local references to keys we're about to wipe.
        self.load_specs.clear();
        let ok = self.client.reset();
        if ok {
            // The client discards every lookup on success.
            self.lookups_in_client.clear();
        }
        Some(ok)
    }
}
