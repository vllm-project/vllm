// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `vllm/v1/hisparse/coordinator.py` (`HiSparseCoordinator`) and the
//! overrides of the HiSparse single-type managers that call into it
//! (`HiSparseSourceManager`, and the parts of `HiSparseHotManager` /
//! `HiSparseResidentManager` that need the coordinator).
//!
//! Python's coordinator holds references to its managers and pools; here the
//! [`KvCacheCoordinator`](super::KvCacheCoordinator) owns them and lends them
//! to every call as a [`HiSparseCtx`].
//!
//! Deviations from Python:
//! - Python calls `_on_block_reused` from inside `BlockPool.get_new_blocks`.
//!   Here the pool queues reused blocks ([`BlockPool::take_reused_blocks`])
//!   and [`HiSparseCoordinator::process_reused_blocks`] handles them right
//!   after each allocation from the device pool, before anything reads the
//!   state they change (the handler reads no pool state).
//! - A pending publication keeps a snapshot of its request
//!   ([`PublicationRequest`]) instead of a reference to it, refreshed on
//!   every `publish_when_ready`; see there.
//! - Python iterates a `set` of request ids (`PYTHONHASHSEED`-dependent
//!   order) to publish the requests whose host writes completed in one
//!   `update_spills`; this port uses the order of their spills.

use indexmap::{IndexMap, IndexSet};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeSet;

use super::block::{BlockHashWithGroupId, BlockId, KvCacheBlockCopy, NULL_BLOCK_ID};
use super::block_pool::BlockPool;
use super::cdiv;
use super::error::KvCacheError;
use super::events::EventExtraKeys;
use super::manager::KvCacheManager;
use super::metrics::KvCacheMetricsCollector;
use super::request::KvRequest;
use super::single_type::{ManagerType, SingleTypeKvCacheManager};
use super::spec::KvCacheConfig;
use crate::hash::BlockHash;

/// Sealed pages this many positions behind the block-table tail stay pinned so
/// a page written by an in-flight step is never handed out under it.
pub const ACTIVE_TAIL_PAGES: usize = 2;

/// Copy one logical KV page between cache-manager and worker-owned tiers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseKvPageTransfer {
    pub transfer_id: u64,
    pub host_block_id: BlockId,
    pub resident_block_ids: Vec<BlockId>,
    pub restore: bool,
}

/// Mirror one contiguous resident-row span into the host cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseKvRowMirror {
    pub source_starts: Vec<usize>,
    pub destination_start: usize,
    pub num_rows: usize,
}

/// GPU block ids of some of a request's resident pages, per resident group.
/// A null block id means the page is read from the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseKvResidencyUpdate {
    pub pages: Vec<usize>,
    pub block_ids: Vec<Vec<BlockId>>,
}

/// The managers and pools of the KV cache coordinator, lent to the HiSparse
/// coordinator for one call.
pub(crate) struct HiSparseCtx<'a> {
    pub managers: &'a mut [SingleTypeKvCacheManager],
    /// The shared device pool.
    pub device: &'a mut BlockPool,
    /// The private pool of the host source group.
    pub host: &'a mut BlockPool,
}

impl HiSparseCtx<'_> {
    /// The host pool publishes into the device pool's event queue.
    fn forward_host_events(&mut self) {
        self.host.forward_events_to(self.device);
    }
}

/// The request data a deferred host publication reads, captured while the
/// request is at hand.
///
/// Python keeps the request itself and reads it when the host writes
/// complete, possibly after it was freed. The captured hashes and tokens
/// cover every page the publication can reach, which are final once
/// computed. Deviation: block hashes that only appear after the capture (a
/// block whose hash waits for lookahead tokens) are not published until the
/// next capture, where Python publishes them as soon as they exist.
#[derive(Clone, Debug, Default)]
pub(crate) struct PublicationRequest {
    request_id: String,
    block_hashes: Vec<BlockHash>,
    num_tokens: usize,
    num_prompt_tokens: usize,
    num_computed_tokens: usize,
    num_preemptions: usize,
    shared_prefix_boundary: usize,
    /// Only captured with KV cache events.
    events: Option<EventData>,
}

#[derive(Clone, Debug, Default)]
struct EventData {
    /// `all_token_ids` up to the publication's computed tokens.
    token_ids: Vec<i32>,
    lora_id: Option<i64>,
    lora_name: Option<String>,
    session_id: Option<String>,
    mm_feature_ranges: Vec<(usize, usize)>,
    /// `event_extra_keys` of the token ranges a publication can emit events
    /// for: each page, and the partial prompt tail.
    extra_keys: FxHashMap<(usize, usize), (Option<EventExtraKeys>, isize)>,
    /// Pages whose extra keys are captured, and the next multi-modal index.
    num_key_pages: usize,
    next_mm_idx: isize,
}

impl PublicationRequest {
    /// Capture `request` for a publication of up to `num_computed_tokens`.
    fn capture(
        &mut self,
        request: &impl KvRequest,
        num_computed_tokens: usize,
        page_size: usize,
        hash_block_size: usize,
        events: bool,
    ) {
        if self.request_id != request.request_id() {
            *self = Self {
                request_id: request.request_id().to_owned(),
                ..Self::default()
            };
        }
        // The hashes chain over their prefix: extend while the captured ones
        // are still a prefix of the request's.
        let hashes = request.block_hashes();
        let n = self.block_hashes.len();
        if n > hashes.len() || (n > 0 && self.block_hashes[n - 1] != hashes[n - 1]) {
            self.block_hashes.clear();
            self.events = None;
        }
        let n = self.block_hashes.len();
        self.block_hashes.extend_from_slice(&hashes[n..]);
        self.num_tokens = request.num_tokens();
        self.num_prompt_tokens = request.num_prompt_tokens();
        self.num_computed_tokens = request.num_computed_tokens();
        self.num_preemptions = request.num_preemptions();
        self.shared_prefix_boundary = request.shared_prefix_boundary();
        if !events {
            return;
        }
        let data = self.events.get_or_insert_with(|| EventData {
            lora_id: request.lora_id(),
            lora_name: request.lora_name(),
            session_id: request.session_id(),
            mm_feature_ranges: (0..request.num_mm_features())
                .map(|i| request.mm_feature_range(i))
                .collect(),
            ..EventData::default()
        });
        let num_tokens = num_computed_tokens.min(request.num_tokens());
        if data.token_ids.len() < num_tokens {
            let start = data.token_ids.len();
            data.token_ids.extend(request.token_ids(start, num_tokens));
        }
        let num_pages = num_computed_tokens / page_size;
        for page in data.num_key_pages..num_pages {
            let (start, end) = (page * page_size, (page + 1) * page_size);
            let (keys, next) = request.event_extra_keys(start, end, data.next_mm_idx);
            data.next_mm_idx = next;
            data.extra_keys.insert((start, end), (keys, next));
        }
        data.num_key_pages = data.num_key_pages.max(num_pages);
        // The partial prompt tail `FullAttentionManager._cache_partial_tail_block`
        // registers, for either EAGLE setting.
        let num_prompt_tokens = request.num_prompt_tokens();
        for token_limit in [num_prompt_tokens, num_prompt_tokens.saturating_sub(1)] {
            let boundary = (token_limit / hash_block_size).min(hashes.len()) * hash_block_size;
            if boundary == 0 || boundary.is_multiple_of(page_size) {
                continue;
            }
            let start = boundary - hash_block_size;
            data.extra_keys
                .entry((start, boundary))
                .or_insert_with(|| request.event_extra_keys(start, boundary, 0));
        }
    }

    fn events(&self) -> &EventData {
        self.events
            .as_ref()
            .expect("KV cache event data is captured with events enabled")
    }
}

impl KvRequest for PublicationRequest {
    fn request_id(&self) -> &str {
        &self.request_id
    }

    fn block_hashes(&self) -> &[BlockHash] {
        &self.block_hashes
    }

    fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    fn num_prompt_tokens(&self) -> usize {
        self.num_prompt_tokens
    }

    fn num_computed_tokens(&self) -> usize {
        self.num_computed_tokens
    }

    fn num_in_flight_tokens(&self) -> usize {
        0
    }

    fn num_preemptions(&self) -> usize {
        self.num_preemptions
    }

    fn shared_prefix_boundary(&self) -> usize {
        self.shared_prefix_boundary
    }

    fn prefix_cache_hit_limit(&self) -> Option<usize> {
        None
    }

    fn is_waiting_or_preempted(&self) -> bool {
        false
    }

    fn token_ids(&self, start: usize, end: usize) -> Vec<i32> {
        let tokens = &self.events().token_ids;
        let end = end.min(tokens.len());
        tokens[start.min(end)..end].to_vec()
    }

    fn lora_id(&self) -> Option<i64> {
        self.events().lora_id
    }

    fn lora_name(&self) -> Option<String> {
        self.events().lora_name.clone()
    }

    fn session_id(&self) -> Option<String> {
        self.events().session_id.clone()
    }

    fn event_extra_keys(
        &self,
        start: usize,
        end: usize,
        _start_mm_idx: isize,
    ) -> (Option<EventExtraKeys>, isize) {
        self.events()
            .extra_keys
            .get(&(start, end))
            .cloned()
            .unwrap_or_else(|| panic!("no captured extra keys for tokens [{start}, {end})"))
    }

    fn num_mm_features(&self) -> usize {
        self.events().mm_feature_ranges.len()
    }

    fn mm_feature_range(&self, idx: usize) -> (usize, usize) {
        self.events().mm_feature_ranges[idx]
    }
}

#[derive(Debug)]
struct PendingPublication {
    request: PublicationRequest,
    num_computed_tokens: usize,
    num_pages: usize,
    retention_interval: Option<usize>,
    replay_boundaries: Vec<usize>,
    detached_blocks: Option<Vec<BlockId>>,
    num_cached_blocks: usize,
}

/// Residency bookkeeping for one request.
///
/// A resident page moves through three states: dirty (only the GPU copy
/// exists), clean (`valid_pages`: the host copy is complete) and unpinned
/// (`unpinned_pages`: clean, and its allocation reference was released so
/// the pool may reuse it). `pinned_clean` tracks clean pages still holding
/// their reference, which only happens before the request can read from host.
#[derive(Debug, Default)]
pub struct RequestState {
    pub valid_pages: FxHashSet<usize>,
    pub ready_prefix_pages: usize,
    /// page -> spill id.
    pub pending_pages: FxHashMap<usize, u64>,
    publication: Option<PendingPublication>,
    pub copies_recorded_blocks: usize,
    pub pinned_clean: BTreeSet<usize>,
    pub unpinned_pages: FxHashSet<usize>,
    /// Prefix pages whose GPU copies are adopted after the admitting allocation.
    pub pages_to_adopt: usize,
    /// The worker holds this request's residency for every page below
    /// synced_pages, except the stale pages changed since it was sent.
    pub synced_pages: usize,
    pub stale_pages: BTreeSet<usize>,
}

impl RequestState {
    /// Whether a host publication is pending.
    pub fn has_publication(&self) -> bool {
        self.publication.is_some()
    }
}

/// Identifies a [`RequestState`]: Python spills hold the state object itself,
/// which outlives its request when it is freed with spills in flight.
type StateId = u64;

#[derive(Debug)]
struct PendingSpill {
    transfer_id: u64,
    page: (String, usize),
    state: StateId,
    host_block: BlockId,
    resident_blocks: Vec<BlockId>,
    restore: bool,
    expected_worker_completions: usize,
    worker_completions: usize,
    enqueue_applied: bool,
}

/// Own HiSparse host allocation, host publication, spills, and GPU residency.
///
/// GPU-resident pages are a write-back cache of the host tier. Once a page's
/// host copy is durable and its owner can read from host (it has, or has
/// asked for, a hot region), the page's allocation reference is released with
/// `BlockPool.unpin_blocks`: the block keeps being read, but the pool counts
/// it as free and may hand it out, at which point the owner stops reading it.
/// Residency changes reach the worker through `take_residency_updates`, not
/// the block table, which stays append-only.
#[derive(Debug)]
pub struct HiSparseCoordinator {
    /// Spec decode step tokens an import needs beyond its sequence.
    pub import_step_tokens: usize,
    /// Whether a connector bound this coordinator ([`Self::bind`]).
    bound: bool,
    pub num_reprefillable_tokens: usize,
    /// Group ids of the resident managers, in group order.
    pub resident_group_ids: Vec<usize>,
    /// Group ids of the hot managers, in group order.
    pub hot_group_ids: Vec<usize>,
    pub host_group_id: usize,
    pub max_spill_pages: usize,
    pub transition_watermark: usize,
    /// Host copies handed to the worker, by destination block id, until it
    /// reports having run them.
    retained_copies: FxHashMap<BlockId, (BlockId, BlockId)>,
    /// request -> prefix length an external load is still filling in.
    pending_imports: FxHashMap<String, usize>,
    /// request -> full-sequence host pages of an admitted import. A waiting
    /// import holds host pages and is never preempted, so the pages it has
    /// yet to allocate stay reserved against later admissions.
    import_pages: FxHashMap<String, usize>,
    import_reservations: FxHashMap<String, usize>,
    num_reserved_import_pages: i64,
    /// The import the admission gate last sized, kept for its allocation.
    sized_import: Option<(String, usize)>,
    /// Requests whose resident pages were all non-null when last scanned.
    /// Pages only become null at admission or import (before the first
    /// scan) and when the pool reuses an unpinned page (_lose_page).
    resident_requests: FxHashSet<String>,
    spills_to_send: Vec<SparseKvPageTransfer>,
    pending_spills: IndexMap<u64, PendingSpill>,
    request_states: FxHashMap<String, StateId>,
    /// All live states: those of `request_states` and detached ones that
    /// in-flight spills still reference.
    states: FxHashMap<StateId, RequestState>,
    next_state_id: StateId,
    next_spill_id: u64,
    /// Published host block hash -> GPU copies of that page, readable until
    /// the pool reuses them. Lets a host prefix hit come back GPU-resident.
    copies: FxHashMap<BlockHashWithGroupId, Vec<BlockId>>,
    copy_by_block: FxHashMap<BlockId, BlockHashWithGroupId>,
    /// Unpinned GPU block id -> (request, page) pairs still reading it.
    owners: FxHashMap<BlockId, IndexSet<(String, usize)>>,
}

impl HiSparseCoordinator {
    /// `HiSparseCoordinator.__init__` with `bind_host_pool`: validates the
    /// HiSparse groups and builds the host pool of the source group.
    ///
    /// Returns `None` without a host source group. Python builds the
    /// coordinator when the HiSparse connector binds to the KV cache manager;
    /// this port builds it with the manager (with `import_step_tokens` 0
    /// until [`Self::set_import_step_tokens`]).
    pub(crate) fn new(
        kv_cache_config: &KvCacheConfig,
        managers: &[SingleTypeKvCacheManager],
        device: &BlockPool,
        max_model_len: usize,
        num_reprefillable_tokens: usize,
    ) -> Result<Option<(Self, BlockPool)>, KvCacheError> {
        let invalid = |message: &str| {
            Err(KvCacheError::InvalidConfig {
                message: message.to_owned(),
            })
        };
        let groups = &kv_cache_config.kv_cache_groups;
        let mut resident_group_ids = Vec::new();
        let mut hot_group_ids = Vec::new();
        let mut host_group_id = None;
        for (group_id, (group, manager)) in groups.iter().zip(managers).enumerate() {
            match manager.manager_type {
                ManagerType::HiSparseResident => {
                    assert!(!group.host_resident);
                    resident_group_ids.push(group_id);
                }
                ManagerType::HiSparseHot => {
                    assert!(!group.host_resident);
                    hot_group_ids.push(group_id);
                }
                ManagerType::HiSparseSource => {
                    if host_group_id.is_some() {
                        return invalid("Only one HiSparse host group is supported.");
                    }
                    host_group_id = Some(group_id);
                }
                _ => {}
            }
        }
        let Some(host_group_id) = host_group_id else {
            if !resident_group_ids.is_empty() || !hot_group_ids.is_empty() {
                return invalid("No HiSparse cache group is configured.");
            }
            return Ok(None);
        };
        let Some(num_host_blocks) = kv_cache_config.hisparse_host_num_blocks else {
            return invalid("HiSparse host group needs host capacity.");
        };
        let host_manager = &managers[host_group_id];
        let mut max_spill_pages = 0;
        let mut transition_watermark = 0;
        if let Some(&first_resident) = resident_group_ids.first() {
            if host_group_id > first_resident {
                return invalid("HiSparse host group must precede resident groups.");
            }
            let resident_block_size = managers[first_resident].block_size;
            if resident_group_ids
                .iter()
                .any(|&g| managers[g].block_size != resident_block_size)
            {
                return invalid("HiSparse resident cache groups must use one block size.");
            }
            if host_manager.block_size != resident_block_size {
                return invalid("HiSparse host and resident block sizes must match.");
            }
            max_spill_pages = max_model_len / resident_block_size;
            // Requests start reading from host once the shared pool runs this
            // low, so admissions find evictable pages rather than pinned ones.
            let hot_cost: usize =
                hot_group_ids.iter().map(|&g| managers[g].hisparse.blocks_per_request).sum();
            transition_watermark = hot_cost.max(kv_cache_config.num_blocks / 10);
        }
        // `bind_host_pool`: replace the device pool this group was built with
        // by a host one.
        let mut host = BlockPool::new(
            num_host_blocks,
            host_manager.enable_caching && host_manager.kv_cache_spec.prefix_cacheable(),
            device.hash_block_size,
            device.enable_kv_cache_events,
            device
                .metrics_collector
                .as_ref()
                .map(|m| KvCacheMetricsCollector::new(m.sample_rate)),
            0,
            super::events::MEDIUM_CPU,
        );
        host.use_int_block_hashes = device.use_int_block_hashes;
        Ok(Some((
            Self {
                import_step_tokens: 0,
                bound: false,
                num_reprefillable_tokens,
                resident_group_ids,
                hot_group_ids,
                host_group_id,
                max_spill_pages,
                transition_watermark,
                retained_copies: FxHashMap::default(),
                pending_imports: FxHashMap::default(),
                import_pages: FxHashMap::default(),
                import_reservations: FxHashMap::default(),
                num_reserved_import_pages: 0,
                sized_import: None,
                resident_requests: FxHashSet::default(),
                spills_to_send: Vec::new(),
                pending_spills: IndexMap::new(),
                request_states: FxHashMap::default(),
                states: FxHashMap::default(),
                next_state_id: 0,
                next_spill_id: 0,
                copies: FxHashMap::default(),
                copy_by_block: FxHashMap::default(),
                owners: FxHashMap::default(),
            },
            host,
        )))
    }

    /// Set the spec decode step tokens an import needs beyond its sequence
    /// (`get_hisparse_coordinator(..., import_step_tokens=...)`).
    pub fn set_import_step_tokens(&mut self, import_step_tokens: usize) {
        self.import_step_tokens = import_step_tokens;
    }

    /// `get_hisparse_coordinator`: the first call fixes `import_step_tokens`
    /// (0 if `None`); later calls must pass `None` or the same value.
    pub fn bind(&mut self, import_step_tokens: Option<usize>) {
        if self.bound {
            assert!(
                import_step_tokens.is_none_or(|n| n == self.import_step_tokens),
                "the HiSparse coordinator is bound with other import_step_tokens"
            );
            return;
        }
        self.bound = true;
        self.import_step_tokens = import_step_tokens.unwrap_or(0);
    }

    fn has_residents(&self) -> bool {
        !self.resident_group_ids.is_empty()
    }

    fn get_request_state(&mut self, request_id: &str) -> StateId {
        if let Some(&id) = self.request_states.get(request_id) {
            return id;
        }
        let id = self.next_state_id;
        self.next_state_id += 1;
        self.states.insert(id, RequestState::default());
        self.request_states.insert(request_id.to_owned(), id);
        id
    }

    fn state(&mut self, id: StateId) -> &mut RequestState {
        self.states.get_mut(&id).expect("live request state")
    }

    fn host_blocks<'c>(&self, cx: &'c HiSparseCtx<'_>, request_id: &str) -> Option<&'c [BlockId]> {
        cx.managers[self.host_group_id].req_blocks(request_id)
    }

    fn host_block_size(&self, cx: &HiSparseCtx<'_>) -> usize {
        cx.managers[self.host_group_id].block_size
    }

    /// Account for a prefix hit on host pages the request now references.
    fn commit_computed_blocks(
        &mut self,
        cx: &HiSparseCtx<'_>,
        request_id: &str,
        num_host_pages: usize,
    ) {
        self.resident_requests.remove(request_id);
        if !self.has_residents() {
            return;
        }
        let num_host_blocks = self.host_blocks(cx, request_id).map_or(0, <[_]>::len);
        let num_host_pages = num_host_pages.min(num_host_blocks);
        if num_host_pages == 0 {
            return;
        }
        let id = self.get_request_state(request_id);
        let state = self.state(id);
        state.valid_pages.extend(0..num_host_pages);
        state.ready_prefix_pages = state.ready_prefix_pages.max(num_host_pages);
        // Adopting pins copies out of the free pool, so it must wait until the
        // allocation admission counted on has run (see update_residency).
        state.pages_to_adopt = state.pages_to_adopt.max(num_host_pages);
        state.copies_recorded_blocks = state.copies_recorded_blocks.max(num_host_pages);
    }

    /// Drain host copy-on-write work, retaining both endpoints.
    ///
    /// The worker runs the copies in the step that carries them, so the
    /// endpoints are released when that step's output comes back.
    pub(crate) fn take_host_block_copies(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
    ) -> Vec<KvCacheBlockCopy> {
        let pairs = cx.managers[self.host_group_id].take_host_cow_copies();
        for &(source_block, cow_block) in &pairs {
            self.retained_copies.insert(cow_block, (source_block, cow_block));
        }
        pairs
            .into_iter()
            .map(|(src_block_id, dst_block_id)| KvCacheBlockCopy {
                src_block_id,
                dst_block_id,
            })
            .collect()
    }

    /// Release the endpoints of host copies the worker reports having run.
    pub(crate) fn release_completed_host_copies(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        dst_block_ids: impl IntoIterator<Item = BlockId>,
    ) {
        let mut blocks = Vec::new();
        for dst_block_id in dst_block_ids {
            if let Some((src, dst)) = self.retained_copies.remove(&dst_block_id) {
                blocks.extend([src, dst]);
            }
        }
        if blocks.is_empty() {
            return;
        }
        cx.host.free_blocks(blocks.into_iter().rev(), false);
    }

    // ------------------------------------------------------------------
    // GPU copies of published host pages
    // ------------------------------------------------------------------

    /// Point null prefix pages at readable GPU copies of their contents.
    ///
    /// Runs after the hit length is fully reconciled, so copy hits and misses
    /// can be scattered per page without affecting the prefix hit.
    fn adopt_copies(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        request_id: &str,
        id: StateId,
        host_blocks: &[BlockId],
    ) {
        if self.copies.is_empty() {
            return;
        }
        for (host_idx, &host_block) in host_blocks.iter().enumerate() {
            let blk = cx.host.block(host_block);
            if blk.is_null {
                continue;
            }
            let Some(host_hash) = &blk.block_hash else {
                continue;
            };
            let Some(blocks) = self.copies.get(host_hash) else {
                continue;
            };
            for (&group_id, &block) in self.resident_group_ids.iter().zip(blocks) {
                if cx.managers[group_id].adopt_resident_page(request_id, host_idx, block) {
                    cx.device.touch(&[block]);
                    let state = self.states.get_mut(&id).expect("live request state");
                    state.pinned_clean.insert(host_idx);
                    if host_idx < state.synced_pages {
                        state.stale_pages.insert(host_idx);
                    }
                }
            }
        }
    }

    fn adopt_pending_copies(&mut self, cx: &mut HiSparseCtx<'_>, request_id: &str, id: StateId) {
        let pages_to_adopt = self.state(id).pages_to_adopt;
        if pages_to_adopt > 0 {
            let host_blocks = self.host_blocks(cx, request_id).unwrap_or_default();
            let host_blocks = host_blocks[..pages_to_adopt.min(host_blocks.len())].to_vec();
            self.adopt_copies(cx, request_id, id, &host_blocks);
            self.state(id).pages_to_adopt = 0;
        }
    }

    /// Index the GPU copies of just-published host blocks for later hits.
    fn record_copies(
        &mut self,
        cx: &HiSparseCtx<'_>,
        request_id: &str,
        num_computed_tokens: usize,
    ) {
        if !self.has_residents() {
            return;
        }
        let Some(host_blocks) = self.host_blocks(cx, request_id).filter(|b| !b.is_empty()) else {
            return;
        };
        let id = self.get_request_state(request_id);
        let num_host_blocks =
            (num_computed_tokens / self.host_block_size(cx)).min(host_blocks.len());
        let start = self.state(id).copies_recorded_blocks;
        for (host_idx, &host_block) in
            host_blocks.iter().enumerate().take(num_host_blocks).skip(start)
        {
            if self.state(id).pending_pages.contains_key(&host_idx) {
                continue;
            }
            self.record_copy(cx, request_id, host_idx, host_block);
        }
        let state = self.state(id);
        state.copies_recorded_blocks = state.copies_recorded_blocks.max(num_host_blocks);
    }

    fn record_copy(
        &mut self,
        cx: &HiSparseCtx<'_>,
        request_id: &str,
        page_idx: usize,
        host_block: BlockId,
    ) {
        let blk = cx.host.block(host_block);
        if blk.is_null {
            return;
        }
        let Some(host_hash) = blk.block_hash.clone() else {
            return;
        };
        let mut blocks = Vec::with_capacity(self.resident_group_ids.len());
        for &group_id in &self.resident_group_ids {
            let Some(block) = cx.managers[group_id].get_resident_page(request_id, page_idx) else {
                return;
            };
            blocks.push(block);
        }
        self.drop_copy(&host_hash);
        for &block in &blocks {
            self.copy_by_block.insert(block, host_hash.clone());
        }
        self.copies.insert(host_hash, blocks);
    }

    fn drop_copy(&mut self, host_hash: &BlockHashWithGroupId) {
        let Some(blocks) = self.copies.remove(host_hash) else {
            return;
        };
        for block in blocks {
            self.copy_by_block.remove(&block);
        }
    }

    // ------------------------------------------------------------------
    // Residency: pinning, unpinning and losing pages
    // ------------------------------------------------------------------

    /// Whether resident pages may be dropped under the request.
    fn can_read_from_host(&self, cx: &HiSparseCtx<'_>, request_id: &str) -> bool {
        !self.hot_group_ids.is_empty()
            && self.hot_group_ids.iter().all(|&g| cx.managers[g].has_hot(request_id))
    }

    /// Whether the request holds the resident pages admission reserved.
    ///
    /// Admission caps a request's resident pages at the in-flight window,
    /// assuming older pages move to host; keeping them pinned past it lets a
    /// chunked prefill outgrow the pool it was admitted into.
    fn fills_admission_window(&self, cx: &HiSparseCtx<'_>, request_id: &str) -> bool {
        self.resident_group_ids.iter().any(|&g| {
            let manager = &cx.managers[g];
            manager.num_req_blocks(request_id)
                >= manager
                    .max_admission_blocks_per_request
                    .expect("HiSparse resident managers cap admission")
        })
    }

    fn resident_page_blocks(
        &self,
        cx: &HiSparseCtx<'_>,
        request_id: &str,
        page_idx: usize,
    ) -> Option<Vec<BlockId>> {
        let mut blocks = Vec::with_capacity(self.resident_group_ids.len());
        for &group_id in &self.resident_group_ids {
            let manager = &cx.managers[group_id];
            let num_pages = manager.num_req_blocks(request_id);
            let block = manager.get_resident_page(request_id, page_idx);
            if page_idx + ACTIVE_TAIL_PAGES >= num_pages {
                return None;
            }
            blocks.push(block?);
        }
        Some(blocks)
    }

    fn unpin_page(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        request_id: &str,
        id: StateId,
        page_idx: usize,
    ) -> bool {
        let state = self.state(id);
        if state.pending_pages.contains_key(&page_idx) || !state.valid_pages.contains(&page_idx) {
            return false;
        }
        let Some(blocks) = self.resident_page_blocks(cx, request_id, page_idx) else {
            return false;
        };
        for block in blocks {
            self.owners.entry(block).or_default().insert((request_id.to_owned(), page_idx));
            cx.device.unpin_blocks(&[block]);
        }
        let state = self.state(id);
        state.pinned_clean.remove(&page_idx);
        state.unpinned_pages.insert(page_idx);
        true
    }

    fn unpin_clean_pages(&mut self, cx: &mut HiSparseCtx<'_>, request_id: &str, id: StateId) {
        // Tail first, so a prefix loses its tail pages before its head.
        let pages: Vec<usize> = self.state(id).pinned_clean.iter().rev().copied().collect();
        for page_idx in pages {
            self.unpin_page(cx, request_id, id, page_idx);
        }
    }

    fn page_became_clean(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        request_id: &str,
        id: StateId,
        page_idx: usize,
    ) {
        self.state(id).pinned_clean.insert(page_idx);
        if self.can_read_from_host(cx, request_id) {
            self.unpin_page(cx, request_id, id, page_idx);
        }
    }

    /// Handle the unpinned blocks the device pool reused since the last call
    /// (`_on_block_reused`, which Python runs from `get_new_blocks`).
    pub(crate) fn process_reused_blocks(&mut self, cx: &mut HiSparseCtx<'_>) {
        for block in cx.device.take_reused_blocks() {
            if let Some(host_hash) = self.copy_by_block.get(&block).cloned() {
                self.drop_copy(&host_hash);
            }
            for (request_id, page_idx) in self.owners.remove(&block).unwrap_or_default() {
                self.lose_page(cx, &request_id, page_idx);
            }
        }
    }

    fn lose_page(&mut self, cx: &mut HiSparseCtx<'_>, request_id: &str, page_idx: usize) {
        self.resident_requests.remove(request_id);
        let Some(&id) = self.request_states.get(request_id) else {
            return;
        };
        for &group_id in &self.resident_group_ids {
            let Some(block) = cx.managers[group_id].drop_resident_page(request_id, page_idx) else {
                continue;
            };
            if let Some(owners) = self.owners.get_mut(&block) {
                owners.shift_remove(&(request_id.to_owned(), page_idx));
            }
        }
        let state = self.state(id);
        state.unpinned_pages.remove(&page_idx);
        if page_idx < state.synced_pages {
            state.stale_pages.insert(page_idx);
        }
        for &group_id in &self.hot_group_ids {
            cx.managers[group_id].require_hot(request_id);
        }
    }

    /// Per-step residency policy for a scheduled request.
    ///
    /// A request that can read from host releases every clean sealed page to
    /// the pool. One that cannot keeps its pages pinned until the shared pool
    /// runs low, or until it holds the resident pages admission reserved for
    /// it, then asks for a hot region. Pages remain pinned until that region is
    /// allocated on a subsequent scheduling pass.
    pub(crate) fn update_residency(&mut self, cx: &mut HiSparseCtx<'_>, request_id: &str) {
        if !self.has_residents() {
            return;
        }
        let id = self.get_request_state(request_id);
        self.adopt_pending_copies(cx, request_id, id);
        if !self.can_read_from_host(cx, request_id) {
            if cx.device.get_num_free_blocks() >= self.transition_watermark
                && !self.fills_admission_window(cx, request_id)
            {
                return;
            }
            for &group_id in &self.hot_group_ids {
                cx.managers[group_id].require_hot(request_id);
            }
            return;
        }
        self.unpin_clean_pages(cx, request_id, id);
    }

    // ------------------------------------------------------------------
    // Host publication and spills
    // ------------------------------------------------------------------

    /// Run the per-step residency work for each scheduled request.
    ///
    /// `requests` pairs a request id with the number of tokens computed
    /// once the step completes. As in `KVCacheCoordinator.cache_blocks`, the
    /// last `num_reprefillable_tokens` are not final yet and are not written
    /// back.
    pub(crate) fn advance_scheduled<'r>(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        requests: impl IntoIterator<Item = (&'r str, usize)>,
    ) {
        if !self.has_residents() {
            return;
        }
        for (request_id, num_tokens) in requests {
            self.plan_prefix_materialization(
                cx,
                request_id,
                num_tokens.saturating_sub(self.num_reprefillable_tokens),
            );
            self.update_residency(cx, request_id);
        }
    }

    /// Eagerly write back every sealed page that has no host copy yet.
    pub(crate) fn plan_prefix_materialization(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        request_id: &str,
        num_computed_tokens: usize,
    ) {
        if !self.has_residents() {
            return;
        }
        let host_block_size = self.host_block_size(cx);
        let num_pages = num_computed_tokens / host_block_size;
        let importing_pages = cdiv(
            self.pending_imports.get(request_id).copied().unwrap_or(0),
            host_block_size,
        );
        let id = self.get_request_state(request_id);
        let mut budget = self.max_spill_pages.saturating_sub(self.spills_to_send.len());
        let first_page = self.state(id).ready_prefix_pages.max(importing_pages);
        for page_idx in first_page..num_pages {
            if budget == 0 {
                break;
            }
            let state = self.state(id);
            if state.pending_pages.contains_key(&page_idx) || state.valid_pages.contains(&page_idx)
            {
                continue;
            }
            if !self
                .resident_group_ids
                .iter()
                .all(|&g| cx.managers[g].get_resident_page(request_id, page_idx).is_some())
            {
                continue;
            }
            if self.plan_page_transfer(cx, request_id, page_idx, false) {
                budget -= 1;
            }
        }
    }

    /// Publish host-source hashes of the pages that are already durable.
    pub(crate) fn publish_when_ready(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        request: &impl KvRequest,
        num_computed_tokens: usize,
        retention_interval: Option<usize>,
        replay_boundaries: &[usize],
    ) {
        let request_id = request.request_id();
        let page_size = self.host_block_size(cx);
        let hash_block_size = cx.host.hash_block_size;
        let events = cx.host.enable_kv_cache_events;
        let id = self.get_request_state(request_id);
        let state = self.state(id);
        // Reuse the previous capture: it only grows while the request runs.
        let mut captured = state.publication.take().map(|p| p.request).unwrap_or_default();
        captured.capture(
            request,
            num_computed_tokens,
            page_size,
            hash_block_size,
            events,
        );
        state.publication = Some(PendingPublication {
            request: captured,
            num_computed_tokens,
            num_pages: num_computed_tokens / page_size,
            retention_interval,
            replay_boundaries: replay_boundaries.to_vec(),
            detached_blocks: None,
            num_cached_blocks: 0,
        });
        self.publish_host_blocks_if_ready(cx, request_id);
    }

    /// Publish the durable prefix of the pending publication.
    ///
    /// Host writes trail a long prefill by about a chunk, so waiting for the
    /// whole computed prefix would publish nothing until the prefill ends, and
    /// a prefill preempted before then would recompute from the start.
    fn publish_host_blocks_if_ready(&mut self, cx: &mut HiSparseCtx<'_>, request_id: &str) {
        let Some(&id) = self.request_states.get(request_id) else {
            return;
        };
        let host_block_size = self.host_block_size(cx);
        let state = self.states.get_mut(&id).expect("live request state");
        let ready_prefix_pages = state.ready_prefix_pages;
        let Some(publication) = &state.publication else {
            return;
        };
        let num_tokens = publication.num_computed_tokens.min(ready_prefix_pages * host_block_size);
        // `HiSparseSourceManager.publish_blocks`.
        cx.managers[self.host_group_id].cache_blocks(
            cx.host,
            &publication.request,
            num_tokens,
            publication.retention_interval,
            &publication.replay_boundaries,
        );
        cx.forward_host_events();
        let done = ready_prefix_pages >= publication.num_pages;
        self.record_copies(cx, request_id, num_tokens);
        if done {
            self.state(id).publication = None;
        }
    }

    /// Note a prefix an external load is populating in host pages.
    fn record_pending_host_import(&mut self, request_id: &str, num_tokens: usize) {
        self.resident_requests.remove(request_id);
        self.pending_imports.insert(request_id.to_owned(), num_tokens);
        if let Some((sized_id, num_pages)) = self.sized_import.take()
            && sized_id == request_id
        {
            self.import_pages.insert(sized_id, num_pages);
        }
    }

    /// Note the full-sequence host pages of an import being admitted.
    ///
    /// The admission gate runs in the same `allocate_slots` call as the
    /// allocation, so the next `record_pending_host_import` is this
    /// request's if it is admitted; a size left by a rejected admission is
    /// discarded.
    fn size_import(&mut self, request_id: &str, num_pages: usize) {
        self.sized_import = Some((request_id.to_owned(), num_pages));
    }

    /// Host pages admitted imports still need for their sequence.
    pub fn reserved_import_pages(&self) -> i64 {
        self.num_reserved_import_pages
    }

    fn update_import_reservation(&mut self, request_id: &str, num_held_pages: usize) {
        let Some(&pages) = self.import_pages.get(request_id) else {
            return;
        };
        self.set_import_reservation(request_id, pages.saturating_sub(num_held_pages));
        if num_held_pages >= pages {
            self.import_pages.remove(request_id);
        }
    }

    fn set_import_reservation(&mut self, request_id: &str, num_pages: usize) {
        let previous = self.import_reservations.remove(request_id).unwrap_or(0);
        self.num_reserved_import_pages += num_pages as i64 - previous as i64;
        if num_pages > 0 {
            self.import_reservations.insert(request_id.to_owned(), num_pages);
        }
    }

    pub(crate) fn finish_host_import(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        request_id: &str,
        failed: bool,
    ) {
        if let Some(num_tokens) = self.pending_imports.remove(request_id)
            && !failed
        {
            self.complete_host_import(cx, request_id, num_tokens);
        }
    }

    /// Publish externally populated host pages after connector completion.
    pub(crate) fn complete_host_import(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        request_id: &str,
        num_computed_tokens: usize,
    ) {
        if !self.has_residents() {
            return;
        }
        let block_size = cx.managers[self.resident_group_ids[0]].block_size;
        let num_pages = num_computed_tokens / block_size;
        let id = self.get_request_state(request_id);
        let state = self.state(id);
        state.valid_pages = (0..num_pages).collect();
        state.ready_prefix_pages = num_pages;
        if num_computed_tokens > 0 {
            // A local prefix hit's surviving GPU copies cost no copy, so adopt
            // them before promoting the rest.
            self.adopt_pending_copies(cx, request_id, id);
            let tail_page = (num_computed_tokens - 1) / block_size;
            let promoted = self.promote_imported_pages(cx, request_id, tail_page);
            for page_idx in std::iter::once(tail_page).chain(promoted) {
                self.plan_page_transfer(cx, request_id, page_idx, true);
            }
        }
        self.publish_host_blocks_if_ready(cx, request_id);
    }

    /// Back an import's first `num_pages` host pages with new GPU pages.
    ///
    /// Imports land in the host tier. While the GPU pool has room above the
    /// transition watermark, the imported pages are restored into resident
    /// pages too, so decode reads them like a locally computed prefix. They are
    /// clean copies of durable host pages: the residency policy unpins them,
    /// and the pool reclaims them only after every other free block.
    fn promote_imported_pages(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        request_id: &str,
        num_pages: usize,
    ) -> Vec<usize> {
        let num_managers = self.resident_group_ids.len();
        let spare = cx.device.get_num_free_blocks() as i64 - self.transition_watermark as i64;
        let host_blocks = cx.managers[self.host_group_id].req_blocks_entry(request_id).clone();
        let mut num_pages = num_pages.min(host_blocks.len());
        for &group_id in &self.resident_group_ids {
            num_pages = num_pages.min(cx.managers[group_id].num_req_blocks(request_id));
        }
        let id = self.get_request_state(request_id);
        let limit = (spare.max(0) as usize) / num_managers;
        let pages: Vec<usize> = (0..num_pages)
            .filter(|&page_idx| {
                host_blocks[page_idx] != NULL_BLOCK_ID
                    && !self.states[&id].pending_pages.contains_key(&page_idx)
                    && self
                        .resident_group_ids
                        .iter()
                        .all(|&g| cx.managers[g].get_resident_page(request_id, page_idx).is_none())
            })
            .take(limit)
            .collect();
        if pages.is_empty() {
            return pages;
        }
        let blocks = cx.device.get_new_blocks(pages.len() * num_managers);
        self.process_reused_blocks(cx);
        for (i, &page_idx) in pages.iter().enumerate() {
            for (slot, &group_id) in self.resident_group_ids.iter().enumerate() {
                let adopted = cx.managers[group_id].adopt_resident_page(
                    request_id,
                    page_idx,
                    blocks[i * num_managers + slot],
                );
                assert!(adopted, "Cannot promote page {page_idx} of {request_id}");
            }
            let state = self.state(id);
            if page_idx < state.synced_pages {
                state.stale_pages.insert(page_idx);
            }
        }
        pages
    }

    fn plan_page_transfer(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        request_id: &str,
        page_idx: usize,
        restore: bool,
    ) -> bool {
        let id = self.get_request_state(request_id);
        if self.state(id).pending_pages.contains_key(&page_idx) {
            return false;
        }
        let Some(&host_block) = self.host_blocks(cx, request_id).and_then(|b| b.get(page_idx))
        else {
            return false;
        };
        if host_block == NULL_BLOCK_ID {
            return false;
        }
        let mut blocks = Vec::with_capacity(self.resident_group_ids.len());
        for &group_id in &self.resident_group_ids {
            let Some(block) = cx.managers[group_id].get_resident_page(request_id, page_idx) else {
                return false;
            };
            blocks.push(block);
        }

        cx.host.touch(&[host_block]);
        for &block in &blocks {
            cx.device.touch(&[block]);
        }
        let spill_id = self.next_spill_id;
        self.next_spill_id += 1;
        let plan = SparseKvPageTransfer {
            transfer_id: spill_id,
            host_block_id: host_block,
            resident_block_ids: blocks.clone(),
            restore,
        };
        self.pending_spills.insert(
            spill_id,
            PendingSpill {
                transfer_id: spill_id,
                page: (request_id.to_owned(), page_idx),
                state: id,
                host_block,
                resident_blocks: blocks,
                restore,
                expected_worker_completions: 0,
                worker_completions: 0,
                enqueue_applied: false,
            },
        );
        self.state(id).pending_pages.insert(page_idx, spill_id);
        self.spills_to_send.push(plan);
        true
    }

    // ------------------------------------------------------------------
    // Worker-facing views
    // ------------------------------------------------------------------

    /// Map scheduled rows to scheduler-owned resident and host blocks.
    ///
    /// `requests` are `(request_id, num_computed_tokens,
    /// num_scheduled_tokens)`.
    pub(crate) fn build_row_mirrors<'r>(
        &self,
        cx: &HiSparseCtx<'_>,
        requests: impl IntoIterator<Item = (&'r str, usize, usize)>,
    ) -> Vec<SparseKvRowMirror> {
        if !self.has_residents() {
            return Vec::new();
        }
        let block_size = cx.managers[self.resident_group_ids[0]].block_size;
        let mut mirrors = Vec::new();
        for (request_id, num_computed_tokens, num_scheduled_tokens) in requests {
            let Some(host_blocks) = self.host_blocks(cx, request_id) else {
                continue;
            };
            if self
                .resident_group_ids
                .iter()
                .any(|&g| cx.managers[g].req_blocks(request_id).is_none())
            {
                continue;
            }
            let mut token_position = num_computed_tokens;
            let end_position = token_position + num_scheduled_tokens;
            while token_position < end_position {
                let (page_idx, row_offset) =
                    (token_position / block_size, token_position % block_size);
                let num_rows = (block_size - row_offset).min(end_position - token_position);
                // A page without a GPU copy has no mirror source, and a page
                // without a host block has no destination. Skip just that page:
                // its neighbours in the window are still mirrorable, and a
                // window only ever moves forward, so dropping them here would
                // leave their host rows permanently stale.
                let source_starts: Option<Vec<usize>> = self
                    .resident_group_ids
                    .iter()
                    .map(|&g| {
                        cx.managers[g]
                            .get_resident_page(request_id, page_idx)
                            .map(|block| block as usize * block_size + row_offset)
                    })
                    .collect();
                let host_block = host_blocks.get(page_idx).copied().unwrap_or(NULL_BLOCK_ID);
                if let Some(source_starts) = source_starts
                    && host_block != NULL_BLOCK_ID
                {
                    mirrors.push(SparseKvRowMirror {
                        source_starts,
                        destination_start: host_block as usize * block_size + row_offset,
                        num_rows,
                    });
                }
                token_position += num_rows;
            }
        }
        mirrors
    }

    /// Return whether every scheduled request can read only resident KV.
    pub(crate) fn all_context_pages_resident<'r>(
        &mut self,
        cx: &HiSparseCtx<'_>,
        requests: impl IntoIterator<Item = (&'r str, usize, usize)>,
    ) -> bool {
        if !self.has_residents() {
            return false;
        }
        let first = &cx.managers[self.resident_group_ids[0]];
        for (request_id, num_computed_tokens, num_scheduled_tokens) in requests {
            let num_pages = cdiv(num_computed_tokens + num_scheduled_tokens, first.block_size);
            if first.num_req_blocks(request_id) < num_pages {
                return false;
            }
            if self.resident_requests.contains(request_id) {
                continue;
            }
            if self
                .resident_group_ids
                .iter()
                .any(|&g| cx.managers[g].get_residency_row(request_id, 0).contains(&NULL_BLOCK_ID))
            {
                return false;
            }
            self.resident_requests.insert(request_id.to_owned());
        }
        true
    }

    /// Resident block ids the worker lacks for the scheduled requests:
    /// stale pages plus pages appended since the last update.
    ///
    /// Requests that are not scheduled keep their changes until they are:
    /// only a scheduled request's residency is read.
    pub(crate) fn take_residency_updates<'r>(
        &mut self,
        cx: &HiSparseCtx<'_>,
        request_ids: impl IntoIterator<Item = &'r str>,
    ) -> IndexMap<String, SparseKvResidencyUpdate> {
        let mut updates = IndexMap::new();
        if !self.has_residents() {
            return updates;
        }
        for request_id in request_ids {
            let id = self.get_request_state(request_id);
            let num_pages = cx.managers[self.resident_group_ids[0]].num_req_blocks(request_id);
            let state = self.state(id);
            let start = state.synced_pages;
            if state.stale_pages.is_empty() && start >= num_pages {
                continue;
            }
            let stale_pages: Vec<usize> = state.stale_pages.iter().copied().collect();
            let block_ids = self
                .resident_group_ids
                .iter()
                .map(|&g| {
                    let manager = &cx.managers[g];
                    let mut group_block_ids: Vec<BlockId> = stale_pages
                        .iter()
                        .map(|&page| {
                            manager.get_resident_page(request_id, page).unwrap_or(NULL_BLOCK_ID)
                        })
                        .collect();
                    group_block_ids.extend(manager.get_residency_row(request_id, start));
                    group_block_ids
                })
                .collect();
            let mut pages = stale_pages;
            pages.extend(start..num_pages);
            updates.insert(
                request_id.to_owned(),
                SparseKvResidencyUpdate { pages, block_ids },
            );
            let state = self.state(id);
            state.stale_pages.clear();
            state.synced_pages = num_pages;
        }
        updates
    }

    /// Package residency decisions for the worker connector: the page
    /// transfers planned since the last command, or `None` without resident
    /// groups.
    pub fn build_offload_command(&mut self) -> Option<Vec<SparseKvPageTransfer>> {
        if !self.has_residents() {
            return None;
        }
        Some(std::mem::take(&mut self.spills_to_send))
    }

    /// Whether an in-flight spill will free GPU or host blocks on completion.
    pub(crate) fn has_pending_reclamation(&self, cx: &HiSparseCtx<'_>) -> bool {
        self.pending_spills.values().any(|pending| {
            cx.host.block(pending.host_block).ref_cnt == 1
                || (self.request_states.contains_key(&pending.page.0)
                    && self.can_read_from_host(cx, &pending.page.0))
        })
    }

    pub fn has_pending_work(&self) -> bool {
        !self.spills_to_send.is_empty()
            || !self.pending_spills.is_empty()
            || !self.retained_copies.is_empty()
    }

    /// The residency bookkeeping of a request (`request_states`).
    pub fn request_state(&self, request_id: &str) -> Option<&RequestState> {
        self.request_states.get(request_id).map(|id| &self.states[id])
    }

    /// The ids of the requests with residency bookkeeping.
    pub fn request_state_ids(&self) -> impl Iterator<Item = &str> + '_ {
        self.request_states.keys().map(String::as_str)
    }

    /// The ids of the spills in flight (`pending_spills`).
    pub fn pending_spill_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.pending_spills.keys().copied()
    }

    /// The transfers planned but not yet packaged (`spills_to_send`).
    pub fn spills_to_send(&self) -> &[SparseKvPageTransfer] {
        &self.spills_to_send
    }

    /// The published host block hashes with readable GPU copies (`copies`).
    pub fn copies(&self) -> &FxHashMap<BlockHashWithGroupId, Vec<BlockId>> {
        &self.copies
    }

    pub(crate) fn update_spills(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        enqueued_counts: impl IntoIterator<Item = (u64, usize)>,
        completed_counts: impl IntoIterator<Item = (u64, usize)>,
    ) {
        for (spill_id, count) in enqueued_counts {
            if let Some(pending) = self.pending_spills.get_mut(&spill_id)
                && pending.expected_worker_completions == 0
            {
                pending.expected_worker_completions = count;
            }
        }
        for (spill_id, count) in completed_counts {
            if let Some(pending) = self.pending_spills.get_mut(&spill_id) {
                pending.worker_completions += count;
            }
        }
        self.apply_enqueued_spills(cx);
        self.complete_host_writes(cx);
    }

    /// Drop the GPU pins once every worker has completed the transfer.
    fn apply_enqueued_spills(&mut self, cx: &mut HiSparseCtx<'_>) {
        for pending in self.pending_spills.values_mut() {
            if pending.enqueue_applied
                || pending.expected_worker_completions == 0
                || pending.worker_completions < pending.expected_worker_completions
            {
                continue;
            }
            for &block in &pending.resident_blocks {
                cx.device.free_blocks([block], false);
            }
            pending.enqueue_applied = true;
        }
    }

    fn complete_host_writes(&mut self, cx: &mut HiSparseCtx<'_>) {
        let completed: Vec<u64> = self
            .pending_spills
            .values()
            .filter(|p| {
                p.enqueue_applied
                    && p.expected_worker_completions > 0
                    && p.worker_completions >= p.expected_worker_completions
            })
            .map(|p| p.transfer_id)
            .collect();
        let mut completed_request_ids: IndexSet<String> = IndexSet::new();
        for spill_id in completed {
            let pending = self.pending_spills.shift_remove(&spill_id).expect("pending spill");
            let (request_id, page_idx) = &pending.page;
            let page_idx = *page_idx;
            self.state(pending.state).pending_pages.remove(&page_idx);
            let is_current = self.request_states.get(request_id) == Some(&pending.state);
            if is_current {
                let state = self.state(pending.state);
                if !pending.restore {
                    state.valid_pages.insert(page_idx);
                }
                if state.valid_pages.contains(&page_idx) {
                    self.page_became_clean(cx, request_id, pending.state, page_idx);
                    if pending.restore {
                        self.record_copy(cx, request_id, page_idx, pending.host_block);
                    }
                }
                completed_request_ids.insert(request_id.clone());
            } else {
                let state = self.state(pending.state);
                if state.publication.is_some() {
                    if !pending.restore {
                        state.valid_pages.insert(page_idx);
                    }
                    self.publish_detached_blocks(cx, pending.state);
                }
                if self.states[&pending.state].pending_pages.is_empty() {
                    self.states.remove(&pending.state);
                }
            }
            cx.host.free_blocks([pending.host_block], false);
        }
        for request_id in completed_request_ids {
            let id = self.request_states[&request_id];
            let state = self.state(id);
            while state.valid_pages.contains(&state.ready_prefix_pages) {
                state.ready_prefix_pages += 1;
            }
            self.publish_host_blocks_if_ready(cx, &request_id);
        }
    }

    fn publish_detached_blocks(&mut self, cx: &mut HiSparseCtx<'_>, id: StateId) {
        let host_block_size = self.host_block_size(cx);
        let host_group_id = self.host_group_id;
        let state = self.state(id);
        if state.publication.is_none() || !state.pending_pages.is_empty() {
            return;
        }
        if state.publication.as_ref().is_some_and(|p| p.detached_blocks.is_none()) {
            return;
        }
        while state.valid_pages.contains(&state.ready_prefix_pages) {
            state.ready_prefix_pages += 1;
        }
        let publication = state.publication.take().expect("checked above");
        let blocks = publication.detached_blocks.expect("checked above");
        // The host group uses full-attention retention. Only sealed pages with
        // completed copies can outlive the request, including on cancellation.
        cx.host.cache_full_blocks(
            &publication.request,
            &blocks,
            publication.num_cached_blocks,
            state.ready_prefix_pages.min(publication.num_pages),
            host_block_size,
            host_group_id,
            None,
        );
        cx.host.free_blocks(blocks.into_iter().rev(), false);
        cx.forward_host_events();
    }

    /// Detach the request; its clean pages stay readable copies in the pool.
    fn free(&mut self, cx: &mut HiSparseCtx<'_>, request_id: &str, finished: bool) {
        self.resident_requests.remove(request_id);
        self.pending_imports.remove(request_id);
        self.import_pages.remove(request_id);
        self.set_import_reservation(request_id, 0);
        let Some(id) = self.request_states.remove(request_id) else {
            return;
        };
        let host_group_id = self.host_group_id;
        let state = self.states.get_mut(&id).expect("live request state");
        let has_pending_pages = !state.pending_pages.is_empty();
        match state.publication.as_mut().filter(|_| finished && has_pending_pages) {
            Some(publication) => {
                let host_manager = &mut cx.managers[host_group_id];
                let host_blocks = host_manager.req_blocks_entry(request_id);
                let detached = host_blocks[..publication.num_pages.min(host_blocks.len())].to_vec();
                publication.num_cached_blocks =
                    host_manager.num_cached_blocks(request_id).unwrap_or(0);
                // Completed pages need leases too: another allocation may evict
                // them before the remaining copies finish and the prefix publishes.
                cx.host.touch(&detached);
                publication.detached_blocks = Some(detached);
            }
            // Preemption can cancel a scheduled forward after its spills were
            // planned, so its copies do not prove that the KV was computed.
            None => state.publication = None,
        }
        for &group_id in &self.resident_group_ids {
            let manager = &mut cx.managers[group_id];
            let blocks = manager.get_residency_row(request_id, 0);
            // Tail first, so a prefix loses its tail pages before its head.
            for page_idx in (0..blocks.len()).rev() {
                let block = blocks[page_idx];
                if block == NULL_BLOCK_ID {
                    continue;
                }
                if state.unpinned_pages.contains(&page_idx) {
                    if let Some(owners) = self.owners.get_mut(&block) {
                        owners.shift_remove(&(request_id.to_owned(), page_idx));
                    }
                } else if state.valid_pages.contains(&page_idx)
                    && !state.pending_pages.contains_key(&page_idx)
                {
                    cx.device.unpin_blocks(&[block]);
                } else {
                    continue;
                }
                manager.drop_resident_page(request_id, page_idx);
            }
        }
        if state.pending_pages.is_empty() {
            self.states.remove(&id);
        }
    }

    // ------------------------------------------------------------------
    // Overrides of the HiSparse single-type managers
    // ------------------------------------------------------------------

    /// `HiSparseSourceManager.get_num_blocks_to_allocate`.
    pub(crate) fn source_get_num_blocks_to_allocate(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        request_id: &str,
        mut num_tokens: usize,
        new_computed_blocks: &[BlockId],
        total_computed_tokens: usize,
        num_local_computed_tokens: usize,
        apply_admission_cap: bool,
    ) -> i64 {
        let manager = &cx.managers[self.host_group_id];
        let mut available = cx.host.get_num_free_blocks() as i64;
        if apply_admission_cap && total_computed_tokens > num_local_computed_tokens {
            num_tokens += self.import_step_tokens;
            available -= self.num_reserved_import_pages;
            let num_pages = cdiv(num_tokens, manager.block_size);
            self.size_import(request_id, num_pages);
        }
        let required = cx.managers[self.host_group_id].base_get_num_blocks_to_allocate(
            cx.host,
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            false,
        );
        // Host blocks come from a private pool, so they never count against
        // the GPU pool. Use the same admission sentinel as Mamba instead.
        if required > available {
            assert!(self.has_residents());
            return cx.device.num_gpu_blocks as i64 + 1;
        }
        0
    }

    /// `HiSparseSourceManager.allocate_external_computed_blocks`.
    pub(crate) fn source_allocate_external_computed_blocks(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        request_id: &str,
        num_local_computed_tokens: usize,
        num_external_computed_tokens: usize,
        record_for_zeroing: bool,
    ) {
        if num_external_computed_tokens == 0 {
            return;
        }
        // The connector writes these pages; only a successful receive makes
        // them readable, not advancing the request's computed-token count.
        self.record_pending_host_import(
            request_id,
            num_local_computed_tokens + num_external_computed_tokens,
        );
        let manager = &mut cx.managers[self.host_group_id];
        manager.allocate_external_computed_blocks(
            cx.host,
            request_id,
            num_local_computed_tokens,
            num_external_computed_tokens,
            record_for_zeroing,
        );
        let num_held = manager.req_blocks_entry(request_id).len();
        cx.forward_host_events();
        self.update_import_reservation(request_id, num_held);
    }

    /// `HiSparseSourceManager.allocate_new_blocks`.
    pub(crate) fn source_allocate_new_blocks(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        request_id: &str,
        num_tokens: usize,
    ) -> Vec<BlockId> {
        let manager = &mut cx.managers[self.host_group_id];
        let new_blocks = manager.base_allocate_new_blocks(cx.host, request_id, num_tokens);
        cx.forward_host_events();
        if !new_blocks.is_empty() {
            let num_held = cx.managers[self.host_group_id].num_req_blocks(request_id);
            self.update_import_reservation(request_id, num_held);
        }
        new_blocks
    }

    /// `HiSparseResidentManager.add_local_computed_blocks`.
    pub(crate) fn resident_add_local_computed_blocks(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        group_id: usize,
        request_id: &str,
        new_computed_blocks: &[BlockId],
        num_local_computed_tokens: usize,
    ) {
        let num_host_pages = cx.managers[group_id].resident_add_local_computed_blocks(
            request_id,
            new_computed_blocks,
            num_local_computed_tokens,
        );
        self.commit_computed_blocks(cx, request_id, num_host_pages);
    }

    /// `HiSparseSourceManager.pop_blocks_for_free` and
    /// `HiSparseResidentManager.pop_blocks_for_free`.
    pub(crate) fn pop_blocks_for_free(
        &mut self,
        cx: &mut HiSparseCtx<'_>,
        group_id: usize,
        request_id: &str,
        finished: bool,
    ) -> Vec<BlockId> {
        self.free(cx, request_id, finished);
        let manager = &mut cx.managers[group_id];
        match manager.manager_type {
            ManagerType::HiSparseResident => manager.resident_pop_blocks_for_free(request_id),
            _ => manager.pop_blocks_for_free(request_id),
        }
    }
}

/// The HiSparse coordinator of a KV cache manager with the state it works on
/// (`get_hisparse_coordinator(kv_cache_manager)`), for the HiSparse KV
/// connector.
pub struct HiSparseHandle<'a> {
    hisparse: &'a mut HiSparseCoordinator,
    cx: HiSparseCtx<'a>,
}

impl KvCacheManager {
    /// The HiSparse coordinator, with a HiSparse host source group.
    pub fn hisparse(&self) -> Option<&HiSparseCoordinator> {
        self.coordinator.hisparse.as_ref()
    }

    /// The HiSparse coordinator, to drive it.
    pub fn hisparse_mut(&mut self) -> Option<HiSparseHandle<'_>> {
        self.coordinator.sync_hisparse();
        let (hisparse, cx) = self.coordinator.hisparse_parts()?;
        Some(HiSparseHandle { hisparse, cx })
    }
}

impl HiSparseHandle<'_> {
    pub fn coordinator(&self) -> &HiSparseCoordinator {
        self.hisparse
    }

    pub fn coordinator_mut(&mut self) -> &mut HiSparseCoordinator {
        self.hisparse
    }

    /// See [`HiSparseCoordinator::set_import_step_tokens`].
    pub fn set_import_step_tokens(&mut self, import_step_tokens: usize) {
        self.hisparse.set_import_step_tokens(import_step_tokens);
    }

    /// See [`HiSparseCoordinator::bind`].
    pub fn bind(&mut self, import_step_tokens: Option<usize>) {
        self.hisparse.bind(import_step_tokens);
    }

    /// Run the per-step residency work for each scheduled request: pairs of
    /// a request id and the number of tokens computed once the step completes.
    pub fn advance_scheduled<'r>(&mut self, requests: impl IntoIterator<Item = (&'r str, usize)>) {
        self.hisparse.advance_scheduled(&mut self.cx, requests);
    }

    /// Per-step residency policy for a scheduled request.
    pub fn update_residency(&mut self, request_id: &str) {
        self.hisparse.update_residency(&mut self.cx, request_id);
    }

    /// Eagerly write back every sealed page that has no host copy yet.
    pub fn plan_prefix_materialization(&mut self, request_id: &str, num_computed_tokens: usize) {
        self.hisparse
            .plan_prefix_materialization(&mut self.cx, request_id, num_computed_tokens);
    }

    /// The page transfers of `SparseKVOffloadCommand`, or `None` without
    /// resident groups.
    pub fn build_offload_command(&mut self) -> Option<Vec<SparseKvPageTransfer>> {
        self.hisparse.build_offload_command()
    }

    /// Drain host copy-on-write work, retaining both endpoints.
    pub fn take_host_block_copies(&mut self) -> Vec<KvCacheBlockCopy> {
        self.hisparse.take_host_block_copies(&mut self.cx)
    }

    /// Release the endpoints of host copies the worker reports having run.
    pub fn release_completed_host_copies(
        &mut self,
        dst_block_ids: impl IntoIterator<Item = BlockId>,
    ) {
        self.hisparse.release_completed_host_copies(&mut self.cx, dst_block_ids);
    }

    /// Map scheduled rows (`(request_id, num_computed_tokens,
    /// num_scheduled_tokens)`) to scheduler-owned resident and host blocks.
    pub fn build_row_mirrors<'r>(
        &self,
        requests: impl IntoIterator<Item = (&'r str, usize, usize)>,
    ) -> Vec<SparseKvRowMirror> {
        self.hisparse.build_row_mirrors(&self.cx, requests)
    }

    /// Return whether every scheduled request can read only resident KV.
    pub fn all_context_pages_resident<'r>(
        &mut self,
        requests: impl IntoIterator<Item = (&'r str, usize, usize)>,
    ) -> bool {
        self.hisparse.all_context_pages_resident(&self.cx, requests)
    }

    /// Resident block ids the worker lacks for the scheduled requests.
    pub fn take_residency_updates<'r>(
        &mut self,
        request_ids: impl IntoIterator<Item = &'r str>,
    ) -> IndexMap<String, SparseKvResidencyUpdate> {
        self.hisparse.take_residency_updates(&self.cx, request_ids)
    }

    pub fn finish_host_import(&mut self, request_id: &str, failed: bool) {
        self.hisparse.finish_host_import(&mut self.cx, request_id, failed);
    }

    /// Publish externally populated host pages after connector completion.
    pub fn complete_host_import(&mut self, request_id: &str, num_computed_tokens: usize) {
        self.hisparse
            .complete_host_import(&mut self.cx, request_id, num_computed_tokens);
    }

    pub fn update_spills(
        &mut self,
        enqueued_counts: impl IntoIterator<Item = (u64, usize)>,
        completed_counts: impl IntoIterator<Item = (u64, usize)>,
    ) {
        self.hisparse.update_spills(&mut self.cx, enqueued_counts, completed_counts);
    }

    /// Whether an in-flight spill will free GPU or host blocks on completion.
    pub fn has_pending_reclamation(&self) -> bool {
        self.hisparse.has_pending_reclamation(&self.cx)
    }

    pub fn has_pending_work(&self) -> bool {
        self.hisparse.has_pending_work()
    }
}
