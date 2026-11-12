// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `single_type_kv_cache_manager.py`: the per-group managers.
//!
//! Python's class hierarchy becomes one [`SingleTypeKvCacheManager`] struct
//! with a [`ManagerType`] tag; methods that a subclass overrides dispatch on
//! the tag, keeping Python's dynamic dispatch (e.g. the base
//! `remove_skipped_blocks` calling Mamba's `_remove_blocks_in_range`).
//! Managers do not own the block pool; every method that touches blocks takes
//! it as an argument.
//!
//! The HiSparse managers (`HiSparseSourceManager`, `HiSparseHotManager`,
//! `HiSparseResidentManager`) call into their shared HiSparse coordinator, so
//! their overrides live in [`super::hisparse`] and are dispatched by the KV
//! cache coordinator; `hisparse` here holds their per-manager state.

mod hisparse;
mod hit;
mod mamba;

use rustc_hash::{FxHashMap, FxHashSet};

pub use self::hit::CacheHit;
use super::block::{BlockHashWithGroupId, BlockId, NULL_BLOCK_ID, trim_null_head};
use super::block_pool::BlockPool;
use super::cdiv;
use super::error::KvCacheError;
use super::request::KvRequest;
use super::spec::{KvCacheGroupRole, KvCacheSpec, KvCacheSpecType, MambaCacheMode};

/// The manager class registered for a spec (`KVCacheSpecRegistry`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ManagerType {
    /// `FullAttentionManager`: `FullAttentionSpec`, `MLAAttentionSpec`,
    /// `HiddenStateCacheSpec`.
    FullAttention,
    /// `RSWAManager` (a `FullAttentionManager`).
    Rswa,
    /// `SlidingWindowManager`: `SlidingWindowSpec`, `SlidingWindowMLASpec`.
    SlidingWindow,
    /// `CircularBufferManager` (a `FullAttentionManager`).
    CircularBuffer,
    /// `ChunkedLocalAttentionManager`.
    ChunkedLocalAttention,
    /// `MambaManager`.
    Mamba,
    /// `CrossAttentionManager`.
    CrossAttention,
    /// `SinkFullAttentionManager` (a `FullAttentionManager`).
    SinkFullAttention,
    /// `HiSparseSourceManager` (a `FullAttentionManager`): the host tier,
    /// with a private host block pool. Selected by the
    /// `KVCacheGroupRole.HISPARSE_SOURCE` role.
    HiSparseSource,
    /// `HiSparseHotManager`: `HiSparseHotSpec`.
    HiSparseHot,
    /// `HiSparseResidentManager`: `HiSparseResidentSpec`.
    HiSparseResident,
}

impl ManagerType {
    /// The manager class registered for a group's spec and role
    /// (`KVCacheSpecRegistry.get_manager_class`).
    pub fn for_group(spec: &KvCacheSpec, role: KvCacheGroupRole) -> Result<Self, KvCacheError> {
        // A role-registered manager takes precedence; only the HiSparse source
        // role registers one.
        if role == KvCacheGroupRole::HiSparseSource {
            return Ok(Self::HiSparseSource);
        }
        Ok(match spec.spec_type {
            KvCacheSpecType::FullAttention
            | KvCacheSpecType::MlaAttention
            | KvCacheSpecType::HiddenStateCache => Self::FullAttention,
            KvCacheSpecType::Rswa { .. } => Self::Rswa,
            KvCacheSpecType::SinkFullAttention { .. } => Self::SinkFullAttention,
            KvCacheSpecType::SlidingWindow { .. } | KvCacheSpecType::SlidingWindowMla { .. } => {
                Self::SlidingWindow
            }
            KvCacheSpecType::ChunkedLocalAttention { .. } => Self::ChunkedLocalAttention,
            KvCacheSpecType::CircularBuffer => Self::CircularBuffer,
            KvCacheSpecType::Mamba { .. } => Self::Mamba,
            KvCacheSpecType::CrossAttention => Self::CrossAttention,
            KvCacheSpecType::EncoderOnlyAttention => {
                return Err(KvCacheError::Unsupported {
                    message: "No manager registered for KVCacheSpec EncoderOnlyAttentionSpec"
                        .to_owned(),
                });
            }
            KvCacheSpecType::HiSparseHot { .. } => Self::HiSparseHot,
            KvCacheSpecType::HiSparseResident => Self::HiSparseResident,
        })
    }

    /// Whether this is one of the HiSparse managers, whose overrides the
    /// coordinator dispatches to its HiSparse coordinator.
    pub fn is_hisparse(self) -> bool {
        matches!(
            self,
            Self::HiSparseSource | Self::HiSparseHot | Self::HiSparseResident
        )
    }

    /// `supports_fine_grained_hash_lookup` class attribute.
    pub fn supports_fine_grained_hash_lookup(self) -> bool {
        matches!(
            self,
            Self::FullAttention
                | Self::Rswa
                | Self::SinkFullAttention
                | Self::Mamba
                | Self::HiSparseSource
        )
    }

    /// Keep this group's longer prefix until external cache lookup completes.
    /// The host tier keeps prefixes the device groups have already lost.
    pub fn retains_longer_hit(self) -> bool {
        self == Self::HiSparseSource
    }
}

/// Python list indexing with a possibly negative index.
fn py_index(len: usize, idx: i64) -> usize {
    let resolved = if idx < 0 { len as i64 + idx } else { idx };
    assert!(
        (0..len as i64).contains(&resolved),
        "list index out of range"
    );
    resolved as usize
}

/// A boundary state handed to a KV connector: `(req_id, group_id, block,
/// boundary_tokens)`.
pub type PendingBoundaryStateOffload = (String, usize, BlockId, usize);

/// State of `MambaManager`.
#[derive(Debug, Default)]
pub(crate) struct MambaState {
    pub(crate) mamba_cache_mode: Option<MambaCacheMode>,
    pub(crate) num_speculative_blocks: usize,
    pub(crate) has_prefill_checkpoint_blocks: bool,
    /// Mamba checkpoints follow Eagle's global replay boundary.
    pub(crate) drop_eagle_checkpoint_block: bool,
    pub(crate) cached_blocks_this_step: FxHashSet<BlockHashWithGroupId>,
    /// Mapping from request ID to the index of the block allocated in the
    /// previous step. (Align mode only, like the following fields.)
    pub(crate) last_state_block_idx: FxHashMap<String, i64>,
    pub(crate) num_retired_blocks: FxHashMap<String, usize>,
    /// The set of the requests that have been allocated blocks
    pub(crate) allocated_block_reqs: FxHashSet<String>,
    /// checkpoint position and reserved block index for the current
    /// allocation.
    pub(crate) checkpoints: FxHashMap<String, (usize, usize)>,
    /// Requests that registered their own last-prompt-boundary partial
    /// tail (producers). A later CoW hands its private copy to the
    /// connector; a request that finishes first hands off this table
    /// source directly.
    pub(crate) producer_partial_tail_reqs: FxHashMap<String, (BlockId, usize)>,
}

impl MambaState {
    fn is_align(&self) -> bool {
        self.mamba_cache_mode == Some(MambaCacheMode::Align)
    }
}

/// A manager that handles the kv cache management logic of one specific
/// type of attention layer.
#[derive(Debug)]
pub struct SingleTypeKvCacheManager {
    pub manager_type: ManagerType,
    /// The scheduling granularity (LCM of all group block sizes); a multiple
    /// of this manager's `block_size`.
    pub scheduler_block_size: usize,
    /// Hybrid fine-grained lookup may lower this after all participating
    /// managers have been validated by the coordinator.
    pub cache_hit_alignment_tokens: usize,
    pub dcp_world_size: usize,
    pub pcp_world_size: usize,
    /// The block size for this manager; used for actual block allocation.
    pub block_size: usize,
    pub kv_cache_spec: KvCacheSpec,
    pub enable_caching: bool,
    /// Recycling-aware per-request block cap used by
    /// `get_num_blocks_to_allocate`. Only set for spec types that recycle
    /// blocks across chunks (SWA, chunked-local).
    pub max_admission_blocks_per_request: Option<usize>,
    /// Record newly allocated block ids only when worker-side zeroing will
    /// consume them and this manager holds a spec type that gets zeroed.
    record_new_block_ids: bool,
    pub new_block_ids: Vec<BlockId>,
    /// Mapping from request ID to blocks to track the blocks allocated
    /// for each request, so that we can free the blocks when the request
    /// is finished.
    pub(crate) req_to_blocks: FxHashMap<String, Vec<BlockId>>,
    /// {req_id: The number of cached blocks for this given request}
    /// This is used to track the number of cached blocks for each request.
    /// This is only used to track the RUNNING requests, we do not track the
    /// data for preempted ones.
    pub(crate) num_cached_block: FxHashMap<String, usize>,
    pub kv_cache_group_id: usize,
    /// Whether this group's prefix-cache hits drop the EAGLE/MTP lookahead
    /// block. Only consulted by managers whose hit logic is sparse within an
    /// aligned segment (SWA). Initialized lazily by the coordinator after
    /// determining the attention groups.
    pub use_eagle: bool,
    /// `CacheConfig.enable_mamba_shared_prefix_checkpoint`, narrowed and set
    /// by `KVCacheManager`; only an EAGLE Mamba "align" group ever gets it.
    pub shared_prefix_checkpoint: bool,
    /// Partial-hit copy-on-write bookkeeping. Populated only by fine-grained
    /// managers (full attention, mamba "align"); harmlessly empty elsewhere.
    pub(crate) partial_hit_reqs: FxHashMap<String, (usize, BlockId)>,
    pending_cow_copies: Vec<(BlockId, BlockId)>,
    /// Boundary-state offload hand-off for external KV connectors. A mamba
    /// "align" block table is not append-only (interior states are
    /// nulled/freed and speculative blocks relocate in place), so a
    /// connector cannot resolve its state blocks positionally. Record
    /// (request, group, block, exact token boundary) for each committed
    /// boundary state so a connector can offload the right block under the
    /// right hash. Populated only by mamba "align".
    pending_boundary_state_offloads: Vec<PendingBoundaryStateOffload>,
    pub(crate) mamba: MambaState,
    /// State of the HiSparse hot and resident managers.
    pub(crate) hisparse: hisparse::HiSparseManagerState,
    /// `RSWASpec.rswa_window`.
    rswa_window: usize,
}

impl SingleTypeKvCacheManager {
    /// `get_manager_for_kv_cache_spec` followed by the manager constructor.
    pub fn new(
        kv_cache_spec: KvCacheSpec,
        role: KvCacheGroupRole,
        max_in_flight_tokens: usize,
        max_model_len: usize,
        enable_caching: bool,
        kv_cache_group_id: usize,
        scheduler_block_size: usize,
        dcp_world_size: usize,
        pcp_world_size: usize,
        needs_kv_cache_zeroing: bool,
    ) -> Result<Self, KvCacheError> {
        let manager_type = ManagerType::for_group(&kv_cache_spec, role)?;
        // SlidingWindow / ChunkedLocalAttention managers recycle blocks;
        // the runtime admission cap must match the recycling-aware bound the
        // startup pool sizer uses (single source of truth: the spec method).
        let max_admission_blocks_per_request =
            kv_cache_spec.max_admission_blocks_per_request(max_in_flight_tokens, max_model_len);
        let dcp_world_size = if kv_cache_spec.dcp_sharded {
            dcp_world_size
        } else {
            1
        };
        let pcp_world_size = if kv_cache_spec.dcp_sharded {
            pcp_world_size
        } else {
            1
        };
        let block_size = kv_cache_spec.block_size * dcp_world_size;
        // Deviation: `SinkFullAttentionManager.__init__` does not accept
        // `needs_kv_cache_zeroing` (Python raises TypeError when the
        // coordinator passes it); here it simply records no ids.
        let record_new_block_ids = needs_kv_cache_zeroing
            && kv_cache_spec.is_attention()
            && !matches!(kv_cache_spec.spec_type, KvCacheSpecType::CircularBuffer)
            && manager_type != ManagerType::SinkFullAttention;

        let mut mamba = MambaState::default();
        let mut hisparse = hisparse::HiSparseManagerState::default();
        let mut rswa_window = 0;
        match kv_cache_spec.spec_type {
            KvCacheSpecType::Mamba {
                mamba_cache_mode,
                num_speculative_blocks,
                num_prefill_checkpoint_blocks,
                ..
            } => {
                assert!(
                    mamba_cache_mode == MambaCacheMode::Align || !enable_caching,
                    "Mamba prefix caching requires mamba_cache_mode='align'"
                );
                mamba.mamba_cache_mode = Some(mamba_cache_mode);
                mamba.num_speculative_blocks = num_speculative_blocks;
                mamba.has_prefill_checkpoint_blocks =
                    mamba_cache_mode == MambaCacheMode::Align && num_prefill_checkpoint_blocks > 0;
            }
            KvCacheSpecType::Rswa { rswa_window: w } => rswa_window = w,
            KvCacheSpecType::HiSparseHot { blocks_per_request } => {
                hisparse.blocks_per_request = blocks_per_request;
            }
            KvCacheSpecType::SinkFullAttention { sink_len } => {
                assert!(
                    sink_len.is_some_and(|s| s > 0 && s.is_multiple_of(block_size)),
                    "sink_len must be a positive multiple of block_size"
                );
            }
            _ => {}
        }

        Ok(Self {
            manager_type,
            scheduler_block_size,
            cache_hit_alignment_tokens: scheduler_block_size,
            dcp_world_size,
            pcp_world_size,
            block_size,
            kv_cache_spec,
            enable_caching,
            max_admission_blocks_per_request,
            record_new_block_ids,
            new_block_ids: Vec::new(),
            req_to_blocks: FxHashMap::default(),
            num_cached_block: FxHashMap::default(),
            kv_cache_group_id,
            use_eagle: false,
            shared_prefix_checkpoint: false,
            partial_hit_reqs: FxHashMap::default(),
            pending_cow_copies: Vec::new(),
            pending_boundary_state_offloads: Vec::new(),
            mamba,
            hisparse,
            rswa_window,
        })
    }

    fn is_mamba_align(&self) -> bool {
        self.manager_type == ManagerType::Mamba && self.mamba.is_align()
    }

    /// Whether positional offload scans can follow this block table.
    ///
    /// True means already-scanned indices will not be reused for a different
    /// token range. Managers may still null blocks that left their retention
    /// window, as long as the cursor's positional history remains valid.
    pub fn has_positionally_stable_blocks(&self) -> bool {
        // Align-mode Mamba can null interior states and relocate speculative
        // blocks in place. Other modes retain positional identity.
        !self.is_mamba_align()
    }

    /// The request's block table, if any.
    pub fn req_blocks(&self, request_id: &str) -> Option<&[BlockId]> {
        self.req_to_blocks.get(request_id).map(Vec::as_slice)
    }

    /// The ids of the requests with a block table (`req_to_blocks` keys).
    pub fn req_ids(&self) -> impl Iterator<Item = &str> {
        self.req_to_blocks.keys().map(String::as_str)
    }

    /// The number of the request's blocks that are cached, if tracked.
    pub fn num_cached_blocks(&self, request_id: &str) -> Option<usize> {
        self.num_cached_block.get(request_id).copied()
    }

    /// Whether the manager tracks the request as running.
    pub fn has_cached_block_entry(&self, request_id: &str) -> bool {
        self.num_cached_block.contains_key(request_id)
    }

    /// `self.req_to_blocks[request_id]` on the Python `defaultdict`, which
    /// creates an empty entry.
    pub(crate) fn req_blocks_entry(&mut self, request_id: &str) -> &mut Vec<BlockId> {
        if !self.req_to_blocks.contains_key(request_id) {
            self.req_to_blocks.insert(request_id.to_owned(), Vec::new());
        }
        self.req_to_blocks.get_mut(request_id).unwrap()
    }

    pub(crate) fn num_req_blocks(&self, request_id: &str) -> usize {
        self.req_to_blocks.get(request_id).map_or(0, Vec::len)
    }

    fn get_num_evictable_blocks(pool: &BlockPool, blocks: &[BlockId]) -> i64 {
        blocks
            .iter()
            .filter(|&&b| {
                let blk = pool.block(b);
                blk.ref_cnt == 0 && !blk.is_null
            })
            .count() as i64
    }

    fn has_partial_local_hit(
        &self,
        new_computed_blocks: &[BlockId],
        num_local_computed_tokens: usize,
    ) -> bool {
        // The local prefix-cache hit ends inside one of this manager's
        // blocks: the shared tail block needs CoW.
        !new_computed_blocks.is_empty()
            && !num_local_computed_tokens.is_multiple_of(self.block_size)
    }

    /// Get the number of blocks needed to be allocated for the request.
    ///
    /// Args:
    ///     request_id: The request ID.
    ///     num_tokens: The total number of tokens that need a slot (including
    ///         tokens that are already allocated).
    ///     new_computed_blocks: The new computed blocks just hitting the
    ///         prefix caching.
    ///     total_computed_tokens: Include both local and external computed
    ///         tokens.
    ///     num_local_computed_tokens: The number of local prefix-cache computed
    ///         tokens.
    ///     num_tokens_main_model: The number of tokens for the main model (aka target
    ///         model in spec decode). w/o spec decode, it is num_tokens;
    ///         with spec decode, it is num_tokens - num_lookahead_tokens.
    ///     apply_admission_cap: If True, clamp by `num_required_blocks` by
    ///         `max_admission_blocks_per_request`for recycling-aware specs
    ///         (SWA, chunked-local).
    ///     prefill_end: The token index the request's prefill ends at, the
    ///         same value the scheduler splits chunks against. Mamba reserves
    ///         a prefill checkpoint only on the chunk reaching it; 0 (dense
    ///         retention) reserves one on every chunk.
    ///
    /// Returns:
    ///     The number of blocks to allocate. Signed: Mamba "align" can return
    ///     a negative count, which Python adds up as is.
    pub fn get_num_blocks_to_allocate(
        &mut self,
        pool: &BlockPool,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &[BlockId],
        total_computed_tokens: usize,
        num_local_computed_tokens: usize,
        num_tokens_main_model: usize,
        apply_admission_cap: bool,
        prefill_end: usize,
    ) -> i64 {
        match self.manager_type {
            ManagerType::CircularBuffer => {
                if self.num_req_blocks(request_id) > 0 {
                    0
                } else {
                    1
                }
            }
            ManagerType::Mamba => self.mamba_get_num_blocks_to_allocate(
                pool,
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap,
                prefill_end,
            ),
            _ => self.base_get_num_blocks_to_allocate(
                pool,
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
                apply_admission_cap,
            ),
        }
    }

    pub(crate) fn base_get_num_blocks_to_allocate(
        &self,
        pool: &BlockPool,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &[BlockId],
        total_computed_tokens: usize,
        num_local_computed_tokens: usize,
        apply_admission_cap: bool,
    ) -> i64 {
        let mut num_required_blocks = cdiv(num_tokens, self.block_size);
        if apply_admission_cap && let Some(cap) = self.max_admission_blocks_per_request {
            // Recycling-aware specs (SWA, chunked-local) cap the per-request
            // reservation here so admission matches the startup pool sizer
            // (`SlidingWindowSpec.max_admission_blocks_per_request` / its
            // chunked-local counterpart). `remove_skipped_blocks` runs from
            // `allocate_slots` before each chunk's `get_num_blocks_to_allocate`,
            // so per-request peak real-held blocks <= this cap, which keeps
            // `sum(reservations) <= pool` <=> `sum(peak_real_held) <= pool`.
            // Drift between the two would re-introduce the deadlock from
            // issue #39734 or, worse, mid-prefill OOM.
            num_required_blocks = num_required_blocks.min(cap);
        }
        let num_req_blocks = self.num_req_blocks(request_id);

        if self.num_cached_block.contains_key(request_id) {
            // Fast-path: a running request won't have any new prefix-cache hits.
            assert!(new_computed_blocks.is_empty());
            // NOTE: With speculative decoding, request's blocks may be allocated
            // for draft tokens which are later rejected. In this case,
            // num_required_blocks may be smaller than num_req_blocks.
            return (num_required_blocks as i64 - num_req_blocks as i64).max(0);
        }

        let num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens);
        let num_local_computed_blocks = new_computed_blocks.len() + num_req_blocks;
        // Number of whole blocks that are skipped by the attention window.
        // If nothing is skipped, this is 0.
        let num_skipped_blocks = num_skipped_tokens.div_euclid(self.block_size as i64);
        // We need blocks for the non-skipped suffix. If there are still
        // local-computed blocks inside the window, they contribute to the
        // required capacity; otherwise, skipped blocks dominate.
        let mut num_new_blocks = (num_required_blocks as i64
            - num_skipped_blocks.max(num_local_computed_blocks as i64))
        .max(0);

        // Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
        // of blocks are skipped; `num_req_blocks` of those may already be in
        // `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
        let num_skipped_new_computed_blocks =
            (num_skipped_blocks - num_req_blocks as i64).max(0) as usize;

        // If a computed block is an eviction candidate (in the free queue and
        // ref_cnt == 0), it will be removed from the free queue when touched by
        // the allocated request, so we must count it in the free-capacity check.
        let skipped = num_skipped_new_computed_blocks.min(new_computed_blocks.len());
        let num_evictable_blocks =
            Self::get_num_evictable_blocks(pool, &new_computed_blocks[skipped..]);
        if self.has_partial_local_hit(new_computed_blocks, num_local_computed_tokens) {
            // Reserve the extra block that allocate_new_blocks pulls for the
            // partial-hit CoW redirect.
            num_new_blocks += 1;
        }
        num_new_blocks + num_evictable_blocks
    }

    /// Add the locally cached (prefix-hit) blocks to the request:
    /// 1. Touch the computed blocks (paired with adding them to `req_blocks`)
    ///    so their ref_cnt exactly tracks the referencing requests.
    ///
    /// 1.5. (Optional) For sliding window, skipped blocks are padded with nulls.
    /// 2. Add the remaining computed blocks.
    pub fn add_local_computed_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        new_computed_blocks: &[BlockId],
        num_local_computed_tokens: usize,
        num_external_computed_tokens: usize,
    ) {
        match self.manager_type {
            ManagerType::CircularBuffer => return,
            ManagerType::CrossAttention => {
                // We do not cache blocks for cross-attention to be shared between
                // requests, so  `new_computed_blocks` should always be empty.
                assert!(new_computed_blocks.is_empty());
                return;
            }
            _ => {}
        }
        // The coordinator only calls this for first-time allocations (running
        // requests are short-circuited there), so the request has no blocks yet.
        assert!(self.req_blocks_entry(request_id).is_empty());
        let num_total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens;
        let num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens);
        let num_skipped_blocks = num_skipped_tokens.div_euclid(self.block_size as i64);
        let mut new_computed_blocks = new_computed_blocks;
        if num_skipped_blocks > 0 {
            // It is possible that all new computed blocks are skipped when
            // num_skipped_blocks > len(new_computed_blocks).
            let skip = (num_skipped_blocks as usize).min(new_computed_blocks.len());
            new_computed_blocks = &new_computed_blocks[skip..];
        }

        // Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching {
            pool.touch(new_computed_blocks);
        } else {
            assert!(
                new_computed_blocks.is_empty(),
                "Computed blocks should be empty when prefix caching is disabled"
            );
        }

        let has_partial_hit =
            self.has_partial_local_hit(new_computed_blocks, num_local_computed_tokens);
        let block_size = self.block_size;
        let req_blocks = self.req_blocks_entry(request_id);
        // Skip blocks are padded with null blocks.
        req_blocks.resize(num_skipped_blocks.max(0) as usize, NULL_BLOCK_ID);
        // Add the remaining computed blocks.
        req_blocks.extend_from_slice(new_computed_blocks);
        let num_blocks = req_blocks.len();
        // All cached hits (including skipped nulls) are already cached; mark
        // them so cache_blocks() will not try to re-cache blocks that already
        // have a block_hash set.
        self.num_cached_block.insert(request_id.to_owned(), num_blocks);
        if has_partial_hit {
            // Record the partial tail for the CoW redirect in
            // allocate_new_blocks; cap the cached count at the full blocks so
            // cache_blocks() re-caches the private copy once full.
            let block_idx = num_local_computed_tokens / block_size;
            let tail = *new_computed_blocks.last().unwrap();
            self.partial_hit_reqs.insert(request_id.to_owned(), (block_idx, tail));
            self.num_cached_block.insert(request_id.to_owned(), block_idx);
        }
    }

    /// Allocate new blocks for external (KV-connector) computed tokens.
    ///
    /// Must run only after every group's local blocks have been touched via
    /// `add_local_computed_blocks`, so this group's `get_new_blocks` cannot
    /// evict another group's cache-hit blocks (issue #33775).
    ///
    /// Args:
    ///     request_id: The request ID.
    ///     num_local_computed_tokens: The number of local computed tokens.
    ///     num_external_computed_tokens: The number of external computed tokens.
    ///     record_for_zeroing: Whether the new blocks need zeroing. False when
    ///         the load writes them after this step, which zeroing would race.
    pub fn allocate_external_computed_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        num_local_computed_tokens: usize,
        num_external_computed_tokens: usize,
        record_for_zeroing: bool,
    ) {
        match self.manager_type {
            ManagerType::CircularBuffer => {
                self.claim_ring_block(pool, request_id, record_for_zeroing);
                return;
            }
            // Cross-attention does not use prefix caching / external KV loads.
            ManagerType::CrossAttention => return,
            _ => {}
        }
        let num_total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens;
        let num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens);
        let mut num_external = num_external_computed_tokens as i64;
        if num_skipped_tokens > 0 {
            // Some external computed tokens may be skipped too.
            num_external =
                (num_total_computed_tokens as i64 - num_skipped_tokens).min(num_external);
        }
        if num_external <= 0 {
            return;
        }

        let block_size = self.block_size;
        let num_req_blocks = self.req_blocks_entry(request_id).len();
        let num_new_blocks =
            cdiv(num_total_computed_tokens, block_size).saturating_sub(num_req_blocks);
        let allocated_blocks = pool.get_new_blocks(num_new_blocks);
        self.req_blocks_entry(request_id).extend_from_slice(&allocated_blocks);
        if self.record_new_block_ids && record_for_zeroing {
            self.new_block_ids.extend_from_slice(&allocated_blocks);
        }
    }

    /// Allocate new blocks for the request to give it at least `num_tokens`
    /// token slots.
    ///
    /// Args:
    ///     request_id: The request ID.
    ///     num_tokens: The total number of tokens that need a slot (including
    ///         tokens that are already allocated).
    ///     num_tokens_main_model: The number of tokens for the main model (aka target
    ///         model in spec decode). w/o spec decode, it is num_tokens;
    ///         with spec decode, it is num_tokens - num_lookahead_tokens.
    ///
    /// Returns:
    ///     The new allocated blocks.
    pub fn allocate_new_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        num_tokens: usize,
        num_tokens_main_model: usize,
    ) -> Vec<BlockId> {
        match self.manager_type {
            ManagerType::CircularBuffer => self.claim_ring_block(pool, request_id, true),
            ManagerType::Mamba => {
                self.mamba_allocate_new_blocks(pool, request_id, num_tokens, num_tokens_main_model)
            }
            _ => self.base_allocate_new_blocks(pool, request_id, num_tokens),
        }
    }

    pub(crate) fn base_allocate_new_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        num_tokens: usize,
    ) -> Vec<BlockId> {
        let mut cow_blocks = Vec::new();
        if let Some((block_idx, source_block)) = self.partial_hit_reqs.remove(request_id) {
            // Partial hit: redirect the shared tail to a private CoW block.
            // Replacing in place keeps the length-based allocation below
            // correct; the extra block was reserved by
            // get_num_blocks_to_allocate.
            let cow_block = pool.get_new_blocks(1)[0];
            self.apply_cow(pool, request_id, block_idx, source_block, cow_block);
            self.new_block_ids.push(cow_block);
            cow_blocks.push(cow_block);
        }

        let num_required_blocks = cdiv(num_tokens, self.block_size);
        let num_req_blocks = self.req_blocks_entry(request_id).len();
        if num_required_blocks <= num_req_blocks {
            return cow_blocks;
        }
        let new_blocks = pool.get_new_blocks(num_required_blocks - num_req_blocks);
        self.req_blocks_entry(request_id).extend_from_slice(&new_blocks);
        if self.record_new_block_ids {
            self.new_block_ids.extend_from_slice(&new_blocks);
        }
        cow_blocks.extend(new_blocks);
        cow_blocks
    }

    /// `CircularBufferManager`: claims the ring's single block per request.
    fn claim_ring_block(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        record_for_zeroing: bool,
    ) -> Vec<BlockId> {
        if !self.req_blocks_entry(request_id).is_empty() {
            return Vec::new();
        }
        let new_blocks = pool.get_new_blocks(1);
        self.req_blocks_entry(request_id).extend_from_slice(&new_blocks);
        if self.record_new_block_ids && record_for_zeroing {
            self.new_block_ids.extend_from_slice(&new_blocks);
        }
        new_blocks
    }

    /// Whether this manager's new blocks are zeroed by the worker.
    pub fn records_new_block_ids(&self) -> bool {
        // Host blocks are never zeroed by the worker.
        self.record_new_block_ids && self.manager_type != ManagerType::HiSparseSource
    }

    /// Drain and return block IDs allocated since the last call.
    pub fn take_new_block_ids(&mut self) -> Vec<BlockId> {
        let ids = std::mem::take(&mut self.new_block_ids);
        if self.manager_type == ManagerType::HiSparseSource {
            return Vec::new();
        }
        ids
    }

    /// Drain pending CoW source and destination block pairs.
    pub fn take_pending_cow_copies(&mut self) -> Vec<(BlockId, BlockId)> {
        // Host copies never reach the worker's generic block-copy path; the
        // HiSparse coordinator drains them (`take_host_cow_copies`).
        if self.manager_type == ManagerType::HiSparseSource {
            return Vec::new();
        }
        std::mem::take(&mut self.pending_cow_copies)
    }

    /// `HiSparseSourceManager.take_host_cow_copies`: drain host copies for
    /// the HiSparse coordinator to hand to the connector.
    pub(crate) fn take_host_cow_copies(&mut self) -> Vec<(BlockId, BlockId)> {
        std::mem::take(&mut self.pending_cow_copies)
    }

    /// Drain producer boundary-state hand-offs.
    ///
    /// Entries are `(req_id, group_id, block, boundary_tokens)`.
    ///
    /// Only mamba "align" populates this. The blocks are not kept alive by
    /// the request block table for the whole request, so a caller that reads
    /// them asynchronously must pin them first.
    pub fn take_pending_boundary_state_offloads(&mut self) -> Vec<PendingBoundaryStateOffload> {
        std::mem::take(&mut self.pending_boundary_state_offloads)
    }

    /// Finalize a producer partial tail when its request finishes.
    pub fn finalize_partial_tail_offload(
        &mut self,
        request_id: &str,
        num_computed_tokens: usize,
        num_in_flight_tokens: usize,
    ) -> Option<(usize, BlockId, usize)> {
        if !self.is_mamba_align() {
            return None;
        }
        let (source_block, boundary_tokens) =
            self.mamba.producer_partial_tail_reqs.remove(request_id)?;
        if num_in_flight_tokens != 0 || num_computed_tokens != boundary_tokens {
            return None;
        }
        Some((self.kv_cache_group_id, source_block, boundary_tokens))
    }

    /// Redirect a partial prefix-cache hit to a private CoW block.
    ///
    /// Both copy endpoints stay retained until the copy has run on the worker,
    /// so a same-step free cannot recycle them: `source_block` keeps its
    /// hit-ref, `cow_block` takes an extra ref beyond the one handed to the
    /// request.
    fn apply_cow(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        block_idx: usize,
        source_block: BlockId,
        cow_block: BlockId,
    ) {
        let req_blocks = self.req_blocks_entry(request_id);
        assert!(block_idx < req_blocks.len());
        assert!(req_blocks[block_idx] == source_block);
        let src = pool.block(source_block);
        assert!(!src.is_null && src.ref_cnt > 0);
        req_blocks[block_idx] = cow_block;
        self.pending_cow_copies.push((source_block, cow_block));
        pool.block_mut(cow_block).ref_cnt += 1;
    }

    /// Cache the blocks for the request.
    ///
    /// Args:
    ///     request: The request.
    ///     num_tokens: The total number of tokens that need to be cached
    ///         (including tokens that are already cached).
    ///     retention_interval: Sparse local-checkpoint granularity. `None`
    ///         keeps dense checkpointing; `0` keeps only the latest replay
    ///         boundary; a positive multiple of `scheduler_block_size` keeps
    ///         a tail once per that-sized segment. Only SWA acts on it.
    ///     replay_boundaries: Positions a later request replaying this prompt
    ///         can resume at, from `replay_boundaries`.
    pub fn cache_blocks(
        &mut self,
        pool: &mut BlockPool,
        request: &impl KvRequest,
        num_tokens: usize,
        retention_interval: Option<usize>,
        replay_boundaries: &[usize],
    ) {
        match self.manager_type {
            ManagerType::CircularBuffer => {}
            ManagerType::CrossAttention => {
                // We do not cache blocks for cross-attention to be shared between
                // requests, so this method is not relevant.
                panic!("Should not be called as prefix caching is disabled.")
            }
            ManagerType::Mamba => self.mamba_cache_blocks(
                pool,
                request,
                num_tokens,
                retention_interval,
                replay_boundaries,
            ),
            // `_HiSparseAuxiliaryManager`: the host source owns prefix caching.
            ManagerType::HiSparseHot | ManagerType::HiSparseResident => {}
            ManagerType::FullAttention
            | ManagerType::Rswa
            | ManagerType::SinkFullAttention
            | ManagerType::HiSparseSource => {
                self.base_cache_blocks(
                    pool,
                    request,
                    num_tokens,
                    retention_interval,
                    replay_boundaries,
                );
                if self.block_size == pool.hash_block_size {
                    return;
                }
                self.full_attention_cache_partial_tail_block(pool, request, num_tokens);
            }
            ManagerType::SlidingWindow | ManagerType::ChunkedLocalAttention => self
                .base_cache_blocks(
                    pool,
                    request,
                    num_tokens,
                    retention_interval,
                    replay_boundaries,
                ),
        }
    }

    fn base_cache_blocks(
        &mut self,
        pool: &mut BlockPool,
        request: &impl KvRequest,
        num_tokens: usize,
        retention_interval: Option<usize>,
        replay_boundaries: &[usize],
    ) {
        if !self.kv_cache_spec.prefix_cacheable() {
            return;
        }
        let request_id = request.request_id();
        let num_cached_blocks = self.num_cached_block.get(request_id).copied().unwrap_or(0);
        let num_hashed_tokens = request.block_hashes().len() * pool.hash_block_size;
        let num_full_blocks = num_tokens.min(num_hashed_tokens) / self.block_size;

        if num_cached_blocks >= num_full_blocks {
            return;
        }

        // Token boundaries whose reachable tail must be retained under sparse
        // retention: every position a replaying sibling can resume at (see
        // ``replay_boundaries``) and any detected shared-prefix junction.
        let mut reachable_boundaries: Vec<i64> =
            replay_boundaries.iter().map(|&b| b as i64).collect();
        let shared_prefix_boundary = request.shared_prefix_boundary();
        if shared_prefix_boundary != 0 {
            reachable_boundaries.push(shared_prefix_boundary as i64);
            if self.use_eagle {
                // A sibling that diverges right after the junction cannot use
                // the block past it as the EAGLE peek, so its hit lands one
                // block lower.
                reachable_boundaries.push(shared_prefix_boundary as i64 - self.block_size as i64);
            }
        }

        let block_mask = self.manager_type.reachable_block_mask(
            num_cached_blocks,
            num_full_blocks,
            Some(self.cache_hit_alignment_tokens),
            &self.kv_cache_spec,
            self.use_eagle,
            retention_interval,
            &reachable_boundaries,
            self.dcp_world_size,
            None,
        );
        let (block_size, group_id) = (self.block_size, self.kv_cache_group_id);
        let blocks = self.req_blocks_entry(request_id);
        pool.cache_full_blocks(
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            group_id,
            block_mask.as_deref(),
        );

        self.num_cached_block.insert(request_id.to_owned(), num_full_blocks);
    }

    /// Cache the prompt tail when it ends inside a cache block.
    ///
    /// Only the final prompt hash boundary is registered as a partial
    /// prefix-cache entry; intermediate hash boundaries inside the same cache
    /// block are intentionally skipped.
    fn full_attention_cache_partial_tail_block(
        &mut self,
        pool: &mut BlockPool,
        request: &impl KvRequest,
        num_tokens: usize,
    ) {
        let hash_block_size = pool.hash_block_size;
        // A resend matches at most `prompt_len - 1` tokens, but EAGLE lookups
        // read the whole prompt before dropping a hash unit.
        let token_limit = if self.use_eagle {
            request.num_prompt_tokens()
        } else {
            request.num_prompt_tokens().saturating_sub(1)
        };
        let boundary_tokens =
            (token_limit / hash_block_size).min(request.block_hashes().len()) * hash_block_size;
        if boundary_tokens == 0 || boundary_tokens > num_tokens {
            return;
        }
        if boundary_tokens.is_multiple_of(self.block_size) {
            return;
        }

        let block_idx = boundary_tokens / self.block_size;
        let blocks = self.req_blocks_entry(request.request_id());
        if block_idx >= blocks.len() {
            return;
        }
        let block = blocks[block_idx];
        pool.cache_partial_block(
            request,
            block,
            boundary_tokens,
            self.kv_cache_group_id,
            self.block_size,
            false,
        );
    }

    /// Pop the request's bookkeeping and return its blocks without yet
    /// returning them to the block pool. The caller is responsible for
    /// eventually passing the returned blocks to `block_pool.free_blocks`,
    /// freeing them in reverse order (so that tail blocks are evicted first).
    ///
    /// Returns:
    ///     The request's blocks in allocation order, without the leading
    ///     null blocks (e.g. a sliding-window table's skipped prefix).
    pub fn pop_blocks_for_free(&mut self, request_id: &str) -> Vec<BlockId> {
        if self.is_mamba_align() {
            let mamba = &mut self.mamba;
            mamba.allocated_block_reqs.remove(request_id);
            mamba.last_state_block_idx.remove(request_id);
            mamba.num_retired_blocks.remove(request_id);
            mamba.checkpoints.remove(request_id);
            mamba.producer_partial_tail_reqs.remove(request_id);
            // An offer is only guaranteed to hold committed bytes until the end
            // of the pass that made it. This request's blocks are going back to
            // the pool now, so drop its not-yet-offered hand-offs rather than
            // let a connector claim a block another request may already have
            // been handed.
            self.pending_boundary_state_offloads.retain(|entry| entry.0 != request_id);
        }
        // Default to [] in case a request is freed (aborted) before alloc.
        let req_blocks = self.req_to_blocks.remove(request_id).unwrap_or_default();
        self.num_cached_block.remove(request_id);
        self.partial_hit_reqs.remove(request_id);
        let nhl = super::block::null_head_len(&req_blocks);
        if nhl == 0 {
            req_blocks
        } else {
            trim_null_head(&req_blocks).to_vec()
        }
    }

    /// Free the blocks for the request.
    pub fn free(&mut self, pool: &mut BlockPool, request_id: &str) {
        // Free blocks in reverse order so that the tail blocks are freed first.
        let blocks = self.pop_blocks_for_free(request_id);
        pool.free_blocks(blocks.into_iter().rev(), false);
    }

    /// Free blocks in `[first_block, last_block)` and replace with null_block.
    ///
    /// Iterates backward so newly-evictable tail blocks are reached even after
    /// earlier blocks in the range were nulled in a prior call.
    fn remove_blocks_in_range(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        first_block: usize,
        last_block: usize,
    ) {
        if self.manager_type == ManagerType::Mamba && self.mamba.is_align() {
            self.mamba_remove_blocks_in_range(pool, request_id, first_block, last_block);
            return;
        }
        let Some(blocks) = self.req_to_blocks.get_mut(request_id) else {
            return;
        };
        if first_block >= last_block {
            return;
        }
        let last_block = last_block.min(blocks.len());

        let mut freed = Vec::new();
        for i in (first_block..last_block).rev() {
            if blocks[i] == NULL_BLOCK_ID {
                break;
            }
            freed.push(blocks[i]);
            blocks[i] = NULL_BLOCK_ID;
        }
        if !freed.is_empty() {
            pool.free_blocks(freed, false);
        }
    }

    /// Remove and free blocks no longer needed for attention computation.
    ///
    /// The removed blocks should be replaced by null_block.
    ///
    /// Args:
    ///     request_id: The request ID.
    ///     processed_computed_tokens: Computed-token prefix length covering
    ///         fully processed and committed tokens only (safe to free).
    ///     num_prompt_tokens: Optional prompt length for attention types (e.g.
    ///         R-SWA) that evict a middle gap rather than a head prefix. Ignored
    ///         by the default implementation.
    pub fn remove_skipped_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        processed_computed_tokens: usize,
        num_prompt_tokens: Option<usize>,
    ) {
        match self.manager_type {
            ManagerType::CircularBuffer => {}
            ManagerType::Rswa => self.rswa_remove_skipped_blocks(
                pool,
                request_id,
                processed_computed_tokens,
                num_prompt_tokens,
            ),
            ManagerType::Mamba => {
                self.mamba_remove_skipped_blocks(pool, request_id, processed_computed_tokens)
            }
            _ => self.base_remove_skipped_blocks(pool, request_id, processed_computed_tokens),
        }
    }

    fn base_remove_skipped_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        processed_computed_tokens: usize,
    ) {
        // Remove the blocks that will be skipped during attention computation.
        let num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens);
        if num_skipped_tokens <= 0 {
            // This indicates that ALL tokens are inside attention window.
            // Thus we do not need to free any blocks outside attention window.
            // A typical case is full attention that we never free any token
            // before the request is finished.
            return;
        }
        let block_size = self.block_size;
        let num_blocks = self.req_blocks_entry(request_id).len();
        let num_skipped_blocks = num_skipped_tokens as usize / block_size;
        // `num_skipped_tokens` may include tokens that haven't been allocated yet
        // (e.g., when the attention window moves into the external computed tokens
        // range), so we must cap to the number of blocks that currently exist for
        // this request.
        let num_skipped_blocks = num_skipped_blocks.min(num_blocks);
        self.remove_blocks_in_range(pool, request_id, 0, num_skipped_blocks);
    }

    /// Free gap blocks that are no longer needed for attention.
    ///
    /// Gap = blocks entirely within
    ///     [ceil(prefix_len / block_size) * block_size,
    ///      max(prefix_len, processed_computed_tokens - rswa_window))
    ///
    /// Freed blocks are replaced with null_block in req_to_blocks so the
    /// block_table passed to FA4 is valid (null_block KV is all-zero;
    /// rswa_mask_mod marks gap positions as non-visible so FA4 skips them).
    fn rswa_remove_skipped_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        processed_computed_tokens: usize,
        num_prompt_tokens: Option<usize>,
    ) {
        let Some(num_prompt_tokens) = num_prompt_tokens else {
            self.base_remove_skipped_blocks(pool, request_id, processed_computed_tokens);
            return;
        };
        let bs = self.block_size;
        // First block fully after the prefill boundary.
        let first_gap_block = cdiv(num_prompt_tokens, bs);
        // Decode window start position; blocks before this are evictable.
        let window_start =
            num_prompt_tokens.max(processed_computed_tokens.saturating_sub(self.rswa_window));
        let last_gap_block = window_start / bs; // exclusive upper bound
        self.remove_blocks_in_range(pool, request_id, first_gap_block, last_gap_block);
    }

    /// Get the number of tokens that will be skipped for attention computation.
    ///
    /// Signed because Mamba returns `num_computed_tokens - 1`, which is -1
    /// for 0.
    pub fn get_num_skipped_tokens(&self, num_computed_tokens: usize) -> i64 {
        let n = num_computed_tokens as i64;
        match self.kv_cache_spec.spec_type {
            // Python dispatches on the manager class; specs map 1:1 below.
            _ if self.manager_type == ManagerType::CircularBuffer => 0,
            KvCacheSpecType::SlidingWindow {
                sliding_window,
                extra_retained_tokens,
            }
            | KvCacheSpecType::SlidingWindowMla {
                sliding_window,
                extra_retained_tokens,
                ..
            } => {
                // For sliding window, this corresponds to the tokens that are prior to
                // the current sliding window.
                //
                // The trailing edge of the window is extended by `extra_retained_tokens`
                // so that those extra trailing tokens' blocks are retained (but not
                // attended).
                (n - sliding_window as i64 + 1 - extra_retained_tokens as i64).max(0)
            }
            KvCacheSpecType::ChunkedLocalAttention {
                attention_chunk_size,
            } => {
                // For chunked local attention, this corresponds to the tokens that are on
                // the left side of the current chunk.
                n / attention_chunk_size as i64 * attention_chunk_size as i64
            }
            // Mamba only need to keep the state of the last computed token, so we
            // return num_computed_tokens - 1.
            KvCacheSpecType::Mamba { .. } => n - 1,
            // The default behavior is to not skip any tokens.
            _ => 0,
        }
    }

    pub fn new_step_starts(&mut self) {
        if self.manager_type == ManagerType::Mamba {
            self.mamba.cached_blocks_this_step.clear();
        }
    }
}
