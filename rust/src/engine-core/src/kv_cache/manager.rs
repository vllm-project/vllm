// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `kv_cache_manager.py`: `KVCacheBlocks` and `KVCacheManager`.

use std::borrow::Cow;

use indexmap::IndexMap;

use super::block::{BlockId, KvCacheBlockCopy, trim_null_head};
use super::block_pool::BlockPool;
use super::cdiv;
use super::coordinator::{CoordinatorKind, KvCacheCoordinator};
use super::error::KvCacheError;
use super::events::KvCacheEvent;
use super::metrics::{KvCacheMetricsCollector, PrefixCacheStats};
use super::request::KvRequest;
use super::single_type::ManagerType;
use super::spec::KvCacheConfig;

/// The allocation result of KVCacheManager, work as the interface between
/// Scheduler and KVCacheManager, to hide KVCacheManager's internal data
/// structure from the Scheduler.
///
/// `blocks[i][j]` refers to the i-th kv_cache_group and the j-th block of
/// tokens. Python shares one immutable empty instance; here an instance whose
/// groups are all empty plays that role (see [`KvCacheManager::allocate_slots`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvCacheBlocks {
    pub blocks: Vec<Vec<BlockId>>,
}

impl KvCacheBlocks {
    pub fn new(blocks: Vec<Vec<BlockId>>) -> Self {
        Self { blocks }
    }

    /// Whether every group is empty (Python: the shared empty instance).
    pub fn is_empty(&self) -> bool {
        self.blocks.iter().all(Vec::is_empty)
    }

    /// Adds two KVCacheBlocks instances.
    pub fn add(&self, other: &KvCacheBlocks) -> KvCacheBlocks {
        KvCacheBlocks::new(
            self.blocks
                .iter()
                .zip(&other.blocks)
                .map(|(a, b)| a.iter().chain(b).copied().collect())
                .collect(),
        )
    }

    /// Converts the KVCacheBlocks instance to block_ids.
    ///
    /// Args:
    ///     allow_none: Return None when every selected group is empty.
    ///     group_ids: KV cache groups to include. Includes all groups by default.
    pub fn get_block_ids(
        &self,
        allow_none: bool,
        group_ids: Option<&[usize]>,
    ) -> Option<Vec<Vec<BlockId>>> {
        let groups: Vec<&Vec<BlockId>> = match group_ids {
            None => self.blocks.iter().collect(),
            Some(ids) => ids.iter().map(|&g| &self.blocks[g]).collect(),
        };
        if allow_none && groups.iter().all(|g| g.is_empty()) {
            return None;
        }
        Some(groups.into_iter().cloned().collect())
    }

    /// Get block_ids of unhashed blocks from KVCacheBlocks instance.
    pub fn get_unhashed_block_ids(&self, pool: &BlockPool) -> Vec<BlockId> {
        assert!(self.blocks.len() == 1, "Only one group is supported");
        self.blocks[0]
            .iter()
            .copied()
            .filter(|&b| pool.block(b).block_hash.is_none())
            .collect()
    }

    /// Get block_ids of unhashed blocks from KVCacheBlocks instance.
    ///
    /// `group_pool` maps a group to its block pool (see
    /// [`KvCacheManager::group_pool`]).
    pub fn get_unhashed_block_ids_all_groups<'a>(
        &self,
        group_pool: impl Fn(usize) -> &'a BlockPool,
    ) -> Vec<Vec<BlockId>> {
        // Skip padding blocks.
        self.blocks
            .iter()
            .enumerate()
            .map(|(group_id, group)| {
                let pool = group_pool(group_id);
                trim_null_head(group)
                    .iter()
                    .copied()
                    .filter(|&b| {
                        let blk = pool.block(b);
                        blk.block_hash.is_none() && !blk.is_null
                    })
                    .collect()
            })
            .collect()
    }

    /// Creates a new KVCacheBlocks instance with no blocks.
    pub fn new_empty(&self) -> KvCacheBlocks {
        KvCacheBlocks::new(vec![Vec::new(); self.blocks.len()])
    }
}

/// The constructor arguments of `KVCacheManager` besides the config.
#[derive(Debug)]
pub struct KvCacheManagerConfig {
    pub max_model_len: usize,
    pub scheduler_block_size: usize,
    pub hash_block_size: usize,
    /// When unset, fall back to `max_model_len` so the recycling-aware cap
    /// collapses to the prior (uncapped) admission behavior.
    pub max_in_flight_tokens: Option<usize>,
    pub enable_caching: bool,
    pub use_eagle: bool,
    pub num_hash_lookahead_tokens: usize,
    pub num_prefill_lookahead: usize,
    pub log_stats: bool,
    pub enable_kv_cache_events: bool,
    /// `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`.
    pub kv_events_use_int_block_hashes: bool,
    pub dcp_world_size: usize,
    pub pcp_world_size: usize,
    pub metrics_collector: Option<KvCacheMetricsCollector>,
    pub watermark: f64,
    pub enable_mamba_shared_prefix_checkpoint: bool,
}

impl KvCacheManagerConfig {
    /// The Python defaults for the optional arguments.
    pub fn new(max_model_len: usize, scheduler_block_size: usize, hash_block_size: usize) -> Self {
        Self {
            max_model_len,
            scheduler_block_size,
            hash_block_size,
            max_in_flight_tokens: None,
            enable_caching: true,
            use_eagle: false,
            num_hash_lookahead_tokens: 0,
            num_prefill_lookahead: 0,
            log_stats: false,
            enable_kv_cache_events: false,
            kv_events_use_int_block_hashes: false,
            dcp_world_size: 1,
            pcp_world_size: 1,
            metrics_collector: None,
            watermark: 0.0,
            enable_mamba_shared_prefix_checkpoint: false,
        }
    }
}

/// The optional arguments of `KVCacheManager.allocate_slots`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocateSlotsArgs<'a> {
    /// The number of new computed tokens just hitting the prefix caching,
    /// excluding external tokens.
    pub num_new_computed_tokens: usize,
    /// The cached blocks for the above new computed tokens, grouped as a
    /// tuple by kv cache groups.
    pub new_computed_blocks: Option<&'a KvCacheBlocks>,
    /// The number of speculative tokens to allocate.
    pub num_lookahead_tokens: usize,
    /// The number of tokens that their KV caches are not cached by vLLM but
    /// cached by the connector.
    pub num_external_computed_tokens: usize,
    /// Whether to skip caching the blocks.
    pub delay_cache_blocks: bool,
    /// The number of encoder tokens to allocate for cross-attention.
    pub num_encoder_tokens: usize,
    /// Only allocate blocks if the KV cache has enough free blocks to hold the
    /// full sequence.
    pub full_sequence_must_fit: bool,
    /// Number of free blocks that must be left available for other in-flight
    /// sequences to complete.
    pub reserved_blocks: usize,
    /// Whether any requests are already scheduled to run this step, controls
    /// whether watermark is applied. Python defaults this to True.
    pub has_scheduled_reqs: Option<bool>,
    /// Groups whose external-token blocks will be written by an async load
    /// and must not be zeroed concurrently.
    pub skip_zeroing_group_ids: &'a [usize],
}

/// A boundary state for a KV connector: `(group_id, block_id, boundary_tokens)`.
pub type BoundaryStateOffload = (usize, BlockId, usize);

#[derive(Debug)]
pub struct KvCacheManager {
    pub max_model_len: usize,
    pub enable_caching: bool,
    pub enable_kv_cache_events: bool,
    pub use_eagle: bool,
    pub log_stats: bool,
    prefix_cache_stats: Option<PrefixCacheStats>,
    pub coordinator: KvCacheCoordinator,
    /// One predicate, read by both sides of the feature, so the scheduler
    /// cannot end a chunk at a junction the manager would refuse.
    pub mamba_shared_prefix_checkpoint: bool,
    pub num_kv_cache_groups: usize,
    pub retained_hit_group_ids: Vec<usize>,
    pub kv_cache_config: KvCacheConfig,
    /// Watermark: minimum number of KV cache blocks to keep free when
    /// admitting waiting/preempted requests, to avoid frequent preemptions.
    pub watermark_blocks: usize,
    kv_cache_event_metadata: Vec<(&'static str, Option<usize>)>,
}

impl KvCacheManager {
    pub fn new(
        kv_cache_config: KvCacheConfig,
        config: KvCacheManagerConfig,
    ) -> Result<Self, KvCacheError> {
        let max_in_flight_tokens = config.max_in_flight_tokens.unwrap_or(config.max_model_len);
        let mut coordinator = KvCacheCoordinator::new(
            kv_cache_config.clone(),
            config.max_model_len,
            max_in_flight_tokens,
            config.use_eagle,
            config.enable_caching,
            config.enable_kv_cache_events,
            config.dcp_world_size,
            config.pcp_world_size,
            config.scheduler_block_size,
            config.hash_block_size,
            config.metrics_collector,
            config.num_hash_lookahead_tokens,
            config.num_prefill_lookahead,
            true,
        )?;
        coordinator.block_pool.use_int_block_hashes = config.kv_events_use_int_block_hashes;
        if let Some(host) = &mut coordinator.host_pool {
            host.use_int_block_hashes = config.kv_events_use_int_block_hashes;
        }
        // Multi-module MTP is excluded because ``cache_blocks`` then hands the
        // manager ``num_computed - num_reprefillable`` rather than the chunk end.
        let mamba_shared_prefix_checkpoint = config.enable_mamba_shared_prefix_checkpoint
            && !coordinator.eagle_group_ids.is_empty()
            && coordinator.enable_partial_hash_hits
            && coordinator.num_reprefillable_tokens == 0;
        if mamba_shared_prefix_checkpoint {
            for manager in &mut coordinator.single_type_managers {
                if manager.manager_type == ManagerType::Mamba {
                    manager.shared_prefix_checkpoint = true;
                }
            }
        }
        let retained_hit_group_ids = coordinator
            .single_type_managers
            .iter()
            .filter(|m| m.manager_type.retains_longer_hit())
            .map(|m| m.kv_cache_group_id)
            .collect();
        assert!(config.watermark >= 0.0, "watermark must be non-negative");
        let watermark_blocks = (config.watermark * kv_cache_config.num_blocks as f64) as usize;
        let kv_cache_event_metadata = kv_cache_config
            .kv_cache_groups
            .iter()
            .map(|g| (g.kv_cache_spec.kind_str(), g.kv_cache_spec.sliding_window()))
            .collect();
        Ok(Self {
            max_model_len: config.max_model_len,
            enable_caching: config.enable_caching,
            enable_kv_cache_events: config.enable_kv_cache_events,
            use_eagle: config.use_eagle,
            log_stats: config.log_stats,
            prefix_cache_stats: config.log_stats.then(PrefixCacheStats::default),
            coordinator,
            mamba_shared_prefix_checkpoint,
            num_kv_cache_groups: kv_cache_config.kv_cache_groups.len(),
            retained_hit_group_ids,
            kv_cache_config,
            watermark_blocks,
            kv_cache_event_metadata,
        })
    }

    pub fn block_pool(&self) -> &BlockPool {
        &self.coordinator.block_pool
    }

    /// The (device) block pool, for callers (KV connectors) that touch and
    /// free blocks directly. Blocks from [`Self::pop_blocks_for_free`] go back
    /// through [`Self::free_blocks`], which also routes HiSparse host blocks.
    pub fn block_pool_mut(&mut self) -> &mut BlockPool {
        &mut self.coordinator.block_pool
    }

    /// The private block pool of the HiSparse host source group, if any.
    pub fn host_block_pool(&self) -> Option<&BlockPool> {
        self.coordinator.host_pool.as_ref()
    }

    pub fn host_block_pool_mut(&mut self) -> Option<&mut BlockPool> {
        self.coordinator.host_pool.as_mut()
    }

    /// The block pool of a group (`manager.block_pool`): the HiSparse host
    /// pool for the host source group, else the device pool.
    pub fn group_pool(&self, group_id: usize) -> &BlockPool {
        self.coordinator.group_pool(group_id)
    }

    /// Free blocks returned by [`Self::pop_blocks_for_free`] (or retained
    /// copy endpoints), routing HiSparse host blocks
    /// ([`HOST_BLOCK_FLAG`](super::HOST_BLOCK_FLAG)) to the host pool as
    /// Python's `block_pool.free_blocks` does.
    pub fn free_blocks(
        &mut self,
        ordered_blocks: impl IntoIterator<Item = BlockId>,
        evict_first: bool,
    ) {
        self.coordinator.free_mixed_blocks(ordered_blocks, evict_first, false);
    }

    /// [`Self::free_blocks`] as `free_blocks` of the HiSparse host pool if
    /// `host`: that pool's blocks are freed first.
    pub fn free_blocks_from(
        &mut self,
        ordered_blocks: impl IntoIterator<Item = BlockId>,
        evict_first: bool,
        host: bool,
    ) {
        self.coordinator.free_mixed_blocks(ordered_blocks, evict_first, host);
    }

    /// Run `f` on the device pool, or on the HiSparse host pool if `host`,
    /// for callers that operate on a pool directly; then forwards the host
    /// pool's events and handles the blocks the device pool reused, as
    /// Python's pools do as part of the operation.
    pub fn with_pool_mut<R>(&mut self, host: bool, f: impl FnOnce(&mut BlockPool) -> R) -> R {
        let out = match (&mut self.coordinator.host_pool, host) {
            (Some(pool), true) => f(pool),
            _ => f(&mut self.coordinator.block_pool),
        };
        self.coordinator.forward_host_events();
        self.coordinator.sync_hisparse();
        out
    }

    pub fn empty_kv_cache_blocks(&self) -> KvCacheBlocks {
        KvCacheBlocks::new(vec![Vec::new(); self.num_kv_cache_groups])
    }

    /// Get the KV cache usage (between 0.0 and 1.0).
    pub fn usage(&self) -> f64 {
        self.coordinator.block_pool.get_usage()
    }

    /// Get (and reset) the prefix cache stats, or None if logging is disabled.
    pub fn make_prefix_cache_stats(&mut self) -> Option<PrefixCacheStats> {
        if !self.log_stats {
            return None;
        }
        self.prefix_cache_stats.replace(PrefixCacheStats::default())
    }

    /// Whether a local prefix cache lookup may be run for this request.
    pub fn prefix_cache_lookup_enabled(&self, request: &impl KvRequest) -> bool {
        self.enable_caching && request.prefix_cache_hit_limit() != Some(0)
    }

    /// How many of a request's tokens are already in the prefix cache.
    ///
    /// Unlike `get_computed_blocks`, this only probes the cache: it takes no
    /// references, allocates no blocks and emits no events, so it is safe to
    /// call for a request that may not be scheduled.
    pub fn get_num_cached_tokens(&self, request: &impl KvRequest) -> usize {
        if !self.prefix_cache_lookup_enabled(request) {
            return 0;
        }
        // Mirrors `get_computed_blocks`: the last token is always recomputed.
        let (_, num_computed_tokens, _) = self.coordinator.find_longest_cache_hit(
            request.block_hashes(),
            request.num_tokens().saturating_sub(1),
        );
        num_computed_tokens
    }

    pub fn record_prefix_cache_stats(&mut self, request: &impl KvRequest, num_hits: usize) {
        // Don't count a request that skipped the cache lookup.
        if !self.log_stats || !self.prefix_cache_lookup_enabled(request) {
            return;
        }
        self.prefix_cache_stats
            .as_mut()
            .expect("prefix cache stats exist when log_stats is set")
            .record(
                request.num_tokens(),
                num_hits,
                request.num_preemptions() > 0,
            );
    }

    /// Get the computed (cached) blocks for the request.
    /// Note that the computed blocks must be full.
    ///
    /// Args:
    ///     request: The request to get the computed blocks.
    ///     max_cache_hit_length: Upper bound on the hit length. Defaults to
    ///         `request.get_max_cache_hit_length()`; explicit bounds can only
    ///         reduce the request's cache hit limit.
    ///
    /// Returns:
    ///     A tuple containing:
    ///         - A list of blocks that are computed for the request.
    ///         - The number of computed tokens.
    ///         - `shared_prefix_boundary`: the block-aligned token position of
    ///           a shared prefix that a sparse-retention group (Mamba / sliding
    ///           window) has not cached yet (Marconi-style APC), or 0 if none.
    ///           Pinned so sparse prefix-cache retention does not drop
    ///           the junction and defeat cross-request reuse.
    pub fn get_computed_blocks(
        &mut self,
        request: &impl KvRequest,
        max_cache_hit_length: Option<usize>,
    ) -> (KvCacheBlocks, usize, usize) {
        // We skip finding the prefix cache hit when prefix caching is
        // disabled or the request is marked as skipping kv cache read
        // (which happens when the request requires prompt logprobs
        // or calls a pooling model with all pooling).
        if !self.prefix_cache_lookup_enabled(request) {
            return (self.empty_kv_cache_blocks(), 0, 0);
        }

        // NOTE: When all tokens hit the cache, we must recompute the last token
        // to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        let max_cache_hit_length = request.get_max_cache_hit_length(max_cache_hit_length);
        let (computed_blocks, num_new_computed_tokens, num_uncached) = self
            .coordinator
            .find_longest_cache_hit(request.block_hashes(), max_cache_hit_length);

        // When kv_cache_report_mode is "full", emit BlockStored events
        // for the reused prefix cache blocks so that external consumers
        // (e.g. gateway) can learn about them.
        if num_new_computed_tokens > 0
            && self.enable_kv_cache_events
            && request.kv_cache_report_mode_full()
        {
            self.coordinator.emit_cached_block_events(request, &computed_blocks);
        }

        // The junction to pin is where the lagging sparse-retention group stops
        // (``num_new_computed_tokens``) plus the uncached shared prefix -- i.e.
        // the longest single-group hit. Sub-block gaps are left to the mask,
        // which floors to the alignment boundary (a no-op there).
        let shared_prefix_boundary = if num_uncached > 0 {
            num_new_computed_tokens + num_uncached
        } else {
            0
        };

        (
            self.create_kv_cache_blocks(computed_blocks),
            num_new_computed_tokens,
            shared_prefix_boundary,
        )
    }

    /// Local prefix-cache lookup for a request scheduled with a KV connector.
    ///
    /// Hybrid (Mamba + full-attention) models can have per-group prefix hits
    /// diverge under block pressure: the full-attention tail may be evicted
    /// while a deeper Mamba state survives, or vice versa. Report the
    /// full-attention hit as the local prefix - the connector transfers the
    /// remaining suffix and the Mamba state is transferred unconditionally by
    /// nixl's `_apply_prefix_caching` - and flag when that hit ran deeper
    /// than a lagging group. Such a hit only has a valid Mamba state at its
    /// boundary if the connector supplies it, so the caller must fall back to
    /// `get_computed_blocks` to reconcile when no external tokens are found.
    ///
    /// Non-hybrid models and already-convergent hits use `get_computed_blocks`.
    ///
    /// Returns:
    ///     The `get_computed_blocks` triple (blocks, number of local computed
    ///     tokens, shared-prefix boundary) plus `hit_diverged`.
    pub fn get_computed_blocks_for_connector(
        &mut self,
        request: &impl KvRequest,
        max_cache_hit_length: Option<usize>,
    ) -> (KvCacheBlocks, usize, usize, bool) {
        let Some(fa_group_id) = self.coordinator.full_attention_group_id.filter(|_| {
            self.kv_cache_config.has_mamba_layers()
                && self.coordinator.kind == CoordinatorKind::Hybrid
        }) else {
            let (blocks, n, boundary) = self.get_computed_blocks(request, max_cache_hit_length);
            return (blocks, n, boundary, false);
        };

        if !self.prefix_cache_lookup_enabled(request) {
            return (self.empty_kv_cache_blocks(), 0, 0, false);
        }

        let max_cache_hit_length = request.get_max_cache_hit_length(max_cache_hit_length);
        let (computed, per_group_hits) = self
            .coordinator
            .find_longest_cache_hit_per_group(request.block_hashes(), max_cache_hit_length);
        let fa_hit = per_group_hits[fa_group_id];
        if per_group_hits.iter().any(|&hit| hit > fa_hit) {
            // A lagging group hit deeper than full attention means its
            // full-attention blocks were evicted; use the reconciled boundary
            // that every group agrees on.
            let (blocks, n, boundary) =
                self.get_computed_blocks(request, Some(max_cache_hit_length));
            return (blocks, n, boundary, false);
        }

        let num_local = fa_hit;
        let min_hit = per_group_hits.iter().copied().min().unwrap_or(0);
        // Per-group lookups do not detect an uncached shared prefix (boundary 0).
        (
            self.create_kv_cache_blocks(computed),
            num_local,
            0,
            min_hit < num_local,
        )
    }

    /// Add slots for a request with new tokens to append.
    ///
    /// Args:
    ///     request: The request to allocate slots.
    ///     num_new_tokens: The number of new tokens to be allocated and computed.
    ///     args: The optional arguments, see [`AllocateSlotsArgs`].
    ///
    /// Blocks layout:
    /// ```text
    /// ----------------------------------------------------------------------
    /// | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
    /// ----------------------------------------------------------------------
    ///                                           |   < to be computed >     |
    /// ----------------------------------------------------------------------
    ///                           |            < to be allocated >           |
    /// ----------------------------------------------------------------------
    ///                           | < to be cached (roughly, |
    ///                           | details below)>          |
    /// ----------------------------------------------------------------------
    /// | Prefix-cached tokens from either vLLM   |
    /// | or connector. Can be safely removed if  |
    /// | they are outside sliding window.        |
    /// ----------------------------------------------------------------------
    /// |   < cached by vLLM >    | not cached by |
    ///                           | vLLM, but     |
    /// | ref_cnt  | ref_cnt not  | cached by     |
    /// | increased| increased yet| connector     |
    /// ----------------------------------------------------------------------
    /// ```
    ///
    /// NOTE: for new tokens which include both verified and unverified draft
    /// tokens, we only cache the verified tokens (by capping the number at
    /// `request.num_tokens`).
    ///
    /// The allocation has three stages:
    /// - Free unnecessary blocks in `comp` and check
    ///   if we have sufficient free blocks (return None if not).
    /// - Handle prefix tokens (`comp + new_comp + ext_comp`):
    ///     - Free unnecessary blocks (e.g. outside sliding window)
    ///     - Allocate new blocks for `ext_comp` tokens inside
    ///       sliding window
    /// - Allocate new blocks for tokens to be computed (`new + lookahead`)
    ///
    /// Returns:
    ///     A list of new allocated blocks, or `None` if they do not fit.
    pub fn allocate_slots(
        &mut self,
        request: &impl KvRequest,
        num_new_tokens: usize,
        args: AllocateSlotsArgs<'_>,
    ) -> Result<Option<KvCacheBlocks>, KvCacheError> {
        let AllocateSlotsArgs {
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs,
            skip_zeroing_group_ids,
        } = args;
        let has_scheduled_reqs = has_scheduled_reqs.unwrap_or(true);
        // A step may need no slots of its own while still adopting computed
        // tokens: an async KV load, or a chunk that ends inside the replayed
        // range of a hit (SWA bounded replay).
        if num_new_tokens == 0 && num_external_computed_tokens == 0 && num_new_computed_tokens == 0
        {
            return Err(KvCacheError::InvalidArgument {
                message: "num_new_tokens must be greater than 0 when there are no computed \
                          tokens to adopt"
                    .to_owned(),
            });
        }

        let request_id = request.request_id();
        // Python distinguishes the shared empty `KVCacheBlocks` by identity; it is
        // the only all-empty instance the manager hands out, so all-empty
        // input is treated as that instance.
        let mut new_computed_block_list: Cow<'_, [Vec<BlockId>]> = Cow::Borrowed(&[]);
        let mut is_empty_singleton = true;
        if let Some(blocks) = new_computed_blocks
            && !blocks.is_empty()
        {
            is_empty_singleton = false;
            new_computed_block_list = Cow::Borrowed(&blocks.blocks);
            if !self.retained_hit_group_ids.is_empty() {
                // Only retain the prefix covered by the local and external hits.
                let reused = num_new_computed_tokens + num_external_computed_tokens;
                let groups = new_computed_block_list.to_mut();
                for &group_id in &self.retained_hit_group_ids {
                    let block_size = self.coordinator.single_type_managers[group_id].block_size;
                    groups[group_id].truncate(cdiv(reused, block_size));
                }
            }
        }

        // The number of computed tokens is the number of computed tokens plus
        // the new prefix caching hits
        let num_local_computed_tokens = request.num_computed_tokens() + num_new_computed_tokens;
        let total_computed_tokens =
            (num_local_computed_tokens + num_external_computed_tokens).min(self.max_model_len);

        // The watermark is applied to waiting/preempted requests only, and only
        // when there's at least one request already scheduled.
        let watermark_blocks = if has_scheduled_reqs && request.is_waiting_or_preempted() {
            self.watermark_blocks
        } else {
            0
        };

        // Matches the scheduler's own prefill boundary: `num_tokens - 1`
        // extends it to resumed requests replaying their output tokens.
        let prefill_end = request.num_prompt_tokens().max(request.num_tokens().saturating_sub(1));

        if full_sequence_must_fit {
            // First check and fail if the full request sequence won't fit.
            let full_num_tokens = request.num_tokens().min(self.max_model_len);
            let num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id,
                full_num_tokens,
                &new_computed_block_list,
                num_encoder_tokens,
                total_computed_tokens,
                num_local_computed_tokens,
                full_num_tokens,
                true,
                prefill_end,
            );
            let required_blocks = num_blocks_to_allocate + watermark_blocks as i64;
            if required_blocks > self.coordinator.block_pool.get_num_free_blocks() as i64 {
                return Ok(None);
            }
        }

        let num_tokens_main_model = total_computed_tokens + num_new_tokens;
        let num_tokens_need_slot =
            (num_tokens_main_model + num_lookahead_tokens).min(self.max_model_len);

        // Free the blocks that are skipped during the attention computation
        // (e.g., tokens outside the sliding window).
        // We can do this even if we cannot schedule this request due to
        // insufficient free blocks.
        // Should call this function before allocating new blocks to reduce
        // the number of evicted blocks.
        // Free on the processed-token basis: in-flight steps' attention windows
        // still read blocks below the optimistic boundary, and rejected spec
        // tokens can roll it back.
        self.coordinator.remove_skipped_blocks(
            request_id,
            total_computed_tokens.saturating_sub(request.num_in_flight_tokens()),
            Some(request.num_prompt_tokens()),
        );

        let num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id,
            num_tokens_need_slot,
            &new_computed_block_list,
            num_encoder_tokens,
            num_local_computed_tokens + num_external_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            false,
            prefill_end,
        );

        // Keep `reserved_blocks` free for other in-flight sequences, and an
        // additional watermark of headroom for waiting/preempted admissions.
        let available_blocks =
            self.coordinator.block_pool.get_num_free_blocks() as i64 - reserved_blocks as i64;
        let required_blocks = num_blocks_to_allocate + watermark_blocks as i64;
        if required_blocks > available_blocks {
            // Cannot allocate new blocks
            return Ok(None);
        }

        if !is_empty_singleton || num_external_computed_tokens > 0 {
            // Append the new computed blocks to the request blocks until now to
            // avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id,
                &new_computed_block_list,
                num_local_computed_tokens,
                num_external_computed_tokens,
                skip_zeroing_group_ids,
            );
        }

        let new_blocks = self.coordinator.allocate_new_blocks(
            request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        );

        // P/D: delay caching blocks if we have to recv from
        // remote. Update state for locally cached blocks.
        if delay_cache_blocks {
            return Ok(Some(self.create_kv_cache_blocks(new_blocks)));
        }

        // NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        // + num_external_computed_tokens + num_new_tokens, but must exclude
        // "non-committable" tokens (e.g., draft tokens that could be rejected).
        // Therefore, we cap the number at `request.num_tokens`, ensuring only
        // "finalized" tokens are cached.
        let num_tokens_to_cache =
            (total_computed_tokens + num_new_tokens).min(request.num_tokens());
        self.coordinator.cache_blocks(request, num_tokens_to_cache);

        Ok(Some(self.create_kv_cache_blocks(new_blocks)))
    }

    /// Free the blocks allocated for the request.
    /// We free the blocks in reverse order so that the tail blocks are evicted
    /// first when caching is enabled.
    ///
    /// `finished`: whether the request finished (`request.is_finished()`);
    /// HiSparse still publishes a finished request's host prefix once its
    /// in-flight host writes complete.
    pub fn free(&mut self, request_id: &str, finished: bool) {
        self.coordinator.free(request_id, finished);
    }

    /// Remove the blocks that are no longer needed from `blocks` and replace
    /// the removed blocks with null_block.
    pub fn remove_skipped_blocks(
        &mut self,
        request_id: &str,
        processed_computed_tokens: usize,
        num_prompt_tokens: Option<usize>,
    ) {
        self.coordinator.remove_skipped_blocks(
            request_id,
            processed_computed_tokens,
            num_prompt_tokens,
        );
    }

    /// Pop the request's bookkeeping and return its blocks without
    /// returning them to the block pool. The caller must eventually free
    /// them with [`Self::free_blocks`] in reverse order (so that tail blocks
    /// are evicted first).
    ///
    /// `finished`: see [`Self::free`].
    pub fn pop_blocks_for_free(&mut self, request_id: &str, finished: bool) -> Vec<BlockId> {
        self.coordinator.pop_blocks_for_free(request_id, finished)
    }

    /// Evict blocks from the prefix cache by their block IDs.
    pub fn evict_blocks(&mut self, block_ids: impl IntoIterator<Item = BlockId>) {
        self.coordinator.block_pool.evict_blocks(block_ids);
    }

    /// Reset prefix cache. This function may be used in RLHF
    /// flows to invalidate prefix caching after the weights are updated,
    /// or used for resetting prefix caching status for benchmarking.
    pub fn reset_prefix_cache(&mut self) -> bool {
        if !self.coordinator.reset_prefix_cache() {
            return false;
        }
        if self.log_stats {
            self.prefix_cache_stats
                .as_mut()
                .expect("prefix cache stats exist when log_stats is set")
                .reset = true;
        }
        true
    }

    /// Take the KV cache events from the block pool.
    pub fn take_events(&mut self) -> Vec<KvCacheEvent> {
        let mut events = self.coordinator.block_pool.take_events();
        for event in &mut events {
            let KvCacheEvent::BlockStored(event) = event else {
                continue;
            };
            let Some(group_idx) = event.group_idx else {
                continue;
            };
            // TODO: Python logs a warning for an unknown group index.
            let Some(&(kind, sliding_window)) = self.kv_cache_event_metadata.get(group_idx) else {
                continue;
            };
            // Annotate here so BlockPool can keep emitting structural cache
            // events without owning semantic KV cache spec metadata.
            event.kv_cache_spec_kind = Some(kind);
            event.kv_cache_spec_sliding_window = sliding_window;
        }
        events
    }

    /// The request's live block table of each group, without copying (Python's
    /// `get_blocks` hands out the live lists).
    pub fn get_block_tables(&self, request_id: &str) -> Vec<&[BlockId]> {
        self.coordinator
            .single_type_managers
            .iter()
            .map(|m| m.req_blocks(request_id).unwrap_or_default())
            .collect()
    }

    /// Get the blocks of a request.
    pub fn get_blocks(&self, request_id: &str) -> KvCacheBlocks {
        self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))
    }

    /// Get the block ids of a request.
    pub fn get_block_ids(&self, request_id: &str) -> Vec<Vec<BlockId>> {
        self.coordinator.get_blocks(request_id)
    }

    /// Get block ids covering the request's computed tokens.
    pub fn get_block_ids_for_computed_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Vec<Vec<BlockId>> {
        self.get_block_ids(request_id)
            .into_iter()
            .zip(&self.kv_cache_config.kv_cache_groups)
            .map(|(mut ids, group)| {
                let spec = &group.kv_cache_spec;
                if spec.is_attention() && !spec.is_cross_or_encoder_only() {
                    ids.truncate(cdiv(num_computed_tokens, spec.block_size));
                }
                ids
            })
            .collect()
    }

    /// Estimate the number of tokens cached by the request.
    pub fn estimate_cached_tokens(&self, request: &impl KvRequest) -> usize {
        let mut cached_tokens: Option<usize> = None;
        for (group_id, (group, blocks)) in self
            .kv_cache_config
            .kv_cache_groups
            .iter()
            .zip(self.coordinator.get_blocks(request.request_id()))
            .enumerate()
        {
            let pool = self.coordinator.group_pool(group_id);
            if !group.kv_cache_spec.prefix_cacheable() {
                continue;
            }
            if group.kv_cache_spec.is_cross_or_encoder_only() {
                // Cross-attention and encoder-only groups are not prefix cached.
                continue;
            }
            // Coverage grows along the block table: the last hashed block has the most.
            let group_cached_tokens = blocks
                .iter()
                .rev()
                .find_map(|&b| pool.block(b).block_hash_num_tokens.filter(|&n| n != 0))
                .unwrap_or(0);
            cached_tokens =
                Some(cached_tokens.map_or(group_cached_tokens, |c| c.min(group_cached_tokens)));
        }
        cached_tokens.unwrap_or(0)
    }

    /// Cache the blocks for the request, if enabled.
    pub fn cache_blocks(&mut self, request: &impl KvRequest, num_computed_tokens: usize) {
        if self.enable_caching {
            self.coordinator.cache_blocks(request, num_computed_tokens);
        }
    }

    pub fn create_kv_cache_blocks(&self, blocks: Vec<Vec<BlockId>>) -> KvCacheBlocks {
        // Only create new KVCacheBlocks for non-empty blocks
        if blocks.iter().any(|b| !b.is_empty()) {
            KvCacheBlocks::new(blocks)
        } else {
            self.empty_kv_cache_blocks()
        }
    }

    /// Return a lookup-result view truncated at an aligned token endpoint.
    ///
    /// An external hit can supply the final Mamba state even when the local
    /// Mamba group ends before this endpoint. Other groups must cover it.
    /// Pure slicing: refcounts are untouched and `blocks` is not mutated.
    pub fn truncate_computed_blocks(
        &self,
        blocks: &KvCacheBlocks,
        num_computed_tokens: usize,
    ) -> KvCacheBlocks {
        assert_eq!(
            blocks.blocks.len(),
            self.coordinator.single_type_managers.len()
        );
        let truncated = blocks
            .blocks
            .iter()
            .zip(&self.coordinator.single_type_managers)
            .zip(&self.kv_cache_config.kv_cache_groups)
            .map(|((group_blocks, manager), group)| {
                if !group.kv_cache_spec.prefix_cacheable() {
                    // Scratch groups (e.g. the CSA compressor ring) hold a fixed
                    // block covering no token range, so a lookup result never
                    // carries computed blocks for them and their block size does
                    // not divide the endpoint.
                    assert!(group_blocks.is_empty());
                    return Vec::new();
                }
                assert!(num_computed_tokens.is_multiple_of(manager.block_size));
                let mut num_blocks = num_computed_tokens / manager.block_size;
                if group.kv_cache_spec.is_mamba() {
                    num_blocks = num_blocks.min(group_blocks.len());
                } else {
                    assert!(num_blocks <= group_blocks.len());
                }
                group_blocks[..num_blocks].to_vec()
            })
            .collect();
        self.create_kv_cache_blocks(truncated)
    }

    /// Drain and return new attention block IDs for zeroing.
    pub fn take_new_block_ids(&mut self) -> Vec<BlockId> {
        let mut ids = Vec::new();
        for mgr in &mut self.coordinator.single_type_managers {
            ids.extend(mgr.take_new_block_ids());
        }
        ids
    }

    /// Re-record the request's blocks from start_token onwards for
    /// zeroing, e.g. blocks a failed async KV load left unwritten.
    ///
    /// start_token must be block-aligned: zeroing a partially-valid block
    /// would wipe its valid prefix.
    pub fn record_blocks_for_zeroing(&mut self, request_id: &str, start_token: usize) {
        for mgr in &mut self.coordinator.single_type_managers {
            if mgr.records_new_block_ids() {
                assert!(start_token.is_multiple_of(mgr.block_size));
                let start_idx = start_token / mgr.block_size;
                let blocks = mgr.req_blocks(request_id).unwrap_or_default();
                let tail = blocks.get(start_idx..).unwrap_or_default().to_vec();
                mgr.new_block_ids.extend(tail);
            }
        }
    }

    /// Drain pending copies and return their retained endpoints.
    pub fn take_kv_cache_block_copies(&mut self) -> (Vec<KvCacheBlockCopy>, Vec<BlockId>) {
        let mut pending_copies = Vec::new();
        for mgr in &mut self.coordinator.single_type_managers {
            pending_copies.extend(mgr.take_pending_cow_copies());
        }
        let copies = pending_copies
            .iter()
            .map(|&(src_block_id, dst_block_id)| KvCacheBlockCopy {
                src_block_id,
                dst_block_id,
            })
            .collect();
        let retained_blocks = pending_copies.iter().flat_map(|&(src, dst)| [src, dst]).collect();
        (copies, retained_blocks)
    }

    /// Drain this step's boundary-state hand-offs for a KV connector.
    ///
    /// Returns `{request_id: [(group_id, block_id, boundary_tokens), ...]}`
    /// for mamba "align" boundary states: the
    /// request's committed boundary-state snapshots and, on a sub-block partial
    /// hit, the CoW copy of its last-prompt-boundary state. Only mamba "align"
    /// groups contribute; empty otherwise. A connector reads the referenced
    /// blocks — never resolving them positionally — and offloads them so a
    /// later request can hit that prefix.
    pub fn take_boundary_state_offloads(&mut self) -> IndexMap<String, Vec<BoundaryStateOffload>> {
        let mut offloads: IndexMap<String, Vec<BoundaryStateOffload>> = IndexMap::new();
        for mgr in &mut self.coordinator.single_type_managers {
            for (req_id, group_id, block, boundary_tokens) in
                mgr.take_pending_boundary_state_offloads()
            {
                offloads.entry(req_id).or_default().push((group_id, block, boundary_tokens));
            }
        }
        offloads
    }

    /// Consume safe producer partial tails when a request finishes.
    ///
    /// A mamba align table block is a valid boundary source only if no later
    /// token was forwarded. The connector pins and queues the exact table
    /// block before request cleanup, then releases the pin when every worker
    /// reports the store job complete.
    pub fn finalize_partial_tail_offloads(
        &mut self,
        request: &impl KvRequest,
    ) -> Vec<BoundaryStateOffload> {
        let mut offloads = Vec::new();
        for mgr in &mut self.coordinator.single_type_managers {
            if let Some(finalized) = mgr.finalize_partial_tail_offload(
                request.request_id(),
                request.num_computed_tokens(),
                request.num_in_flight_tokens(),
            ) {
                offloads.push(finalized);
            }
        }
        offloads
    }

    /// Notify the coordinator that a new step is starting.
    pub fn new_step_starts(&mut self) {
        self.coordinator.new_step_starts();
    }
}
