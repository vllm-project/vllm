// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `kv_cache_coordinator.py`: coordinate the KV cache of different KV cache
//! groups.
//!
//! The three Python coordinator classes share one struct tagged by
//! [`CoordinatorKind`].

use std::collections::BTreeSet;

use super::block::{BlockId, HOST_BLOCK_FLAG};
use super::block_hash_list::{eagle_proof_margin, partial_hash_hits_enabled};
use super::block_pool::{BlockPool, CachedBlockLookup};
use super::cdiv;
use super::error::KvCacheError;
use super::hisparse::{HiSparseCoordinator, HiSparseCtx};
use super::metrics::KvCacheMetricsCollector;
use super::request::KvRequest;
use super::single_type::{ManagerType, SingleTypeKvCacheManager};
use super::spec::{KvCacheConfig, KvCacheSpec};
use crate::hash::BlockHash;

fn validate_prefix_cache_retention_interval(
    retention_interval: Option<usize>,
    scheduler_block_size: usize,
    kv_cache_config: &KvCacheConfig,
) -> Result<(), KvCacheError> {
    let Some(retention_interval) = retention_interval else {
        return Ok(());
    };

    // Retention sparsifies sliding-window and Mamba (linear-attention)
    // checkpoints; full-attention and chunked-local groups cache densely and
    // ignore it (their hit granularity must stay fine).
    if !kv_cache_config
        .kv_cache_groups
        .iter()
        .any(|g| g.kv_cache_spec.is_sliding_window() || g.kv_cache_spec.is_mamba())
    {
        if retention_interval == 0 {
            return Ok(());
        }
        return Err(KvCacheError::InvalidConfig {
            message: "prefix_cache_retention_interval is set but this model has no \
                      sliding-window or Mamba KV cache group, so retention has no effect. \
                      Set it to 0 (it only applies to sliding-window and Mamba attention)."
                .to_owned(),
        });
    }

    // Python also rejects negative values, which `usize` cannot hold.
    if retention_interval % scheduler_block_size != 0 {
        return Err(KvCacheError::InvalidConfig {
            message: format!(
                "prefix_cache_retention_interval ({retention_interval}) must be non-negative \
                 and a multiple of scheduler_block_size ({scheduler_block_size})."
            ),
        });
    }
    Ok(())
}

/// Positions a later request replaying a prompt can resume at.
///
/// A hit is the shortest across all groups, so every group retains state
/// at each position; EAGLE groups also keep the block above, which they
/// match and drop back from (see `reachable_block_mask`).
///
/// Two positions are reachable: a resend of the identical prompt is capped
/// at `num_tokens - 1` (its last token is recomputed for logits), a
/// longer sibling matches the final aligned block. They differ only on a
/// block-aligned prompt, where retaining just the higher one collapses the
/// resend's hit to 0. The alignment is the scheduler block size, not the
/// finer hash granularity, which would over-estimate the reach.
pub fn replay_boundaries(
    num_prompt_tokens: usize,
    scheduler_block_size: usize,
    use_eagle: bool,
) -> Vec<usize> {
    if !use_eagle {
        // Python yields -1 for an empty prompt; prompts are never empty.
        return vec![num_prompt_tokens.saturating_sub(1)];
    }
    let block = scheduler_block_size;
    let resend = num_prompt_tokens.saturating_sub(1) / block * block;
    let extension = num_prompt_tokens / block * block;
    let set: BTreeSet<usize> = [
        resend.saturating_sub(block),
        extension.saturating_sub(block),
    ]
    .into_iter()
    .collect();
    set.into_iter().collect()
}

/// Which Python coordinator class this is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinatorKind {
    /// `KVCacheCoordinatorNoPrefixCache`: to use if prefix caching is disabled
    /// or unsupported. Supports arbitrary numbers of KV cache groups
    /// (including 0 groups).
    NoPrefixCache,
    /// `UnitaryKVCacheCoordinator`: for models with only one KV cache group.
    Unitary,
    /// `HybridKVCacheCoordinator`: for hybrid models with multiple KV cache
    /// types, and thus multiple kv cache groups.
    Hybrid,
}

/// KV cache groups that share one spec, batched together for a single
/// cache-hit lookup.
///
/// `use_eagle` is True iff any member group is an EAGLE/MTP group. Members
/// sharing a spec are cached and looked up jointly, so the EAGLE last-block drop
/// is necessarily decided for the whole spec group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecGroup {
    pub spec: KvCacheSpec,
    pub group_ids: Vec<usize>,
    pub manager_type: ManagerType,
    pub use_eagle: bool,
}

/// Coordinate the KV cache of different KV cache groups.
#[derive(Debug)]
pub struct KvCacheCoordinator {
    pub kind: CoordinatorKind,
    pub kv_cache_config: KvCacheConfig,
    /// The scheduling granularity (LCM of all group block sizes), must be a multiple
    /// of the hash_block_size and the block size of each group.
    pub scheduler_block_size: usize,
    pub num_reprefillable_tokens: usize,
    pub block_pool: BlockPool,
    /// KV cache group indices that get the EAGLE last-block drop.
    pub eagle_group_ids: BTreeSet<usize>,
    /// Lookahead hashes clear ``use_eagle``: groups stay flagged, none drops.
    pub drop_eagle_group_ids: BTreeSet<usize>,
    pub single_type_managers: Vec<SingleTypeKvCacheManager>,
    pub group_block_sizes: Vec<usize>,
    /// A positive retention interval must be a multiple of the base hit granularity
    /// (``scheduler_block_size``) to land on real cache-hit boundaries.
    /// 0 = keep only the latest replay boundary; None = dense;
    pub retention_interval: Option<usize>,
    /// Fine-grained hash hits (hybrid only).
    pub enable_partial_hash_hits: bool,
    pub hash_block_size: usize,
    /// Hybrid only: groups batched by spec, full attention first.
    pub attention_groups: Vec<SpecGroup>,
    /// Dense reference group for per-group lookups (None when the model
    /// has no full-attention layers).
    pub full_attention_group_id: Option<usize>,
    /// The HiSparse coordinator, with a HiSparse host source group.
    pub hisparse: Option<HiSparseCoordinator>,
    /// The private pool of the HiSparse host source group
    /// (`HiSparseSourceManager.bind_host_pool`); the other groups share
    /// `block_pool`.
    pub host_pool: Option<BlockPool>,
}

/// Run `$body` with `$pool` bound to the lookup of group `$group_id`: the
/// HiSparse host pool for the host group when `$host` is given, else
/// `$lookup`.
macro_rules! with_group_lookup {
    ($self:ident, $group_id:expr, $lookup:expr, $host:expr, |$pool:ident| $body:expr) => {
        match $host.filter(|_| $self.host_group_id() == Some($group_id)) {
            Some($pool) => $body,
            None => {
                let $pool = $lookup;
                $body
            }
        }
    };
}

impl KvCacheCoordinator {
    /// `get_kv_cache_coordinator`.
    pub fn new(
        kv_cache_config: KvCacheConfig,
        max_model_len: usize,
        max_in_flight_tokens: usize,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: usize,
        pcp_world_size: usize,
        scheduler_block_size: usize,
        hash_block_size: usize,
        metrics_collector: Option<KvCacheMetricsCollector>,
        num_hash_lookahead_tokens: usize,
        num_prefill_lookahead: usize,
        allow_partial_hash_hits: bool,
    ) -> Result<Self, KvCacheError> {
        let kind = if !enable_caching {
            CoordinatorKind::NoPrefixCache
        } else if kv_cache_config.kv_cache_groups.len() == 1 {
            CoordinatorKind::Unitary
        } else {
            CoordinatorKind::Hybrid
        };
        let mut coordinator = Self::new_base(
            kind,
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
            pcp_world_size,
            scheduler_block_size,
            hash_block_size,
            metrics_collector,
            num_hash_lookahead_tokens,
            num_prefill_lookahead,
        )?;
        match kind {
            CoordinatorKind::NoPrefixCache => {}
            CoordinatorKind::Unitary => coordinator.init_unitary(enable_caching, hash_block_size),
            CoordinatorKind::Hybrid => coordinator.init_hybrid(
                hash_block_size,
                dcp_world_size,
                pcp_world_size,
                allow_partial_hash_hits,
            ),
        }
        Ok(coordinator)
    }

    fn new_base(
        kind: CoordinatorKind,
        kv_cache_config: KvCacheConfig,
        max_model_len: usize,
        max_in_flight_tokens: usize,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: usize,
        pcp_world_size: usize,
        scheduler_block_size: usize,
        hash_block_size: usize,
        metrics_collector: Option<KvCacheMetricsCollector>,
        num_hash_lookahead_tokens: usize,
        num_prefill_lookahead: usize,
    ) -> Result<Self, KvCacheError> {
        assert!(
            scheduler_block_size.is_multiple_of(hash_block_size)
                && kv_cache_config
                    .kv_cache_groups
                    .iter()
                    .all(|g| scheduler_block_size.is_multiple_of(g.kv_cache_spec.block_size))
        );
        let num_reprefillable_tokens = num_prefill_lookahead.saturating_sub(1);

        let block_pool = BlockPool::new(
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
            num_hash_lookahead_tokens,
            super::events::MEDIUM_GPU,
        );

        let num_groups = kv_cache_config.kv_cache_groups.len();
        let mut eagle_group_ids: BTreeSet<usize> = kv_cache_config
            .kv_cache_groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.is_eagle_group)
            .map(|(i, _)| i)
            .collect();
        // Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle && eagle_group_ids.is_empty() {
            eagle_group_ids = (0..num_groups).collect();
        }
        let drop_eagle_group_ids = if use_eagle {
            eagle_group_ids.clone()
        } else {
            BTreeSet::new()
        };

        // During chunked prefill with EAGLE, the single next prefill lookahead
        // token past the chunk boundary is combined with the final hidden state
        // and written to the KV cache. Therefore, the final chunk token must be
        // excluded from prefix cache hits to prevent requests from acquiring the
        // KV cache slot polluted with the next prefill token, which may or may not
        // be present after the matching prefix. The last-block drop handles this
        // edge case. During multi-module MTP, the issue generalizes to a prefill
        // lookahead of num_speculative_tokens, so the dropped tail must be large
        // enough to contain them. Hits land on scheduler-block boundaries (see
        // `_cache_hit_alignment_tokens`), so the excluded tail is
        // scheduler_block_size, not the group's own block size.
        if enable_caching
            && !eagle_group_ids.is_empty()
            && scheduler_block_size < num_prefill_lookahead
        {
            return Err(KvCacheError::InvalidConfig {
                message: format!(
                    "Multi-module MTP with prefix caching requires scheduler_block_size \
                     (={scheduler_block_size}) >= num_speculative_tokens (={num_prefill_lookahead})."
                ),
            });
        }

        let mut single_type_managers = Vec::with_capacity(num_groups);
        for (i, group) in kv_cache_config.kv_cache_groups.iter().enumerate() {
            single_type_managers.push(SingleTypeKvCacheManager::new(
                group.kv_cache_spec.clone(),
                group.role,
                max_in_flight_tokens,
                max_model_len,
                enable_caching,
                i,
                scheduler_block_size,
                dcp_world_size,
                pcp_world_size,
                kv_cache_config.needs_kv_cache_zeroing,
            )?);
        }
        // Match Mamba checkpoints to Eagle's attention replay boundary.
        if use_eagle {
            for manager in &mut single_type_managers {
                if manager.manager_type == ManagerType::Mamba {
                    manager.mamba.drop_eagle_checkpoint_block = true;
                }
            }
        }
        let group_block_sizes = single_type_managers.iter().map(|m| m.block_size).collect();
        let (hisparse, host_pool) = match HiSparseCoordinator::new(
            &kv_cache_config,
            &single_type_managers,
            &block_pool,
            max_model_len,
            num_reprefillable_tokens,
        )? {
            Some((hisparse, host_pool)) => (Some(hisparse), Some(host_pool)),
            None => (None, None),
        };

        let retention_interval = kv_cache_config.prefix_cache_retention_interval;
        validate_prefix_cache_retention_interval(
            retention_interval,
            scheduler_block_size,
            &kv_cache_config,
        )?;

        Ok(Self {
            kind,
            kv_cache_config,
            scheduler_block_size,
            num_reprefillable_tokens,
            block_pool,
            eagle_group_ids,
            drop_eagle_group_ids,
            single_type_managers,
            group_block_sizes,
            retention_interval,
            enable_partial_hash_hits: false,
            hash_block_size,
            attention_groups: Vec::new(),
            full_attention_group_id: None,
            hisparse,
            host_pool,
        })
    }

    /// The HiSparse host source group, if any.
    pub fn host_group_id(&self) -> Option<usize> {
        self.hisparse.as_ref().map(|h| h.host_group_id)
    }

    /// The block pool of a group: the HiSparse host pool for the host
    /// source group, else the shared device pool (`manager.block_pool`).
    pub fn group_pool(&self, group_id: usize) -> &BlockPool {
        match &self.host_pool {
            Some(host) if self.host_group_id() == Some(group_id) => host,
            _ => &self.block_pool,
        }
    }

    /// The HiSparse coordinator and the state it works on.
    pub(crate) fn hisparse_parts(&mut self) -> Option<(&mut HiSparseCoordinator, HiSparseCtx<'_>)> {
        let hisparse = self.hisparse.as_mut()?;
        Some((
            hisparse,
            HiSparseCtx {
                managers: &mut self.single_type_managers,
                device: &mut self.block_pool,
                host: self.host_pool.as_mut().expect("HiSparse has a host pool"),
            },
        ))
    }

    /// Handle the unpinned blocks the device pool reused since the last call
    /// (`HiSparseCoordinator._on_block_reused`, which Python runs from
    /// `get_new_blocks`).
    pub fn sync_hisparse(&mut self) {
        if let Some((hisparse, mut cx)) = self.hisparse_parts() {
            hisparse.process_reused_blocks(&mut cx);
        }
    }

    /// The managers, the device pool and the pool of group `group_id`, for a
    /// manager call that touches blocks.
    fn manager_and_pool(
        &mut self,
        group_id: usize,
    ) -> (&mut SingleTypeKvCacheManager, &mut BlockPool) {
        let manager = &mut self.single_type_managers[group_id];
        match &mut self.host_pool {
            Some(host) if manager.manager_type == ManagerType::HiSparseSource => (manager, host),
            _ => (manager, &mut self.block_pool),
        }
    }

    /// After an operation on the HiSparse host pool: its events go to the
    /// device pool's queue (`SharedEventQueueBlockPool`).
    pub(crate) fn forward_host_events(&mut self) {
        if let Some(host) = &mut self.host_pool {
            host.forward_events_to(&mut self.block_pool);
        }
    }

    fn init_unitary(&mut self, enable_caching: bool, hash_block_size: usize) {
        let block_size = self.single_type_managers[0].block_size;
        // For models using only Mamba, block_size is set to max_model_len when
        // prefix caching is disabled, and hash_block_size validation is skipped.
        assert!(
            !enable_caching || hash_block_size == block_size,
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        );
        assert!(
            self.kv_cache_config.kv_cache_groups.len() == 1,
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        );
        // Single group; useless but just set ``use_eagle`` for consistency regardless.
        self.single_type_managers[0].use_eagle = self.drop_eagle_group_ids.contains(&0);
    }

    fn init_hybrid(
        &mut self,
        hash_block_size: usize,
        dcp_world_size: usize,
        pcp_world_size: usize,
        allow_partial_hash_hits: bool,
    ) {
        // Only groups that participate in prefix caching must satisfy the
        // divisibility constraint; groups that opt out (e.g. circular buffers
        // such as the GLM-5.3-Flash indexer tail) are scratch and excluded.
        let cacheable_block_sizes: Vec<usize> = self
            .single_type_managers
            .iter()
            .zip(&self.kv_cache_config.kv_cache_groups)
            .filter(|(_, g)| g.kv_cache_spec.prefix_cacheable())
            .map(|(m, _)| m.block_size)
            .collect();
        assert!(
            cacheable_block_sizes.iter().all(|bs| bs.is_multiple_of(hash_block_size)),
            "Each KV cache group's real block_size must be divisible by hash_block_size. \
             block_sizes={cacheable_block_sizes:?}, hash_block_size={hash_block_size}"
        );
        for g in &self.kv_cache_config.kv_cache_groups {
            let spec = &g.kv_cache_spec;
            let replicated = !spec.dcp_sharded;
            if pcp_world_size > 1 {
                assert!(
                    spec.is_full_attention() || (spec.is_attention() && replicated),
                    "PCP only supports full attention and replicated draft groups."
                );
            }
            if dcp_world_size > 1 {
                assert!(
                    spec.is_full_attention() || replicated,
                    "DCP with hybrid KV cache layouts only supports full-attention, Mamba, \
                     and replicated draft groups, got: {:?}.",
                    spec.spec_type
                );
            }
        }
        // Fine-grained hash hits require Mamba "align" and compatible cache
        // managers in every group. TP needs hashing finer than the Mamba block;
        // DCP accepts equality because it scales the effective full-attention
        // block instead.
        let manager_types: Vec<ManagerType> =
            self.single_type_managers.iter().map(|m| m.manager_type).collect();
        self.enable_partial_hash_hits = allow_partial_hash_hits
            && partial_hash_hits_enabled(
                &self.kv_cache_config.kv_cache_groups,
                hash_block_size,
                dcp_world_size,
                Some(&manager_types),
            );
        let cache_hit_alignment_tokens = self.cache_hit_alignment_tokens();
        for manager in &mut self.single_type_managers {
            manager.cache_hit_alignment_tokens = cache_hit_alignment_tokens;
        }
        self.verify_and_split_kv_cache_groups();
    }

    /// Fine-grained partial hits may return hash-block-aligned lengths;
    /// otherwise it must stay scheduler-block-aligned.
    pub fn cache_hit_alignment_tokens(&self) -> usize {
        if self.enable_partial_hash_hits {
            self.hash_block_size
        } else {
            self.scheduler_block_size
        }
    }

    /// Groups KV cache groups by their spec type for efficient batch processing
    /// during cache hit lookup.
    fn verify_and_split_kv_cache_groups(&mut self) {
        let mut attention_groups: Vec<SpecGroup> = Vec::new();
        for (i, g) in self.kv_cache_config.kv_cache_groups.iter().enumerate() {
            // Skip groups that opt out of prefix caching (e.g. circular
            // buffers): their blocks are per-request scratch, never
            // shareable, so they must not participate in hit lookup (their
            // manager-level hooks already no-op). Their slot in the per-group
            // hit tuple stays empty.
            if !g.kv_cache_spec.prefix_cacheable() {
                continue;
            }
            let manager_type = self.single_type_managers[i].manager_type;
            let spec = &g.kv_cache_spec;
            let use_eagle = self.drop_eagle_group_ids.contains(&i);

            // Try to find an existing group with the same spec. All managers
            // share one block pool here.
            if let Some(group) = attention_groups
                .iter_mut()
                .find(|group| group.spec == *spec && group.manager_type == manager_type)
            {
                group.group_ids.push(i);
                group.use_eagle |= use_eagle;
            } else {
                attention_groups.push(SpecGroup {
                    spec: spec.clone(),
                    group_ids: vec![i],
                    manager_type,
                    use_eagle,
                });
            }
        }

        assert!(
            !attention_groups.is_empty(),
            "HybridKVCacheCoordinator requires at least one cacheable group."
        );

        // Put full attention first: its efficient left-to-right scan provides
        // a tighter initial bound, reducing work for subsequent groups.
        attention_groups.sort_by_key(|g| !g.spec.is_full_attention());

        // Dense reference group for per-group lookups (None when the model
        // has no full-attention layers): full attention is downward-closed,
        // so any group reporting a longer per-group hit implies the union of
        // per-group hits is not consistent at a single boundary (#46453).
        let first = &attention_groups[0];
        self.full_attention_group_id = first.spec.is_full_attention().then(|| first.group_ids[0]);

        // Propagate the eagle bit to each manager (default to ``use_eagle=False``).
        for group in &attention_groups {
            if group.use_eagle {
                for &gid in &group.group_ids {
                    self.single_type_managers[gid].use_eagle = true;
                }
            }
        }
        self.attention_groups = attention_groups;
    }

    /// Get the number of device blocks needed to be allocated for the request.
    ///
    /// Args:
    ///     request_id: The request ID.
    ///     num_tokens: The total number of tokens that need a slot (including
    ///         tokens that are already allocated).
    ///     new_computed_blocks: The new computed blocks just hitting the
    ///         prefix caching.
    ///     num_encoder_tokens: The number of encoder tokens for allocating
    ///         blocks for cross-attention.
    ///     total_computed_tokens: Include both local and external tokens.
    ///     num_local_computed_tokens: The number of local prefix-cache computed
    ///         tokens.
    ///     num_tokens_main_model: The number of tokens for the main model (aka target
    ///         model in spec decode). w/o spec decode, it is num_tokens;
    ///         with spec decode, it is num_tokens - num_lookahead_tokens.
    ///     apply_admission_cap: If True, apply the recycling-aware
    ///         per-request admission cap (SWA / chunked-local). Set only by
    ///         the full-sequence admission gate; per-step allocation must
    ///         leave it False so the predictor matches `allocate_new_blocks`.
    ///     prefill_end: The token index the request's prefill ends at, the
    ///         same value the scheduler splits chunks against. Under sparse
    ///         retention Mamba reserves a prefill checkpoint only on the chunk
    ///         reaching it; under dense retention every chunk publishes a
    ///         state, so it is ignored (0).
    ///
    /// Returns:
    ///     The number of blocks to allocate (signed, see
    ///     [`SingleTypeKvCacheManager::get_num_blocks_to_allocate`]).
    pub fn get_num_blocks_to_allocate(
        &mut self,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &[Vec<BlockId>],
        num_encoder_tokens: usize,
        total_computed_tokens: usize,
        num_local_computed_tokens: usize,
        num_tokens_main_model: usize,
        apply_admission_cap: bool,
        prefill_end: usize,
    ) -> i64 {
        let mut num_blocks_to_allocate = 0;
        let prefill_end = if self.retention_interval != Some(0) {
            0
        } else {
            prefill_end
        };
        self.sync_hisparse();
        for i in 0..self.single_type_managers.len() {
            num_blocks_to_allocate +=
                if self.single_type_managers[i].manager_type == ManagerType::CrossAttention {
                    // For cross-attention, we issue a single static allocation
                    // of blocks based on the number of encoder input tokens.
                    self.group_get_num_blocks_to_allocate(
                        i,
                        request_id,
                        num_encoder_tokens,
                        &[],
                        0,
                        0,
                        num_encoder_tokens,
                        apply_admission_cap,
                        0,
                    )
                } else {
                    self.group_get_num_blocks_to_allocate(
                        i,
                        request_id,
                        num_tokens,
                        new_computed_blocks.get(i).map_or(&[][..], Vec::as_slice),
                        total_computed_tokens,
                        num_local_computed_tokens,
                        num_tokens_main_model,
                        apply_admission_cap,
                        prefill_end,
                    )
                };
        }
        num_blocks_to_allocate
    }

    /// `manager.get_num_blocks_to_allocate` of group `group_id`.
    pub fn group_get_num_blocks_to_allocate(
        &mut self,
        group_id: usize,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &[BlockId],
        total_computed_tokens: usize,
        num_local_computed_tokens: usize,
        num_tokens_main_model: usize,
        apply_admission_cap: bool,
        prefill_end: usize,
    ) -> i64 {
        let manager = &self.single_type_managers[group_id];
        match manager.manager_type {
            ManagerType::HiSparseSource => {
                let (hisparse, mut cx) = self.hisparse_parts().expect("HiSparse group");
                hisparse.source_get_num_blocks_to_allocate(
                    &mut cx,
                    request_id,
                    num_tokens,
                    new_computed_blocks,
                    total_computed_tokens,
                    num_local_computed_tokens,
                    apply_admission_cap,
                )
            }
            ManagerType::HiSparseHot => manager.hot_get_num_blocks_to_allocate(
                request_id,
                new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
            ),
            ManagerType::HiSparseResident => manager.resident_get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
                apply_admission_cap,
            ),
            _ => self.single_type_managers[group_id].get_num_blocks_to_allocate(
                &self.block_pool,
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap,
                prefill_end,
            ),
        }
    }

    /// Add the new computed blocks to the request. Optionally allocate new
    /// blocks for external computed tokens (if any).
    ///
    /// Args:
    ///     request_id: The request ID.
    ///     new_computed_blocks: The new computed blocks just hitting the
    ///         prefix cache.
    ///     num_local_computed_tokens: The number of local computed tokens.
    ///     num_external_computed_tokens: The number of external computed tokens.
    ///     skip_zeroing_group_ids: Groups whose external-token blocks are
    ///         written by an async load and must not be zeroed.
    pub fn allocate_new_computed_blocks(
        &mut self,
        request_id: &str,
        new_computed_blocks: &[Vec<BlockId>],
        num_local_computed_tokens: usize,
        num_external_computed_tokens: usize,
        skip_zeroing_group_ids: &[usize],
    ) {
        // A running request is already tracked in num_cached_block and won't
        // have new prefix-cache hits, so this is a no-op for it.
        if self.single_type_managers.iter().any(|m| m.has_cached_block_entry(request_id)) {
            assert!(new_computed_blocks.iter().all(Vec::is_empty));
            return;
        }

        // Two-phase allocation (issue #33775): first touch every group's local
        // cache-hit blocks, then allocate external blocks for every group. This
        // ensures an earlier group's external `get_new_blocks` cannot evict a
        // later group's not-yet-touched cache-hit blocks.
        self.sync_hisparse();
        let has_resident_managers =
            self.hisparse.as_ref().is_some_and(|h| !h.resident_group_ids.is_empty());
        for i in 0..self.single_type_managers.len() {
            let new_computed_blocks = new_computed_blocks.get(i).map_or(&[][..], Vec::as_slice);
            match self.single_type_managers[i].manager_type {
                ManagerType::HiSparseHot => self.single_type_managers[i]
                    .hot_add_local_computed_blocks(
                        request_id,
                        new_computed_blocks,
                        num_local_computed_tokens,
                        has_resident_managers,
                    ),
                ManagerType::HiSparseResident => {
                    let (hisparse, mut cx) = self.hisparse_parts().expect("HiSparse group");
                    hisparse.resident_add_local_computed_blocks(
                        &mut cx,
                        i,
                        request_id,
                        new_computed_blocks,
                        num_local_computed_tokens,
                    );
                }
                _ => {
                    let (manager, pool) = self.manager_and_pool(i);
                    manager.add_local_computed_blocks(
                        pool,
                        request_id,
                        new_computed_blocks,
                        num_local_computed_tokens,
                        num_external_computed_tokens,
                    );
                }
            }
        }
        if num_external_computed_tokens > 0 {
            for i in 0..self.single_type_managers.len() {
                let record_for_zeroing = !skip_zeroing_group_ids.contains(&i);
                match self.single_type_managers[i].manager_type {
                    ManagerType::HiSparseSource => {
                        let (hisparse, mut cx) = self.hisparse_parts().expect("HiSparse group");
                        hisparse.source_allocate_external_computed_blocks(
                            &mut cx,
                            request_id,
                            num_local_computed_tokens,
                            num_external_computed_tokens,
                            record_for_zeroing,
                        );
                    }
                    ManagerType::HiSparseHot => {
                        self.single_type_managers[i].require_hot(request_id);
                    }
                    ManagerType::HiSparseResident => {
                        self.single_type_managers[i].resident_allocate_external_computed_blocks(
                            &mut self.block_pool,
                            request_id,
                            num_local_computed_tokens,
                            num_external_computed_tokens,
                        );
                    }
                    _ => self.single_type_managers[i].allocate_external_computed_blocks(
                        &mut self.block_pool,
                        request_id,
                        num_local_computed_tokens,
                        num_external_computed_tokens,
                        record_for_zeroing,
                    ),
                }
                self.sync_hisparse();
            }
        }
    }

    /// Allocate new blocks for the request to give it at least `num_tokens`
    /// token slots.
    pub fn allocate_new_blocks(
        &mut self,
        request_id: &str,
        num_tokens: usize,
        num_tokens_main_model: usize,
        num_encoder_tokens: usize,
    ) -> Vec<Vec<BlockId>> {
        self.sync_hisparse();
        let mut new_blocks = Vec::with_capacity(self.single_type_managers.len());
        for i in 0..self.single_type_managers.len() {
            let num_tokens =
                if self.single_type_managers[i].manager_type == ManagerType::CrossAttention {
                    num_encoder_tokens
                } else {
                    num_tokens
                };
            new_blocks.push(self.group_allocate_new_blocks(
                i,
                request_id,
                num_tokens,
                num_tokens_main_model,
            ));
        }
        new_blocks
    }

    /// `manager.allocate_new_blocks` of group `group_id`.
    pub fn group_allocate_new_blocks(
        &mut self,
        group_id: usize,
        request_id: &str,
        num_tokens: usize,
        num_tokens_main_model: usize,
    ) -> Vec<BlockId> {
        let manager = &mut self.single_type_managers[group_id];
        let blocks = match manager.manager_type {
            ManagerType::HiSparseSource => {
                let (hisparse, mut cx) = self.hisparse_parts().expect("HiSparse group");
                hisparse.source_allocate_new_blocks(&mut cx, request_id, num_tokens)
            }
            ManagerType::HiSparseHot => {
                manager.hot_allocate_new_blocks(&mut self.block_pool, request_id)
            }
            ManagerType::HiSparseResident => {
                manager.resident_allocate_new_blocks(&mut self.block_pool, request_id, num_tokens)
            }
            _ => manager.allocate_new_blocks(
                &mut self.block_pool,
                request_id,
                num_tokens,
                num_tokens_main_model,
            ),
        };
        self.sync_hisparse();
        blocks
    }

    /// Largest prefix of `num_tokens` a future cache hit could match.
    ///
    /// Hits are `scheduler_block_size`-aligned (see
    /// `find_longest_cache_hit`) unless fine-grained partial hash hits are
    /// enabled, in which case no rounding applies -- rounding even to
    /// `hash_block_size` would re-register a privatized Mamba tail.
    fn align_cacheable(&self, num_tokens: usize) -> usize {
        if self.enable_partial_hash_hits {
            return num_tokens;
        }
        num_tokens / self.scheduler_block_size * self.scheduler_block_size
    }

    /// Return the prefix eligible for hashing in this group.
    pub fn get_num_cacheable_tokens(
        &self,
        num_computed_tokens: usize,
        kv_cache_group_id: usize,
    ) -> usize {
        if self.kind != CoordinatorKind::Hybrid {
            // Only cache tokens with finalized KV. The last num_reprefillable_tokens
            // tokens can be re-prefilled during multi-module MTP.
            return num_computed_tokens.saturating_sub(self.num_reprefillable_tokens);
        }
        let cached_num_computed_tokens = self.align_cacheable(num_computed_tokens);
        let manager = &self.single_type_managers[kv_cache_group_id];
        // EAGLE matches one block past the aligned boundary, then drops it.
        if manager.use_eagle && cached_num_computed_tokens > 0 {
            let num_finalized_computed_tokens =
                num_computed_tokens.saturating_sub(self.num_reprefillable_tokens);
            return num_finalized_computed_tokens
                .min(self.align_cacheable(num_finalized_computed_tokens) + manager.block_size);
        }
        cached_num_computed_tokens
    }

    /// Cache the blocks for the request.
    ///
    /// Args:
    ///     request: The request.
    ///     num_computed_tokens: The total number of tokens
    ///         that need to be cached
    ///         (including tokens that are already cached).
    pub fn cache_blocks(&mut self, request: &impl KvRequest, num_computed_tokens: usize) {
        let boundaries = replay_boundaries(
            request.num_prompt_tokens(),
            self.scheduler_block_size,
            !self.drop_eagle_group_ids.is_empty(),
        );
        for group_id in 0..self.single_type_managers.len() {
            if !self.single_type_managers[group_id].enable_caching {
                continue;
            }
            let num_tokens = self.get_num_cacheable_tokens(num_computed_tokens, group_id);
            let retention_interval = self.retention_interval;
            if self.single_type_managers[group_id].manager_type == ManagerType::HiSparseSource {
                let (hisparse, mut cx) = self.hisparse_parts().expect("HiSparse group");
                hisparse.publish_when_ready(
                    &mut cx,
                    request,
                    num_tokens,
                    retention_interval,
                    &boundaries,
                );
                continue;
            }
            self.single_type_managers[group_id].cache_blocks(
                &mut self.block_pool,
                request,
                num_tokens,
                retention_interval,
                &boundaries,
            );
        }
    }

    pub fn emit_cached_block_events(
        &mut self,
        request: &impl KvRequest,
        computed_blocks: &[Vec<BlockId>],
    ) {
        for (group_idx, group_blocks) in computed_blocks.iter().enumerate() {
            if !group_blocks.is_empty() {
                let block_size =
                    self.kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec.block_size;
                let (_, pool) = self.manager_and_pool(group_idx);
                pool.emit_cached_block_events(request, group_blocks.len(), block_size, group_idx);
                self.forward_host_events();
            }
        }
    }

    /// Reset each manager's pool once.
    pub fn reset_prefix_cache(&mut self) -> bool {
        // Python iterates `dict.fromkeys(manager.block_pool ...)`, which is
        // empty without managers.
        if self.single_type_managers.is_empty() {
            return true;
        }
        // Pools in the order their managers first use them; every pool is
        // reset even if an earlier one fails.
        let host_group_id = self.host_group_id();
        let ok = match (&mut self.host_pool, host_group_id) {
            (Some(host), Some(0)) => {
                let host_ok = host.reset_prefix_cache();
                self.forward_host_events();
                let device_ok =
                    self.single_type_managers.len() == 1 || self.block_pool.reset_prefix_cache();
                host_ok & device_ok
            }
            (Some(host), _) => {
                let device_ok = self.block_pool.reset_prefix_cache();
                let host_ok = host.reset_prefix_cache();
                self.forward_host_events();
                device_ok & host_ok
            }
            (None, _) => self.block_pool.reset_prefix_cache(),
        };
        self.sync_hisparse();
        ok
    }

    /// Free the blocks for the request.
    ///
    /// `finished`: whether the request finished (`request.is_finished()`),
    /// which only HiSparse reads: the host prefix of a finished request is
    /// still published once its in-flight host writes complete.
    pub fn free(&mut self, request_id: &str, finished: bool) {
        self.sync_hisparse();
        for i in 0..self.single_type_managers.len() {
            let blocks = self.pop_group_blocks_for_free(i, request_id, finished);
            // Free blocks in reverse order so that the tail blocks are freed first.
            let (_, pool) = self.manager_and_pool(i);
            pool.free_blocks(blocks.into_iter().rev(), false);
        }
    }

    /// `manager.pop_blocks_for_free` of group `group_id`.
    fn pop_group_blocks_for_free(
        &mut self,
        group_id: usize,
        request_id: &str,
        finished: bool,
    ) -> Vec<BlockId> {
        match self.single_type_managers[group_id].manager_type {
            ManagerType::HiSparseSource | ManagerType::HiSparseResident => {
                let (hisparse, mut cx) = self.hisparse_parts().expect("HiSparse group");
                hisparse.pop_blocks_for_free(&mut cx, group_id, request_id, finished)
            }
            ManagerType::HiSparseHot => {
                self.single_type_managers[group_id].hot_pop_blocks_for_free(request_id)
            }
            _ => self.single_type_managers[group_id].pop_blocks_for_free(request_id),
        }
    }

    /// Pop the request's bookkeeping from all single-type managers and
    /// return its blocks without returning them to the block pool. The
    /// caller must eventually pass the returned blocks to
    /// `KvCacheManager::free_blocks`, freeing them in reverse order (so that
    /// tail blocks are evicted first). HiSparse host blocks carry
    /// [`HOST_BLOCK_FLAG`].
    ///
    /// `finished`: see [`Self::free`].
    pub fn pop_blocks_for_free(&mut self, request_id: &str, finished: bool) -> Vec<BlockId> {
        self.sync_hisparse();
        let mut blocks = Vec::new();
        for i in 0..self.single_type_managers.len() {
            let group_blocks = self.pop_group_blocks_for_free(i, request_id, finished);
            if self.host_group_id() == Some(i) {
                blocks.extend(group_blocks.into_iter().map(|b| b | HOST_BLOCK_FLAG));
            } else {
                blocks.extend(group_blocks);
            }
        }
        blocks
    }

    /// Free blocks that may mix the device and HiSparse host pools
    /// ([`HOST_BLOCK_FLAG`]), as `BlockPool.free_blocks` on `own` (the host
    /// pool if `host_first`, else the device pool) does in Python: the own
    /// pool's blocks first, then the other pool's, each in the given order.
    pub fn free_mixed_blocks(
        &mut self,
        ordered_blocks: impl IntoIterator<Item = BlockId>,
        evict_first: bool,
        host_first: bool,
    ) {
        let Some(host) = &mut self.host_pool else {
            self.block_pool.free_blocks(ordered_blocks, evict_first);
            return;
        };
        let (host_blocks, device_blocks): (Vec<BlockId>, Vec<BlockId>) =
            ordered_blocks.into_iter().partition(|&b| b & HOST_BLOCK_FLAG != 0);
        let host_blocks = host_blocks.into_iter().map(|b| b & !HOST_BLOCK_FLAG);
        if host_first {
            host.free_blocks(host_blocks, evict_first);
            self.block_pool.free_blocks(device_blocks, evict_first);
        } else {
            self.block_pool.free_blocks(device_blocks, evict_first);
            host.free_blocks(host_blocks, evict_first);
        }
    }

    /// Remove the blocks that are no longer needed from `blocks` and replace
    /// the removed blocks with null_block.
    ///
    /// Args:
    ///     request_id: The request ID.
    ///     processed_computed_tokens: Computed-token prefix length covering
    ///         fully processed and committed tokens only (safe to free).
    ///     num_prompt_tokens: Optional prompt length. R-SWA managers use this to
    ///         free gap blocks between the prefill tail and decode window; other
    ///         manager types ignore it.
    pub fn remove_skipped_blocks(
        &mut self,
        request_id: &str,
        processed_computed_tokens: usize,
        num_prompt_tokens: Option<usize>,
    ) {
        for i in 0..self.single_type_managers.len() {
            let (manager, pool) = self.manager_and_pool(i);
            manager.remove_skipped_blocks(
                pool,
                request_id,
                processed_computed_tokens,
                num_prompt_tokens,
            );
        }
    }

    /// Get the blocks for the request.
    ///
    /// Deviation: Python returns the managers' live lists; this returns copies.
    pub fn get_blocks(&self, request_id: &str) -> Vec<Vec<BlockId>> {
        self.single_type_managers
            .iter()
            .map(|m| m.req_blocks(request_id).map(<[BlockId]>::to_vec).unwrap_or_default())
            .collect()
    }

    /// Returns the per-group hit blocks, the hit length, and the number of
    /// `num_uncached_common_prefix_tokens` (a shared prefix that a
    /// sparse-retention group has not cached yet; 0 unless hybrid).
    pub fn find_longest_cache_hit(
        &self,
        block_hashes: &[BlockHash],
        max_cache_hit_length: usize,
    ) -> (Vec<Vec<BlockId>>, usize, usize) {
        self.find_longest_cache_hit_with(
            &self.block_pool,
            self.host_pool.as_ref(),
            block_hashes,
            max_cache_hit_length,
        )
    }

    /// [`Self::find_longest_cache_hit`] against an arbitrary cache presence
    /// lookup (e.g. an external tier) instead of the local block pools.
    pub fn find_longest_cache_hit_in(
        &self,
        lookup: &impl CachedBlockLookup,
        block_hashes: &[BlockHash],
        max_cache_hit_length: usize,
    ) -> (Vec<Vec<BlockId>>, usize, usize) {
        self.find_longest_cache_hit_with(lookup, None, block_hashes, max_cache_hit_length)
    }

    /// Looks up the HiSparse host group in `host` if given, the other groups
    /// in `lookup`.
    fn find_longest_cache_hit_with(
        &self,
        lookup: &impl CachedBlockLookup,
        host: Option<&BlockPool>,
        block_hashes: &[BlockHash],
        max_cache_hit_length: usize,
    ) -> (Vec<Vec<BlockId>>, usize, usize) {
        match self.kind {
            CoordinatorKind::NoPrefixCache => {
                (vec![Vec::new(); self.single_type_managers.len()], 0, 0)
            }
            CoordinatorKind::Unitary => {
                let manager = &self.single_type_managers[0];
                let (hit_blocks, hit_length) = with_group_lookup!(self, 0, lookup, host, |pool| {
                    manager.manager_type.find_longest_cache_hit(
                        block_hashes,
                        max_cache_hit_length,
                        &[0],
                        pool,
                        &manager.kv_cache_spec,
                        self.drop_eagle_group_ids.contains(&0),
                        manager.block_size,
                        manager.dcp_world_size,
                        manager.pcp_world_size,
                    )
                });
                // Single group: nothing "uncached common" -- no other group to lag it.
                (hit_blocks, hit_length, 0)
            }
            CoordinatorKind::Hybrid => {
                self.hybrid_find_longest_cache_hit(lookup, host, block_hashes, max_cache_hit_length)
            }
        }
    }

    /// Find the longest cache hit using an iterative fixed-point algorithm.
    ///
    /// Each attention type either accepts the current candidate length or
    /// reduces it. If any type reduces the length, restart checks over all
    /// types. This converges because length monotonically decreases and is
    /// bounded below by 0.
    fn hybrid_find_longest_cache_hit(
        &self,
        lookup: &impl CachedBlockLookup,
        host: Option<&BlockPool>,
        block_hashes: &[BlockHash],
        max_cache_hit_length: usize,
    ) -> (Vec<Vec<BlockId>>, usize, usize) {
        let num_groups = self.kv_cache_config.kv_cache_groups.len();
        let mut hit_length = max_cache_hit_length;
        let mut longest_hit_length = 0;
        let mut hit_blocks_by_group: Vec<Option<Vec<BlockId>>> = vec![None; num_groups];
        let mut hit_length_by_group = vec![0usize; num_groups];
        let alignment_tokens = self.cache_hit_alignment_tokens();

        // Simple hybrid (1 full attn + 1 other): one iteration suffices.
        // Full attn is always first if it exists.
        let is_simple_hybrid =
            self.attention_groups.len() == 2 && self.attention_groups[0].spec.is_full_attention();

        // Attention-group indices whose EAGLE drop is verified at the current
        // ``curr_hit_length``. Each eagle group applies the drop at most once
        // per candidate length (see issue #32802).
        let mut eagle_verified: BTreeSet<usize> = BTreeSet::new();

        loop {
            let mut curr_hit_length = hit_length;

            for (idx, group) in self.attention_groups.iter().enumerate() {
                let first_group_id = group.group_ids[0];
                let manager = &self.single_type_managers[first_group_id];
                // DCP/PCP shard each block's KV across ranks, so the manager's
                // effective block size may exceed the spec's.
                let group_block_size = manager.block_size;
                if group.spec.is_full_attention() && hit_blocks_by_group[first_group_id].is_some() {
                    // Full attention is downward-closed: we only need to look
                    // up cached blocks once; on subsequent iterations just trim
                    // to the (reduced) current hit length.
                    curr_hit_length = curr_hit_length.min(hit_length_by_group[first_group_id]);
                    continue;
                }

                let drop_eagle_block = group.use_eagle && !eagle_verified.contains(&idx);

                let mut max_length = curr_hit_length;
                // Eagle matches one extra drop unit (one hash unit for
                // fine-grained managers, else one cache block) and then drops
                // it, landing back at the candidate length. No margin for
                // mamba: its finder never drops (draft models have no mamba
                // layers), so the hit would grow past the candidate.
                if drop_eagle_block && !group.spec.is_mamba() {
                    let eagle_margin = eagle_proof_margin(
                        group_block_size,
                        self.hash_block_size,
                        self.enable_partial_hash_hits
                            && group.manager_type.supports_fine_grained_hash_lookup(),
                    );
                    max_length = (curr_hit_length + eagle_margin)
                        .min(block_hashes.len() * self.hash_block_size);
                }
                let (hit_blocks, new_hit_length) =
                    with_group_lookup!(self, first_group_id, lookup, host, |pool| {
                        group.manager_type.find_longest_cache_hit(
                            block_hashes,
                            max_length,
                            &group.group_ids,
                            pool,
                            &group.spec,
                            drop_eagle_block,
                            alignment_tokens,
                            manager.dcp_world_size,
                            manager.pcp_world_size,
                        )
                    });
                if drop_eagle_block {
                    eagle_verified.insert(idx);
                } else if new_hit_length < curr_hit_length {
                    // length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear();
                }
                curr_hit_length = new_hit_length;
                for (&group_id, blocks) in group.group_ids.iter().zip(hit_blocks) {
                    hit_blocks_by_group[group_id] = Some(blocks);
                    hit_length_by_group[group_id] = new_hit_length;
                }

                longest_hit_length = longest_hit_length.max(curr_hit_length);
            }

            if curr_hit_length >= hit_length {
                break;
            }
            hit_length = curr_hit_length;
            if is_simple_hybrid {
                break;
            }
        }

        // Truncate every full-attention group (target and draft) blocks
        // to final hit_length.
        for group in &self.attention_groups {
            if !group.spec.is_full_attention() {
                continue;
            }
            let manager = &self.single_type_managers[group.group_ids[0]];
            if manager.manager_type.retains_longer_hit() {
                continue;
            }
            let num_blocks = cdiv(hit_length, manager.block_size);
            for &group_id in &group.group_ids {
                if let Some(blks) = &mut hit_blocks_by_group[group_id] {
                    blks.truncate(num_blocks);
                    hit_length_by_group[group_id] = hit_length;
                }
            }
        }

        // Uncached shared prefix detection: if any attn. group cached a longer
        // prefix than the reconciled hit, it is an uncached common prefix across
        // requests that a sparse-retention group hasn't cached yet.
        let num_uncached_common_prefix_tokens = longest_hit_length.saturating_sub(hit_length);
        let cache_hit_blocks =
            hit_blocks_by_group.into_iter().map(Option::unwrap_or_default).collect();
        (
            cache_hit_blocks,
            hit_length,
            num_uncached_common_prefix_tokens,
        )
    }

    /// Like find_longest_cache_hit but evaluates each group independently.
    /// Hybrid coordinators only.
    ///
    /// Returns:
    ///     (blocks_per_group, hit_lengths_per_group)
    pub fn find_longest_cache_hit_per_group(
        &self,
        block_hashes: &[BlockHash],
        max_cache_hit_length: usize,
    ) -> (Vec<Vec<BlockId>>, Vec<usize>) {
        assert_eq!(self.kind, CoordinatorKind::Hybrid);
        let num_groups = self.kv_cache_config.kv_cache_groups.len();
        let mut hit_blocks = vec![Vec::new(); num_groups];
        let mut hit_lengths = vec![0usize; num_groups];

        for group in &self.attention_groups {
            let manager = &self.single_type_managers[group.group_ids[0]];
            let mut lookup_length = max_cache_hit_length;
            if group.use_eagle && !group.spec.is_mamba() {
                let eagle_margin = eagle_proof_margin(
                    manager.block_size,
                    self.hash_block_size,
                    self.enable_partial_hash_hits
                        && group.manager_type.supports_fine_grained_hash_lookup(),
                );
                lookup_length = (max_cache_hit_length + eagle_margin)
                    .min(block_hashes.len() * self.hash_block_size);
            }
            let (blocks, group_hit) = group.manager_type.find_longest_cache_hit(
                block_hashes,
                lookup_length,
                &group.group_ids,
                self.group_pool(group.group_ids[0]),
                &group.spec,
                group.use_eagle,
                self.cache_hit_alignment_tokens(),
                manager.dcp_world_size,
                manager.pcp_world_size,
            );
            for (&gid, blks) in group.group_ids.iter().zip(blocks) {
                hit_blocks[gid] = blks;
                hit_lengths[gid] = group_hit;
            }
        }
        (hit_blocks, hit_lengths)
    }

    /// Notify each manager that a new step is starting.
    pub fn new_step_starts(&mut self) {
        for manager in &mut self.single_type_managers {
            manager.new_step_starts();
        }
    }
}
