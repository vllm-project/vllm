// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `BlockHashToBlockMap` and `BlockPool` of `block_pool.py`.

use std::hash::BuildHasherDefault;

use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;

use super::block::{BlockHashWithGroupId, BlockId, KvCacheBlock, NULL_BLOCK_ID};
use super::block_hash_list::resolve_block_hashes;
use super::events::{
    AllBlocksCleared, BlockRemoved, BlockStored, EventExtraKeys, ExternalBlockHash, KvCacheEvent,
};
use super::free_queue::FreeKvCacheBlockQueue;
use super::metrics::KvCacheMetricsCollector;
use super::request::{KvRequest, mm_features_in_window_lo};
use crate::hash::BlockHash;

type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// The per-group cached blocks of one block hash, as returned by
/// `BlockPool.get_cached_block`.
pub type CachedBlocks = SmallVec<[BlockId; 4]>;

/// The prefix-cache presence lookup that the cache-hit search runs against.
///
/// Implemented by [`BlockPool`]; external tiers (e.g. the Mooncake store's
/// `ExternalCachedBlockPool`) can implement it to reuse the same hit search.
/// The null block is always [`NULL_BLOCK_ID`].
pub trait CachedBlockLookup {
    /// The block size at which the request block hashes are computed.
    fn hash_block_size(&self) -> usize;

    /// Get the cached block by the block hash for each group in
    /// `kv_cache_group_ids`, or None if cache miss for any group.
    /// If there are duplicated blocks, we return the first block in the cache.
    fn get_cached_block(
        &self,
        block_hash: &BlockHash,
        kv_cache_group_ids: &[usize],
    ) -> Option<CachedBlocks>;
}

/// Cache of blocks that are used for prefix caching, from block hash to
/// one or more cached blocks of the pool.
///
/// A cached block is a full block with a block hash that can be used
/// for prefix caching.
/// The cached block may be used by running requests or in the
/// free_block_queue that could potentially be evicted.
///
/// NOTE #1: We currently don't de-duplicate the blocks in the cache,
/// meaning that if a block becomes full and is cached, we don't check
/// if there is already an identical block in the cache. This is because
/// we want to make sure the allocated block IDs won't change so that
/// block tables are append-only.
#[derive(Clone, Debug, Default)]
pub struct BlockHashToBlockMap {
    /// Block id of one block per key.
    cache: FxHashMap<BlockHashWithGroupId, BlockId>,
    /// All block ids, in insertion order, of keys cached by several blocks.
    dup_ids: FxHashMap<BlockHashWithGroupId, IndexSet<BlockId, FxBuildHasher>>,
}

impl BlockHashToBlockMap {
    /// Gets any block with the given block hash key.
    pub fn get_one_block(&self, key: &BlockHashWithGroupId) -> Option<BlockId> {
        self.cache.get(key).copied()
    }

    /// Checks whether the key maps to the given block ID.
    pub fn contain(&self, key: &BlockHashWithGroupId, block_id: BlockId) -> bool {
        if let Some(dup_ids) = self.dup_ids.get(key) {
            return dup_ids.contains(&block_id);
        }
        self.cache.get(key) == Some(&block_id)
    }

    /// Inserts the KVCacheBlock to the cache
    pub fn insert(&mut self, key: BlockHashWithGroupId, block_id: BlockId) {
        let existing = *self.cache.entry(key.clone()).or_insert(block_id);
        if existing != block_id {
            self.dup_ids
                .entry(key)
                .or_insert_with(|| IndexSet::from_iter([existing]))
                .insert(block_id);
        }
    }

    /// Pops block_id from the blocks cached under key, if it is there.
    pub fn pop(&mut self, key: &BlockHashWithGroupId, block_id: BlockId) -> Option<BlockId> {
        match self.dup_ids.get_mut(key) {
            None => {
                if self.cache.get(key) != Some(&block_id) {
                    return None;
                }
                self.cache.remove(key);
            }
            Some(dup_ids) => {
                if !dup_ids.shift_remove(&block_id) {
                    return None;
                }
                let first = *dup_ids.first().expect("dup ids hold at least one id");
                let remaining = dup_ids.len();
                *self.cache.get_mut(key).expect("dup key is cached") = first;
                if remaining == 1 {
                    self.dup_ids.remove(key);
                }
            }
        }
        Some(block_id)
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.dup_ids.clear();
    }
}

/// BlockPool that manages KVCacheBlocks.
/// It provides methods to allocate, free and cache the kv cache blocks. The
/// free_block_queue stores the free blocks in eviction order to enable
/// allocation, free, and cache eviction. The cached_block_hash_to_block
/// maps between block hash and cached block to support finding cached blocks
/// by their block hash.
pub struct BlockPool {
    pub num_gpu_blocks: usize,
    /// Storage medium reported in KV cache events.
    pub medium: String,
    pub enable_caching: bool,
    /// The block size of which the block hashes are computed.
    /// The actual block size usually equals hash_block_size, but in cases
    /// where different KV cache groups have different block sizes, the
    /// actual block size can be a multiple of hash_block_size.
    pub hash_block_size: usize,
    /// Tokens after each block covered by its hash.
    pub num_hash_lookahead_tokens: usize,
    /// All kv-cache blocks.
    blocks: Vec<KvCacheBlock>,
    /// Free block queue that constructs and manipulates a doubly linked
    /// list of free blocks (including eviction candidates when caching is
    /// enabled).
    pub free_block_queue: FreeKvCacheBlockQueue,
    /// Cache for block lookup
    pub cached_block_hash_to_block: BlockHashToBlockMap,
    /// Secondary (partial) hash keys per block, besides its primary
    /// `block_hash`.
    ///
    /// Deviation: Python keeps an unordered `set`, whose iteration order (and
    /// thus the order of `BlockRemoved` events for secondary keys) depends on
    /// `PYTHONHASHSEED`; this port uses insertion order.
    cached_block_hashes_by_block: FxHashMap<BlockId, IndexSet<BlockHashWithGroupId, FxBuildHasher>>,
    /// To represent a placeholder block with block_id=0.
    /// The ref_cnt of null_block is not maintained, needs special care to
    /// avoid freeing it.
    pub null_block: BlockId,
    pub enable_kv_cache_events: bool,
    /// `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`.
    pub use_int_block_hashes: bool,
    kv_event_queue: Vec<KvCacheEvent>,
    pub metrics_collector: Option<KvCacheMetricsCollector>,
    /// Blocks released with `unpin_blocks` whose contents are still being
    /// read until the pool reuses them.
    ///
    /// Python registers an `on_reuse` callback per block and calls it from
    /// `get_new_blocks`; here the reused blocks queue up in `reused_blocks`
    /// for their reader (the HiSparse coordinator, the only one) to drain
    /// with [`Self::take_reused_blocks`] before it next reads its state.
    reuse_watchers: IndexSet<BlockId, FxBuildHasher>,
    reused_blocks: Vec<BlockId>,
}

impl std::fmt::Debug for BlockPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockPool")
            .field("num_gpu_blocks", &self.num_gpu_blocks)
            .field("num_free_blocks", &self.get_num_free_blocks())
            .field("num_cached_keys", &self.cached_block_hash_to_block.len())
            .finish_non_exhaustive()
    }
}

impl CachedBlockLookup for BlockPool {
    fn hash_block_size(&self) -> usize {
        self.hash_block_size
    }

    fn get_cached_block(
        &self,
        block_hash: &BlockHash,
        kv_cache_group_ids: &[usize],
    ) -> Option<CachedBlocks> {
        BlockPool::get_cached_block(self, block_hash, kv_cache_group_ids)
    }
}

impl BlockPool {
    pub fn new(
        num_gpu_blocks: usize,
        enable_caching: bool,
        hash_block_size: usize,
        enable_kv_cache_events: bool,
        metrics_collector: Option<KvCacheMetricsCollector>,
        num_hash_lookahead_tokens: usize,
        medium: &str,
    ) -> Self {
        assert!(num_gpu_blocks > 0);
        let blocks = (0..num_gpu_blocks).map(|i| KvCacheBlock::new(i as BlockId)).collect();
        let mut free_block_queue = FreeKvCacheBlockQueue::new(num_gpu_blocks);
        let null_block = free_block_queue.popleft();
        debug_assert_eq!(null_block, NULL_BLOCK_ID);
        let mut pool = Self {
            num_gpu_blocks,
            medium: medium.to_owned(),
            enable_caching,
            hash_block_size,
            num_hash_lookahead_tokens,
            blocks,
            free_block_queue,
            cached_block_hash_to_block: BlockHashToBlockMap::default(),
            cached_block_hashes_by_block: FxHashMap::default(),
            null_block,
            enable_kv_cache_events,
            use_int_block_hashes: false,
            kv_event_queue: Vec::new(),
            metrics_collector,
            reuse_watchers: IndexSet::default(),
            reused_blocks: Vec::new(),
        };
        pool.blocks[null_block as usize].is_null = true;
        pool
    }

    /// All kv-cache blocks, indexed by block id.
    pub fn blocks(&self) -> &[KvCacheBlock] {
        &self.blocks
    }

    pub fn block(&self, block_id: BlockId) -> &KvCacheBlock {
        &self.blocks[block_id as usize]
    }

    /// Mutable access to a block, for the managers' ref-count bookkeeping.
    pub(crate) fn block_mut(&mut self, block_id: BlockId) -> &mut KvCacheBlock {
        &mut self.blocks[block_id as usize]
    }

    fn convert_hash(&self, hash: &BlockHash) -> ExternalBlockHash {
        ExternalBlockHash::convert(hash, self.use_int_block_hashes)
    }

    /// Get the cached block by the block hash for each group in
    /// `kv_cache_group_ids`, or None if cache miss for any group.
    /// If there are duplicated blocks, we return the first block in the cache.
    pub fn get_cached_block(
        &self,
        block_hash: &BlockHash,
        kv_cache_group_ids: &[usize],
    ) -> Option<CachedBlocks> {
        let mut cached_blocks = CachedBlocks::new();
        let mut key = BlockHashWithGroupId::new(block_hash.clone(), 0);
        for &group_id in kv_cache_group_ids {
            key.group_id = group_id as u32;
            cached_blocks.push(self.cached_block_hash_to_block.get_one_block(&key)?);
        }
        Some(cached_blocks)
    }

    /// Cache a list of full blocks for prefix caching.
    /// This function takes a list of blocks that will have their block hash
    /// metadata to be updated and cached. Given a request, it updates the
    /// metadata for each block and caching it in the
    /// `cached_block_hash_to_block`.
    /// The block hashes values are computed by the Request object immediately
    /// when it is created and when new tokens are appended.
    ///
    /// Args:
    ///     request: The request to cache the blocks.
    ///     blocks: All blocks in the request.
    ///     num_cached_blocks: The number of blocks that are already cached.
    ///     num_full_blocks: The number of blocks that are full and should
    ///         be cached after this function.
    ///     block_size: Number of tokens in each block.
    ///     kv_cache_group_id: The id of the KV cache group.
    ///     block_mask: Optional mask aligned with
    ///         `blocks[num_cached_blocks:num_full_blocks]`. When provided,
    ///         blocks where the mask is False are skipped (treated like null
    ///         blocks). Used by groups whose `find_longest_cache_hit` only
    ///         consults a subset of blocks (e.g. SWA tail-window), so blocks
    ///         that can never serve a hit stay out of the prefix-cache hash
    ///         map.
    pub fn cache_full_blocks(
        &mut self,
        request: &impl KvRequest,
        blocks: &[BlockId],
        num_cached_blocks: usize,
        num_full_blocks: usize,
        block_size: usize,
        kv_cache_group_id: usize,
        block_mask: Option<&[bool]>,
    ) {
        if num_cached_blocks >= num_full_blocks {
            return;
        }
        // Python slice semantics.
        let new_full_blocks =
            &blocks[num_cached_blocks.min(blocks.len())..num_full_blocks.min(blocks.len())];
        assert!(block_mask.is_none_or(|mask| mask.len() == new_full_blocks.len()));
        let block_hashes = resolve_block_hashes(
            request.block_hashes(),
            self.hash_block_size,
            block_size,
            false,
            None,
        );

        let mut new_hashes: Option<Vec<ExternalBlockHash>> =
            self.enable_kv_cache_events.then(Vec::new);
        let mut key = BlockHashWithGroupId::new(BlockHash(Default::default()), kv_cache_group_id);
        for (i, &blk) in new_full_blocks.iter().enumerate() {
            // Some blocks may be null or masked out when enabling sparse attention
            // like sliding window attention, or Mamba models with prefix-caching
            // in align mode. We skip null blocks here.
            if self.blocks[blk as usize].is_null || block_mask.is_some_and(|mask| !mask[i]) {
                continue;
            }
            let block_hash = block_hashes.get(num_cached_blocks + i);
            let num_hash_tokens = (num_cached_blocks + i + 1) * block_size;

            // Update and added the full block to the cache.
            key.block_hash = block_hash.clone();
            if self.blocks[blk as usize].block_hash.is_some() {
                // The only valid case where a "new full block" already has a
                // hash is partial->full promotion of the same cache block.
                let n = self.blocks[blk as usize].block_hash_num_tokens;
                assert!(n.is_some_and(|n| n < num_hash_tokens));
                let removed_hashes = self.remove_cached_block_hashes(blk);
                self.emit_block_removed_events(&removed_hashes);
            }
            self.insert_block_hash(key.clone(), blk, Some(num_hash_tokens));
            if let Some(new_hashes) = &mut new_hashes {
                new_hashes.push(self.convert_hash(block_hash));
            }
        }

        if let Some(new_hashes) = new_hashes {
            let parent_block_hash = (num_cached_blocks > 0)
                .then(|| self.convert_hash(block_hashes.get(num_cached_blocks - 1)));

            // Calculate token range for the blocks being cached
            let start_token_idx = num_cached_blocks * block_size;
            let end_token_idx = num_full_blocks * block_size;

            // Generate extra keys for each block individually.
            // Each block may have different extra_keys (e.g., different MM
            // features, or cache_salt only for the first block).
            // Skip null/masked-out blocks to match the length of new_hashes.
            let mut extra_keys_list = Vec::new();
            let mut lookahead_token_ids = (self.num_hash_lookahead_tokens > 0).then(Vec::new);
            let mut curr_mm_idx = 0;
            for i in num_cached_blocks..num_full_blocks {
                if self.blocks[blocks[i] as usize].is_null {
                    continue;
                }
                if block_mask.is_some_and(|mask| !mask[i - num_cached_blocks]) {
                    continue;
                }
                let block_start = i * block_size;
                let block_end = block_start + block_size;
                let (extra_keys, next_mm_idx) =
                    self.generate_event_extra_keys(request, block_start, block_end, curr_mm_idx);
                curr_mm_idx = next_mm_idx;
                extra_keys_list.push(extra_keys);
                if let Some(lookahead) = &mut lookahead_token_ids {
                    lookahead.push(self.get_lookahead_token_ids(request, block_end));
                }
            }

            let event = self.build_block_stored_event(
                request,
                new_hashes,
                parent_block_hash,
                start_token_idx,
                end_token_idx,
                block_size,
                kv_cache_group_id,
                extra_keys_list,
                lookahead_token_ids,
            );
            self.kv_event_queue.push(KvCacheEvent::BlockStored(event));
        }
    }

    /// Event extra keys cover the same window as the block hash.
    fn generate_event_extra_keys(
        &self,
        request: &impl KvRequest,
        start: usize,
        end: usize,
        start_mm_idx: isize,
    ) -> (Option<EventExtraKeys>, isize) {
        if self.num_hash_lookahead_tokens > 0 {
            // `generate_lookahead_block_hash_extra_keys`: rewind the MM index to
            // retain inputs overlapping the next block.
            let (extra_keys, mut mm_idx) =
                request.event_extra_keys(start, end + self.num_hash_lookahead_tokens, start_mm_idx);
            while mm_idx > 0 {
                let (offset, length) = request.mm_feature_range(mm_idx as usize - 1);
                if offset + length <= end {
                    break;
                }
                mm_idx -= 1;
            }
            return (extra_keys, mm_idx);
        }
        request.event_extra_keys(start, end, start_mm_idx)
    }

    fn get_lookahead_token_ids(&self, request: &impl KvRequest, block_end: usize) -> Vec<i32> {
        request.token_ids(block_end, block_end + self.num_hash_lookahead_tokens)
    }

    /// Build a `BlockStored` KV event for `request`.
    ///
    /// Shared by `cache_full_blocks` (newly cached blocks) and
    /// `emit_cached_block_events` (prefix-cache-reused blocks) so both emit
    /// identical event shapes for downstream consumers.
    fn build_block_stored_event(
        &self,
        request: &impl KvRequest,
        block_hashes: Vec<ExternalBlockHash>,
        parent_block_hash: Option<ExternalBlockHash>,
        start_token_idx: usize,
        end_token_idx: usize,
        block_size: usize,
        kv_cache_group_id: usize,
        extra_keys_list: Vec<Option<EventExtraKeys>>,
        lookahead_token_ids: Option<Vec<Vec<i32>>>,
    ) -> BlockStored {
        BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids: request.token_ids(start_token_idx, end_token_idx),
            block_size,
            lora_id: request.lora_id(),
            medium: Some(self.medium.clone()),
            lora_name: request.lora_name(),
            // `to_event_extra_keys`: None for an empty list.
            extra_keys: (!extra_keys_list.is_empty()).then_some(extra_keys_list),
            group_idx: Some(kv_cache_group_id),
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            session_id: request.session_id(),
            lookahead_token_ids,
        }
    }

    /// Generate BlockStored events for blocks reused from prefix cache.
    ///
    /// Unlike cache_full_blocks(), this does NOT modify block state —
    /// the blocks are already cached. It only generates events so that
    /// external consumers (e.g. gateway) can learn about reused blocks.
    pub fn emit_cached_block_events(
        &mut self,
        request: &impl KvRequest,
        num_cached_blocks: usize,
        block_size: usize,
        kv_cache_group_id: usize,
    ) {
        if !self.enable_kv_cache_events || num_cached_blocks == 0 {
            return;
        }
        let block_hashes = resolve_block_hashes(
            request.block_hashes(),
            self.hash_block_size,
            block_size,
            false,
            None,
        );

        // Collect external hashes and extra_keys for cached blocks.
        let mut cached_hashes = Vec::with_capacity(num_cached_blocks);
        let mut extra_keys_list = Vec::with_capacity(num_cached_blocks);
        let mut lookahead_token_ids = (self.num_hash_lookahead_tokens > 0).then(Vec::new);
        let mut curr_mm_idx = 0;
        for i in 0..num_cached_blocks {
            let block_start = i * block_size;
            let block_end = block_start + block_size;
            cached_hashes.push(self.convert_hash(block_hashes.get(i)));
            let (extra_keys, next_mm_idx) =
                self.generate_event_extra_keys(request, block_start, block_end, curr_mm_idx);
            curr_mm_idx = next_mm_idx;
            extra_keys_list.push(extra_keys);
            if let Some(lookahead) = &mut lookahead_token_ids {
                lookahead.push(self.get_lookahead_token_ids(request, block_end));
            }
        }

        // Prefix-cache hits always form a contiguous prefix starting at block 0,
        // so the first (and thus the whole group's) parent block hash is None.
        let event = self.build_block_stored_event(
            request,
            cached_hashes,
            None,
            0,
            num_cached_blocks * block_size,
            block_size,
            kv_cache_group_id,
            extra_keys_list,
            lookahead_token_ids,
        );
        self.kv_event_queue.push(KvCacheEvent::BlockStored(event));
    }

    /// Register a partial prefix-cache entry for an existing block.
    ///
    /// Prefix-cache keys normally identify full cache blocks. A partial entry
    /// makes an existing cache block reachable from a fine-grained prefix
    /// boundary inside that block without allocating or copying a new
    /// `KVCacheBlock`.
    ///
    /// The partial entry is lookup metadata owned by `block`. If `block`
    /// has no primary hash, the key becomes its primary hash. If the block
    /// already has a primary hash, the partial entry is tracked in
    /// `cached_block_hashes_by_block` so eviction, reset, and promotion can
    /// remove every hash key that points to the block.
    ///
    /// Args:
    ///     request: Request whose token IDs and block hashes define the
    ///         partial entry.
    ///     block: Existing cache block to make reachable from the partial
    ///         prefix boundary.
    ///     num_tokens: Prefix length represented by the partial entry. It
    ///         must be a positive multiple of `self.hash_block_size` and
    ///         cannot exceed the request's computed block hashes.
    ///     kv_cache_group_id: KV cache group that owns the partial entry.
    ///     block_size: Cache block size for the owning group. The partial
    ///         entry hash itself is always the prefix-chain hash at
    ///         `num_tokens`; `block_size` must be a multiple of the hash
    ///         block size.
    ///     replace_existing_hashes: Whether the block contents were replaced
    ///         and all existing cache entries must be removed before the new
    ///         entry is registered.
    ///
    /// Returns:
    ///     The hash key with group ID if a partial entry can be registered;
    ///     otherwise `None` for null blocks.
    pub fn cache_partial_block(
        &mut self,
        request: &impl KvRequest,
        block: BlockId,
        num_tokens: usize,
        kv_cache_group_id: usize,
        block_size: usize,
        replace_existing_hashes: bool,
    ) -> Option<BlockHashWithGroupId> {
        if self.blocks[block as usize].is_null {
            return None;
        }

        assert!(block_size.is_multiple_of(self.hash_block_size));
        let block_hash = self.get_partial_block_hash(request, num_tokens).clone();
        let num_hash_blocks = num_tokens / self.hash_block_size;
        let key = BlockHashWithGroupId::new(block_hash.clone(), kv_cache_group_id);
        let mut already_cached = self.blocks[block as usize].block_hash.as_ref() == Some(&key)
            || self.cached_block_hash_to_block.contain(&key, block);
        if replace_existing_hashes {
            let removed_hashes = self.remove_cached_block_hashes(block);
            self.emit_block_removed_events(&removed_hashes);
            already_cached = false;
        } else if !already_cached
            && self.blocks[block as usize].block_hash.is_some()
            && self.blocks[block as usize]
                .block_hash_num_tokens
                .is_some_and(|n| n < num_hash_blocks * self.hash_block_size)
        {
            let removed_hashes = self.remove_cached_block_hashes(block);
            self.emit_block_removed_events(&removed_hashes);
        }
        self.insert_block_hash(
            key.clone(),
            block,
            Some(num_hash_blocks * self.hash_block_size),
        );
        if self.enable_kv_cache_events && !already_cached {
            let (parent_hash, block_start) =
                self.get_partial_block_parent_hash_and_start(request, num_tokens);
            let parent_block_hash = parent_hash.map(|h| self.convert_hash(h));
            let block_end = num_tokens;
            let mut curr_mm_idx = 0;
            let num_mm_features = request.num_mm_features();
            if block_start > 0 && num_mm_features > 0 {
                let (offset, length) = request.mm_feature_range(num_mm_features - 1);
                if offset + length > block_start {
                    curr_mm_idx = mm_features_in_window_lo(request, block_start) as isize;
                }
            }
            let (extra_keys, _) =
                self.generate_event_extra_keys(request, block_start, block_end, curr_mm_idx);
            let event = BlockStored {
                block_hashes: vec![self.convert_hash(&block_hash)],
                parent_block_hash,
                token_ids: request.token_ids(block_start, block_end),
                block_size: block_end - block_start,
                lora_id: request.lora_id(),
                medium: Some(self.medium.clone()),
                lora_name: request.lora_name(),
                extra_keys: Some(vec![extra_keys]),
                group_idx: Some(kv_cache_group_id),
                kv_cache_spec_kind: None,
                kv_cache_spec_sliding_window: None,
                session_id: request.session_id(),
                lookahead_token_ids: (self.num_hash_lookahead_tokens > 0)
                    .then(|| vec![self.get_lookahead_token_ids(request, block_end)]),
            };
            self.kv_event_queue.push(KvCacheEvent::BlockStored(event));
        }
        Some(key)
    }

    fn get_partial_block_hash<'r>(
        &self,
        request: &'r impl KvRequest,
        num_tokens: usize,
    ) -> &'r BlockHash {
        assert!(num_tokens.is_multiple_of(self.hash_block_size));
        let num_hash_blocks = num_tokens / self.hash_block_size;
        let block_hashes = request.block_hashes();
        assert!(0 < num_hash_blocks && num_hash_blocks <= block_hashes.len());

        // Each hash_block_size hash chains over its full prefix, so the partial
        // entry for any group block size is the hash at that prefix boundary.
        &block_hashes[num_hash_blocks - 1]
    }

    fn get_partial_block_parent_hash_and_start<'r>(
        &self,
        request: &'r impl KvRequest,
        num_tokens: usize,
    ) -> (Option<&'r BlockHash>, usize) {
        let num_hash_blocks = num_tokens / self.hash_block_size;
        let parent_hash =
            (num_hash_blocks > 1).then(|| &request.block_hashes()[num_hash_blocks - 2]);
        let block_start = (num_hash_blocks - 1) * self.hash_block_size;
        (parent_hash, block_start)
    }

    fn remove_cached_block_hashes(&mut self, block: BlockId) -> Vec<BlockHashWithGroupId> {
        // A block only has secondary hashes alongside a primary one.
        let Some(primary) = self.blocks[block as usize].block_hash.clone() else {
            return Vec::new();
        };
        let secondary = self.cached_block_hashes_by_block.remove(&block);
        let mut removed_hashes = Vec::with_capacity(1 + secondary.as_ref().map_or(0, |s| s.len()));
        for block_hash in std::iter::once(primary).chain(secondary.into_iter().flatten()) {
            if self.cached_block_hash_to_block.pop(&block_hash, block).is_some() {
                removed_hashes.push(block_hash);
            }
        }
        self.blocks[block as usize].reset_hash();
        removed_hashes
    }

    fn emit_block_removed_events(&mut self, block_hashes: &[BlockHashWithGroupId]) {
        if !self.enable_kv_cache_events {
            return;
        }
        for block_hash in block_hashes {
            let event = BlockRemoved {
                block_hashes: vec![self.convert_hash(&block_hash.block_hash)],
                medium: Some(self.medium.clone()),
                group_idx: Some(block_hash.group_id as usize),
            };
            self.kv_event_queue.push(KvCacheEvent::BlockRemoved(event));
        }
    }

    fn insert_block_hash(
        &mut self,
        key: BlockHashWithGroupId,
        block: BlockId,
        num_tokens: Option<usize>,
    ) {
        let blk = &mut self.blocks[block as usize];
        match &blk.block_hash {
            // A block without a hash has no entries in the cache yet.
            None => blk.set_block_hash(key.clone(), num_tokens),
            Some(primary) => {
                if *primary == key || self.cached_block_hash_to_block.contain(&key, block) {
                    return;
                }
                self.cached_block_hashes_by_block.entry(block).or_default().insert(key.clone());
            }
        }
        self.cached_block_hash_to_block.insert(key, block);
    }

    /// Re-point `src_block`'s prefix-cache entries to `dst_block`.
    ///
    /// Used when the request owning `src_block` keeps writing into it
    /// : the prefix cache holds a private copy (`dst_block`)
    /// under the same hashes instead. Entries stay live; no events emitted.
    pub fn move_block_hashes(&mut self, src_block: BlockId, dst_block: BlockId) {
        assert!(self.blocks[dst_block as usize].block_hash.is_none());
        assert!(!self.cached_block_hashes_by_block.contains_key(&dst_block));
        let num_tokens = self.blocks[src_block as usize].block_hash_num_tokens;
        for block_hash in self.remove_cached_block_hashes(src_block) {
            // `num_tokens` only applies to the first (primary) insertion.
            self.insert_block_hash(block_hash, dst_block, num_tokens);
        }
    }

    /// Get new blocks from the free block pool.
    ///
    /// Note that we do not check block cache in this function.
    ///
    /// Panics (Python raises `ValueError`) if the pool does not have
    /// `num_blocks` free blocks; callers check capacity first.
    pub fn get_new_blocks(&mut self, num_blocks: usize) -> Vec<BlockId> {
        if num_blocks > self.get_num_free_blocks() {
            panic!("Cannot get {num_blocks} free blocks from the pool");
        }

        let ret = self.free_block_queue.popleft_n(num_blocks);

        if !self.reuse_watchers.is_empty() {
            self.notify_reuse(&ret);
        }

        // In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching {
            if self.metrics_collector.is_some() {
                for &block in &ret {
                    self.maybe_evict_cached_block(block);
                    let blk = &mut self.blocks[block as usize];
                    assert!(blk.ref_cnt == 0);
                    blk.ref_cnt = 1;
                    if let Some(metrics) = &mut self.metrics_collector {
                        metrics.on_block_allocated(block);
                    }
                }
            } else {
                for &block in &ret {
                    // Most blocks were never cached, so skip the call when possible.
                    if self.blocks[block as usize].block_hash.is_some() {
                        self.maybe_evict_cached_block(block);
                    }
                    let blk = &mut self.blocks[block as usize];
                    assert!(blk.ref_cnt == 0);
                    blk.ref_cnt = 1;
                }
            }
        } else {
            for &block in &ret {
                let blk = &mut self.blocks[block as usize];
                assert!(blk.ref_cnt == 0);
                blk.ref_cnt = 1;
                if let Some(metrics) = &mut self.metrics_collector {
                    metrics.on_block_allocated(block);
                }
            }
        }
        ret
    }

    fn notify_reuse(&mut self, blocks: &[BlockId]) {
        for &block in blocks {
            if self.reuse_watchers.shift_remove(&block) {
                self.reused_blocks.push(block);
            }
        }
    }

    /// Release references to blocks that stay readable until reused.
    ///
    /// The blocks become last-resort eviction candidates regardless of prefix
    /// caching: they join the tail of the free queue and count as free.
    /// Once `get_new_blocks` hands a block out (or the cache is reset), it is
    /// reported by [`Self::take_reused_blocks`], so the caller can stop
    /// reading it.
    pub fn unpin_blocks(&mut self, blocks: &[BlockId]) {
        let mut released = Vec::new();
        for &block in blocks {
            let blk = &mut self.blocks[block as usize];
            assert!(blk.ref_cnt > 0 && !blk.is_null);
            blk.ref_cnt -= 1;
            self.reuse_watchers.insert(block);
            if blk.ref_cnt == 0 {
                released.push(block);
            }
        }
        self.free_block_queue.append_n(&released);
    }

    /// Drain the unpinned blocks the pool reused since the last call, in
    /// reuse order (the order Python calls their `on_reuse` callbacks).
    pub fn take_reused_blocks(&mut self) -> Vec<BlockId> {
        std::mem::take(&mut self.reused_blocks)
    }

    /// Move this pool's queued KV events to the end of `owner`'s queue.
    ///
    /// Python's `SharedEventQueueBlockPool` (the HiSparse host pool) appends
    /// to its owner's live queue; the owner of this pool calls this after
    /// each operation on it, keeping the events in order.
    pub(crate) fn forward_events_to(&mut self, owner: &mut BlockPool) {
        if !self.kv_event_queue.is_empty() {
            owner.kv_event_queue.append(&mut self.kv_event_queue);
        }
    }

    /// If a block is cached in `cached_block_hash_to_block`, we reset its hash
    /// metadata and evict it from the cache.
    ///
    /// Returns:
    ///     True if the block is evicted, False otherwise.
    pub(crate) fn maybe_evict_cached_block(&mut self, block: BlockId) -> bool {
        // Clean up metrics tracking first to prevent leaks
        if let Some(metrics) = &mut self.metrics_collector {
            metrics.on_block_evicted(block);
        }

        let evicted_hashes = self.remove_cached_block_hashes(block);
        if evicted_hashes.is_empty() {
            // The block doesn't have hash, eviction is not needed
            return false;
        }
        self.emit_block_removed_events(&evicted_hashes);
        true
    }

    /// Touch a block increases its reference count by 1, and may remove
    /// the block from the free queue. This is used when a block is hit by
    /// another request with the same prefix.
    pub fn touch(&mut self, blocks: &[BlockId]) {
        for &block in blocks {
            let blk = &mut self.blocks[block as usize];
            if blk.is_null {
                continue;
            }
            // ref_cnt=0 means this block is in the free list (i.e. eviction
            // candidate), so remove it.
            if blk.ref_cnt == 0 {
                self.free_block_queue.remove(block);
            }
            blk.ref_cnt += 1;
            if let Some(metrics) = &mut self.metrics_collector {
                metrics.on_block_accessed(block);
            }
        }
    }

    /// Return whether a block can be mutated by its sole owner.
    pub fn is_block_writable(&self, block: BlockId) -> bool {
        let blk = &self.blocks[block as usize];
        !blk.is_null && blk.ref_cnt == 1 && blk.block_hash.is_none()
    }

    /// Free a list of blocks. The blocks should be ordered by their
    /// eviction priority, where the first block will be evicted first.
    ///
    /// Args:
    ///     ordered_blocks: A list of blocks to free ordered by their eviction
    ///         priority.
    ///     evict_first: Also put freed cached blocks at the front of the free
    ///         queue, so they are evicted before other cached blocks. For
    ///         blocks whose contents another tier already holds.
    ///
    /// Python routes blocks owned by another pool (HiSparse host blocks) to
    /// that pool; here [`KvCacheManager::free_blocks`] does, for block lists
    /// that mix pools.
    ///
    /// [`KvCacheManager::free_blocks`]: super::KvCacheManager::free_blocks
    pub fn free_blocks(
        &mut self,
        ordered_blocks: impl IntoIterator<Item = BlockId>,
        evict_first: bool,
    ) {
        // Identify blocks with hash (LRU cache) and without it (never match APC)
        let mut blocks_to_evict_last = Vec::new();
        let mut blocks_to_evict_first = Vec::new();
        for block in ordered_blocks {
            let blk = &mut self.blocks[block as usize];
            if blk.is_null {
                // never pooled; its ref_cnt is not maintained
                continue;
            }
            blk.ref_cnt -= 1;
            if blk.ref_cnt == 0 {
                if blk.block_hash.is_none() || !self.enable_caching || evict_first {
                    // LIFO reuse of non-cached blocks for better GPU locality,
                    // and of cached ones when asked to evict them first.
                    blocks_to_evict_first.push(block);
                } else {
                    // FIFO reuse of cached blocks for LRU eviction behavior.
                    blocks_to_evict_last.push(block);
                }
            }
        }
        // Blocks to reuse first are prepended to the front of the free queue.
        self.free_block_queue.prepend_n(&blocks_to_evict_first);
        // Blocks to reuse last are appended to the end of the free queue.
        self.free_block_queue.append_n(&blocks_to_evict_last);
    }

    /// Evict blocks from the prefix cache by their block IDs.
    ///
    /// only evicts blocks that are currently cached (have a hash). blocks
    /// with ref_cnt > 0 are not freed from the block pool, only evicted
    /// from the prefix cache hash table.
    ///
    /// Python takes a `set[int]`; events follow the iteration order of
    /// `block_ids`, so pass the set's iteration order for parity.
    pub fn evict_blocks(&mut self, block_ids: impl IntoIterator<Item = BlockId>) {
        for block_id in block_ids {
            assert!(
                (block_id as usize) < self.blocks.len(),
                "Invalid block_id {block_id} >= {}. This indicates a bug in the KV connector - \
                 workers should only report block IDs that were allocated by the scheduler.",
                self.blocks.len()
            );
            self.maybe_evict_cached_block(block_id);
        }
    }

    /// Reset prefix cache. This function may be used in RLHF
    /// flows to invalidate prefix caching after the weights are updated,
    /// or used for resetting prefix caching status for benchmarking.
    ///
    /// Returns:
    ///     bool: True if the prefix cache is successfully reset,
    ///     False otherwise.
    pub fn reset_prefix_cache(&mut self) -> bool {
        let num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks();
        if num_used_blocks != 1 {
            // The null block is always marked as used.
            // TODO: Python logs "Failed to reset prefix cache because some
            // blocks (%d) are not freed yet".
            return false;
        }

        // Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block.clear();
        self.cached_block_hashes_by_block.clear();
        if !self.reuse_watchers.is_empty() {
            let watched: Vec<BlockId> = self.reuse_watchers.iter().copied().collect();
            self.notify_reuse(&watched);
        }

        // Remove all hashes from all blocks.
        for block in &mut self.blocks {
            block.reset_hash();
        }

        if let Some(metrics) = &mut self.metrics_collector {
            metrics.reset();
        }

        if self.enable_kv_cache_events {
            self.kv_event_queue.push(KvCacheEvent::AllBlocksCleared(AllBlocksCleared));
        }
        true
    }

    /// Get the number of free blocks in the pool.
    /// `cached_block_hashes_by_block`: each block's secondary (partial)
    /// hash keys, besides its primary `block_hash`.
    pub fn cached_block_hashes_by_block(
        &self,
    ) -> impl Iterator<Item = (BlockId, &IndexSet<BlockHashWithGroupId, FxBuildHasher>)> {
        self.cached_block_hashes_by_block.iter().map(|(&b, hashes)| (b, hashes))
    }

    pub fn get_num_free_blocks(&self) -> usize {
        self.free_block_queue.num_free_blocks
    }

    /// Get the KV cache usage (between 0.0 and 1.0).
    pub fn get_usage(&self) -> f64 {
        // Subtract 1 to account for null block.
        let total_gpu_blocks = self.num_gpu_blocks - 1;
        if total_gpu_blocks == 0 {
            return 0.0;
        }
        1.0 - (self.get_num_free_blocks() as f64 / total_gpu_blocks as f64)
    }

    /// Atomically takes all events and clears the queue.
    pub fn take_events(&mut self) -> Vec<KvCacheEvent> {
        if !self.enable_kv_cache_events {
            return Vec::new();
        }
        std::mem::take(&mut self.kv_event_queue)
    }
}
