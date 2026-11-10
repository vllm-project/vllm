// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The class methods of the single-type managers: `find_longest_cache_hit` and
//! `reachable_block_mask`. They only read the cache through
//! [`CachedBlockLookup`], so external tiers can reuse them.

use super::ManagerType;
use crate::hash::BlockHash;
use crate::kv_cache::block::{BlockId, NULL_BLOCK_ID};
use crate::kv_cache::block_hash_list::{BlockHashListWithBlockSize, resolve_block_hashes};
use crate::kv_cache::block_pool::{CachedBlockLookup, CachedBlocks};
use crate::kv_cache::cdiv;
use crate::kv_cache::spec::{KvCacheSpec, KvCacheSpecType};

/// The per-group hit blocks (skipped blocks replaced by the null block) and
/// the exact cache-hit length in tokens.
pub type CacheHit = (Vec<Vec<BlockId>>, usize);

/// `get_cached_block` with Python truthiness: a hit for an empty group list
/// is an empty (falsy) list.
fn lookup(
    pool: &impl CachedBlockLookup,
    hash: &BlockHash,
    group_ids: &[usize],
) -> Option<CachedBlocks> {
    pool.get_cached_block(hash, group_ids).filter(|blocks| !blocks.is_empty())
}

fn push_hit(computed: &mut [Vec<BlockId>], cached: &CachedBlocks) {
    for (computed, &cached) in computed.iter_mut().zip(cached) {
        computed.push(cached);
    }
}

/// Floor division of a Python int.
fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

impl ManagerType {
    /// Get the longest cache hit prefix of the blocks that is not longer than
    /// `max_length`. The prefix should be a common prefix hit for all the
    /// kv cache groups in `kv_cache_group_ids`. If no cache hit is found,
    /// return an empty list.
    /// If eagle is enabled, drop the last matched block to force recompute the
    /// last block to get the required hidden states for eagle drafting head.
    /// For multi-module MTP, this recompute also rewrites the dropped block's
    /// draft-layer KVs, which depend on up to num_speculative_tokens - 1
    /// tokens past the matched prefix (i.e. on the cache writer's
    /// continuation, which the block hash does not cover); the coordinator
    /// asserts the block size covers that window.
    ///
    /// Args:
    ///     block_hashes: The block hashes of the request.
    ///     max_length: The maximum length of the cache hit prefix.
    ///     kv_cache_group_ids: The ids of the kv cache groups.
    ///     block_pool: The block pool.
    ///     kv_cache_spec: The kv cache spec.
    ///     drop_eagle_block: Whether to drop the last matched block for EAGLE/MTP.
    ///         Always False for non-EAGLE/MTP groups, but can be False for EAGLE/MTP
    ///         groups too if the last block is already dropped (e.g., in a
    ///         convergence loop in `find_longest_cache_hit`).
    ///     alignment_tokens: The returned cache hit length (in tokens) should
    ///         be a multiple of this value (in tokens). By default, it should
    ///         be set to the block_size.
    ///     dcp_world_size: The world size of decode context parallelism.
    ///     pcp_world_size: The world size of prefill context parallelism.
    ///
    /// Returns:
    ///     A tuple containing cached blocks and the exact cache-hit length in
    ///     tokens. The cached block tuple has skipped blocks replaced by null
    ///     blocks for each kv cache group in `kv_cache_group_ids`.
    ///     For example, sliding window manager should return a list like
    ///     ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) for block size 4
    ///     and sliding window 8 and len(kv_cache_group_ids) = 1.
    pub fn find_longest_cache_hit(
        self,
        block_hashes: &[BlockHash],
        max_length: usize,
        kv_cache_group_ids: &[usize],
        block_pool: &impl CachedBlockLookup,
        kv_cache_spec: &KvCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: usize,
        dcp_world_size: usize,
        pcp_world_size: usize,
    ) -> CacheHit {
        match self {
            Self::FullAttention | Self::Rswa | Self::SinkFullAttention | Self::HiSparseSource => {
                full_attention_hit(
                    self,
                    block_hashes,
                    max_length,
                    kv_cache_group_ids,
                    block_pool,
                    kv_cache_spec,
                    drop_eagle_block,
                    alignment_tokens,
                    dcp_world_size,
                )
            }
            Self::SlidingWindow => sliding_window_hit(
                block_hashes,
                max_length,
                kv_cache_group_ids,
                block_pool,
                kv_cache_spec,
                drop_eagle_block,
                alignment_tokens,
                dcp_world_size,
                pcp_world_size,
            ),
            // `_HiSparseAuxiliaryManager`: the host source owns prefix caching.
            Self::CircularBuffer | Self::HiSparseHot | Self::HiSparseResident => {
                (vec![Vec::new(); kv_cache_group_ids.len()], 0)
            }
            Self::ChunkedLocalAttention => chunked_local_hit(
                block_hashes,
                max_length,
                kv_cache_group_ids,
                block_pool,
                kv_cache_spec,
                drop_eagle_block,
                alignment_tokens,
                dcp_world_size,
                pcp_world_size,
            ),
            Self::Mamba => mamba_hit(
                block_hashes,
                max_length,
                kv_cache_group_ids,
                block_pool,
                kv_cache_spec,
                alignment_tokens,
                dcp_world_size,
                pcp_world_size,
            ),
            Self::CrossAttention => {
                assert!(
                    matches!(kv_cache_spec.spec_type, KvCacheSpecType::CrossAttention),
                    "CrossAttentionManager can only be used for cross-attention groups"
                );
                // Cross-attention does not benefit from prefix caching since:
                // 1. Encoder states are unique per request (different audio/image
                //    inputs)
                // 2. Encoder states are computed once per request, not incrementally
                // 3. No reusable prefix exists between different multimodal inputs
                panic!("CrossAttentionManager does not support caching")
            }
        }
    }

    /// Per-block mask for `cache_full_blocks`. `None` means cache
    /// every (non-null) block — the default for full attention.
    ///
    /// Subclasses with sparse hit semantics (SWA / Mamba) override this to skip
    /// blocks that can never serve a hit at any alignment-aligned prefix length.
    /// `reachable_boundaries` are token positions whose reachable tail must be
    /// retained; the base (dense) policy ignores them.
    /// `final_segment_end_block` is the exclusive end of the request's final
    /// segment. It may be later than `end_block` while prefill is still in
    /// progress. Non-EAGLE sparse managers may keep its reachable tail. The
    /// base policy already keeps every block, so it ignores this value.
    ///
    /// Boundaries are signed: Python may pass `shared_prefix_boundary -
    /// block_size`, which can be negative (and then selects nothing).
    pub fn reachable_block_mask(
        self,
        start_block: usize,
        end_block: usize,
        alignment_tokens: Option<usize>,
        kv_cache_spec: &KvCacheSpec,
        use_eagle: bool,
        retention_interval: Option<usize>,
        reachable_boundaries: &[i64],
        dcp_world_size: usize,
        final_segment_end_block: Option<usize>,
    ) -> Option<Vec<bool>> {
        match self {
            Self::SlidingWindow => sliding_window_mask(
                start_block,
                end_block,
                alignment_tokens,
                kv_cache_spec,
                use_eagle,
                retention_interval,
                reachable_boundaries,
                dcp_world_size,
                final_segment_end_block,
            ),
            Self::Mamba => mamba_mask(
                start_block,
                end_block,
                alignment_tokens,
                kv_cache_spec,
                retention_interval,
                reachable_boundaries,
            ),
            _ => None,
        }
    }
}

fn full_attention_hit(
    manager_type: ManagerType,
    block_hashes: &[BlockHash],
    max_length: usize,
    kv_cache_group_ids: &[usize],
    block_pool: &impl CachedBlockLookup,
    kv_cache_spec: &KvCacheSpec,
    drop_eagle_block: bool,
    alignment_tokens: usize,
    dcp_world_size: usize,
) -> CacheHit {
    assert!(
        kv_cache_spec.is_full_attention()
            || matches!(
                kv_cache_spec.spec_type,
                KvCacheSpecType::ChunkedLocalAttention { .. }
            ),
        "FullAttentionManager can only be used for full attention and chunked local attention groups"
    );
    let mut block_size = kv_cache_spec.block_size;
    if dcp_world_size > 1 {
        // DCP shards each block's KV across ranks; hashes must be viewed at
        // the sharded block size.
        block_size *= dcp_world_size;
    }
    let resolved = resolve_block_hashes(
        block_hashes,
        block_pool.hash_block_size(),
        block_size,
        manager_type.supports_fine_grained_hash_lookup(),
        Some(alignment_tokens),
    );

    // Fine-grained mode (alignment_tokens == hash_block_size <
    // block_size): resolve_block_hashes kept the raw hash-granularity
    // list so interior boundaries can be probed.
    let fine_grained = alignment_tokens < block_size && block_size.is_multiple_of(alignment_tokens);
    let full_block_hashes = if fine_grained {
        assert_eq!(resolved.scale_factor, 1);
        BlockHashListWithBlockSize::new(block_hashes, alignment_tokens, block_size)
    } else {
        resolved
    };

    let mut computed_blocks = vec![Vec::new(); kv_cache_group_ids.len()];
    // Phase 1: longest run of cached full blocks from the start. A missing
    // block implies every later block misses too (chained hashes).
    for block_hash in full_block_hashes.iter().take(max_length / block_size) {
        let Some(cached_block) = lookup(block_pool, block_hash, kv_cache_group_ids) else {
            break;
        };
        push_hit(&mut computed_blocks, &cached_block);
    }
    let num_full_hit_blocks = computed_blocks[0].len();
    let mut hit_length = num_full_hit_blocks * block_size;

    // Phase 2 (fine-grained only): extend into the first non-full block by
    // probing its interior hash boundaries high-to-low (longest hit first).
    if fine_grained {
        let scale_factor = block_size / alignment_tokens;
        let first_partial_idx = num_full_hit_blocks * scale_factor;
        let max_partial_idx = (first_partial_idx + scale_factor - 1)
            .min(max_length / alignment_tokens)
            .min(block_hashes.len());
        for fine_idx in (first_partial_idx..max_partial_idx).rev() {
            let Some(cached_tail) = lookup(block_pool, &block_hashes[fine_idx], kv_cache_group_ids)
            else {
                continue;
            };
            push_hit(&mut computed_blocks, &cached_tail);
            hit_length = (fine_idx + 1) * alignment_tokens;
            break;
        }
    }

    // Eagle needs the tokens right before the generation point recomputed:
    // drop one hash unit when fine-grained (the tail block's KV is
    // append-only, so it still covers the reduced length), else one cache
    // block.
    if drop_eagle_block && hit_length > 0 {
        hit_length = hit_length.saturating_sub(alignment_tokens.min(block_size));
    }
    // Round down to the alignment; a no-op when fine-grained (hits land on
    // hash boundaries by construction) and when alignment_tokens ==
    // block_size. Then trim blocks past the new tail.
    hit_length -= hit_length % alignment_tokens;
    let num_blocks = cdiv(hit_length, block_size);
    for computed in &mut computed_blocks {
        computed.truncate(num_blocks);
    }
    (computed_blocks, hit_length)
}

/// The number of contiguous blocks needed for a sliding-window prefix hit.
pub(crate) fn contiguous_blocks_for_hit(
    window_size: usize,
    block_size: usize,
    use_eagle: bool,
) -> usize {
    let mut blocks = cdiv(window_size - 1, block_size);
    if use_eagle {
        // Need to drop the last matched block if eagle is enabled. For
        // sliding window layer, we achieve this by increasing the number of
        // contiguous blocks needed for prefix cache hit by one and dropping
        // the last matched block.
        blocks += 1;
    }
    blocks
}

fn sliding_window_hit(
    block_hashes: &[BlockHash],
    max_length: usize,
    kv_cache_group_ids: &[usize],
    block_pool: &impl CachedBlockLookup,
    kv_cache_spec: &KvCacheSpec,
    drop_eagle_block: bool,
    alignment_tokens: usize,
    dcp_world_size: usize,
    pcp_world_size: usize,
) -> CacheHit {
    let sliding_window = kv_cache_spec
        .sliding_window()
        .expect("SlidingWindowManager can only be used for sliding window groups");
    assert!(
        dcp_world_size == 1,
        "DCP not support sliding window attn now."
    );
    assert!(
        pcp_world_size == 1,
        "PCP not support sliding window attn now."
    );
    let block_size = kv_cache_spec.block_size;
    // Sliding-window cache hits must stay at the group's physical block
    // granularity. resolve_block_hashes() converts finer-grained hashes to
    // that view when the hybrid-cache alignment is smaller than block_size.
    let block_hashes = resolve_block_hashes(
        block_hashes,
        block_pool.hash_block_size(),
        block_size,
        ManagerType::SlidingWindow.supports_fine_grained_hash_lookup(),
        Some(alignment_tokens),
    );

    // The number of contiguous blocks needed for a prefix cache hit.
    let sliding_window_contiguous_blocks =
        contiguous_blocks_for_hit(sliding_window, block_size, drop_eagle_block);

    // TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
    // optimize the time complexity from O(max_num_blocks) to
    // O(max_num_blocks / sliding_window_contiguous_blocks +
    // sliding_window_contiguous_blocks),
    // which is good for low cache hit rate scenarios.
    let max_num_blocks = max_length / block_size;
    let mut computed_blocks = vec![vec![NULL_BLOCK_ID; max_num_blocks]; kv_cache_group_ids.len()];
    let mut num_contiguous_blocks = 0;
    let mut match_found = false;
    // Search from right to left and early stop when a match is found.
    for i in (0..max_num_blocks).rev() {
        if let Some(cached_block) = lookup(block_pool, block_hashes.get(i), kv_cache_group_ids) {
            // Skip prefix matching check if the block is not aligned with
            // `alignment_tokens`.
            if num_contiguous_blocks == 0 && block_size != alignment_tokens {
                let post_pop_blocks = if drop_eagle_block { i } else { i + 1 };
                if !(post_pop_blocks * block_size).is_multiple_of(alignment_tokens) {
                    continue;
                }
            }
            // Add the cached block to the computed blocks.
            for (computed, &cached) in computed_blocks.iter_mut().zip(&cached_block) {
                computed[i] = cached;
            }
            num_contiguous_blocks += 1;
            if num_contiguous_blocks >= sliding_window_contiguous_blocks {
                // Trim the trailing blocks.
                // E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                // when sliding_window_contiguous_blocks=2.
                for computed in &mut computed_blocks {
                    computed.truncate(i + num_contiguous_blocks);
                }
                match_found = true;
                break;
            }
        } else {
            num_contiguous_blocks = 0;
        }
    }
    let realign = |computed_blocks: &mut Vec<Vec<BlockId>>| {
        while block_size != alignment_tokens // Faster for common case.
            && !(computed_blocks[0].len() * block_size).is_multiple_of(alignment_tokens)
        {
            for computed in computed_blocks.iter_mut() {
                computed.pop();
            }
        }
    };
    if !match_found {
        // The first `num_contiguous_blocks` is a cache hit even if
        // `num_contiguous_blocks < sliding_window_contiguous_blocks`.
        for computed in &mut computed_blocks {
            computed.truncate(num_contiguous_blocks);
        }
        realign(&mut computed_blocks);
    }
    if drop_eagle_block && !computed_blocks[0].is_empty() {
        for computed in &mut computed_blocks {
            computed.pop();
        }
        // Re-align after eagle pop: the pop may break the alignment
        // when block_size != alignment_tokens (hybrid models with
        // different page sizes, e.g. Gemma4).
        realign(&mut computed_blocks);
    }
    let hit_length = computed_blocks[0].len() * block_size;
    (computed_blocks, hit_length)
}

/// For chunked local attention, we need to find the longest cache hit
/// prefix of the blocks that is not longer than `max_length`. The prefix
/// should be a common prefix hit for all the kv cache groups in
/// `kv_cache_group_ids`. If no cache hit is found, return an empty list.
/// note we mark as computed if the whole block is outside of the local
/// window, and set the block as null. Examples:
///
/// 1. Attention chunk size of 8, block size of 4, max length of 15
///    for next token at 15th (zero-indexed), 8th - 14th tokens are in
///    the window(needs lookup), 0th - 7th are not in the window,
///    so they are already marked as computed. We check the complete
///    block3 (8th - 11th tokens), Assume block 3 is hit, we will return
///    [null, null, block 3], otherwise, we return [null, null]
///
/// 2. Attention chunk size of 8, block size of 4, max length of 16
///    for next token at 16th (zero-indexed), 0th - 15th tokens are not
///    in the window, so they are already marked as computed.
///    we return 4 blocks[null, null, null, null]
fn chunked_local_hit(
    block_hashes: &[BlockHash],
    max_length: usize,
    kv_cache_group_ids: &[usize],
    block_pool: &impl CachedBlockLookup,
    kv_cache_spec: &KvCacheSpec,
    drop_eagle_block: bool,
    alignment_tokens: usize,
    dcp_world_size: usize,
    pcp_world_size: usize,
) -> CacheHit {
    let KvCacheSpecType::ChunkedLocalAttention {
        attention_chunk_size,
    } = kv_cache_spec.spec_type
    else {
        panic!("ChunkedLocalAttentionManager can only be used for chunked local attention groups");
    };
    assert!(
        !drop_eagle_block,
        "Hybrid KV cache is not supported for eagle + chunked local attention."
    );
    assert!(
        dcp_world_size == 1,
        "DCP not support chunked local attn now."
    );
    assert!(
        pcp_world_size == 1,
        "PCP not support chunked local attn now."
    );
    let block_size = kv_cache_spec.block_size;
    assert!(
        block_size == alignment_tokens,
        "KV cache groups with different block sizes are not compatible with chunked local attention now"
    );
    let block_hashes = resolve_block_hashes(
        block_hashes,
        block_pool.hash_block_size(),
        block_size,
        ManagerType::ChunkedLocalAttention.supports_fine_grained_hash_lookup(),
        Some(alignment_tokens),
    );
    let max_num_blocks = max_length / block_size;
    let local_attention_start_idx = if max_length > 0 {
        max_length / attention_chunk_size * attention_chunk_size
    } else {
        0
    };
    // we marked blocks out of window as computed
    // with null blocks, and blocks inside window based on cache lookup
    // result [null] [null] ... [null] [hit block 1 (1st block contain
    // last window)] [hit block 2] ... [hit block x]
    let local_attention_start_block_idx = local_attention_start_idx / block_size;
    let mut computed_blocks =
        vec![vec![NULL_BLOCK_ID; local_attention_start_block_idx]; kv_cache_group_ids.len()];
    for i in local_attention_start_block_idx..max_num_blocks {
        let Some(cached_block) = lookup(block_pool, block_hashes.get(i), kv_cache_group_ids) else {
            break;
        };
        push_hit(&mut computed_blocks, &cached_block);
    }
    let hit_length = computed_blocks[0].len() * block_size;
    (computed_blocks, hit_length)
}

fn mamba_hit(
    block_hashes: &[BlockHash],
    max_length: usize,
    kv_cache_group_ids: &[usize],
    block_pool: &impl CachedBlockLookup,
    kv_cache_spec: &KvCacheSpec,
    alignment_tokens: usize,
    dcp_world_size: usize,
    pcp_world_size: usize,
) -> CacheHit {
    assert!(
        kv_cache_spec.is_mamba(),
        "MambaManager can only be used for mamba groups"
    );
    assert!(dcp_world_size == 1, "DCP not support mamba now.");
    assert!(pcp_world_size == 1, "PCP not support mamba now.");
    let block_size = kv_cache_spec.block_size;
    let resolved = resolve_block_hashes(
        block_hashes,
        block_pool.hash_block_size(),
        block_size,
        ManagerType::Mamba.supports_fine_grained_hash_lookup(),
        Some(alignment_tokens),
    );
    let mut computed_blocks = vec![Vec::new(); kv_cache_group_ids.len()];
    let mut hit_length = 0;

    if alignment_tokens < block_size && block_size.is_multiple_of(alignment_tokens) {
        assert_eq!(resolved.scale_factor, 1);
        let hash_block_size = alignment_tokens;
        let scale_factor = block_size / hash_block_size;
        let max_num_partial_units = (max_length / hash_block_size).min(block_hashes.len());
        for fine_idx in (0..max_num_partial_units).rev() {
            let num_tokens = (fine_idx + 1) * hash_block_size;
            if let Some(cached_block) =
                lookup(block_pool, &block_hashes[fine_idx], kv_cache_group_ids)
            {
                let block_idx = fine_idx / scale_factor;
                for (computed, &cached) in computed_blocks.iter_mut().zip(&cached_block) {
                    computed.resize(block_idx, NULL_BLOCK_ID);
                    computed.push(cached);
                }
                hit_length = num_tokens;
                break;
            }
        }
        return (computed_blocks, hit_length);
    }

    let max_num_blocks = max_length / block_size;
    // Search from right to left and early stop when a match is found.
    for i in (0..max_num_blocks).rev() {
        if let Some(cached_block) = lookup(block_pool, resolved.get(i), kv_cache_group_ids) {
            // When enable Mamba prefix caching, `block_size` will be aligned
            // across full attention layers and Mamba layers to ensure the
            // prefix hit length aligned at block
            if block_size != alignment_tokens // Faster for common case.
                && !((i + 1) * block_size).is_multiple_of(alignment_tokens)
            {
                continue;
            }
            for (computed, &cached) in computed_blocks.iter_mut().zip(&cached_block) {
                // the hit length logic later assumes:
                //  hit_length = len(hit_blocks_other_attn[0])
                //               * self.other_block_size
                // so we insert dummy blocks at the beginning:
                computed.resize(i, NULL_BLOCK_ID);
                computed.push(cached);
            }
            hit_length = (i + 1) * block_size;
            break; // we just need the last match - early stopping
        }
    }
    (computed_blocks, hit_length)
}

fn sliding_window_mask(
    start_block: usize,
    end_block: usize,
    alignment_tokens: Option<usize>,
    kv_cache_spec: &KvCacheSpec,
    use_eagle: bool,
    retention_interval: Option<usize>,
    reachable_boundaries: &[i64],
    dcp_world_size: usize,
    final_segment_end_block: Option<usize>,
) -> Option<Vec<bool>> {
    let sliding_window = kv_cache_spec
        .sliding_window()
        .expect("SlidingWindowManager can only be used for sliding window groups");
    // Fast path: when the coordinator imposes no alignment constraint.
    let alignment_tokens = alignment_tokens?;
    let block_size = kv_cache_spec.block_size * dcp_world_size;
    if alignment_tokens % block_size != 0 {
        // The mask is block-granular, so a sub-block alignment cannot be
        // represented exactly. This happens for hybrid offloading, where
        // ``alignment_tokens`` is the full-attention chunk size and need not
        // be a multiple of this SWA group's (DCP-scaled) block size (e.g.
        // Gemma). Fall back to dense: every block is reachable, which never
        // drops a block that could serve a hit.
        return None;
    }

    // Contiguous blocks a hit needs at a boundary (incl. the EAGLE peek).
    let need = contiguous_blocks_for_hit(sliding_window, block_size, use_eagle);
    // The matched run's right edge sits on the aligned boundary block when
    // EAGLE peeks one block past it (shift=1), otherwise on the last block
    // before the boundary (shift=0).
    let shift = usize::from(use_eagle);

    let mut mask = vec![false; end_block.saturating_sub(start_block)];

    // (1) Segment-boundary tails. ``retention_interval``:
    //   None -> dense (a tail at every ``alignment_tokens`` boundary);
    //   0    -> no dense tails (only the replay boundary below);
    //   >0   -> a tail once per ``retention_interval``-sized segment.
    let segment_tokens = match retention_interval {
        None => Some(alignment_tokens),
        Some(0) => None,
        Some(interval) => Some(interval),
    };
    if let Some(segment_tokens) = segment_tokens {
        let per_segment = segment_tokens / block_size;
        if need >= per_segment {
            // Every block is reachable; cache them all.
            return None;
        }
        for i in start_block..end_block {
            if i >= shift && (i - shift) % per_segment >= per_segment - need {
                mask[i - start_block] = true;
            }
        }

        if let Some(final_segment_end_block) = final_segment_end_block
            && !use_eagle
        {
            let final_segment_size = final_segment_end_block % per_segment;
            if final_segment_size != 0 {
                let final_segment_start = final_segment_end_block - final_segment_size;
                let final_tail_start =
                    final_segment_start.max(final_segment_end_block.saturating_sub(need));
                let final_tail_end = end_block.min(final_segment_end_block);
                for i in start_block.max(final_tail_start)..final_tail_end {
                    mask[i - start_block] = true;
                }
            }
        }
    }

    // (2) Reachable-boundary tails: the replay boundary (``num_prompt - 1``,
    // capped by ``get_computed_blocks``) and any shared-prefix junction. Both
    // land before segments would cover them under sparse retention, so keep
    // the ``need``-block tail ending on each boundary explicitly.
    if retention_interval.is_some() {
        let (alignment, bs) = (alignment_tokens as i64, block_size as i64);
        for &boundary_tokens in reachable_boundaries {
            let aligned = floor_div(boundary_tokens, alignment) * alignment;
            let end = floor_div(aligned, bs) + shift as i64;
            let lo = (start_block as i64).max(end - need as i64);
            let hi = (end_block as i64).min(end);
            for j in lo..hi {
                mask[(j - start_block as i64) as usize] = true;
            }
        }
    }

    Some(mask)
}

/// Sparse Mamba state-snapshot retention.
///
/// `retention_interval`:
///
///   `None` -> dense (cache every block; default, unchanged behavior)
///   `0`    -> keep only the `reachable_boundaries` states
///   `> 0`  -> keep one state per `retention_interval`-sized segment
///
/// `reachable_boundaries` are proven reuse points (the replay boundary and
/// any cross-request shared-prefix junction, Marconi-style APC); their
/// boundary state is always kept so sparse retention does not defeat reuse.
fn mamba_mask(
    start_block: usize,
    end_block: usize,
    alignment_tokens: Option<usize>,
    kv_cache_spec: &KvCacheSpec,
    retention_interval: Option<usize>,
    reachable_boundaries: &[i64],
) -> Option<Vec<bool>> {
    // Dense caching (default) or no alignment constraint imposed.
    let retention_interval = retention_interval?;
    let alignment_tokens = alignment_tokens?;
    assert!(kv_cache_spec.is_mamba());
    let block_size = kv_cache_spec.block_size;
    let mut mask = vec![false; end_block.saturating_sub(start_block)];

    // (1) Segment-boundary states. A Mamba hit needs exactly the single
    // state block ending on the boundary (no window, and draft models have
    // no mamba layers, so no eagle shift). Block ``i`` ends at token
    // ``(i + 1) * block_size``.
    if retention_interval != 0 {
        let per_segment = retention_interval / block_size;
        if per_segment <= 1 {
            // Interval at/below the block size: every block is a boundary.
            return None;
        }
        let first_boundary = (start_block + per_segment) / per_segment * per_segment - 1;
        for i in ((first_boundary - start_block)..mask.len()).step_by(per_segment) {
            mask[i] = true;
        }
    }

    // (2) Reachable-boundary states: the replay boundary (``num_prompt - 1``,
    // capped by ``get_computed_blocks``) and any shared-prefix junction, both
    // of which segments would otherwise skip under sparse retention. A Mamba
    // hit needs exactly the single state block ending on the boundary.
    let (alignment, bs) = (alignment_tokens as i64, block_size as i64);
    for &boundary_tokens in reachable_boundaries {
        let aligned = floor_div(boundary_tokens, alignment) * alignment;
        let boundary_block = floor_div(aligned, bs) - 1;
        if start_block as i64 <= boundary_block && boundary_block < end_block as i64 {
            mask[(boundary_block - start_block as i64) as usize] = true;
        }
    }

    Some(mask)
}
