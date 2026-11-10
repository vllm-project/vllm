// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `BlockHashListWithBlockSize`, `resolve_block_hashes` and the hash-granularity
//! helpers of `kv_cache_utils.py`.

use super::single_type::ManagerType;
use super::spec::{KvCacheGroupSpec, KvCacheSpec, MambaCacheMode};
use crate::hash::BlockHash;

/// Convert block-hash granularity from `hash_block_size` to `target_block_size`.
/// Used when KV cache groups have different block sizes: `hash_block_size`
/// is the size used to compute the original `block_hashes`; `target_block_size`
/// is the group's actual block size.
///
/// Currently, only scaling up by an integer factor is supported (i.e.,
/// `target_block_size` is a multiple of `hash_block_size`). Conversion is
/// performed lazily on access for efficiency. Each `hash_block_size` hash is
/// already chained over its entire prefix, so the hash at the last
/// `hash_block_size` boundary of a `target_block_size` block uniquely
/// fingerprints that block's prefix; we use it directly.
///
/// Example (`hash_block_size` = 16, `target_block_size` = 32):
/// the second 16-size hash already covers tokens 0-31, so it is the 32-size
/// hash:
///
/// ```text
/// Block hashes with block_size 16:
/// | Token Range | 0-15 | 16-31 | 32-47 | 48-63 |
/// |-------------|------|-------|-------|-------|
/// | Hash        | A    | B     | C     | D     |
///
/// Block hashes with block_size 32:
/// | Token Range | 0-31 | 32-63 |
/// |-------------|------|-------|
/// | Hash        | B    | D     |
/// ```
///
/// A scale factor of 1 is the raw list itself, so this type also stands in
/// for Python's `BlockHashList` union.
#[derive(Clone, Copy, Debug)]
pub struct BlockHashListWithBlockSize<'a> {
    pub block_hashes: &'a [BlockHash],
    pub scale_factor: usize,
}

impl<'a> BlockHashListWithBlockSize<'a> {
    pub fn new(
        block_hashes: &'a [BlockHash],
        hash_block_size: usize,
        target_block_size: usize,
    ) -> Self {
        assert!(target_block_size.is_multiple_of(hash_block_size));
        Self {
            block_hashes,
            scale_factor: target_block_size / hash_block_size,
        }
    }

    /// The raw list viewed at its own granularity.
    pub fn raw(block_hashes: &'a [BlockHash]) -> Self {
        Self {
            block_hashes,
            scale_factor: 1,
        }
    }

    pub fn len(&self) -> usize {
        self.block_hashes.len() / self.scale_factor
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `self[idx]`. Panics where Python raises `IndexError`.
    pub fn get(&self, idx: usize) -> &'a BlockHash {
        // The last hash_block_size hash within the target block already chains
        // over the whole prefix, so it is the target block's hash.
        &self.block_hashes[(idx + 1) * self.scale_factor - 1]
    }

    pub fn iter(&self) -> impl Iterator<Item = &'a BlockHash> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }
}

/// Resolve the block-hash view at `block_size`.
///
/// When `block_size` equals `hash_block_size`, reuse the precomputed block
/// hashes directly; otherwise view them at `block_size` granularity.
/// Fine-grained lookup keeps the original hashes for partial cache hits.
///
/// Only raw lists are accepted: every Python caller passes `request.block_hashes`
/// (the "already a block-size view" branch is unreachable from the managers).
pub fn resolve_block_hashes(
    block_hashes: &[BlockHash],
    hash_block_size: usize,
    block_size: usize,
    supports_fine_grained_hash_lookup: bool,
    alignment_tokens: Option<usize>,
) -> BlockHashListWithBlockSize<'_> {
    if block_size == hash_block_size {
        return BlockHashListWithBlockSize::raw(block_hashes);
    }
    // Fine-grained partial hits keep the raw hashes. The caller passes
    // alignment_tokens = hash_block_size to enable them, else >= block_size.
    if supports_fine_grained_hash_lookup
        && let Some(alignment_tokens) = alignment_tokens
        && alignment_tokens < block_size
        && block_size.is_multiple_of(alignment_tokens)
    {
        return BlockHashListWithBlockSize::raw(block_hashes);
    }
    assert!(block_size.is_multiple_of(hash_block_size));
    BlockHashListWithBlockSize::new(block_hashes, hash_block_size, block_size)
}

/// Return the token span of a cache block under DCP.
pub(crate) fn resolve_dcp_kv_block_size(spec: &KvCacheSpec, dcp_world_size: usize) -> usize {
    spec.block_size * if spec.dcp_sharded { dcp_world_size } else { 1 }
}

/// Whether aligned Mamba states support sub-block prefix-cache hits.
///
/// If `manager_types` (one per group) is given, also require every other
/// prefix-cacheable group to support fine-grained lookups.
pub fn partial_hash_hits_enabled(
    kv_cache_groups: &[KvCacheGroupSpec],
    hash_block_size: usize,
    dcp_world_size: usize,
    manager_types: Option<&[ManagerType]>,
) -> bool {
    if !kv_cache_groups.iter().any(|group| {
        let spec = &group.kv_cache_spec;
        spec.mamba_cache_mode() == Some(MambaCacheMode::Align)
            && ((dcp_world_size == 1 && spec.block_size > hash_block_size)
                || (dcp_world_size > 1 && spec.block_size >= hash_block_size))
    }) {
        return false;
    }
    let Some(manager_types) = manager_types else {
        return true;
    };
    // TODO: Python logs a warning naming the unsupported managers.
    let has_unsupported = kv_cache_groups.iter().zip(manager_types).any(|(group, manager_type)| {
        group.kv_cache_spec.prefix_cacheable()
            && !manager_type.supports_fine_grained_hash_lookup()
            && resolve_dcp_kv_block_size(&group.kv_cache_spec, dcp_world_size) != hash_block_size
    });
    !has_unsupported
}

/// Tokens an EAGLE group matches past a cache hit before dropping them.
///
/// Fine-grained lookups drop one hash unit; others drop one cache block.
pub fn eagle_proof_margin(
    block_size: usize,
    hash_block_size: usize,
    fine_grained_lookup: bool,
) -> usize {
    if fine_grained_lookup && block_size > hash_block_size {
        return hash_block_size;
    }
    block_size
}
