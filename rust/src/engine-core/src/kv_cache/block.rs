// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `KVCacheBlock` and the block-hash key helpers of `kv_cache_utils.py`.

use crate::hash::BlockHash;

/// Index of a block in its [`BlockPool`](super::BlockPool), ranging from 0 to
/// `num_gpu_blocks - 1`.
pub type BlockId = u32;

/// The placeholder block: the first block popped from a fresh pool.
pub const NULL_BLOCK_ID: BlockId = 0;

/// Marks a HiSparse host-pool block id in block lists that mix the device and
/// host pools: those [`KvCacheManager::pop_blocks_for_free`] returns, to be
/// handed back to [`KvCacheManager::free_blocks`]. Python blocks carry their
/// pool instead (`KVCacheBlock.pool`). Block tables never carry the flag: a
/// group's blocks all come from one pool.
///
/// [`KvCacheManager::pop_blocks_for_free`]: super::KvCacheManager::pop_blocks_for_free
/// [`KvCacheManager::free_blocks`]: super::KvCacheManager::free_blocks
pub const HOST_BLOCK_FLAG: BlockId = 1 << 31;

/// `BlockHashWithGroupId` combines a `BlockHash` with its KV cache group ID.
///
/// Python packs the group id as 4 big-endian bytes appended to the block hash
/// bytes; this port keeps the two components apart (see
/// [`Self::to_packed_bytes`] for the Python representation), which avoids
/// building a byte string per block.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BlockHashWithGroupId {
    pub block_hash: BlockHash,
    pub group_id: u32,
}

impl BlockHashWithGroupId {
    /// `make_block_hash_with_group_id`.
    pub fn new(block_hash: BlockHash, group_id: usize) -> Self {
        Self {
            block_hash,
            group_id: u32::try_from(group_id).expect("group id must fit in 4 bytes"),
        }
    }

    /// The Python `BlockHashWithGroupId` bytes: the block hash followed by the
    /// group id encoded using 4 bytes in big-endian order.
    pub fn to_packed_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.block_hash.0.len() + 4);
        out.extend_from_slice(self.block_hash.as_bytes());
        out.extend_from_slice(&self.group_id.to_be_bytes());
        out
    }
}

impl std::fmt::Debug for BlockHashWithGroupId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}@{}", self.block_hash, self.group_id)
    }
}

/// KV-cache block metadata.
///
/// The free-list links (`prev_free_block` / `next_free_block` in Python) live
/// in [`FreeKvCacheBlockQueue`](super::FreeKvCacheBlockQueue). Python's
/// `pool` back-reference, which routes frees of HiSparse host blocks to their
/// own pool, is [`HOST_BLOCK_FLAG`] here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCacheBlock {
    /// Block ID, ranging from 0 to num_gpu_blocks - 1.
    pub block_id: BlockId,
    /// Reference count. Signed like the Python `int`, which an unbalanced free
    /// may drive negative without an error.
    pub ref_cnt: i32,
    /// The hash key (block hash + group id) of the block, only available
    /// when the block is full and cached.
    pub block_hash: Option<BlockHashWithGroupId>,
    /// Number of prefix tokens covered by block_hash. For full blocks this is
    /// the full block boundary; partial entries can end inside a cache block.
    pub block_hash_num_tokens: Option<usize>,
    /// Whether the block is a null block that should never be cached.
    pub is_null: bool,
}

impl KvCacheBlock {
    pub fn new(block_id: BlockId) -> Self {
        Self {
            block_id,
            ref_cnt: 0,
            block_hash: None,
            block_hash_num_tokens: None,
            is_null: false,
        }
    }

    pub fn set_block_hash(&mut self, block_hash: BlockHashWithGroupId, num_tokens: Option<usize>) {
        assert!(
            self.block_hash.is_none() && self.block_hash_num_tokens.is_none(),
            "The block already has a hash. This should not happen."
        );
        self.block_hash = Some(block_hash);
        self.block_hash_num_tokens = num_tokens;
    }

    /// Reset the block hash when the block is evicted.
    pub fn reset_hash(&mut self) {
        self.block_hash = None;
        self.block_hash_num_tokens = None;
    }
}

/// Number of leading null blocks, e.g. a sliding-window table's skipped prefix.
///
/// Block tables reference the null block only by [`NULL_BLOCK_ID`], so this
/// counts the leading run of that id.
pub fn null_head_len(blocks: &[BlockId]) -> usize {
    blocks.iter().take_while(|&&b| b == NULL_BLOCK_ID).count()
}

/// `blocks` without its leading null blocks; `blocks` itself if none.
pub fn trim_null_head(blocks: &[BlockId]) -> &[BlockId] {
    &blocks[null_head_len(blocks)..]
}

/// A pending block copy (`KVCacheBlockCopy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCacheBlockCopy {
    pub src_block_id: BlockId,
    pub dst_block_id: BlockId,
}
