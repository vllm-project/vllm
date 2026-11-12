// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Port of the vLLM v1 KV cache management stack (`vllm/v1/core/`):
//! `kv_cache_utils.py` (blocks, free queue, hash views), `block_pool.py`,
//! `single_type_kv_cache_manager.py`, `kv_cache_coordinator.py` and
//! `kv_cache_manager.py`.
//!
//! The Python implementation is the behavioral reference: for the same inputs
//! this port allocates the same block ids, evicts in the same order, reports
//! the same cache hits and emits the same KV cache events. Deliberate
//! deviations are documented where they occur.
//!
//! Data layout differs from Python where it buys performance without changing
//! behavior: blocks live in an arena owned by [`BlockPool`] and are referred to
//! by [`BlockId`] (`u32`), the free queue is an index-based intrusive list, and
//! the prefix-cache map is keyed by `(BlockHash, group_id)` instead of packed
//! bytes. The null block is block 0, as in Python.
//!
//! Single-type managers do not hold a reference to the block pool; the
//! coordinator owns both and passes the pool to every manager call.

// The methods mirror the Python signatures; the workspace allows this lint too.
#![allow(clippy::too_many_arguments)]

mod block;
mod block_hash_list;
mod block_pool;
mod coordinator;
mod error;
mod events;
mod free_queue;
mod hisparse;
mod manager;
mod metrics;
mod request;
mod single_type;
mod spec;

#[cfg(test)]
mod tests;

pub use self::block::{
    BlockHashWithGroupId, BlockId, HOST_BLOCK_FLAG, KvCacheBlock, KvCacheBlockCopy, NULL_BLOCK_ID,
    null_head_len, trim_null_head,
};
pub use self::block_hash_list::{
    BlockHashListWithBlockSize, eagle_proof_margin, partial_hash_hits_enabled, resolve_block_hashes,
};
pub use self::block_pool::{BlockHashToBlockMap, BlockPool, CachedBlockLookup, CachedBlocks};
pub use self::coordinator::{CoordinatorKind, KvCacheCoordinator, SpecGroup, replay_boundaries};
pub use self::error::KvCacheError;
pub use self::events::{
    AllBlocksCleared, BlockRemoved, BlockStored, EventExtraKey, EventExtraKeys, ExternalBlockHash,
    KvCacheEvent, MEDIUM_CPU, MEDIUM_GPU, MEDIUM_STORAGE,
};
pub use self::free_queue::FreeKvCacheBlockQueue;
pub use self::hisparse::{
    ACTIVE_TAIL_PAGES, HiSparseCoordinator, HiSparseHandle, RequestState, SparseKvPageTransfer,
    SparseKvResidencyUpdate, SparseKvRowMirror,
};
pub use self::manager::{
    AllocateSlotsArgs, BoundaryStateOffload, KvCacheBlocks, KvCacheManager, KvCacheManagerConfig,
};
pub use self::metrics::{KvCacheEvictionEvent, KvCacheMetricsCollector, PrefixCacheStats};
pub use self::request::KvRequest;
pub use self::single_type::{
    CacheHit, ManagerType, PendingBoundaryStateOffload, SingleTypeKvCacheManager,
};
pub use self::spec::{
    KvCacheConfig, KvCacheGroupRole, KvCacheGroupSpec, KvCacheSpec, KvCacheSpecType,
    MambaCacheMode, get_mamba_prefill_checkpoint_position, is_mamba_prefill_checkpoint_valid,
};

/// `vllm.utils.math_utils.cdiv` for non-negative operands.
#[inline]
pub(crate) fn cdiv(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}
