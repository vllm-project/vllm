# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.utils.gc_utils import gc_untrack
from vllm.v1.core.grant_table import BlockState, GrantError, GrantTable
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    generate_lookahead_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
    resolve_block_hashes,
    to_event_extra_keys,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """Cache of blocks that are used for prefix caching, from block hash to
    one or more cached blocks of the pool.

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: Entries hold block ids, not blocks: the cyclic GC does not track a
    dict of only bytes and ints, so cached blocks add nothing to each collection.
    """

    def __init__(self, blocks: list[KVCacheBlock]):
        self._blocks = blocks
        # Block id of one block per key.
        # Also untracked explicitly: Python 3.14+ no longer untracks dicts of
        # atomic values by itself.
        self._cache: dict[BlockHashWithGroupId, int] = gc_untrack({})
        # All block ids, in insertion order, of keys cached by several blocks.
        # Kept apart so that `_cache` holds no containers.
        self._dup_ids: dict[BlockHashWithGroupId, dict[int, None]] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """Gets any block with the given block hash key."""
        block_id = self._cache.get(key)
        return None if block_id is None else self._blocks[block_id]

    def contain(self, key: BlockHashWithGroupId, block_id: int) -> bool:
        """Checks whether the key maps to the given block ID."""
        dup_ids = self._dup_ids.get(key)
        if dup_ids is not None:
            return block_id in dup_ids
        return self._cache.get(key) == block_id

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """Inserts the KVCacheBlock to the cache"""
        block_id = self._cache.setdefault(key, block.block_id)
        if block_id != block.block_id:
            self._dup_ids.setdefault(key, {block_id: None})[block.block_id] = None

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """Pops block_id from the blocks cached under key, if it is there."""
        dup_ids = self._dup_ids.get(key)
        if dup_ids is None:
            if self._cache.get(key) != block_id:
                return None
            del self._cache[key]
        else:
            if dup_ids.pop(block_id, False) is False:
                return None
            self._cache[key] = next(iter(dup_ids))
            if len(dup_ids) == 1:
                del self._dup_ids[key]
        return self._blocks[block_id]

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._dup_ids.clear()


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
        num_hash_lookahead_tokens: Tokens after each block covered by its hash.
        medium: Storage medium reported in KV cache events.
        grant_table: The GrantTable governing this pool's block ownership. A
            default one (all groups at `hash_block_size`) is built if None.

    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
        num_hash_lookahead_tokens: int = 0,
        medium: str = MEDIUM_GPU,
        grant_table: GrantTable | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        if grant_table is None:
            grant_table = GrantTable(num_gpu_blocks, [hash_block_size], enable_caching)
        assert grant_table.num_blocks == num_gpu_blocks
        self.grant_table = grant_table
        self.num_gpu_blocks = num_gpu_blocks
        self.medium = medium
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        self.num_hash_lookahead_tokens = num_hash_lookahead_tokens
        self._weak_self = weakref.ref(self)
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx, pool=self._weak_self) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap(
            self.blocks
        )
        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector
        # Callbacks for blocks released with ``unpin_blocks`` whose contents
        # are still being read until the pool reuses them.
        self._reuse_watchers: dict[int, Callable[[KVCacheBlock], None]] = {}
        # The free queue holds FREE and CACHED_IDLE blocks only; count the
        # latter so allocation skips the eviction scan when there are none.
        self._num_cached_in_queue = 0
        # CACHED_IDLE blocks a step may still read (not drainable yet); they
        # count as free but `make_room` cannot evict them.
        self._undrainable_cached: set[int] = set()
        self._system_owners: dict[str, int] = {}
        # Percy M2 (T2, I3, I8): prefix-cache entries registered before their
        # slots were acked. They are published only once a processed step
        # confirms the slots (`flush_publications`), and dropped if the block
        # is released, revoked or re-granted first (the gen is checked).
        # block_id -> (request, gen, col, block_size, group)
        self._pending_full: dict[int, tuple[Request, int, int, int, int]] = {}
        # block_id -> (request, gen, num_tokens, group, block_size, replace)
        self._pending_partial: dict[int, tuple[Request, int, int, int, int, bool]] = {}
        self._pending_by_req: dict[str, set[int]] = {}
        # CoW destinations whose moved entries wait for their copy step:
        # dst block_id -> (hashes, num_tokens, valid, copy step seq, src, gen).
        self._pending_cow: dict[
            int, tuple[list[BlockHashWithGroupId], int | None, int, int, int, int]
        ] = {}

    def system_owner(self, name: str = "pool") -> int:
        """Owner id for blocks granted outside any request (connectors, host
        pools). One id per caller name, so `check_writes` can tell them apart
        from requests and from each other."""
        owner = self._system_owners.get(name)
        if owner is None:
            owner = self.grant_table.owner_for(f"\0percy-system:{name}")
            self._system_owners[name] = owner
        return owner

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.

        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
            block_mask: Optional mask aligned with
                `blocks[num_cached_blocks:num_full_blocks]`. When provided,
                blocks where the mask is False are skipped (treated like null
                blocks). Used by groups whose `find_longest_cache_hit` only
                consults a subset of blocks (e.g. SWA tail-window), so blocks
                that can never serve a hit stay out of the prefix-cache hash
                map.

        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert block_mask is None or len(block_mask) == len(new_full_blocks)
        self._cache_new_full_blocks(
            request,
            new_full_blocks,
            num_cached_blocks,
            block_size,
            kv_cache_group_id,
            block_mask,
            defer=True,
        )

    def _cache_new_full_blocks(
        self,
        request: Request,
        new_full_blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None,
        defer: bool,
    ) -> None:
        """Publish and hash `new_full_blocks`, the request's columns from
        `num_cached_blocks` on. With `defer`, an owned block whose slots are
        not confirmed yet is registered for publication on its ack (T2)."""
        num_full_blocks = num_cached_blocks + len(new_full_blocks)
        # T2 (I3): only blocks whose slots are all confirmed are published.
        published = self._publish_full_blocks(new_full_blocks, block_mask)
        if defer:
            for i, blk in enumerate(new_full_blocks):
                if (
                    not published[i]
                    and not blk.is_null
                    and (block_mask is None or block_mask[i])
                    and self.grant_table.state(blk.block_id) == BlockState.EXCLUSIVE
                ):
                    self._add_pending(
                        self._pending_full,
                        request,
                        blk,
                        (num_cached_blocks + i, block_size, kv_cache_group_id),
                    )
        block_hashes = resolve_block_hashes(
            request.block_hashes, self.hash_block_size, block_size
        )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        # Appended per block below: much cheaper than packing the group id each time.
        group_suffix = make_block_hash_with_group_id(BlockHash(b""), kv_cache_group_id)
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix caching.
            # We skip null blocks here.
            if not published[i]:
                continue
            block_hash = new_block_hashes[i]
            num_hash_tokens = (num_cached_blocks + i + 1) * block_size

            # Update and added the full block to the cache.
            block_hash_with_group_id = BlockHashWithGroupId(block_hash + group_suffix)
            if blk.block_hash is not None:
                # The only valid case where a "new full block" already has a
                # hash is partial->full promotion of the same cache block.
                assert (
                    blk.block_hash_num_tokens is not None
                    and blk.block_hash_num_tokens < num_hash_tokens
                )
                removed_hashes = self._remove_cached_block_hashes(blk)
                self._emit_block_removed_events(removed_hashes)
            self._insert_block_hash(
                block_hash_with_group_id,
                blk,
                num_tokens=num_hash_tokens,
            )
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null/masked-out blocks to match the length of new_hashes.
            extra_keys_list: list[tuple[Any, ...] | None] = []
            lookahead_token_ids: list[list[int]] | None = (
                [] if self.num_hash_lookahead_tokens else None
            )
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if not published[i - num_cached_blocks]:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = self.generate_event_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)
                if lookahead_token_ids is not None:
                    lookahead_token_ids.append(
                        self.get_lookahead_token_ids(request, block_end)
                    )

            self.kv_event_queue.append(
                self._build_block_stored_event(
                    request,
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    start_token_idx=start_token_idx,
                    end_token_idx=end_token_idx,
                    block_size=block_size,
                    kv_cache_group_id=kv_cache_group_id,
                    extra_keys_list=extra_keys_list,
                    lookahead_token_ids=lookahead_token_ids,
                )
            )

    def _publish_full_blocks(
        self, blocks: list[KVCacheBlock], block_mask: list[bool] | None
    ) -> list[bool]:
        """Publish the non-null, unmasked blocks the GrantTable accepts.

        Returns, per block, whether it was published (and may be hashed).
        """
        published = [False] * len(blocks)
        candidates = [
            i
            for i, blk in enumerate(blocks)
            if not blk.is_null and (block_mask is None or block_mask[i])
        ]
        if not candidates:
            return published
        ids = np.fromiter(
            (blocks[i].block_id for i in candidates),
            dtype=np.int64,
            count=len(candidates),
        )
        ok = self.grant_table.publishable_mask(ids)
        if ok.any():
            accepted = self.grant_table.publish(ids[ok])
            assert accepted
        for i, keep in zip(candidates, ok.tolist()):
            published[i] = keep
        return published

    def generate_event_extra_keys(
        self, request: Request, start: int, end: int, start_mm_idx: int
    ) -> tuple[tuple[Any, ...] | None, int]:
        # Event extra keys cover the same window as the block hash.
        if self.num_hash_lookahead_tokens:
            return generate_lookahead_block_hash_extra_keys(
                request, start, end, end + self.num_hash_lookahead_tokens, start_mm_idx
            )
        return generate_block_hash_extra_keys(request, start, end, start_mm_idx)

    def get_lookahead_token_ids(self, request: Request, block_end: int) -> list[int]:
        return request.all_token_ids[
            block_end : block_end + self.num_hash_lookahead_tokens
        ]

    def _build_block_stored_event(
        self,
        request: Request,
        block_hashes: list[ExternalBlockHash] | None,
        parent_block_hash: ExternalBlockHash | None,
        start_token_idx: int,
        end_token_idx: int,
        block_size: int,
        kv_cache_group_id: int,
        extra_keys_list: list[tuple[Any, ...] | None],
        lookahead_token_ids: list[list[int]] | None,
    ) -> BlockStored:
        """Build a `BlockStored` KV event for `request`.

        Shared by `cache_full_blocks` (newly cached blocks) and
        `emit_cached_block_events` (prefix-cache-reused blocks) so both emit
        identical event shapes for downstream consumers.
        """
        return BlockStored(
            block_hashes=block_hashes,
            parent_block_hash=parent_block_hash,
            token_ids=request.all_token_ids[start_token_idx:end_token_idx],
            block_size=block_size,
            lora_id=request.lora_request.adapter_id if request.lora_request else None,
            medium=self.medium,
            lora_name=request.lora_request.name if request.lora_request else None,
            extra_keys=to_event_extra_keys(extra_keys_list),
            group_idx=kv_cache_group_id,
            session_id=request.session_id,
            lookahead_token_ids=lookahead_token_ids,
        )

    def emit_cached_block_events(
        self,
        request: Request,
        num_cached_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Generate BlockStored events for blocks reused from prefix cache.

        Unlike cache_full_blocks(), this does NOT modify block state —
        the blocks are already cached. It only generates events so that
        external consumers (e.g. gateway) can learn about reused blocks.

        Args:
            request: The request whose prefix cache blocks were reused.
            num_cached_blocks: Number of blocks that were cache hits.
            block_size: Number of tokens per block.
            kv_cache_group_id: The KV cache group ID.

        """
        if not self.enable_kv_cache_events or num_cached_blocks == 0:
            return

        block_hashes = resolve_block_hashes(
            request.block_hashes, self.hash_block_size, block_size
        )

        # Collect external hashes and extra_keys for cached blocks.
        cached_hashes: list[ExternalBlockHash] = []
        extra_keys_list: list[tuple[Any, ...] | None] = []
        lookahead_token_ids: list[list[int]] | None = (
            [] if self.num_hash_lookahead_tokens else None
        )
        curr_mm_idx = 0
        for i in range(num_cached_blocks):
            block_start = i * block_size
            block_end = block_start + block_size
            cached_hashes.append(maybe_convert_block_hash(block_hashes[i]))
            extra_keys, curr_mm_idx = self.generate_event_extra_keys(
                request, block_start, block_end, curr_mm_idx
            )
            extra_keys_list.append(extra_keys)
            if lookahead_token_ids is not None:
                lookahead_token_ids.append(
                    self.get_lookahead_token_ids(request, block_end)
                )

        if not cached_hashes:
            return

        # Prefix-cache hits always form a contiguous prefix starting at block 0,
        # so the first (and thus the whole group's) parent block hash is None.
        parent_block_hash: ExternalBlockHash | None = None
        start_token_idx = 0
        end_token_idx = num_cached_blocks * block_size

        logger.debug(
            "EmitCachedBlock event: block_size=%d, "
            "num_cached_blocks=%d, parent_block_hash=%s, "
            "token_ids_len=%d, group_idx=%s",
            block_size,
            num_cached_blocks,
            parent_block_hash,
            len(request.all_token_ids[start_token_idx:end_token_idx]),
            kv_cache_group_id,
        )

        self.kv_event_queue.append(
            self._build_block_stored_event(
                request,
                block_hashes=cached_hashes,
                parent_block_hash=parent_block_hash,
                start_token_idx=start_token_idx,
                end_token_idx=end_token_idx,
                block_size=block_size,
                kv_cache_group_id=kv_cache_group_id,
                extra_keys_list=extra_keys_list,
                lookahead_token_ids=lookahead_token_ids,
            )
        )

    def cache_partial_block(
        self,
        request: Request,
        block: KVCacheBlock,
        num_tokens: int,
        kv_cache_group_id: int,
        block_size: int,
        replace_existing_hashes: bool = False,
    ) -> BlockHashWithGroupId | None:
        """Register a partial prefix-cache entry for an existing block.

        Prefix-cache keys normally identify full cache blocks. A partial entry
        makes an existing cache block reachable from a fine-grained prefix
        boundary inside that block without allocating or copying a new
        `KVCacheBlock`.

        The partial entry is lookup metadata owned by `block`. If `block`
        has no primary hash, the key becomes its primary hash. If the block
        already has a primary hash, the partial entry is tracked in
        `cached_block_hashes_by_block` so eviction, reset, and promotion can
        remove every hash key that points to the block.

        Args:
            request: Request whose token IDs and block hashes define the
                partial entry.
            block: Existing cache block to make reachable from the partial
                prefix boundary.
            num_tokens: Prefix length represented by the partial entry. It
                must be a positive multiple of `self.hash_block_size` and
                cannot exceed the request's computed block hashes.
            kv_cache_group_id: KV cache group that owns the partial entry.
            block_size: Cache block size for the owning group. The partial
                entry hash itself is always the prefix-chain hash at
                `num_tokens`; `block_size` must be a multiple of the hash
                block size.
            replace_existing_hashes: Whether the block contents were replaced
                and all existing cache entries must be removed before the new
                entry is registered.

        Returns:
            The hash key with group ID if a partial entry can be registered;
            otherwise `None` for null blocks.

        """
        if block.is_null:
            return None

        assert block_size % self.hash_block_size == 0
        block_hash = self._get_partial_block_hash(request, num_tokens)
        # A partial entry points at the block's first slots up to the
        # boundary; they must be confirmed (scoped I3, see `_seal_or_unhash`).
        if not self.grant_table.backs_partial_entry(
            block.block_id, (num_tokens - 1) % block_size + 1
        ):
            # Percy M2: register it for when a processed step acks the slots;
            # the returned key names the entry it will publish.
            if self.grant_table.state(block.block_id) != BlockState.EXCLUSIVE:
                return None
            self._add_pending(
                self._pending_partial,
                request,
                block,
                (
                    num_tokens,
                    kv_cache_group_id,
                    block_size,
                    replace_existing_hashes,
                ),
            )
            return make_block_hash_with_group_id(block_hash, kv_cache_group_id)
        num_hash_blocks = num_tokens // self.hash_block_size
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, kv_cache_group_id
        )
        already_cached = block.block_hash == block_hash_with_group_id or (
            self.cached_block_hash_to_block.contain(
                block_hash_with_group_id, block.block_id
            )
        )
        if replace_existing_hashes:
            removed_hashes = self._remove_cached_block_hashes(block)
            self._emit_block_removed_events(removed_hashes)
            already_cached = False
        elif (
            not already_cached
            and block.block_hash is not None
            and block.block_hash_num_tokens is not None
            and block.block_hash_num_tokens < num_hash_blocks * self.hash_block_size
        ):
            removed_hashes = self._remove_cached_block_hashes(block)
            self._emit_block_removed_events(removed_hashes)
        self._insert_block_hash(
            block_hash_with_group_id,
            block,
            num_tokens=num_hash_blocks * self.hash_block_size,
        )
        if self.enable_kv_cache_events and not already_cached:
            parent_hash, block_start = self._get_partial_block_parent_hash_and_start(
                request, num_tokens
            )
            parent_block_hash = (
                maybe_convert_block_hash(parent_hash)
                if parent_hash is not None
                else None
            )
            block_end = num_tokens
            curr_mm_idx = 0
            mm_features = request.mm_features
            if block_start > 0 and mm_features:
                last_mm_pos = mm_features[-1].mm_position
                if last_mm_pos.offset + last_mm_pos.length > block_start:
                    curr_mm_idx, _ = get_mm_features_in_window(
                        mm_features, block_start, block_end
                    )
            extra_keys, _ = self.generate_event_extra_keys(
                request, block_start, block_end, curr_mm_idx
            )
            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=[maybe_convert_block_hash(block_hash)],
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[block_start:block_end],
                    block_size=block_end - block_start,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=self.medium,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=to_event_extra_keys([extra_keys]),
                    group_idx=kv_cache_group_id,
                    session_id=request.session_id,
                    lookahead_token_ids=[
                        self.get_lookahead_token_ids(request, block_end)
                    ]
                    if self.num_hash_lookahead_tokens
                    else None,
                )
            )
        return block_hash_with_group_id

    def _get_partial_block_hash(
        self,
        request: Request,
        num_tokens: int,
    ) -> BlockHash:
        assert num_tokens % self.hash_block_size == 0
        num_hash_blocks = num_tokens // self.hash_block_size
        assert 0 < num_hash_blocks <= len(request.block_hashes)

        # Each hash_block_size hash chains over its full prefix, so the partial
        # entry for any group block size is the hash at that prefix boundary.
        return request.block_hashes[num_hash_blocks - 1]

    def _get_partial_block_parent_hash_and_start(
        self,
        request: Request,
        num_tokens: int,
    ) -> tuple[BlockHash | None, int]:
        num_hash_blocks = num_tokens // self.hash_block_size
        parent_hash = (
            request.block_hashes[num_hash_blocks - 2] if num_hash_blocks > 1 else None
        )
        block_start = (num_hash_blocks - 1) * self.hash_block_size
        return parent_hash, block_start

    def _remove_cached_block_hashes(
        self,
        block: KVCacheBlock,
    ) -> list[BlockHashWithGroupId]:
        # A block only has secondary hashes alongside a primary one.
        if block.block_hash is None:
            return []

        block_hashes = [block.block_hash]
        block_hashes.extend(self.cached_block_hashes_by_block.pop(block.block_id, ()))

        removed_hashes: list[BlockHashWithGroupId] = []
        for block_hash in block_hashes:
            if (
                self.cached_block_hash_to_block.pop(block_hash, block.block_id)
                is not None
            ):
                removed_hashes.append(block_hash)
        block.reset_hash()
        return removed_hashes

    def _emit_block_removed_events(
        self,
        block_hashes: list[BlockHashWithGroupId],
    ) -> None:
        if not self.enable_kv_cache_events:
            return
        for block_hash in block_hashes:
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=self.medium,
                    group_idx=get_group_id(block_hash),
                )
            )

    def _insert_block_hash(
        self,
        block_hash_with_group_id: BlockHashWithGroupId,
        block: KVCacheBlock,
        num_tokens: int | None,
    ) -> None:
        if block.block_hash is None:
            # A block without a hash has no entries in the cache yet.
            block.set_block_hash(block_hash_with_group_id, num_tokens=num_tokens)
        elif (
            block.block_hash == block_hash_with_group_id
            or self.cached_block_hash_to_block.contain(
                block_hash_with_group_id, block.block_id
            )
        ):
            return
        else:
            self.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(
                block_hash_with_group_id
            )
        self.cached_block_hash_to_block.insert(block_hash_with_group_id, block)

    def move_block_hashes(
        self,
        src_block: KVCacheBlock,
        dst_block: KVCacheBlock,
    ) -> None:
        """Re-point `src_block`'s prefix-cache entries to `dst_block`.

        Used when the request owning `src_block` keeps writing into it
        : the prefix cache holds a private copy (`dst_block`)
        under the same hashes instead. Entries stay live; no events emitted.
        """
        assert dst_block.block_hash is None
        assert dst_block.block_id not in self.cached_block_hashes_by_block
        src_id = src_block.block_id
        num_tokens = src_block.block_hash_num_tokens
        valid = self.grant_table.valid(src_id)
        hashes = self._remove_cached_block_hashes(src_block)
        # Entries registered on src but not acked yet follow the copy too:
        # src's slots are overwritten by the next forward.
        pending = self._pending_partial.pop(src_id, None)
        if pending is not None:
            request, gen, entry_tokens, group, _, _ = pending
            self._unlink_pending(request.request_id, src_id)
            if gen == self.grant_table.gen(src_id):
                key = make_block_hash_with_group_id(
                    self._get_partial_block_hash(request, entry_tokens), group
                )
                if key not in hashes:
                    hashes.append(key)
                if num_tokens is None or entry_tokens > num_tokens:
                    num_tokens = entry_tokens
        if not hashes:
            return
        # Percy M2: dst holds src's confirmed slots only once the copy has run,
        # in the step being scheduled. Pin dst and publish its entries when
        # that step is processed (`drain`); until then lookups miss.
        self.touch([dst_block], record_access=False)
        self._pending_cow[dst_block.block_id] = (
            hashes,
            num_tokens,
            valid,
            self.grant_table.latest_sched_seq + 1,
            src_id,
            self.grant_table.gen(src_id),
        )

    def make_room(self, num_blocks: int) -> bool:
        """Make the first `num_blocks` free-queue entries FREE (T8).

        The eviction half of allocation, kept apart from `get_new_blocks` so a
        cached block is never evicted and re-granted in one call. Walks the
        queue in eviction order, revokes cached-idle blocks no step or peer can
        still touch (skipping the others, which stay cached), drains them and
        moves them back to the head in the walk order. In the common case no
        block is skipped and the order matches eviction-and-reuse exactly.

        Args:
            num_blocks: The number of blocks the next `get_new_blocks` call
                will take.

        Returns:
            Whether `num_blocks` FREE blocks now lead the free queue.

        """
        if num_blocks <= 0:
            return True
        queue = self.free_block_queue
        if self._num_cached_in_queue == 0:
            return queue.num_free_blocks >= num_blocks
        chosen: list[KVCacheBlock] = []
        to_evict: list[KVCacheBlock] = []
        skipped = False
        for block in queue.iter_blocks_after(None):
            if len(chosen) == num_blocks:
                break
            if block.block_hash is None:
                chosen.append(block)
            elif self.grant_table.drainable_mask([block.block_id])[0]:
                chosen.append(block)
                to_evict.append(block)
            else:
                skipped = True
        if to_evict:
            for block in to_evict:
                if self.metrics_collector:
                    self.metrics_collector.on_block_evicted(block)
                self._emit_block_removed_events(self._remove_cached_block_hashes(block))
            self.grant_table.revoke([b.block_id for b in to_evict], 0)
            self._num_cached_in_queue -= len(to_evict)
            drained = self.grant_table.drain()
            assert set(b.block_id for b in to_evict).issubset(drained.tolist())
            if skipped:
                # Bring the drained blocks ahead of the cached ones skipped.
                for block in chosen:
                    queue.remove(block)
                self.free_block_queue.prepend_n(chosen)
            self._queue_drained(drained, [])
        return len(chosen) >= num_blocks

    def get_new_blocks(
        self,
        num_blocks: int,
        owner: int | None = None,
        group: int = 0,
        domain: int = 0,
    ) -> list[KVCacheBlock]:
        """Grant new blocks from the head of the free queue (T1).

        Never evicts: the caller runs `make_room(num_blocks)` first.

        Args:
            num_blocks: The number of blocks to allocate.
            owner: The GrantTable owner id the blocks are granted to; None
                uses this pool's system owner.
            group: The KV cache group the blocks belong to.
            domain: The sharing domain of the owner (I9).

        Returns:
            A list of new block.

        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
        if num_blocks == 0:
            return []
        if self._num_cached_in_queue:
            for i, block in enumerate(self.free_block_queue.iter_blocks_after(None)):
                if i == num_blocks:
                    break
                if block.block_hash is not None:
                    raise ValueError(
                        f"Cannot get {num_blocks} free blocks: a cached block "
                        "leads the free queue; call make_room() first"
                    )

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
        if owner is None:
            owner = self.system_owner()
        self.grant_table.grant([b.block_id for b in ret], owner, group, domain)

        if self._reuse_watchers:
            self._notify_reuse(ret)

        metrics_collector = self.metrics_collector
        for block in ret:
            assert block.ref_cnt == 0 and block.block_hash is None
            block.ref_cnt = 1
            if metrics_collector:
                if self.enable_caching:
                    metrics_collector.on_block_evicted(block)
                metrics_collector.on_block_allocated(block)
        return ret

    def _notify_reuse(self, blocks: Iterable[KVCacheBlock]) -> None:
        for block in blocks:
            watcher = self._reuse_watchers.pop(block.block_id, None)
            if watcher is not None:
                watcher(block)

    def unpin_blocks(
        self,
        blocks: Iterable[KVCacheBlock],
        on_reuse: Callable[[KVCacheBlock], None],
    ) -> None:
        """Release references to blocks that stay readable until reused.

        The blocks become last-resort eviction candidates regardless of prefix
        caching: they join the tail of the free queue and count as free.
        `on_reuse` fires when `get_new_blocks` hands a block out (or the
        cache is reset), so the caller can stop reading it.
        """
        blocks = list(blocks)
        for block in blocks:
            assert block.ref_cnt > 0 and not block.is_null
            self._reuse_watchers[block.block_id] = on_reuse
        self._release(blocks, None, prepend=False, prepend_cached=False)

    def deprioritize_blocks(self, blocks: Iterable[KVCacheBlock]) -> None:
        """Deprioritize free blocks so they are evicted after uncached blocks
        but before normal LRU-cached blocks, in FIFO order.

        Blocks still referenced (`ref_cnt > 0`) or not in the free queue are
        left unchanged.
        """
        moved: list[KVCacheBlock] = []
        for block in blocks:
            if block.ref_cnt == 0 and block.prev_free_block is not None:
                self.free_block_queue.remove(block)
                moved.append(block)
        self.free_block_queue.insert_before_cached(moved)

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        A cached-idle block is revoked (T8) and drained; a published block
        still held is force-revoked (T7) and drains once released. Partial
        entries on a block its owner still writes are lookup metadata only.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.

        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)
        # Entries still waiting for their ack are dropped with the block's.
        self._drop_pending([block.block_id])

        evicted_hashes = self._remove_cached_block_hashes(block)
        if not evicted_hashes:
            # The block doesn't have hash, eviction is not needed
            return False

        self._emit_block_removed_events(evicted_hashes)
        state = self.grant_table.state(block.block_id)
        if state == BlockState.CACHED_IDLE:
            self.grant_table.revoke([block.block_id], 0)
            self._num_cached_in_queue -= 1
            drained = self.grant_table.drain()
            if block.block_id not in set(drained.tolist()):
                self.free_block_queue.remove(block)
            self._queue_drained(drained, [])
        elif state == BlockState.SHARED:
            self.grant_table.revoke([block.block_id], 0)
        return True

    def touch(
        self,
        blocks: Sequence[KVCacheBlock],
        group: int | None = None,
        domain: int | None = None,
        record_access: bool = True,
    ) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        A prefix hit passes the hitter's `group` and `domain` (I9); an
        internal pin passes None. An internal pin of a FREE block (a released
        page re-adopted before reuse, e.g. HiSparse) re-grants it to this
        pool's system owner.

        Args:
            blocks: A list of blocks to touch.
            group: The KV cache group of the hit, or None for a pin.
            domain: The sharing domain of the hit, or None for a pin.
            record_access: Whether the touch counts as a metrics access
                (False for copy retentions).

        Raises:
            GrantError: The GrantTable refused the share; nothing changed.

        """
        blocks = [block for block in blocks if not block.is_null]
        if not blocks:
            return
        ids = np.fromiter(
            (block.block_id for block in blocks), dtype=np.int64, count=len(blocks)
        )
        states = self.grant_table.state_of(ids)
        free = states == BlockState.FREE
        if free.any():
            if group is not None or domain is not None:
                raise GrantError("touch: a prefix hit found a free block")
            free_ids, first = np.unique(ids[free], return_index=True)
            # Re-grant each FREE block once; later occurrences become shares.
            again = np.ones(int(free.sum()), dtype=bool)
            again[first] = False
            share_ids = np.concatenate([ids[~free], ids[free][again]])
            if share_ids.size and not self._can_share(share_ids, ids[~free]):
                raise GrantError("touch: the GrantTable refused the share")
            self.grant_table.grant(free_ids, self.system_owner("reacquire"), 0, 0)
            if share_ids.size:
                shared = self.grant_table.share(share_ids)
                assert shared
        elif not self.grant_table.share(ids, group, domain):
            raise GrantError("touch: the GrantTable refused the share")
        for block, state in zip(blocks, states.tolist()):
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and block.prev_free_block is not None:
                self.free_block_queue.remove(block)
                if state == BlockState.CACHED_IDLE:
                    self._num_cached_in_queue -= 1
            block.ref_cnt += 1
            if record_access and self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)
        self._assert_readers(blocks)

    def _can_share(self, share_ids: np.ndarray, held_ids: np.ndarray) -> bool:
        states = self.grant_table.state_of(held_ids)
        return bool(
            np.all(
                (states == BlockState.SHARED)
                | (states == BlockState.CACHED_IDLE)
                | (states == BlockState.EXCLUSIVE)
            )
        )

    def is_block_writable(self, block: KVCacheBlock) -> bool:
        """Return whether a block can be mutated by its sole owner: it holds
        the write grant (EXCLUSIVE), no other holder pins it, and no
        prefix-cache entry points at it."""
        return (
            not block.is_null
            and self.grant_table.is_writable(block.block_id)
            and block.ref_cnt == 1
            and block.block_hash is None
        )

    def free_blocks(
        self,
        ordered_blocks: Iterable[KVCacheBlock],
        evict_first: bool = False,
        step_seq: int | None = None,
    ) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Each free releases one holder in the GrantTable. A block whose last
        holder leaves either stays cached (published: CACHED_IDLE) or is
        revoked; revoked blocks reach the free queue only through `drain`.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
            evict_first: Also put freed cached blocks at the front of the free
                queue, so they are evicted before other cached blocks. For
                blocks whose contents another tier already holds.
            step_seq: The last step that may use the blocks through the
                releasing holder; defaults to the latest scheduled step.

        """
        other_pools: dict[BlockPool, list[KVCacheBlock]] = {}
        own: list[KVCacheBlock] = []
        for block in ordered_blocks:
            if block.is_null:  # never pooled; its ref_cnt is not maintained
                continue
            if block.pool is not None and block.pool is not self._weak_self:
                pool = block.pool()
                if pool is not None:
                    other_pools.setdefault(pool, []).append(block)
                continue
            own.append(block)
        self._release(own, step_seq, prepend=True, prepend_cached=evict_first)
        for pool, blocks in other_pools.items():
            pool.free_blocks(blocks, evict_first=evict_first, step_seq=step_seq)

    def _release(
        self,
        blocks: list[KVCacheBlock],
        step_seq: int | None,
        prepend: bool,
        prepend_cached: bool,
    ) -> None:
        """Release one holder per entry of `blocks` and queue the blocks
        that became FREE (drained) or CACHED_IDLE, in `blocks` order."""
        if not blocks:
            return
        if step_seq is None:
            step_seq = self.grant_table.latest_sched_seq
        released: list[KVCacheBlock] = []
        for block in blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                released.append(block)
        if self._pending_full or self._pending_partial:
            # The last holder is leaving: publish what its acks confirmed,
            # drop the rest (never published, so nothing can hit it).
            self._flush_pending_blocks([b.block_id for b in released])
            self._drop_pending([b.block_id for b in released])
        for block in released:
            if block.block_hash is not None:
                self._seal_or_unhash(block)
        self.grant_table.release([b.block_id for b in blocks], step_seq)
        self._assert_readers(blocks)

        states = self.grant_table.state_of([b.block_id for b in released]).tolist()
        stale = [
            b
            for b, st in zip(released, states)
            if st == BlockState.CACHED_IDLE and b.block_hash is None
        ]
        if stale:
            # Published, then its entries moved away: nothing can hit it.
            self.grant_table.revoke([b.block_id for b in stale], 0)
            states = self.grant_table.state_of([b.block_id for b in released]).tolist()
        drained = self.grant_table.drain()
        drained_set = set(drained.tolist())
        front: list[KVCacheBlock] = []
        back: list[KVCacheBlock] = []
        for block, state in zip(released, states):
            if state == BlockState.CACHED_IDLE:
                self._num_cached_in_queue += 1
                if not self.grant_table.drainable_mask([block.block_id])[0]:
                    self._undrainable_cached.add(block.block_id)
                (front if prepend_cached else back).append(block)
            elif block.block_id in drained_set:
                (front if prepend else back).append(block)
        self._queue_drained(drained, released, front, back, prepend)

    def _queue_drained(
        self,
        drained: np.ndarray,
        handled: list[KVCacheBlock],
        front: list[KVCacheBlock] | None = None,
        back: list[KVCacheBlock] | None = None,
        prepend: bool = True,
    ) -> None:
        """Put blocks into the free queue: `front` at the head and `back` at
        the tail, in order, plus any other drained block not yet queued."""
        front = front if front is not None else []
        back = back if back is not None else []
        if drained.size:
            handled_ids = {b.block_id for b in handled}
            for block_id in drained.tolist():
                block = self.blocks[block_id]
                if block_id not in handled_ids and block.prev_free_block is None:
                    (front if prepend else back).append(block)
        # Blocks to reuse first are prepended to the front of the free queue.
        self.free_block_queue.prepend_n(front)
        # Blocks to reuse last are appended to the end of the free queue.
        self.free_block_queue.append_n(back)

    def _add_pending(
        self,
        pending: dict,
        request: Request,
        block: KVCacheBlock,
        info: tuple,
    ) -> None:
        block_id = block.block_id
        pending[block_id] = (request, self.grant_table.gen(block_id), *info)
        self._pending_by_req.setdefault(request.request_id, set()).add(block_id)

    def _drop_pending(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            for pending in (self._pending_full, self._pending_partial):
                entry = pending.pop(block_id, None)
                if entry is not None:
                    self._unlink_pending(entry[0].request_id, block_id)

    def _unlink_pending(self, request_id: str, block_id: int) -> None:
        ids = self._pending_by_req.get(request_id)
        if ids is None:
            return
        if block_id not in self._pending_full and block_id not in (
            self._pending_partial
        ):
            ids.discard(block_id)
        if not ids:
            del self._pending_by_req[request_id]

    def is_publication_pending(self, block_id: int) -> bool:
        """Whether the block has an entry waiting for its slots' ack."""
        return (
            block_id in self._pending_full
            or block_id in self._pending_partial
            or block_id in self._pending_cow
        )

    def num_pending_publications(self) -> int:
        """Prefix-cache entries waiting for their slots' acks."""
        return (
            len(self._pending_full)
            + len(self._pending_partial)
            + len(self._pending_cow)
        )

    def flush_publications(self, request_id: str) -> None:
        """Percy M2: publish the request's pending entries whose slots a
        processed step has now confirmed (T2 on ack)."""
        block_ids = self._pending_by_req.get(request_id)
        if block_ids:
            self._flush_pending_blocks(list(block_ids))

    def _flush_pending_blocks(self, block_ids: Sequence[int]) -> None:
        grant_table = self.grant_table
        for block_id in sorted(block_ids):
            entry = self._pending_partial.get(block_id)
            if entry is None:
                continue
            request, gen, num_tokens, group, block_size, replace = entry
            if grant_table.gen(block_id) != gen:
                self._pending_partial.pop(block_id)
                self._unlink_pending(request.request_id, block_id)
                continue
            if not grant_table.backs_partial_entry(
                block_id, (num_tokens - 1) % block_size + 1
            ):
                continue
            self._pending_partial.pop(block_id)
            self._unlink_pending(request.request_id, block_id)
            self.cache_partial_block(
                request=request,
                block=self.blocks[block_id],
                num_tokens=num_tokens,
                kv_cache_group_id=group,
                block_size=block_size,
                replace_existing_hashes=replace,
            )
        ready: list[tuple[int, int, int]] = []
        for block_id in block_ids:
            entry = self._pending_full.get(block_id)
            if entry is None:
                continue
            request, gen, col, block_size, group = entry
            if grant_table.gen(block_id) != gen:
                self._pending_full.pop(block_id)
                self._unlink_pending(request.request_id, block_id)
                continue
            if grant_table.publishable_mask([block_id])[0]:
                ready.append((group, col, block_id))
        # Parents before children, so events arrive in prefix order.
        for group, col, block_id in sorted(ready):
            request, _, _, block_size, _ = self._pending_full.pop(block_id)
            self._unlink_pending(request.request_id, block_id)
            block = self.blocks[block_id]
            if block.block_hash is not None and (
                block.block_hash_num_tokens is None
                or block.block_hash_num_tokens >= (col + 1) * block_size
            ):
                continue
            self._cache_new_full_blocks(
                request, [block], col, block_size, group, None, defer=False
            )

    def _flush_pending_cow(
        self, processed_seq: int
    ) -> list[list[BlockHashWithGroupId]]:
        """Publish CoW destinations whose copy step has been processed, then
        drop the pin taken in `move_block_hashes`. Returns the entries made
        visible."""
        published: list[list[BlockHashWithGroupId]] = []
        done = [
            block_id
            for block_id, entry in self._pending_cow.items()
            if entry[3] <= processed_seq
        ]
        for block_id in done:
            hashes, num_tokens, valid, copy_seq, src_id, src_gen = (
                self._pending_cow.pop(block_id)
            )
            block = self.blocks[block_id]
            state = self.grant_table.state(block_id)
            if state == BlockState.EXCLUSIVE and block.block_hash is None:
                # The copy wrote src's slots as of the steps before it, which
                # are processed too: that is dst's ack.
                if self.grant_table.gen(src_id) == src_gen:
                    valid = max(valid, self.grant_table.valid(src_id))
                self.grant_table.confirm_valid([block_id], valid)
                if num_tokens is None or self.grant_table.backs_partial_entry(
                    block_id,
                    (num_tokens - 1)
                    % self.grant_table.block_size_of(self.grant_table.group(block_id))
                    + 1,
                ):
                    for block_hash in hashes:
                        self._insert_block_hash(block_hash, block, num_tokens)
                    published.append(hashes)
                    if self.grant_table.publishable_mask([block_id])[0]:
                        assert self.grant_table.publish([block_id])
                else:
                    self._emit_block_removed_events(hashes)
            else:
                self._emit_block_removed_events(hashes)
            self._release([block], copy_seq, prepend=True, prepend_cached=False)
        return published

    def drain(self, processed_seq: int | None = None) -> int:
        """T9: return DRAINING blocks no step or peer can still touch to the
        free queue (the only path that adds freed blocks to it).

        Args:
            processed_seq: The latest step whose output has been processed;
                None keeps the GrantTable's current value.

        Returns:
            The number of blocks drained.

        """
        if self._pending_cow and processed_seq is not None:
            self._flush_pending_cow(processed_seq)
        drained = self.grant_table.drain(processed_seq)
        self._queue_drained(drained, [])
        return int(drained.size)

    def _seal_or_unhash(self, block: KVCacheBlock) -> None:
        """The last holder of an owned block with prefix-cache entries is
        leaving. Freeze it (scoped I3: every entry points at confirmed slots,
        and no writer remains) or, if its slots are not confirmed, drop the
        entries so the block can be revoked."""
        if self.grant_table.state(block.block_id) != BlockState.EXCLUSIVE:
            return
        size = self.grant_table.block_size_of(self.grant_table.group(block.block_id))
        num_tokens = block.block_hash_num_tokens
        num_slots = size if num_tokens is None else (num_tokens - 1) % size + 1
        if not self.grant_table.publish_partial([block.block_id], num_slots):
            self._emit_block_removed_events(self._remove_cached_block_hashes(block))

    def _assert_readers(self, blocks: Sequence[KVCacheBlock]) -> None:
        """ref_cnt stays as the LRU counter; it must equal GrantTable readers."""
        readers = self.grant_table.readers_of([b.block_id for b in blocks])
        assert all(b.ref_cnt == r for b, r in zip(blocks, readers.tolist())), (
            "KVCacheBlock.ref_cnt diverged from GrantTable readers"
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """Evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.

        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.

        """
        # Blocks still DRAINING (released, waiting for an in-flight step) hold
        # no prefix-cache entry and have no holder: they do not block a reset.
        num_held_blocks = self.grant_table.num_held
        if num_held_blocks:
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_held_blocks,
            )
            return False

        # No block is held, so no entry can still be waiting for an ack.
        assert not self._pending_cow
        self._pending_full.clear()
        self._pending_partial.clear()
        self._pending_by_req.clear()

        # Remove all hashes so that no new blocks will hit. Clear in place: a
        # new map created after the startup gc.freeze() would not be frozen.
        self.cached_block_hash_to_block.clear()
        self.cached_block_hashes_by_block.clear()
        if self._reuse_watchers:
            self._notify_reuse(
                [self.blocks[block_id] for block_id in list(self._reuse_watchers)]
            )

        # Every cached block is idle in the free queue: revoke them (T8) in
        # place; any a step may still read leaves the queue until drained.
        cached = [block for block in self.blocks if block.block_hash is not None]
        if cached:
            self.grant_table.revoke([b.block_id for b in cached], 0)
            self._num_cached_in_queue = 0
            drained = set(self.grant_table.drain().tolist())
            for block in cached:
                if block.block_id not in drained:
                    self.free_block_queue.remove(block)

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()
        self.free_block_queue.reset_cached_head()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.

        """
        return self.free_block_queue.num_free_blocks

    def get_num_allocatable_blocks(self) -> int:
        """Free blocks a `make_room` + `get_new_blocks` pair can hand out now:
        the free queue minus cached-idle blocks an in-flight step may still
        read (they cannot be evicted until drainable).

        Returns:
            The number of blocks allocatable in this scheduling step.

        """
        if self._undrainable_cached:
            ids = np.fromiter(self._undrainable_cached, dtype=np.int64)
            still = (
                self.grant_table.state_of(ids) == BlockState.CACHED_IDLE
            ) & ~self.grant_table.drainable_mask(ids)
            self._undrainable_cached = set(ids[still].tolist())
        return self.free_block_queue.num_free_blocks - len(self._undrainable_cached)

    def get_num_uncached_free_blocks(self, limit: int) -> int:
        """Count, up to `limit`, the leading uncached free blocks that
        `get_new_blocks` hands out before any cached prefix block."""
        count = 0
        for block in self.free_block_queue.iter_blocks_after(None):
            if count >= limit or block.block_hash is not None:
                break
            count += 1
        return count

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).

        """
        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.

        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
