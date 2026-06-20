# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    LargeBlockMeta,
    generate_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


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
        large_block_factor: Number `N` of small (attention) blocks that one
            large (mamba) block spans. ``N == 1`` (default) means flat / legacy
            behaviour. ``N > 1`` enables hierarchical allocation: small blocks
            are dispensed from inside parent large blocks, and large blocks
            are managed by their own free-queue.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
        large_block_factor: int = 1,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        assert isinstance(large_block_factor, int) and large_block_factor >= 1
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        self.large_block_factor = large_block_factor
        # All kv-cache blocks (small / attention granularity).
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        if large_block_factor == 1:
            # Flat / legacy mode: one global free queue over all blocks. No
            # large-block hierarchy.
            self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
            self.large_block_metas: list[LargeBlockMeta] = []
            self.free_large_block_queue: FreeKVCacheBlockQueue | None = None

            # To represent a placeholder block with block_id=0.
            # The ref_cnt of null_block is not maintained, needs special care to
            # avoid freeing it.
            self.null_block = self.free_block_queue.popleft()
            self.null_block.is_null = True
        else:
            # Hierarchical mode: small blocks live inside parent large blocks
            # and are dispensed via per-meta cursors. The flat ``free_block_queue``
            # is created empty and unused; ``free_large_block_queue`` owns
            # whole large blocks.
            num_large_blocks = num_gpu_blocks // large_block_factor
            assert num_large_blocks >= 2, (
                "Hierarchical block pool requires at least 2 large blocks "
                "(one is reserved for the null block)."
            )
            self.large_block_metas = [
                LargeBlockMeta(
                    large_block=KVCacheBlock(L),
                    small_blocks=self.blocks[
                        L * large_block_factor : (L + 1) * large_block_factor
                    ],
                )
                for L in range(num_large_blocks)
            ]
            # Reserve large block 0 to host the null small block; never enters
            # any free queue.
            self.large_block_metas[0].num_small_in_use = large_block_factor
            self.large_block_metas[0].next_small_idx = large_block_factor
            self.null_block = self.large_block_metas[0].small_blocks[0]
            self.null_block.is_null = True
            # Truncated free queue carrying only large blocks for ids >= 1.
            self.free_block_queue = FreeKVCacheBlockQueue([])
            self.free_large_block_queue = FreeKVCacheBlockQueue(
                [meta.large_block for meta in self.large_block_metas[1:]]
            )

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

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
                ``blocks[num_cached_blocks:num_full_blocks]``. When provided,
                blocks where the mask is False are skipped (treated like null
                blocks). Used by groups whose ``find_longest_cache_hit`` only
                consults a subset of blocks (e.g. SWA tail-window), so blocks
                that can never serve a hit stay out of the prefix-cache hash
                map.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        assert block_mask is None or len(block_mask) == len(new_full_blocks)
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix-caching
            # in align mode. We skip null blocks here.
            if blk.is_null or (block_mask is not None and not block_mask[i]):
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
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
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                if block_mask is not None and not block_mask[i - num_cached_blocks]:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[start_token_idx:end_token_idx],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=extra_keys_list if extra_keys_list else None,
                    group_idx=kv_cache_group_id,
                )
            )

    def get_new_blocks(
        self,
        num_blocks: int,
        large_block: bool = False,
        last_hit_block_id: int | None = None,
    ) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.
            large_block: If True (only valid when ``large_block_factor > 1``),
                allocate whole large blocks (returning ``KVCacheBlock`` objects
                whose ``block_id`` is the large id). If False, allocate small
                blocks (the legacy semantics).
            last_hit_block_id: Optional small-block id of the last prefix-cache
                hit for this request. When provided, the allocator tries to
                continue allocating small blocks from the same parent large
                block to keep the request's blocks within one large block.
                Ignored for ``large_block=True`` and for legacy flat pools.

        Returns:
            A list of new blocks.
        """
        if num_blocks == 0:
            return []
        capacity = (
            self.get_num_free_large_blocks()
            if large_block
            else self.get_num_free_blocks()
        )
        if num_blocks > capacity:
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        if self.large_block_factor == 1:
            # Legacy / flat path. ``large_block`` and ``last_hit_block_id`` are
            # no-ops here; small ids and large ids coincide.
            ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
        elif large_block:
            assert self.free_large_block_queue is not None
            large_blocks = self.free_large_block_queue.popleft_n(num_blocks)
            for blk in large_blocks:
                meta = self.large_block_metas[blk.block_id]
                meta.num_small_in_use = self.large_block_factor
                meta.next_small_idx = self.large_block_factor
            ret = large_blocks
        else:
            ret = self._dispense_small_blocks(num_blocks, last_hit_block_id)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    def _dispense_small_blocks(
        self, num_blocks: int, last_hit_block_id: int | None
    ) -> list[KVCacheBlock]:
        """Hierarchical small-block dispensing.

        Tries to continue inside the partial parent large block of
        ``last_hit_block_id`` (when capacity remains there), then opens fresh
        large blocks one at a time.
        """
        assert self.free_large_block_queue is not None
        N = self.large_block_factor
        ret: list[KVCacheBlock] = []
        remaining = num_blocks

        # 1. Try to continue inside the parent of last_hit_block_id.
        if last_hit_block_id is not None:
            parent_L = last_hit_block_id // N
            if 0 <= parent_L < len(self.large_block_metas):
                meta = self.large_block_metas[parent_L]
                # The parent is a "partial" meta iff its cursor is in
                # (0, N): fully-free metas live in the free-large queue;
                # fully-consumed metas have next_small_idx == N.
                if 0 < meta.next_small_idx < N:
                    avail = N - meta.next_small_idx
                    take = min(avail, remaining)
                    for j in range(take):
                        small = meta.small_blocks[meta.next_small_idx + j]
                        ret.append(small)
                    meta.next_small_idx += take
                    meta.num_small_in_use += take
                    remaining -= take

        # 2. Open fresh large blocks as needed.
        while remaining > 0:
            large_blk = self.free_large_block_queue.popleft()
            meta = self.large_block_metas[large_blk.block_id]
            assert meta.next_small_idx == 0
            take = min(N, remaining)
            for j in range(take):
                ret.append(meta.small_blocks[j])
            meta.next_small_idx = take
            meta.num_small_in_use = take
            remaining -= take

        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                    group_idx=get_group_id(block_hash),
                )
            )
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        N = self.large_block_factor
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                if N == 1:
                    self.free_block_queue.remove(block)
                else:
                    # Hierarchical: figure out whether this is a large or a
                    # small block by which list its block_id indexes into.
                    # Mamba large blocks are touched as the ``large_block``
                    # KVCacheBlock inside a meta; small blocks live inside a
                    # meta's ``small_blocks`` list at idx (block_id % N).
                    assert self.free_large_block_queue is not None

                    # TODO: Is this a valid condition?
                    meta_for_large = (
                        self.large_block_metas[block.block_id]
                        if block.block_id < len(self.large_block_metas)
                        else None
                    )
                    if (
                        meta_for_large is not None
                        and meta_for_large.large_block is block
                    ):
                        # Large-block re-touch: reclaim from free-large.
                        self.free_large_block_queue.remove(block)
                        meta_for_large.num_small_in_use = N
                        meta_for_large.next_small_idx = N
                    else:
                        # Small-block re-touch: parent meta may be partial
                        # (still hosting smalls) or recycled (sitting in
                        # free-large). Handle both.
                        parent = self.large_block_metas[block.block_id // N]
                        if parent.num_small_in_use == 0 and parent.next_small_idx == 0:
                            # Recycled — pull parent back out of free-large.
                            self.free_large_block_queue.remove(parent.large_block)
                        slot_idx = (block.block_id % N) + 1
                        if parent.next_small_idx < slot_idx:
                            parent.next_small_idx = slot_idx
                        parent.num_small_in_use += 1
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(
        self,
        ordered_blocks: Iterable[KVCacheBlock],
        large_block: bool = False,
    ) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
            large_block: If True (hierarchical mode only), the passed
                ``KVCacheBlock`` ids are large ids and the corresponding meta
                is reset wholesale.
        """
        if self.large_block_factor == 1:
            # Legacy / flat path.
            blocks_with_hash = []
            blocks_without_hash = []
            for block in ordered_blocks:
                block.ref_cnt -= 1
                if block.ref_cnt == 0 and not block.is_null:
                    if block.block_hash is None:
                        blocks_without_hash.append(block)
                    else:
                        blocks_with_hash.append(block)

            self.free_block_queue.prepend_n(blocks_without_hash)
            self.free_block_queue.append_n(blocks_with_hash)
            return

        assert self.free_large_block_queue is not None
        if large_block:
            # Free whole large blocks: reset their metas and return them to
            # the free-large queue. The large_block KVCacheBlock itself can
            # carry a hash too (mamba prefix caching), so we honour the same
            # eviction-order convention as the small-block path.
            large_with_hash: list[KVCacheBlock] = []
            large_without_hash: list[KVCacheBlock] = []
            for blk in ordered_blocks:
                blk.ref_cnt -= 1
                if blk.ref_cnt == 0 and not blk.is_null:
                    meta = self.large_block_metas[blk.block_id]
                    meta.num_small_in_use = 0
                    meta.next_small_idx = 0
                    if blk.block_hash is None:
                        large_without_hash.append(blk)
                    else:
                        large_with_hash.append(blk)
            self.free_large_block_queue.prepend_n(large_without_hash)
            self.free_large_block_queue.append_n(large_with_hash)
            return

        # Hierarchical small-block free. Mirrors the flat path's "linger"
        # semantics: a freed but still-cached small can serve a future hit
        # via ``touch`` until its parent meta is recycled. Cache eviction
        # for smalls happens lazily in ``get_new_blocks`` (the per-block
        # ``_maybe_evict_cached_block`` loop) and at meta-recycle time
        # below.
        N = self.large_block_factor
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                meta = self.large_block_metas[block.block_id // N]
                meta.num_small_in_use -= 1
                # Recycle the meta when it has no in-use smalls. We don't
                # require ``next_small_idx == N`` because a partial meta
                # whose request finished early would otherwise leak.
                if meta.num_small_in_use == 0 and meta.next_small_idx > 0:
                    meta.next_small_idx = 0
                    # Tail-append so existing small-block free behaviour
                    # (LRU order) is preserved as much as possible.
                    self.free_large_block_queue.append(meta.large_block)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

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
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free (small) blocks the pool can still dispense.

        In hierarchical mode this counts: full free-large capacity (as
        ``N * num_free_large``) plus the residual cursor capacity of every
        partial meta. Smalls that were freed inside a still-partial meta
        don't count — they're inaccessible until that meta is recycled
        (anti-fragmentation invariant).
        """
        if self.large_block_factor == 1:
            return self.free_block_queue.num_free_blocks
        assert self.free_large_block_queue is not None
        N = self.large_block_factor
        free = self.free_large_block_queue.num_free_blocks * N
        # TODO: Can ignore partial ; Note can be freed from head too
        # for meta in self.large_block_metas:
        #     if 0 < meta.next_small_idx < N:
        #         free += N - meta.next_small_idx
        return free

    def get_num_free_large_blocks(self) -> int:
        """Number of fully-free large blocks (hierarchical mode only)."""
        if self.large_block_factor == 1:
            # In flat mode there's no separate large-block notion.
            return self.free_block_queue.num_free_blocks
        assert self.free_large_block_queue is not None
        return self.free_large_block_queue.num_free_blocks

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
