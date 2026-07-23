# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rust-backed drop-in for `vllm.v1.core.block_pool.BlockPool`.

Delegates the hot alloc/free/touch loops to the `vllm_rs` PyO3 crate; keeps
the event-queue and `cache_full_blocks` (which couples to `Request.block_hashes`)
in Python, since those paths pull in user-facing dataclasses that aren't worth
porting.

Enable with env var: VLLM_USE_RUST_BLOCK_POOL=1
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import vllm_rs

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
    BlockHashList,
    BlockHashListWithBlockSize,
    ExternalBlockHash,
    generate_block_hash_extra_keys,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class RustBlockPool:
    """Same public surface as `vllm.v1.core.block_pool.BlockPool`, but hot
    methods delegate to `vllm_rs.BlockPool`."""

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size

        self._rs = vllm_rs.BlockPool(
            num_gpu_blocks, enable_caching, hash_block_size
        )

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []
        self.metrics_collector = metrics_collector

    # ---- pass-through attributes ----

    @property
    def blocks(self):
        # Exposed for a handful of callers (evict_blocks in connector land).
        # Lazy build from Rust each access is expensive; cache on demand.
        # Most callers just iterate, so a lightweight list is fine.
        n = self.num_gpu_blocks
        return [self._rs.get_block(i) for i in range(n)]

    @property
    def free_block_queue(self):
        return self._rs.free_block_queue

    @property
    def cached_block_hash_to_block(self):
        return self._rs.cached_block_hash_to_block

    @property
    def null_block(self):
        return self._rs.null_block

    # ---- hot methods: delegate to Rust ----

    def get_num_free_blocks(self) -> int:
        return self._rs.get_num_free_blocks()

    def get_usage(self) -> float:
        return self._rs.get_usage()

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        return self._rs.get_cached_block(bytes(block_hash), list(kv_cache_group_ids))

    def get_new_blocks(self, num_blocks: int):
        blocks = self._rs.get_new_blocks(num_blocks)
        if self.metrics_collector:
            for b in blocks:
                self.metrics_collector.on_block_allocated(b)
        return blocks

    def touch(self, blocks: Sequence[Any]) -> None:
        self._rs.touch(blocks)
        if self.metrics_collector:
            for b in blocks:
                self.metrics_collector.on_block_accessed(b)

    def free_blocks(self, ordered_blocks: Iterable[Any]) -> None:
        self._rs.free_blocks(ordered_blocks)

    def reset_prefix_cache(self) -> bool:
        num_used = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used != 1:  # null block stays "used"
            logger.warning(
                "Failed to reset prefix cache because some blocks (%d) are "
                "not freed yet",
                num_used - 1,
            )
            return False
        self._rs.reset_prefix_cache()
        if self.metrics_collector:
            self.metrics_collector.reset()
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())
        logger.info("Successfully reset prefix cache")
        return True

    # ---- non-hot methods: kept in Python ----

    def cache_full_blocks(
        self,
        request: "Request",
        blocks: list,
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        # Lifted verbatim from BlockPool.cache_full_blocks — same semantics,
        # just writes through self._rs.cached_block_hash_to_block instead of
        # a Python dict. That hash-map is shared with the Rust allocator, so
        # inserts become immediately visible to the next get_cached_block call.
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            block_hashes: BlockHashList = request.block_hashes
        else:
            assert block_size % self.hash_block_size == 0
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        cache_map = self._rs.cached_block_hash_to_block

        # Fast path: when events are off we don't need the side-output
        # `new_hashes` list and can delegate the whole loop to Rust.
        if new_hashes is None and isinstance(new_block_hashes, list):
            self._rs.cache_full_blocks_fast(
                blocks, new_block_hashes, num_cached_blocks,
                num_full_blocks, kv_cache_group_id,
            )
        else:
            for i, blk in enumerate(new_full_blocks):
                if blk.is_null:
                    continue
                assert blk.block_hash is None
                block_hash = new_block_hashes[i]
                block_hash_with_group_id = make_block_hash_with_group_id(
                    block_hash, kv_cache_group_id
                )
                blk.block_hash = block_hash_with_group_id
                cache_map.insert(block_hash_with_group_id, blk)
                if new_hashes is not None:
                    new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
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
                )
            )

    def evict_blocks(self, block_ids: set[int]) -> None:
        cache_map = self._rs.cached_block_hash_to_block
        for block_id in block_ids:
            assert 0 <= block_id < self.num_gpu_blocks
            block = self._rs.get_block(block_id)
            if block is None:
                continue
            block_hash = block.block_hash
            if block_hash is None:
                continue
            popped = cache_map.pop(bytes(block_hash), block_id)
            if popped is None:
                continue
            block.reset_hash()
            if self.enable_kv_cache_events:
                self.kv_event_queue.append(
                    BlockRemoved(
                        block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                        medium=MEDIUM_GPU,
                    )
                )

    def take_events(self) -> list[KVCacheEvent]:
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
