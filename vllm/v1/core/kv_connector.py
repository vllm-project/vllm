# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete SchedulerContext backed by a BlockPool."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVCacheBlockView,
    SchedulerContext,
)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool


class KVConnectorSchedulerContext(SchedulerContext):
    """SchedulerContext backed by a BlockPool.

    Created by the scheduler and bound to connectors via
    ``bind_scheduler_context``.
    """

    def __init__(self, block_pool: BlockPool):
        self._block_pool = block_pool

    def get_block(self, block_id: int) -> KVCacheBlockView:
        return self._block_pool.blocks[block_id]  # type: ignore[return-value]

    def touch(self, block_ids: list[int]) -> None:
        pool = self._block_pool
        pool.touch([pool.blocks[bid] for bid in block_ids])

    def free_blocks(self, block_ids: Iterable[int]) -> None:
        pool = self._block_pool
        pool.free_blocks(pool.blocks[bid] for bid in block_ids)

    def iter_free_blocks(
        self, after_block_id: int | None = None
    ) -> Iterator[KVCacheBlockView]:
        pool = self._block_pool
        free_queue = pool.free_block_queue
        tail = free_queue.fake_free_list_tail

        if after_block_id is not None:
            node = pool.blocks[after_block_id].next_free_block
        else:
            node = free_queue.fake_free_list_head.next_free_block

        while node is not None and node is not tail:
            yield node  # type: ignore[misc]
            node = node.next_free_block
