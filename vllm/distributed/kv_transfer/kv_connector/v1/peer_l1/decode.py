# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Release-batch ordering and bounded retirement of redundant decode rows."""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice, takewhile

import numpy as np

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock

from .session import peer_key
from .stats import PeerL1Stats


@dataclass(slots=True)
class _Idle:
    block: KVCacheBlock
    key: bytes
    hint: bytes


class DecodeCacheOrder:
    """Prefer redundant decode rows for eviction on the scheduler thread.

    Placement is advisory. Bounded maintenance and release-time pressure checks
    preserve spare empty rows without altering the pool's lease protection.
    """

    def __init__(
        self,
        pool: BlockPool,
        query: Callable[[list[bytes]], np.ndarray | None],
        stats: PeerL1Stats,
        scan_blocks: int,
    ):
        self.pool, self.query, self.stats = pool, query, stats
        self.scan_blocks = scan_blocks
        self._under_pressure = False
        self._idle: dict[int, _Idle] = {}
        self._candidates: OrderedDict[int, _Idle] = OrderedDict()

    def forget(self, block: KVCacheBlock) -> None:
        self._candidates.pop(block.block_id, None)
        self._idle.pop(block.block_id, None)

    def on_reuse(self, block: KVCacheBlock) -> None:
        self.forget(block)

    def _has_spare_rows(self) -> bool:
        """Whether at least `scan_blocks` empty rows lead the free queue."""
        leading = islice(
            self.pool.free_block_queue.iter_blocks_after(None), self.scan_blocks
        )
        empty = sum(1 for _ in takewhile(lambda b: b.block_hash is None, leading))
        return empty >= self.scan_blocks

    def classify(self, blocks: list[KVCacheBlock]) -> list[bool]:
        """Return front-placement flags for one newly free cached batch.

        The pool calls this on its scheduler thread before inserting the batch.
        Contended hints conservatively leave rows at their normal LRU position.
        """
        # Rows freed within this step can restore empty headroom. Recheck
        # before moving cached rows ahead of those empty rows.
        if self._under_pressure and self._has_spare_rows():
            self._under_pressure = False
        if not self._under_pressure:
            return [False] * len(blocks)
        keys, hint_keys = [], []
        for block in blocks:
            assert block.block_hash is not None
            keys.append(block.block_hash)
            previous = self._idle.get(block.block_id)
            hint_keys.append(
                previous.hint
                if previous is not None and previous.key == block.block_hash
                else peer_key(block.block_hash)
            )
        hints = self.query(hint_keys)
        if hints is None:
            flags = [False] * len(blocks)
        else:
            flags = [bool(value) for value in hints]
        for block, key, redundant, hint in zip(
            blocks, keys, flags, hint_keys, strict=True
        ):
            assert key is not None
            self.forget(block)
            entry = _Idle(block, key, hint)
            self._idle[block.block_id] = entry
            if redundant:
                self._candidates[block.block_id] = entry
        for block, redundant in zip(reversed(blocks), reversed(flags), strict=True):
            if redundant:
                self._candidates.move_to_end(block.block_id, last=False)
        self.stats.inc("decode_order_rows", action="front", value=sum(flags))
        return flags

    def scan(self) -> None:
        """Move free candidates to the tail after their prefill copies disappear."""
        queue = self.pool.free_block_queue
        batch = list(islice(self._candidates.values(), self.scan_blocks))
        hints = self.query([entry.hint for entry in batch]) if batch else []
        self.stats.inc("decode_order_rows", action="scanned", value=len(batch))
        if hints is None:
            return
        for entry, present in zip(batch, hints, strict=True):
            block = entry.block
            # A row can be reused and freed again under the same block ID.
            # Only the exact record from this free period may move it.
            if self._idle.get(block.block_id) is not entry:
                continue
            if (
                block.ref_cnt
                or block.next_free_block is None
                or block.block_hash != entry.key
            ):
                self.forget(block)
                continue
            if not present:
                queue.remove(block)
                queue.append(block)
                del self._candidates[block.block_id]
                self.stats.inc("decode_order_rows", action="retired")
            else:
                self._candidates.move_to_end(block.block_id)

    def finish_step(self) -> None:
        """Bound withdrawal work and sample empty headroom for subsequent releases."""
        self.scan()
        self._under_pressure = not self._has_spare_rows()

    def reset(self) -> None:
        self._under_pressure = False
        self._idle.clear()
        self._candidates.clear()
