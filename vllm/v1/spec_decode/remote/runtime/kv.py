# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-only KV block pool.

The standalone process owns its draft KV cache outright: blocks are handed
out per request slot and returned when the sequence closes. Running out of
blocks fails the request that needed them; resident sequences are never
evicted to make room.
"""

import numpy as np

from vllm.utils.math_utils import cdiv

NULL_BLOCK_ID = 0
"""Reserved block that padded tokens write to; never handed out."""


class DraftKVExhausted(Exception):
    """The pool cannot grow a sequence to the requested length."""


class DraftBlockPool:
    """Fixed pool of KV blocks keyed by request slot."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 2:
            raise ValueError("a draft KV pool needs the null block plus one")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Only the prefix is free. Keep storage allocated so cleanup never
        # needs to grow a list while recovering from an allocation failure.
        self._free = list(range(num_blocks - 1, NULL_BLOCK_ID, -1))
        self._num_free = len(self._free)
        self._owned: dict[int, list[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return self._num_free

    def blocks(self, req_index: int) -> tuple[int, ...]:
        return tuple(self._owned.get(req_index, ()))

    def ensure(self, req_index: int, num_positions: int) -> list[int]:
        """Grow ``req_index`` to cover positions ``[0, num_positions)``.

        Returns the block ids appended, in order. Allocation is all or
        nothing: on DraftKVExhausted the slot keeps its previous blocks.
        """
        owned = self._owned.get(req_index, [])
        missing = cdiv(num_positions, self.block_size) - len(owned)
        if missing <= 0:
            return []
        if missing > self._num_free:
            raise DraftKVExhausted(
                f"need {missing} more draft KV blocks, {self._num_free} free"
            )
        # Prepare allocation-backed metadata before consuming free capacity.
        remaining = self._num_free - missing
        new_blocks = self._free[remaining : self._num_free][::-1]
        self._owned[req_index] = owned + new_blocks
        self._num_free = remaining
        return new_blocks

    def release(self, req_index: int) -> None:
        owned = self._owned.get(req_index)
        if owned is None:
            return
        returned = self._num_free + len(owned)
        for index, block_id in enumerate(reversed(owned), self._num_free):
            self._free[index] = block_id
        # Until all writes succeed these cells are outside the free prefix.
        del self._owned[req_index]
        self._num_free = returned

    def slot_mapping(self, req_index: int, positions: np.ndarray) -> np.ndarray:
        """KV slot of each position; positions must be covered by ``ensure``."""
        blocks = np.asarray(self._owned[req_index], dtype=np.int64)
        return blocks[positions // self.block_size] * self.block_size + (
            positions % self.block_size
        )
