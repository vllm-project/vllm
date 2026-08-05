# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-owned fixed-size tail buffers for execution artifacts."""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _RequestTail:
    slot: int
    block_start: int
    length: int = 0


class RoutedExpertsArtifactBuffer:
    """Stage at most one incomplete logical block per active request."""

    def __init__(
        self,
        dtype: np.dtype[Any],
        shape_per_token: tuple[int, ...],
        block_size: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
    ) -> None:
        self.dtype = dtype
        self.shape_per_token = shape_per_token
        self.block_size = block_size
        self._rows: np.ndarray = np.empty(
            (2 * max_num_seqs, block_size, *shape_per_token), dtype=dtype
        )
        max_retained_blocks = (max_num_batched_tokens + block_size - 1) // block_size
        self._retained_rows: np.ndarray = np.empty(
            (max_retained_blocks, block_size, *shape_per_token), dtype=dtype
        )
        self._free_slots = list(range(len(self._rows) - 1, -1, -1))
        self._completed_slots: dict[int, int] = {}
        self._free_retained_slots = list(range(max_retained_blocks - 1, -1, -1))
        self._retained_slots: dict[int, int] = {}
        self._requests: dict[Hashable, _RequestTail] = {}

    def _tail(self, request_id: Hashable, block_start: int) -> _RequestTail:
        tail = self._requests.get(request_id)
        if tail is not None:
            if tail.block_start != block_start:
                raise RuntimeError(
                    "artifact capture skipped an incomplete block: "
                    f"request={request_id}, expected={tail.block_start}, "
                    f"actual={block_start}"
                )
            return tail
        if not self._free_slots:
            raise RuntimeError("artifact request-buffer pool is exhausted")
        tail = _RequestTail(self._free_slots.pop(), block_start)
        self._requests[request_id] = tail
        return tail

    def capture(
        self, request_id: Hashable, token_start: int, rows: np.ndarray
    ) -> list[tuple[int, np.ndarray]]:
        """Stage rows and return completed blocks without retaining them."""
        rows = np.asarray(rows)
        if rows.shape[1:] != self.shape_per_token:
            raise RuntimeError("routed-experts capture profile changed")
        rows = rows.astype(self.dtype, copy=False)
        if token_start < 0:
            raise ValueError("artifact token start must be non-negative")

        tail = self._requests.get(request_id)
        if tail is not None and token_start < tail.block_start:
            skipped = min(tail.block_start - token_start, len(rows))
            rows = rows[skipped:]
            token_start += skipped

        completed: list[tuple[int, np.ndarray]] = []
        offset = 0
        while offset < len(rows):
            position = token_start + offset
            block_start = position // self.block_size * self.block_size
            local_start = position - block_start

            # Full aligned input blocks do not need to touch the tail pool.
            if (
                request_id not in self._requests
                and local_start == 0
                and len(rows) - offset >= self.block_size
            ):
                completed.append((block_start, rows[offset : offset + self.block_size]))
                offset += self.block_size
                continue

            tail = self._tail(request_id, block_start)
            if local_start > tail.length:
                raise RuntimeError(
                    "artifact capture is not contiguous: "
                    f"request={request_id}, expected<={block_start + tail.length}, "
                    f"actual={position}"
                )
            count = min(self.block_size - local_start, len(rows) - offset)
            self._rows[tail.slot, local_start : local_start + count] = rows[
                offset : offset + count
            ]
            tail.length = max(tail.length, local_start + count)
            offset += count
            if tail.length == self.block_size:
                block = self._rows[tail.slot]
                del self._requests[request_id]
                self._completed_slots[id(block)] = tail.slot
                completed.append((block_start, block))

        return completed

    def read(
        self, request_id: Hashable, token_start: int, token_end: int
    ) -> np.ndarray:
        tail = self._requests.get(request_id)
        if tail is None:
            raise RuntimeError(f"artifact buffer is missing request {request_id}")
        local_start = token_start - tail.block_start
        local_end = token_end - tail.block_start
        if local_start < 0 or local_end > tail.length or local_start >= local_end:
            raise RuntimeError(
                "artifact range is unavailable: "
                f"request={request_id}, range=[{token_start}, {token_end}), "
                f"available=[{tail.block_start}, {tail.block_start + tail.length})"
            )
        return self._rows[tail.slot, local_start:local_end].copy()

    def retain_block(self, rows: np.ndarray) -> np.ndarray:
        """Retain one unkeyed block after the current capture call."""
        if id(rows) in self._completed_slots:
            return rows
        if not self._free_retained_slots:
            raise RuntimeError("artifact retained-block pool is exhausted")
        slot = self._free_retained_slots.pop()
        retained = self._retained_rows[slot]
        retained[...] = rows
        self._retained_slots[id(retained)] = slot
        return retained

    def release_block(self, rows: np.ndarray) -> None:
        slot = self._completed_slots.pop(id(rows), None)
        if slot is not None:
            self._free_slots.append(slot)
            return
        slot = self._retained_slots.pop(id(rows), None)
        if slot is not None:
            self._free_retained_slots.append(slot)

    def _release(self, request_id: Hashable) -> None:
        tail = self._requests.pop(request_id)
        self._free_slots.append(tail.slot)

    def discard(self, request_id: Hashable) -> None:
        if request_id in self._requests:
            self._release(request_id)

    def reset(self) -> None:
        for request_id in list(self._requests):
            self._release(request_id)
        self._free_slots.extend(self._completed_slots.values())
        self._completed_slots.clear()
        self._retained_slots.clear()
        self._free_retained_slots[:] = range(len(self._retained_rows) - 1, -1, -1)
