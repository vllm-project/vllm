# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-owned logical buffers for execution artifacts."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _RequestBuffer:
    start: int
    rows: np.ndarray
    length: int


class RoutedExpertsArtifactBuffer:
    """Keep the uncommitted logical R3 suffix for each request."""

    def __init__(
        self,
        dtype: np.dtype[Any],
        shape_per_token: tuple[int, ...],
        block_size: int = 16,
    ) -> None:
        self.dtype = dtype
        self.shape_per_token = shape_per_token
        self.block_size = block_size
        self._requests: dict[str, _RequestBuffer] = {}

    def _allocate(self, num_rows: int) -> np.ndarray:
        capacity = max(self.block_size, num_rows)
        return np.empty((capacity, *self.shape_per_token), dtype=self.dtype)

    def capture(self, request_id: str, token_start: int, rows: np.ndarray) -> None:
        rows = np.asarray(rows)
        if rows.shape[1:] != self.shape_per_token:
            raise RuntimeError("routed-experts capture profile changed")
        rows = rows.astype(self.dtype, copy=False)
        if token_start < 0:
            raise ValueError("artifact token start must be non-negative")
        if not len(rows):
            return

        current = self._requests.get(request_id)
        if current is None:
            storage = self._allocate(len(rows))
            storage[: len(rows)] = rows
            self._requests[request_id] = _RequestBuffer(token_start, storage, len(rows))
            return

        token_end = token_start + len(rows)
        current_end = current.start + current.length
        if token_end <= current.start:
            return
        if token_start < current.start:
            rows = rows[current.start - token_start :]
            token_start = current.start
        if token_start > current_end:
            raise RuntimeError(
                "artifact capture is not contiguous: "
                f"request={request_id}, expected<={current_end}, actual={token_start}"
            )

        overlap = min(current_end - token_start, len(rows))
        if overlap > 0:
            local_start = token_start - current.start
            current.rows[local_start : local_start + overlap] = rows[:overlap]
        if overlap < len(rows):
            appended = rows[overlap:]
            required = current.length + len(appended)
            if required > len(current.rows):
                storage = self._allocate(max(required, 2 * len(current.rows)))
                storage[: current.length] = current.rows[: current.length]
                current.rows = storage
            current.rows[current.length : required] = appended
            current.length = required

    def read(
        self,
        request_id: str,
        token_start: int,
        token_end: int,
    ) -> np.ndarray:
        current = self._requests.get(request_id)
        if current is None:
            raise RuntimeError(f"artifact buffer is missing request {request_id}")
        local_start = token_start - current.start
        local_end = token_end - current.start
        if local_start < 0 or local_end > current.length or local_start >= local_end:
            raise RuntimeError(
                "artifact range is unavailable: "
                f"request={request_id}, range=[{token_start}, {token_end}), "
                f"available=[{current.start}, {current.start + current.length})"
            )
        return np.array(current.rows[local_start:local_end], copy=True)

    def release_through(self, request_id: str, token_end: int) -> None:
        current = self._requests.get(request_id)
        if current is None or token_end <= current.start:
            return
        local_end = min(token_end - current.start, current.length)
        remaining = current.length - local_end
        if remaining:
            current.rows[:remaining] = current.rows[local_end : current.length]
        current.length = remaining
        current.start += local_end
        if not current.length:
            del self._requests[request_id]

    def discard(self, request_id: str) -> None:
        self._requests.pop(request_id, None)
