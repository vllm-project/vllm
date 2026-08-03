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


class RoutedExpertsArtifactBuffer:
    """Keep the uncommitted logical R3 suffix for each request."""

    def __init__(
        self,
        dtype: np.dtype[Any],
        shape_per_token: tuple[int, ...],
    ) -> None:
        self.dtype = dtype
        self.shape_per_token = shape_per_token
        self._requests: dict[str, _RequestBuffer] = {}

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
            self._requests[request_id] = _RequestBuffer(
                token_start, np.array(rows, copy=True)
            )
            return

        token_end = token_start + len(rows)
        current_end = current.start + len(current.rows)
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
            current.rows = np.concatenate((current.rows, rows[overlap:]))

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
        if local_start < 0 or local_end > len(current.rows) or local_start >= local_end:
            raise RuntimeError(
                "artifact range is unavailable: "
                f"request={request_id}, range=[{token_start}, {token_end}), "
                f"available=[{current.start}, {current.start + len(current.rows)})"
            )
        return np.array(current.rows[local_start:local_end], copy=True)

    def release_through(self, request_id: str, token_end: int) -> None:
        current = self._requests.get(request_id)
        if current is None or token_end <= current.start:
            return
        local_end = min(token_end - current.start, len(current.rows))
        current.rows = current.rows[local_end:]
        current.start += local_end
        if not len(current.rows):
            del self._requests[request_id]

    def discard(self, request_id: str) -> None:
        self._requests.pop(request_id, None)
