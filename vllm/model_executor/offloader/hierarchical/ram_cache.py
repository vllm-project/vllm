# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned RAM expert cache with LFRU eviction and learned pins."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.model_executor.offloader.base import should_pin_memory

logger = init_logger(__name__)


@dataclass
class RamFrame:
    layer_id: int
    expert_id: int
    offset: int
    nbytes: int
    heat: float = 0.0
    last_access: float = 0.0
    pinned: bool = False
    valid: bool = False


class PinnedExpertRamCache:
    """Fixed pinned-byte arena holding expert rows for fast H2D."""

    def __init__(
        self,
        capacity_bytes: int,
        row_nbytes: int,
        *,
        device: torch.device | None = None,
    ):
        self.row_nbytes = row_nbytes
        self.capacity_bytes = max(0, capacity_bytes)
        pin = should_pin_memory()
        num_frames = max(1, self.capacity_bytes // max(row_nbytes, 1)) if row_nbytes else 0
        self.num_frames = num_frames
        self._arena: torch.Tensor | None = None
        if num_frames > 0 and row_nbytes > 0:
            self._arena = torch.empty(
                num_frames * row_nbytes,
                dtype=torch.uint8,
                pin_memory=pin,
            )
            logger.info(
                "PinnedExpertRamCache: %d frames × %d bytes (%.3f GiB, pin=%s)",
                num_frames,
                row_nbytes,
                num_frames * row_nbytes / 1024**3,
                pin,
            )
        self._frames: list[RamFrame] = [
            RamFrame(layer_id=-1, expert_id=-1, offset=i * row_nbytes, nbytes=row_nbytes)
            for i in range(num_frames)
        ]
        # (layer, expert) -> frame index
        self._index: dict[tuple[int, int], int] = {}
        self._free: list[int] = list(range(num_frames))
        self._clock = 0.0

    @property
    def enabled(self) -> bool:
        return self._arena is not None and self.num_frames > 0

    def _touch(self, frame: RamFrame) -> None:
        self._clock = time.monotonic()
        frame.last_access = self._clock
        frame.heat = frame.heat * 0.99 + 1.0

    def get(self, layer_id: int, expert_id: int) -> torch.Tensor | None:
        """Return a view of the cached row or None on miss."""
        key = (layer_id, expert_id)
        idx = self._index.get(key)
        if idx is None:
            return None
        frame = self._frames[idx]
        if not frame.valid or self._arena is None:
            return None
        self._touch(frame)
        return self._arena[frame.offset : frame.offset + frame.nbytes]

    def put(
        self,
        layer_id: int,
        expert_id: int,
        row: torch.Tensor,
        *,
        pinned: bool = False,
    ) -> torch.Tensor:
        """Insert/replace an expert row; returns the pinned view."""
        assert self._arena is not None
        assert row.numel() == self.row_nbytes
        key = (layer_id, expert_id)
        if key in self._index:
            idx = self._index[key]
            frame = self._frames[idx]
            self._arena[frame.offset : frame.offset + frame.nbytes].copy_(
                row.view(torch.uint8).reshape(-1)
            )
            frame.pinned = frame.pinned or pinned
            frame.valid = True
            self._touch(frame)
            return self._arena[frame.offset : frame.offset + frame.nbytes]

        idx = self._alloc_frame(pinned=pinned)
        frame = self._frames[idx]
        # Evict previous occupant from index
        old_key = (frame.layer_id, frame.expert_id)
        if frame.valid and old_key in self._index and self._index[old_key] == idx:
            del self._index[old_key]
        self._arena[frame.offset : frame.offset + frame.nbytes].copy_(
            row.view(torch.uint8).reshape(-1)
        )
        frame.layer_id = layer_id
        frame.expert_id = expert_id
        frame.pinned = pinned
        frame.valid = True
        self._index[key] = idx
        self._touch(frame)
        return self._arena[frame.offset : frame.offset + frame.nbytes]

    def pin(self, layer_id: int, expert_id: int) -> None:
        idx = self._index.get((layer_id, expert_id))
        if idx is not None:
            self._frames[idx].pinned = True

    def _alloc_frame(self, *, pinned: bool) -> int:
        if self._free:
            return self._free.pop()
        # LFRU among non-pinned frames: minimize heat / recency score
        best_idx = -1
        best_score = float("inf")
        now = time.monotonic()
        for i, frame in enumerate(self._frames):
            if frame.pinned:
                continue
            age = max(1e-3, now - frame.last_access)
            score = frame.heat / age
            if score < best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            # All pinned — evict coldest pinned as last resort
            best_idx = min(
                range(len(self._frames)),
                key=lambda i: self._frames[i].heat,
            )
            self._frames[best_idx].pinned = False
        return best_idx

    def repin_hottest(
        self,
        layer_id: int,
        hot_experts: list[int],
        *,
        max_swaps: int = 4,
    ) -> int:
        """Live LFRU repin: mark hot experts pinned (up to max_swaps new pins)."""
        swaps = 0
        for e in hot_experts:
            if swaps >= max_swaps:
                break
            key = (layer_id, e)
            if key not in self._index:
                continue
            frame = self._frames[self._index[key]]
            if not frame.pinned:
                frame.pinned = True
                swaps += 1
        return swaps
