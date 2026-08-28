# SPDX-License-Identifier: Apache-2.0
"""Synchronous CPU-to-CPU block copy backend."""

from __future__ import annotations

from typing import Any

import torch


class ImmediateEvent:
    """torch.Event-compatible completion marker for synchronous CPU I/O."""

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class CpuCopyBackend:
    def __init__(self) -> None:
        self._active: dict[str, torch.Tensor] = {}
        self._offload: dict[str, torch.Tensor] = {}

    def init(
        self,
        active_caches: dict[str, torch.Tensor],
        offload_caches: dict[str, torch.Tensor],
    ) -> None:
        self._active = active_caches
        self._offload = offload_caches

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, Any]],
        wait_event: Any | None = None,
    ) -> None:
        del wait_event
        source = self._active if is_store else self._offload
        target = self._offload if is_store else self._active
        for src_block, dst_block in zip(src_blocks, dst_blocks, strict=True):
            for name in source:
                target[name][dst_block].copy_(source[name][src_block])
        events_list.append((event_idx, ImmediateEvent()))

    def shutdown(self) -> None:
        return None
