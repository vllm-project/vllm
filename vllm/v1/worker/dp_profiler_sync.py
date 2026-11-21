# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

from vllm.distributed.parallel_state import GroupCoordinator

ProfileSettings = tuple[str | None, int | None, int | None]


class DPProfilerSync:
    """Synchronize profiler lifecycle requests across a DP group."""

    def __init__(self) -> None:
        self.pending_start: ProfileSettings | None = None
        self.pending_stop = False
        self._start_now: ProfileSettings | None = None
        self._stop_now = False

    def request_start(
        self,
        profile_prefix: str | None = None,
        delay_iterations: int | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self.pending_start = (profile_prefix, delay_iterations, max_iterations)

    def request_stop(self) -> None:
        self.pending_start = None
        self._start_now = None
        self.pending_stop = True

    def observe(
        self,
        start_flags: Sequence[int],
        stop_flags: Sequence[int],
        group: GroupCoordinator,
    ) -> None:
        """Latch the agreed transition and share the selected start settings."""
        if any(stop_flags):
            self._stop_now = True
            self._start_now = None
            self.pending_start = None
            return

        source = next((rank for rank, flag in enumerate(start_flags) if flag), None)
        if source is None:
            return
        local = self.pending_start if group.rank_in_group == source else None
        self._start_now = group.broadcast_object(local, src=source)

    def consume_stop(self) -> bool:
        if not self._stop_now:
            return False
        self._stop_now = False
        self.pending_stop = False
        return True

    def consume_start(self) -> ProfileSettings | None:
        settings = self._start_now
        if settings is not None:
            self._start_now = None
            self.pending_start = None
        return settings
