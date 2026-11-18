# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.config import ParallelConfig


@dataclass
class _Share:
    row_size: int
    provide: Callable[[], list[float]]
    receive: Callable[[list[list[float]]], object]
    on_idle: Callable[[], object] | None


class DPStateSync:
    """The DP engine core's cross-rank state sync, offered to the scheduler
    under DP attention + EP.

    Args:
        parallel_config: Runs the all-reduce.
        group: Returns the current DP group (elastic EP can replace it).

    """

    def __init__(
        self, parallel_config: "ParallelConfig", group: Callable[[], "ProcessGroup"]
    ) -> None:
        self._parallel_config = parallel_config
        self._group = group
        self._shares: list[_Share] = []

    @property
    def rank(self) -> int:
        return self._group().rank()

    @property
    def size(self) -> int:
        return self._group().size()

    @property
    def every_step(self) -> bool:
        """Whether the state must be synced every step, not at an interval."""
        return bool(self._shares)

    def share(
        self,
        row_size: int,
        provide: Callable[[], list[float]],
        receive: Callable[[list[list[float]]], object],
        on_idle: Callable[[], object] | None = None,
    ) -> None:
        """Shares a row of floats per rank on every step from now on.

        Each sync, every rank's `provide()` row is all-gathered and
        `receive(rows)` is called on every rank, with rows in rank order.
        `on_idle()` is called when the sync finds every rank idle.
        """
        self._shares.append(_Share(row_size, provide, receive, on_idle))

    def sync(self, has_unfinished: bool, pending_pause: bool) -> tuple[bool, bool]:
        """Runs the DP state all-reduce, carrying the shared rows.

        Returns:
            (has_unfinished_global, pause_consensus)

        """
        group = self._group()
        gather = None
        if self._shares:
            row = [v for s in self._shares for v in s.provide()]
            assert len(row) == sum(s.row_size for s in self._shares)
            gather = torch.zeros(group.size(), len(row), dtype=torch.float64)
            gather[group.rank()] = torch.tensor(row, dtype=torch.float64)
        has_unfinished_global, pause_consensus = self._parallel_config.sync_dp_state(
            group,
            has_unfinished=has_unfinished,
            pending_pause=pending_pause,
            gather=gather,
        )
        if gather is not None:
            rows = gather.tolist()
            start = 0
            for s in self._shares:
                end = start + s.row_size
                s.receive([r[start:end] for r in rows])
                start = end
            if not has_unfinished_global:
                for s in self._shares:
                    if s.on_idle is not None:
                        s.on_idle()
        return has_unfinished_global, pause_consensus
