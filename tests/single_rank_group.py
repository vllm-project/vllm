# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A single-rank stand-in for the parallel groups, to build layers in unit tests
without initializing `torch.distributed`."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch

import vllm.distributed.parallel_state as parallel_state

_GROUP_NAMES = ("_WORLD", "_TP", "_ETP", "_DCP", "_PP", "_DP", "_EP", "_PCP")


class SingleRankGroup:
    """No-op single-rank stand-in for a compiled `GroupCoordinator`."""

    world_size = 1
    rank_in_group = 0
    local_rank = 0
    is_first_rank = True
    is_last_rank = True

    @property
    def device_group(self):
        return self

    @property
    def cpu_group(self):
        return self

    @property
    def device_communicator(self):
        return self

    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 1

    def all_reduce(self, input_: torch.Tensor, *args: Any, **kwargs: Any):
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = 0):
        return input_

    def all_gatherv(self, inputs: Any, dim: int = 0, sizes: Any = None):
        return inputs

    def reduce_scatter(self, input_: torch.Tensor, dim: int = 0):
        return input_

    def reduce_scatterv(self, input_: Any, dim: int = 0, sizes: Any = None):
        return input_

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        return input_

    def send(self, *args: Any, **kwargs: Any) -> None:
        return None

    def recv(self, *args: Any, **kwargs: Any) -> None:
        return None

    def barrier(self) -> None:
        return None

    def destroy(self) -> None:
        return None


@contextmanager
def override_parallel_groups(group: "SingleRankGroup") -> Generator[None]:
    """Use `group` for all the parallel groups of `parallel_state`."""
    saved = {name: getattr(parallel_state, name) for name in _GROUP_NAMES}
    for name in _GROUP_NAMES:
        setattr(parallel_state, name, group)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(parallel_state, name, value)
