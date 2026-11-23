# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A single-rank stand-in for the parallel groups, to build layers in unit tests
without initializing `torch.distributed`."""

from contextlib import ContextDecorator
from functools import wraps
from inspect import isgeneratorfunction
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


class _ParallelGroupOverride(ContextDecorator):
    def __init__(self, group: SingleRankGroup):
        self.group = group
        self.saved: list[dict[str, Any]] = []

    def __enter__(self) -> None:
        self.saved.append(
            {name: getattr(parallel_state, name) for name in _GROUP_NAMES}
        )
        for name in _GROUP_NAMES:
            setattr(parallel_state, name, self.group)

    def __exit__(self, *exc: Any) -> None:
        for name, value in self.saved.pop().items():
            setattr(parallel_state, name, value)

    def _recreate_cm(self) -> "_ParallelGroupOverride":
        return _ParallelGroupOverride(self.group)

    def __call__(self, function):
        if isgeneratorfunction(function):

            @wraps(function)
            def wrapped(*args: Any, **kwargs: Any):
                with self._recreate_cm():
                    return (yield from function(*args, **kwargs))
        else:
            wrapped = super().__call__(function)

        decorated: Any = wrapped
        decorated._uses_single_rank_groups = True
        return decorated


def override_parallel_groups(group: SingleRankGroup) -> _ParallelGroupOverride:
    """Use `group` for all parallel groups, including across generator yields."""
    return _ParallelGroupOverride(group)
