# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch stand-ins for the model runner's Triton kernels.

Patched over the kernel objects in the GPU modules so the surrounding entry
points run unmodified; see ``vllm.v1.worker.cpu.shm``.
"""

from collections.abc import Callable
from typing import Any

# Launch options Triton consumes itself; they carry no meaning for a torch
# implementation and are dropped rather than forwarded.
_LAUNCH_OPTIONS = ("num_warps", "num_stages", "num_ctas", "maxnreg")


class TorchKernel:
    """Mimics ``kernel[grid](*args, **kwargs)`` for a plain torch callable.

    The wrapped function receives ``grid`` first, since a few kernels derive
    their request count from the launch grid rather than from an argument.
    """

    def __init__(self, fn: Callable[..., Any]):
        self._fn = fn

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., Any]:
        def launch(*args: Any, **kwargs: Any) -> Any:
            for option in _LAUNCH_OPTIONS:
                kwargs.pop(option, None)
            return self._fn(grid, *args, **kwargs)

        return launch


def next_power_of_2(n: int) -> int:
    """``triton.next_power_of_2`` is absent from vLLM's Triton placeholder."""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()
