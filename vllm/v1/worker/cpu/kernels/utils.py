# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import functools
from collections.abc import Callable
from typing import Any

import numba
import torch


@functools.lru_cache(maxsize=1024)
def ptr_view(addr: int, dtype: torch.dtype, numel: int) -> torch.Tensor:
    """Zero-copy tensor over `numel` elements at host address `addr`.

    Kernels that address several tensors through a pointer array get raw
    addresses; on CPU these are host addresses and can be viewed directly.
    """
    buf = (ctypes.c_byte * (numel * dtype.itemsize)).from_address(addr)
    return torch.frombuffer(buf, dtype=dtype)


def numba_kernel(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps a numba port of a Triton kernel, called as `fn(grid, *args)`.

    Tensor arguments are passed as zero-copy numpy views.
    """
    jitted = numba.njit(cache=True, nogil=True)(fn)

    @functools.wraps(fn)
    def launch(*args: Any, grid: tuple[int, ...], **kwargs: Any) -> None:
        jitted(
            grid,
            *map(_to_numpy, args),
            **{name: _to_numpy(arg) for name, arg in kwargs.items()},
        )

    return launch


def _to_numpy(arg: Any) -> Any:
    return arg.numpy() if isinstance(arg, torch.Tensor) else arg
